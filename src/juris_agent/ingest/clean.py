"""F1 法条清洗与条级聚合：`data/raw/*.docx` → 一条一行的 `data/laws.jsonl`。

**为什么必须按「条」切**

评测的 gold 标注单位就是「条」（法条编号）。切分粒度与评测粒度不一致，Recall 就没法算。
所以这里不做任何「聪明」的分块，只认 `第X条`；分块策略是 F2 的事，它在条级结果上再做。

**这份文件要处理的四个真实脏点**（都是把 8 部法规抽出来看过之后写的，不是设想的）：

1. **目录区**。目录里的章节标题和正文的章节标题**一模一样**（`第一章　总　　则`），
   没法靠模式区分。解法是个巧合但可靠：目录在前、正文在后，取「第一条之前**最后**一个
   章节标题行」当正文起点——目录里的同名条目在它前面，正文自己的章节标题在后面。
   民法典的目录有 140 多行，靠这条兜住。
2. **行首空白**。司法解释（一）的条文行前面有两个全角空格，`^第X条` 会整部漏掉
   （实测：59 个含「条」的行，一条都没匹配上）。正则必须容忍行首空白。
3. **条文跨行**。第二条的第二款是独立段落、不带 `第X条` 前缀，必须并回上一条。
   民法典有 1260 条，这个逻辑错一次就是全盘错。
4. **生效日期不能靠正则**。带修正的法律，附则里至今保留**原始**施行条款，正则抓到的
   是旧版生效日。所以生效日期以 `configs/laws_manifest.yaml` 为准，正则只当备选，
   并在记录里标来源（见 `effective_date_source`）。

**输出契约**：每条记录的 `article_key`（形如 `lcl#47`）是跨模块的稳定主键，
gold 标注、引用校验、邻条扩展全部用它对齐。**改这个格式等于改全套评测的接口。**
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import date
from pathlib import Path
from typing import Iterable

import yaml
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# 常量：正则与中文数字
# --------------------------------------------------------------------------- #

# 行首允许空白（全角/半角/BOM）；条号后必须跟分隔符，否则会把
# 「依据劳动合同法第八十五条规定」这类**行内引用**误当成条文开头。
_ARTICLE_RE = re.compile(
    r"^[ \t　﻿]*"
    r"第(?P<num>[一二三四五六七八九十百千零〇两]+)条"
    r"(?:之(?P<sub>[一二三四五六七八九十]+))?"
    r"[ \t　]"
)

_HEADING_RE = re.compile(
    r"^[ \t　﻿]*"
    r"第(?P<num>[一二三四五六七八九十]+)(?P<kind>[编章节])"
    r"[ \t　]*(?P<title>.*)$"
)

# 正文自述的施行日期：只在清单缺项时当备选
_EFFECTIVE_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日起施行")

_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}

# 条文正文为空时用这个占位，避免下游把空 chunk 当「检索不到」和「数据坏」混淆
_EMPTY_TEXT = "(本条无正文)"


def cn_to_int(s: str) -> int:
    """中文数字转 int。支持到千位（民法典最大 第一千二百六十条）。

    只有「十」前面省略了数字（`十五` = 15，不是 0*10+5），所以要 `num or 1`。
    """
    total = 0
    section = 0
    num = 0
    for ch in s:
        if ch in _CN_DIGITS:
            num = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            section += (num or 1) * _CN_UNITS[ch]
            num = 0
        else:
            raise ValueError(f"无法解析的中文数字: {ch!r}（在 {s!r} 中）")
    return total + section + num


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #


class LawMeta(BaseModel):
    """`configs/laws_manifest.yaml` 里一条法规的元信息。"""

    id: str
    file: str
    name: str
    short: str
    version_date: date | None = None
    effective_date: date | None = None
    verified: bool = False
    note: str = ""


class Article(BaseModel):
    """一条法条。这是整个语料的原子单位。"""

    article_key: str  # 稳定主键，如 `lcl#47`（gold / 引用 / 邻条扩展都靠它）
    law_id: str
    law_name: str
    law_short: str
    article_no: str  # 中文条号原文，如 `第四十七条`
    article_num: int  # 数字条号，用于邻条扩展与连续性校验
    article_sub: str | None = None  # `第X条之Y` 的 Y
    chapter: str | None = None  # 最近一层「章」
    section: str | None = None  # 最近一层「节」（换章时清空）
    text: str
    char_count: int = 0
    version_date: date | None = None
    effective_date: date | None = None
    # manifest = 人工清单（权威）/ text-regex = 从正文抓的备选 / none = 都没有
    effective_date_source: str = "none"
    source_file: str = ""

    def citation(self) -> str:
        """给人看/给模型引用的形式：`《劳动合同法》第四十七条`。"""
        sub = f"之{self.article_sub}" if self.article_sub else ""
        return f"《{self.law_short}》{self.article_no}{sub}"


# --------------------------------------------------------------------------- #
# docx → 纯文本
# --------------------------------------------------------------------------- #


def docx_to_text(path: Path) -> str:
    """把 .docx 抽成纯文本（段落边界 → 换行）。

    故意用标准库 zipfile 而不是 python-docx：这里只需要 `word/document.xml` 的文本，
    为一个纯文本抽取引一个解析库不划算，也少一个依赖要维护。
    """
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)  # 制表位
    xml = re.sub(r"</w:p>", "\n", xml)  # 段落边界
    text = re.sub(r"<[^>]+>", "", xml)  # 其余标签全剥掉
    text = text.replace(" ", " ")  # 不换行空格
    return text


def read_source(path: Path) -> str:
    """按扩展名取纯文本。除 .docx 外也认 .txt / .md，方便以后手工补法规。"""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return docx_to_text(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"不支持的语料格式: {path.name}")


# --------------------------------------------------------------------------- #
# 正文切分
# --------------------------------------------------------------------------- #


def _compact_heading(kind: str, num_zh: str, title: str) -> str:
    """`第一章　总　　则` → `第一章 总则`。

    法条标题里用全角空格做字距对齐（`总　　则`），直接留着会污染嵌入和关键词匹配。
    """
    title = re.sub(r"[　\s]+", "", title)
    return f"第{num_zh}{kind} {title}".strip()


def _strip_body_front(lines: list[str]) -> int:
    """返回正文起始行号：第一条之前**最后**一个章节标题行；没有标题就取第一条本身。

    这是绕开目录区的关键（见模块 docstring 第 1 点）。
    """
    first_art = next((i for i, l in enumerate(lines) if _ARTICLE_RE.match(l)), None)
    if first_art is None:
        return len(lines)
    headings_before = [i for i, l in enumerate(lines[:first_art]) if _HEADING_RE.match(l)]
    return headings_before[-1] if headings_before else first_art


def parse_articles(text: str, meta: LawMeta) -> list[Article]:
    """把一部法规的纯文本切成 `Article` 列表。"""
    lines = text.replace("﻿", "").split("\n")
    start = _strip_body_front(lines)

    effective_from_text: date | None = None
    if m := _EFFECTIVE_RE.search(text):
        effective_from_text = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    if meta.effective_date is not None:
        eff, eff_src = meta.effective_date, "manifest"
    elif effective_from_text is not None:
        eff, eff_src = effective_from_text, "text-regex"
    else:
        eff, eff_src = None, "none"

    articles: list[Article] = []
    chapter: str | None = None
    section: str | None = None
    cur: dict | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal cur, buf
        if cur is None:
            return
        body = "\n".join(p for p in buf if p.strip()).strip()
        articles.append(
            Article(
                **cur,
                text=body or _EMPTY_TEXT,
                char_count=len(body),
                version_date=meta.version_date,
                effective_date=eff,
                effective_date_source=eff_src,
                source_file=meta.file,
            )
        )
        cur, buf = None, []

    for line in lines[start:]:
        if _HEADING_RE.match(line):
            # 标题行不属于任何条文——必须先 flush，否则会被并进上一条的正文
            flush()
            h = _HEADING_RE.match(line)
            assert h is not None
            label = _compact_heading(h.group("kind"), h.group("num"), h.group("title"))
            if h.group("kind") == "节":
                section = label
            else:  # 编/章：章以下无节
                chapter, section = label, None
            continue

        if (a := _ARTICLE_RE.match(line)) is not None:
            flush()
            num_zh, sub = a.group("num"), a.group("sub")
            num = cn_to_int(num_zh)
            key = f"{meta.id}#{num}" + (f"之{sub}" if sub else "")
            cur = {
                "article_key": key,
                "law_id": meta.id,
                "law_name": meta.name,
                "law_short": meta.short,
                "article_no": f"第{num_zh}条",
                "article_num": num,
                "article_sub": sub,
                "chapter": chapter,
                "section": section,
            }
            rest = line[a.end() :].strip()
            buf = [rest] if rest else []
            continue

        if cur is not None:
            buf.append(line.strip())

    flush()
    return articles


# --------------------------------------------------------------------------- #
# 清单与全量清洗
# --------------------------------------------------------------------------- #


def load_manifest(path: Path) -> list[LawMeta]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [LawMeta(**item) for item in raw["laws"]]


def clean_corpus(
    raw_dir: Path, manifest: Iterable[LawMeta]
) -> tuple[list[Article], list[str]]:
    """清洗整个语料。

    返回 `(按清单顺序排序的条文列表, 警告列表)`。警告不抛异常——语料缺一两个文件
    应该让流程继续，但必须让人看见。
    """
    warnings: list[str] = []
    articles: list[Article] = []

    for meta in manifest:
        path = raw_dir / meta.file
        if not path.exists():
            warnings.append(f"清单里的文件不存在，已跳过: {meta.file}")
            continue
        parsed = parse_articles(read_source(path), meta)
        if not parsed:
            warnings.append(f"未切出任何条文（正则或格式变了？）: {meta.file}")
            continue
        nums = [a.article_num for a in parsed]
        expected = list(range(1, len(nums) + 1))
        if nums != expected:
            missing = sorted(set(expected) - set(nums))
            warnings.append(
                f"{meta.short}: 条号不连续，缺失 {missing[:8]}{'…' if len(missing) > 8 else ''}"
            )
        if not meta.verified:
            warnings.append(f"{meta.short}: effective_date 未经人工核对（manifest.verified=false）")
        articles.extend(parsed)

    return articles, warnings


def corpus_hash(articles: Iterable[Article]) -> str:
    """语料指纹。指标元信息里必须带它，否则数字无法复现。"""
    h = hashlib.sha256()
    for a in articles:
        h.update(a.article_key.encode("utf-8"))
        h.update(b"\0")
        h.update(a.text.encode("utf-8"))
        h.update(b"\1")
    return h.hexdigest()[:16]


def write_jsonl(articles: Iterable[Article], out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    items = list(articles)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for a in items:
            fh.write(a.model_dump_json() + "\n")
    return corpus_hash(items)


def load_articles(path: Path | None = None) -> list[Article]:
    """读回 `laws.jsonl`。检索与评测层都用它，不必各自解析一遍。

    `path=None` 时按项目默认位置找（config 是局部 import——纯解析的单元测试
    不该被迫拉起 pydantic-settings）。
    """
    if path is None:
        from juris_agent.config import get_settings

        path = get_settings().paths.laws
    return [
        Article(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------- #
# CLI 入口（scripts/ingest_corpus.py 调它）
# --------------------------------------------------------------------------- #


def summarize(articles: list[Article]) -> list[dict]:
    """按法规汇总条数与字数，供报表与自检用。"""
    by_law: dict[str, dict] = {}
    for a in articles:
        row = by_law.setdefault(
            a.law_id,
            {
                "law_id": a.law_id,
                "law_short": a.law_short,
                "n_articles": 0,
                "n_chars": 0,
                "effective_date": str(a.effective_date) if a.effective_date else None,
                "effective_date_source": a.effective_date_source,
                "with_chapter": 0,
            },
        )
        row["n_articles"] += 1
        row["n_chars"] += a.char_count
        if a.chapter:
            row["with_chapter"] += 1
    return list(by_law.values())


__all__ = [
    "Article",
    "LawMeta",
    "clean_corpus",
    "cn_to_int",
    "corpus_hash",
    "docx_to_text",
    "load_articles",
    "load_manifest",
    "parse_articles",
    "read_source",
    "summarize",
    "write_jsonl",
]
