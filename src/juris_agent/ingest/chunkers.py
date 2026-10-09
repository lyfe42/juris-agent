"""F2 切分器：4 个臂，供消融矩阵对比「切分策略值多少分」。

面试官会问「你的切分策略怎么定的」——只说「我按条切」是答不上来的，得有一张对照表。
所以这里刻意实现四个粒度不同的切法，而不是一个好切法：

| 臂 | 切法 | 想验证的假设 |
|---|---|---|
| `fixed_512` | 定长 512 字、重叠 64，**不看条界** | 不尊重条界的切法会不会把条文劈开、拉低召回 |
| `by_article` | 一条一块 | 基础臂：粒度与 gold 完全一致，理论上占便宜 |
| `sentence_window` | 句滑窗（3 句 / 步长 1） | 更细的粒度会不会在短问题上更好 |
| `by_article_plus_context` | 一条一块，前置所属章标题 | 条文正文缺关键词时，章标题能不能补上 |

**每个 chunk 都带 `article_keys`**（它覆盖了哪些条），这是与 gold 对齐的接口——
检索命中一个 chunk，只要它覆盖的条里含 gold 条，就算命中。非按条切的两个臂
因此可能「一块覆盖多条」，这正是要观察的效应。

切分不改写原文一字：`text` 是原样的条文（`fixed_512` 会跨条，也照原样拼）。
"""

from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel

from juris_agent.ingest.clean import Article


class Chunk(BaseModel):
    chunk_id: str
    text: str
    article_keys: list[str]  # 覆盖的条文，用于与 gold 对齐
    primary_key: str  # 主条，用于引用展示
    law_id: str
    char_count: int = 0


_SENT_SPLIT = re.compile(r"(?<=[。；！？])")


def _sentences(text: str) -> list[str]:
    """按中文句末标点切句。保留标点，丢掉空白句。"""
    out: list[str] = []
    for para in text.split("\n"):
        for s in _SENT_SPLIT.split(para):
            s = s.strip()
            if s:
                out.append(s)
    return out


def by_article(articles: list[Article]) -> list[Chunk]:
    """一条一块。粒度与 gold 一致的基础臂。"""
    return [
        Chunk(
            chunk_id=f"{a.article_key}",
            text=a.text,
            article_keys=[a.article_key],
            primary_key=a.article_key,
            law_id=a.law_id,
            char_count=a.char_count,
        )
        for a in articles
    ]


def by_article_plus_context(articles: list[Article]) -> list[Chunk]:
    """一条一块，正文前加上所属章/节标题。

    动机是真实的：用户问「试用期最长多久」，条文正文里有「试用期」，
    但问「竞业限制人员范围」，条文用的是「高级管理人员」，而「竞业限制」这个词
    出现在章标题里——把标题补进去能补上这类词汇鸿沟。
    """
    chunks: list[Chunk] = []
    for a in articles:
        ctx = " › ".join(x for x in (a.chapter, a.section) if x)
        text = f"【{ctx}】\n{a.text}" if ctx else a.text
        chunks.append(
            Chunk(
                chunk_id=f"{a.article_key}+ctx",
                text=text,
                article_keys=[a.article_key],
                primary_key=a.article_key,
                law_id=a.law_id,
                char_count=len(text),
            )
        )
    return chunks


def fixed_window(
    articles: list[Article], size: int = 512, overlap: int = 64
) -> list[Chunk]:
    """定长窗口，**完全不看条界**——条文会被从中间劈开。

    实现要点：按法规拼成一整块文本，同时记住每条在整块里的字符区间；
    切窗口时，凡是与窗口有交集的条都记进 `article_keys`。这样"劈开的半条"
    也算覆盖了那条，不会因为切坏而对某个臂不公平。
    """
    if size <= overlap:
        raise ValueError(f"size({size}) 必须大于 overlap({overlap})，否则窗口不前进")

    by_law: dict[str, list[Article]] = defaultdict(list)
    for a in articles:
        by_law[a.law_id].append(a)

    chunks: list[Chunk] = []
    step = size - overlap
    for law_id, arts in by_law.items():
        arts = sorted(arts, key=lambda x: (x.article_num, x.article_sub or ""))
        blob_parts: list[str] = []
        spans: list[tuple[int, int, str]] = []
        pos = 0
        for a in arts:
            blob_parts.append(a.text)
            spans.append((pos, pos + len(a.text), a.article_key))
            pos += len(a.text) + 1  # 拼接用的 "\n"
        blob = "\n".join(blob_parts)

        for start in range(0, len(blob), step):
            end = min(start + size, len(blob))
            covered = [k for s, e, k in spans if s < end and e > start]
            if not covered:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{law_id}:fw{start}",
                    text=blob[start:end],
                    article_keys=covered,
                    primary_key=covered[0],
                    law_id=law_id,
                    char_count=end - start,
                )
            )
            if end >= len(blob):
                break
    return chunks


def sentence_window(
    articles: list[Article], window: int = 3, stride: int = 1
) -> list[Chunk]:
    """句滑窗。比 by_article 更细，用来观察"过细的粒度"是否伤害召回。"""
    if window < 1 or stride < 1:
        raise ValueError("window 与 stride 必须 >= 1")

    by_law: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for a in articles:
        for s in _sentences(a.text):
            by_law[a.law_id].append((a.article_key, s))

    chunks: list[Chunk] = []
    for law_id, sents in by_law.items():
        for i in range(0, len(sents), stride):
            block = sents[i : i + window]
            if not block:
                break
            keys: list[str] = []
            for k, _ in block:
                if k not in keys:
                    keys.append(k)
            text = "".join(s for _, s in block)
            chunks.append(
                Chunk(
                    chunk_id=f"{law_id}:sw{i}",
                    text=text,
                    article_keys=keys,
                    primary_key=keys[0],
                    law_id=law_id,
                    char_count=len(text),
                )
            )
            if i + window >= len(sents):
                break
    return chunks


CHUNKERS = {
    "fixed_512": fixed_window,
    "by_article": by_article,
    "sentence_window": sentence_window,
    "by_article_plus_context": by_article_plus_context,
}
"""消融用的名字 → 函数。`configs/ablation.yaml` 里就写这些名字。"""


__all__ = [
    "CHUNKERS",
    "Chunk",
    "by_article",
    "by_article_plus_context",
    "fixed_window",
    "sentence_window",
]
