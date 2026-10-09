"""F23 检索指标：**两种 Recall 一起报**，外加 MRR@10 与 nDCG@5。

**为什么必须有两种 Recall**

只报「命中即算」的 Recall 会让多条款组合题完全看不见。举个数：
一条题的 gold 有 3 条（第四十七条、第四十六条、第八十七条），系统只召回其中 1 条——
- 命中口径 `Recall@k_hit`：命中了 → 记 **1.0**，看起来完美；
- 覆盖口径 `Recall@k_art`：1/3 → 记 **0.33**，暴露了漏掉两条。

同一份结果，两个数差了 3 倍。**只报前者就是在骗自己。** 而 gold 恰好是法条编号列表，
两个口径都能零 LLM 成本地算出来——这是选法律领域当载体的核心收益。

**两个口径的定义**
- `Recall@k_hit`：top-k 里**只要有一条** gold 就算命中（1/0 平均）。衡量"有没有找到门"。
- `Recall@k_art`：top-k 覆盖了 gold 的**比例**（逐题求覆盖率再平均）。衡量"找全了没有"。

**MRR@10 未命中记 0，不跳过**——跳过未命中会把困难题从分母里剔掉，指标虚高。
这是排序指标最常见的实现错误，测试里有一条专门盯它。

**nDCG@5** 用**条级二元相关**：把 chunk 排名展开成条级排名（见 `expand_ranking`），
相关 = 该条在 gold 里；IDCG 按 `min(k, |gold|)` 个理想相关位算。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from juris_agent.ingest.chunkers import Chunk

DEFAULT_KS: tuple[int, ...] = (1, 5, 10)


# --------------------------------------------------------------------------- #
# gold
# --------------------------------------------------------------------------- #


class GoldItem(BaseModel):
    """一条评测题。`gold` 是法条编号列表——这是"答案天然已知"的体现。"""

    id: str
    query: str
    gold: list[str]
    qtype: str = "single_fact"  # single_fact / multi_article / calculation / temporal
    difficulty: str = "normal"  # easy / normal / hard
    # machine = LLM 反向构造；human = 人工口语改写。**必须分开报**——
    # machine 造的题措辞贴着原文，指标天然偏高（见 need.md F4 的说明）。
    source: str = "machine"
    # dev/test 严格隔离：调参只看 dev，报数只认 test。
    # 不隔离的话，反复在同一个集合上调 knobs（RRF 的 k、融合权重），
    # 迟早会把这个集合"调熟"——指标涨了，泛化没涨。
    split: str = "dev"
    note: str = ""


def load_gold(path: Path) -> list[GoldItem]:
    items = [GoldItem(**json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    seen = {i.id for i in items}
    if len(seen) != len(items):
        raise ValueError("gold 里有重复 id——评测集会重复计分")
    return items


def gold_hash(items: Iterable[GoldItem]) -> str:
    h = hashlib.sha256()
    for i in sorted(items, key=lambda x: x.id):
        h.update(f"{i.id}|{i.query}|{','.join(i.gold)}\n".encode("utf-8"))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 排名展开：chunk 级 → 条级
# --------------------------------------------------------------------------- #


def expand_ranking(chunks: Sequence[Chunk], max_keys: int | None = None) -> list[str]:
    """把检索到的 chunk 序列展开成**条级**排名（去重，保留首次出现的名次）。

    非按条切的臂（`fixed_512`）一块可能覆盖多条，展开后一条 chunk 会贡献多个条级名次。
    这是有意为之：块大就该看到它覆盖得多。代价（顺便覆盖了无关的条、注入了更多 token）
    由精排和成本报表体现，不由这个指标体现——**别把它当成对按条切的公平性声明**。
    """
    keys: list[str] = []
    seen: set[str] = set()
    for c in chunks:
        for k in c.article_keys:
            if k not in seen:
                seen.add(k)
                keys.append(k)
                if max_keys is not None and len(keys) >= max_keys:
                    return keys
    return keys


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #


def _ndcg_at(ranked: Sequence[str], gold: set[str], k: int) -> float:
    import math

    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, key in enumerate(ranked[:k], start=1)
        if key in gold
    )
    ideal_hits = min(k, len(gold))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def score_one(
    gold: Iterable[str], ranked: Sequence[str], ks: Sequence[int] = DEFAULT_KS, mrr_k: int = 10
) -> dict[str, float]:
    """单题打分。`ranked` 是条级排名（用 `expand_ranking` 得到）。"""
    gold_set = set(gold)
    out: dict[str, float] = {}
    for k in ks:
        top = ranked[:k]
        out[f"recall_hit@{k}"] = 1.0 if any(x in gold_set for x in top) else 0.0
        out[f"recall_art@{k}"] = (len(set(top) & gold_set) / len(gold_set)) if gold_set else 0.0
    # MRR：未命中记 0（不是跳过）
    rr = 0.0
    for i, key in enumerate(ranked[:mrr_k], start=1):
        if key in gold_set:
            rr = 1.0 / i
            break
    out[f"mrr@{mrr_k}"] = rr
    out["ndcg@5"] = _ndcg_at(ranked, gold_set, 5)
    return out


class Metrics(BaseModel):
    n: int
    values: dict[str, float]

    @classmethod
    def mean_of(cls, rows: list[dict[str, float]]) -> "Metrics":
        if not rows:
            return cls(n=0, values={})
        keys = rows[0].keys()
        return cls(n=len(rows), values={k: sum(r[k] for r in rows) / len(rows) for k in keys})


class EvalReport(BaseModel):
    overall: Metrics
    by_difficulty: dict[str, Metrics] = Field(default_factory=dict)
    by_qtype: dict[str, Metrics] = Field(default_factory=dict)
    by_source: dict[str, Metrics] = Field(default_factory=dict)
    # dev/test 切片。**必须落在报告里，不能只在命令行上过滤**：
    # `--split dev` 能把数据滤干净，但输出里看不出这一行是哪种切片，
    # 于是 dev 上调参的 0.86 和 test 上报数的 0.79 会被并排贴进同一张表，
    # 谁也没法看出其中一个是"见过的题"。按切片同时报出来，就没法这么糊。
    by_split: dict[str, Metrics] = Field(default_factory=dict)
    meta: dict = Field(default_factory=dict)

    def to_markdown(self, title: str) -> str:
        """产出消融表的一行/一表。列名固定，方便多条臂拼成一张矩阵。"""
        cols = ["recall_hit@1", "recall_hit@5", "recall_hit@10",
                "recall_art@5", "recall_art@10", "mrr@10", "ndcg@5"]
        lines = [f"## {title}", ""]
        if self.meta:
            lines.append("meta: " + ", ".join(f"`{k}={v}`" for k, v in self.meta.items()))
            lines.append("")
        header = "| 切片 | n | " + " | ".join(c.split("@")[0] + "@" + c.split("@")[1] for c in cols) + " |"
        lines.append(header)
        lines.append("|---" * (len(cols) + 2) + "|")

        def row(name: str, m: Metrics) -> str:
            if m.n == 0:
                return f"| {name} | 0 | " + " | ".join("—" for _ in cols) + " |"
            cells = " | ".join(f"{m.values.get(c, 0):.3f}" for c in cols)
            return f"| {name} | {m.n} | {cells} |"

        lines.append(row("**全部**", self.overall))
        for label, group in (
            ("难度", self.by_difficulty),
            ("题型", self.by_qtype),
            ("来源", self.by_source),
            ("切片", self.by_split),
        ):
            for k, m in sorted(group.items()):
                lines.append(row(f"{label}:{k}", m))
        lines.append("")
        return "\n".join(lines)


def evaluate(
    gold_items: Sequence[GoldItem],
    rankings: dict[str, Sequence[str]],
    ks: Sequence[int] = DEFAULT_KS,
    meta: dict | None = None,
) -> EvalReport:
    """`rankings` 是 `{gold_id: 条级排名}`。缺 id 的题按"零召回"计入，**不静默跳过**。"""
    per_query: list[tuple[GoldItem, dict[str, float]]] = []
    for g in gold_items:
        ranked = rankings.get(g.id, [])
        per_query.append((g, score_one(g.gold, ranked, ks)))

    def group(attr: str) -> dict[str, Metrics]:
        buckets: dict[str, list[dict[str, float]]] = {}
        for g, v in per_query:
            buckets.setdefault(getattr(g, attr), []).append(v)
        return {k: Metrics.mean_of(v) for k, v in sorted(buckets.items())}

    return EvalReport(
        overall=Metrics.mean_of([v for _, v in per_query]),
        by_difficulty=group("difficulty"),
        by_qtype=group("qtype"),
        by_source=group("source"),
        by_split=group("split"),
        meta=meta or {},
    )


__all__ = [
    "DEFAULT_KS",
    "EvalReport",
    "GoldItem",
    "Metrics",
    "evaluate",
    "expand_ranking",
    "gold_hash",
    "load_gold",
    "score_one",
]
