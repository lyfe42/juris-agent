"""F10 七段检索编排 + 埋点。

   改写 → BM25 → 稠密 → 融合 → 精排 → 邻条扩展 → 去重

## 为什么要"编排"而不是把这几段写成一坨

**每一段都要能被单独关掉。** 这是消融矩阵（F24）的前提——如果组件是硬连的，
就只能测"全开 vs 全关"，测不出"哪一段贡献了多少"。所以这里用 `ArmConfig`
描述一条臂，`build_pipeline` 按配置装配，消融 runner 只负责枚举配置。

## 为什么要埋点（而不是只看总耗时）

面试问"一次检索耗时花在哪"，答案是**分段报出来的**
（BM25 3ms / Dense 12ms / 融合 1ms / 精排 310ms …），不是"大概 300 多毫秒"。
埋点还顺带解决一个排查问题：某次检索结果为空，是**没有候选**（检索层的问题）
还是**候选被过滤光了**（过滤层的问题）——`n_in/n_out` 一眼看出来。

## 邻条扩展

命中了第四十七条，往往还需要第四十六条（"什么情况下给补偿"）和第四十八条。
法条不是孤立的，**"相邻条"是最廉价、最不需要模型的上下文扩展**——
比任何 rerank 都便宜，而且能补上 gold 里"应当同时引用"的那些条。

代价是注入 prompt 的 token 变多、噪声变多，所以要能关（`neighbor_window=0`），
并且**扩展来的条在排序上排在原始命中之后**——不能让"顺带捞的"挤掉"真正命中的"。
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from juris_agent.ingest.chunkers import Chunk
from juris_agent.ingest.clean import Article
from juris_agent.retrieval.fusion import DEFAULT_RRF_K, FUSERS, rrf_fuse
from juris_agent.retrieval.rerank import Reranker
from juris_agent.retrieval.rewrite import IdentityRewriter, Rewriter, SynonymRewriter, SynonymTable


# --------------------------------------------------------------------------- #
# 一条臂的配置
# --------------------------------------------------------------------------- #


class ArmConfig(BaseModel):
    """描述消融矩阵里的一行。**只用数据描述，不含任何逻辑。**

    这样矩阵可以写成 `configs/ablation.yaml`，一行一个 `ArmConfig`；
    也可以由 runner 程序化枚举。两种方式产出的是同一个对象。
    """

    name: str = "arm"

    # 切分臂
    chunker: str = "by_article"

    # 召回路（可多选）
    use_bm25: bool = True
    use_dense: bool = False
    use_dict: bool = True  # BM25 的法律领域词典开关

    # 融合
    fusion: str = "rrf"  # rrf | weighted_sum
    rrf_k: int = DEFAULT_RRF_K
    # 每一路在融合里的权重。**默认 1.0 = 等权**，但等权是有前提的：
    # 两路的质量得差不多。实测（269 题）稠密单独 hit@5=0.907、BM25 单独 0.606，
    # 等权融合只有 0.866——**加一路更弱的检索器把更强的那个拖下去了 4 个点**，
    # 且 MRR@10 随 BM25 权重单调下降（0.768 → 0.689）。
    # 所以这两个权重不是摆设，是消融里必须扫的一轴（见 arms_weighting）。
    sparse_weight: float = 1.0
    dense_weight: float = 1.0

    # 精排
    use_rerank: bool = False

    # 改写
    rewrite: str = "none"  # none | synonym | llm

    # 邻条扩展
    neighbor_window: int = 0

    # 候选规模
    top_k: int = 20  # 每一路召回多少
    fused_k: int = 20  # 融合后留多少交给精排

    # 稠密后端（exact | faiss_ivf）
    dense_backend: str = "exact"

    def describe(self) -> str:
        """人类可读的一行标签，直接进消融表。"""
        parts = [self.chunker]
        paths = []
        if self.use_bm25:
            paths.append("bm25" + ("" if self.use_dict else "-nodict"))
        if self.use_dense:
            paths.append(f"dense[{self.dense_backend}]")
        parts.append("+".join(paths) or "none")
        if len(paths) > 1:
            parts.append(f"|{self.fusion}(k={self.rrf_k})")
            if self.sparse_weight != 1.0 or self.dense_weight != 1.0:
                parts.append(f"w={self.sparse_weight:g}/{self.dense_weight:g}")
        if self.use_rerank:
            parts.append("|rerank")
        if self.rewrite != "none":
            parts.append(f"|rw:{self.rewrite}")
        if self.neighbor_window:
            parts.append(f"|nb{self.neighbor_window}")
        return " ".join(parts)


# --------------------------------------------------------------------------- #
# 埋点
# --------------------------------------------------------------------------- #


class StageStat(BaseModel):
    """一段的耗时与候选数。`n_in/n_out` 是排查"结果为什么空了"的第一手证据。"""

    stage: str
    elapsed_ms: float
    n_in: int = 0
    n_out: int = 0
    note: str = ""


class PipelineResult(BaseModel):
    chunks: list[Chunk]
    stages: list[StageStat] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def elapsed_ms(self) -> float:
        return sum(s.elapsed_ms for s in self.stages)

    def expand_article_keys(self) -> list[str]:
        """chunk 级结果 → 条级排名（去重保首次）。评测层要的就是这个。"""
        keys: list[str] = []
        seen: set[str] = set()
        for c in self.chunks:
            for k in c.article_keys:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        return keys


class _Timer:
    """`with _Timer() as t:` 的极简实现，用来避免每段都写两遍 `perf_counter`。"""

    def __enter__(self) -> "_Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = (time.perf_counter() - self.t0) * 1000


# --------------------------------------------------------------------------- #
# 邻条扩展
# --------------------------------------------------------------------------- #


def expand_neighbors(
    chunks: list[Chunk], articles_by_key: dict[str, Article], window: int
) -> list[Chunk]:
    """把命中条文的前后 N 条**追加**到结果尾部（不改变已有顺序）。

    `Article` 里存的是整条正文，这里临时包成 `Chunk`——它的 `article_keys`
    只有自己一条，所以不会污染评测的条级排名口径。
    """
    if window <= 0:
        return chunks
    have = {c.primary_key for c in chunks}
    extra: list[Chunk] = []
    for c in list(chunks):  # 注意：基于**原始命中**向外扩，不是边扩边看
        base = articles_by_key.get(c.primary_key)
        if base is None:
            continue
        for delta in range(-window, window + 1):
            if delta == 0:
                continue
            target = f"{base.law_id}#{base.article_num + delta}"
            if target in have or target in {e.primary_key for e in extra}:
                continue
            art = articles_by_key.get(target)
            if art is None:
                continue
            extra.append(
                Chunk(
                    chunk_id=f"{target}+nb",
                    text=art.text,
                    article_keys=[target],
                    primary_key=target,
                    law_id=art.law_id,
                    char_count=art.char_count,
                )
            )
    return chunks + extra


# --------------------------------------------------------------------------- #
# 管道
# --------------------------------------------------------------------------- #


class RetrievalPipeline:
    """把各段装起来。**每段都允许为 None**——None 就是"这一臂关掉了这段"。"""

    def __init__(
        self,
        config: ArmConfig,
        sparse: Any | None = None,
        dense: Any | None = None,
        reranker: Reranker | None = None,
        rewriter: Rewriter | None = None,
        articles_by_key: dict[str, Article] | None = None,
    ) -> None:
        self.config = config
        self.sparse = sparse
        self.dense = dense
        self.reranker = reranker
        self.rewriter = rewriter or IdentityRewriter()
        self.articles_by_key = articles_by_key or {}
        if not sparse and not dense:
            raise ValueError("至少要有一路召回（sparse 或 dense）")

    def search(self, query: str) -> PipelineResult:
        cfg = self.config
        stages: list[StageStat] = []
        warnings: list[str] = []

        # --- 1. 改写 ---------------------------------------------------- #
        with _Timer() as t:
            variants = self.rewriter.rewrite(query)
        stages.append(StageStat(stage="rewrite", elapsed_ms=t.ms, n_in=1, n_out=len(variants)))

        # --- 2/3. 多路召回（每个变体各查一遍） ---------------------------- #
        def retrieve_one(retriever: Any, stage_name: str) -> list[list[tuple[Chunk, float]]]:
            """每个变体一路结果。返回 `[[(chunk, score), ...], ...]`。"""
            per_variant: list[list[tuple[Chunk, float]]] = []
            with _Timer() as t:
                for v in variants:
                    per_variant.append(retriever.search(v, top_k=cfg.top_k))
            stages.append(
                StageStat(
                    stage=stage_name,
                    elapsed_ms=t.ms,
                    n_in=len(variants),
                    n_out=len({c.chunk_id for r in per_variant for c, _ in r}),
                    note=f"{len(variants)} 个查询变体" if len(variants) > 1 else "",
                )
            )
            return per_variant

        sparse_per_variant = retrieve_one(self.sparse, "bm25") if self.sparse else None
        dense_per_variant = retrieve_one(self.dense, "dense") if self.dense else None

        # --- 4. 融合 ---------------------------------------------------- #
        with _Timer() as t:
            fused = self._fuse(sparse_per_variant, dense_per_variant)
        stages.append(
            StageStat(stage=f"fuse/{cfg.fusion}", elapsed_ms=t.ms, n_in=0, n_out=len(fused))
        )
        by_id = {c.chunk_id: c for c in self._all_chunks(sparse_per_variant, dense_per_variant)}
        candidates = [by_id[cid] for cid, _ in fused if cid in by_id]

        # --- 5. 精排（含超时降级） ---------------------------------------- #
        if self.reranker is not None:
            r = self.reranker.rerank(query, candidates, top_k=cfg.fused_k)
            candidates = r.chunks
            if r.warning:
                warnings.append(r.warning)
            stages.append(
                StageStat(
                    stage="rerank",
                    elapsed_ms=r.elapsed_ms,
                    n_in=r.n_candidates,
                    n_out=len(r.chunks),
                    note="降级" if r.degraded else "",
                )
            )
        else:
            candidates = candidates[: cfg.fused_k]

        # --- 6. 邻条扩展 ------------------------------------------------- #
        if cfg.neighbor_window:
            with _Timer() as t:
                before = len(candidates)
                candidates = expand_neighbors(candidates, self.articles_by_key, cfg.neighbor_window)
            stages.append(
                StageStat(
                    stage="neighbor",
                    elapsed_ms=t.ms,
                    n_in=before,
                    n_out=len(candidates),
                    note=f"±{cfg.neighbor_window} 条",
                )
            )

        # --- 7. 去重 ------------------------------------------------------ #
        with _Timer() as t:
            before = len(candidates)
            seen_id: set[str] = set()
            seen_primary: set[str] = set()
            deduped: list[Chunk] = []
            for c in candidates:
                # 两级去重：chunk_id 去重同一块（多个变体可能命中同一块），
                # primary_key 去重同一**条**被不同块覆盖（按条切 + 邻条扩展时会撞）
                if c.chunk_id in seen_id or c.primary_key in seen_primary:
                    continue
                seen_id.add(c.chunk_id)
                seen_primary.add(c.primary_key)
                deduped.append(c)
        stages.append(
            StageStat(stage="dedup", elapsed_ms=t.ms, n_in=before, n_out=len(deduped))
        )

        return PipelineResult(chunks=deduped, stages=stages, warnings=warnings)

    # -- 内部 ------------------------------------------------------------- #

    @staticmethod
    def _all_chunks(*per_variant_groups: Any) -> list[Chunk]:
        out: list[Chunk] = []
        for group in per_variant_groups:
            if not group:
                continue
            for per_variant in group:
                for c, _ in per_variant:
                    out.append(c)
        return out

    def _fuse(self, sparse: Any, dense: Any) -> list[tuple[str, float]]:
        """把"每个查询变体 × 每一路"的所有排名一起融合。

        **变体之间也走融合，而不是取并集。** 原因：同一 chunk 被多个变体命中，
        说明它更可能是真的相关——这是个免费的置信度信号，并集会丢掉它。

        权重按**路**给（同一路的多个变体共享一个权重），不是按变体给：
        "这一路有多可信"是关于**检索器**的判断，不是关于某条改写式的判断。
        """
        cfg = self.config
        rankings: list[list[str]] = []
        ranked_with_scores: list[list[tuple[str, float]]] = []
        weights: list[float] = []
        for group, w in ((sparse, cfg.sparse_weight), (dense, cfg.dense_weight)):
            if not group:
                continue
            for per_variant in group:
                rankings.append([c.chunk_id for c, _ in per_variant])
                ranked_with_scores.append([(c.chunk_id, s) for c, s in per_variant])
                weights.append(w)
        if not rankings:
            return []
        if cfg.fusion == "rrf":
            # 用配置里的 k，而不是 rrf_fuse 的默认值——消融要扫 k
            return rrf_fuse(rankings, k=cfg.rrf_k, weights=weights)
        fuser = FUSERS.get(cfg.fusion)
        if fuser is None:
            raise ValueError(f"未知融合方式 {cfg.fusion!r}，可选 {sorted(FUSERS)}")
        return fuser(ranked_with_scores, weights=weights)


__all__ = [
    "ArmConfig",
    "PipelineResult",
    "RetrievalPipeline",
    "StageStat",
    "expand_neighbors",
]
