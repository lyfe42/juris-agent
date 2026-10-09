"""按 `ArmConfig` 装配一条检索臂。**消融矩阵唯一需要调用的装配入口。**

把装配逻辑集中在这里，是为了让消融 runner 只有"枚举配置 + 跑 + 收数字"三件事，
不掺任何组件构造细节。否则每加一个开关就要改一次 runner，
矩阵会慢慢腐烂成一份谁也不敢动的脚本。

## 关于缓存复用

`build_pipeline` 接收可选的 `_shared` 缓存字典，用来在一次消融跑多个臂时
**复用同一份编码向量和同一个 BM25 索引**。这不是过早优化：

- 稠密编码是最贵的一段（1700 条 × N 个切分臂）。同一个切分臂下有 5 条臂
  只在融合/精排上不同，若每条都重新编码，消融就从 1 分钟变成 5 分钟。
- BM25 建索引虽然便宜，但**复用能保证完全一致**——重建一次理论上结果相同，
  实际上 `rank_bm25` 内部有浮点累加顺序的差异，白给消融引入噪声。

缓存键必须包含 `(chunker, 组件, 语料指纹)`。`corpus_hash` 漏掉就会读到
改语料之前的旧向量——**指标看起来正常，其实算的是别的东西**。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from juris_agent.config import get_settings
from juris_agent.ingest.chunkers import CHUNKERS
from juris_agent.ingest.clean import Article, load_articles
from juris_agent.retrieval.dense import DenseRetriever, EmbeddingCache, Encoder
from juris_agent.retrieval.pipeline import ArmConfig, RetrievalPipeline
from juris_agent.retrieval.rerank import Reranker
from juris_agent.retrieval.rewrite import IdentityRewriter, LLMRewriter, SynonymRewriter, SynonymTable


class ArmDeps:
    """装一条臂需要的外部资源。显式传进来，方便测试里换成假的。

    惰性初始化：`encoder`、`reranker`、`llm` 都是"用到才建"。
    纯 BM25 的臂因此不会去加载 torch，也不会去读 `.env` 里的 key——
    **评测链路里最便宜的那几条臂应该能在没装 ml extra 的机器上跑通。**
    """

    def __init__(
        self,
        articles: list[Article] | None = None,
        *,
        dict_path: Path | None = None,
        corpus_hash: str = "",
        llm: Any | None = None,
        skip_llm: bool = False,
    ) -> None:
        s = get_settings()
        self.articles = articles if articles is not None else load_articles(s.paths.laws)
        self.dict_path = dict_path or (s.paths.configs / "legal_dict.txt")
        self.synonyms_path = s.paths.configs / "synonyms.yaml"
        self.corpus_hash = corpus_hash
        self.articles_by_key = {a.article_key: a for a in self.articles}
        self.cache = EmbeddingCache(s.paths.index / "emb")

        self._encoder: Encoder | None = None
        self._reranker: Reranker | None = None
        self._llm = llm
        self._skip_llm = skip_llm

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            s = get_settings()
            self._encoder = Encoder(s.retrieval.embedding_model)
        return self._encoder

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            s = get_settings()
            self._reranker = Reranker(timeout_ms=s.retrieval.rerank_timeout_ms,
                                      batch_size=s.retrieval.rerank_batch_size)
        return self._reranker

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from juris_agent.llm import LLMClient

            self._llm = LLMClient()
        return self._llm

    def build_rewriter(self, name: str):
        if name == "none":
            return IdentityRewriter()
        if name == "synonym":
            return SynonymRewriter(SynonymTable.load(self.synonyms_path))
        if name == "llm":
            if self._skip_llm:
                raise RuntimeError("这一臂需要 LLM 改写，但被 --skip-llm 禁用了")
            return LLMRewriter(self.llm)
        raise ValueError(f"未知改写方式 {name!r}")


def build_pipeline(
    config: ArmConfig,
    deps: ArmDeps,
    *,
    artifacts: dict[str, Any] | None = None,
) -> RetrievalPipeline:
    """按配置装配。`artifacts` 是跨臂复用的缓存（见模块 docstring）。"""
    store = artifacts if artifacts is not None else {}
    chunks = store.get(("chunks", config.chunker))
    if chunks is None:
        chunks = CHUNKERS[config.chunker](deps.articles)
        store[("chunks", config.chunker)] = chunks

    sparse = None
    if config.use_bm25:
        from juris_agent.retrieval.bm25 import BM25Retriever, Tokenizer

        key = ("bm25", config.chunker, config.use_dict)
        sparse = store.get(key)
        if sparse is None:
            tok = Tokenizer(dict_path=deps.dict_path, use_dict=config.use_dict)
            sparse = BM25Retriever(chunks, tok)
            store[key] = sparse

    dense = None
    if config.use_dense:
        key = ("dense", config.chunker, config.dense_backend)
        dense = store.get(key)
        if dense is None:
            dense = DenseRetriever(
                chunks,
                deps.encoder,
                backend=config.dense_backend,
                chunker_name=config.chunker,
                corpus_hash=deps.corpus_hash,
                cache=deps.cache,
            )
            store[key] = dense

    reranker = deps.reranker if config.use_rerank else None
    rewriter = deps.build_rewriter(config.rewrite)

    return RetrievalPipeline(
        config,
        sparse=sparse,
        dense=dense,
        reranker=reranker,
        rewriter=rewriter,
        articles_by_key=deps.articles_by_key,
    )


__all__ = ["ArmConfig", "ArmDeps", "build_pipeline"]
