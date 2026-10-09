"""检索层：稠密（F5）/ BM25（F6）→ RRF 融合（F7）→ 精排（F8）→ 改写（F9）→ 编排（F10）。

同一段文本会有两套 id 体系，别混淆：
- `article_key`（`lcl#47`）：**条**级主键，跨模块稳定，gold 标注与引用校验用它。
- `chunk_id`（`lcl#47` 或 `lcl:fw1024`）：**块**级主键，随切分策略而变。

检索在 chunk 上排序，评测在 article 上算分——两者靠 `Chunk.article_keys` 对齐。

## 这个 `__init__` 刻意只导出"轻"的东西

`dense`（要 torch）与 `rerank`（要 transformers）**不在这里导出**，
`build_pipeline` 也不导出。原因：`import juris_agent.retrieval` 应该是廉价的——
BM25 那条臂不需要 2.5GB 的 ml extra 就能跑，评测链路上最便宜的部分
不该被最贵的部分拖累。要用稠密/精排就显式
`from juris_agent.retrieval.dense import DenseRetriever`。

（这里曾经写过一版把所有东西都 re-export 的 `__init__`，结果是纯 BM25 的评测
也要先 import torch 再等十秒。删掉了。）
"""

from __future__ import annotations

from juris_agent.retrieval.bm25 import BM25Retriever, Tokenizer, load_legal_dict
from juris_agent.retrieval.fusion import (
    DEFAULT_RRF_K,
    FUSERS,
    normalize_scores,
    rrf_fuse,
    weighted_sum_fuse,
)
from juris_agent.retrieval.pipeline import (
    ArmConfig,
    PipelineResult,
    RetrievalPipeline,
    StageStat,
    expand_neighbors,
)
from juris_agent.retrieval.rewrite import (
    REWRITERS,
    IdentityRewriter,
    LLMRewriter,
    SynonymRewriter,
    SynonymTable,
)

__all__ = [
    "DEFAULT_RRF_K",
    "FUSERS",
    "REWRITERS",
    "ArmConfig",
    "BM25Retriever",
    "IdentityRewriter",
    "LLMRewriter",
    "PipelineResult",
    "RetrievalPipeline",
    "StageStat",
    "SynonymRewriter",
    "SynonymTable",
    "Tokenizer",
    "expand_neighbors",
    "load_legal_dict",
    "normalize_scores",
    "rrf_fuse",
    "weighted_sum_fuse",
]
