"""评测层：检索指标（F23）→ 消融矩阵（F24）→ 引用/拒答（F25）→ 裁判（F26）→ 归因（F27）。

**这一层是项目最不可替代的资产。** 产品壳谁都能写，能零 LLM 成本、
程序化算出检索质量的地方不多——法律领域的法条编号让这件事成立。
"""

from __future__ import annotations

from juris_agent.eval.retrieval_eval import (
    EvalReport,
    GoldItem,
    Metrics,
    evaluate,
    expand_ranking,
    gold_hash,
    load_gold,
    score_one,
)

__all__ = [
    "EvalReport",
    "GoldItem",
    "Metrics",
    "evaluate",
    "expand_ranking",
    "gold_hash",
    "load_gold",
    "score_one",
]
