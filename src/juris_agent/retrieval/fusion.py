"""F7 多路召回融合：RRF 与加权求和，**两个都实现**。

面试官问"多路召回怎么合并"，标准答案不是"我用 RRF"，而是**"我为什么不用加权求和，
以及我测出来的差是多少"**。所以这里必须有两个实现和一条对照轴。

## 加权求和的根本问题：两个分数不同量纲，且量纲随查询变

BM25 的分数是**无上界**的（取决于词频和文档长度），余弦相似度在 `[-1, 1]`。
直接加权相加等于让 BM25 单方面决定排序。所以加权求和必须**先归一化**，常见做法是
每题 min-max 拉到 `[0,1]`：

    norm(s) = (s - min_s) / (max_s - min_s)

这个式子有两个致命处：

1. **min/max 是每题现算的**，所以"这一路第一名"永远被拉到 1.0，
   哪怕它的绝对分数低得可笑。BM25 在某题上全军覆没（最高分 0.3），
   归一化后照样产生一个 1.0 的"冠军"，然后以满权重混进来。
2. **归一化后的权重不再是权重的意义。** 你写 `0.5*BM25 + 0.5*Dense`，
   但实际影响力取决于两路各自的分数分布宽度——同一组权重，
   换一道题行为就变了。**权重不可移植。**

## RRF 为什么绕开了这个问题

    RRF(d) = Σ_i  w_i / (k + rank_i(d))

只用**名次**，不用分数。名次是无量纲的，两路天然可比；`k` 起到"削弱头部"
的作用（`k` 越小头部越强势，`k → ∞` 退化成按被召回次数排序）。

代价：**丢掉了分数的信息量**。一路"以 0.99 的相似度遥遥领先"和"以 0.51 险胜"
在 RRF 眼里完全一样。这是 RRF 用简单换鲁棒的自觉取舍。

所以本项目在 RRF **之后**接精排（F8）把细粒度排序补回来——
"粗排求全、精排求准"的分工，而不是指望融合层一次到位。
"""

from __future__ import annotations

from collections.abc import Sequence

# 文献与工程实践里的常用默认值。F24 会扫 {1, 10, 30, 60, 200} 出一张曲线，
# 用它来回答"k 该怎么选"而不是"我看别人写 60"。
DEFAULT_RRF_K = 60


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion。

    参数
    ----
    rankings
        每一路召回的结果，**按名次从好到坏**排列的 id 列表。同一 id 在列表内
        重复只算**首次出现的名次**（去重保首次）——重复计入会让某一路
        靠刷同一个 id 就抬高总分。
    k
        平滑常数。必须 > 0；`k=0` 时第一名得无穷大分，退化成"谁拿过第一谁赢"。
    weights
        可选的每路权重，长度须与 `rankings` 一致。**默认全 1**：
        RRF 的卖点之一就是不加权也够用，加权重是额外自由度，得有数字支持才加。

    返回 `[(id, 融合分)]`，按分数降序；分数相同则按 id 字典序（**保证可复现**，
    否则同分项的相对顺序会随字典插入顺序漂移，两次跑出的排名不一样）。
    """
    if k <= 0:
        raise ValueError(f"k 必须 > 0，收到 {k}")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError(f"weights 长度 {len(weights)} 与 rankings 路数 {len(rankings)} 不符")

    ws = list(weights) if weights is not None else [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranked, w in zip(rankings, ws):
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranked, start=1):  # 名次从 1 开始
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    # 丢掉总分为 0 的文档。**这不是微小的清理，是个真实的 bug 修复**：
    # 权重为 0 的那一路，它独有的文档得分正好是 0——但从分数上讲，"没被召回"
    # 和"被召回了但权重为 0"都会落到这里。前者本就不该出现，
    # 后者等价于那一路被关掉。所以 0 分 ⇒ 不该出现在结果里。
    #
    # 症状（这行代码之前）：`sparse_weight=0` 的臂与 `use_bm25=False` 的臂结果
    # 不一致——前者会多出一批只有 BM25 召回的条，于是"关掉一路"变成"降权但不关"，
    # 权重扫描那一组的自检行失效。测试 `test_zero_weight_turns_a_path_off` 盯着它。
    #
    # 注意加权求和**不**做这个过滤：那边 0 是 min-max 之后的"最后一名"，
    # 语义是"召回了但垫底"，与"没召回"是不同的事（见该函数 docstring）。
    return sorted(
        ((d, s) for d, s in scores.items() if s > 0.0),
        key=lambda kv: (-kv[1], kv[0]),
    )


def normalize_scores(pairs: Sequence[tuple[str, float]]) -> dict[str, float]:
    """把 `[(id, 原始分)]` min-max 拉到 `[0, 1]`。

    单元素或全等值时返回全 1.0 而不是全 0.0——`max == min` 时原式是 0/0。
    取 1.0 是"这一路只有它，它就代表这一路"，与 RRF 的行为一致。
    """
    if not pairs:
        return {}
    vals = [s for _, s in pairs]
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {i: 1.0 for i, _ in pairs}
    return {i: (s - lo) / (hi - lo) for i, s in pairs}


def weighted_sum_fuse(
    ranked_with_scores: Sequence[Sequence[tuple[str, float]]],
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """加权求和融合（**对照臂**，见模块 docstring 说明它为什么会输）。

    `ranked_with_scores` 每路是 `[(id, 原始分)]`，内部做 min-max 归一化。
    没被某一路召回的 id 在该路上记 0 分——注意这个 0 与"归一化后的最低分"
    是同一个值，所以"未召回"和"召回了但排最后"在这里**无法区分**，
    这是加权求和的又一处信息损失。
    """
    if weights is not None and len(weights) != len(ranked_with_scores):
        raise ValueError("weights 路数不匹配")
    ws = list(weights) if weights is not None else [1.0] * len(ranked_with_scores)
    total: dict[str, float] = {}
    for pairs, w in zip(ranked_with_scores, ws):
        for doc_id, s in normalize_scores(pairs).items():
            total[doc_id] = total.get(doc_id, 0.0) + w * s
    return sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))


FUSERS = {"rrf": rrf_fuse, "weighted_sum": weighted_sum_fuse}
"""消融里可选的两条融合路径。F24 的「RRF vs 加权求和」一轴就是切这个。"""


__all__ = [
    "DEFAULT_RRF_K",
    "FUSERS",
    "normalize_scores",
    "rrf_fuse",
    "weighted_sum_fuse",
]
