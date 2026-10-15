"""把「检索管道」接到「Agent 的工具」上的那根线（F13）。

## 为什么需要这一层

现在项目里有两种人对检索的说法不一样：

- **评测层**要的是 `PipelineResult.chunks`——它只关心**顺序**（排名对不对）。
- **`search_law` 工具**要的是 `list[dict]`，字段名必须对得上 `LawArticleHit`
  （`article_key` / `law` / `article_no` / `text` / `score` / …），因为它要
  序列化成 JSON 喂给大模型，模型只能看见 JSON 字段。

如果让 `RetrievalPipeline` 直接吐 `dict`，评测层就得反过来把 dict 还原成对象——
两边都被对方的格式污染。所以中间加一个**适配器**：管道只管检索，
适配器只管翻译。这就是设计模式里说的 Adapter，但这里不用记这个名字，
记住"**让检索层不认识 LLM，让工具层不认识 Chunk**"就够了。

## ⚠️ 分数（score）这一栏有三个坑，必须说清楚

1. **它不是相似度。** 典型的 RRF 融合分长这样：`1/(60+1) = 0.0164`、
   `1/(60+2) = 0.0161`——数字小得离谱，而且 **0.0164 和 0.0161 只差 2%**，
   但那是第 1 名和第 2 名的差别。所以它**只能比大小，不能当"相关度 90%"读**。
   界面上如果要显示"相关度"，得先做归一化（本文件给了 `normalize_scores`）。
2. **它跨臂不可比。** 开了精排是 CrossEncoder 分（可正可负、无上界），
   没开精排是倒数排名分（恒正、恒小于 1/60）。同一个问题在两条臂下分数完全不同，
   **不能拿 A 臂的 0.9 和 B 臂的 0.02 比谁更相关**。
3. **0.0 是一个哨兵值，不是"不相关"。** 邻条扩展（`neighbor_window`）带出来的条文
   本来就没有分数——它是"顺带捞的上下文"，不是检索命中的。这里给它 0.0，
   并在 `note` 里标明来源，免得下游把它当成"得分最低的命中"。
"""

from __future__ import annotations

from typing import Any

from juris_agent.ingest.clean import Article
from juris_agent.retrieval.pipeline import PipelineResult, RetrievalPipeline


def normalize_scores(scores: list[float]) -> list[float]:
    """把一组分数线性拉到 `[0, 1]`，**只为了显示**。

    两种退化情况必须处理，否则界面会显示 `NaN` 或 `-0.0` 这种鬼东西：

    - **全部相等**（比如整条臂没开精排、或只有一条结果）：没有"谁更高"这回事，
      统一给 1.0。而不是除以 0。
    - **只有一个元素**：同上，给 1.0。

    ⚠️ 归一化**会丢掉绝对信息**：跟参考库不匹配（分数全在 0.1 附近）本来是个
    "这次检索质量很差"的信号，归一化之后第一名照样是 1.0。所以
    **归一化只能用于展示，不能用于设阈值判断"检索是否可信"**——那个判断要用原始分数。
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


class PipelineRetriever:
    """`search_law` 工具要的那种检索器：`search(query, top_k) -> list[dict]`。

    它同时满足 `juris_agent.tools.builtin.Retriever` 这个 Protocol
    （结构类型，不用显式继承——只要方法签名对得上就行）。

    **不做缓存、不做重试、不做熔断**——那些是 `tools/executor.py` 的职责。
    这一层故意保持"哑"：给什么查什么，查完如实翻译。责任分清楚，
    出问题时才能一眼看出是"检索不准"还是"工具层没兜住"。
    """

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        articles_by_key: dict[str, Article],
        *,
        min_score: float | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.articles_by_key = articles_by_key
        # 低于这个分数的结果直接丢掉。**默认 None = 不过滤**——
        # 理由：这个语料只有 1730 条，两路召回 + 精排之后留下的本来就不多，
        # 再加一道阈值等于凭空制造"检索为空"的失败。等有真实日志了再按日志定阈值。
        self.min_score = min_score
        # 最近一次检索的埋点，给可观测性用（F28）。不上锁：每个请求一个 retriever 实例，
        # 跨请求共享一个实例时才需要加锁，而那种用法本项目不用。
        self.last_stages: list[dict[str, Any]] = []
        self.last_warnings: list[str] = []

    # -- 主体 --------------------------------------------------------------- #

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """检索并翻译成 `LawArticleHit` 能直接 `model_validate` 的字典。

        `top_k` 是**返回条数**，而管道里的 `cfg.top_k` 是**每路召回条数**。
        两者是两回事：管道拿 20 条去融合、精排，最后这里只要前 8 条。
        所以正常的调用参数是"管道 wide、这里 narrow"——**召回要宽，注入要窄**。
        """
        if not query or not query.strip():
            # 空查询直接返回空，不去打扰 BM25 / 向量模型。
            # 这不是"提前优化"，是因为 BM25 对空串会返回全 0 分的一堆结果，
            # 那些结果会以"得分最低的命中"混进上下文，比空结果更糟。
            self.last_stages = []
            self.last_warnings = ["查询为空，未执行检索"]
            return []

        result: PipelineResult = self.pipeline.search(query)
        self.last_stages = [s.model_dump() for s in result.stages]
        self.last_warnings = list(result.warnings)

        out: list[dict[str, Any]] = []
        for c in result.chunks:
            if len(out) >= top_k:
                break
            art = self.articles_by_key.get(c.primary_key)
            if art is None:
                # 语料换过、而索引是旧的，就会走到这里。**不静默跳过**——
                # 静默跳过会让"检索到了 8 条、注入只有 5 条"看起来像是模型挑的。
                self.last_warnings.append(
                    f"索引里的 {c.primary_key} 在当前语料中不存在（索引与语料不同步，请重建索引）"
                )
                continue

            score = result.rank_scores.get(c.chunk_id, 0.0)
            if self.min_score is not None and score < self.min_score and score != 0.0:
                continue

            out.append(
                {
                    "article_key": art.article_key,
                    "law": art.law_short,
                    # 中文条号原文（`第四十七条`），给模型看的
                    "article_no": art.article_no,
                    # 邻条扩展来的：分数是哨兵 0.0，注明来源，好让下游知道它不是命中
                    "text": c.text,
                    "score": score,
                    "version": art.version_date.isoformat() if art.version_date else None,
                    "effective_date": (
                        art.effective_date.isoformat() if art.effective_date else None
                    ),
                    "citation": art.citation(),  # 《劳动合同法》第四十七条
                    "source": "neighbor" if score == 0.0 else "hit",
                }
            )
        return out

    # -- 便于调试 ----------------------------------------------------------- #

    def describe_stages(self) -> str:
        """把最近一次检索的分段耗时拼成一行，给日志和 Web 界面用。

        示例：`改写 3ms → bm25 12ms → dense 45ms → fuse/rrf 1ms → rerank 210ms → dedup 0ms`
        面试问"一次检索慢在哪"时，这一行就是答案。
        """
        if not self.last_stages:
            return "（尚未检索）"
        parts = [f"{s['stage']} {s['elapsed_ms']:.0f}ms" for s in self.last_stages]
        return " → ".join(parts)


__all__ = ["PipelineRetriever", "normalize_scores"]
