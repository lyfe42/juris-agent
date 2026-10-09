"""F8 精排：cross-encoder 重排 + **超时降级**。

两件事，第二件比第一件重要。

## 一、为什么要精排

融合层（F7）是"粗排"：它只用名次，不知道 query 和某个具体条文的真实相关性。
cross-encoder 把 `(query, 条文)` **拼在一起过一遍模型**，能看见两者的交互，
精度显著高于双塔（bi-encoder）——稠密检索是双塔，query 和文档各自编码成向量，
只在最后算一次点积，**两者从未"见过面"**。

代价：cross-encoder 要 `O(候选数)` 次前向，无法预计算。所以它只能作用在
top-k 这一小撮上，且**它是这条链上唯一一个慢到可能拖垮响应的环节**。

## 二、超时降级：这才是这个模块存在的意义

一次 `rerank` 要跑 20 次前向。模型没预热、GPU 被别的进程占了、首次加载要读盘——
任何一件都能让它从 300ms 变成 3s。**检索链上其它每一段都是微秒到毫秒级，
只有这一段有真实的超时风险。**

于是：给它一个 `timeout_ms`，**超了就退回融合层的排序**接着走，
并在 `warnings` 里如实说明"本次结果是降级后的"，而不是让整个请求失败。

这比"精排提升 X 点"更值得讲——因为**精排提升的是平均质量，
超时降级保的是尾部可用性**。没有降级，一次抖动就能让用户看到 500。

### 一个必须说清的局限

超时后线程**不会被杀死**（Python 无法强杀线程），那次前向会在后台跑完。
所以"超时"省下的不是算力，是**用户的等待**。文档里也不该写成"及时取消"——
那是假的。真要能取消得用进程池，代价是每个请求多一次进程间拷贝，
对这个规模不划算。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Callable, Protocol

from pydantic import BaseModel, Field

from juris_agent.ingest.chunkers import Chunk


class RerankResult(BaseModel):
    """精排结果。**降级信息必须随结果一起回传**，不能只写日志。"""

    chunks: list[Chunk]
    scores: list[float] = Field(default_factory=list)
    degraded: bool = False
    warning: str | None = None
    elapsed_ms: float = 0.0
    n_candidates: int = 0


class Scorer(Protocol):
    """打分器接口。`predict` 收 `[(query, passage)]` 返回每对的分数。"""

    def predict(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]: ...


class CrossEncoderScorer:
    """`sentence-transformers` 的 CrossEncoder 包装。延迟加载模型。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)

    def predict(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        if not pairs:
            return []
        out = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return [float(x) for x in out]


class Reranker:
    """精排 + 超时降级。

    `scorer` 可注入——测试里塞一个 `sleep` 的假打分器就能断言降级路径，
    不需要真的把 GPU 跑超时（那样测不稳也没法在 CI 跑）。
    """

    def __init__(
        self,
        scorer: Scorer | None = None,
        *,
        timeout_ms: int = 800,
        batch_size: int = 32,
    ) -> None:
        self._scorer = scorer
        self.timeout_ms = timeout_ms
        self.batch_size = batch_size

    @property
    def scorer(self) -> Scorer:
        if self._scorer is None:
            self._scorer = CrossEncoderScorer()
        return self._scorer

    def rerank(
        self,
        query: str,
        candidates: list[Chunk],
        top_k: int | None = None,
    ) -> RerankResult:
        """对候选重排。**任何异常都降级，不向上抛。**

        注意这里连"打分器加载失败/打分抛错"也一并降级：精排是**质量增强**，
        不是**功能必需**——它挂了不该让整个检索挂掉。这与工具层"降级实现（若有）"
        是同一条原则，只是这里没有备用实现，备用实现**就是**融合排序本身。
        """
        t0 = time.perf_counter()
        if not candidates:
            return RerankResult(chunks=[], n_candidates=0, elapsed_ms=0.0)
        if not query.strip():
            return RerankResult(
                chunks=candidates[:top_k] if top_k else candidates,
                n_candidates=len(candidates),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        fallback = candidates[:top_k] if top_k else candidates
        pairs = [(query, c.text) for c in candidates]

        # 用**一个一次性线程池**跑，这样即使超时也不会把池子里的其它任务拖住；
        # 代价是每次 rerank 建一个线程（微秒级，可忽略）。
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rerank")
        try:
            future = pool.submit(self.scorer.predict, pairs, self.batch_size)
            try:
                scores = future.result(timeout=self.timeout_ms / 1000.0)
            except FutureTimeout:
                return RerankResult(
                    chunks=fallback,
                    scores=[],
                    degraded=True,
                    warning=(
                        f"精排超时（>{self.timeout_ms}ms），已降级为融合层排序。"
                        "注意：超时的那次前向仍在后台跑完，省下的是用户等待而非算力。"
                    ),
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                    n_candidates=len(candidates),
                )
            except Exception as e:  # 打分器本身出错 → 同样降级
                return RerankResult(
                    chunks=fallback,
                    scores=[],
                    degraded=True,
                    warning=f"精排失败（{type(e).__name__}: {e}），已降级为融合层排序。",
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                    n_candidates=len(candidates),
                )
        finally:
            # 不等待未完成的任务（wait=False），这正是"不阻塞响应"的落点
            pool.shutdown(wait=False)

        order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        ordered = [candidates[i] for i in order]
        ordered_scores = [scores[i] for i in order]
        if top_k:
            ordered = ordered[:top_k]
            ordered_scores = ordered_scores[:top_k]
        return RerankResult(
            chunks=ordered,
            scores=ordered_scores,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            n_candidates=len(candidates),
        )


def make_noop_scorer() -> Callable[[list[tuple[str, str]], int], list[float]]:
    """一个恒返回 0 的打分器——用于"精排关掉"的消融臂（保持链路形状不变）。"""

    def _predict(pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        return [0.0] * len(pairs)

    return _predict


__all__ = ["CrossEncoderScorer", "RerankResult", "Reranker", "Scorer", "make_noop_scorer"]
