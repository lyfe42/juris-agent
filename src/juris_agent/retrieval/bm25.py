"""F6 BM25 关键词检索。

为什么法律检索不能只有向量：**法条里有大量"专属词"**——「第四十七条」「经济补偿金」
「无固定期限劳动合同」。稠密向量对这些词不敏感（`第47条` 和 `第48条` 在向量空间里
几乎重合），但用户和判决书恰恰满口都是它们。BM25 抓精确词，向量抓语义，两者互补——
这就是 F7 要融合的动机。

**分词是这个模块的全部难度。** jieba 默认词典会把「经济补偿金」切成
`经济 / 补偿 / 金`，于是用户查「经济补偿」时，「补偿金」这个变体和「补偿」对不上；
「无固定期限劳动合同」更惨，会被切散。所以要有法律领域词典
（`configs/legal_dict.txt`，jieba 用户词典格式）。

词典开与不开必须能切换——`use_dict` 参数就是消融的一臂：
    「有无领域词典的 Recall 差异」是 F6 验收要看的数字。
"""

from __future__ import annotations

import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from juris_agent.ingest.chunkers import Chunk

# 只丢真正的虚词。**「不得」「应当」「可以」这类一律保留**——它们在法律文本里
# 是实义词（「不得解除」和「可以解除」是两回事），丢了对检索是净损失。
_DEFAULT_STOPWORDS = frozenset(
    "的 了 和 与 及 或 在 为 是 对 等 所 其 之 者 有 被 由 于 而 并 也 中 上 下 内 外".split()
)

# 纯标点/空白的 token 丢掉；`\w` 在 Unicode 下含汉字，所以中文 token 不会被误伤
_PUNCT_ONLY = re.compile(r"^[\W_]+$")

def load_legal_dict(path: Path) -> bool:
    """把法律领域词典加进**全局** jieba。

    ⚠️ 只给"我只跑一臂"的场景用（比如 REPL 或单臂脚本）。**消融实验不要用它**：
    `jieba.load_userdict` 改的是进程级全局状态，一个进程里跑多臂时，先加载的那臂
    会把词典泄漏给后面所有臂——实测过一次，`dict=on` 和 `dict=off` 的数字一模一样，
    差点得出"领域词典没用"的错误结论。要跑对照就用 `Tokenizer` 实例，它持有独立的
    `jieba.Tokenizer`，词典互不影响。
    """
    if not path.exists():
        return False
    jieba.load_userdict(str(path))
    return True


class Tokenizer:
    """中文分词 + 过滤。

    每个实例持有**独立的 `jieba.Tokenizer`**（不是全局那个 `jieba` 模块），
    所以 `use_dict=True/False` 的两个实例可以同时存在、互不污染——这是消融矩阵
    能在同一个进程里跑完的前提。
    """

    def __init__(
        self,
        dict_path: Path | None = None,
        use_dict: bool = True,
        stopwords: frozenset[str] = _DEFAULT_STOPWORDS,
    ) -> None:
        self._jieba = jieba.Tokenizer()
        self.use_dict = bool(use_dict and dict_path is not None and dict_path.exists())
        if self.use_dict:
            assert dict_path is not None
            self._jieba.load_userdict(str(dict_path))
        self.stopwords = stopwords

    def __call__(self, text: str) -> list[str]:
        out: list[str] = []
        for tok in self._jieba.lcut(text, cut_all=False):
            tok = tok.strip().lower()
            if not tok or _PUNCT_ONLY.match(tok) or tok in self.stopwords:
                continue
            out.append(tok)
        return out


class BM25Retriever:
    """内存版 BM25。1700 条语料下建索引不到一秒，不值得持久化。"""

    def __init__(self, chunks: list[Chunk], tokenizer: Tokenizer, k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("chunks 为空——先跑 ingest，别空索引")
        self.chunks = chunks
        self.tokenizer = tokenizer
        self._corpus_tokens = [tokenizer(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens, k1=k1, b=b)

    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        """返回 `[(chunk, 分数)]`，按分数降序。分数为 0 的不返回（关键词一个都没中）。"""
        q = self.tokenizer(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[tuple[Chunk, float]] = []
        for i in ranked[: top_k * 3]:
            if scores[i] <= 0:
                break
            out.append((self.chunks[i], float(scores[i])))
            if len(out) >= top_k:
                break
        return out

    def rank(self, query: str, top_k: int = 20) -> list[str]:
        """只返回 `chunk_id` 序列，供 RRF 融合用（融合只需要排名，不需要分数）。"""
        return [c.chunk_id for c, _ in self.search(query, top_k)]


__all__ = ["BM25Retriever", "Tokenizer", "load_legal_dict"]
