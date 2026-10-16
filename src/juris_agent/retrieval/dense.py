"""F5 稠密检索：bge 向量 + 可插拔的近邻索引。

## 为什么核心用 numpy 而不是向量数据库

语料是 **1730 条 × 768 维**，整个矩阵 `1730 × 768 × 4B ≈ 5.3 MB`。
在这个规模上：

| 做法 | 一次查询 | 召回 | 额外依赖 |
|---|---|---|---|
| numpy 暴力点积 | **微秒级** | **精确（100%）** | 0 |
| FAISS IVF 近似 | 微秒级 | 有损 | faiss-cpu |
| Chroma / Milvus | 毫秒级（还有网络/客户端开销） | 精确 | 300MB+（chromadb 会拖进 kubernetes / onnxruntime / grpcio） |

向量库的唯一卖点是**在数据量大到暴力算不动时**用近似换速度。1730 条离那个边界
差着好几个数量级，所以引进向量库**一分指标都换不来，只换来依赖和冷启动开销**。
面试被问"为什么用向量库"而答不出"因为数据量大"，是硬伤；反过来，能说清
"我算过这个规模不需要"，才是加分项。

FAISS 仍然保留成可选后端——不是因为它更快，而是因为 `IVF` 的**近似损失是可以测的**：
F24 里"精确 vs IVF(nprobe=1)"这条轴能给出一个真实的 Recall 差值，
这比"Chroma vs FAISS"（两者都是精确检索，Recall 必然相同，差异只有 QPS）有意义得多。

## bge 的查询指令：一个不写就会白掉几个点的坑

`bge-*-zh-v1.5` 系列在**短查询检索长文档**的场景下要求给**查询**加一句指令前缀：

    为这个句子生成表示以用于检索相关文章：

**文档侧不加，查询侧加**——两侧不对称。这是模型训练时就这么定的，不是可选项。
漏掉它不会报错、不会崩，只会静默地少几个点的 Recall，非常难发现。
所以这里把它做成显式参数 `query_instruction`，并在消融里留了开关。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from juris_agent.ingest.chunkers import Chunk

# bge 官方在短查询-长文档检索时建议的查询指令（文档侧不加）
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


# --------------------------------------------------------------------------- #
# 编码
# --------------------------------------------------------------------------- #


class Encoder:
    """sentence-transformers 的薄封装。

    只暴露本项目需要的东西，好处是**编码参数只有一处**——`normalize_embeddings=True`
    这条如果散落在多个脚本里，早晚会有一个地方忘了写，然后余弦相似度变成内积，
    排序在长文本上悄悄出错。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-zh-v1.5",
        device: str | None = None,
        batch_size: int = 64,
        query_instruction: str = BGE_QUERY_INSTRUCTION,
    ) -> None:
        # 延迟 import：`sentence_transformers` 会拉起 torch，而工具层的测试
        # 只该依赖 pydantic。放在类里 import 保证"不碰检索就不会加载 torch"。
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        # ⚠️ 维度这个属性**改过名**：sentence-transformers 较新的版本里叫
        # `get_embedding_dimension`，旧的（以及本项目当前锁定的那个版本）
        # 叫 `get_sentence_embedding_dimension`。直接在代码里写新名字会在
        # 旧版本上 `AttributeError`，写旧名字会在新版本上打一条
        # `FutureWarning`（就是 CLI 里那行噪音，它会在真机上到处出现）。
        #
        # 所以两个都试：优先新名字，没有就退回旧名字。
        # **这比"直接钉死一个版本"好**——钉版本会让想升级的人先改这里，
        # 而他改的时候不会知道为什么这里有这个名字。
        get_dim = getattr(self.model, "get_embedding_dimension", None) or getattr(
            self.model, "get_sentence_embedding_dimension"
        )
        self.dim = int(get_dim())

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """文档侧：**不加**指令前缀。返回 L2 归一化后的 `(n, dim)` float32。"""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """查询侧：加指令前缀。"""
        prefixed = [f"{self.query_instruction}{t}" for t in texts]
        return self.encode_documents(prefixed)


# --------------------------------------------------------------------------- #
# 近邻索引后端
# --------------------------------------------------------------------------- #


class VectorBackend(Protocol):
    name: str

    def add(self, vectors: np.ndarray) -> None: ...
    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]: ...


class NumpyIndex:
    """精确暴力检索。向量已归一化，所以点积 == 余弦相似度。

    为什么不排序整个数组：`argpartition` 取 top-k 是 O(n)，全排序是 O(n log n)。
    1730 条时两者都是微秒级，但写成 `argpartition` 顺便说明了"我知道这里可以不做全排序"。
    """

    name = "exact"

    def __init__(self) -> None:
        self._m: np.ndarray | None = None

    def add(self, vectors: np.ndarray) -> None:
        self._m = np.ascontiguousarray(vectors.astype(np.float32))

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        assert self._m is not None, "先 add()"
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        scores = self._m @ q
        k = min(top_k, len(scores))
        if k <= 0:
            return []
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]


class FaissIndex:
    """FAISS `IVF` 近似索引——**故意做成有损的**，用来量化近似损失。

    `nprobe=1` 时只扫 `nlist` 个簇里的 1 个，`n=1730` 下漏检是必然的。
    这不是配置错误，是要观察的对象（见 F24 的"精确 vs 近似"一轴）。
    想让它接近精确就调大 `nprobe`。
    """

    name = "faiss_ivf"

    def __init__(self, nlist: int = 16, nprobe: int = 1) -> None:
        import faiss  # 延迟到用时才 import

        self._faiss = faiss
        self.nlist = nlist
        self.nprobe = nprobe
        self._index = None

    def add(self, vectors: np.ndarray) -> None:
        faiss = self._faiss
        v = np.ascontiguousarray(vectors.astype(np.float32))
        dim = v.shape[1]
        # 内积索引（向量已归一化 → 等价于余弦）
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
        # n 太小时 train 会因簇内样本太少而告警；至少要有 nlist 个训练点
        index.train(v)
        index.add(v)
        index.nprobe = self.nprobe
        self._index = index

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        assert self._index is not None, "先 add()"
        q = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
        k = min(top_k, self._index.ntotal)
        if k <= 0:
            return []
        scores, idx = self._index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]


BACKENDS = {"exact": NumpyIndex, "faiss_ivf": FaissIndex}
"""消融里可选的索引后端名。默认 `exact`。"""


# --------------------------------------------------------------------------- #
# 磁盘缓存
# --------------------------------------------------------------------------- #


class EmbeddingCache:
    """把编码结果按 `(模型, 切分臂, 语料指纹)` 落盘。

    **不做这一步，消融矩阵就没法跑。** 一次全矩阵要编码 4 个切分臂 × 1700 条 =
    近 7000 次前向；就算 GPU 也要几十秒，重跑一次就再花一次。落盘之后换后端、
    换 k、换融合权重都**不需要重新编码**，消融才能真的"一条命令跑完"。

    文件名里带 `corpus_hash` 是关键：语料改了但名字没变，会读到**过期向量**并
    算出一组看起来正常、实则错误的历史指标——这类错误最难查。
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _stem(self, model: str, chunker: str, corpus_hash: str) -> str:
        safe = model.replace("/", "__")
        return f"{safe}__{chunker}__{corpus_hash}"

    def path_for(self, model: str, chunker: str, corpus_hash: str) -> Path:
        return self.root / f"{self._stem(model, chunker, corpus_hash)}.npz"

    def load(self, model: str, chunker: str, corpus_hash: str) -> tuple[np.ndarray, list[str]] | None:
        p = self.path_for(model, chunker, corpus_hash)
        if not p.exists():
            return None
        with np.load(p, allow_pickle=False) as z:
            return z["vectors"], [str(x) for x in z["chunk_ids"]]

    def save(
        self, model: str, chunker: str, corpus_hash: str, vectors: np.ndarray, chunk_ids: list[str]
    ) -> None:
        # ⚠️ 这里**不能**用 `dtype=object`。object 数组存进 npz 会走 pickle，
        # 而读取端用 `allow_pickle=False`（这是对的，npz 可能来自不可信来源），
        # 于是读的时候直接 `ValueError: Object arrays cannot be loaded...`。
        # 用 numpy 的原生 Unicode 数组（`<U…`）就没有这个问题——
        # 它是定长字符串，不涉及 pickle。
        np.savez_compressed(
            self.path_for(model, chunker, corpus_hash),
            vectors=vectors,
            chunk_ids=np.asarray(chunk_ids),
        )


# --------------------------------------------------------------------------- #
# 检索器
# --------------------------------------------------------------------------- #


class DenseRetriever:
    """稠密检索。构造时吃 chunk 列表，内部按需编码或读缓存。"""

    def __init__(
        self,
        chunks: list[Chunk],
        encoder: Encoder,
        *,
        backend: str = "exact",
        chunker_name: str = "",
        corpus_hash: str = "",
        cache: EmbeddingCache | None = None,
        backend_kwargs: dict | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("chunks 为空——先跑 ingest，别空索引")
        self.chunks = chunks
        self.encoder = encoder
        self.corpus_hash = corpus_hash
        self.cache = cache

        vectors = None
        if cache is not None and corpus_hash:
            cached = cache.load(encoder.model_name, chunker_name, corpus_hash)
            if cached is not None:
                vecs, ids = cached
                # 缓存与当前 chunk 列表必须逐条对齐。数量对了还不够——顺序错位
                # 会让"第 47 条"的向量挂到别的条上，检索结果看起来正常但全是错的。
                if len(ids) == len(chunks) and ids == [c.chunk_id for c in chunks]:
                    vectors = vecs

        if vectors is None:
            vectors = encoder.encode_documents([c.text for c in chunks])
            if cache is not None and corpus_hash:
                cache.save(
                    encoder.model_name, chunker_name, corpus_hash, vectors,
                    [c.chunk_id for c in chunks],
                )

        if backend not in BACKENDS:
            raise ValueError(f"未知后端 {backend!r}，可选：{sorted(BACKENDS)}")
        self.backend_name = backend
        self.index: VectorBackend = BACKENDS[backend](**(backend_kwargs or {}))
        self.index.add(vectors)

    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        if not query.strip():
            return []
        qv = self.encoder.encode_queries([query])[0]
        return [(self.chunks[i], s) for i, s in self.index.search(qv, top_k)]

    def rank(self, query: str, top_k: int = 20) -> list[str]:
        return [c.chunk_id for c, _ in self.search(query, top_k)]
