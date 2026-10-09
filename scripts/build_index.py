"""F3 建索引：把语料编码成向量并落盘。

## 为什么这个脚本值得单独存在（而不是在 pipeline 里顺手编码）

编码是**整条链上最贵的一步**：4 个切分臂 × 约 1700 条 ≈ 7000 次前向。
把它从评测里拆出来单独跑，换来三件事：

1. **消融不用重复编码。** 一次全矩阵里有 20 多条臂，其中大部分只在融合/精排/
   改写上不同，切分臂是一样的。缓存命中后这些臂几乎瞬间起跑。
2. **失败可归因。** "跑不起来"是编码挂了还是检索挂了？分开跑就一眼知道。
3. **`corpus_hash` 的校验点。** 索引文件名带语料指纹，语料一变就自然失配、
   触发重编码。**这是防止"读到过期向量却算出正常指标"的唯一机制**——
   那类错误不会报错，只会让历史数字变成谎言。

## 关于 Chroma

原计划（need.md F3）这里是「bge 编码 → 存 Chroma 持久化目录」。**改成了
直接存 `.npz`**，理由写在 `retrieval/dense.py` 的模块 docstring 里，
一句话：1730 条 × 768 维 = 5.3MB，暴力点积是微秒级且精确，
向量库在这个规模上换不来一分指标。`--backend faiss_ivf` 仍可建 FAISS 索引，
用来做「精确 vs 近似」的对照。

用法：
    python scripts/build_index.py                 # 全部 4 个切分臂，精确后端
    python scripts/build_index.py --chunker by_article
    python scripts/build_index.py --query "试用期最长多久"   # 建完顺手验一次
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from juris_agent.config import ensure_dirs, get_settings  # noqa: E402
from juris_agent.ingest.chunkers import CHUNKERS  # noqa: E402
from juris_agent.ingest.clean import load_articles  # noqa: E402

DEFAULT_CHUNKERS = ["by_article", "by_article_plus_context", "sentence_window", "fixed_512"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="建稠密索引（F3）")
    ap.add_argument("--chunker", action="append", choices=sorted(CHUNKERS), default=None,
                    help="可重复；默认全部 4 个切分臂")
    ap.add_argument("--model", default=None, help="覆盖 embedding 模型名")
    ap.add_argument("--query", default=None, help="建完后用这条 query 验一次 top-5")
    ap.add_argument("--force", action="store_true", help="忽略已有缓存，强制重编码")
    args = ap.parse_args(argv)

    s = get_settings()
    ensure_dirs(s)
    if not s.paths.laws.exists():
        print("[FATAL] 没有 data/laws.jsonl，先跑: python scripts/ingest_corpus.py", file=sys.stderr)
        return 2

    corpus_meta = (
        json.loads(s.paths.corpus_meta.read_text(encoding="utf-8")) if s.paths.corpus_meta.exists() else {}
    )
    corpus_hash = corpus_meta.get("corpus_hash")
    if not corpus_hash:
        print("[FATAL] 缺 data/index/corpus_meta.json（含 corpus_hash）——先跑 ingest_corpus.py",
              file=sys.stderr)
        return 2

    articles = load_articles(s.paths.laws)
    chunkers = args.chunker or DEFAULT_CHUNKERS

    from juris_agent.retrieval.dense import EmbeddingCache, Encoder

    model_name = args.model or s.retrieval.embedding_model
    print(f"语料 {len(articles)} 条　corpus_hash={corpus_hash}")
    print(f"编码模型 {model_name}")

    t0 = time.perf_counter()
    encoder = Encoder(model_name)
    print(f"模型加载完成（{time.perf_counter()-t0:.1f}s，device={encoder.device}，dim={encoder.dim}）")

    if args.force:
        cache = EmbeddingCache(s.paths.index / "emb")
        for c in chunkers:
            p = cache.path_for(model_name, c, corpus_hash)
            if p.exists():
                p.unlink()
                print(f"  已删除旧缓存 {p.name}")

    from juris_agent.retrieval.dense import DenseRetriever

    cache = EmbeddingCache(s.paths.index / "emb")
    summary: list[dict] = []
    for name in chunkers:
        chunks = CHUNKERS[name](articles)
        cached_before = cache.path_for(model_name, name, corpus_hash).exists()
        t1 = time.perf_counter()
        retriever = DenseRetriever(
            chunks, encoder, backend="exact", chunker_name=name,
            corpus_hash=corpus_hash, cache=cache,
        )
        dt = time.perf_counter() - t1
        n_hit = 0
        if args.query:
            hits = retriever.search(args.query, top_k=5)
            n_hit = len(hits)
        summary.append({"chunker": name, "n_chunks": len(chunks),
                        "from_cache": cached_before, "seconds": round(dt, 2)})
        print(f"  {name:<26} chunks={len(chunks):>5}  {'缓存命中' if cached_before else '新编码':<8} {dt:6.2f}s")

    if args.query:
        print(f"\n验证查询：{args.query!r}")
        # 用最后一个切分臂的结果展示（通常就是 by_article）
        retriever = DenseRetriever(
            CHUNKERS[chunkers[-1]](articles), encoder, chunker_name=chunkers[-1],
            corpus_hash=corpus_hash, cache=cache,
        )
        for c, sc in retriever.search(args.query, top_k=5):
            preview = c.text[:56].replace("\n", " ")
            print(f"  {sc:.4f}  {c.primary_key:<16} {preview}…")

    meta_path = s.paths.index / "emb_index_meta.json"
    meta_path.write_text(
        json.dumps(
            {"model": model_name, "dim": encoder.dim, "corpus_hash": corpus_hash,
             "arms": summary, "device": encoder.device},
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {meta_path}")
    print(f"总耗时 {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
