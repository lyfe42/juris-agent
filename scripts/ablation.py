"""F24 消融矩阵 runner：一条命令跑完所有对照臂，产出 `reports/ablation.md`。

## 为什么要有"分组"

全矩阵一次跑完要编码多个切分臂、加载精排模型、还要调 LLM 改写。
开发时想快速看一眼 BM25 那几行也等好几分钟，人就会开始"懒得跑"，
然后消融表慢慢过期、和代码对不上。

所以按**依赖**分组：`chunking` 组只跑 BM25（毫秒级、不用 GPU、
不用 API key），`neural` 组才碰稠密/精排，`llm` 组才花钱。
`--group chunking` 是开发循环里最常用的那条命令。

    python scripts/ablation.py --group chunking      # 秒级，无 GPU 无 key
    python scripts/ablation.py --group neural        # 要 ml extra
    python scripts/ablation.py --group all           # 全跑

## 每张表的元信息

每行都带 `n / gold_hash / corpus_hash / 组件配置`。这不是装饰：
面试官问"这个 0.571 是什么条件下测的"，要能立刻答上来；
更重要的是**自己回头看得懂**——三个月后看到"hit@5 = 0.62"而不知道
当时测评集是哪一份，这个数字就作废了。

## 延迟怎么报

`p50/p95` 是**真实测量**的端到端检索耗时（`time.perf_counter` 包住整条管道），
不是模拟量。这与工具层故障注入实验形成对比——那边报的是模拟延迟，
两个表里都会写清楚，**不能让读者分不清哪个是实测**。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from juris_agent.config import ensure_dirs, get_settings  # noqa: E402
from juris_agent.eval.retrieval_eval import (  # noqa: E402
    EvalReport,
    GoldItem,
    evaluate,
    gold_hash,
    load_gold,
)
from juris_agent.retrieval.build import ArmDeps, build_pipeline  # noqa: E402
from juris_agent.retrieval.pipeline import ArmConfig  # noqa: E402


# --------------------------------------------------------------------------- #
# 臂的定义
# --------------------------------------------------------------------------- #

CHUNKERS = ["by_article", "by_article_plus_context", "sentence_window", "fixed_512"]


def arms_chunking() -> list[ArmConfig]:
    """A 组：切分 × 词典。**只用 BM25**，所以无 GPU、无 key、秒级。

    这组回答两个问题：
    1. 切分策略值多少分？（4 个切分臂对比）
    2. 法律领域词典有没有用？（dict on/off 配对）
    """
    arms = []
    for c in CHUNKERS:
        for d in (True, False):
            arms.append(
                ArmConfig(
                    name=f"A/{c}/dict={'on' if d else 'off'}",
                    chunker=c,
                    use_bm25=True,
                    use_dict=d,
                )
            )
    return arms


def arms_dense() -> list[ArmConfig]:
    """B 组：稠密单独、两路融合、融合方式对比。需要 ml extra。"""
    return [
        ArmConfig(name="B/dense-only", chunker="by_article", use_bm25=False, use_dense=True),
        ArmConfig(name="B/bm25-only", chunker="by_article", use_bm25=True, use_dense=False),
        ArmConfig(name="B/bm25+dense RRF", chunker="by_article", use_bm25=True, use_dense=True, fusion="rrf"),
        ArmConfig(name="B/bm25+dense 加权求和", chunker="by_article", use_bm25=True, use_dense=True, fusion="weighted_sum"),
    ]


def arms_neural() -> list[ArmConfig]:
    """C 组：在最佳召回组合上逐个加装——精排、邻条扩展。

    实测（269 题）：精排是**单项收益最大的部件**（hit@5 0.866 → 0.944，
    MRR@10 0.689 → 0.819），代价是 p50 从 12ms 涨到 222ms。

    但**邻条扩展这一臂在指标上恒等于基线**（`+精排+邻条` 也恒等于 `+精排`）。
    这不是 bug，是必然：扩展出来的条一律**追加在结果尾部**（设计上不让
    "顺带捞的"挤掉"真正命中的"），于是条级排名的前 k 个位置完全不变。
    逐题验证过：269/269 题的前 10 个条级排名与基线逐位相同。

    所以这一臂**要留着**——留着才能说明"邻条扩展在本项目的指标口径下
    无法被证实"，而不是让它悄悄消失、让读者以为它有用。
    它声称的价值在**生成层**（给模型更多上下文），那需要另一个实验来测。
    """
    base = dict(chunker="by_article", use_bm25=True, use_dense=True, fusion="rrf")
    return [
        ArmConfig(name="C/融合（基线）", **base),
        ArmConfig(name="C/+精排", use_rerank=True, **base),
        ArmConfig(name="C/+邻条扩展±1", neighbor_window=1, **base),
        ArmConfig(name="C/+精排+邻条", use_rerank=True, neighbor_window=1, **base),
    ]


def arms_rewrite() -> list[ArmConfig]:
    """D 组：查询改写。同义词表（免费）vs LLM 改写（花钱）——两条独立对照轴。"""
    base = dict(chunker="by_article", use_bm25=True, use_dense=True, fusion="rrf")
    return [
        ArmConfig(name="D/无改写（基线）", rewrite="none", **base),
        ArmConfig(name="D/+同义词表", rewrite="synonym", **base),
        ArmConfig(name="D/+LLM 改写", rewrite="llm", **base),
    ]


def arms_weighting() -> list[ArmConfig]:
    """G 组：融合权重。

    **这一组是被实测逼出来的。** B 组跑出了反直觉的结果：
    稠密单独 hit@5 = 0.907，BM25 单独 0.606，而两者等权 RRF 融合只有 0.866
    ——**融合比最强的单路还差**。（早先在 21 题人工集上是 0.857 / 0.571 / 0.714，
    方向一致、幅度更大；小样本会把差距夸大。）

    机制假设：等权意味着"我认定两路同样可信"，但它们的质量差着 30 个点。
    BM25 那种"字面撞上一个词"的命中（例如 query 里有个「公司」，
    民法典里到处都有「公司」）会被当成与稠密的语义命中同等的证据，
    于是把噪声推进 top-5。

    实测结果比假设更有意思——**两个指标说的是两件事**：

        BM25 权重   0     0.25   0.50   0.75   1.00
        hit@5     0.907  0.903  0.903  0.903  0.866
        MRR@10    0.768  0.730  0.717  0.713  0.689

    `hit@5` 在 0.25～0.75 持平，看着"加一点 BM25 无害"；但 `MRR@10` **单调下降**。
    只看 hit@5 会得出"加点没关系"，看 MRR 才知道它一直在把正确答案往下挤。
    **所以这个扫描的价值不在于找到最优权重，而在于它同时证明了
    两个指标都不能省。**

    结论：**在这个语料上 BM25 不该进候选池**（它的 info 已经被稠密覆盖，
    而多出来的部分是噪声）。这比"我用了混合检索"有信息量得多。

    `w=0.0` 那行是**自检**：它必须与 dense-only **逐题完全一致**。
    这条自检曾经失败过——`rrf_fuse` 会把零权重那一路的文档以 0 分插进结果，
    于是"关掉一路"变成了"降权但不关"。修法见 `retrieval/fusion.py` 里的注释。
    """
    base = dict(chunker="by_article", use_bm25=True, use_dense=True, fusion="rrf")
    return [
        ArmConfig(name=f"G/权重 bm25={w:g}", sparse_weight=w, **base)
        for w in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]


def arms_k_scan() -> list[ArmConfig]:
    """E 组：RRF 的 k 扫描。回答"k=60 是怎么定的"而不是"我看别人写 60"。"""
    return [
        ArmConfig(
            name=f"E/RRF k={k}", chunker="by_article", use_bm25=True, use_dense=True,
            fusion="rrf", rrf_k=k,
        )
        for k in (1, 10, 30, 60, 200)
    ]


def arms_backend() -> list[ArmConfig]:
    """F 组：精确 vs 近似。**量化近似检索的损失**，这是"要不要向量库"的实测依据。"""
    return [
        ArmConfig(name="F/dense 精确", chunker="by_article", use_bm25=False, use_dense=True, dense_backend="exact"),
        ArmConfig(name="F/dense IVF(nprobe=1)", chunker="by_article", use_bm25=False, use_dense=True, dense_backend="faiss_ivf"),
    ]


GROUPS = {
    "chunking": arms_chunking,
    "dense": arms_dense,
    "weighting": arms_weighting,
    "neural": arms_neural,
    "rewrite": arms_rewrite,
    "k_scan": arms_k_scan,
    "backend": arms_backend,
}

# 分组名 → 该组臂名前缀。runner 靠它把 ArmsRun 分回自己那一组。
GROUP_PREFIX = {
    "chunking": "A/",
    "dense": "B/",
    "weighting": "G/",
    "neural": "C/",
    "rewrite": "D/",
    "k_scan": "E/",
    "backend": "F/",
}


# --------------------------------------------------------------------------- #
# 跑
# --------------------------------------------------------------------------- #

METRIC_COLS = [
    "recall_hit@1", "recall_hit@5", "recall_hit@10",
    "recall_art@5", "recall_art@10", "mrr@10", "ndcg@5",
]
COL_LABEL = {
    "recall_hit@1": "hit@1", "recall_hit@5": "hit@5", "recall_hit@10": "hit@10",
    "recall_art@5": "art@5", "recall_art@10": "art@10",
    "mrr@10": "MRR@10", "ndcg@5": "nDCG@5",
}


class ArmRun:
    def __init__(self, name: str, report: EvalReport, latencies: list[float], describe: str,
                 extra: dict | None = None, stages: dict[str, list[float]] | None = None,
                 config: ArmConfig | None = None) -> None:
        self.name = name
        self.report = report
        self.latencies = sorted(latencies)
        self.describe = describe
        self.extra = extra or {}
        self.config = config
        # stage 名 → 该臂在每个查询上这一段的耗时列表。用来出**分段耗时表**：
        # 面试问"一次检索耗时花在哪"要靠它答，只报总 p50 是答不上来的。
        self.stages = stages or {}

    @property
    def p50(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0

    @property
    def p95(self) -> float:
        if not self.latencies:
            return 0.0
        idx = min(len(self.latencies) - 1, int(round(0.95 * (len(self.latencies) - 1))))
        return self.latencies[idx]

    @property
    def calls_llm(self) -> bool:
        """这条臂的检索路径上会不会调 LLM（目前只有改写）。"""
        return bool(self.config and self.config.rewrite == "llm")

    def row(self, latency_is_real: bool = True) -> str:
        v = self.report.overall.values
        cells = " | ".join(f"{v.get(c, 0):.3f}" for c in METRIC_COLS)
        # 走 LLM 的臂在缓存命中时延时会塌掉两个数量级（实测 ≈6.7 秒 → 42ms），
        # 不是真实推理耗时。标出来，别让读者拿缓存命中延迟当性能数据。
        mark = "" if (latency_is_real or not self.calls_llm) else " ⚠️"
        return (f"| `{self.name}` | {self.report.overall.n} | {cells} "
                f"| {self.p50:.0f}{mark} | {self.p95:.0f}{mark} |")


def run_arm(
    cfg: ArmConfig,
    deps: ArmDeps,
    gold: list[GoldItem],
    artifacts: dict,
    *,
    warmup: bool = True,
) -> ArmRun:
    pipe = build_pipeline(cfg, deps, artifacts=artifacts)

    rankings: dict[str, list[str]] = {}
    latencies: list[float] = []
    warnings_seen: set[str] = set()
    stage_ms: dict[str, list[float]] = {}
    for i, g in enumerate(gold):
        if warmup and i == 0:
            pipe.search(g.query)  # 首次调用含模型预热/缓存填充，不计入延迟
        t0 = time.perf_counter()
        res = pipe.search(g.query)
        latencies.append((time.perf_counter() - t0) * 1000)
        rankings[g.id] = res.expand_article_keys()
        warnings_seen.update(res.warnings)
        for st in res.stages:
            stage_ms.setdefault(st.stage, []).append(st.elapsed_ms)

    meta = {
        "arm": cfg.name,
        "chunker": cfg.chunker,
        "legal_dict": str(cfg.use_dict),
        "n_gold": len(gold),
        "gold_hash": gold_hash(gold),
        "corpus_hash": deps.corpus_hash,
    }
    rep = evaluate(gold, rankings, meta=meta)
    extra: dict = {}
    if warnings_seen:
        extra["warnings"] = sorted(warnings_seen)
    # LLM 改写失败会**静默降级**成"这一题没改写"（见 rewrite.LLMRewriter 的 docstring）。
    # 不报出来的话，这一臂的数字里混着一成没改写的题，而看表的人无从知道。
    rewriter = getattr(pipe, "rewriter", None)
    if getattr(rewriter, "failures", 0):
        extra["rewrite_failures"] = {
            "count": rewriter.failures,
            "calls": rewriter.calls,
            "rate": round(rewriter.failure_rate, 4),
            "reasons": dict(rewriter.failure_reasons),
        }
    return ArmRun(cfg.name, rep, latencies, cfg.describe(), extra, stages=stage_ms, config=cfg)


def latency_table(runs: list[ArmRun]) -> list[str]:
    """分段耗时表：回答"一次检索耗时花在哪"。

    每段取**中位数**（不是均值）：某段在某题上根本没跑（例如精排降级、
    邻条关闭），均值会被这些"缺席"稀释得看不出来，中位数不会。

    表的列是**并集**——不同臂开的段不一样，只列第一个臂有的列会把后面的臂截掉。
    缺的那格写 `—` 而不是 0：`0` 会被读成"这段不耗时"，而事实是"这段没开"。
    """
    if not runs:
        return []
    cols: list[str] = []
    for r in runs:
        for s in r.stages:
            if s not in cols:
                cols.append(s)
    head = "| 臂 | " + " | ".join(cols) + " | 合计 |"
    lines = ["### 分段耗时（中位数 ms）", "", head, "|---" * (len(cols) + 2) + "|"]
    for r in runs:
        cells = []
        for c in cols:
            v = r.stages.get(c)
            cells.append(f"{statistics.median(v):.1f}" if v else "—")
        lines.append(f"| `{r.name}` | " + " | ".join(cells) + f" | {r.p50:.0f} |")
    lines.append("")
    lines.append("> `—` = 这条臂没开这一段（不是耗时为零）。`合计` = 端到端 p50，")
    lines.append("> 与各段中位数之和不必相等——总和的分布中位数不等于各段中位数之和。")
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="检索消融矩阵（F24）")
    ap.add_argument("--group", action="append", choices=sorted(GROUPS) + ["all"], default=None,
                    help="可重复；默认 chunking")
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"],
                    help="按 dev/test 切片（调参看 dev，报数认 test）")
    ap.add_argument("--source", default="all", choices=["all", "machine", "human"])
    ap.add_argument("--gold", default=None, help="额外/替换的评测集路径（默认自动合并 human + machine）")
    ap.add_argument("--out", default="reports/ablation.md")
    ap.add_argument("--json-out", default="reports/ablation.json")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要跑的臂，不实际执行")
    args = ap.parse_args(argv)

    s = get_settings()
    ensure_dirs(s)

    if not s.paths.laws.exists():
        print("[FATAL] 没有 data/laws.jsonl，先跑: python scripts/ingest_corpus.py", file=sys.stderr)
        return 2

    # --- 载入评测集（human + machine 合并） ----------------------------- #
    gold_paths: list[Path] = []
    if args.gold:
        gold_paths.append(Path(args.gold))
    else:
        human = s.paths.eval / "retrieval_gold.jsonl"
        machine = s.paths.eval / "retrieval_gold_machine.jsonl"
        for p in (human, machine):
            if p.exists():
                gold_paths.append(p)
    if not gold_paths:
        print("[FATAL] 找不到任何评测集", file=sys.stderr)
        return 2

    gold: list[GoldItem] = []
    for p in gold_paths:
        gold.extend(load_gold(p))
    if args.split != "all":
        gold = [g for g in gold if g.split == args.split]
    if args.source != "all":
        gold = [g for g in gold if g.source == args.source]
    if not gold:
        print("[FATAL] 过滤后评测集为空", file=sys.stderr)
        return 2

    corpus_meta = (
        json.loads(s.paths.corpus_meta.read_text(encoding="utf-8")) if s.paths.corpus_meta.exists() else {}
    )
    corpus_hash = corpus_meta.get("corpus_hash", "unknown")

    groups = args.group or ["chunking"]
    if "all" in groups:
        groups = sorted(GROUPS)
    # 去重保序
    seen: set[str] = set()
    groups = [g for g in groups if not (g in seen or seen.add(g))]

    print(f"corpus_hash={corpus_hash}  gold_hash={gold_hash(gold)}  n_gold={len(gold)} "
          f"(source={args.source}, split={args.split})")
    print(f"分组: {groups}\n")

    deps = ArmDeps(corpus_hash=corpus_hash, skip_llm=("rewrite" not in groups))
    artifacts: dict = {}
    runs: list[ArmRun] = []

    for gname in groups:
        arms = GROUPS[gname]()
        for cfg in arms:
            if args.dry_run:
                print(f"  [{gname}] {cfg.name:<34} {cfg.describe()}")
                continue
            t0 = time.perf_counter()
            try:
                run = run_arm(cfg, deps, gold, artifacts)
            except Exception as e:
                print(f"  [{gname}] {cfg.name:<34} FAILED: {type(e).__name__}: {e}")
                continue
            runs.append(run)
            v = run.report.overall.values
            print(f"  [{gname}] {cfg.name:<34} hit@5={v.get('recall_hit@5', 0):.3f} "
                  f"art@5={v.get('recall_art@5', 0):.3f} mrr={v.get('mrr@10', 0):.3f} "
                  f"p50={run.p50:.0f}ms  ({time.perf_counter()-t0:.1f}s)")

    if args.dry_run or not runs:
        return 0

    # --- 出表 ------------------------------------------------------------ #
    header = "| 臂 | n | " + " | ".join(COL_LABEL[c] for c in METRIC_COLS) + " | p50 ms | p95 ms |"
    llm_mode = s.llm_mode
    # `record` / `replay` 都会**先读缓存**，所以第二次跑起任何 LLM 调用的耗时
    # 都变成"读一个 JSON 文件"。实测同一个臂：缓存未命中时 p50 ≈ 6.7~6.8 秒
    # （三次采样 6684 / 6729 / 6792ms，p95 达 12.1 秒），缓存命中后 p50 ≈ 42ms
    # ——**差约 160 倍，而指标一模一样**。
    # 这时候表里的延迟是"缓存命中延迟"，不是推理延迟，必须标出来。
    latency_is_real = llm_mode == "off"
    lines = [
        "# 检索消融矩阵（自动生成，勿手改）",
        "",
        f"生成命令：`python scripts/ablation.py --group {' --group '.join(groups)} "
        f"--split {args.split} --source {args.source}`",
        "",
        f"- `corpus_hash`: `{corpus_hash}`",
        f"- `gold_hash`: `{gold_hash(gold)}`　n = {len(gold)}　source = `{args.source}`　split = `{args.split}`",
        "- 每路候选 top_k = 20，融合后 fused_k = 20",
        f"- LLM 模式：**`{llm_mode}`**"
        + ("（`off` = 每次都真调模型，**延迟是真实的**）" if latency_is_real
           else "（`record`/`replay` 先读缓存，⚠️ 走 LLM 的臂其延迟是**缓存命中延迟**）"),
        ("- **延迟是实测**（`perf_counter` 包住整条管道，含首题预热剔除）。"
         if latency_is_real else
         "- **不带 ⚠️ 的臂，延迟是实测**（`perf_counter` 包住整条管道，"
         "含首题预热剔除）；**带 ⚠️ 的臂延迟无效**，见下。"),
        "",
        "> **两种 Recall 必须一起看。** `hit@k` = top-k 里有任意一条 gold 即命中；",
        "> `art@k` = 覆盖了 gold 的比例。多条款组合题上，只报前者会虚高数倍。",
        "",
        "> ⚠️ **缓存会摧毁延迟测量。** 在 `record`/`replay` 模式下，带 ⚠️ 的臂",
        "> 跑的是缓存命中。实测同一个 LLM 改写臂：未缓存 p50 ≈ **6.8 秒**，",
        "> 缓存命中 p50 ≈ **42 毫秒**——**差约 160 倍，而指标一模一样**。",
        "> 要报 LLM 臂的真实延迟，必须用 `JURIS_AGENT_LLM_MODE=off` 单独跑一次。",
        "> 指标不受影响（缓存是内容寻址的，命中即同一份输出）。",
        ">",
        "> ⚠️ **另外，走 LLM 的臂其指标本身只是「单次采样」，不等于稳定值**：",
        "> 这个托管模型在 `temperature=0` 下也不可复现（实测同一道题问两次，20/20 不同），",
        "> 缓存只是把**某一次**采样钉住了。要报可信区间，必须多次采样报均值±std。",
        "",
    ]
    for gname in groups:
        grp_runs = [r for r in runs if r.name.startswith(GROUP_PREFIX[gname])]
        if not grp_runs:
            continue
        lines.append(f"## {gname}")
        lines.append("")
        lines.append(header)
        lines.append("|---" * (len(METRIC_COLS) + 3) + "|")
        lines.extend(r.row(latency_is_real=latency_is_real) for r in grp_runs)
        lines.append("")
        for r in grp_runs:
            if r.extra.get("warnings"):
                lines.append(f"- `{r.name}` 触发降级：{'；'.join(r.extra['warnings'])}")
            rf = r.extra.get("rewrite_failures")
            if rf:
                reasons = " / ".join(f"{k}: {v}" for k, v in rf["reasons"].items())
                lines.append(
                    f"- ⚠️ `{r.name}` 有 **{rf['count']}/{rf['calls']} 次改写失败"
                    f"（{rf['rate']:.1%}）**，这些题静默退化成「不改写」，"
                    f"**这一行的指标已被稀释。** 原因分布：{reasons}。"
                    f"失败调用**不入缓存**（存了会把一次网络抖动永久钉进 fixture），"
                    f"所以这个次数每次重跑都会变——它本身不是定值。"
                )
        if any(r.extra.get("warnings") or r.extra.get("rewrite_failures") for r in grp_runs):
            lines.append("")

    lines.extend(latency_table(runs))

    out = s.paths.root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    json_out = s.paths.root / args.json_out
    json_out.write_text(
        json.dumps(
            {
                "corpus_hash": corpus_hash,
                "gold_hash": gold_hash(gold),
                "n_gold": len(gold),
                "source": args.source,
                "split": args.split,
                "llm_mode": llm_mode,
                "latency_is_real": latency_is_real,
                "arms": [
                    {
                        "name": r.name,
                        "describe": r.describe,
                        "n": r.report.overall.n,
                        "metrics": r.report.overall.values,
                        "p50_ms": round(r.p50, 2),
                        "p95_ms": round(r.p95, 2),
                        "by_source": {k: v.values for k, v in r.report.by_source.items()},
                        "by_qtype": {k: v.values for k, v in r.report.by_qtype.items()},
                        "by_split": {k: v.values for k, v in r.report.by_split.items()},
                        "stage_p50_ms": {
                            k: round(statistics.median(v), 3) for k, v in r.stages.items()
                        },
                        "warnings": r.extra.get("warnings", []),
                        "rewrite_failures": r.extra.get("rewrite_failures"),
                    }
                    for r in runs
                ],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}\nwrote {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
