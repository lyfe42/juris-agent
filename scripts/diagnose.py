"""检索差异诊断：**当一组臂的排名和直觉不符时，用来看是谁赢在哪道题上。**

## 为什么必须有这个脚本

消融表只告诉你"融合比稠密单独差 0.14"。它**不告诉你为什么**。
只有把逐条的排名摊开，才能看出是哪种失败在发生：

- 是 BM25 把语义正确的条挤出了 top-5？（噪声挤占）
- 还是 BM25 和稠密各自召回了不同的 gold，融合本该double win，却因为
  排序规则被压到了 5 名开外？（融合规则问题）
- 还是 gold 本身有漏标，导致"看起来赢了"的那条其实不算命中？（评测集问题）

三种原因的修法完全不同。**没有这个脚本，就只能对着一个更低的数字猜。**

## 输出

`reports/diagnosis.md`：
1. 逐臂总览
2. **独赢题**：只有某一臂命中、其他臂都miss的题（这些题最能说明该臂的独特价值）
3. **分歧题**：臂与臂名次差 ≥ 5 的题，附各自召回的 top-3 条文，供人眼判断谁对

用法：
    python scripts/diagnose.py --group dense
    python scripts/diagnose.py --arms "B/dense-only" "B/bm25+dense RRF" --topk 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from juris_agent.config import ensure_dirs, get_settings  # noqa: E402
from juris_agent.eval.retrieval_eval import GoldItem, load_gold  # noqa: E402
from juris_agent.retrieval.build import ArmDeps, build_pipeline  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation import GROUPS, GROUPS as _G  # noqa: E402  (复用臂的定义，避免两处漂移)


def first_hit_rank(ranked: list[str], gold: list[str]) -> int | None:
    """gold 中任一条在排名里的最好位次（1 起）。没命中返回 None。"""
    gs = set(gold)
    for i, key in enumerate(ranked, start=1):
        if key in gs:
            return i
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="检索差异诊断")
    ap.add_argument("--group", default="dense", choices=sorted(GROUPS),
                    help="用消融里预定义的一组臂")
    ap.add_argument("--arms", nargs="*", default=None, help="按名字挑（须在 --group 的臂里）")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-cases", type=int, default=12, help="每类最多列几道题")
    ap.add_argument("--source", default="all", choices=["all", "machine", "human"])
    ap.add_argument("--out", default="reports/diagnosis.md")
    args = ap.parse_args(argv)

    s = get_settings()
    ensure_dirs(s)
    if not s.paths.laws.exists():
        print("[FATAL] 先跑 scripts/ingest_corpus.py", file=sys.stderr)
        return 2

    gold: list[GoldItem] = []
    for p in (s.paths.eval / "retrieval_gold.jsonl", s.paths.eval / "retrieval_gold_machine.jsonl"):
        if p.exists():
            gold.extend(load_gold(p))
    if args.source != "all":
        gold = [g for g in gold if g.source == args.source]
    if not gold:
        print("[FATAL] 评测集为空", file=sys.stderr)
        return 2

    arms = GROUPS[args.group]()
    if args.arms:
        wanted = set(args.arms)
        arms = [a for a in arms if a.name in wanted]
        missing = wanted - {a.name for a in arms}
        if missing:
            print(f"[FATAL] 组 {args.group} 里没有这些臂: {sorted(missing)}", file=sys.stderr)
            print(f"        可选: {[a.name for a in GROUPS[args.group]()]}", file=sys.stderr)
            return 2

    corpus_meta_path = s.paths.corpus_meta
    import json

    corpus_hash = (
        json.loads(corpus_meta_path.read_text(encoding="utf-8")).get("corpus_hash", "unknown")
        if corpus_meta_path.exists() else "unknown"
    )
    deps = ArmDeps(corpus_hash=corpus_hash, skip_llm=True)
    artifacts: dict = {}

    # ranks[arm][gold_id] = 首位命中名次（None = 未命中）
    ranks: dict[str, dict[str, int | None]] = {}
    topos: dict[str, dict[str, list[str]]] = {}
    for cfg in arms:
        pipe = build_pipeline(cfg, deps, artifacts=artifacts)
        r_map: dict[str, int | None] = {}
        t_map: dict[str, list[str]] = {}
        for g in gold:
            ranked = pipe.search(g.query).expand_article_keys()
            r_map[g.id] = first_hit_rank(ranked, g.gold)
            t_map[g.id] = ranked[: args.topk]
        ranks[cfg.name] = r_map
        topos[cfg.name] = t_map
        n_hit = sum(1 for v in r_map.values() if v)
        print(f"  {cfg.name:<34} hit {n_hit}/{len(gold)}  (top{args.topk} 口径)")

    names = [c.name for c in arms]
    lines = [
        "# 检索差异诊断（自动生成）",
        "",
        f"- 评测集 n={len(gold)}　source={args.source}",
        f"- 命中口径：gold 中任一条出现在 top-{args.topk}",
        "",
        "## 逐臂总览",
        "",
        "| 臂 | 命中题数 | 平均首位名次（仅命中题） |",
        "|---|---:|---:|",
    ]
    for n in names:
        hits = [v for v in ranks[n].values() if v]
        avg = sum(hits) / len(hits) if hits else 0.0
        lines.append(f"| `{n}` | {len(hits)}/{len(gold)} | {avg:.2f} |")
    lines.append("")

    # --- 独赢题 --------------------------------------------------------- #
    lines.append("## 独赢题（只有这一臂命中）")
    lines.append("")
    lines.append("这类题最能说明某个臂的**不可替代性**。如果所有臂都没有独赢题，")
    lines.append("说明它们高度重叠——多跑一路只是在烧算力。")
    lines.append("")
    any_unique = False
    for n in names:
        uniq = [
            g for g in gold
            if ranks[n][g.id] is not None
            and all(ranks[m][g.id] is None for m in names if m != n)
        ]
        if not uniq:
            continue
        any_unique = True
        lines.append(f"### 只有 `{n}` 命中（{len(uniq)} 题）")
        lines.append("")
        for g in uniq[: args.max_cases]:
            lines.append(f"- **{g.id}**（{g.source}/{g.difficulty}）{g.query}")
            lines.append(f"  - gold: {g.gold}　该臂首位名次: {ranks[n][g.id]}")
            lines.append(f"  - 该臂 top-{args.topk}: {topos[n][g.id]}")
            others = {m: ranks[m][g.id] for m in names if m != n}
            lines.append(f"  - 其他臂名次: {others}")
        lines.append("")
    if not any_unique:
        lines.append("（没有任何臂有独赢题——它们高度重叠。）")
        lines.append("")

    # --- 分歧题 --------------------------------------------------------- #
    lines.append("## 分歧题（任两臂名次差 ≥ 5，或一命中一未命中）")
    lines.append("")
    lines.append("这里要**人眼看谁对**：是融合把正确答案挤下去了，还是另一臂召回了")
    lines.append("评测集漏标的平行条款（那属于 gold 的问题，不是系统的问题）。")
    lines.append("")
    shown = 0
    for g in gold:
        if shown >= args.max_cases:
            break
        vals = {n: ranks[n][g.id] for n in names}
        hit_vals = [v for v in vals.values() if v is not None]
        miss_vals = [v for v in vals.values() if v is None]
        spread = (max(hit_vals) - min(hit_vals)) if len(hit_vals) > 1 else 0
        if not (miss_vals and hit_vals) and spread < 5:
            continue
        shown += 1
        lines.append(f"### {g.id}　{g.query}")
        lines.append("")
        lines.append(f"- gold: {g.gold}　({g.source}/{g.qtype}/{g.difficulty})")
        lines.append("")
        lines.append("| 臂 | 首位命中名次 | top-3 条级排名 |")
        lines.append("|---|---:|---|")
        for n in names:
            r = ranks[n][g.id]
            lines.append(f"| `{n}` | {r if r else '**未命中**'} | {topos[n][g.id][:3]} |")
        lines.append("")
    if shown == 0:
        lines.append("（没有显著分歧的题。）")
        lines.append("")

    out = s.paths.root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
