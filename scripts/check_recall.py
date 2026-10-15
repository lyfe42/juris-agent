"""F34 Recall 回归门禁：把"检索变差了"这件事做**红**，而不是靠人盯表。

## 这个脚本在解决什么问题

消融表 `reports/ablation.md` 是**快照**，不是**门禁**。它回答"当时哪个臂赢了"，
但没人会在每次改完之后重跑它、再逐行对照上一版。于是真实情况是：

    改了一处检索，跑一遍单测（全绿，因为单测只断言"函数返回了东西"），
    提交。三个星期后回头看消融表，发现 hit@5 掉了 4 个点，
    但已经不知道是哪一次改的掉的了。

**单测测的是"跑不跑得通"，门禁测的是"跑得对不对"。** 两件事，缺一个都会漏。

## 用法

    python scripts/check_recall.py                    # 跑所有能在本机跑的门禁
    python scripts/check_recall.py --only bm25-test   # 只跑一条
    python scripts/check_recall.py --update-baseline  # 重新标定基线（见下）

退出码：`0` 全过 / `1` 有指标掉了 / `2` 环境不齐（缺语料、缺 ml extra）/ `3` 基线过期。

## ⚠️ 三个刻意的设计决定，每个都对应一种"门禁会骗人"的失败方式

### 1. 语料或评测集一变，**拒绝比较**，而不是照常比

基线数字是"在某一份语料 + 某一份评测集上"测出来的。这两个东西任何一个变了，
数字变化**都是合法的**——你新加了两部法规，Recall 本来就该变。

这时候老实做一个差值比较，会有两种结果，**两种都比不比较更坏**：
- 数字涨了 → 测试绿，你以为没事，其实是语料变了；
- 数字跌了 → 测试红，你去查一个根本不存在的 bug，查一下午。

所以基线里一并存 `corpus_hash` 和 `gold_hash`。对不上就直接拒绝比较并退出码 3，
提示"先确认这版改动是不是真的改变了语料/评测集；确认无误再 `--update-baseline` 重新标定"。
**把"该不该重新标定"这个判断明确交回给人**，而不是让脚本替你猜。

### 2. 门禁看 `recall_art@5`，**不是** `recall_hit@5`

这是本项目 F23 的核心结论，门禁必须跟着它走。两道口径的差别：
- `recall_hit@5`：top-5 里**只要蒙中一条** gold 就算满分。多条款组合题召回 1/3 也是满分。
- `recall_art@5`：按**条**算覆盖率。召回 1/3 就是 0.33。

最弱的臂上差得最明显（`fixed_512` 的 0.394 vs 0.370）。更麻烦的是——
**答应用户的时候，漏掉的那两条才是答错的原因**，而 hit 口径对这件事完全看不见。
所以门禁卡 `recall_art@5`；`hit@5` 只作为参考列打出来，不进判定。

### 3. CI 默认只跑**不装 ml extra 也能跑的那条**

一个"需要下载 2.5GB torch 才能跑"的门禁，跑不了几次就会变成注释里的一行字。
**跑不起来的门禁等于没有门禁。**

所以 `configs/recall_gate.yaml` 里每条门禁带一个 `requires` 字段：
`bm25` 组只要 numpy（CI 里必跑），`ml` 组要 sentence-transformers
（本机跑，CI 里自动标 SKIP 并**在汇总里说明跳过了什么**——
跳过不算失败，但必须让人看见自己少跑了哪一条）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from juris_agent.config import PROJECT_ROOT, get_settings  # noqa: E402
from juris_agent.eval.retrieval_eval import (  # noqa: E402
    GoldItem,
    evaluate,
    gold_hash,
    load_gold,
)
from juris_agent.retrieval.build import ArmDeps, build_pipeline  # noqa: E402
from juris_agent.retrieval.pipeline import ArmConfig  # noqa: E402

GATE_CONFIG = PROJECT_ROOT / "configs" / "recall_gate.yaml"

# 退出码。**分开是有用的**：CI 里三种码可以接不同的处理
# （1 = 真的退化了，该拦住；2 = 环境问题，该修 CI 配置；3 = 基线过期，该人工确认）。
EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_ENV = 2
EXIT_STALE = 3


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #


class Gate(BaseModel):
    """一条门禁：跑哪条臂、在哪个切片上、哪些指标不许低于多少。"""

    name: str
    # 人话说明这条门禁在守什么。**必须写**：以后它红了，看的人得知道
    # 该去翻哪一块代码，而不是对着一个数字发呆。
    guards: str = ""
    arm: dict = Field(default_factory=dict)  # ArmConfig 的字段，程序化组成一条臂
    split: str = "test"  # all | dev | test
    source: str = "all"  # all | human | machine
    requires: str = "bm25"  # bm25（只要 numpy）| ml（要 sentence-transformers）
    # 指标下限。**只列该守的那几个**，不是全指标都卡——
    # 每多卡一个指标，就多一分"因为无关的抖动而变红"的概率，
    # 然后大家就学会忽略红色了。
    min_values: dict[str, float] = Field(default_factory=dict)
    tolerance: float = 0.01  # 允许比基线低这么多（测量噪声 + 无关改动的副作用）
    # 标定出来的基线。首次跑 `--update-baseline` 生成。
    baseline: dict[str, float] = Field(default_factory=dict)
    baseline_hashes: dict[str, str] = Field(default_factory=dict)


class GateFile(BaseModel):
    version: int = 1
    gates: list[Gate] = Field(default_factory=list)


def has_module(name: str) -> bool:
    """不 import、只查能不能 import。**查一下就够了，别真去加载 torch。**"""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def requirement_met(req: str) -> tuple[bool, str]:
    """这条门禁能不能在本机跑，跑不了是缺什么。"""
    if req == "bm25":
        return True, ""
    if req == "ml":
        missing = [m for m in ("sentence_transformers", "transformers") if not has_module(m)]
        if missing:
            return False, "缺 " + "、".join(missing) + "（`uv sync --extra ml`）"
        return True, ""
    return False, f"不认识的 requires={req!r}"


# --------------------------------------------------------------------------- #
# 跑一条门禁
# --------------------------------------------------------------------------- #


def load_gold_for(split: str, source: str, gold_path: Path | None) -> tuple[list[GoldItem], str]:
    """载评测集并切片。返回 (题目, 切片说明)。

    ⚠️ **`split` 的默认值是个容易踩的坑**：`GoldItem.split` 默认 `"dev"`，
    而人工那 21 条（`retrieval_gold.jsonl`）**压根没有 `split` 字段**，
    于是它们全都落在 dev 里。也就是说 `--split test` 实际只跑机器造的那 93 条。
    这不是 bug，但**必须在输出里说清楚**，否则"test 上 0.9"会被读成
    "在全部评测集上 0.9"。
    """
    s = get_settings()
    paths: list[Path] = []
    if gold_path is not None:
        paths.append(gold_path)
    else:
        for p in (
            s.paths.eval / "retrieval_gold.jsonl",
            s.paths.eval / "retrieval_gold_machine.jsonl",
        ):
            if p.exists():
                paths.append(p)
    items: list[GoldItem] = []
    for p in paths:
        items.extend(load_gold(p))
    n_all = len(items)
    if split != "all":
        items = [g for g in items if g.split == split]
    if source != "all":
        items = [g for g in items if g.source == source]
    return items, f"{len(items)}/{n_all} 题"


def run_gate(gate: Gate, gold: list[GoldItem], corpus_hash: str) -> dict[str, float]:
    """跑一条臂，返回指标字典。

    **和 `ablation.py` 走的是同一条路**（`build_pipeline` + `evaluate`），
    这是刻意的：门禁和消融表要是各跑一套流程，两者迟早对不上，
    然后就会出现"消融表上没退化、门禁红了"这种没人能解释的局面。
    """
    cfg = ArmConfig(name=gate.name, **gate.arm)
    deps = ArmDeps(corpus_hash=corpus_hash, skip_llm=True)
    pipe = build_pipeline(cfg, deps, artifacts={})

    rankings: dict[str, list[str]] = {}
    for i, g in enumerate(gold):
        if i == 0:
            pipe.search(g.query)  # 首次调用含预热，不计分也不计时
        rankings[g.id] = pipe.search(g.query).expand_article_keys()

    rep = evaluate(gold, rankings)
    return dict(rep.overall.values)


# --------------------------------------------------------------------------- #
# 判定
# --------------------------------------------------------------------------- #


class Verdict(BaseModel):
    gate: str
    status: str  # PASS | FAIL | SKIP | STALE | NEW
    detail: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)


def judge(gate: Gate, metrics: dict[str, float], hashes: dict[str, str]) -> Verdict:
    """三个问题依次问：能不能比 → 该不该比 → 比出来怎么样。顺序不能换。"""
    # ① 有基线吗？没有就只能标定，不能判定。
    if not gate.baseline:
        return Verdict(
            gate=gate.name, status="NEW", metrics=metrics,
            detail="还没有基线（首次运行请加 --update-baseline 标定）",
        )

    # ② 基线和这次测的是同一份语料 + 同一份评测集吗？
    #    ⚠️ **这一步必须在比数值之前**。见文件头「拒绝比较」那一节：
    #    拿两份不同数据的数字做差，结果无论涨跌都是**假的**。
    for key, label in (("corpus_hash", "语料"), ("gold_hash", "评测集"), ("split", "切片")):
        want = gate.baseline_hashes.get(key)
        got = hashes.get(key)
        # split 不参与 hash 校验，只记录——它本来就在配置里，改了配置自然会一起更新。
        if key == "split":
            continue
        if want and got and want != got:
            return Verdict(
                gate=gate.name, status="STALE", metrics=metrics,
                detail=(
                    f"{label}变了（基线 {want} → 现在 {got}）。"
                    f"**这不能比**：数字变化是合法的，比出来的涨跌是假的。"
                    f"先确认这版改动确实动了{label}，再 --update-baseline 重新标定。"
                ),
            )

    # ③ 比数值。**先看下限再看基线**，两者都要满足：
    #    min_values 是绝对底线（掉到这儿以下一定是坏事），
    #    baseline-tolerance 是相对退化（和上次比掉了多少）。
    #    只有相对值时，"一直很低但一直没变"会永远绿；
    #    只有绝对底线时，一次小退步只要没过线就没人管。
    worst = ""
    for metric, floor in gate.min_values.items():
        got = metrics.get(metric)
        if got is None:
            return Verdict(
                gate=gate.name, status="FAIL", metrics=metrics,
                detail=f"指标 {metric} 没算出来——门禁配错了（拼写？），这比退化更该修",
            )
        base = gate.baseline.get(metric)
        if base is not None and got < base - gate.tolerance:
            worst = (
                f"{metric} 从 {base:.4f} 掉到 {got:.4f}"
                f"（跌 {base - got:.4f}，超过容差 {gate.tolerance}）"
            )
            break
        if got < floor:
            worst = f"{metric} = {got:.4f}，低于绝对下限 {floor}"
            break

    if worst:
        return Verdict(gate=gate.name, status="FAIL", metrics=metrics, detail=worst)
    return Verdict(gate=gate.name, status="PASS", metrics=metrics)


# --------------------------------------------------------------------------- #
# 显示
# --------------------------------------------------------------------------- #

_COLORS = {"PASS": "\033[32m", "FAIL": "\033[31m", "SKIP": "\033[33m", "STALE": "\033[33m",
           "NEW": "\033[36m"}
_RESET = "\033[0m"


def fmt_row(v: Verdict, gate: Gate) -> str:
    c = _COLORS.get(v.status, "")
    watched = " ".join(
        f"{m}={v.metrics.get(m, float('nan')):.3f}" for m in gate.min_values
    )
    return f"  {c}{v.status:<5}{_RESET} {v.gate:<16} {watched}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recall 回归门禁（F34）")
    ap.add_argument("--config", default=str(GATE_CONFIG))
    ap.add_argument("--only", action="append", default=None, help="只跑名字匹配的门禁（可重复）")
    ap.add_argument("--split", default=None, help="覆盖配置里的切片（调试用）")
    ap.add_argument("--update-baseline", action="store_true",
                    help="把这次测出来的值写成新基线（**会覆盖旧基线，先确认语料/评测集确实该变**）")
    ap.add_argument("--gold", default=None, help="换一份评测集（调试用）")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[FATAL] 找不到门禁配置：{cfg_path}", file=sys.stderr)
        return EXIT_ENV
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    gate_file = GateFile(**raw)

    s = get_settings()
    if not s.paths.laws.exists():
        print("[FATAL] 没有 data/laws.jsonl，先跑: python scripts/ingest_corpus.py", file=sys.stderr)
        return EXIT_ENV
    corpus_meta = (
        json.loads(s.paths.corpus_meta.read_text(encoding="utf-8"))
        if s.paths.corpus_meta.exists() else {}
    )
    corpus_hash = corpus_meta.get("corpus_hash", "unknown")

    gates = gate_file.gates
    if args.only:
        gates = [g for g in gates if g.name in set(args.only)]
        if not gates:
            print(f"[FATAL] --only 没匹配到任何门禁，现有：{[g.name for g in gate_file.gates]}",
                  file=sys.stderr)
            return EXIT_ENV

    print(f"语料 corpus_hash={corpus_hash}")
    print(f"门禁 {len(gates)} 条\n")

    verdicts: list[Verdict] = []
    for gate in gates:
        if args.split:
            gate.split = args.split
        ok, why = requirement_met(gate.requires)
        if not ok:
            v = Verdict(
                gate=gate.name, status="SKIP",
                detail=f"本机跑不了（{why}）。**这条没跑，不是通过了。**",
            )
            verdicts.append(v)
            print(fmt_row(v, gate) + f"   {v.detail}")
            continue

        gold, shape = load_gold_for(gate.split, gate.source, Path(args.gold) if args.gold else None)
        if not gold:
            v = Verdict(gate=gate.name, status="SKIP",
                        detail=f"切片后没有题（split={gate.split} source={gate.source}）")
            verdicts.append(v)
            print(fmt_row(v, gate) + f"   {v.detail}")
            continue

        metrics = run_gate(gate, gold, corpus_hash)
        hashes = {
            "corpus_hash": corpus_hash,
            "gold_hash": gold_hash(gold),
            "split": gate.split,
            "n": str(len(gold)),
        }

        v = judge(gate, metrics, hashes)
        if args.update_baseline:
            v.status = "NEW"
            v.detail = "已写入基线"
        verdicts.append(v)
        print(fmt_row(v, gate) + f"   [{shape}]" + (f"  {v.detail}" if v.detail else ""))

        if args.update_baseline:
            gate.baseline = {m: round(metrics[m], 6) for m in gate.min_values if m in metrics}
            gate.baseline_hashes = hashes

        if gate.guards and v.status in ("FAIL", "STALE"):
            print(f"        ↳ 这条门禁守的是：{gate.guards}")

    # ---- 汇总 ---------------------------------------------------------- #
    print()
    n_pass = sum(v.status == "PASS" for v in verdicts)
    n_fail = sum(v.status == "FAIL" for v in verdicts)
    n_skip = sum(v.status == "SKIP" for v in verdicts)
    n_stale = sum(v.status == "STALE" for v in verdicts)
    n_new = sum(v.status == "NEW" for v in verdicts)

    if args.update_baseline:
        out = {
            "version": gate_file.version,
            "gates": [g.model_dump() for g in gate_file.gates],
        }
        cfg_path.write_text(
            "# F34 Recall 回归门禁。**由 scripts/check_recall.py 读写**。\n"
            "#\n"
            "# 改 `min_values` / `tolerance` 是改策略，随便改。\n"
            "# 改 `baseline` 之后请再跑一次 `--update-baseline`，让 `baseline_hashes` 跟上，\n"
            "# 否则门禁会判定「基线过期」（状态 STALE）而拒绝比较。\n"
            "#\n"
            "# ⚠️ **只抄 baseline 里的数字、不管 baseline_hashes，是这个文件最典型的误用**：\n"
            "# 那个数字只在「某一份语料 + 某一份评测集」上成立。语料或评测集一变，\n"
            "# 数字变是应该的，拿它做差得到的涨跌是**假的**——真跌了你会去查一个\n"
            "# 不存在的 bug，真涨了你会以为没事。详见 check_recall.py 的文件头。\n"
            "#\n"
            "# 生成命令：python scripts/check_recall.py --update-baseline\n"
            + yaml.safe_dump(out, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(f"已写入 {cfg_path}")
        return EXIT_OK

    parts = [f"{n_pass} 通过"]
    if n_new:
        parts.append(f"{n_new} 未标定")
    if n_fail:
        parts.append(f"{n_fail} 退化")
    if n_stale:
        parts.append(f"{n_stale} 基线过期")
    if n_skip:
        parts.append(f"{n_skip} 跳过")
    print("汇总：" + "、".join(parts))

    if n_skip:
        skipped = "、".join(v.gate for v in verdicts if v.status == "SKIP")
        print(f"⚠️ 跳过的（**没跑 ≠ 通过**）：{skipped}")

    if n_fail:
        print("\n退化的门禁：")
        for v in verdicts:
            if v.status == "FAIL":
                print(f"  · {v.gate}：{v.detail}")
        return EXIT_REGRESSION
    if n_stale:
        return EXIT_STALE
    if n_new:
        print("\n有门禁还没标定基线，跑一次 `--update-baseline` 即可。")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
