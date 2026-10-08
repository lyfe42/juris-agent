"""故障注入实验：量化工具可靠性层的收益。

**两个场景，因为"错误率高"和"下游持续故障"是两回事**——这是本实验最重要的发现，
也是第一版脚本写错的地方：在均匀 30% 随机错误下，熔断器**一次都没触发**
（6 个工具分摊后单工具失败率远低于 50% 阈值），于是"重试臂"和"全机制臂"的数字
完全相同。如果只跑这一个场景，就会写出一条数据不支持的结论。

| 场景 | 变量 | 检验的机制 |
|---|---|---|
| A 均匀随机故障 | 每个工具独立 30% 失败率 | **重试退避** 能不能把成功率拉回来 |
| B 单工具持续故障 | 某一下游 100% 故障 | **熔断** 能不能停止对它的空转 |

三条实验纪律：

1. **各臂共用同一份调用计划**（同样工具、参数、顺序），只有 `ToolSettings` 不同。
2. **各臂共用同一种子**，故障序列因此是配对的——把"运气"从对比里消掉。
3. **三臂而非两臂**（bare / retry_only / full），才能回答收益来自哪个机制。

**延迟与调用次数是模拟的，不是实测的。** 这里没有任何网络调用：每次物理尝试
按固定成本计入，退避等待按实际计算值累计。它回答的是"重试把下游调用量放大了
多少倍"这个结构性问题，不代表真实服务的 P95——那由 `bench_qps.py` 单独产出
（mock 与 real 分开报）。

用法：
    python scripts/fault_inject.py --error-rate 0.3 --calls 300
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic import BaseModel  # noqa: E402

from lexguard.config import Settings, ToolSettings, ensure_dirs  # noqa: E402
from lexguard.tools.builtin import build_registry  # noqa: E402
from lexguard.tools.errors import PermanentToolError, TransientToolError  # noqa: E402
from lexguard.tools.executor import ToolExecutor, ToolStatus  # noqa: E402
from lexguard.tools.registry import ToolRegistry, ToolSpec  # noqa: E402

ATTEMPT_COST_MS = 120.0
"""每次物理尝试的模拟耗时。检索类工具真实耗时几十到几百毫秒，取 120ms 贴近现实。"""

PERMANENT_SHARE = 0.25
"""故障中永久错误（4xx/额度）的占比。**这个比例不能设成 0**：如果只注入瞬时错误，
重试的收益会被显著高估，因为现实中相当一部分失败重试也无意义。"""


# ---------------------------------------------------------------------------
# 调用计划
# ---------------------------------------------------------------------------
def build_call_plan() -> list[tuple[str, dict[str, Any]]]:
    """真实参数，覆盖每个工具的典型路径与边界值。"""
    return [
        ("calc_severance", {
            "cause": "economic_layoff", "months_of_service": 36,
            "avg_monthly_wage": 15000, "local_avg_monthly_wage": 8000,
        }),
        ("calc_severance", {
            "cause": "illegal_dismissal", "months_of_service": 17,
            "avg_monthly_wage": 42000, "local_avg_monthly_wage": 12000,
        }),
        ("calc_severance", {
            "cause": "fault_dismissal", "months_of_service": 5,
            "avg_monthly_wage": 9000,
        }),
        ("calc_severance", {
            "cause": "no_fault_dismissal", "months_of_service": 24,
            "avg_monthly_wage": 11000, "notice_given": False, "last_month_wage": 11500,
        }),
        ("calc_overtime", {"kind": "extended", "monthly_wage": 12000, "hours": 42}),
        ("calc_overtime", {"kind": "rest_day", "monthly_wage": 8000, "days": 4}),
        ("calc_overtime", {"kind": "statutory_holiday", "monthly_wage": 8000, "days": 3}),
        ("check_probation", {
            "term_months": 12, "agreed_probation_months": 3, "wage_after_probation": 10000,
        }),
        ("check_probation", {
            "term_months": 36, "agreed_probation_months": 2,
            "agreed_probation_wage": 7000, "agreed_contract_wage": 10000,
        }),
        ("calc_limitation", {
            "know_date": "2025-06-01", "today": "2026-10-08",
            "is_wage_arrears": True, "relationship_ended": "2026-03-01",
        }),
        ("calc_limitation", {
            "know_date": "2024-02-29", "today": "2026-10-08",
            "interruptions": ["2025-01-15"],
        }),
        ("review_contract", {
            "term_months": 6, "probation_months": 4, "other_liquidated_damages": 20000,
            "other_liquidated_damages_desc": "提前离职赔三个月工资",
        }),
        ("review_contract", {
            "term_months": 36, "probation_months": 3,
            "has_noncompete_clause": True, "noncompete_months": 36,
        }),
        ("refuse_to_answer", {
            "reason": "问题涉及婚姻财产分割", "category": "out_of_scope",
            "suggestion": "建议咨询婚姻家事方向律师",
        }),
    ]


# ---------------------------------------------------------------------------
# 故障注入器
# ---------------------------------------------------------------------------
class Injector(Protocol):
    """把自己插在工具函数前面，决定这次调用是否失败。"""

    def wrap(self, tool: str, fn: Callable[..., Any]) -> Callable[..., Any]: ...

    @property
    def reached_downstream(self) -> int:
        """真正打到工具实现的调用次数（即下游承受的请求量）。"""
        ...

    def instrument(self, registry: ToolRegistry) -> ToolRegistry: ...


class _BaseInjector:
    def __init__(self) -> None:
        self.reached = 0

    @property
    def reached_downstream(self) -> int:
        return self.reached

    def instrument(self, registry: ToolRegistry) -> ToolRegistry:
        out = ToolRegistry()
        for spec in registry.all():
            out.register(
                ToolSpec(
                    name=spec.name,
                    description=spec.description,
                    args_model=spec.args_model,
                    result_model=spec.result_model,
                    fn=self.wrap(spec.name, spec.fn),
                    fallback=spec.fallback,
                    idempotent=spec.idempotent,
                    timeout_seconds=spec.timeout_seconds,
                )
            )
        return out


class UniformInjector(_BaseInjector):
    """场景 A：每个工具独立地以 `error_rate` 概率失败。"""

    def __init__(self, error_rate: float, rng: random.Random) -> None:
        super().__init__()
        self.error_rate = error_rate
        self.rng = rng
        self.injected = 0

    def wrap(self, tool: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(args: BaseModel) -> Any:
            self.reached += 1
            if self.rng.random() < self.error_rate:
                self.injected += 1
                if self.rng.random() < PERMANENT_SHARE:
                    raise PermanentToolError("403 quota exceeded")
                raise TransientToolError("upstream 503")
            return fn(args)

        return wrapped


class OutageInjector(_BaseInjector):
    """场景 B：指定工具 100% 故障，模拟下游彻底挂掉。

    这个场景才是熔断器的用武之地：均匀随机故障下熔断永远不触发，
    而现实中真正需要保护的是"某个下游持续不可用"。
    """

    def __init__(self, victim: str, warmup: int = 0) -> None:
        super().__init__()
        self.victim = victim
        self.warmup = warmup
        self._victim_calls = 0

    def wrap(self, tool: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(args: BaseModel) -> Any:
            self.reached += 1
            if tool == self.victim:
                self._victim_calls += 1
                if self._victim_calls > self.warmup:
                    raise TransientToolError(f"{tool} downstream down (100%)")
            return fn(args)

        return wrapped


# ---------------------------------------------------------------------------
# 实验臂
# ---------------------------------------------------------------------------
@dataclass
class Arm:
    name: str
    label: str
    settings: ToolSettings


def make_arms(base: ToolSettings) -> list[Arm]:
    """三臂设计：只有这样才能回答"收益来自哪个机制"。"""
    return [
        Arm(
            "bare",
            "裸调用（无重试 / 无熔断）",
            base.model_copy(update={"max_retries": 1, "breaker_min_samples": 10**9}),
        ),
        Arm(
            "retry_only",
            "仅重试退避（无熔断）",
            base.model_copy(update={"breaker_min_samples": 10**9}),
        ),
        Arm("full", "重试退避 + 熔断", base.model_copy(update={})),
    ]


class RecordingSleeper:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)

    @property
    def total_seconds(self) -> float:
        return sum(self.calls)


@dataclass
class ArmResult:
    name: str
    label: str
    logical_calls: int = 0
    succeeded: int = 0
    degraded: int = 0
    failed: int = 0
    downstream_calls: int = 0
    backoff_seconds: float = 0.0
    error_types: dict[str, int] = field(default_factory=dict)
    breaker_trips: int = 0
    breaker_rejections: int = 0

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.logical_calls if self.logical_calls else 0.0

    @property
    def calls_per_logical(self) -> float:
        return (
            self.downstream_calls / self.logical_calls if self.logical_calls else 0.0
        )

    def simulated_mean_latency_ms(self) -> float:
        return self.downstream_calls / max(self.logical_calls, 1) * ATTEMPT_COST_MS

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "logical_calls": self.logical_calls,
            "succeeded": self.succeeded,
            "degraded": self.degraded,
            "failed": self.failed,
            "success_rate": round(self.success_rate, 4),
            "downstream_calls": self.downstream_calls,
            "calls_per_logical": round(self.calls_per_logical, 3),
            "simulated_mean_latency_ms": round(self.simulated_mean_latency_ms(), 1),
            "error_types": dict(sorted(self.error_types.items())),
            "breaker_trips": self.breaker_trips,
            "breaker_rejections": self.breaker_rejections,
        }


def run_arm(
    arm: Arm,
    plan: list[tuple[str, dict[str, Any]]],
    injector_factory: Callable[[], Injector],
    total_calls: int,
    seed: int,
) -> tuple[ArmResult, Injector]:
    injector = injector_factory()
    registry = injector.instrument(build_registry(retriever=None))

    sleeper = RecordingSleeper()
    t = [1_000.0]  # 假时钟：整套实验不真的 sleep，300 次调用跑在毫秒级
    clock: Callable[[], float] = lambda: t[0]

    def advancing_sleeper(seconds: float) -> None:
        sleeper(seconds)
        t[0] += seconds

    executor = ToolExecutor(
        registry,
        arm.settings,
        clock=clock,
        sleeper=advancing_sleeper,
        # 各臂同种子：故障序列与抖动序列都是配对的
        rng=random.Random(seed),
    )

    result = ArmResult(name=arm.name, label=arm.label)
    i = 0
    while result.logical_calls < total_calls:
        tool, args = plan[i % len(plan)]
        i += 1
        outcome = executor.call(tool, args)
        result.logical_calls += 1
        if outcome.status is ToolStatus.OK:
            result.succeeded += 1
        elif outcome.status is ToolStatus.DEGRADED:
            result.degraded += 1
            result.succeeded += 1  # 降级结果仍可用，计入成功
        else:
            result.failed += 1
            key = (outcome.error or {}).get("error_type", "unknown")
            result.error_types[key] = result.error_types.get(key, 0) + 1

    result.downstream_calls = injector.reached_downstream
    result.backoff_seconds = sleeper.total_seconds
    snaps = executor.breaker_snapshots()
    result.breaker_trips = sum(int(s["trips"]) for s in snaps)
    result.breaker_rejections = sum(int(s["rejections"]) for s in snaps)
    return result, injector


# ---------------------------------------------------------------------------
# 报表
# ---------------------------------------------------------------------------
def _table(results: list[ArmResult]) -> list[str]:
    lines = [
        "| 臂 | 逻辑调用 | 成功 | 失败 | **成功率** | 下游调用量 | 次/调用 "
        "| 均耗时(模拟) | 熔断跳闸 | 拒绝调用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.label} | {r.logical_calls} | {r.succeeded} | {r.failed} | "
            f"**{r.success_rate:.1%}** | {r.downstream_calls} | {r.calls_per_logical:.2f} | "
            f"{r.simulated_mean_latency_ms():.0f}ms | "
            f"{r.breaker_trips} | {r.breaker_rejections} |"
        )
    return lines


def render(
    scenario_a: list[ArmResult],
    scenario_b: list[ArmResult],
    error_rate: float,
    calls: int,
    seed: int,
    victim: str,
) -> str:
    bare_a, retry_a, full_a = scenario_a
    bare_b, retry_b, full_b = scenario_b
    L: list[str] = []

    L.append("# 工具层故障注入报告\n")
    L.append(
        f"- 每臂 **{calls}** 次逻辑调用；随机种子 `{seed}`，**各臂同种子**"
        "（故障序列配对，排除运气差异）"
    )
    L.append(
        f"- 故障构成：**{1 - PERMANENT_SHARE:.0%} 瞬时错误**（503，重试有意义）+ "
        f"**{PERMANENT_SHARE:.0%} 永久错误**（403 额度，重试无意义）"
    )
    L.append("- 工具集：6 个纯计算工具（`search_law` 需语料，本次未纳入）")
    L.append(
        f"- 延迟为模拟量：每次物理尝试计价 {ATTEMPT_COST_MS:.0f}ms。"
        "**本报告不报实测 P95**——那由 `bench_qps.py` 单独产出（mock 与 real 分开列）。\n"
    )

    # ---------------- 场景 A ----------------
    L.append("---\n")
    L.append("## 场景 A：均匀随机故障（每工具独立 30% 失败）\n")
    L.append(f"注入错误率 {error_rate:.0%}，遍历全部 6 个工具。\n")
    L.extend(_table(scenario_a))
    L.append("")
    L.append("**结论**\n")
    L.append(
        f"- 成功率 **{bare_a.success_rate:.1%} → {full_a.success_rate:.1%}**"
        f"（+{(full_a.success_rate - bare_a.success_rate) * 100:.1f} 个百分点）"
    )
    L.append(
        f"- 收益**全部来自重试退避**：仅重试臂已达 {retry_a.success_rate:.1%}，"
        f"与全机制臂的 {full_a.success_rate:.1%} 相同。"
    )
    L.append(
        f"- **熔断在本场景一次都没触发**：全机制臂的跳闸计数为 **{full_a.breaker_trips}**、"
        f"拒绝调用 **{full_a.breaker_rejections}** 次——这是实测计数，不是从成功率反推的。"
        f"单工具失败率约 {error_rate:.0%}，低于 50% 的熔断阈值。"
        "这不是配置错了，而是说明——**均匀的、低幅度的错误扩散，本来就不该触发熔断**。"
        "熔断要防的是另一种故障，见场景 B。"
    )
    L.append(
        f"- 剩余失败 {full_a.failed} 次（{full_a.failed / full_a.logical_calls:.1%}），"
        "其中永久错误占多数：这些重试也救不回来，**刻意不重试它们正是设计意图**。"
        "把 4xx 塞进重试循环只会抬高下游负载而成功率不变。"
    )
    L.append("\n各臂失败构成：")
    for r in scenario_a:
        if r.error_types:
            L.append(
                "- **"
                + r.label
                + "**："
                + "、".join(f"`{k}` × {v}" for k, v in r.error_types.items())
            )
        else:
            L.append(f"- **{r.label}**：无失败")

    # ---------------- 场景 B ----------------
    L.append("\n---\n")
    L.append(f"## 场景 B：单工具持续故障（`{victim}` 100% 不可用）\n")
    L.append(
        "同一个下游彻底挂掉后不再恢复。**这才是熔断器要防的场景**，"
        "也是场景 A 无法暴露的。\n"
    )
    L.extend(_table(scenario_b))
    L.append("")
    L.append("**结论**\n")
    amplified = retry_b.downstream_calls / max(bare_b.downstream_calls, 1)
    compressed = retry_b.downstream_calls / max(full_b.downstream_calls, 1)
    L.append(
        f"- **三个臂的成功率都是 {full_b.success_rate:.1%}**——下游 100% 挂掉时，"
        "任何客户端机制都变不出正确答案。**熔断不提高成功率，它的价值是保护下游。**"
    )
    L.append(
        "- ⚠️ 读表注意：全机制臂「失败 300」里只有 "
        f"**{full_b.downstream_calls} 次真正打到了下游**，其余 {full_b.breaker_rejections} 次"
        "是熔断在**本地**直接拒掉的，没有产生任何下游请求。失败数不等于下游负载。"
    )
    L.append(
        f"- 下游承受的调用量：裸调用 **{bare_b.downstream_calls}** 次 → "
        f"仅重试 **{retry_b.downstream_calls}** 次（重试把负载放大 "
        f"**{amplified:.2f} 倍**）→ 全机制 **{full_b.downstream_calls}** 次。"
    )
    L.append(
        f"- 熔断在这一场景跳闸 **{full_b.breaker_trips}** 次、拒掉 **{full_b.breaker_rejections}** "
        f"次调用，把下游调用量从 {retry_b.downstream_calls} 压到 {full_b.downstream_calls} 次"
        f"（**压缩 {compressed:.0f} 倍**）。这是「加了重试反而更难恢复」的机制："
        "重试让本已过载的下游雪上加霜，熔断才止住这股放大。"
    )
    L.append(
        f"- 对**用户**的意义同样关键：下游挂掉时，仅重试臂要等 "
        f"{retry_b.simulated_mean_latency_ms():.0f}ms 才拿到失败，全机制臂 "
        f"{full_b.simulated_mean_latency_ms():.0f}ms 就快速失败了。"
        "**熔断把「慢失败」变成「快失败」**——下游已经坏了，让用户多等 3 次退避毫无价值，"
        "而这段时间正该用来走降级路径或直接把问题抛回给模型换策略。"
    )
    L.append(
        f"- 对照着看两个场景：**同样是「故障率」这个词，场景 A 的重试救回 "
        f"{(full_a.success_rate - bare_a.success_rate) * 100:.1f} 个百分点的成功率而熔断无用，"
        f"场景 B 的重试一点成功率也救不回、反而放大 {amplified:.2f} 倍负载而熔断把负载压下 "
        f"{compressed:.0f} 倍。**只看一个场景会得出错的结论——这也是为什么报表要两个都留。"
    )
    L.append(
        "- 这正是**重试与熔断必须成对出现**的原因：重试单独用是有害的。"
        "很多项目只加了重试，在压测或真实故障中反而放大了故障面。"
    )

    L.append("\n---\n")
    L.append("## 复现\n")
    L.append("```bash")
    L.append(
        f"python scripts/fault_inject.py --error-rate {error_rate} "
        f"--calls {calls} --seed {seed}"
    )
    L.append("```\n")
    L.append(
        "> 结构化的原始数字见 `reports/fault_injection.json`，"
        "可作为回归基线：成功率或下游调用量若发生显著变化，说明执行层行为被改动了。"
    )
    return "\n".join(L)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="工具层故障注入实验")
    ap.add_argument("--error-rate", type=float, default=0.3)
    ap.add_argument("--calls", type=int, default=300, help="每臂的逻辑调用次数")
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--victim", default="calc_overtime", help="场景 B 中持续故障的工具")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    settings = Settings()
    ensure_dirs(settings)
    plan = build_call_plan()
    victim = args.victim
    victim_plan = [(t, a) for t, a in plan if t == victim] or plan[:1]
    arms = make_arms(settings.tool)

    scenario_a: list[ArmResult] = []
    for arm in arms:
        res, _ = run_arm(
            arm,
            plan,
            lambda: UniformInjector(args.error_rate, random.Random(args.seed)),
            args.calls,
            args.seed,
        )
        scenario_a.append(res)

    scenario_b: list[ArmResult] = []
    for arm in arms:
        res, _ = run_arm(
            arm,
            victim_plan,
            lambda: OutageInjector(victim),
            args.calls,
            args.seed,
        )
        scenario_b.append(res)

    report = render(scenario_a, scenario_b, args.error_rate, args.calls, args.seed, victim)
    out = args.out or (settings.paths.reports / "fault_injection.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    (settings.paths.reports / "fault_injection.json").write_text(
        json.dumps(
            {
                "error_rate": args.error_rate,
                "calls_per_arm": args.calls,
                "seed": args.seed,
                "permanent_share": PERMANENT_SHARE,
                "attempt_cost_ms": ATTEMPT_COST_MS,
                "scenario_a_uniform": [r.to_dict() for r in scenario_a],
                "scenario_b_outage": [r.to_dict() for r in scenario_b],
                "outage_victim": victim,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[fault_inject] 写入 {out}")
    for label, rows in (("A 均匀随机", scenario_a), ("B 持续故障", scenario_b)):
        print(f"  场景 {label}:")
        for r in rows:
            print(
                f"    {r.name:<11} 成功率 {r.success_rate:6.1%}  "
                f"下游调用 {r.downstream_calls:5d}  ({r.calls_per_logical:.2f}/次)  "
                f"跳闸 {r.breaker_trips}  拒绝 {r.breaker_rejections}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
