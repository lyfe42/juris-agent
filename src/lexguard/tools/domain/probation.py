"""试用期合法性判定。

法条依据：

- **《劳动合同法》第19条**：试用期上限按合同期限分档；
  同一用人单位与同一劳动者**只能约定一次**试用期；
  以完成一定工作任务为期限的、或期限不满 3 个月的，**不得约定试用期**；
  试用期**包含在**劳动合同期限内（仅约定试用期 → 期限视为劳动合同期限）。
- **第20条**：试用期工资不得低于本单位相同岗位最低档工资或劳动合同约定工资的
  **80%**，并不得低于用人单位所在地的**最低工资标准**。
- **第83条**：违法约定的试用期**已履行**的，用人单位以**试用期满月工资**为标准，
  按**已履行的超过法定试用期的期间**支付赔偿金。

分档边界（`term_months` 单位：月）：

| 合同期限                     | 试用期上限 |
|------------------------------|-----------|
| < 3 个月（或以完成工作任务为期限） | 0（不得约定） |
| ≥ 3 且 < 12 个月             | 1 个月    |
| ≥ 12 且 < 36 个月            | 2 个月    |
| ≥ 36 个月（含无固定期限）     | 6 个月    |

**边界上的常见错误**：把「三个月以上不满一年」实现成 `term_months > 3 and
term_months < 12`（漏掉正好 3 个月）、把「一年以上」实现成 `> 12`（漏掉正好 12 个月）。
测试里对 3 / 12 / 36 三个点各有断言。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

#: 分档：(合同期限下界含, 上界不含, 试用期上限月数)。按顺序匹配，None 表示无上界。
_PROBATION_TIERS: list[tuple[float, float | None, float]] = [
    (3.0, 12.0, 1.0),
    (12.0, 36.0, 2.0),
    (36.0, None, 6.0),
]


def max_probation_months(term_months: float, *, task_based: bool = False) -> float:
    """第19条的试用期上限（月）。`task_based` 指"以完成一定工作任务为期限"。"""
    if task_based or term_months < 3:
        return 0.0
    for lower, upper, limit in _PROBATION_TIERS:
        if term_months >= lower and (upper is None or term_months < upper):
            return limit
    return 0.0


class ProbationCheck(BaseModel):
    term_months: float
    task_based: bool
    agreed_probation_months: float
    max_probation_months: float
    is_period_legal: bool
    excess_months: float = Field(description="已履行且超过法定上限的月数，第83条赔偿金的乘数")

    probation_wage_ok: bool | None = None
    wage_floor: float | None = None
    agreed_wage_ratio: float | None = None

    penalty_for_excess: float | None = None
    citations: list[str]
    findings: list[str]
    warnings: list[str] = Field(default_factory=list)


def check_probation(
    *,
    term_months: float,
    agreed_probation_months: float,
    task_based: bool = False,
    agreed_probation_wage: float | None = None,
    agreed_contract_wage: float | None = None,
    same_post_lowest_wage: float | None = None,
    local_minimum_wage: float | None = None,
    performed_probation_months: float | None = None,
    wage_after_probation: float | None = None,
) -> ProbationCheck:
    """判定试用期期限与试用期工资是否合法，并算第83条赔偿金。

    `performed_probation_months` 是**实际已履行**的试用期长度。第83条的赔偿金只
    针对"已履行的、超过法定上限的期间"——所以违法约定但一天没履行，是不产生
    赔偿金的。缺省时按 `agreed_probation_months` 视为已全部履行。
    """
    findings: list[str] = []
    warnings: list[str] = []

    limit = max_probation_months(term_months, task_based=task_based)
    excess = max(0.0, agreed_probation_months - limit)
    period_legal = excess == 0

    if task_based or term_months < 3:
        if agreed_probation_months > 0:
            findings.append(
                f"违法：合同期限 {term_months} 个月"
                f"{'（以完成一定工作任务为期限）' if task_based else ''}，"
                f"依法**不得约定试用期**（第19条第3款），"
                f"但约定了 {agreed_probation_months} 个月。"
            )
        else:
            findings.append("合法：该合同依法不得约定试用期，且未约定。")
    elif period_legal:
        findings.append(
            f"合法：合同期限 {term_months} 个月，法定上限 {limit} 个月，"
            f"约定 {agreed_probation_months} 个月，未超限。"
        )
    else:
        findings.append(
            f"违法：合同期限 {term_months} 个月，法定上限 {limit} 个月，"
            f"约定 {agreed_probation_months} 个月，**超出 {excess} 个月**（第19条第1款）。"
        )

    if agreed_probation_months > term_months:
        findings.append(
            f"违法：约定的试用期（{agreed_probation_months} 个月）长于合同期限"
            f"（{term_months} 个月）。第19条第4款规定试用期包含在劳动合同期限内。"
        )

    # —— 第83条赔偿金 ——
    penalty: float | None = None
    if excess > 0:
        performed = (
            performed_probation_months
            if performed_probation_months is not None
            else agreed_probation_months
        )
        performed_excess = max(0.0, min(performed, agreed_probation_months) - limit)
        if performed_excess > 0 and wage_after_probation is not None:
            penalty = round(performed_excess * wage_after_probation, 2)
            findings.append(
                f"第83条赔偿金 = 已履行的超限期间 {performed_excess} 个月 × "
                f"试用期满月工资 {wage_after_probation} = {penalty} 元。"
            )
        elif performed_excess > 0:
            warnings.append(
                f"存在 {performed_excess} 个月已履行的超限试用期，但未提供"
                "「试用期满月工资」，无法计算第83条赔偿金。"
            )
        else:
            findings.append(
                "违法约定的试用期尚未实际履行，第83条赔偿金以「已履行的超过法定"
                "试用期的期间」为基数，故当前不产生赔偿金。"
            )

    # —— 第20条试用期工资 ——
    wage_ok: bool | None = None
    floor: float | None = None
    ratio: float | None = None
    if agreed_probation_wage is not None:
        floors: list[tuple[str, float]] = []
        if agreed_contract_wage is not None:
            floors.append(("劳动合同约定工资的 80%", round(agreed_contract_wage * 0.8, 2)))
        if same_post_lowest_wage is not None:
            floors.append(("本单位相同岗位最低档工资", same_post_lowest_wage))
        if local_minimum_wage is not None:
            floors.append(("当地最低工资标准", local_minimum_wage))

        if floors:
            floor_label, floor = max(floors, key=lambda kv: kv[1])
            wage_ok = agreed_probation_wage >= floor
            if agreed_contract_wage:
                ratio = round(agreed_probation_wage / agreed_contract_wage, 4)
            if wage_ok:
                findings.append(
                    f"合法：试用期工资 {agreed_probation_wage} ≥ 各类下限的最高者"
                    f"（{floor_label} = {floor}）。"
                )
            else:
                findings.append(
                    f"违法：试用期工资 {agreed_probation_wage} 低于「{floor_label}」"
                    f"= {floor}（第20条）。"
                )
        else:
            warnings.append("提供了试用期工资，但未提供任何比较基准（约定工资 / "
            "同岗位最低档 / 当地最低工资），无法判定是否合法。")

    return ProbationCheck(
        term_months=term_months,
        task_based=task_based,
        agreed_probation_months=agreed_probation_months,
        max_probation_months=limit,
        is_period_legal=period_legal and agreed_probation_months <= term_months,
        excess_months=excess,
        probation_wage_ok=wage_ok,
        wage_floor=floor,
        agreed_wage_ratio=ratio,
        penalty_for_excess=penalty,
        citations=["《劳动合同法》第19条", "第20条", "第83条"],
        findings=findings,
        warnings=warnings,
    )
