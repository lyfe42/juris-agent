"""经济补偿 / 赔偿金 / 代通知金计算器。

法条依据（《劳动合同法》，2008-01-01 施行，2012 年修正）：

- **第46条**：列举「用人单位应当支付经济补偿」的七类情形。**这是本模块存在的
  理由**——朴素实现总是把经济补偿算出来，但第39条（劳动者有过错被辞退）、
  劳动者主动辞职、协商一致由劳动者提出，都是**不支付**的。先判断"该不该付"，
  再算"付多少"，顺序反了就会给出对用户不利/有利的错误结论。
- **第47条**：计算公式。每满一年一个月工资；六个月以上不满一年按一年算；
  不满六个月付半个月。月工资高于当地上年度职工月平均工资三倍的，按三倍封顶，
  且**年限最高不超过十二年**。
- **第87条**：违法解除的赔偿金 = 经济补偿标准的**二倍**。
- **实施条例第20条**：代通知金按劳动者**上一个月的工资**标准确定（不是 12 个月平均）。

两个最容易写错的点，都由测试钉死：
1. 「三倍封顶」和「12 年封顶」是**同一个条件的两个后果**——只有月工资 > 3×社平
   时才同时生效。月工资不超三倍时，哪怕干了 30 年也是 30 个月，不封顶。
2. 「六个月以上不满一年」中的六个月**含本数**（`>= 6`），「不满六个月」是 `< 6`。
   5 个月 vs 6 个月的结果是 0.5 个月 vs 1 个月工资，差一倍。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field

from juris_agent.tools.domain import money


class TerminationCause(str, Enum):
    """解除/终止的情形。命名对应法条，不对应"谁提的"这种模糊说法。"""

    # —— 应当支付经济补偿（第46条）——
    EMPLOYEE_FORCED = "employee_forced"
    """第38条：用人单位有过错（未付报酬、未缴社保等），劳动者被迫解除。"""

    MUTUAL_EMPLOYER_PROPOSED = "mutual_employer_proposed"
    """第36条：**由用人单位提出**、协商一致解除。"""

    NO_FAULT_DISMISSAL = "no_fault_dismissal"
    """第40条：无过失性辞退（医疗期满不能工作、不胜任、客观情况重大变化）。"""

    ECONOMIC_LAYOFF = "economic_layoff"
    """第41条：经济性裁员。"""

    FIXED_TERM_EXPIRY = "fixed_term_expiry"
    """第44条第1项：固定期限合同期满终止（用人单位不续订，或降低条件续订）。"""

    # —— 不支付经济补偿 ——
    FAULT_DISMISSAL = "fault_dismissal"
    """第39条：劳动者有过错（试用期不符合录用条件、严重违纪、严重失职、
    双重劳动关系、欺诈胁迫、被追究刑事责任）。"""

    MUTUAL_EMPLOYEE_PROPOSED = "mutual_employee_proposed"
    """第36条：由**劳动者提出**、协商一致解除。法条未列入第46条，故无补偿。"""

    EMPLOYEE_RESIGNATION = "employee_resignation"
    """劳动者依第37条提前 30 日书面通知辞职。"""

    FIXED_TERM_EXPIRY_EMPLOYEE_DECLINED = "fixed_term_expiry_employee_declined"
    """第46条第5项的**除外情形**：用人单位维持或提高约定条件续订，劳动者不同意。
    注意这个除外只豁免"维持或提高"的情形；若单位降低条件，仍应按
    FIXED_TERM_EXPIRY 处理并支付补偿。"""

    # —— 违法解除，赔偿金为二倍 ——
    ILLEGAL_DISMISSAL = "illegal_dismissal"
    """第48条 / 第87条：用人单位违法解除，劳动者不要求继续履行的，
    按经济补偿标准的二倍支付赔偿金。"""


@dataclass(frozen=True)
class _Rule:
    pays: bool
    """是否支付。"""
    multiplier: Decimal
    """1 = 经济补偿标准；2 = 赔偿金（二倍）。"""
    may_owe_notice_pay: bool
    """是否可能另有代通知金（仅第40条适用：未提前 30 日书面通知时）。"""
    article: str
    label: str


_RULES: dict[TerminationCause, _Rule] = {
    TerminationCause.EMPLOYEE_FORCED: _Rule(
        True, Decimal("1"), False, "第38条+第46条第1项", "劳动者被迫解除"
    ),
    TerminationCause.MUTUAL_EMPLOYER_PROPOSED: _Rule(
        True, Decimal("1"), False, "第36条+第46条第2项", "用人单位提出协商一致解除"
    ),
    TerminationCause.NO_FAULT_DISMISSAL: _Rule(
        True, Decimal("1"), True, "第40条+第46条第3项", "无过失性辞退"
    ),
    TerminationCause.ECONOMIC_LAYOFF: _Rule(
        True, Decimal("1"), False, "第41条+第46条第4项", "经济性裁员"
    ),
    TerminationCause.FIXED_TERM_EXPIRY: _Rule(
        True, Decimal("1"), False, "第44条第1项+第46条第5项", "固定期限合同期满终止"
    ),
    TerminationCause.FAULT_DISMISSAL: _Rule(
        False,
        Decimal("0"),
        False,
        "第39条",
        "劳动者有过错的过失性辞退",
    ),
    TerminationCause.MUTUAL_EMPLOYEE_PROPOSED: _Rule(
        False, Decimal("0"), False, "第36条（未列入第46条）", "劳动者提出协商一致解除"
    ),
    TerminationCause.EMPLOYEE_RESIGNATION: _Rule(
        False, Decimal("0"), False, "第37条", "劳动者主动辞职"
    ),
    TerminationCause.FIXED_TERM_EXPIRY_EMPLOYEE_DECLINED: _Rule(
        False,
        Decimal("0"),
        False,
        "第46条第5项除外情形",
        "单位维持或提高条件续订、劳动者不同意",
    ),
    TerminationCause.ILLEGAL_DISMISSAL: _Rule(
        True, Decimal("2"), False, "第48条+第87条", "违法解除（二倍赔偿金）"
    ),
}


def service_multiplier(months_of_service: int) -> Decimal:
    """第47条的人数换算：把工作月数折成"几个月工资"。

    每满一年 1 个月；剩余 6 个月以上（含 6）不满一年按 1 个月；不满 6 个月按 0.5。
    这里**不含** 12 年封顶——封顶是有条件的，见 `compute_severance`。
    """
    if months_of_service < 0:
        raise ValueError("工作月数不能为负")
    full_years, remaining = divmod(months_of_service, 12)
    multiplier = Decimal(full_years)
    if remaining >= 6:
        multiplier += 1
    elif remaining > 0:
        multiplier += Decimal("0.5")
    return multiplier


class SeveranceResult(BaseModel):
    """结构化的计算结果。`steps` 是给人看的算式代入过程，`citations` 是法条。

    产品要求「计算题必须给算式与代入过程，不能只给结论」——所以 `steps` 不是
    日志，是**要渲染到前端的产物**，也是评测里"答案可核验"的依据。
    """

    cause: TerminationCause
    cause_label: str
    pays: bool
    months_of_service: int
    service_multiplier: Decimal = Field(description="折算后的「几个月工资」，已含 12 年封顶")
    monthly_wage_used: Decimal = Field(description="实际用于计算的月工资（可能被三倍封顶）")
    capping_applied: bool = Field(description="是否触发了三倍社平封顶")
    years_cap_applied: bool = Field(description="是否触发了 12 年封顶")
    severance_pay: Decimal = Field(description="经济补偿（不含代通知金、不含二倍）")
    compensation_total: Decimal = Field(description="本条情形下应付的解除补偿总额")
    notice_pay: Decimal = Field(default=Decimal("0.00"), description="代通知金（第40条未提前通知时）")
    total_payable: Decimal
    citations: list[str]
    steps: list[str]
    warnings: list[str] = Field(default_factory=list)


def compute_severance(
    *,
    cause: TerminationCause,
    months_of_service: int,
    avg_monthly_wage: Decimal | int | float | str,
    local_avg_monthly_wage: Decimal | int | float | str | None = None,
    notice_given: bool | None = None,
    last_month_wage: Decimal | int | float | str | None = None,
) -> SeveranceResult:
    """按第47条计算经济补偿。

    Args:
        cause: 解除/终止情形，决定"该不该付"以及倍率（见 `_RULES`）。
        months_of_service: 在本单位的**工作月数**（累计，不要求整月对齐）。
        avg_monthly_wage: 解除或终止前 **12 个月的平均工资**（第47条第3款的口径）。
        local_avg_monthly_wage: 用人单位所在直辖市/设区的市公布的上年度**职工月平均
            工资**。为 None 时无法适用三倍封顶——结果里会给出 warning，因为
            一线城市高薪场景下这会让金额显著偏高。
        notice_given: 第40条情形下是否提前 30 日书面通知。`False` 则加一个月代通知金。
            **`None` 表示事实未确认**——此时不加钱，但会给出 warning 提示存在这笔
            可能的款项。默认不选 True 或 False 是刻意的：一个计算器不该在事实缺失时
            静默地替用户做选择，那样会让用户少拿一个月的钱却看不到任何提示。
        last_month_wage: 上一个月工资，代通知金按它算（实施条例第20条）。缺省时
            退化用 avg_monthly_wage，并给出 warning。
    """
    rule = _RULES[cause]
    avg = money(avg_monthly_wage)
    steps: list[str] = []
    warnings: list[str] = []

    steps.append(
        f"① 判断是否应付：{rule.article}（{rule.label}）→ "
        f"{'应当支付' if rule.pays else '不需要支付'}"
    )

    if not rule.pays:
        return SeveranceResult(
            cause=cause,
            cause_label=rule.label,
            pays=False,
            months_of_service=months_of_service,
            service_multiplier=Decimal("0"),
            monthly_wage_used=avg,
            capping_applied=False,
            years_cap_applied=False,
            severance_pay=money(0),
            compensation_total=money(0),
            notice_pay=money(0),
            total_payable=money(0),
            citations=[rule.article],
            steps=steps,
            warnings=["不支付经济补偿不等于用人单位的行为一定合法，"
            "请另行核对解除理由本身是否成立（第39条的举证责任在用人单位）。"],
        )

    # ② 年限折算
    raw_multiplier = service_multiplier(months_of_service)
    years, months = divmod(months_of_service, 12)
    if months >= 6:
        basis = f"{years} 年整 + 剩余 {months} 个月（≥6 个月，按一年计）"
    elif months > 0:
        basis = f"{years} 年整 + 剩余 {months} 个月（不满 6 个月，按半个月计）"
    else:
        basis = f"{years} 年整"
    steps.append(
        f"② 工作年限折算（第47条第1款）：{months_of_service} 个月 = {basis} "
        f"→ {raw_multiplier} 个月工资"
    )

    # ③ 三倍封顶 + 12 年封顶（同一条件的两个后果）
    capping_applied = False
    years_cap_applied = False
    wage_used = avg
    multiplier = raw_multiplier

    if local_avg_monthly_wage is None:
        warnings.append(
            "未提供当地上年度职工月平均工资，无法判断是否触发第47条第2款的"
            "「三倍封顶 + 最长 12 年」。若月薪较高，实际金额可能低于此结果。"
        )
    else:
        local_avg = money(local_avg_monthly_wage)
        cap = money(local_avg * 3)
        if avg > cap:
            capping_applied = True
            wage_used = cap
            steps.append(
                f"③ 三倍封顶（第47条第2款）：月工资 {avg} > 当地社平 {local_avg} × 3 "
                f"= {cap} → 按 {cap} 计算"
            )
            if multiplier > 12:
                years_cap_applied = True
                steps.append(
                    f"   年限封顶：折算 {multiplier} 个月 > 12 → 按 12 年封顶"
                )
                multiplier = Decimal("12")
        else:
            steps.append(
                f"③ 未触发三倍封顶：月工资 {avg} ≤ 当地社平 {local_avg} × 3 = {cap}"
                f"，按实际月工资计算，且年限不封顶"
            )

    severance = money(wage_used * multiplier)
    steps.append(f"④ 经济补偿 = {wage_used} × {multiplier} = {severance}")

    # ⑤ 倍率（违法解除二倍赔偿金）
    if rule.multiplier == 2:
        compensation = money(severance * 2)
        steps.append(f"⑤ 违法解除赔偿金（第87条）= {severance} × 2 = {compensation}")
    else:
        compensation = severance
        steps.append("⑤ 非违法解除，不适用第87条二倍标准")

    # ⑥ 代通知金
    notice_pay = money(0)
    if rule.may_owe_notice_pay and notice_given is None:
        steps.append("⑥ 代通知金：未确认是否提前 30 日书面通知，本次不计入")
        warnings.append(
            "本次按「已提前 30 日书面通知」处理，未计入代通知金。若用人单位"
            "**未提前 30 日书面通知**，依第40条可另主张一个月工资作为代通知金"
            "（实施条例第20条：按上一个月工资标准）。"
        )
    elif rule.may_owe_notice_pay and not notice_given:
        notice_base = money(last_month_wage) if last_month_wage is not None else avg
        if last_month_wage is None:
            warnings.append(
                "未提供上一个月工资，代通知金暂按 12 个月平均工资计算；"
                "实施条例第20条规定应按上一个月工资标准确定。"
            )
        notice_pay = notice_base
        steps.append(
            f"⑥ 代通知金（第40条未提前 30 日书面通知，实施条例第20条）"
            f"= 上一个月工资 = {notice_pay}"
        )

    total = money(compensation + notice_pay)
    steps.append(f"⑦ 应付合计 = {compensation} + {notice_pay} = {total}")

    return SeveranceResult(
        cause=cause,
        cause_label=rule.label,
        pays=True,
        months_of_service=months_of_service,
        service_multiplier=multiplier,
        monthly_wage_used=wage_used,
        capping_applied=capping_applied,
        years_cap_applied=years_cap_applied,
        severance_pay=severance,
        compensation_total=compensation,
        notice_pay=notice_pay,
        total_payable=total,
        citations=[rule.article, "第47条"],
        steps=steps,
        warnings=warnings,
    )
