"""加班费计算器。

法条依据：

- **《劳动法》第44条**：三档加成——延长工作时间 150%、休息日 200%、法定休假日 300%。
- **《劳动法》第41条**：加班时长上限（一般每日 ≤1 小时；特殊原因每日 ≤3 小时、
  每月 ≤36 小时）。超限本身是违法，但不影响已发生加班费的支付义务，故只作 warning。
- **劳社部发〔2008〕3号**《关于职工全年月平均工作时间和工资折算问题的通知》：
  月计薪天数 = (365 − 104) ÷ 12 = **21.75** 天；日工资 = 月工资 ÷ 21.75；
  小时工资 = 月工资 ÷ (21.75 × 8)。

**最容易答错的一处**：休息日加班**可以**用补休替代（补了就不付 200%），
但法定休假日加班**不能**用补休替代，必须付 300%。把这两档同等对待是常见错误，
所以 API 里补休参数刻意分开、不共用一个布尔值。

21.75 这个数字**不随法定节假日天数变化**——它的分子只扣了 104 个休息日，没有扣
法定节假日（法定节假日属于"计薪"的日子）。2025 年起全体公民放假的节日由 11 天
增至 13 天（除夕、5 月 2 日），但 21.75 不变。这是个真实的时效性考点。
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field

from juris_agent.tools.domain import money

#: 月计薪天数，(365 − 104) / 12。注意它不随法定节假日数量变化。
MONTHLY_PAY_DAYS = Decimal("21.75")

#: 标准工作日时长，用于把"月工资"折成"小时工资"。
HOURS_PER_DAY = Decimal("8")


class OvertimeKind(str, Enum):
    """加班类型，直接对应第44条的三档。"""

    EXTENDED = "extended"
    """第44条第1项：安排延长工作时间 → 不低于 150%。"""

    REST_DAY = "rest_day"
    """第44条第2项：休息日安排工作又不能安排补休 → 不低于 200%。"""

    STATUTORY_HOLIDAY = "statutory_holiday"
    """第44条第3项：法定休假日安排工作 → 不低于 300%。**补休不能替代。**"""


_RATES: dict[OvertimeKind, Decimal] = {
    OvertimeKind.EXTENDED: Decimal("1.5"),
    OvertimeKind.REST_DAY: Decimal("2.0"),
    OvertimeKind.STATUTORY_HOLIDAY: Decimal("3.0"),
}

_RATE_ARTICLES: dict[OvertimeKind, str] = {
    OvertimeKind.EXTENDED: "第44条第1项",
    OvertimeKind.REST_DAY: "第44条第2项",
    OvertimeKind.STATUTORY_HOLIDAY: "第44条第3项",
}


class OvertimeResult(BaseModel):
    hourly_wage: Decimal
    daily_wage: Decimal
    kind: OvertimeKind
    rate: Decimal
    hours: Decimal
    days: Decimal
    amount: Decimal
    citations: list[str]
    steps: list[str]
    warnings: list[str] = Field(default_factory=list)


def compute_overtime(
    *,
    kind: OvertimeKind,
    monthly_wage: Decimal | int | float | str,
    hours: Decimal | int | float | str = 0,
    days: Decimal | int | float | str = 0,
    rest_day_compensated: bool = False,
) -> OvertimeResult:
    """计算某一档加班费。

    Args:
        kind: 加班类型，决定倍率。
        monthly_wage: 劳动合同约定的月工资（"正常工作时间工资"口径）。
        hours: 延时加班的**小时数**。延时档只能按小时计。
        days: 休息日/法定节假日加班的**天数**。这两档按天计。
        rest_day_compensated: 休息日是否已安排补休。仅对 `REST_DAY` 有意义；
            对法定休假日传入 True 会得到 warning 并**仍然照常计算 300%**。
    """
    rate = _RATES[kind]
    article = _RATE_ARTICLES[kind]
    wage = money(monthly_wage)
    warnings: list[str] = []
    steps: list[str] = []

    daily = money(wage / MONTHLY_PAY_DAYS)
    hourly = money(daily / HOURS_PER_DAY)
    steps.append(
        f"① 日工资 = 月工资 {wage} ÷ 月计薪天数 {MONTHLY_PAY_DAYS} = {daily}"
        f"（劳社部发〔2008〕3号）"
    )
    steps.append(f"   小时工资 = {daily} ÷ {HOURS_PER_DAY} = {hourly}")

    h = Decimal(str(hours))
    d = Decimal(str(days))

    if kind is OvertimeKind.EXTENDED:
        if d:
            warnings.append("延时加班按小时计，传入的 days 参数已忽略。")
        steps.append(
            f"② 第44条第1项：延长工作时间，加成 {rate} 倍 → "
            f"{hourly} × {h} 小时 × {rate}"
        )
        amount = money(hourly * h * rate)
    elif kind is OvertimeKind.REST_DAY:
        if h:
            warnings.append("休息日加班按天计，传入的 hours 参数已忽略。")
        if rest_day_compensated:
            steps.append(
                f"② 第44条第2项：休息日加班**已安排补休** → 不支付 {rate * 100}% 加班费"
            )
            amount = money(0)
            return OvertimeResult(
                hourly_wage=hourly,
                daily_wage=daily,
                kind=kind,
                rate=rate,
                hours=h,
                days=d,
                amount=amount,
                citations=[f"《劳动法》{article}"],
                steps=steps,
                warnings=warnings,
            )
        steps.append(
            f"② 第44条第2项：休息日加班未补休，加成 {rate} 倍 → "
            f"{daily} × {d} 天 × {rate}"
        )
        amount = money(daily * d * rate)
    else:  # STATUTORY_HOLIDAY
        if h:
            warnings.append("法定休假日加班按天计，传入的 hours 参数已忽略。")
        if rest_day_compensated:
            warnings.append(
                "法定休假日加班不得以补休替代，仍须支付 300% 加班费"
                "（《劳动法》第44条第3项与休息日档的区别）。"
            )
        steps.append(
            f"② 第44条第3项：法定休假日加班，加成 {rate} 倍 → "
            f"{daily} × {d} 天 × {rate}"
        )
        amount = money(daily * d * rate)

    steps.append(f"③ 加班费 = {amount}")

    if kind is OvertimeKind.EXTENDED and h > 36:
        warnings.append(
            f"延时加班 {h} 小时已超过《劳动法》第41条「每月不超过 36 小时」的上限。"
            "超过部分仍应支付加班费，同时用人单位可能面临行政处罚。"
        )

    return OvertimeResult(
        hourly_wage=hourly,
        daily_wage=daily,
        kind=kind,
        rate=rate,
        hours=h,
        days=d,
        amount=amount,
        citations=["《劳动法》第44条", "劳社部发〔2008〕3号"],
        steps=steps,
        warnings=warnings,
    )
