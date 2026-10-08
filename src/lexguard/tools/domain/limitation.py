"""劳动仲裁时效计算。

法条依据：**《劳动争议调解仲裁法》第27条**（共四款）。

- **第1款**：仲裁时效期间为**一年**，自当事人**知道或者应当知道**其权利被侵害
  之日起计算。
- **第2款（中断）**：因一方向对方**主张权利**、向有关部门**请求权利救济**、
  或对方**同意履行义务**而中断。**从中断时起重新计算**。
- **第3款（中止）**：因不可抗力或其他正当理由不能申请仲裁的，从中止原因消除
  之日起**继续计算**（注意是"继续"不是"重新"）。
- **第4款（拖欠劳动报酬的特殊规则）**：**劳动关系存续期间**因拖欠劳动报酬发生
  争议，不受第1款一年限制；但**劳动关系终止的，应当自终止之日起一年内提出**。

第4款是这块最容易做错的地方：起点从"知道权利被侵害之日"**换成**"劳动关系终止
之日"，而不是"两个起点的较晚者"。单位拖欠了 5 年工资、今天刚离职的人，时效从
今天起算；而不是因为 5 年前就知道，就认定早已过期。
"""

from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel, Field

#: 一般时效期间，第27条第1款。
LIMITATION_DAYS = 365


def _add_one_year(d: date) -> date:
    """加一年。`date.replace(year=...)` 在 2 月 29 日会抛 ValueError，故单独处理。"""
    try:
        return d.replace(year=d.year + 1)
    except ValueError:  # 2/29 → 次年无此日，退到 2/28
        return d.replace(year=d.year + 1, month=2, day=28)


class LimitationResult(BaseModel):
    base_start: date = Field(description="时效起算日")
    start_reason: str
    deadline: date | None = Field(description="申请仲裁的截止日；None 表示不受一年限制")
    unlimited: bool = Field(description="是否处于第27条第4款「不受一年限制」的状态")
    interrupted: bool = Field(description="是否发生过时效中断")
    interruption_chain: list[tuple[date, date]] = Field(
        default_factory=list, description="(中断日, 该次中断后的新截止日) 序列"
    )
    suspended_days: int = 0
    expired: bool | None = Field(description="截至 today 是否已过时效")
    days_remaining: int | None = None
    citations: list[str]
    steps: list[str]
    warnings: list[str] = Field(default_factory=list)


def compute_limitation(
    *,
    know_date: date,
    today: date | None = None,
    relationship_ended: date | None = None,
    is_wage_arrears: bool = False,
    interruptions: list[date] | None = None,
    suspension_days: int = 0,
) -> LimitationResult:
    """算仲裁时效截止日与是否已过期。

    Args:
        know_date: 知道或应当知道权利被侵害之日（第27条第1款的起算点）。
        today: 用于判断是否已过期，缺省取系统当天。
        relationship_ended: 劳动关系终止之日。为 None 表示关系仍存续。
        is_wage_arrears: 是否为**拖欠劳动报酬**争议（决定第4款是否适用）。
        interruptions: 发生时效中断的日期，顺序不限，内部会排序。
        suspension_days: 因中止而应当顺延的天数（第27条第3款）。
    """
    today = today or date.today()
    steps: list[str] = []
    warnings: list[str] = []

    # —— 起算点 ——
    if is_wage_arrears and relationship_ended is None:
        steps.append(
            "① 属拖欠劳动报酬且劳动关系仍存续 → 适用第27条第4款，"
            "**不受一年仲裁时效限制**"
        )
        return LimitationResult(
            base_start=know_date,
            start_reason="劳动关系存续期间的劳动报酬争议（第27条第4款）",
            deadline=None,
            unlimited=True,
            interrupted=False,
            suspended_days=suspension_days,
            expired=False,
            days_remaining=None,
            citations=["《劳动争议调解仲裁法》第27条第4款"],
            steps=steps,
            warnings=[
                "「不受一年限制」只在劳动关系存续期间有效。一旦离职，须自"
                "劳动关系终止之日起一年内提出，建议尽早固定证据。"
            ],
        )

    if is_wage_arrears:
        base_start = relationship_ended  # type: ignore[assignment]
        reason = "劳动关系已终止的劳动报酬争议，自终止之日起算（第27条第4款）"
        steps.append(f"① 属拖欠劳动报酬，劳动关系已于 {base_start} 终止 → 以此为起算点")
    else:
        base_start = know_date
        reason = "自知道或应当知道权利被侵害之日起算（第27条第1款）"
        steps.append(f"① 一般时效：自知道权利被侵害之日 {base_start} 起算")

    deadline = _add_one_year(base_start)
    steps.append(f"② 一年时效 → 截止日 {deadline}")

    # —— 中断：链式重算 ——
    chain: list[tuple[date, date]] = []
    ignored: list[date] = []
    for d in sorted(interruptions or []):
        if d <= deadline:
            deadline = _add_one_year(d)
            chain.append((d, deadline))
            steps.append(f"③ 中断于 {d}（第27条第2款）→ 重新计算，新截止日 {deadline}")
        else:
            ignored.append(d)
    if ignored:
        warnings.append(
            "以下日期在时效届满之后才发生，不产生中断效果（时效已过不能因事后"
            f"主张而复活）：{', '.join(str(d) for d in ignored)}"
        )

    # —— 中止：顺延 ——
    if suspension_days > 0:
        deadline = deadline + timedelta(days=suspension_days)
        steps.append(
            f"④ 中止顺延 {suspension_days} 天（第27条第3款）→ 截止日 {deadline}"
        )

    remaining = (deadline - today).days
    expired = remaining < 0
    steps.append(
        f"⑤ 截至 {today}：{'已过期 ' + str(-remaining) + ' 天' if expired else '尚余 ' + str(remaining) + ' 天'}"
    )

    if 0 <= remaining <= 30:
        warnings.append(f"仅剩 {remaining} 天，接近时效届满，建议立即申请仲裁。")

    return LimitationResult(
        base_start=base_start,
        start_reason=reason,
        deadline=deadline,
        unlimited=False,
        interrupted=bool(chain),
        interruption_chain=chain,
        suspended_days=suspension_days,
        expired=expired,
        days_remaining=remaining,
        citations=["《劳动争议调解仲裁法》第27条"],
        steps=steps,
        warnings=warnings,
    )
