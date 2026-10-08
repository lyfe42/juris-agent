"""经济补偿计算器的边界值测试。

这份测试的价值在于**把第47条那些"含不含本数"的边界钉死**。这些边界是
naive 实现必然出错的地方，也是面试官最爱挑的地方：

- 工作 5 个月 vs 6 个月 → 半个月 vs 一个月工资，差一倍
- 月工资**正好等于** 3 倍社平 → 不封顶（法条是"高于"）
- 月工资超过 3 倍社平但只干了 5 年 → 封顶生效但不触发 12 年上限
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from lexguard.tools.domain.compensation import (
    TerminationCause,
    compute_severance,
    service_multiplier,
)


class TestServiceMultiplier:
    @pytest.mark.parametrize(
        ("months", "expected"),
        [
            (0, "0"),
            (1, "0.5"),
            (5, "0.5"),      # 不满六个月 → 半个月
            (6, "1"),        # 六个月整 → 六个月以上（含本数）→ 一年
            (11, "1"),       # 六个月以上不满一年 → 一年
            (12, "1"),       # 一年整
            (17, "1.5"),     # 1 年 + 5 个月
            (18, "2"),       # 1 年 + 6 个月（含本数）
            (23, "2"),
            (24, "2"),
            (30, "3"),       # 2 年 + 6 个月
            (360, "30"),     # 30 年，折算阶段不封顶
        ],
    )
    def test_boundaries(self, months: int, expected: str) -> None:
        assert service_multiplier(months) == Decimal(expected)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            service_multiplier(-1)


class TestShouldPay:
    """第46条：先判断该不该付。这是朴素实现最容易漏掉的一步。"""

    @pytest.mark.parametrize(
        "cause",
        [
            TerminationCause.EMPLOYEE_FORCED,
            TerminationCause.MUTUAL_EMPLOYER_PROPOSED,
            TerminationCause.NO_FAULT_DISMISSAL,
            TerminationCause.ECONOMIC_LAYOFF,
            TerminationCause.FIXED_TERM_EXPIRY,
        ],
    )
    def test_pays(self, cause: TerminationCause) -> None:
        r = compute_severance(
            cause=cause, months_of_service=36, avg_monthly_wage=10_000
        )
        assert r.pays is True
        assert r.severance_pay == Decimal("30000.00")

    @pytest.mark.parametrize(
        "cause",
        [
            TerminationCause.FAULT_DISMISSAL,
            TerminationCause.MUTUAL_EMPLOYEE_PROPOSED,
            TerminationCause.EMPLOYEE_RESIGNATION,
            TerminationCause.FIXED_TERM_EXPIRY_EMPLOYEE_DECLINED,
        ],
    )
    def test_does_not_pay(self, cause: TerminationCause) -> None:
        r = compute_severance(
            cause=cause, months_of_service=36, avg_monthly_wage=10_000
        )
        assert r.pays is False
        assert r.total_payable == Decimal("0.00")
        # 即使不付，也必须给出"不付≠对方行为合法"的提示
        assert r.warnings

    def test_illegal_dismissal_is_double(self) -> None:
        r = compute_severance(
            cause=TerminationCause.ILLEGAL_DISMISSAL,
            months_of_service=36,
            avg_monthly_wage=10_000,
        )
        assert r.severance_pay == Decimal("30000.00")
        assert r.compensation_total == Decimal("60000.00")
        assert "第48条" in " ".join(r.citations)


class TestThreeTimeCap:
    """第47条第2款：三倍封顶与 12 年封顶是同一条件的两个后果。"""

    def test_high_wage_triggers_both_caps(self) -> None:
        r = compute_severance(
            cause=TerminationCause.ECONOMIC_LAYOFF,
            months_of_service=240,  # 20 年
            avg_monthly_wage=50_000,
            local_avg_monthly_wage=10_000,  # 3× = 30_000
        )
        assert r.capping_applied is True
        assert r.years_cap_applied is True
        assert r.monthly_wage_used == Decimal("30000.00")
        assert r.service_multiplier == Decimal("12")
        assert r.severance_pay == Decimal("360000.00")

    def test_high_wage_short_tenure_caps_wage_only(self) -> None:
        """月薪超标但工龄短 → 只封工资，不触发 12 年上限。"""
        r = compute_severance(
            cause=TerminationCause.ECONOMIC_LAYOFF,
            months_of_service=60,  # 5 年
            avg_monthly_wage=50_000,
            local_avg_monthly_wage=10_000,
        )
        assert r.capping_applied is True
        assert r.years_cap_applied is False
        assert r.service_multiplier == Decimal("5")
        assert r.severance_pay == Decimal("150000.00")

    def test_exactly_three_times_is_not_capped(self) -> None:
        """法条是「高于」三倍才封顶；正好等于三倍不封顶。"""
        r = compute_severance(
            cause=TerminationCause.ECONOMIC_LAYOFF,
            months_of_service=240,  # 20 年
            avg_monthly_wage=30_000,
            local_avg_monthly_wage=10_000,  # 3× = 30_000，正好相等
        )
        assert r.capping_applied is False
        assert r.years_cap_applied is False
        assert r.service_multiplier == Decimal("20")
        assert r.severance_pay == Decimal("600000.00")

    def test_missing_local_wage_warns(self) -> None:
        r = compute_severance(
            cause=TerminationCause.ECONOMIC_LAYOFF,
            months_of_service=24,
            avg_monthly_wage=80_000,
        )
        assert r.capping_applied is False
        assert any("三倍封顶" in w for w in r.warnings)


class TestNoticePay:
    def test_no_fault_without_notice_adds_one_month(self) -> None:
        r = compute_severance(
            cause=TerminationCause.NO_FAULT_DISMISSAL,
            months_of_service=24,
            avg_monthly_wage=10_000,
            notice_given=False,
            last_month_wage=12_000,
        )
        assert r.notice_pay == Decimal("12000.00")
        assert r.total_payable == Decimal("32000.00")

    def test_notice_given_means_no_extra(self) -> None:
        r = compute_severance(
            cause=TerminationCause.NO_FAULT_DISMISSAL,
            months_of_service=24,
            avg_monthly_wage=10_000,
            notice_given=True,
        )
        assert r.notice_pay == Decimal("0.00")

    def test_unconfirmed_notice_surfaces_the_possible_amount(self) -> None:
        """事实未确认时不能静默按"已通知"处理——那会让用户少拿一个月的钱。"""
        r = compute_severance(
            cause=TerminationCause.NO_FAULT_DISMISSAL,
            months_of_service=24,
            avg_monthly_wage=10_000,
        )
        assert r.notice_pay == Decimal("0.00")
        assert any("代通知金" in w for w in r.warnings)

    def test_notice_pay_not_added_for_other_causes(self) -> None:
        """代通知金只属于第40条。经济性裁员即使"没通知"也不加。"""
        r = compute_severance(
            cause=TerminationCause.ECONOMIC_LAYOFF,
            months_of_service=24,
            avg_monthly_wage=10_000,
            notice_given=False,
        )
        assert r.notice_pay == Decimal("0.00")


class TestPrecision:
    def test_float_input_does_not_leak_binary_error(self) -> None:
        """`Decimal(1234.56)` 会得到 1234.5599999…，`money()` 必须走 str 转换。"""
        r = compute_severance(
            cause=TerminationCause.ECONOMIC_LAYOFF,
            months_of_service=12,
            avg_monthly_wage=1234.56,
        )
        assert r.monthly_wage_used == Decimal("1234.56")
        assert r.severance_pay == Decimal("1234.56")

    def test_rounds_half_up_to_cent(self) -> None:
        r = compute_severance(
            cause=TerminationCause.ECONOMIC_LAYOFF,
            months_of_service=6,
            avg_monthly_wage="3333.335",
        )
        assert r.severance_pay == Decimal("3333.34")


class TestSteps:
    def test_steps_include_arithmetic(self) -> None:
        """产品要求计算题给算式，不能只给结论。"""
        r = compute_severance(
            cause=TerminationCause.ECONOMIC_LAYOFF,
            months_of_service=36,
            avg_monthly_wage=10_000,
        )
        joined = "\n".join(r.steps)
        assert "10000.00 × 3" in joined
        assert "第47条" in joined
