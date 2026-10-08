"""加班费测试。

核心断言：**休息日可用补休替代，法定休假日不可以**。把这两档同等对待是
最典型的错误实现，所以这里对同一组输入分别断言两者的行为差异。
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from juris_agent.tools.domain.overtime import OvertimeKind, compute_overtime


@pytest.fixture
def base() -> dict[str, float]:
    return {"monthly_wage": 10_000}


class TestWageConversion:
    def test_daily_uses_2175(self, base: dict[str, float]) -> None:
        r = compute_overtime(kind=OvertimeKind.EXTENDED, hours=1, **base)
        assert r.daily_wage == Decimal("459.77")  # 10000 / 21.75

    def test_hourly_is_daily_over_eight(self, base: dict[str, float]) -> None:
        r = compute_overtime(kind=OvertimeKind.EXTENDED, hours=1, **base)
        assert r.hourly_wage == Decimal("57.47")  # 459.77 / 8

    def test_cites_the_conversion_rule(self, base: dict[str, float]) -> None:
        r = compute_overtime(kind=OvertimeKind.EXTENDED, hours=1, **base)
        assert any("劳社部发" in c for c in r.citations)


class TestRates:
    def test_extended_is_150_percent(self, base: dict[str, float]) -> None:
        r = compute_overtime(kind=OvertimeKind.EXTENDED, hours=10, **base)
        assert r.rate == Decimal("1.5")
        assert r.amount == Decimal("862.05")  # 57.47 × 10 × 1.5

    def test_rest_day_is_200_percent(self, base: dict[str, float]) -> None:
        r = compute_overtime(kind=OvertimeKind.REST_DAY, days=2, **base)
        assert r.rate == Decimal("2.0")
        assert r.amount == Decimal("1839.08")  # 459.77 × 2 × 2

    def test_statutory_holiday_is_300_percent(self, base: dict[str, float]) -> None:
        r = compute_overtime(kind=OvertimeKind.STATUTORY_HOLIDAY, days=1, **base)
        assert r.rate == Decimal("3.0")
        assert r.amount == Decimal("1379.31")  # 459.77 × 3


class TestCompensatoryRest:
    def test_rest_day_with_comp_time_pays_nothing(self, base: dict[str, float]) -> None:
        r = compute_overtime(
            kind=OvertimeKind.REST_DAY, days=2, rest_day_compensated=True, **base
        )
        assert r.amount == Decimal("0.00")

    def test_statutory_holiday_cannot_be_replaced_by_comp_time(
        self, base: dict[str, float]
    ) -> None:
        """第44条第3项与第2项的关键差别：法定休假日必须照付 300%。"""
        r = compute_overtime(
            kind=OvertimeKind.STATUTORY_HOLIDAY,
            days=1,
            rest_day_compensated=True,
            **base,
        )
        assert r.amount == Decimal("1379.31")
        assert any("不得以补休替代" in w for w in r.warnings)


class TestHoursCap:
    def test_long_overtime_warns_about_article_41(self, base: dict[str, float]) -> None:
        r = compute_overtime(kind=OvertimeKind.EXTENDED, hours=40, **base)
        assert any("第41条" in w for w in r.warnings)
        # 超时仍然要付钱——违法不影响支付义务
        assert r.amount > Decimal("0")

    def test_within_cap_does_not_warn(self, base: dict[str, float]) -> None:
        r = compute_overtime(kind=OvertimeKind.EXTENDED, hours=36, **base)
        assert not any("第41条" in w for w in r.warnings)


class TestWrongUnit:
    def test_says_when_the_other_unit_is_ignored(self, base: dict[str, float]) -> None:
        r = compute_overtime(kind=OvertimeKind.EXTENDED, hours=2, days=3, **base)
        assert any("days 参数已忽略" in w for w in r.warnings)

    def test_steps_show_substitution(self, base: dict[str, float]) -> None:
        r = compute_overtime(kind=OvertimeKind.REST_DAY, days=2, **base)
        assert any("459.77 × 2" in s for s in r.steps)
