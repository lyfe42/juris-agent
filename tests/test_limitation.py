"""仲裁时效测试。

第27条第4款是这块最容易做错的：拖欠劳动报酬的时效起点会**从"知道权利被侵害之日"
换成"劳动关系终止之日"**，而不是取两个起点的较晚者。
"""

from __future__ import annotations

from datetime import date

from juris_agent.tools.domain.limitation import compute_limitation

TODAY = date(2026, 10, 8)


class TestGeneralRule:
    def test_one_year_from_knowledge(self) -> None:
        r = compute_limitation(know_date=date(2026, 1, 1), today=TODAY)
        assert r.deadline == date(2027, 1, 1)
        assert r.expired is False
        # 2026-10-08 → 2027-01-01：10月余 23 天 + 11月 30 + 12月 31 + 1月 1 = 85
        assert r.days_remaining == 85

    def test_expired(self) -> None:
        r = compute_limitation(know_date=date(2025, 1, 1), today=TODAY)
        assert r.deadline == date(2026, 1, 1)
        assert r.expired is True
        assert r.days_remaining is not None and r.days_remaining < 0

    def test_warns_when_close_to_deadline(self) -> None:
        r = compute_limitation(know_date=date(2025, 10, 20), today=TODAY)
        assert any("接近时效届满" in w for w in r.warnings)

    def test_leap_day_start(self) -> None:
        """2 月 29 日起算，次年无此日 → 退到 2 月 28，不能抛异常。"""
        r = compute_limitation(know_date=date(2028, 2, 29), today=date(2028, 3, 1))
        assert r.deadline == date(2029, 2, 28)


class TestWageArrears:
    def test_ongoing_relationship_is_unlimited(self) -> None:
        r = compute_limitation(
            know_date=date(2020, 1, 1),
            today=TODAY,
            is_wage_arrears=True,
            relationship_ended=None,
        )
        assert r.unlimited is True
        assert r.deadline is None
        assert r.expired is False
        assert any("一年内提出" in w for w in r.warnings)

    def test_ended_relationship_starts_from_termination(self) -> None:
        """起点是离职日，不是五年前的那次欠薪——这是第4款的关键。"""
        r = compute_limitation(
            know_date=date(2020, 1, 1),
            today=TODAY,
            is_wage_arrears=True,
            relationship_ended=date(2026, 5, 1),
        )
        assert r.base_start == date(2026, 5, 1)
        assert r.deadline == date(2027, 5, 1)
        assert r.expired is False


class TestInterruption:
    def test_interruption_restarts_the_clock(self) -> None:
        r = compute_limitation(
            know_date=date(2025, 1, 1),
            today=TODAY,
            interruptions=[date(2025, 6, 1)],
        )
        assert r.interrupted is True
        assert r.deadline == date(2026, 6, 1)

    def test_multiple_interruptions_chain(self) -> None:
        r = compute_limitation(
            know_date=date(2025, 1, 1),
            today=TODAY,
            interruptions=[date(2025, 6, 1), date(2026, 4, 1)],
        )
        assert r.deadline == date(2027, 4, 1)
        assert len(r.interruption_chain) == 2

    def test_interruption_after_expiry_has_no_effect(self) -> None:
        """时效已过不能因事后主张而复活。"""
        r = compute_limitation(
            know_date=date(2024, 1, 1),
            today=TODAY,
            interruptions=[date(2026, 9, 1)],  # 截止日 2025-01-01 之后
        )
        assert r.deadline == date(2025, 1, 1)
        assert r.interrupted is False
        assert any("不产生中断效果" in w for w in r.warnings)

    def test_interruption_exactly_on_deadline_counts(self) -> None:
        r = compute_limitation(
            know_date=date(2025, 1, 1),
            today=TODAY,
            interruptions=[date(2026, 1, 1)],  # 正好等于截止日
        )
        assert r.deadline == date(2027, 1, 1)


class TestSuspension:
    def test_suspension_extends_not_restarts(self) -> None:
        r = compute_limitation(
            know_date=date(2026, 1, 1), today=TODAY, suspension_days=30
        )
        assert r.deadline == date(2027, 1, 31)


class TestCitations:
    def test_cites_article_27(self) -> None:
        r = compute_limitation(know_date=date(2026, 1, 1), today=TODAY)
        assert any("第27条" in c for c in r.citations)
