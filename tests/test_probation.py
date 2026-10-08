"""试用期判定测试。

重点是**分档的三个边界点**（3 / 12 / 36 个月）——把「三个月以上」写成 `> 3`
或把「一年以上」写成 `> 12` 都会在那里漏一档。
"""

from __future__ import annotations

import pytest

from juris_agent.tools.domain.probation import check_probation, max_probation_months


class TestTiers:
    @pytest.mark.parametrize(
        ("term", "expected"),
        [
            (0.5, 0.0),   # 不满 3 个月 → 不得约定
            (2.99, 0.0),
            (3.0, 1.0),   # 边界：三个月整 → 落入第一档
            (6.0, 1.0),
            (11.99, 1.0),
            (12.0, 2.0),  # 边界：一年整 → 第二档
            (24.0, 2.0),
            (35.99, 2.0),
            (36.0, 6.0),  # 边界：三年整 → 第三档
            (120.0, 6.0),
        ],
    )
    def test_boundaries(self, term: float, expected: float) -> None:
        assert max_probation_months(term) == expected

    def test_task_based_never_allows_probation(self) -> None:
        assert max_probation_months(60.0, task_based=True) == 0.0


class TestPeriodLegality:
    def test_within_limit_is_legal(self) -> None:
        r = check_probation(term_months=24, agreed_probation_months=2)
        assert r.is_period_legal is True
        assert r.excess_months == 0

    def test_over_limit_is_illegal(self) -> None:
        r = check_probation(term_months=12, agreed_probation_months=3)
        assert r.is_period_legal is False
        assert r.excess_months == 1
        assert any("超出 1.0 个月" in f for f in r.findings)

    def test_probation_longer_than_term_is_illegal(self) -> None:
        r = check_probation(term_months=6, agreed_probation_months=8)
        assert r.is_period_legal is False
        assert any("长于合同期限" in f for f in r.findings)

    def test_no_probation_allowed_for_short_term(self) -> None:
        r = check_probation(term_months=2, agreed_probation_months=1)
        assert r.is_period_legal is False
        assert any("不得约定试用期" in f for f in r.findings)


class TestArticle83Penalty:
    def test_penalty_counts_only_performed_excess(self) -> None:
        """第83条的基数是「已履行的超过法定试用期的期间」。"""
        r = check_probation(
            term_months=6,                 # 上限 1 个月
            agreed_probation_months=3,
            performed_probation_months=3,  # 实际履行满 3 个月
            wage_after_probation=8_000,
        )
        assert r.excess_months == 2
        assert r.penalty_for_excess == 16_000.0  # 2 个月 × 8000

    def test_no_penalty_when_not_yet_performed(self) -> None:
        r = check_probation(
            term_months=6,
            agreed_probation_months=3,
            performed_probation_months=0,
            wage_after_probation=8_000,
        )
        assert r.penalty_for_excess is None
        assert any("尚未实际履行" in f for f in r.findings)

    def test_partial_performance_prorates(self) -> None:
        """只履行了法定上限以内的部分 → 无赔偿金；多履行一个月 → 赔一个月。"""
        within = check_probation(
            term_months=6,                 # 上限 1 个月
            agreed_probation_months=4,
            performed_probation_months=1,  # 恰好只履行到上限
            wage_after_probation=8_000,
        )
        assert within.penalty_for_excess is None

        beyond = check_probation(
            term_months=6,
            agreed_probation_months=4,
            performed_probation_months=2,  # 多履行 1 个月
            wage_after_probation=8_000,
        )
        assert beyond.penalty_for_excess == 8_000.0

    def test_warns_when_base_missing(self) -> None:
        r = check_probation(
            term_months=6, agreed_probation_months=3, performed_probation_months=3
        )
        assert any("试用期满月工资" in w for w in r.warnings)


class TestArticle20Wage:
    def test_below_80_percent_is_illegal(self) -> None:
        r = check_probation(
            term_months=24,
            agreed_probation_months=2,
            agreed_probation_wage=7_000,
            agreed_contract_wage=10_000,
        )
        assert r.probation_wage_ok is False
        assert r.wage_floor == 8_000.0

    def test_exactly_80_percent_is_legal(self) -> None:
        r = check_probation(
            term_months=24,
            agreed_probation_months=2,
            agreed_probation_wage=8_000,
            agreed_contract_wage=10_000,
        )
        assert r.probation_wage_ok is True

    def test_low_contract_wage_min_wage_floor_wins(self) -> None:
        """第20条是「双重下限」：80% 与当地最低工资，取高者。"""
        r = check_probation(
            term_months=24,
            agreed_probation_months=2,
            agreed_probation_wage=2_500,
            agreed_contract_wage=3_000,    # 80% = 2400
            local_minimum_wage=2_500,      # 更高
        )
        assert r.wage_floor == 2_500.0
        assert r.probation_wage_ok is True

    def test_warns_without_any_baseline(self) -> None:
        r = check_probation(
            term_months=24, agreed_probation_months=2, agreed_probation_wage=5_000
        )
        assert r.probation_wage_ok is None
        assert any("比较基准" in w for w in r.warnings)


class TestCitations:
    def test_cites_all_three_articles(self) -> None:
        r = check_probation(term_months=24, agreed_probation_months=2)
        joined = " ".join(r.citations)
        for art in ("第19条", "第20条", "第83条"):
            assert art in joined
