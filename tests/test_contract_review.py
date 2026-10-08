"""合同体检测试。

每个断言对应一条法条。特别是**第25条**——"除服务期和竞业限制外不得约定
由劳动者承担违约金"，这是现实中最常被滥用、也最常被求职者默默接受的条款。
"""

from __future__ import annotations

from lexguard.tools.domain.contract_review import (
    ContractInput,
    Severity,
    review_contract,
)


def _by_citation(findings, needle: str):
    return [f for f in findings if needle in f.citation]


def _titles(findings, severity: Severity) -> list[str]:
    return [f.title for f in findings if f.severity is severity]


class TestWrittenContract:
    def test_missing_written_contract_is_illegal(self) -> None:
        f = review_contract(
            ContractInput(
                written_contract=False, months_without_written_contract=3, monthly_wage=10_000
            )
        )
        assert "未订立书面劳动合同" in _titles(f, Severity.ILLEGAL)
        assert any("30000.0" in x.detail for x in f)

    def test_under_one_month_does_not_trigger_double_wage(self) -> None:
        f = review_contract(
            ContractInput(
                written_contract=False, months_without_written_contract=0.5, monthly_wage=10_000
            )
        )
        assert not any("二倍工资" in x.detail for x in f)

    def test_written_contract_produces_no_finding(self) -> None:
        f = review_contract(ContractInput(written_contract=True))
        assert _by_citation(f, "第10条") == []


class TestMandatoryClauses:
    def test_missing_clauses_is_risky(self) -> None:
        f = review_contract(
            ContractInput(missing_mandatory_clauses=["工作地点", "社会保险"])
        )
        titles = _titles(f, Severity.RISKY)
        assert "缺少法定必备条款" in titles
        assert any("工作地点" in x.detail for x in f)


class TestNonCompete:
    def test_over_two_years_is_illegal(self) -> None:
        f = review_contract(
            ContractInput(has_noncompete_clause=True, noncompete_months=36)
        )
        assert "竞业限制期限超过二年" in _titles(f, Severity.ILLEGAL)

    def test_exactly_two_years_is_allowed(self) -> None:
        f = review_contract(
            ContractInput(has_noncompete_clause=True, noncompete_months=24)
        )
        assert "竞业限制期限超过二年" not in _titles(f, Severity.ILLEGAL)

    def test_no_compensation_is_risky(self) -> None:
        f = review_contract(ContractInput(has_noncompete_clause=True, noncompete_months=12))
        assert "竞业限制未约定经济补偿" in _titles(f, Severity.RISKY)

    def test_wrong_role_is_illegal(self) -> None:
        f = review_contract(
            ContractInput(
                has_noncompete_clause=True,
                noncompete_months=12,
                noncompete_monthly_compensation=3_000,
                is_senior_or_confidential_role=False,
            )
        )
        assert "竞业限制主体不适格" in _titles(f, Severity.ILLEGAL)


class TestArticle25LiquidatedDamages:
    def test_blanket_penalty_clause_is_illegal(self) -> None:
        f = review_contract(
            ContractInput(
                other_liquidated_damages=20_000,
                other_liquidated_damages_desc="提前离职需赔偿三个月工资",
            )
        )
        titles = _titles(f, Severity.ILLEGAL)
        assert "约定由劳动者承担的违约金无效" in titles
        assert any("第25条" in x.citation for x in f)

    def test_absent_clause_produces_nothing(self) -> None:
        f = review_contract(ContractInput(other_liquidated_damages=None))
        assert not any("第25条" in x.citation for x in f)


class TestServicePeriod:
    def test_damages_over_allocated_share_is_illegal(self) -> None:
        """第22条：违约金不得超过尚未履行部分所应分摊的培训费用。"""
        f = review_contract(
            ContractInput(
                has_service_period_clause=True,
                training_cost=30_000,
                service_period_months=36,
                service_period_served_months=12,   # 未履行 24 → 上限 20000
                service_period_liquidated_damages=25_000,
            )
        )
        assert "服务期违约金超过法定上限" in _titles(f, Severity.ILLEGAL)

    def test_damages_within_share_is_ok(self) -> None:
        f = review_contract(
            ContractInput(
                has_service_period_clause=True,
                training_cost=30_000,
                service_period_months=36,
                service_period_served_months=12,
                service_period_liquidated_damages=18_000,
            )
        )
        assert "服务期违约金" in _titles(f, Severity.OK)

    def test_missing_training_cost_is_risky(self) -> None:
        f = review_contract(ContractInput(has_service_period_clause=True))
        assert "服务期违约金缺少培训费用依据" in _titles(f, Severity.RISKY)


class TestProbationIntegration:
    def test_over_limit_probation_reported(self) -> None:
        f = review_contract(ContractInput(term_months=12, probation_months=3))
        assert "试用期约定" in _titles(f, Severity.ILLEGAL)

    def test_low_probation_wage_reported(self) -> None:
        f = review_contract(
            ContractInput(
                term_months=24,
                probation_months=2,
                probation_wage=7_000,
                contract_wage=10_000,
            )
        )
        assert "试用期工资" in _titles(f, Severity.ILLEGAL)

    def test_not_checked_when_term_missing(self) -> None:
        f = review_contract(ContractInput(probation_months=3))
        assert not any("试用期" in x.title for x in f)
