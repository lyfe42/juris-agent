"""劳动合同体检：逐条款比对法定标准，输出违法/风险/正常三级发现。

法条依据：

- **第10条 / 第82条**：建立劳动关系应当订立**书面**劳动合同；自用工之日起超过
  一个月不满一年未订立的，应当**每月支付二倍工资**。
- **第17条**：劳动合同的必备条款（用人单位信息、劳动者信息、期限、工作内容与
  工作地点、工作时间与休息休假、劳动报酬、社会保险、劳动保护与职业危害防护）。
- **第19条 / 第20条 / 第83条**：试用期（委托 `probation` 模块）。
- **第22条**：服务期——用人单位提供**专项培训费用**进行专业技术培训的，可约定
  服务期与违约金，违约金**不得超过**尚未履行部分所应分摊的培训费用。
- **第23条**：竞业限制——可约定，但**在竞业限制期限内应按月给予经济补偿**。
- **第24条**：竞业限制人员限于**高级管理人员、高级技术人员和其他负有保密义务
  的人员**；期限**不得超过二年**。
- **第25条**：**除第22条和第23条两种情形外，不得约定由劳动者承担违约金。**
  这是最常被滥用的条款——"提前离职赔 3 个月工资"之类一律无效。

说明：本模块只判定**可程序化判定的形式合法性**（条款存在与否、数值是否超限），
不评判条款的实质公平性或具体措辞的法律效力——那需要律师。工具的输出里会明确
说清这条边界，避免给出"像法律意见但不是"的答案。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from juris_agent.tools.domain.probation import check_probation


class Severity(str, Enum):
    ILLEGAL = "illegal"
    """违反法律强制性规定，该条款无效或可能引发赔偿/行政处罚。"""
    RISKY = "risky"
    """未必违法，但存在争议风险或对劳动者明显不利，建议协商修改。"""
    OK = "ok"
    """未发现问题。"""


class Finding(BaseModel):
    severity: Severity
    title: str
    detail: str
    citation: str
    suggestion: str | None = None


class ContractInput(BaseModel):
    """待体检的合同要素。字段全部可选：只对提供了的内容做检查，避免臆测。"""

    term_months: float | None = None
    task_based: bool = False

    probation_months: float | None = None
    probation_wage: float | None = None
    contract_wage: float | None = None
    same_post_lowest_wage: float | None = None
    local_minimum_wage: float | None = None

    # 离职相关
    written_contract: bool = True
    months_without_written_contract: float = 0.0
    monthly_wage: float | None = None

    # 服务期 / 竞业限制
    has_service_period_clause: bool = False
    training_cost: float | None = None
    service_period_months: float | None = None
    service_period_served_months: float | None = None
    service_period_liquidated_damages: float | None = None

    has_noncompete_clause: bool = False
    noncompete_months: float | None = None
    noncompete_monthly_compensation: float | None = None
    is_senior_or_confidential_role: bool | None = None

    # 兜底的违约金条款：既非服务期也非竞业限制
    other_liquidated_damages: float | None = None
    other_liquidated_damages_desc: str | None = None

    # 必备条款缺失项（第17条），由调用方给出缺失的条款名
    missing_mandatory_clauses: list[str] = Field(default_factory=list)


def review_contract(c: ContractInput) -> list[Finding]:
    """返回体检发现列表。空列表表示提供了的字段全部合格（不代表合同整体合法）。"""
    out: list[Finding] = []

    # —— 书面劳动合同（第10条、第82条）——
    if not c.written_contract:
        detail = "用人单位自用工之日起应当订立书面劳动合同（第10条）。"
        if c.months_without_written_contract >= 1:
            detail += (
                f"已 {c.months_without_written_contract} 个月未订立，"
                "依第82条应当每月支付二倍工资"
            )
            if c.monthly_wage is not None:
                extra = round(c.months_without_written_contract * c.monthly_wage, 2)
                detail += f"，估算二倍工资差额约 {extra} 元"
            detail += "。"
        out.append(
            Finding(
                severity=Severity.ILLEGAL,
                title="未订立书面劳动合同",
                detail=detail,
                citation="《劳动合同法》第10条、第82条",
                suggestion="要求补签书面合同，并保留实际用工起始日的证据（考勤、工资流水）。",
            )
        )

    # —— 必备条款（第17条）——
    if c.missing_mandatory_clauses:
        out.append(
            Finding(
                severity=Severity.RISKY,
                title="缺少法定必备条款",
                detail="合同缺少以下必备条款：" + "、".join(c.missing_mandatory_clauses),
                citation="《劳动合同法》第17条",
                suggestion="要求用人单位补充完整后再签署。",
            )
        )

    # —— 试用期 ——
    if c.probation_months is not None and c.term_months is not None:
        pc = check_probation(
            term_months=c.term_months,
            task_based=c.task_based,
            agreed_probation_months=c.probation_months,
            agreed_probation_wage=c.probation_wage,
            agreed_contract_wage=c.contract_wage,
            same_post_lowest_wage=c.same_post_lowest_wage,
            local_minimum_wage=c.local_minimum_wage,
            wage_after_probation=c.contract_wage,
        )
        for f in pc.findings:
            if "试用期工资" in f:
                out.append(
                    Finding(
                        severity=Severity.OK if pc.probation_wage_ok else Severity.ILLEGAL,
                        title="试用期工资",
                        detail=f,
                        citation="《劳动合同法》第20条",
                        suggestion=None
                        if pc.probation_wage_ok
                        else "要求按约定工资的 80% 与当地最低工资标准的较高者支付。",
                    )
                )
            elif f.startswith("违法") or f.startswith("第83条"):
                out.append(
                    Finding(
                        severity=Severity.ILLEGAL,
                        title="试用期约定",
                        detail=f,
                        citation="《劳动合同法》第19条、第83条",
                        suggestion="超限部分可主张第83条赔偿金。",
                    )
                )
            elif f.startswith("合法"):
                out.append(
                    Finding(
                        severity=Severity.OK,
                        title="试用期约定",
                        detail=f,
                        citation="《劳动合同法》第19条",
                    )
                )

    # —— 服务期（第22条）——
    if c.has_service_period_clause:
        if c.training_cost is None:
            out.append(
                Finding(
                    severity=Severity.RISKY,
                    title="服务期违约金缺少培训费用依据",
                    detail="第22条要求服务期以用人单位提供**专项培训费用**进行专业技术"
                    "培训为前提。普通入职培训、岗前培训不属于专项培训。",
                    citation="《劳动合同法》第22条",
                    suggestion="要求用人单位出示专项培训费用的凭证。",
                )
            )
        if (
            c.service_period_liquidated_damages is not None
            and c.training_cost is not None
            and c.service_period_months
            and c.service_period_served_months is not None
        ):
            unserved = max(0.0, c.service_period_months - c.service_period_served_months)
            allowed = round(c.training_cost * unserved / c.service_period_months, 2)
            if c.service_period_liquidated_damages > allowed:
                out.append(
                    Finding(
                        severity=Severity.ILLEGAL,
                        title="服务期违约金超过法定上限",
                        detail=f"第22条限定违约金不得超过**尚未履行部分所应分摊的培训费用**。"
                        f"约定 {c.service_period_liquidated_damages} 元 > 上限 {allowed} 元。",
                        citation="《劳动合同法》第22条第2款",
                        suggestion="超出部分无效，可主张按分摊比例降低。",
                    )
                )
            else:
                out.append(
                    Finding(
                        severity=Severity.OK,
                        title="服务期违约金",
                        detail=f"{c.service_period_liquidated_damages} 元未超过分摊上限 {allowed} 元。",
                        citation="《劳动合同法》第22条第2款",
                    )
                )

    # —— 竞业限制（第23条、第24条）——
    if c.has_noncompete_clause:
        if c.noncompete_months is not None and c.noncompete_months > 24:
            out.append(
                Finding(
                    severity=Severity.ILLEGAL,
                    title="竞业限制期限超过二年",
                    detail=f"约定 {c.noncompete_months} 个月，超过第24条规定的**二年**上限。",
                    citation="《劳动合同法》第24条第2款",
                    suggestion="超过二年的部分无效。",
                )
            )
        if c.noncompete_monthly_compensation is None:
            out.append(
                Finding(
                    severity=Severity.RISKY,
                    title="竞业限制未约定经济补偿",
                    detail="第23条要求用人单位在竞业限制期限内**按月给予经济补偿**。"
                    "未约定的，劳动者履行后可主张补偿。",
                    citation="《劳动合同法》第23条第2款",
                    suggestion="明确约定补偿标准与支付方式。",
                )
            )
        if c.is_senior_or_confidential_role is False:
            out.append(
                Finding(
                    severity=Severity.ILLEGAL,
                    title="竞业限制主体不适格",
                    detail="第24条将竞业限制人员限于**高级管理人员、高级技术人员和"
                    "其他负有保密义务的人员**。普通岗位约定竞业限制对其不发生效力。",
                    citation="《劳动合同法》第24条第1款",
                )
            )

    # —— 兜底违约金（第25条）——
    if c.other_liquidated_damages:
        desc = f"（{c.other_liquidated_damages_desc}）" if c.other_liquidated_damages_desc else ""
        out.append(
            Finding(
                severity=Severity.ILLEGAL,
                title="约定由劳动者承担的违约金无效",
                detail=f"第25条规定：除**服务期（第22条）**和**竞业限制（第23条）**两种"
                f"情形外，用人单位不得与劳动者约定由劳动者承担违约金。"
                f"该条款{desc}约定 {c.other_liquidated_damages} 元，属无效条款。",
                citation="《劳动合同法》第25条",
                suggestion="常见的「提前离职赔一个月工资」「未满服务期扣发工资」均属此类。",
            )
        )

    return out
