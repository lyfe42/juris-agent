"""内置工具的注册。

七件套：法条检索、经济补偿计算、加班费计算、试用期判定、仲裁时效计算、合同体检、拒答。

**关于金额参数为什么用 `float` 而不是 `Decimal`**：这两个类型在这个项目里各司其职。
*工具入参*面向 LLM，必须用 JSON 的 number（`float`）；真正的精度保护发生在
`domain.money()`——它用 `Decimal(str(x))` 转换，所以 `1234.56` 这个 float 会精确
变成 `Decimal("1234.56")` 而不是 `Decimal("1234.5599999999999…")`。结果模型里则
一律是 `Decimal`，因为那是要直接展示给用户的金额。转换点集中在 `money()` 一处。

**关于 `search_law` 为什么是可选注册的**：它依赖检索层，而检索层依赖语料。
语料是用户侧的离线任务（导出法规原文并核对版本）。语料没到位时注册一个会抛异常的
工具，比注册一个返回假数据的工具好——后者会让评测结果全部失真且难以察觉。
"""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol

from pydantic import BaseModel, Field

from juris_agent.tools.domain.compensation import (
    SeveranceResult,
    TerminationCause,
    compute_severance,
)
from juris_agent.tools.domain.contract_review import ContractInput, Finding, review_contract
from juris_agent.tools.domain.limitation import LimitationResult, compute_limitation
from juris_agent.tools.domain.overtime import OvertimeKind, OvertimeResult, compute_overtime
from juris_agent.tools.domain.probation import ProbationCheck, check_probation
from juris_agent.tools.registry import ToolRegistry, ToolSpec


# ============================================================================
# 1. 法条检索（依赖语料，故 search_law 可选注册）
# ============================================================================
class LawSearchArgs(BaseModel):
    query: str = Field(description="检索用的自然语言问题或关键词，尽量包含法律术语")
    top_k: int = Field(default=8, ge=1, le=50, description="返回条数")


class LawArticleHit(BaseModel):
    article_key: str = Field(description="法条唯一键，形如《劳动合同法》#47")
    law: str
    article_no: str
    text: str
    score: float
    version: str | None = None
    effective_date: str | None = None


class LawSearchResult(BaseModel):
    query: str
    hits: list[LawArticleHit]
    citations: list[str] = Field(default_factory=list)
    suggested_next_action: str | None = Field(
        default=None,
        description="空结果时给模型的下一步建议。返回空列表而不给建议会让模型原地重试。",
    )
    warnings: list[str] = Field(default_factory=list)


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]: ...


def make_search_law_spec(retriever: Retriever) -> ToolSpec:
    def _impl(args: LawSearchArgs) -> LawSearchResult:
        raw_hits = retriever.search(args.query, args.top_k)
        hits = [LawArticleHit.model_validate(h) for h in raw_hits]
        warnings: list[str] = []
        suggestion = None
        if not hits:
            suggestion = (
                "未检索到相关法条。请换用更通用的法律术语重新检索（例如把"
                "「老板不给我发离职证明」换成「解除劳动合同 出具证明 义务」），"
                "或改用 calc_severance / calc_overtime 等计算工具。"
                "若该问题超出劳动法范围，请调用 refuse_to_answer。"
            )
            warnings.append("检索结果为空")
        return LawSearchResult(
            query=args.query,
            hits=hits,
            citations=[h.article_key for h in hits],
            suggested_next_action=suggestion,
            warnings=warnings,
        )

    return ToolSpec(
        name="search_law",
        description=(
            "【用途】检索劳动法及相关法规的具体条文，返回条文原文与编号，用于回答"
            "「法律怎么规定的」「有没有依据」这类问题。这是回答法律依据问题的首选工具。\n"
            "【不适用】不要用它做数值计算——问「能赔多少钱」应该用 calc_severance，"
            "问「加班费多少」应该用 calc_overtime。也不要用它回答非劳动法领域的问题"
            "（婚姻、刑事、公司股权等），那类问题应调用 refuse_to_answer。\n"
            "【返回】条文列表（含法条编号、原文、生效版本）；无结果时会返回下一步建议。"
        ),
        args_model=LawSearchArgs,
        result_model=LawSearchResult,
        fn=_impl,
        idempotent=True,
        # 检索有本地向量库 IO 与 rerank 推理，给一个真超时
        timeout_seconds=None,
        label="查法条",
        group="查法律",
    )


# ============================================================================
# 2. 经济补偿计算
# ============================================================================
class SeveranceArgs(BaseModel):
    cause: TerminationCause = Field(
        description=(
            "解除或终止的情形。必须选一个最贴近的："
            "employee_forced=单位有过错劳动者被迫解除；"
            "mutual_employer_proposed=单位提出协商一致；"
            "no_fault_dismissal=第40条无过失性辞退（医疗期满/不胜任/客观情况变化）；"
            "economic_layoff=经济性裁员；"
            "fixed_term_expiry=固定期限合同期满终止；"
            "fault_dismissal=第39条劳动者有过错被辞退（**无补偿**）；"
            "mutual_employee_proposed=劳动者提出协商一致（**无补偿**）；"
            "employee_resignation=劳动者主动辞职（**无补偿**）；"
            "fixed_term_expiry_employee_declined=单位维持或提高条件续订而劳动者不同意（**无补偿**）；"
            "illegal_dismissal=违法解除（**二倍赔偿金**）"
        )
    )
    months_of_service: int = Field(ge=0, description="在本单位的工作月数（累计）")
    avg_monthly_wage: float = Field(
        gt=0, description="解除或终止前 12 个月的平均工资（第47条第3款的口径）"
    )
    local_avg_monthly_wage: float | None = Field(
        default=None,
        description="单位所在直辖市/设区的市公布的上年度职工月平均工资。"
        "不给的话无法判断是否触发三倍封顶，高薪场景会算高。",
    )
    notice_given: bool | None = Field(
        default=None,
        description="第40条情形下是否已提前 30 日书面通知。"
        "用户明确说了没提前通知 → false（会加一个月代通知金）；"
        "明确说了已提前通知 → true；用户没提这件事 → 留空不填，"
        "不要猜。",
    )
    last_month_wage: float | None = Field(
        default=None, description="上一个月工资，代通知金按它算（实施条例第20条）"
    )


def _impl_severance(args: SeveranceArgs) -> Any:
    return compute_severance(
        cause=args.cause,
        months_of_service=args.months_of_service,
        avg_monthly_wage=args.avg_monthly_wage,
        local_avg_monthly_wage=args.local_avg_monthly_wage,
        notice_given=args.notice_given,
        last_month_wage=args.last_month_wage,
    )


# ============================================================================
# 3. 加班费计算
# ============================================================================
class OvertimeArgs(BaseModel):
    kind: OvertimeKind = Field(
        description=(
            "加班类型：extended=工作日延长工作时间（150%，按小时）；"
            "rest_day=休息日加班（200%，按天，可补休替代）；"
            "statutory_holiday=法定休假日加班（300%，按天，**补休不能替代**）"
        )
    )
    monthly_wage: float = Field(gt=0, description="劳动合同约定的月工资")
    hours: float = Field(default=0, ge=0, description="延时加班的小时数（仅 extended 用）")
    days: float = Field(default=0, ge=0, description="加班天数（仅 rest_day / statutory_holiday 用）")
    rest_day_compensated: bool = Field(
        default=False, description="休息日是否已安排补休。对法定休假日无意义。"
    )


def _impl_overtime(args: OvertimeArgs) -> Any:
    return compute_overtime(
        kind=args.kind,
        monthly_wage=args.monthly_wage,
        hours=args.hours,
        days=args.days,
        rest_day_compensated=args.rest_day_compensated,
    )


# ============================================================================
# 4. 试用期判定
# ============================================================================
class ProbationArgs(BaseModel):
    term_months: float = Field(gt=0, description="劳动合同期限，单位：月")
    agreed_probation_months: float = Field(ge=0, description="合同约定的试用期长度，单位：月")
    task_based: bool = Field(default=False, description="是否以完成一定工作任务为期限的合同")
    agreed_probation_wage: float | None = Field(default=None, description="约定的试用期月工资")
    agreed_contract_wage: float | None = Field(default=None, description="劳动合同约定的转正月工资")
    same_post_lowest_wage: float | None = Field(default=None, description="本单位相同岗位最低档工资")
    local_minimum_wage: float | None = Field(default=None, description="当地最低工资标准")
    performed_probation_months: float | None = Field(
        default=None, description="实际已履行的试用期月数，用于算第83条赔偿金。缺省视为已全部履行。"
    )
    wage_after_probation: float | None = Field(
        default=None, description="试用期满后的月工资，第83条赔偿金的计算基数"
    )


def _impl_probation(args: ProbationArgs) -> Any:
    return check_probation(**args.model_dump())


# ============================================================================
# 5. 仲裁时效计算
# ============================================================================
class LimitationArgs(BaseModel):
    know_date: date = Field(description="知道或应当知道权利被侵害之日，格式 YYYY-MM-DD")
    today: date | None = Field(default=None, description="用于判断是否已过期的基准日，缺省为今天")
    relationship_ended: date | None = Field(
        default=None, description="劳动关系终止之日。不填表示关系仍存续。"
    )
    is_wage_arrears: bool = Field(default=False, description="是否为拖欠劳动报酬争议")
    interruptions: list[date] = Field(
        default_factory=list, description="发生时效中断的日期（主张权利、请求救济、对方同意履行）"
    )
    suspension_days: int = Field(default=0, ge=0, description="因不可抗力等中止而顺延的天数")


def _impl_limitation(args: LimitationArgs) -> Any:
    return compute_limitation(**args.model_dump())


# ============================================================================
# 6. 合同体检
# ============================================================================
class ContractReviewResult(BaseModel):
    findings: list[Finding]
    illegal_count: int
    risky_count: int
    citations: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "本结果仅对可程序化判定的形式合法性做检查（条款是否存在、数值是否超限），"
        "不构成法律意见。实质公平性与具体措辞的效力认定需咨询执业律师。"
    )


def _impl_contract_review(args: ContractInput) -> ContractReviewResult:
    findings = review_contract(args)
    illegal = sum(1 for f in findings if f.severity.value == "illegal")
    risky = sum(1 for f in findings if f.severity.value == "risky")
    citations = sorted({f.citation for f in findings})
    return ContractReviewResult(
        findings=findings,
        illegal_count=illegal,
        risky_count=risky,
        citations=citations,
    )


# ============================================================================
# 7. 拒答
# ============================================================================
class RefuseArgs(BaseModel):
    reason: str = Field(description="为什么无法回答，一句话说明")
    category: str = Field(
        description="问题所属的类别。用 out_of_scope / needs_personal_data / legal_advice_required "
        "之一或自拟的简短英文标签"
    )
    suggestion: str = Field(description="给用户的可操作建议，例如应当咨询哪类专业人士")


class RefusalResult(BaseModel):
    refused: bool = True
    reason: str
    category: str
    suggestion: str
    citations: list[str] = Field(default_factory=list)


def _impl_refuse(args: RefuseArgs) -> RefusalResult:
    return RefusalResult(**args.model_dump())


# ============================================================================
# 注册
# ============================================================================
def build_registry(retriever: Retriever | None = None) -> ToolRegistry:
    """构造工具注册表。

    `retriever` 为 None 时不注册 `search_law`——见模块 docstring 的解释。
    """
    reg = ToolRegistry()

    if retriever is not None:
        reg.register(make_search_law_spec(retriever))

    reg.register(
        ToolSpec(
            name="calc_severance",
            label="算经济补偿 / 赔偿金",
            group="算钱",
            description=(
                "【用途】计算解除/终止劳动合同的经济补偿、违法解除的二倍赔偿金、"
                "以及第40条情形下的代通知金。**先判断该不该付，再算金额。**\n"
                "【不适用】不要用本工具算加班费（用 calc_overtime）、试用期赔偿金"
                "（用 check_probation）、社保问题（超出范围，用 refuse_to_answer）。\n"
                "【注意】cause 参数决定是否支付，选错会让结论完全相反。若用户描述的"
                "解除原因不明确，应先追问而不是猜测。\n"
                "【返回】金额 + 逐条算式代入过程 + 法条依据 + 风险提示。"
            ),
            args_model=SeveranceArgs,
            result_model=SeveranceResult,
            fn=_impl_severance,
            idempotent=True,
            timeout_seconds=0.0,  # 纯计算，不套线程池超时
        )
    )

    reg.register(
        ToolSpec(
            name="calc_overtime",
            label="算加班费",
            group="算钱",
            description=(
                "【用途】按《劳动法》第44条计算加班费：工作日延长 150%、休息日 200%、"
                "法定休假日 300%。\n"
                "【不适用】不要用它算经济补偿或赔偿金。（用 calc_severance）\n"
                "【注意】休息日加班**可以**以补休替代，法定休假日**不可以**——"
                "调用前先确认加班发生在哪一类日子，这也是用户最容易说错的信息。\n"
                "【返回】日工资/小时工资折算过程 + 加班费 + 第41条时长超限提示。"
            ),
            args_model=OvertimeArgs,
            result_model=OvertimeResult,
            fn=_impl_overtime,
            idempotent=True,
            timeout_seconds=0.0,
        )
    )

    reg.register(
        ToolSpec(
            name="check_probation",
            label="查试用期合不合法",
            group="查合同条款",
            description=(
                "【用途】判定试用期是否合法：期限是否超过第19条的分档上限、"
                "试用期工资是否低于第20条的下限、以及第83条违法约定试用期的赔偿金。\n"
                "【不适用】不用于判断「试用期内被辞退是否合法」——那属于第39条"
                "（试用期不符合录用条件）的解除合法性问题，应检索法条并结合"
                "calc_severance 的 fault_dismissal 情形分析。\n"
                "【返回】分档上限、是否超限、超限月数、赔偿金估算、逐条说明。"
            ),
            args_model=ProbationArgs,
            result_model=ProbationCheck,
            fn=_impl_probation,
            idempotent=True,
            timeout_seconds=0.0,
        )
    )

    reg.register(
        ToolSpec(
            name="calc_limitation",
            label="算仲裁时效还剩多久",
            group="算时点",
            description=(
                "【用途】计算劳动仲裁时效：截止日期、是否已过期、剩余天数。"
                "覆盖第27条的一年时效、时效中断重算、中止顺延、以及拖欠劳动报酬的"
                "特殊规则。\n"
                "【不适用】不能回答「我该不该去仲裁」这类策略问题。\n"
                "【返回】起算日与理由、截止日、是否过期、剩余天数、法条依据。"
            ),
            args_model=LimitationArgs,
            result_model=LimitationResult,
            fn=_impl_limitation,
            idempotent=True,
            timeout_seconds=0.0,
        )
    )

    reg.register(
        ToolSpec(
            name="review_contract",
            label="体检劳动合同",
            group="查合同条款",
            description=(
                "【用途】体检劳动合同文本的条款是否合法：试用期、书面合同与二倍工资、"
                "服务期违约金上限、竞业限制期限与补偿、以及第25条禁止的兜底违约金。\n"
                "【不适用】不评判条款的实质公平性，也不审查非劳动法内容（如股权、保密"
                "之外的商业条款）。\n"
                "【返回】按 illegal / risky / ok 三级分类的发现列表，每条附法条依据与建议。"
            ),
            args_model=ContractInput,
            result_model=ContractReviewResult,
            fn=_impl_contract_review,
            idempotent=True,
            timeout_seconds=0.0,
        )
    )

    reg.register(
        ToolSpec(
            name="refuse_to_answer",
            label="（内部）拒答",
            group="其他",
            description=(
                "【用途】当问题超出劳动法咨询范围、或缺少必要事实而无法可靠回答时，"
                "调用本工具给出结构化的拒答，而不是硬编一个答案。**这是被鼓励的行为，"
                "不是失败**。\n"
                "【什么时候用】问题属于其他法律领域（婚姻、刑事、公司股权、知识产权）；"
                "问题需要律师对具体案情做实质判断；问题缺少关键事实（金额、日期、"
                "解除原因）且用户未提供。\n"
                "【什么时候不用】问题在劳动法范围内且信息足够——那应该正常回答，"
                "滥用本工具会导致过度拒答，被评测记为误拒。\n"
                "【返回】拒答原因、类别、给用户的替代建议。"
            ),
            args_model=RefuseArgs,
            result_model=RefusalResult,
            fn=_impl_refuse,
            idempotent=True,
            timeout_seconds=0.0,
        )
    )

    return reg


__all__ = ["Retriever", "build_registry", "make_search_law_spec"]
