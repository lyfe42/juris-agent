"""路由测试（F14 的验收：4 类输入能路由到正确的下一步）。

⚠️ 这一套测试里有一条**不是**在测"功能"，而是在测"设计约束"：
`test_calc_kind_covers_every_tool` 和 `test_priority_lists_are_subset`。
它们的作用是**在改配置时提醒你还有别的地方要一起改**——
把"漏改一半"这种错误从"上线后才发现"提前到"提交前就红"。见文件末尾的注释。
"""

from __future__ import annotations

import pytest

from juris_agent.agent.router import RouterRules, calc_tool_for, decide


@pytest.fixture(scope="module")
def rules() -> RouterRules:
    return RouterRules.load()


# --------------------------------------------------------------------------- #
# 四类输入的典型例子（F14 验收要求的四类）
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "q",
    [
        "公司不给我出具离职证明怎么办",
        "竞业限制协议没有约定补偿有效吗",
        "劳动合同法第四十七条怎么规定的",
        "被辞退需要提前多久通知",
        "什么情况下公司可以单方解除劳动合同",
    ],
)
def test_law_questions_route_to_law(q: str, rules: RouterRules) -> None:
    assert decide(q, rules=rules).route == "law"


@pytest.mark.parametrize(
    "q,kind",
    [
        ("公司裁员，我干了3年月薪1万，能赔多少", "计算"),
        ("加班费怎么算，我周末加了两天", "加班"),
        # ⚠️ 「试用期」的问题**刻意走 calc 而不是 law**，见下面那条测试的注释
        ("试用期六个月合法吗", "试用期"),
        ("我去年被辞退了，现在还能申请仲裁吗，时效过了没", "时效"),
        # 组合信号命中：没有「劳动合同」这种专业名词，靠「合同」+「问题」认出来
        ("帮我看看这份合同有没有问题", "合同体检"),
        ("这份合同合不合法", "合同体检"),
    ],
)
def test_calc_questions_route_to_calc(q: str, kind: str, rules: RouterRules) -> None:
    d = decide(q, rules=rules)
    assert d.route == "calc", d.reason
    assert d.calc_kind == kind, d.reason


def test_probation_questions_go_to_calc_not_law(rules: RouterRules) -> None:
    """**「试用期最长多久」走 calc，不走 law——这是有意的，不是误判。**

    第一反应会觉得它该走 law（它确实是在问法条）。但走 calc 的收益更大：
    calc 路的工具循环里**同时挂着 `search_law` 和 `check_probation`**，
    模型可以两个都调（先查第 19 条的分档规定，再判断用户的情况合不合法）。
    而 law 路只能检索、不能计算。**路由到"能力更多的支路"是安全的，
    路由到"能力更少的支路"才需要谨慎。**

    代价是每次试用期提问多一次模型调用。换来的是这类问题（本来就要
    "查条文 + 算分档"两件事）能一步到位。
    """
    d = decide("试用期最长能有多久", rules=rules)
    assert d.route == "calc"
    assert d.calc_kind == "试用期"


def test_contract_review_needs_both_groups(rules: RouterRules) -> None:
    """组合信号必须是"每组都中"。

    只提「合同」不算——否则"租赁合同押金不退"会被拉进劳动法。
    （那个句子本来会被越界规则拦住，但**不该依赖另一条规则来兜底**：
    规则之间的隐含依赖是以后加词时最容易踩的坑。）
    """
    assert decide("合同", rules=rules).route != "calc"
    assert decide("我刚才签了合同", rules=rules).route != "calc"


# --------------------------------------------------------------------------- #
# 越界识别：词表天生不全，但**已知的洞必须补上并留个测试**
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "q",
    [
        # 「帮我写一首关于秋天的诗」是**真机演示里踩出来的洞**：
        # 原先一个词都不命中 → 兜底走法条路 → 白检索一遍 + reranker 白跑 3.4 秒，
        # 最后由模型在正文里自己拒绝。模型拒得挺好，但**系统的判定是错的**：
        # `finish_reason` 记成 `answered`，于是"拒答率"这类指标一个都统计不到它。
        "帮我写一首关于秋天的诗",
        "写一首诗",
        "给我讲个笑话",
        "帮我算一道数学题：1+1",
        "今天天气怎么样",
    ],
)
def test_known_off_topic_questions_are_refused_by_the_router(q: str, rules: RouterRules) -> None:
    d = decide(q, rules=rules)
    assert d.route == "out_of_scope", f"{q} 走了 {d.route}（{d.reason}）"


@pytest.mark.parametrize(
    "q",
    [
        # ⚠️ 这一组是给上面那组**踩刹车**的。加越界词最大的风险是**误伤**：
        # 用户正常的需求被当成越界打回去，比漏掉一个越界问题严重得多
        # （前者用户直接走了，后者用户能看到它其实在认真答）。
        # 尤其是「写一份/写一段」这类文稿需求在劳动法场景里很常见
        # （仲裁申请书、情况说明、答辩状），**绝不能因为「写」字被打成越界**。
        "帮我写一份劳动仲裁申请书",
        "帮我写一段情况说明，公司拖欠我工资",
        "劳动合同里的试用期纠纷，被写进了一首诗里，这种情况怎么算",
        "这首诗的作者和公司签了劳动合同，公司不给工资",
    ],
)
def test_writing_requests_that_are_legitimate_are_not_refused(q: str, rules: RouterRules) -> None:
    assert decide(q, rules=rules).route != "out_of_scope", f"{q} 被误判成越界了"


def test_in_scope_signal_beats_out_of_scope(rules: RouterRules) -> None:
    """**顺序不能反**：劳动法信号必须压过越界词。

    句子同时命中两边时（"这首诗写的是劳动合同里的试用期纠纷"），
    说明用户真的在问劳动法，越界词只是顺带出现的。
    判定顺序反过来，这类问题会被直接拒答——**这是最坏的一种错**：
    用户明明是目标用户，却被系统赶走了。

    这条测试是给"以后有人想调整 `decide` 里那几个 if 的顺序"准备的。
    """
    d = decide("这首诗写的是劳动合同里的试用期纠纷，公司不给钱", rules=rules)
    assert d.route in {"calc", "law"}, f"顺序反了，被判成 {d.route}"


@pytest.mark.parametrize(
    "q",
    [
        "离婚时财产怎么分",
        "我朋友醉驾被抓了会判几年",
        "帮我写个爬虫抓股票数据",
        "这个商标被抢注了怎么办",
    ],
)
def test_out_of_scope_questions_refuse(q: str, rules: RouterRules) -> None:
    assert decide(q, rules=rules).route == "out_of_scope"


@pytest.mark.parametrize("q", ["你好", "在吗", "谢谢", "你能做什么"])
def test_chitchat(q: str, rules: RouterRules) -> None:
    assert decide(q, rules=rules).route == "chitchat"


# --------------------------------------------------------------------------- #
# 顺序与边界：这些才是真正会把路由写错的地方
# --------------------------------------------------------------------------- #


def test_labor_signal_beats_out_of_scope_signal(rules: RouterRules) -> None:
    """**这条是整套路由里最重要的一个断言。**

    "离婚" + "公司辞退" 同时出现时，必须判成劳动法问题。
    判定顺序反了（先查越界），一道标准的劳动法题会被当成越界直接扔掉——
    而这类句子在真实对话里非常常见（用户在讲背景时会顺带提到家庭情况）。
    """
    d = decide("我离婚了，公司把我辞退了，能拿多少补偿", rules=rules)
    assert d.route == "calc", d.reason
    assert "离婚" not in d.matched, "不该把越界词当成判据"


def test_greeting_inside_a_real_question_is_not_chitchat(rules: RouterRules) -> None:
    """"你好，我想问下竞业限制" 有 15 个字，**不能**判闲聊。

    闲聊判定要求"命中问候语"**且**"句子够短"。只判前半条会让所有以
    "你好"开头的问题都变成"你好！我能帮你查法条"——用户会觉得系统在装傻。
    """
    d = decide("你好，我想咨询一下竞业限制的问题", rules=rules)
    assert d.route == "law", d.reason


def test_short_followup_uses_history(rules: RouterRules) -> None:
    """多轮追问："那我要是不签呢" 单独看不含任何词，要靠上一轮补全。

    ⚠️ 这是**已知很粗**的补全（拼接上一轮用户发言），但它能把最简单的
    追问接住。真正的做法是继承槽位（F19）。这条测试的意义是
    **把这个能力的存在和它的做法钉住**——不然以后有人重构时把它删了，
    多轮体验会无声退化。
    """
    history = [{"role": "user", "content": "公司要我签竞业限制协议，不签就不给离职证明"}]
    d = decide("那我要是不签呢", history=history, rules=rules)
    assert d.route == "law", d.reason
    assert "history" in d.reason or "上一轮" in d.reason


def test_unknown_question_defaults_to_law_not_refusal(rules: RouterRules) -> None:
    """兜底成 law 而不是拒答。**拒错的代价比答错大**：

    答错了用户能看出来（有引用、可点开原文），拒错了用户直接就走了。
    所以宁可白检索一次。
    """
    assert decide("哎呀这事儿闹的", rules=rules).route == "law"


def test_decision_always_carries_a_reason(rules: RouterRules) -> None:
    """**每个决定都必须带理由。**理由不是给用户看的，是给排查用的：
    线上出现"这句话怎么被拒答了"，第一件事就是看 `route_reason`。"""
    for q in ["你好", "离婚怎么分", "加班费怎么算", "随便说点什么"]:
        d = decide(q, rules=rules)
        assert d.reason.strip(), q


# --------------------------------------------------------------------------- #
# 配置一致性：这几条不是测功能，是测"配置和代码有没有脱节"
# --------------------------------------------------------------------------- #


def _all_calc_kinds(rules: RouterRules) -> list[str]:
    """`router.yaml` 里出现过的全部计算类型（单关键词组 + 组合组）。"""
    return sorted(set(rules.calc) | set(rules.calc_all_of))


def test_calc_tool_for_covers_every_calc_kind(rules: RouterRules) -> None:
    """`router.yaml` 里的每一组，都必须在 `calc_tool_for` 里有对应工具。

    否则会出现"路由认得出这是试用期问题，但没有工具接得住"——
    表现为走到了工具循环、模型不知道该调什么、白烧一次调用。
    """
    for kind in _all_calc_kinds(rules):
        assert calc_tool_for(kind) is not None, (
            f"计算类型 {kind!r} 在 router.yaml 里有词，但 calc_tool_for() 没有对应工具。"
            "加计算器时两处都要改。"
        )


def test_calc_tools_are_registered_tools(rules: RouterRules) -> None:
    """`calc_tool_for` 给的每个名字都必须是**真实注册过**的工具。

    这条断言防的是拼写错误：`calc_severance` 写成 `calc_severence`，
    在路由层完全看不出来（它只是个字符串），要到运行到工具节点才炸。
    """
    from juris_agent.tools.builtin import build_registry

    registered = set(build_registry().names())
    for kind in _all_calc_kinds(rules):
        tool = calc_tool_for(kind)
        assert tool in registered, f"{kind!r} → {tool!r} 不在注册表里（可用：{sorted(registered)}）"


def test_calc_priority_covers_every_calc_kind(rules: RouterRules) -> None:
    """每个计算类型都要出现在优先级列表里，否则它的词永远评不上。

    ⚠️ 这条**不是重复劳动**：`calc_priority` 是手写的有序列表，
    而 `calc_signals` 是另一处。加了新计算器却忘了把它加进优先级列表，
    症状是"这个工具永远不会被路由到"——**静默失效**，没有报错。
    """
    for kind in _all_calc_kinds(rules):
        assert kind in rules.calc_priority, f"{kind!r} 不在 calc_priority 里，永远不会被选中"
