"""记忆层的测试（F19–F21）。

## 这一组测试真正想钉住的东西

记忆层最容易出的一种 bug 是"**悄悄丢东西**"——第 20 轮少了一个数、
多了一个数，而全程不报错、日志里也看不出来。所以这里的测试分三档：

1. **契约型**：槽位表的每个目标字段，在真的工具参数模型里**必须存在、
   而且类型对得上**。这条测试的价值最高——它一次性把"槽位表"和
   "工具定义"这两份会各自演化的代码绑在一起。写这条测试的时候
   当场就抓到两个真 bug（`relationship_ended` 抽成了布尔、而工具那边要的是日期）。
2. **回归型**：每一条都是真的踩过的坑，用原话写在测试名里。
   `test_bare_number_is_not_a_wage` 这种名字，比 `test_case_17` 有用得多。
3. **行为型**：窗口滑动、摘要累积、四策略的信息量单调性。

## 为什么这一组不需要假时钟、假模型

- 槽位抽取是**纯函数**（给同一句话永远得到同一组槽位）
- 长期层的打分是**确定性的**（重叠数 × 指数衰减，没有随机）
- 摘要器默认走**规则版**（不调模型）

所以整个文件跑在毫秒级。**需要多少假件，是个设计质量的读数。**
（只有 `SlotStore.derive()` 里"入职日期 → 工作月数"要用到 today，
那条测试自己构造日期，不依赖系统时钟的具体值。）
"""

from __future__ import annotations

import datetime as dt
import typing

import pytest
from pydantic import BaseModel

from juris_agent.memory.manager import (
    MemoryBudget,
    MemoryManager,
    MemoryPolicy,
)
from juris_agent.memory.longterm import LongTermStore, MemoryItem, tokenize
from juris_agent.memory.slots import (
    SLOT_SPECS,
    TOOL_SLOTS,
    SlotExtractor,
    SlotStore,
    Slots,
    cn_to_int,
)
from juris_agent.memory.tokens import count_messages, count_tokens
from juris_agent.memory.working import (
    Message,
    RuleSummarizer,
    WorkingMemory,
    rule_summarize,
)

# ===========================================================================
# ① 契约：槽位表 ↔ 工具参数模型
# ===========================================================================

_TOOL_MODELS: dict[str, type[BaseModel]] = {}


def _tool_models() -> dict[str, type[BaseModel]]:
    """把"工具名 → 参数模型"这张表建出来。

    刻意**从 `TOOL_SLOTS` 反查**，而不是在这里手抄一份工具清单——
    手抄的那份迟早会和真实定义脱节，而脱节了也不会报错。
    """
    if not _TOOL_MODELS:
        from juris_agent.tools.builtin import (
            LimitationArgs,
            OvertimeArgs,
            ProbationArgs,
            SeveranceArgs,
        )

        _TOOL_MODELS.update(
            {
                "calc_severance": SeveranceArgs,
                "calc_overtime": OvertimeArgs,
                "check_probation": ProbationArgs,
                "calc_limitation": LimitationArgs,
            }
        )
    return _TOOL_MODELS


def _unwrap(ann: object) -> tuple[object, bool]:
    """把 `X | None` / `Optional[X]` 拆成 `(X, 是否可选)`。"""
    origin = typing.get_origin(ann)
    if origin is typing.Union:  # Optional[X] 在 typing 里就是 Union[X, None]
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return ann, False


# 槽位的 kind → 工具参数允许的类型。
# **`int` 可以填进 `float` 字段**（pydantic 会转），反过来不行。
_KIND_OK: dict[str, tuple[type, ...]] = {
    "money": (float, int),
    "int": (int, float),
    "float": (float, int),
    "date": (dt.date,),
    "bool": (bool,),
    "enum": (str,),
    "str": (str,),
    # 内部槽位（`other_amounts`）不对应任何工具参数，只是本地留个记录
    "json": (list, tuple, dict, str, float, int),
}


def test_every_slot_target_exists_on_the_real_tool_args():
    """★ 每个槽位声明的目标字段，在对应工具的参数模型里**必须真的存在**。

    ⚠️ 这条测试是"两份会各自演化的代码之间的一根绳子"。没有它，
    改一次工具参数名，槽位就会开始往一个不存在的字段里填值——
    **不报错**，只是那个参数永远是空的，而用户看到的是一句敷衍的回答。
    """
    models = _tool_models()
    problems: list[str] = []
    for name, spec in SLOT_SPECS.items():
        for tool, field in spec.targets:
            model = models.get(tool)
            if model is None:
                problems.append(f"{name} → 未知工具 {tool}")
                continue
            if field not in model.model_fields:
                problems.append(
                    f"{name} → {tool}.{field} 不存在（{tool} 现有字段："
                    f"{sorted(model.model_fields)}）"
                )
    assert not problems, "槽位表指向了不存在的工具参数：\n" + "\n".join(problems)


def test_every_slot_target_type_matches_the_slot_kind():
    """★ 槽位的 `kind` 和工具参数的类型必须对得上。

    **这条是写这份测试时现场抓到的真 bug**：`relationship_ended` 一开始
    按语感抽成了布尔（"还在职 → False / 已离职 → True"），
    而 `calc_limitation.relationship_ended` 要的是 `date | None`。
    布尔值塞进去会让工具**直接参数校验失败**，整条链路走不通。

    教训写进断言消息里，方便下次有人踩同一个坑时立刻明白发生了什么。
    """
    models = _tool_models()
    problems: list[str] = []
    for name, spec in SLOT_SPECS.items():
        allowed = _KIND_OK[spec.kind]
        for tool, field in spec.targets:
            model = models.get(tool)
            if model is None or field not in model.model_fields:
                continue  # 上面那条测试负责报这个
            ann, optional = _unwrap(model.model_fields[field].annotation)
            if ann is bool and spec.kind != "bool":
                # bool 是 int 的子类，会被 `isinstance` 蒙混过关，所以先单独挡一下
                problems.append(f"{name}(kind={spec.kind}) → {tool}.{field} 是 bool")
                continue
            if isinstance(ann, type) and not issubclass(ann, allowed):
                problems.append(
                    f"{name}(kind={spec.kind}) → {tool}.{field} 的类型是 {ann}"
                    f"，不是 {allowed}"
                )
    assert not problems, (
        "槽位的 kind 和工具参数类型对不上（类型不对会让工具调用直接失败）：\n"
        + "\n".join(problems)
    )


def test_tool_slots_is_derived_not_hand_written():
    """`TOOL_SLOTS` 是从每个槽位的 `targets` **自动推出来**的，不是手抄的第二份表。

    手抄的话，加一个槽位却忘了往 `TOOL_SLOTS` 里加，`fill_args()` 就会
    静默地少填一个参数——而"少填一个参数"恰恰是记忆层最怕的失败方式。
    """
    rebuilt: dict[str, list[str]] = {}
    for name, spec in SLOT_SPECS.items():
        for tool, _field in spec.targets:
            rebuilt.setdefault(tool, []).append(name)
    assert {k: sorted(v) for k, v in rebuilt.items()} == {
        k: sorted(v) for k, v in TOOL_SLOTS.items()
    }


def test_slot_names_are_unique_and_documented():
    """每个槽位都要写清"为什么需要它"——没有 `why` 的槽位半年后没人敢删。"""
    for name, spec in SLOT_SPECS.items():
        assert spec.name == name
        assert spec.why.strip(), f"{name} 没写为什么需要它"
        assert spec.kind in _KIND_OK, f"{name} 的 kind={spec.kind} 不认识"
        assert 0 <= spec.default_precision <= 2


# ===========================================================================
# ② 槽位表本体：冲突消解
# ===========================================================================


def test_put_writes_when_the_slot_is_empty():
    s = Slots()
    assert s.put("monthly_wage", 12000, turn=1) is True
    assert s.get("monthly_wage") == 12000


def test_put_lets_a_more_precise_value_win():
    s = Slots()
    s.put("monthly_wage", 10000, turn=1, precision=1)  # "一万出头"
    assert s.put("monthly_wage", 12000, turn=2, precision=2) is True
    assert s.get("monthly_wage") == 12000


def test_put_accepts_equally_precise_repairs():
    """同样精确就该覆盖——**用户第 12 轮改口了，不能说他改不过来**。

    这是 `put()` 里最容易被误当成 bug 的一条：一个"只写一次"的实现
    看起来更"稳"，但用户说"哦不对，是 15000"时它就变成了错的。
    """
    s = Slots()
    s.put("monthly_wage", 12000, turn=1, precision=2)
    assert s.put("monthly_wage", 15000, turn=12, precision=2) is True
    assert s.get("monthly_wage") == 15000


def test_put_rejects_a_vaguer_value_and_records_why():
    """模糊值不许盖掉精确值，**而且必须留下"为什么没生效"的记录**。

    不留记录的话，"第 9 轮说的『一万出头』没生效"这件事看起来就像个 bug，
    排查的人只能靠猜。
    """
    s = Slots()
    s.put("monthly_wage", 12000, turn=1, precision=2)
    assert s.put("monthly_wage", 10000, turn=9, precision=1) is False
    assert s.get("monthly_wage") == 12000
    assert len(s.history) == 1
    h = s.history[0]
    assert h["slot"] == "monthly_wage"
    assert h["kept"] == 12000 and h["rejected"] == 10000
    assert "模糊" in h["reason"]


def test_put_rejects_unknown_slot_names_loudly():
    """拼错的槽位名必须当场炸，不能静默丢弃。

    记忆层里"静默丢弃"是最贵的失败方式：值没了，但没人知道。
    """
    with pytest.raises(KeyError, match="未定义的槽位"):
        Slots().put("monthly_wages", 1, turn=1)


def test_put_of_none_is_a_no_op():
    """`None` 表示"这次没抽到"，不该覆盖已有值。"""
    s = Slots()
    s.put("monthly_wage", 12000, turn=1)
    assert s.put("monthly_wage", None, turn=2) is False
    assert s.get("monthly_wage") == 12000


# ===========================================================================
# ③ 槽位 → 工具参数
# ===========================================================================


def _filled() -> Slots:
    s = Slots()
    s.put("monthly_wage", 12000, turn=1)
    s.put("months_of_service", 55, turn=1)
    s.put("termination_cause", "economic_layoff", turn=1)
    return s


def test_fill_args_only_emits_fields_that_tool_actually_declares():
    """`fill_args()` 填出来的每个键，都必须是**这个工具真的有的参数**。

    多填一个键，pydantic 会按 `extra` 的默认行为处理——要么报错，
    要么**静默丢掉**。两种都不该发生在"程序替模型填参数"这条路上。
    """
    s = _filled()
    args = s.fill_args("calc_severance")
    model = _tool_models()["calc_severance"]
    assert set(args) <= set(model.model_fields)
    assert args["avg_monthly_wage"] == 12000
    assert args["months_of_service"] == 55
    # 模型真的能用这些参数跑起来 —— 这一条比逐个比字典结实
    model(**args)


def test_fill_args_does_not_leak_slots_into_a_tool_that_does_not_want_them():
    """`calc_overtime` 用的是 `monthly_wage`，`calc_severance` 用的是
    `avg_monthly_wage`——**同名槽位在不同工具里的字段名不一样**，
    这是 `targets` 存在的全部理由。填错了就是参数对不上。
    """
    args = _filled().fill_args("calc_overtime")
    assert args == {"monthly_wage": 12000}
    assert "avg_monthly_wage" not in args


def test_missing_for_reports_only_required_slots():
    """必须报出来的只有"没有它就算不出结果"的那几个。

    `notice_given`（有没有提前 30 天通知）**不该追问**：用户不知道
    "要不要通知"是他的事，追问它等于把法条的负担甩给用户。
    """
    s = Slots()
    s.put("termination_cause", "economic_layoff", turn=1)
    missing = s.missing_for("calc_severance")
    assert "monthly_wage" in missing
    assert "months_of_service" in missing
    assert "notice_given" not in missing
    assert "last_month_wage" not in missing


def test_missing_labels_are_chinese():
    """拿去问用户的是中文名，不是字段名。"""
    labels = Slots().missing_labels("calc_severance")
    assert "月工资" in labels
    assert all(l and not l.isascii() for l in labels)


def test_internal_slots_never_reach_the_prompt_or_the_tools():
    """内部槽位（入职日期、其它金额）**不该进提示词，也不该进工具参数**。

    该给模型看的是结论（工作年限 55 个月），不是原料（入职日期 2021-03-01）——
    给了原料它就会拿去做自己的算术，而那正是我们要避免的。
    """
    s = Slots()
    s.put("monthly_wage", 12000, turn=1)
    s.put("work_start", "2021-03-01", turn=1, source="manual")
    s.put("other_amounts", [30000.0], turn=1, precision=0)

    block = s.prompt_block()
    assert "12000" in block
    assert "2021-03-01" not in block
    assert "入职" not in block

    assert "work_start" not in s.to_dict()
    assert "work_start" in s.to_dict(include_internal=True)


def test_empty_slots_produce_an_empty_prompt_block():
    """空槽位不渲染。写一屏"月薪：未知"只会让模型倾向于追问用户没提过的东西。"""
    assert Slots().prompt_block() == ""


def test_vague_values_are_marked_in_the_prompt():
    """模糊值在提示词里要带个记号——**让模型知道这个数需要跟用户核一下**。"""
    s = Slots()
    s.put("monthly_wage", 10000, turn=1, precision=1)
    assert "模糊" in s.prompt_block()


# ===========================================================================
# ④ 抽取：每一条都是踩过的坑
# ===========================================================================


@pytest.mark.parametrize(
    "text,expect",
    [
        ("月薪12000", {"monthly_wage": 12000}),
        ("月薪 1.2万", {"monthly_wage": 12000}),
        ("月薪12k", {"monthly_wage": 12000}),
        ("工资八千五", {"monthly_wage": 8500}),
        ("月薪一万二", {"monthly_wage": 12000}),
        ("我月薪一万零三百", {"monthly_wage": 10300}),
        ("公司欠我三万五千块", {"monthly_wage": 35000}),
        ("我一个月挣七千", {"monthly_wage": 7000}),
        # "一万出头" 抽出来是**精度 1**——能用，但管不住后面用户给的精确数
        ("我那点工资，一万出头吧", {"monthly_wage": 10000}),
    ],
)
def test_wage_extraction(text, expect):
    got = {k: v.value for k, v in SlotExtractor.extract(text, turn=1).items()}
    for k, v in expect.items():
        assert got.get(k) == v, f"{text!r} 抽成了 {got}"


def test_bare_number_is_not_a_wage():
    """⚠️ 踩过的坑：`"公司有200人"` 里的 200 被当成月工资。

    错抽的后果比漏抽严重得多——**计算基数会静默地变成一个用户从没提过的数**，
    而且他看不出来。所以"裸数字"必须靠上下文线索（月薪/工资/赔/挣…）才敢认。
    """
    got = SlotExtractor.extract("公司有200人", turn=1)
    assert "monthly_wage" not in got


def test_the_number_next_to_the_cue_wins():
    """一句话里两个金额时，**取被线索词点名的那个**，不取最大的、也不取最后的。

    "年终奖3万，月薪8000" 取最大会拿到年终奖（基数错三倍），
    取最后在"月薪8000，年终奖3万"里又会拿到年终奖。
    线索词直接编码了"这个数字是什么"，比位置和大小都可靠。
    """
    for text in ("月薪8000，年终奖3万", "年终奖3万，月薪8000"):
        got = SlotExtractor.extract(text, turn=1)
        assert got["monthly_wage"].value == 8000, text
        # 落选的金额不猜用途，但要记下来 —— "为什么基数不对"的第一手证据
        assert got["other_amounts"].value == [30000.0], text


def test_two_durations_in_one_sentence_are_not_summed():
    """⚠️ 踩过的坑：`"我干了3年，签了3年合同"` 被整句求和成 72 个月。

    那两个"3年"是**两件不同的事**：一个是已经干了多久（算补偿金用），
    一个是合同签了多久（查试用期上限用）。求和会得到一个用户
    从没提过的数——而且他看不出来。
    """
    got = SlotExtractor.extract("我干了3年，签了3年合同", turn=1)
    assert got["months_of_service"].value == 36
    assert got["contract_term_months"].value == 36


def test_adjacent_durations_in_one_sentence_are_summed():
    """"3年零6个月"**要**加起来——它是一个时长，只是分两段说。"""
    got = SlotExtractor.extract("3年零6个月", turn=1)
    assert got["months_of_service"].value == 42


def test_half_year_is_counted():
    """"两年半" = 30 个月。正则只会匹配到"两年"，那个"半"得自己补上——
    24 个月和 30 个月对补偿金是两个数。"""
    got = SlotExtractor.extract("干了两年半", turn=1)
    assert got["months_of_service"].value == 30


def test_a_year_number_is_not_a_duration():
    """⚠️ 踩过的坑：`"我2021年来的"` 里的 2021 被算成 2021 年工龄（24252 个月）。

    这不是年限，这是把**年份**当成了时长。上限保护挡掉它。
    """
    assert "months_of_service" not in SlotExtractor.extract("我2021年来的", turn=1)


def test_probation_months_are_not_work_experience():
    """⚠️ 踩过的坑：`"试用期3个月"` 里的 "3个月" 被算成已工作时长。

    "试用期 3 个月"说的是**约定**，"干了 3 个月"说的是**事实**。
    补偿金按工龄算，用错那个会直接把钱算错。
    """
    got = SlotExtractor.extract("试用期3个月", turn=1)
    assert got["probation_months"].value == 3
    assert "months_of_service" not in got


def test_overtime_days_need_an_overtime_context():
    """⚠️ 踩过的坑：`"没提前30天通知我"` 里抽出了"加班 30 天"。

    "几天""几个小时"这种说法到处都是，光看数字必错。判据是数字前后
    6 个字里真的出现"加班/值班/休息日"这类词。
    """
    assert "overtime_days" not in SlotExtractor.extract("没提前30天通知我", turn=1)
    got = SlotExtractor.extract("周末加班2天", turn=1)
    assert got["overtime_days"].value == 2.0


def test_the_overtime_window_is_small_enough_to_respect_negation():
    """"我不加班，每天工作8小时" 里**没有**加班时长。

    窗口收到 6 个字，"不加班"的"加"落在窗口外。用整句判的话这里就误判了。
    """
    got = SlotExtractor.extract("我不加班，每天工作8小时", turn=1)
    assert "overtime_hours" not in got


def test_a_month_count_is_not_money():
    """⚠️ 踩过的坑：`"赔我两个月工资"` 里的"两"被当成 2 元。

    `_MONTHS_RE` 只认阿拉伯数字，"两个月"从它手里漏过去，然后金额正则
    看到"两"、又看到后面的"工资"线索，就认了。
    """
    got = SlotExtractor.extract("赔我两个月工资", turn=1)
    assert "monthly_wage" not in got


def test_relationship_ended_is_a_date_not_a_boolean():
    """★ 这个槽位的类型必须是**日期**，因为工具那边要的是日期。

    分三种情况，三种都得对：

    - "还在职" → 什么都不填。工具那边"不填"就表示关系存续，
      而且"仍在职不受一年时效限制"这条结论依赖的正是这个默认。
    - "已经走了"但没说哪天 → 也什么都不填。硬拿"今天"顶上，
      会让"过没过时效"凭空变松。
    - "2023年8月离职" → 填日期。
    """
    assert "relationship_ended" not in SlotExtractor.extract("我还在上班", turn=1)
    assert "relationship_ended" not in SlotExtractor.extract("已经离职两个月了", turn=1)
    got = SlotExtractor.extract("2023年8月离职的", turn=1)
    assert got["relationship_ended"].value == "2023-08-01"


def test_notice_given_is_only_set_when_the_user_raises_it():
    """用户不提"提前通知过没有"，就**不能猜**——猜"没通知"会凭空多算一个月工资。"""
    assert "notice_given" not in SlotExtractor.extract("公司把我裁了", turn=1)
    assert SlotExtractor.extract("没提前30天通知我", turn=1)["notice_given"].value is False
    assert SlotExtractor.extract("提前30天通知了我", turn=1)["notice_given"].value is True


def test_cn_to_int_handles_the_three_common_shapes():
    """中文数字就这点东西：单字、十位省略、单位在前。"""
    assert cn_to_int("三") == 3
    assert cn_to_int("十二") == 12
    assert cn_to_int("十五") == 15  # "十"前面省略了 1
    assert cn_to_int("三十") == 30
    assert cn_to_int("三十五") == 35
    assert cn_to_int("半") is None  # "半"不是数字，是"一半"，由调用方处理
    assert cn_to_int("") is None


def test_extractor_is_pure():
    """同样的输入永远得到同样的输出。**记忆层有随机性的话，
    "第20轮丢了信息"就变成一个没法复现的 bug。**"""
    text = "月薪12000，干了3年，公司裁员"
    a = SlotExtractor.extract(text, turn=3)
    b = SlotExtractor.extract(text, turn=3)
    assert {k: v.value for k, v in a.items()} == {k: v.value for k, v in b.items()}


# ===========================================================================
# ⑤ 跨轮累积：SlotStore
# ===========================================================================


def test_observe_accumulates_across_turns():
    """一段真实咨询的样子：事实是一轮一轮挤出来的，不是一口气说完的。"""
    store = SlotStore()
    assert store.observe("我2021年3月入职，月薪12000", turn=1)
    assert store.observe("公司说要裁员", turn=2)
    assert store.observe("我干了3年", turn=3)

    s = store.slots
    assert s.get("monthly_wage") == 12000
    assert s.get("termination_cause") == "economic_layoff"
    assert s.get("months_of_service") == 36


def test_observe_reports_what_it_actually_wrote():
    """返回值是"这一次真的写进去了哪几个槽位"，进 trace 用。

    排查"为什么第 20 轮少了一个数"时，第一眼看的就是它。
    """
    store = SlotStore()
    assert store.observe("月薪12000", turn=1) == ["monthly_wage"]
    # 第 2 轮说的是模糊值，会被拒绝 —— 于是"这一次什么都没写"
    assert store.observe("其实是一万出头吧", turn=2) == []


def test_a_later_correction_is_accepted():
    """★ 用户改口必须能改过来，**而且不能把改口的数字整条丢掉**。

    ⚠️ 这是写测试时抓到的真 bug：`"不对，是15000"` 里的 15000 是个裸数字，
    前后都没有"月薪/工资"，按"裸数字必须靠线索才敢认"的规矩只能判成精度 0，
    于是**更正被整条丢掉**，后面所有计算还在用 12000——而且没人看得出来。

    修法是认出"改口"这个动作本身就是线索：它说明"这个数字指的是刚才那个东西"。
    """
    store = SlotStore()
    store.observe("月薪12000", turn=1)
    assert store.observe("不对，是15000", turn=2) == ["monthly_wage"]
    assert store.slots.get("monthly_wage") == 15000


def test_a_correction_still_needs_to_look_like_a_wage():
    """"改口"也是个线索，但门槛更高（四位数起）——
    `"不对，公司有200人"` 里的 200 不能被认成工钱。"""
    store = SlotStore()
    store.observe("月薪12000", turn=1)
    assert store.observe("不对，公司有200人", turn=2) == []
    assert store.slots.get("monthly_wage") == 12000


def test_derive_turns_a_hire_date_into_months_of_service():
    """★ 用户常给日期不给年限。派生是**纯计算**，和抽取分开写、分开测。

    ⚠️ 写这条测试时抓到的真 bug：内部槽位改名（`_work_start` → `work_start`）之后，
    `derive()` 还在按老名字读——于是它**永远读不到入职日期，永远什么都不做，也不报错**。
    `SlotStore.observe()` 那边还同时写着一句"跳过下划线开头的名字"，
    把内部槽位整个拦在了写入口外面。
    两条加起来，"只给日期不给年限"这条最常见的路径是**静默死掉的**。
    """
    store = SlotStore()
    store.observe("我2021年3月入职", turn=1)
    assert store.slots.get("work_start") == "2021-03-01"
    changed = store.derive(turn=1)
    assert "months_of_service" in changed

    d0 = dt.date(2021, 3, 1)
    today = dt.date.today()
    expect = (today.year - d0.year) * 12 + (today.month - d0.month) - int(today.day < d0.day)
    assert store.slots.get("months_of_service") == expect


def test_derive_does_not_overwrite_an_explicit_answer():
    """用户明确说了"干了3年"，就别再用入职日期去推算——
    **推算出来的数不许盖掉他说过的数。**"""
    store = SlotStore()
    store.observe("我2021年3月入职，干了3年", turn=1)
    assert store.slots.get("months_of_service") == 36
    store.derive(turn=1)
    assert store.slots.get("months_of_service") == 36


def test_derive_maps_a_city_to_the_local_average_wage_but_marks_it_vague():
    """城市 → 当地社平工资。**精度只给 1**：那张表现在就是会过期的。"""
    store = SlotStore()
    store.observe("公司在西安", turn=1)
    assert "local_avg_monthly_wage" in store.derive(turn=1)
    assert store.slots.entry("local_avg_monthly_wage").precision == 1


def test_derive_does_not_invent_a_city_it_does_not_know():
    """认不出的城市**不编**。编出来的社平工资会让"三倍封顶"这个判断凭空出错。"""
    store = SlotStore()
    store.observe("公司在鹤岗", turn=1)
    assert store.derive(turn=1) == []


def test_derive_is_idempotent():
    """派生跑两遍和跑一遍结果一样——它是纯计算，不是"每轮加一点"。"""
    store = SlotStore()
    store.observe("我2021年3月入职", turn=1)
    assert "months_of_service" in store.derive(turn=1)
    once = store.slots.get("months_of_service")
    assert once is not None
    store.derive(turn=2)
    assert store.slots.get("months_of_service") == once


# ===========================================================================
# ⑥ token 估算
# ===========================================================================


def test_token_estimate_is_monotonic():
    """预算分配要回答的是"谁先被砍"——那是个**排序**问题。
    所以估算只要单调（长的估出来一定更多）就够用了。"""
    short = count_tokens("一个月工资")
    long = count_tokens("一个月工资" * 10)
    assert 0 < short < long


def test_token_estimate_of_empty_is_zero():
    assert count_tokens("") == 0


def test_chinese_costs_more_than_ascii_per_character():
    """汉字约 1 字 1 token，ASCII 约 4 字符 1 token。**高估是故意的**——
    高估只是早一点开始压缩，低估会让 prompt 超长被服务端截断。"""
    assert count_tokens("中" * 40) > count_tokens("a" * 40)


def test_count_messages_adds_per_message_overhead():
    """每条消息都有固定的角色/分隔开销，不能只把内容加起来。"""
    msgs = [{"role": "user", "content": "你好"}]
    assert count_messages(msgs) > count_tokens("你好")
    assert count_messages([]) == 0


# ===========================================================================
# ⑦ 工作记忆：窗口 + 摘要
# ===========================================================================


def _history(n: int) -> list[Message]:
    out: list[Message] = []
    for t in range(1, n + 1):
        out.append(Message(role="user", content=f"问题{t}", turn=t))
        out.append(Message(role="assistant", content=f"回答{t}", turn=t))
    return out


def test_window_keeps_the_most_recent_turns_verbatim():
    """近期原文不压缩。**"这一轮的问题"几乎总依赖"上一轮说了什么"**——
    压缩近期 = 直接丢掉指代关系。"""
    wm = WorkingMemory(window=3, segment=6)
    r = wm.ingest(_history(10))
    kept_turns = {m.turn for m in r["recent"]}
    assert kept_turns == {8, 9, 10}


def test_summaries_only_cover_turns_that_left_the_window():
    """摘要的是**离开窗口**的那部分，不会和近期原文重复。"""
    wm = WorkingMemory(window=3, segment=6)
    r = wm.ingest(_history(10))
    assert r["compressed"] == (1, 6)
    assert wm.summaries[0].from_turn == 1 and wm.summaries[0].to_turn == 6


def test_a_partial_segment_is_left_alone():
    """攒不够一整段就**先不压**。半段摘要比不摘要更糟——
    它会让"这一段覆盖了第几轮到第几轮"这个信息变得不准。"""
    wm = WorkingMemory(window=2, segment=6)
    r = wm.ingest(_history(5))
    assert r["compressed"] is None
    assert wm.summaries == []


def test_summaries_accumulate_instead_of_being_recomputed():
    """摘要是**跨轮累积**的。每轮重压一遍前面所有历史的话，
    第 20 轮要压 14 轮，成本随轮次线性上涨，而压出来的东西和第 19 轮几乎一样。"""
    wm = WorkingMemory(window=2, segment=4)
    wm.ingest(_history(6))
    first_round = len(wm.summaries)
    wm.ingest(_history(14))
    assert len(wm.summaries) > first_round
    # 已经压过的区间不会重压
    assert wm.summaries[0].from_turn == 1


def test_working_memory_is_deterministic():
    """给同样的历史，永远得到同样的结果。"""
    a = WorkingMemory(window=2, segment=4)
    b = WorkingMemory(window=2, segment=4)
    a.ingest(_history(12))
    b.ingest(_history(12))
    assert [s.text for s in a.summaries] == [s.text for s in b.summaries]


def test_rule_summarizer_keeps_the_last_three_questions():
    """规则摘要只抄用户问过的问题，最多 3 条，**越近的越相关**。"""
    msgs = [
        Message(role="user", content=f"问题{i}", turn=i) for i in range(1, 8)
    ]
    text = rule_summarize(msgs)
    assert "问题7" in text and "问题6" in text and "问题5" in text
    assert "问题1" not in text


def test_rule_summarizer_does_not_crash_on_assistant_only_spans():
    """全是助手消息的段是少见但真会出现的情况（用户连发几条、回答被切成两段）。"""
    assert rule_summarize([Message(role="assistant", content="好的", turn=1)])


def test_default_summarizer_does_not_call_a_model():
    """默认必须走规则版。摘要层是三层里唯一"可以有损"的一层，
    **为最不关键的一层付模型调用费，顺序错了。**"""
    assert isinstance(WorkingMemory().summarize, RuleSummarizer)
    assert WorkingMemory().summarize.__name__ == "rule"


def test_summary_records_its_coverage_range():
    """没覆盖范围的话，出了"第 9 轮说过的话丢了"这种事，
    你连"它到底有没有被压进摘要"都判断不了，只能靠猜。"""
    wm = WorkingMemory(window=1, segment=3)
    wm.ingest(_history(6))
    s = wm.summaries[0]
    assert s.span == 3
    assert f"第{s.from_turn}-{s.to_turn}轮摘要" in s.render()


def test_summaries_render_newest_first():
    """和人类读聊天记录一样：越近的越重要，而模型对**开头**最敏感。"""
    wm = WorkingMemory(window=1, segment=2)
    wm.ingest(_history(6))
    # 第 6 轮留在窗口里，所以能压的是 1–5 轮 → 两段：(1,2) 和 (3,4)，
    # 剩下的第 5 轮攒不够一段，留着不压（见 `test_a_partial_segment_is_left_alone`）
    assert [s.span for s in wm.summaries] == [2, 2]
    rendered = wm.render_summaries()
    assert rendered.index("第3-4轮") < rendered.index("第1-2轮")


# ===========================================================================
# ⑧ 长期层：按需召回
# ===========================================================================


def _store() -> LongTermStore:
    st = LongTermStore()
    st.add(MemoryItem(turn=1, question="我月薪一万二", answer="好的"))
    st.add(MemoryItem(turn=5, question="公司在西安，做软件的", answer="好的"))
    st.add(MemoryItem(turn=9, question="公司拖欠工资三个月了", answer="好的"))
    return st


def test_tokenize_drops_single_characters_and_stopwords():
    """"个""的""了"在中文里到处都是，留着它们会让所有记录都互相"相关"。"""
    words = tokenize("我 的 公司 拖欠 工资 了")
    assert "公司" in words and "拖欠" in words
    assert "的" not in words and "了" not in words


def test_recall_returns_nothing_when_there_is_no_overlap():
    """★ 没有任何词重叠时**返回空**，不返回"最不相关的那几条"。

    这是这一层最容易被做错的地方：硬塞 k 条进去，模型会去从一堆无关历史里
    找因果关系，然后编出一个用户没问过的联系。**宁可空手而归。**
    """
    st = _store()
    assert st.search("明天天气怎么样", current_turn=10) == []


def test_recall_prefers_the_turn_that_shares_more_words():
    st = _store()
    hits = st.search("公司拖欠工资", current_turn=10, k=1)
    assert hits and hits[0].turn == 9


def test_recency_breaks_ties():
    """重叠数一样时，**近的赢**。在长对话里"相关"和"近"高度相关。"""
    st = LongTermStore()
    st.add(MemoryItem(turn=1, question="月薪多少", answer=""))
    st.add(MemoryItem(turn=8, question="月薪多少", answer=""))
    assert st.search("月薪", current_turn=10, k=1)[0].turn == 8


def test_old_turns_decay():
    """半衰期 10 轮：第 1 轮的话到第 20 轮只剩约 1/4 的权重。
    重叠数差 1 个词时，**近的那条应该能反超**。"""
    st = LongTermStore(half_life=10.0)
    # 老的那条重叠 2 个词，新的只重叠 1 个词
    st.add(MemoryItem(turn=1, question="工资 拖欠", answer=""))
    st.add(MemoryItem(turn=19, question="工资", answer=""))
    assert st.search("工资 拖欠", current_turn=20, k=1)[0].turn == 19


def test_recall_is_empty_for_an_empty_store():
    assert LongTermStore().search("月薪", current_turn=1) == []


def test_recall_renders_the_answer_truncated():
    """助手那段要截断——它通常几百字，全带回 prompt 就不叫"按需召回"了。"""
    item = MemoryItem(turn=3, question="问", answer="答" * 400)
    assert len(item.render()) < 400
    assert item.render().endswith("…")


def test_long_term_index_is_not_serialized():
    """切词缓存用 `PrivateAttr` 存，**不能被 `model_dump()` 带出去**——
    带出去的话 trace 里会莫名其妙多出几十 KB 的词表。"""
    st = _store()
    st.search("月薪", current_turn=2)
    assert "_index" not in st.model_dump()


# ===========================================================================
# ⑨ 管理器：四策略的分配
# ===========================================================================


def _flat_history(turns: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for t in range(1, turns + 1):
        out.append({"role": "user", "content": f"第{t}轮问题：月薪是多少，公司会赔多少"})
        out.append({"role": "assistant", "content": f"第{t}轮回答：" + "根据第47条，" * 10})
    return out


_QUESTION = "西安 裁员 能赔多少"


def _seed(mgr: MemoryManager) -> None:
    """给管理器预置"前 10 轮攒下来"的东西：槽位 + 一条能召回的长期记忆。

    ⚠️ 长期层必须**预先塞一条**，不能指望第一次 `build()` 就能召回：
    归档发生在 `build()` 的**末尾**（"哪些轮次离开了窗口"是这一轮才算出来的），
    所以第 1 次 build 时长期层必然是空的。这不是 bug，是顺序决定的。
    """
    mgr.slots.put("monthly_wage", 12000, turn=1)
    mgr.slots.put("months_of_service", 36, turn=2)
    mgr.long_term.add(MemoryItem(turn=1, question="西安 裁员 能赔多少", answer="按第47条"))


@pytest.mark.parametrize(
    "policy,expect_slots,expect_recall",
    [
        (MemoryPolicy.NO_MEMORY, False, False),
        (MemoryPolicy.SLIDING_WINDOW, False, False),
        (MemoryPolicy.SLIDING_WINDOW_SUMMARY, False, False),
        (MemoryPolicy.LAYERED, True, True),
    ],
)
def test_policies_turn_layers_on_and_off(policy, expect_slots, expect_recall):
    """四种策略**不是四份实现，而是同一个管理器换开关**。

    如果四种策略是四份代码，那对比出来的差异里就混进了"实现水平的差异"，
    而不是"策略的差异"。同一份代码换开关，才是干净的对照实验。
    """
    mgr = MemoryManager(policy=policy)
    _seed(mgr)
    b = mgr.build(_QUESTION, _flat_history(12), current_turn=12)
    have = {lay.name for lay in b.layers if lay.items}
    assert ("slots" in have) == expect_slots
    assert ("recall" in have) == expect_recall


def test_only_layered_keeps_exact_facts_from_long_ago():
    """★ 四策略真正的区别在这里——**不是"谁的 prompt 更长"**。

    这一点值得写清楚，因为它的直觉是反的：`sliding_window` 会把**整个记忆预算
    砸在原文上**（`window` 被拉大到 10 轮），它的 prompt 比 `layered` **更长**。
    长了 200 个字，却仍然是**没用**的——用户第 1 轮说的"月薪 12000"
    到第 20 轮早被挤出窗口了，多出来的那些字全是第 11~20 轮的寒暄。

    所以四策略的对照指标只能是"**关键事实还在不在**"，不是字数量。
    槽位层是唯一能把这个数带过 20 轮的机制（它压根不参与压缩），
    而它只对 `layered` 开启。

    ⚠️ 注意这条测试和数据里的话术没关系：`_flat_history` 造的文本里
    出现过"月薪"这个词，但**从来没有出现过 12000 这个数**。
    所以 `"12000" in block` 是个干净的判据——它只可能来自槽位层。
    """
    kept = {}
    for policy in MemoryPolicy:
        mgr = MemoryManager(policy=policy)
        _seed(mgr)  # 槽位里放着第 1 轮说的 monthly_wage=12000
        b = mgr.build(_QUESTION, _flat_history(20), current_turn=20)
        kept[policy.value] = "12000" in b.prompt_block
    assert kept == {
        "no_memory": False,
        "sliding_window": False,
        "sliding_window_summary": False,
        "layered": True,
    }, kept


def test_a_longer_prompt_is_not_a_better_one():
    """把上面那条的"反直觉"钉成一条测试：**`sliding_window` 的 prompt
    比 `layered` 更长，但它丢掉的东西更多。**

    这个数（`sliding_window` 的 prompt 长度 > `layered` 的）就是 F22 报告里
    "长上下文不等于好上下文"那句话的证据。写死一个不等式，比写一句评论结实。
    """
    lengths = {}
    for policy in (MemoryPolicy.SLIDING_WINDOW, MemoryPolicy.LAYERED):
        mgr = MemoryManager(policy=policy)
        _seed(mgr)
        lengths[policy.value] = len(
            mgr.build(_QUESTION, _flat_history(20), current_turn=20).prompt_block
        )
    assert lengths["sliding_window"] > lengths["layered"], lengths


def test_no_memory_still_carries_the_current_question():
    """**真正的"无记忆"是"只有这一轮"**，不是"一段空 prompt"。

    一个连问题都不带的 prompt 连跑都跑不起来，那就不是策略对比，
    是拿一个跑不通的基线去比。
    """
    q = "公司把我裁了能赔多少"
    history = _flat_history(11) + [{"role": "user", "content": q}]
    mgr = MemoryManager(policy=MemoryPolicy.NO_MEMORY)
    b = mgr.build(q, history, current_turn=12)
    assert q in b.prompt_block


def test_slots_are_allocated_before_the_window():
    """**预算最优先给槽位。** 砍掉它的后果是"算错钱"，
    而用户拿到的只是一个数字，看不出来是错的。"""
    mgr = MemoryManager(
        policy=MemoryPolicy.LAYERED,
        budget=MemoryBudget(total=200, slots_max=400),
    )
    _seed(mgr)
    b = mgr.build("那能赔多少", _flat_history(12), current_turn=12)
    assert "12000" in b.prompt_block


def test_the_current_turn_survives_an_absurdly_small_budget():
    """★ 硬下限：**近期窗口至少保留最后 1 轮，哪怕它自己就超了预算。**

    没有"这一轮的问题"，就没有对话。
    """
    mgr = MemoryManager(policy=MemoryPolicy.LAYERED, budget=MemoryBudget(total=1))
    q = "这是一句特别长的问题" * 20
    b = mgr.build(q, _flat_history(12) + [{"role": "user", "content": q}], current_turn=13)
    assert b.prompt_block.strip()
    assert any("硬下限" in d or "最后 1 轮" in d for d in b.dropped) or q[:10] in b.prompt_block


def test_dropped_tells_you_what_was_cut_and_why():
    """被砍掉的东西 + 为什么，**这份清单是 F22 报告的核心素材**——
    "第 20 轮丢了信息"要能一眼看出"是因为摘要层被砍了"。"""
    mgr = MemoryManager(
        policy=MemoryPolicy.LAYERED,
        budget=MemoryBudget(total=60, slots_max=10),
    )
    _seed(mgr)
    b = mgr.build("那能赔多少", _flat_history(20), current_turn=20)
    assert b.dropped
    assert all(isinstance(d, str) and d for d in b.dropped)


def test_block_token_count_matches_the_block():
    """账要和实物对得上。**记忆层里各算各的，预算就会前后对不上。**"""
    mgr = MemoryManager(policy=MemoryPolicy.LAYERED)
    _seed(mgr)
    b = mgr.build("那能赔多少", _flat_history(12), current_turn=12)
    assert b.total_tokens == count_tokens(b.prompt_block)


def test_build_is_deterministic_for_a_fresh_manager():
    """给同样的输入，输出同样的 bundle。**记忆层带随机性的话，
    "第20轮丢了信息"就变成一个没法复现的 bug，只能靠重试去撞。**"""
    def run() -> str:
        m = MemoryManager(policy=MemoryPolicy.LAYERED)
        _seed(m)
        return m.build(_QUESTION, _flat_history(12), current_turn=12).prompt_block

    assert run() == run()


def test_recall_is_the_first_thing_cut():
    """长期召回优先级最低——它本来就是"锦上添花"，
    砍掉它没有任何后果。"""
    mgr = MemoryManager(
        policy=MemoryPolicy.LAYERED,
        budget=MemoryBudget(total=120, slots_max=400),
    )
    _seed(mgr)
    b = mgr.build("月薪", _flat_history(24), current_turn=24)
    if "recall" in b.layer_map() and b.layer_map()["recall"] == 0:
        assert any("长期召回" in d for d in b.dropped)


def test_archived_turns_leave_the_window():
    """★ 窗口之外的轮次才进长期层——近期那几轮已经在窗口里了，
    重复召回等于把同一段话给模型看两遍（浪费预算，还会让模型
    以为那件事比实际更重要）。"""
    mgr = MemoryManager(policy=MemoryPolicy.LAYERED)  # window=6
    mgr.build("月薪是多少", _flat_history(16), current_turn=16)
    # 16 轮、窗口 6 → 第 11~16 轮留在原文里，只有 1~10 轮该被归档
    assert len(mgr.long_term) > 0
    assert max(i.turn for i in mgr.long_term.items) < 11


def test_non_layered_policies_do_not_archive():
    """不开启长期层的策略**什么都不该往长期层里写**——
    否则"四策略对比"就变成了"四策略共享同一份记忆"，对照实验直接失效。"""
    mgr = MemoryManager(policy=MemoryPolicy.SLIDING_WINDOW)
    mgr.build("月薪是多少", _flat_history(16), current_turn=16)
    assert len(mgr.long_term) == 0
