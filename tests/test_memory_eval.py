"""记忆评测工具的测试（F22）。

## 为什么评测工具自己也需要测试

一个评测脚本错了，**不会报错，只会得出一个错的结论**。
而且它比业务代码的错误更危险：业务代码错了你当场就发现，
评测错了你会拿着那个结论去写文档、去讲给面试官听、
去指挥下一步做什么。**错误会沿着结论往下游传播，而且一路上没有任何红灯。**

这个文件里每一条测试，都对应一个**真的发生过**的错：

1. `test_the_pattern_matches_whole_numbers_only` —— 忘了 `re.escape`，
   骨架里的 `|` 变成正则的"或"，第一个分支是空串，
   于是**它在任何字符串里都命中**。报表里所有 `text_visible` 全是 `True`，
   `no_memory` 的保持率看起来比滑动窗口还高。一个多余的 `|` 把整张表变成噪声。
2. `test_quotable_is_decided_by_the_program_not_by_hand` —— "隔得近"不等于
   "原文里有"。`工作年限 37` 是"三年零一个月"算出来的，原文里没有 37；
   把它和 `月工资 12000` 混在一起比，会得出"插了槽位层就赢麻了"的假结论。
3. `test_a_derived_value_is_not_quotable` / `test_a_date_in_another_format_is_quotable`
   —— 分类器自己的两条边界。
4. `test_the_audit_catches_a_wrong_expectation` —— 体检能不能真的抓错。
   一条**永远抓不到错的体检**是最糟的：它给你一种"数据是干净的"的错觉。
5. `test_a_probe_whose_fact_is_in_the_probe_turn_is_rejected` —— 探针测的
   必须是真的记忆，不是"刚刚那句话还在窗口里"。

## 这一组不跑模型，也不跑全量数据

全量数据在 `data/eval/memory_sessions.yaml`，跑一遍要几秒。
这里用**手搓的小对话**，每条测试只关心一件事。
数据本身由 `audit_dataset` 在 `scripts/memory_ablation.py` 里把关。
"""

from __future__ import annotations

import pytest

from juris_agent.eval.memory_eval import (
    Probe,
    Session,
    audit_dataset,
    classify_quotable,
    expected_fragment_slot,
    load_sessions,
    load_pricing,
    normalize,
    render_report,
    run_policy,
)

# ⚠️ 这里**同时**用两种写法 import 同一个模块，是刻意的：
# `from ... import xxx` 拿公开函数（读起来干净），
# `import ... as M` 拿私有函数（`M._value_pattern`、`M._skeleton`、`M._plain`）。
# 为什么测试要碰私有函数：**下面这几条测的就是"匹配逻辑本身对不对"**——
# 那个逻辑没有公开出口，而它错了会让整份报表变成噪声。
# 要么给它开一个公开出口（多一层没有意义的包装），
# 要么在这里显式地测私有函数。**测试碰私有，比为了测试而改公开接口好。**
import juris_agent.eval.memory_eval as M
from juris_agent.memory.manager import MemoryPolicy

# ===========================================================================
# ① 值匹配：把"文本里有没有这个事实"变成一个可靠的判定
# ===========================================================================


def _found(value: str, text: str) -> bool:
    """测验值正则的探针。**测的是 `_value_pattern`，不是 `str in str`。**"""
    return M._value_pattern(value).search(M._skeleton(text)) is not None


def test_the_pattern_matches_whole_numbers_only():
    """★ 不加数字边界的话，`3` 会在 `3000` 里命中。

    这是本文件里最值钱的一条测试。判据不能是"子串在不在"——
    "加班 3 小时"和"月薪 3000"里的 `3` 是同一个字符串，但不是同一个数。
    """
    assert _found("3", "平时每天加班3小时")
    assert not _found("3", "月薪3000块")
    assert not _found("1200", "月薪12000块")  # `1200` 是 `12000` 的前缀
    assert _found("12000", "月薪12000块")


def test_the_pattern_does_not_match_everything():
    """★ 忘了 `re.escape` 的那次事故。

    骨架里内容字符带 `|` 前缀，`|` 在正则里是"或"。不转义的话，
    `False` 编出来的正则是 `(?<!\\d)|F|a|l|s|e(?!\\d)`——
    **第一个分支是空串，匹配任何位置**。

    这条测试用"一个绝不存在的值"来钉它：如果连它都能命中，说明正则是通配的。
    """
    assert not _found("False", "没提前通知我")
    assert not _found("True", "公司欠我工资")
    assert not _found("西安", "公司在郑州")
    assert not _found("ZZZ", "随便一句话")


def test_a_date_in_another_format_is_quotable():
    """同一个日期换个写法（`2024-05-10` vs `2024年5月10日`）要认得出。

    这是"骨架"这层设计的全部理由：用户和程序写日期的方式天然不同，
    判定不该被这种差别影响。
    """
    assert _found("2024-05-10", "我2024年5月10日被辞退的")
    assert _found("2024-05-10", "2024/05/10 离职")
    assert _found("2024-05-10", "离职日 2024-5-10")
    assert _found("2024-05-10", "槽位里写着 - 劳动关系终止之日：2024-05-10")


def test_float_and_int_are_the_same_value():
    """槽位里存的是 `3.0`，用户写的是 `3`，**这是同一个事实**。"""
    assert M._plain(3.0) == "3"
    assert M._plain(12000) == "12000"
    assert _found(M._plain(3.0), "加班3小时")


def test_leading_zeros_do_not_matter():
    """`05` 和 `5` 是同一个月份。不去前导零的话，日期永远匹配不上。"""
    assert M._skeleton("2024-05-10") == M._skeleton("2024年5月10日")


# ===========================================================================
# ② 两类事实：照抄型 vs 换算型
# ===========================================================================


def test_quotable_is_decided_by_the_program_not_by_hand():
    """★ 分类必须程序判，不能人在 YAML 里标。

    人在 YAML 里标 `quotable: true`，标错了**没有任何东西会红**——
    你会拿着一个错的分类去画曲线。算出来的不会骗人。
    """
    # 照抄型：原话里就是这个数
    assert classify_quotable("12000", "我月薪12000，公司现在说要裁员")
    assert classify_quotable("西安", "公司在西安，做软件的")
    # 换算型：原话里根本没有这个写法
    assert not classify_quotable("37", "我干了三年零一个月")
    assert not classify_quotable("False", "他们让我这周就走，没提前通知")


def test_a_derived_value_is_not_quotable():
    """★ `37` 这个数**只存在于槽位里**，用户从来没说过。

    这就是整个记忆层存在的理由，也是"隔得近 ≠ 原文里有"的活标本。
    """
    assert not classify_quotable("37", "我干了三年零一个月")


def test_a_one_digit_value_is_never_quotable():
    """一位数的值用子串找必然误命中（`3` 到处都是），所以保守地归到换算型。

    **判不了的就不判**——归到保守那一侧，比给出一个好看但假的数强。
    """
    assert not classify_quotable("3", "平时每天加班3小时")


# ===========================================================================
# ③ 探针期望值 ↔ 槽位表
# ===========================================================================


def test_fragment_label_lookup_ignores_spacing():
    """`是否提前30日通知` 和 `是否提前 30 日通知` 必须都能查到。

    槽位的标签里带空格（是给人读的），而写数据集的人不会去对齐它。
    第一版直接拿原标签当键，差一个空格就报"这不是任何槽位的标签"——
    看上去像写错了槽位名，排查方向直接跑偏。
    """
    assert expected_fragment_slot("月工资：12000") == "monthly_wage"
    assert expected_fragment_slot("是否提前30日通知：False") == "notice_given"
    assert expected_fragment_slot("是否提前 30 日通知：True") == "notice_given"


def test_an_unknown_label_fails_loudly():
    """认不出来的标签必须当场炸。

    静默返回 None 的话，"标签写错了"会表现为"这条探针永远不通过"，
    而你会去记忆层里找一个根本不存在的 bug。
    """
    with pytest.raises(KeyError, match="不是任何槽位的标签"):
        expected_fragment_slot("月工资水平：12000")


def test_normalize_folds_rendering_differences():
    """归一化只做两件事：去空白、去掉整数尾巴上的 `.0`。"""
    assert normalize("- 月工资：12000.0") == "-月工资：12000"
    assert normalize("- 是否提前 30 日通知：False") == "-是否提前30日通知：False"
    # ⚠️ 不能把小数的有效位削掉
    assert normalize("12000.05") == "12000.05"


# ===========================================================================
# ④ 数据集体检：**抓不到错的体检比没有体检更糟**
# ===========================================================================


def _tiny(**over) -> Session:
    """一段 8 轮的小对话，用来喂给体检。默认是**干净**的。"""
    base = dict(
        id="t",
        turns=[
            "我2021年3月入职，月薪12000，公司在西安",
            "公司说要裁员",
            "裁员名单里有我",
            "他们说是经营困难",
            "我平时加班挺多的",
            "社保一直交着",
            "那经济补偿怎么算",
            "那到底能赔多少",
        ],
        probes=[Probe(turn=8, expects=["月工资：12000"])],
    )
    base.update(over)
    return Session(**base)


def test_a_clean_dataset_has_no_complaints():
    assert audit_dataset([_tiny()]) == []


def test_the_audit_catches_a_wrong_expectation():
    """期望值写错了（实际是 12000，你写了 99999）——必须报出来。"""
    s = _tiny(probes=[Probe(turn=8, expects=["月工资：99999"])])
    problems = audit_dataset([s])
    assert problems and "99999" in problems[0]


def test_the_audit_catches_an_empty_slot():
    """这条事实在这段对话里**压根没出现过**——也必须报出来。

    （用 `合同期限` 而不是 `城市`：`_tiny` 的第 1 轮里有"西安"，
    拿城市当例子会被"值对不上"先行拦下，测不到"槽位是空的"这条路。）
    """
    s = _tiny(probes=[Probe(turn=8, expects=["合同期限（月）：24"])])
    problems = audit_dataset([s])
    assert problems and "根本是空的" in problems[0]


def test_the_audit_catches_a_zero_latency_probe():
    """★ 探针的事实如果是**这一轮才说的**，那它测的是"最近原文"，不是记忆。

    这种探针看起来会"全部通过"，很容易被当成好数据——
    实际上它对四种策略一个都分辨不了。
    """
    s = _tiny(probes=[Probe(turn=1, expects=["月工资：12000"])])
    problems = audit_dataset([s])
    assert any("这一轮才写进去的" in p for p in problems)


def test_the_audit_catches_a_value_that_is_in_the_probe_turn():
    """★ 最隐蔽的一种：值就写在探针那一轮的原话里。

    那样它进提示词靠的是"最近原文"，和记忆层没关系。
    这条是"体检第 4 条"，加它的理由就是它太容易骗过前三条。
    """
    turns = _tiny().turns
    turns = turns[:7] + ["我月薪还是12000，能赔多少"]
    s = _tiny(turns=turns, probes=[Probe(turn=8, expects=["月工资：12000"])])
    problems = audit_dataset([s])
    assert any("就写在探针那一轮的用户原话里" in p for p in problems)


# ===========================================================================
# ⑤ 跑一遍：四种策略真的能分出高下
# ===========================================================================


def _long_session() -> Session:
    """12 轮：事实在第 1 轮，探针在第 12 轮，隔 11 轮。

    11 轮 > 窗口 6 轮，也 > 滑动窗口策略被拉大到的 10 轮。
    **所以这是"两种机制分道扬镳"的那一段**——
    滑动窗口盖不住了，只有槽位层还能把 12000 带过来。
    """
    turns = ["我月薪12000，公司说要裁员"]
    turns += [f"第{i}轮随便问问" for i in range(2, 12)]
    turns += ["那到底能赔多少"]
    assert len(turns) == 12
    return Session(id="t12", turns=turns, probes=[Probe(turn=12, expects=["月工资：12000"])])


def test_the_four_policies_are_ordered():
    """★ 保持率必须按"记忆能力"单调递增。

    这是 `MemoryPolicy` 那个枚举的顺序所承诺的东西。
    **不单调就说明实现里有 bug，而不是"策略之间没有好坏"**——
    所以这条断言是有方向的，不是"随便跑跑看"。
    """
    s = _long_session()
    price = load_pricing()
    rates = [run_policy([s], p, price_in=price).retention for p in MemoryPolicy]
    assert rates == sorted(rates), rates
    # 最弱的基线必须是 0：它只带最后一轮，第 12 轮看不到第 1 轮那句话
    assert rates[0] == 0.0
    # 最强的那一个必须拿满
    assert rates[-1] == 1.0


def test_gap_is_computed_not_hand_written():
    """「隔了多少轮」是从槽位自己的 `turn` 里算出来的。

    手写这个数一定会和轮次文本漂移，而漂移了也不报错——
    只是曲线画错。这里钉住它的算法：探针轮次 - 槽位写入轮次。
    """
    s = _long_session()
    r = run_policy([s], MemoryPolicy.NO_MEMORY, price_in=load_pricing())
    assert r.rows[0].gap == 11  # 第 12 轮问、第 1 轮说的


def test_text_visible_and_structured_are_two_different_signals():
    """★ 这两个信号分开报，是这份评测不骗人的关键。

    - `structured`：以槽位行（`- 月工资：12000`）的形式出现 → 只有 `layered` 有
    - `text_visible`：值在提示词里找得到（原文或槽位行都算）→ 四种策略都可能
    """
    s = _long_session()
    price = load_pricing()

    layered = run_policy([s], MemoryPolicy.LAYERED, price_in=price).rows[0]
    assert layered.structured and layered.text_visible

    sw = run_policy([s], MemoryPolicy.SLIDING_WINDOW, price_in=price).rows[0]
    assert not sw.structured  # 滑动窗口不渲染槽位行
    assert not sw.text_visible  # 第 1 轮那句话早被挤出窗口了
    assert not sw.passed


def test_the_headline_metric_uses_only_quotable_facts():
    """★ 主指标只看照抄型。混进换算型会得出"插了槽位层就赢麻了"的假结论。

    换句话讲：**这个测试防的是我自己吹牛。**
    """
    s = _long_session()
    r = run_policy([s], MemoryPolicy.LAYERED, price_in=load_pricing())
    assert r.n_quotable == 1 and r.n_computed == 0
    assert r.retention_quotable == r.retention


def test_a_computed_fact_is_counted_separately():
    """换算型的事实（"三年零一个月" → 37）单独计数，不进主指标。"""
    turns = ["我干了三年零一个月，公司要裁我"]
    turns += [f"第{i}轮随便问问" for i in range(2, 10)]
    turns += ["那到底能赔多少"]
    s = Session(id="c", turns=turns, probes=[Probe(turn=10, expects=["工作年限（月）：37"])])
    assert audit_dataset([s]) == []
    r = run_policy([s], MemoryPolicy.LAYERED, price_in=load_pricing())
    assert r.n_quotable == 0 and r.n_computed == 1
    assert r.retention_computed == 1.0
    # ⚠️ 主指标的分母是 0，不是 1 —— 没有可对比的样本时不能编一个数出来
    assert r.retention_quotable == 0.0


# ===========================================================================
# ⑥ 报表：结论必须跟着数据走
# ===========================================================================


def test_the_report_warns_about_its_own_limits():
    """★ 报表必须自带"我测不到什么"。

    **一份不写局限的评测报告，读者会自动把代理指标当成真实指标**——
    然后拿着它去下结论。所以这几句话是报表的一部分，不是可选装饰。
    """
    sessions = load_sessions()
    price = load_pricing()
    md = render_report(
        [run_policy(sessions, p, price_in=price) for p in MemoryPolicy], sessions, price
    )
    assert "测不到" in md
    assert "同义反复" in md
    assert "提示词里有 ≠ 模型会用" in md


def test_the_report_does_not_pretend_the_recall_layer_is_the_hero():
    """★ 那张"各层谁挣的分"的表必须出现。

    没有它，`layered` 的 1.000 会被读成"四层架构很强"；
    有了它，读者一眼看到长期召回层只在一小部分探针上有内容。
    """
    sessions = load_sessions()
    price = load_pricing()
    md = render_report(
        [run_policy(sessions, p, price_in=price) for p in MemoryPolicy], sessions, price
    )
    assert "到底是谁挣的" in md
    assert "`slots`" in md


def test_the_report_is_a_function_of_its_inputs():
    """★ 同一个输入必须得到同一份输出——**不读模块级全局**。

    第一版把单价放在模块全局里再由 `main()` 赋值，
    结果从测试里调 `render_report()` 会读到默认值，
    报表里就出现一列和实际不符的钱，而且没有任何东西会红。
    """
    sessions = load_sessions()[:2]
    res = [run_policy(sessions, MemoryPolicy.LAYERED, price_in=2.0)]
    a = render_report(res, sessions, 2.0)
    b = render_report(res, sessions, 2.0)
    assert a == b
    c = render_report(res, sessions, 99.0)
    assert a != c  # 单价真的进了报表
    assert "99.0" in c


# ===========================================================================
# ⑦ 真的数据集：它自己得是干净的
# ===========================================================================


def test_the_real_dataset_passes_its_own_audit():
    """★ 全量数据集的体检。

    这条测试的价值在于：**它把"数据集写错了"和"实现有 bug"分开报**。
    没有它，15 段 × 20 轮里任何一个数字写错，都会表现为
    "某个策略在某一轮莫名其妙地挂了"，然后你花两小时去查记忆层。
    """
    problems = audit_dataset(load_sessions())
    assert problems == [], "\n".join(problems)


def test_every_probe_expectation_is_a_known_slot():
    """数据集里每一条期望值的标签都必须是真槽位的标签。

    （体检内部也会调这个，但单独列一条是因为它的失败信息最有用：
    它直接告诉你"改名的那个槽位，评测集还没跟着改"。）
    """
    for s in load_sessions():
        for probe in s.probes:
            for frag in probe.expects:
                expected_fragment_slot(frag)


def test_the_dataset_actually_spans_the_window_boundary():
    """★ 数据集必须真的跨过窗口边界，否则整份报告没有意义。

    如果所有探针都只隔 1~3 轮，那么四种策略的差距会被压平——
    **不是"策略没差别"，是"这份数据问不出差别"。**

    窗口是 6 轮、滑动窗口策略被拉到 10 轮，所以必须有隔 11 轮以上的探针。
    """
    gaps = []
    for s in load_sessions():
        turns = {i for i in range(1, len(s.turns) + 1)}
        assert turns  # 防呆
        for probe in s.probes:
            gaps.append(probe.turn)
    assert max(gaps) >= 18, "最晚的探针至少要到第 18 轮，否则跨不过 10 轮窗口"
