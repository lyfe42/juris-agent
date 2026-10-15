r"""长对话记忆评测（F22）。★

## 要回答的那个问题

> 用户第 1 轮说了月薪，第 20 轮才问"那到底赔多少"。
> **中间隔了 19 轮，那个数字还在不在？**

这不是"上下文窗口够不够大"的问题（1M 上下文，20 轮塞得下）。
是"**该记住的东西，有没有被压在别的东西下面**"的问题。

## 怎么测才不是自欺欺人

### 一、测的是"提示词里有没有"，不是"模型答得对不对"

理由：

1. **便宜且可复现。** 15 段 × 20 轮 × 4 个策略，如果每轮都要调模型，
   那是 1200 次调用、几百块钱、几小时，而且 `temperature=0` 也不可复现
   （实测过，见 `reports/diagnosis.md`）。
2. **它测的正是记忆层。** 记忆层的产出就是那段提示词。**它有没有把数字放进去，
   是一个纯粹的、可判定的问题**；模型拿到之后答不答得对，是另一层的事。
3. **失败可归因。** "没进去"和"进去了但模型没用"是两种完全不同的 bug，
   分开测才能分开修。混在一起测，你只知道"答错了"。

⚠️ **这个选择的局限必须写在报告里**：提示词里有这个数字，
不等于模型一定会用；也不等于提示词里没有它就一定答错（模型可能从
最近几轮的上下文里推出来）。所以这里的数字是**上界**，
不是"端到端正确率"。**把代理指标说成主指标，是评测里最常见的自欺。**

### 二、期望值写成"标签：值"的完整片段

`月工资：12000` 而不是 `12000`。这样：

- 不会误命中（`12000` 这个串可能出现在任何地方）
- **数据集自己能体检**：`audit_dataset()` 会在探针那一轮先跑一遍抽取，
  确认"这个槽位此时确实就是这个值"。对不上就报错——
  手写的评测集一定会写错，靠检查比靠自觉可靠。

### 三、"隔了多少轮"从槽位自己的记录里读，不手填

每条 `SlotValue` 都带一个 `turn`（哪一轮写进去的）。所以
"隔了多少轮" = `探针轮次 - 槽位.turn`，**自动算出来**。
手填这个数一定会和上面那堆轮次文本漂移，而漂移了也不报错——
只是曲线画错。**能算出来的东西不要手写。**

### 四、必须逐轮重放，不能只跑探针那一轮

⚠️ **这是本文件里最容易做错、也最不容易被发现的一步。**

`layered` 策略的长期召回层，是靠 `build()` **每一轮**把"离开窗口的轮次"
归档进去攒起来的（见 `MemoryManager._archive`）。如果只跑探针那一轮，
长期层**从头到尾都是空的**——于是 `layered` 看起来和
`sliding_window_summary` 一样差，而你会得出"长期召回没用"的结论。

**这个结论是评测方法造出来的，不是真的。** 所以：
一段对话从头到尾跑完，每个探针轮次**顺手截一张快照**。

### 五、四个策略比的是同一份数据、同一份代码

`MemoryManager` 换一个 policy 就是另外一个策略。四种策略**不是四份实现**——
否则比出来的差异里混着"实现水平的差异"，那就不叫对照实验了。

### 六、成本列怎么算

记忆层的 token 数 × `configs/pricing.yaml` 里的牌价。
**它是估算不是账单**：不含重试、不含失败调用、不含缓存命中。
报告里每一处金额都注明"按本表牌价换算"。

## 输出

- `reports/memory_ablation.md`：主表 + 按"隔了多少轮"分的保持率曲线
- `reports/memory_ablation.json`：原始数据（每一条探针的通过/不通过）
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import yaml
from pydantic import BaseModel, Field

from juris_agent import PROJECT_ROOT
from juris_agent.memory.manager import MemoryBundle, MemoryManager, MemoryPolicy
from juris_agent.memory.slots import SLOT_SPECS

# --------------------------------------------------------------------------- #
# 归一化：把"渲染出来的样子"和"期望值"对齐
# --------------------------------------------------------------------------- #
#
# ⚠️ 这两个函数**必须放在文件最上面**，因为它们被下面的标签表用到。
# 第一版放在文件中间，结果标签表在模块加载时就要调它——NameError。
# **"谁在模块级被调用"决定了"谁必须写在前面"**，这跟逻辑上的从属关系无关。

_WS = re.compile(r"\s+")
# 浮点数渲染出来是 `12000.0`，而人写期望值会写 `12000`。
# **只在后面不是数字的时候才去掉 `.0`**，否则 `12000.05` 会被削成 `12000.5`。
_TRAILING_ZERO = re.compile(r"(\d+)\.0(?![0-9])")


def normalize(text: str) -> str:
    """去掉所有空白 + 去掉整数后面的 `.0`。

    为什么要去空白：槽位标签里带空格（`是否提前 30 日通知`），
    而人写期望值时不会刻意对齐那个空格。**让"格式的差异"不影响判定**，
    否则你会花一下午调一个跟结论无关的字符串。
    """
    return _TRAILING_ZERO.sub(r"\1", _WS.sub("", text))


def normalize_label(label: str) -> str:
    """标签专用：**只去空白，不去 `.0`**。

    槽位的标签里不会出现数字，所以 `.0` 那条规则在这里没有意义；
    但把它顺手去掉会更糟——万一哪天有个标签叫「赔偿 2.0 倍」，
    两条路会归一化成不同的字符串，然后就对不上了。
    **归一化函数只做"确实需要归一化"的那一件事。**
    """
    return _WS.sub("", label)


# "当分隔符看"的字符。**只有这些**，别的一律当内容。
#
# 为什么要这么个集合：同一个事实在原文里和槽位里写法不同——
# 原文是「2024年5月10日」，槽位里是「2024-05-10」。要问
# "这个事实在不在提示词里"，得先让两种写法变成同一个东西。
#
# ⚠️ 但**不能把所有非数字都当分隔符**：那样「西安」会变成空字符串，
# 「True」也会变成空字符串——它们的"内容"就是那些字。
# 所以分隔符是一张**明确列出来**的表，不是一条"非数字"的规则。
# 「年」「月」「日」在这里当分隔符，是因为它们在日期里纯粹起分隔作用。
_SEP_CHARS = " \t\n\r-—–/.:：。、，,；;()（）[]【】《》\"'`~·　年月日"
_SEP = "\u0001"  # 一个不会出现在正文里的哨兵字符


def _plain(value: object) -> str:
    """槽位值 → "人看到的那串字"。**浮点数按整数渲染。**

    `3.0` 和 `3` 是同一个事实，但 `str(3.0) == "3.0"`。槽位层渲染时
    走的是 `str()`，所以提示词里是 `加班小时数：3.0`——而 `normalize()`
    会把那个尾巴去掉（见上面的 `_TRAILING_ZERO`）。这里跟着一起按整数渲染，
    两条路才对得上。**同一个值在两处渲染成不同的样子，是"看的人对不上"的源头。**
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _skeleton(text: str) -> str:
    """把文本压成"骨架"：数字串保留（去掉前导零），分隔符全部归一成一个哨兵。

    - `2024-05-10` → `2024·5·10`
    - `2024年5月10日` → `2024·5·10`（末尾的哨兵被 strip 掉）
    - `12000` → `12000`；`西安` → `西安`；`True` → `True`

    前导零也要去：`05` 和 `5` 是同一个月份，但字符串上是两个东西。
    """
    parts: list[str] = []
    for ch in text:
        if ch.isdigit():
            parts.append(ch)
        elif ch in _SEP_CHARS:
            parts.append(_SEP)
        else:
            parts.append("|" + ch)  # 内容字符：加个前缀，保证不和数字串粘连
    # 数字串单独处理：需要看"连续数字"才能去前导零
    out = "".join(parts)
    out = re.sub(r"\d+", lambda m: (m.group(0).lstrip("0") or "0"), out)
    out = re.sub(f"{_SEP}+", _SEP, out)
    return out.strip(_SEP)


def _value_pattern(value: object) -> re.Pattern[str]:
    """把槽位值编成一个"认得出它、又不会认错"的正则。

    ⚠️ **两头必须用 `(?<!\\d)` / `(?!\\d)` 把数字锁住。** 不锁的话
    「加班小时数 3」会在「3000」里命中——那是一个完全不同的数，
    而报表会因此多出几个假的"通过"。

    ⚠️⚠️ **每一段都要 `re.escape`。** 骨架里的内容字符是带 `|` 前缀的
    （见 `_skeleton`），`|` 在正则里是**或**。第一版忘了转义，
    值 `False` 编出来的正则成了 `(?<!\\d)|F|a|l|s|e(?!\\d)`——
    第一个分支是**空串**，于是它在**任何**字符串里都命中。

    后果是：报表里所有 `text_visible` 全是 `True`，
    `no_memory` 的保持率看起来高达 0.88，比 `sliding_window` 还高。
    **一个多余的 `|` 把整张表变成了噪声，而且它不报错、不抛异常，
    只是数字全都变得很好看。** 这类 bug 只能靠"这个数不合理"来发现——
    所以看到"隔得越久记得越牢"这种反直觉的结果，第一反应必须是去查度量，
    而不是去写结论。
    """
    sk = _skeleton(_plain(value))
    chunks = [re.escape(c) for c in sk.split(_SEP) if c]
    body = r"[^\d]{0,3}".join(chunks)
    return re.compile(r"(?<!\d)" + body + r"(?!\d)")


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #

# 标签（如"月工资"）→ 槽位名。**从 SLOT_SPECS 反查，不手抄。**
# 手抄的后果是：改一个槽位的中文名，评测集悄悄失效，而没有任何东西会红。
#
# ⚠️ **键是"去掉空白"之后的标签。** 因为槽位的中文名里带空格
# （`是否提前 30 日通知`，那个空格是给人读的），而数据集里写期望值时
# 没人会去对齐它。第一版直接用原标签当键，结果数据集里写
# `是否提前30日通知` 就查不到——报的错还是"这不是任何槽位的标签"，
# 看上去像数据集写错了槽位名，其实是差一个空格。
# **凡是"人肉输入的字符串"，比对之前一律先归一化。**
_LABEL_TO_SLOT: dict[str, str] = {normalize_label(s.label): name for name, s in SLOT_SPECS.items()}

# 按"隔了多少轮"分桶。**桶的边界不是等分的，是照着窗口大小划的**：
# 生产窗口是 6 轮，所以 1–5（还在窗口里）/ 6–10（刚出窗口）/ 11+（早就出去了）
# 这三档才有区分度。等分桶（1-7/8-14/15-20）会把"刚出窗口"和"早就出窗口"
# 混在一起，而这两者的失败原因完全不同。
_GAP_BUCKETS: tuple[tuple[int, int], ...] = ((0, 5), (6, 10), (11, 10_000))


class Probe(BaseModel):
    """一条探针：在第 `turn` 轮问，提示词里必须出现 `expects` 里的每一条。"""

    turn: int
    expects: list[str]


class Session(BaseModel):
    """一段 20 轮的长对话。"""

    id: str
    title: str = ""
    turns: list[str]
    probes: list[Probe]

    def history_at(self, turn: int) -> list[dict[str, str]]:
        """第 `turn` 轮**开始时**的完整历史（不含这一轮本身）。

        ⚠️ 助手回复是**造出来的**，不是真调模型来的。理由：
        这段评测只关心"用户说过什么"，而造一句假回复比真调模型便宜得多、
        也稳得多。唯一要守住的约束是**它不能包含探针期望的那串数字**——
        所以这里刻意写成一句不含任何数字的套话。
        要是哪天有人把温度、模型名写进这句回复，"最近原文"就多了一条
        漏进测试的通路，指标立刻失真。
        """
        out: list[dict[str, str]] = []
        for i, text in enumerate(self.turns[: turn - 1], start=1):
            out.append({"role": "user", "content": text})
            out.append({"role": "assistant", "content": f"（第{i}轮回答）已按劳动法相关条款分析。"})
        return out


class ProbeResult(BaseModel):
    """一条探针 × 一个策略的结果。**报表里的每一格就是它。**"""

    session: str
    turn: int
    expect: str
    slot: str
    gap: int
    """这个事实是**多少轮之前**说的。自动从槽位的 `turn` 算出来。"""

    value: str
    """槽位值渲染成的那串字（`月工资：12000` → `12000`）。"""

    quotable: bool
    """**这条事实能不能在原文里照抄。** 见 `classify_quotable()`。

    `True`：用户原话里就有这个数（"月薪 12000"）——任何策略只要把那句话
    留在窗口里，模型就看得见。**这一类才能拿来横向对比四种策略。**
    `False`：原文里根本没有这个值，它是算出来的或词法归一化过的
    （"三年零一个月" → `37`；"没提前通知" → `False`）。**这一类只有槽位层
    能给出**，所以它不是"四种策略的对比"，而是"槽位层在不在"的检查。"""

    text_visible: bool
    """按"数字骨架"能在提示词里找到这个值（原文或槽位行都算）。"""

    structured: bool
    """以槽位行（`- 月工资：12000`）的形式出现。"""

    passed: bool
    """`text_visible or structured`。**这就是"模型能不能看见这条事实"。**"""

    tokens: int
    cost_cny: float


class PolicyResult(BaseModel):
    policy: str
    n_probes: int
    n_passed: int
    retention: float

    n_quotable: int = 0
    n_quotable_passed: int = 0
    retention_quotable: float = 0.0
    """**主指标**：只看"原文里能照抄"的那些事实。四策略唯一公平的对比口径。"""

    n_computed: int = 0
    n_computed_passed: int = 0
    retention_computed: float = 0.0
    """**另一类**：原文里没有这个写法的（算出来的、归一化过的）。"""

    mean_tokens: float
    total_cost_cny: float
    by_gap: dict[str, float] = Field(default_factory=dict)
    """**只统计 `quotable` 的那一半**——混进来会让曲线被"槽位层有没有开"带偏。"""

    rows: list[ProbeResult] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 读数据
# --------------------------------------------------------------------------- #


def load_sessions(path: Path | None = None) -> list[Session]:
    p = path or (PROJECT_ROOT / "data" / "eval" / "memory_sessions.yaml")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return [Session(**s) for s in raw]


def expected_fragment_slot(fragment: str) -> str:
    """`"月工资：12000"` → `"monthly_wage"`。**认不出来就当场炸。**

    静默返回 None 的话，数据集里写错一个标签就会变成"这条探针永远不通过"，
    而你会以为是记忆层的问题——**排查方向被一个笔误带偏，是最贵的一种错。**
    """
    label = normalize_label(fragment.split("：", 1)[0])
    if label not in _LABEL_TO_SLOT:
        raise KeyError(
            f"探针期望值 {fragment!r} 里的标签 {label!r} 不是任何槽位的标签。"
            f"已知标签：{sorted(_LABEL_TO_SLOT)}"
        )
    return _LABEL_TO_SLOT[label]


# --------------------------------------------------------------------------- #
# 逐轮重放：四策略共用的一份"历史"
# --------------------------------------------------------------------------- #


class Replay(BaseModel):
    """一段对话跑完之后的产物：**每个探针轮次的记忆快照** + 槽位表。

    做成一个显式的对象而不是两组并列的返回值，是为了让"快照和槽位表
    必须是同一时刻的"这件事**在类型上就成立**——分开传的话，
    迟早有人先拿第 8 轮的快照、再拿第 20 轮的槽位表去算 gap。
    """

    snapshots: dict[int, MemoryBundle] = Field(default_factory=dict)
    """探针轮次 → 那一轮组装出来的记忆块。"""

    slot_turns: dict[str, int] = Field(default_factory=dict)
    """槽位名 → 它是第几轮写进去的。**gap 全靠它算。**"""


def replay(session: Session, policy: MemoryPolicy, probe_turns: set[int]) -> Replay:
    """把整段对话**逐轮**喂给一个 `MemoryManager`。

    ⚠️ 每个策略、每段对话都是一个**全新的管理器**，不共享任何状态。
    否则"第 3 段对话的长期记忆"会漏进"第 5 段"，比出来的差异就说不清是谁造成的。

    顺序是 `observe → derive → build`，一步都不能少：

    - `observe` 把这一轮说的话抽成槽位（这一轮说的，这一轮就该用上）
    - `derive` 跑派生（入职日期 → 工作月数、城市 → 社平工资）
    - `build` 组装**这一轮**要给模型的提示词，顺手把离开窗口的轮次归档进长期层

    **`build` 必须每轮都调**，见模块 docstring 第四节。
    """
    mgr = MemoryManager(policy=policy)
    out = Replay()

    for i, text in enumerate(session.turns, start=1):
        mgr.observe(text, turn=i)
        mgr.derive(turn=i)
        bundle = mgr.build(text, session.history_at(i), current_turn=i)
        if i in probe_turns:
            out.snapshots[i] = bundle

    out.slot_turns = {k: v.turn for k, v in mgr.slots.values.items()}
    return out


# --------------------------------------------------------------------------- #
# 数据集体检
# --------------------------------------------------------------------------- #


def audit_dataset(sessions: Iterable[Session]) -> list[str]:
    """跑探针之前先验一遍数据。**返回问题清单，空列表表示数据是干净的。**

    查四件事：

    1. **轮次够不够**：探针的轮次不能超出这段对话的长度。
    2. **探针那一轮，槽位真的是那个值吗**：重放到探针那一轮，先看槽位表。
       对不上说明**数据集写错了**，不是记忆层的问题。
       （比如"干了三年"实际抽出来是 36 个月，而你期望值里写了 37。）
    3. **事实不能是"这一轮才说的"**：如果槽位的写入轮次 ≥ 探针轮次，
       那这个事实根本没经过记忆层——它是刚说出口的。这种探针**没有信息量**，
       它测的是"这一轮的话在不在提示词里"，答案必然是"在"。
    4. **期望值不能是"探针那一轮原话里本来就有"的字**：那样它进提示词靠的是
       "最近原文"，不是记忆。这条最隐蔽——**它会让你误以为某个策略
       "记住"了，其实它只是还没忘掉刚才那句话。**

    体检走的是**与策略无关**的路径（一律用 `layered` 重放），
    这样"数据的问题"和"策略的行为"就是两件分开的事。
    """
    problems: list[str] = []
    for s in sessions:
        if len(s.turns) < 5:
            problems.append(f"{s.id}: 只有 {len(s.turns)} 轮，太短了")

        for probe in s.probes:
            if not 1 <= probe.turn <= len(s.turns):
                problems.append(f"{s.id}: 探针轮次 {probe.turn} 超出对话长度")
                continue

            mgr = MemoryManager(policy=MemoryPolicy.LAYERED)
            for i, turn_text in enumerate(s.turns[: probe.turn], start=1):
                mgr.observe(turn_text, turn=i)
                mgr.derive(turn=i)

            for frag in probe.expects:
                slot = expected_fragment_slot(frag)
                entry = mgr.slots.entry(slot)
                if entry is None:
                    problems.append(
                        f"{s.id} 第{probe.turn}轮：期望 {frag!r}，但槽位 {slot} 此时**根本是空的**"
                    )
                    continue
                rendered = normalize(f"- {SLOT_SPECS[slot].label}：{entry.value}")
                if normalize(frag) not in rendered:
                    problems.append(
                        f"{s.id} 第{probe.turn}轮：期望 {frag!r}，"
                        f"实际槽位是 {entry.value!r}（第{entry.turn}轮写入）"
                    )
                if entry.turn >= probe.turn:
                    problems.append(
                        f"{s.id} 第{probe.turn}轮：槽位 {slot} 是**这一轮才写进去的**，"
                        f"探针没有测到任何记忆"
                    )
                # ⚠️ 用 `_plain()` 而不是 `str()`：槽位里存的是 `12000.0`，
                # 而用户原话写的是 `12000`。直接 `str()` 比较的话，
                # **这一条体检永远抓不到东西**——它静默地放过所有该抓的情况，
                # 而你还以为数据是干净的。（写这条测试的时候当场抓到的。）
                if _plain(entry.value) in s.turns[probe.turn - 1]:
                    problems.append(
                        f"{s.id} 第{probe.turn}轮：期望值 {entry.value!r} "
                        f"**就写在探针那一轮的用户原话里**——这条探针测的是"
                        f"「最近原文」，不是记忆；换个数字或者换个轮次问"
                    )
    return problems


# --------------------------------------------------------------------------- #
# 跑
# --------------------------------------------------------------------------- #


def _rate(hits: int, total: int) -> float:
    return (hits / total) if total else 0.0


def layer_usage(sessions: list[Session], policy: MemoryPolicy) -> dict[str, int]:
    """在**所有探针轮次**上，每一层实际放进去几条（0 条就是"这一层没帮上忙"）。

    ⚠️ 这个诊断是专门用来**防止自我表扬**的。

    你看 `layered` 的保持率是 1.000，很容易顺手写成"四层架构效果好"。
    但一查就知道：这 1.000 分几乎全是**槽位层**拿的，
    长期召回层在大部分探针上**一条都没召回**（"没有词重叠的历史"）。
    如果报告只写总分，读者会以为是四层一起立功——
    **那是把"我插了一个槽位表"包装成了"我的四层架构很强"。**
    这种话在面试里被追问一句就散了，不如自己先写清楚。
    """
    tally: dict[str, int] = {}
    n = 0
    for s in sessions:
        rp = replay(s, policy, {p.turn for p in s.probes})
        for b in rp.snapshots.values():
            n += 1
            for lay in b.layers:
                tally[lay.name] = tally.get(lay.name, 0) + (1 if lay.items else 0)
    tally["_探针轮次总数"] = n
    return tally


def run_policy(
    sessions: list[Session], policy: MemoryPolicy, *, price_in: float
) -> PolicyResult:
    """把一个策略跑完所有探针。

    ⚠️ **每个探针都从同一条起跑线出发**：同一段对话、同一个策略、
    同样逐轮重放。唯一的变量是"在第几轮问"。
    """
    rows: list[ProbeResult] = []
    for s in sessions:
        probe_turns = {p.turn for p in s.probes}
        rp = replay(s, policy, probe_turns)

        for probe in s.probes:
            bundle = rp.snapshots[probe.turn]
            block = normalize(bundle.prompt_block)
            skel = _skeleton(bundle.prompt_block)

            for frag in probe.expects:
                slot = expected_fragment_slot(frag)
                written = rp.slot_turns.get(slot, probe.turn)
                raw_value = frag.split("：", 1)[1]

                # 值本身（去标签）——用来判"原文里看得见吗"。
                # 我们拿不到 SlotValue 的原对象，但**不需要**：
                # 期望值里的那一串字就是渲染出来的样子（数据集体检已经钉住了这点）。
                text_visible = _value_pattern(raw_value).search(skel) is not None
                structured = normalize(frag) in block
                quotable = classify_quotable(raw_value, s.turns[written - 1])

                rows.append(
                    ProbeResult(
                        session=s.id,
                        turn=probe.turn,
                        expect=frag,
                        slot=slot,
                        gap=probe.turn - written,
                        value=raw_value,
                        quotable=quotable,
                        text_visible=text_visible,
                        structured=structured,
                        passed=text_visible or structured,
                        tokens=bundle.total_tokens,
                        cost_cny=bundle.total_tokens / 1_000_000 * price_in,
                    )
                )

    # 曲线只画"照抄型"——另一类不适合横向对比，理由见 `classify_quotable`
    by_gap: dict[int, list[bool]] = {}
    for r in rows:
        if r.quotable:
            by_gap.setdefault(_gap_bucket(r.gap), []).append(r.passed)

    n_passed = sum(r.passed for r in rows)
    q = [r for r in rows if r.quotable]
    c = [r for r in rows if not r.quotable]
    return PolicyResult(
        policy=policy.value,
        n_probes=len(rows),
        n_passed=n_passed,
        retention=_rate(n_passed, len(rows)),
        n_quotable=len(q),
        n_quotable_passed=sum(r.passed for r in q),
        retention_quotable=_rate(sum(r.passed for r in q), len(q)),
        n_computed=len(c),
        n_computed_passed=sum(r.passed for r in c),
        retention_computed=_rate(sum(r.passed for r in c), len(c)),
        mean_tokens=(sum(r.tokens for r in rows) / len(rows)) if rows else 0.0,
        total_cost_cny=sum(r.cost_cny for r in rows),
        by_gap={_bucket_label(k): _rate(sum(v), len(v)) for k, v in sorted(by_gap.items())},
        rows=rows,
    )


def classify_quotable(value: object, source_turn_text: str) -> bool:
    """**这条事实能不能在原文里照抄？** 判据：值的骨架在它被写进去的那一轮原文里。

    这一步是本文件里**最关键的一个判断**，因为它决定了报表怎么读：

    | | 例子 | 原文里有吗 | 谁能承载它 |
    |---|---|---|---|
    | 照抄型 | 月工资 `12000`、城市 `西安`、离职日 `2024-05-10` | 有 | 窗口 / 摘要 / 槽位都行 |
    | 换算型 | 工作年限 `37`（原文是"三年零一个月"）、是否通知 `False`（原文是"没提前通知"） | **没有** | **只有槽位层** |

    ⚠️ **不分开会得出一个假结论。** 第一版把两类混在一起算，
    结果是 `0 / 0 / 0 / 1.0`——看上去像"槽位层赢麻了"，
    实际上那三个 0 是**结构性的**：`月工资：12000` 这种写法
    只有槽位层渲染得出来，别的策略永远不可能命中。
    **那不是测量，那是同一句话换了个说法。**
    """
    text = _plain(value)
    if len(_skeleton(text).replace(_SEP, "")) < 2:
        # 一位数的值（加班 3 小时）用子串找必然误命中，不敢判，归到"换算型"保守处理
        return False
    return _value_pattern(value).search(_skeleton(source_turn_text)) is not None


def _gap_bucket(gap: int) -> int:
    """返回桶的下界，用它当字典的键。"""
    for lo, hi in _GAP_BUCKETS:
        if lo <= gap <= hi:
            return lo
    return _GAP_BUCKETS[0][0]


def _bucket_label(lo: int) -> str:
    for a, b in _GAP_BUCKETS:
        if a == lo:
            return f"{a}-{b}" if b < 10_000 else f"{a}+"
    return str(lo)


# --------------------------------------------------------------------------- #
# 报表
# --------------------------------------------------------------------------- #


def render_report(results: list[PolicyResult], sessions: list[Session], price_in: float) -> str:
    """生成 `reports/memory_ablation.md`。

    ⚠️ **单价是参数，不是模块级全局。** 第一版写成了全局变量再由 `main()`
    赋值——那样从测试里调 `render_report()` 会读到模块默认值，
    报表里就会出现一列和实际不符的钱，而且没有任何东西会红。
    **藏在模块状态里的输入，是"同一个函数两次跑出不同结果"的经典来源。**
    """
    n_expects = sum(len(p.expects) for s in sessions for p in s.probes)
    buckets = sorted({_gap_bucket(r.gap) for res in results for r in res.rows})

    L: list[str] = []
    L.append("# 长对话记忆消融（自动生成，勿手改）")
    L.append("")
    L.append("生成命令：`python scripts/memory_ablation.py`")
    L.append("")
    L.append(f"- 评测集：`data/eval/memory_sessions.yaml`，{len(sessions)} 段 × 20 轮")
    L.append(
        f"- 探针 {sum(len(s.probes) for s in sessions)} 处、期望值 {n_expects} 条"
        "（一处探针可以同时查好几个槽位）"
    )
    L.append("- 记忆预算：`MemoryBudget()` 默认值（total=2048 token，窗口 6 轮）")
    L.append(
        f"- 成本按 `configs/pricing.yaml` 的牌价换算（输入 {price_in} 元/百万 token），"
        "**是估算不是账单**"
    )
    L.append("")
    L.append("## 先说清楚这个指标是什么")
    L.append("")
    L.append("**测的是「提示词里有没有那个值」，不是「模型答得对不对」。**")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    L.append("| 测什么 | 组装出来的提示词里，关键槽位值还在不在 |")
    L.append("| 怎么判 | 期望值写成 `月工资：12000` 这样的完整片段，做子串匹配 |")
    L.append("| 为什么这么测 | 便宜、可复现、失败可归因（「没进去」和「进去了但没用上」是两种 bug） |")
    L.append("| ⚠️ 局限 | 提示词里有 ≠ 模型会用。所以这是**上界**，不是端到端正确率 |")
    L.append("")
    L.append("「隔了多少轮」不是手填的，是从每个槽位自己的 `turn` 记录里算出来的——")
    L.append("**能算出来的数不要手写**，手写的一定会和轮次文本漂移，而漂移了也不报错。")
    L.append("")
    L.append("## 事实分成两类，**这条比主表重要**")
    L.append("")
    L.append("同一个「记住」，对两类事实是完全不同的难度：")
    L.append("")
    L.append("| 类型 | 例子 | 用户原话里长什么样 | 谁承载得住 |")
    L.append("|---|---|---|---|")
    L.append("| **照抄型** | 月工资 `12000`、城市 `西安`、离职日 `2024-05-10` | 原话里**就是**这个值 | 窗口 / 摘要 / 槽位都行 |")
    L.append("| **换算型** | 工作年限 `37`、是否提前通知 `False` | 「三年零一个月」「没提前通知」——**原文里没有这个值** | **只有槽位层** |")
    L.append("")
    L.append("**为什么必须分开报：** 换算型的事实，原文里根本不存在那个写法，")
    L.append("所以对它们来说「四种策略对比」是个假命题——只有槽位层答得出来，")
    L.append("剩下三个必然是 0。**把两类混在一起，你会得到一个好像是"
             "「插了槽位层就赢麻了」的表，其实那个差距是结构性的，不是测出来的。**")
    L.append("")
    L.append("分类是**程序判的**（`classify_quotable`：拿值的骨架去它的原文那一轮里找），")
    L.append("不是人在 YAML 里标的——标错了没人看得出来，算出来的不会骗人。")
    L.append("")
    L.append("## 主表")
    L.append("")
    L.append("| 策略 | **照抄型保持率** | 照抄型 通过/总数 | 换算型保持率 | 换算型 通过/总数 | 平均提示词 token | 估算成本（元/千次咨询） |")
    L.append("|---|---|---|---|---|---|---|")
    for r in results:
        cost_1k = r.mean_tokens / 1_000_000 * price_in * 1000
        L.append(
            f"| `{r.policy}` | **{r.retention_quotable:.3f}** "
            f"| {r.n_quotable_passed}/{r.n_quotable} "
            f"| {r.retention_computed:.3f} | {r.n_computed_passed}/{r.n_computed} "
            f"| {r.mean_tokens:.0f} | {cost_1k:.2f} |"
        )
    L.append("")
    L.append("（成本列 = 平均提示词 token × 输入牌价 ÷ 1e6 × 1000）")
    L.append("")
    L.append("**读表的方式：**「照抄型」那一列才是四种策略真正的区别所在——")
    L.append("它问的是「用户几十轮前说的那句话，还在不在上下文里」。")
    L.append("「换算型」那一列**不是对比**，它是在确认槽位层有没有正常工作；")
    L.append("除 `layered` 外全为 0 是设计使然，不是缺陷。")
    L.append("")

    # ---- 保持率曲线 ----
    L.append("## 保持率曲线：隔得越久，还记得吗")
    L.append("")
    L.append("横轴是「这个事实是多少轮之前说的」。**这一栏才是这份报告的重点**——")
    L.append("主表只有一个总数，它会把「最近的记得牢」和「远的全忘光」平均掉。")
    L.append("")
    L.append("| 策略 | " + " | ".join(_bucket_label(b) + " 轮前" for b in buckets) + " |")
    L.append("|---" * (len(buckets) + 1) + "|")
    for r in results:
        cells = [
            f"{r.by_gap[lb]:.2f}" if (lb := _bucket_label(b)) in r.by_gap else "—"
            for b in buckets
        ]
        L.append(f"| `{r.policy}` | " + " | ".join(cells) + " |")
    L.append("")

    # ---- 逐条明细 ----
    L.append("## 逐条明细（只列失败的那些）")
    L.append("")
    L.append("「哪一段、第几轮、缺了哪个槽位、隔了多少轮、**谁挂了**」——")
    L.append("这几个信息齐了，才能判断是「设计如此」还是「实现有 bug」。")
    L.append("")
    L.append("⚠️ 按「探针」聚合成一行，不是每条失败一行。第一版一条一行的写法")
    L.append("一共 130 行，读的人第一条就划走了——**明细表的作用是让人能扫一眼"
             "看出模式，不是把原始数据倒出来**（原始数据在 `memory_ablation.json` 里，"
             "谁要谁去拿）。")
    L.append("")
    L.append("| 对话 | 第几轮 | 期望 | 隔了多少轮 | 挂了的策略 |")
    L.append("|---|---|---|---|---|")
    grouped: dict[tuple[str, int, str, int], list[str]] = {}
    for r in results:
        for row in r.rows:
            if not row.passed:
                grouped.setdefault((row.session, row.turn, row.expect, row.gap), []).append(r.policy)
    if grouped:
        for (sess, turn, expect, gap), pols in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
            names = "、".join(f"`{p}`" for p in pols)
            L.append(f"| {sess} | {turn} | `{expect}` | {gap} | {names} |")
    else:
        L.append("| — | — | — | — | 全部通过 |")
    L.append("")
    L.append("**看最后一列。** 挂满四个策略的，多半是「这个事实本来就该丢」；")
    L.append("只挂在某几个策略上的，才是有信息量的那一行——")
    L.append("**同一个事实、同一段对话，差别只在「记忆策略怎么组装」，那才是策略的功劳。**")
    L.append("")

    # ---- 逐段 ----
    L.append("## 逐段保持率")
    L.append("")
    L.append("| 对话 | " + " | ".join(f"`{r.policy}`" for r in results) + " |")
    L.append("|---" * (len(results) + 1) + "|")
    for s in sessions:
        cells = []
        for r in results:
            sub = [x for x in r.rows if x.session == s.id]
            cells.append(f"{sum(x.passed for x in sub)}/{len(sub)}" if sub else "—")
        L.append(f"| {s.id} | " + " | ".join(cells) + " |")
    L.append("")

    # ---- 各层的实际贡献 ----
    L.append("## layered 那 1.000 分，到底是谁挣的")
    L.append("")
    L.append("下表统计：在**所有探针轮次**里，每一层实际放进去几条。")
    L.append("0 = 这一层那一轮什么都没帮上。")
    L.append("")
    usage = layer_usage(sessions, MemoryPolicy.LAYERED)
    total_pt = usage.pop("_探针轮次总数", 0)
    L.append("| 层 | 有内容的探针轮次数 | 占比 |")
    L.append("|---|---|---|")
    L.append(f"| 探针轮次总数 | {total_pt} | 100% |")
    for name, cnt in sorted(usage.items(), key=lambda kv: -kv[1]):
        pct = (cnt / total_pt * 100) if total_pt else 0.0
        L.append(f"| `{name}` | {cnt} | {pct:.0f}% |")
    L.append("")
    L.append("**这张表是用来泼冷水的，别跳过。**「四层架构拿了满分」这句话是错的——")
    L.append("分数几乎全部来自**槽位层**，它是唯一一个「只要开了就必然命中」的层。")
    L.append("长期召回层在相当多的探针上**一条都没召回**（检索词和问题没有重叠），")
    L.append("所以它在这份评测里的贡献远小于它的名气。")
    L.append("**把「我插了一个结构化的槽位表」讲成「我的四层记忆架构很强」，")
    L.append("被追问一句就散了；自己先把这行数字摆出来，反而是最稳的。**")
    L.append("")
    L.append("## 这份评测**测不到**什么")
    L.append("")
    L.append("写结论之前先把边界划清楚——下面这几条都是真的，不是免责声明：")
    L.append("")
    L.append("1. **槽位覆盖不到的事实，这里一条都没测。** 探针查的全是槽位表里的字段。")
    L.append("   用户说过的「我下周三要去仲裁」这种**没有对应槽位**的信息，")
    L.append("   本评测的保持率**不含它**——那部分只能靠摘要和原文，")
    L.append("   而这两层在题目里正好是弱项。")
    L.append("2. **`layered` 在换算型上恒为 1.000 是同义反复。** 槽位层就是为这些事实")
    L.append("   设计的，它不渲染出来才是 bug。**这个 1.000 不代表任何「效果」。**")
    L.append("3. **提示词里有 ≠ 模型会用。** 这是全文最要紧的一条局限，见开头那张表。")
    L.append("4. **助手回复是假的**（一句不含数字的套话），所以真实对话里")
    L.append("   助手引用过的事实，在真实的窗口里占的位置和这里不一样。")
    L.append("5. **摘要是规则拼的，不是模型写的。** 换成模型摘要，这一列大概率会变——")
    L.append("   但那就换来了一个不可复现的数（`temperature=0` 也不可复现，")
    L.append("   见 `reports/diagnosis.md` 的实测）。**这是刻意的取舍，不是没做。**")
    L.append("")
    L.append("## 从这张表里读出来的事")
    L.append("")
    L.append(_conclusions(results))
    L.append("")
    return "\n".join(L)


def _conclusions(results: list[PolicyResult]) -> str:
    """把"表里的数字说明什么"写成几句话。

    ⚠️ **数字全部从 `results` 里算出来，不手写。** 这份报告每次重跑数字都会变，
    手写的结论过两次就变成谎话——而读者不会去核对，他们只会读那段话。
    （判断条件仍然是手写的，这没法避免；但至少"数字"和"话"不会脱节。）
    """
    by = {r.policy: r for r in results}
    out: list[str] = []
    n = 0

    sw = by.get(MemoryPolicy.SLIDING_WINDOW.value)
    lay = by.get(MemoryPolicy.LAYERED.value)
    nm = by.get(MemoryPolicy.NO_MEMORY.value)
    sws = by.get(MemoryPolicy.SLIDING_WINDOW_SUMMARY.value)

    # 结论一律**只用照抄型那半**说话。换算型那半不是对比，见 report 里的说明。
    if nm:
        n += 1
        out.append(
            f"{n}. **基线先摆出来：`{nm.policy}` 的照抄型保持率是 "
            f"{nm.retention_quotable:.3f}（{nm.n_quotable_passed}/{nm.n_quotable}）。**\n"
            "   ——它只带最后一轮的原文，所以它记住的永远是「刚刚那句话」。"
            "**它不是「没用」，它是标尺**：后面每个策略的收益，都要先减掉这个数。"
        )

    if sw and lay:
        n += 1
        delta_tok = sw.mean_tokens - lay.mean_tokens
        pct = (delta_tok / lay.mean_tokens * 100) if lay.mean_tokens else 0.0
        longer = delta_tok > 0
        # 曲线形状比总分更能说明问题：滑动窗口是**硬边界**，一过窗口全没；
        # layered 是**平的**，因为槽位层不吃窗口那一套。
        def curve(r: PolicyResult) -> str:
            return "、".join(f"{k} 轮前 {v:.2f}" for k, v in r.by_gap.items())

        out.append(
            f"{n}. **两个方案花的 token 差不多（`{sw.policy}` {sw.mean_tokens:.0f} vs "
            f"`{lay.policy}` {lay.mean_tokens:.0f}，"
            f"{'多' if longer else '少'} {abs(pct):.0f}%），"
            f"但**曲线的形状完全不一样**。**\n"
            f"   - `{sw.policy}`：{curve(sw)}\n"
            f"   - `{lay.policy}`：{curve(lay)}\n"
            "   滑动窗口是**硬边界**——窗口盖得住的记得住，盖不住的一律归零，"
            "没有中间地带。这就是它危险的地方：**它在第 10 轮还是满分，"
            "第 11 轮就变成 0 分，而用户不会知道正在跨过那条线。**\n"
            "   `layered` 是平的，因为它把「必须记住的东西」从窗口里搬出来、"
            "放进一个不吃压缩的槽位表——**花同样的钱，把硬边界抹掉了。**"
        )

    if sws and sw:
        n += 1
        delta = sws.retention_quotable - sw.retention_quotable
        out.append(
            f"{n}. **摘要（`{sws.policy}` = {sws.retention_quotable:.3f}）对比纯滑动窗口"
            f"（`{sw.policy}` = {sw.retention_quotable:.3f}）："
            f"{'高' if delta >= 0 else '低'} {abs(delta):.3f}。**\n"
            "   这一栏的数字**是反直觉的**，得说清楚为什么：两者**总预算一样**，"
            "区别在怎么花——`sliding_window` 把预算全砸在原文上（窗口被拉到 10 轮），"
            "`sliding_window_summary` 保持 6 轮原文 + 拿剩下的预算放摘要。\n"
            "   而这里的摘要是**规则拼出来的**：它从一段历史里挑几句用户原话串起来。"
            f"挑中的句子就是全部——没被挑中的，连数字带上下文一起没了。"
            "**摘要能把「聊过什么」带过去，带不过去精确的值。"
            "这就是为什么槽位层必须单独存在：精确的事实绝不交给有损压缩承载。**"
        )

    if lay:
        n += 1
        out.append(
            f"{n}. **`{lay.policy}` 在换算型上是 {lay.retention_computed:.3f}。"
            "别把这行当成成绩。**\n"
            "   换算型的事实（「三年零一个月」→ 工作年限 37、「没提前通知」→ False）"
            "**在原话里根本找不到那个写法**，只有槽位层能给出。"
            "所以对它们来说「四种策略对比」是假命题——"
            "**1.000 的意思是「槽位层正常工作」，不是「效果好」。**"
        )

    if not out:
        out.append("（策略不全，凑不出结论；用完整的四个策略重跑。）")
    return "\n\n".join(out)


def load_pricing(path: Path | None = None) -> float:
    """读 `configs/pricing.yaml` 里 `deepseek-flash` 的**输入**单价。

    只取输入价，因为这里量的全是**提示词**——记忆层的产出就是一段要发出去的文字。
    **单价不写死在代码里**，理由见 `configs/pricing.yaml` 顶部：
    报价会变，而"改了个代码里的常量"这种事在 review 里根本看不出来。
    """
    p = path or (PROJECT_ROOT / "configs" / "pricing.yaml")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return float(data["models"]["deepseek-flash"]["input"])


# --------------------------------------------------------------------------- #
# 命令行入口
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="长对话记忆消融（F22）")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "memory_ablation.md"))
    parser.add_argument("--json", default=str(PROJECT_ROOT / "reports" / "memory_ablation.json"))
    parser.add_argument(
        "--policies",
        nargs="*",
        default=[p.value for p in MemoryPolicy],
        help="只跑其中几个策略（调试用）。**出正式报告时别省**——"
        "四个策略齐全，结论那一段才拼得完整。",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="跳过数据集体检。⚠️ 只在数据已确认干净、想省几秒时用；"
        "体检是防止「评测集写错了却被当成实现 bug」的唯一防线。",
    )
    args = parser.parse_args(argv)

    sessions = load_sessions()

    if not args.no_audit:
        problems = audit_dataset(sessions)
        if problems:
            print("⚠️ 评测集本身有问题，先修数据再看结果：")
            for p in problems:
                print("   -", p)
            return 2

    price_in = load_pricing()

    results: list[PolicyResult] = []
    for name in args.policies:
        r = run_policy(sessions, MemoryPolicy(name), price_in=price_in)
        results.append(r)
        print(f"{name:24s} 保持率 {r.retention:.3f}  通过 {r.n_passed}/{r.n_probes}")

    Path(args.out).write_text(render_report(results, sessions, price_in), encoding="utf-8")
    Path(args.json).write_text(
        json.dumps(
            {"price_in": price_in, "policies": [r.model_dump() for r in results]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n已写出 {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
