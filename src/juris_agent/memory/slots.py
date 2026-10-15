"""结构化槽位（F19）。★ **本文件是整个记忆层的地基。**

## 先说清楚它解决什么问题（面试就爱从这个问）

一个 20 轮的真实咨询对话是这样的：

```
第 1 轮   用户：我 2021 年 3 月入职，月薪 12000，现在公司要裁我
第 7 轮   用户：（聊了一堆加班的事）
第 14 轮  用户：那我到底能拿多少？
第 20 轮  用户：那要是我不签呢？
```

到第 14 轮时，"12000" 这个数已经离得很远了。如果记忆层靠**摘要**来承载历史，
摘要里会写成"用户月薪较高"或者"月薪一万出头"——**"一万出头"和"12000"在
法律计算里是两个不同的答案**（差 4000 块）。

更糟的是这种错**不会报错**：模型拿到"一万出头"照样能算出个数，用户也看不出
哪里不对。摘要的失真是一种**静默的**失真。

> 所以核心结论是：**精确的事实不能用有损压缩承载。**
> 摘要负责"知道发生过什么"，槽位负责"知道确切的数是多少"。

## 槽位是什么

就是一张**结构化的、不参与摘要压缩的**小表：

```json
{"monthly_wage": {"value": 12000, "turn": 1, "quote": "月薪 12000"},
 "months_of_service": {"value": 36, "turn": 1, "quote": "2021 年 3 月入职"}}
```

它有三条硬性质，每条都是踩出来的：

### 1. 只抽"会改变计算结果"的字段

槽位不是 CRM（客户管理系统），**不抽姓名、公司名、岗位**——那些字段再准，
也算不出多一分钱。判断标准很硬：**这个字段会不会作为参数传进某个计算工具？**
会 → 抽；不会 → 不抽。

好处是这张表天然有界（十几个字段，不是几十个），而且**每个槽位都能说清
"它为什么在"**（对应哪个法条、影响哪一步计算）。

### 2. 每个值都带出处（`turn` + `quote`）

只有值没有出处，出了问题没法查。用户说"我没说过月薪 12000"，你得能回一句
"第 1 轮您说的是『月薪 12000，现在公司要裁我』"。**可溯源不只是对法条的要求，
对记忆里的数字同样适用。**

### 3. 模糊值永远不会覆盖精确值（`precision`）

这是本文件唯一一处"有点绕"的逻辑，也是最容易做错的地方：

```
第 1 轮  用户：月薪 12000        → 精确（precision=2）
第 9 轮  用户：我那点工资，一万出头吧   → 模糊（precision=1）
```

朴素的"后写的覆盖先写的"会让第 9 轮把 12000 冲成"一万出头"。**这是错的**——
第 9 轮那句话不是新信息，是**同一件事的模糊复述**。

规则：**精确值只能被精确值覆盖；模糊值只在"这个槽位还是空的"时才写进去。**

那什么时候精确值该被覆盖？用户改口了的时候（"记错了，是 15000"）——
那就是一个新的精确值，正常覆盖。**"精度不够就不许覆盖"这条规则不会挡住合法更新，
只会挡住模糊复述。**

## 槽位怎么填

**规则优先，模型兜底**（和路由、引用校验一个套路）：

- 正则能抽的（金额、日期、年限、情形关键词）**一律用正则**——免费、微秒级、
  100% 可复现，而且**能精确说出是从哪个片段抽出来的**（`quote` 直接就是命中的那段）。
- 正则抽不到的交给模型（可选）。**默认关着**——理由和检索改写一样：
  一次模型调用换来的召回，抵不上它引入的延迟和不可复现。

## 收益在哪（这是这块最该讲的一句话）

抽出来的槽位不只是"给模型看一眼"，它还能**直接预填工具参数**（见
`fill_args()`）。老话是"第 20 轮把第 1 轮的金额丢了"，实际表现是
**"第 20 轮模型又问了一遍月薪多少"**——用户答过三遍了，体验极差。
有了槽位，工具参数里 `avg_monthly_wage=12000` 是**程序填的**，
根本不经过模型，也就没机会丢。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from juris_agent.tools.builtin import (
    LimitationArgs,
    OvertimeArgs,
    ProbationArgs,
    SeveranceArgs,
)

# ============================================================================
# 槽位表
# ============================================================================


@dataclass(frozen=True)
class SlotSpec:
    """一个槽位的定义。

    `targets` 是**这个槽位最终会流向哪里**：一串 `(工具名, 参数字段名)`。
    它不是装饰——`tests/test_memory.py` 会拿它去核对字段真的存在，
    拼错了会直接红。**这就是"槽位表"和"工具参数"之间不会漂移的机制。**
    """

    name: str
    label: str
    """中文名。会出现在提示词里，所以写成用户看得懂的说法。"""
    kind: str
    """`money / int / float / date / enum / bool / str / json`。
    决定正则怎么抽、怎么转类型，**也是和工具参数类型对表的依据**——
    `tests/test_memory.py` 里有一条测试拿它逐个核对工具参数模型的类型注解，
    类型对不上会让工具调用直接校验失败。"""
    why: str
    """为什么需要这个槽位——**必须写清它影响哪一步计算**，
    不然半年后没人敢删它。"""
    targets: tuple[tuple[str, str], ...] = ()
    default_precision: int = 2
    """2 = 用户明确说了数字；1 = 用户说得模糊（"一万出头"）；0 = 猜的。"""
    required: bool = True
    """`required=True` 表示"没有它这个工具算不出结果"，会被 `missing_for()` 报出来。
    像 `notice_given`（有没有提前 30 天通知）这种**用户不提就不能猜**的字段，
    `required=False`——它是可选参数，缺了按法条默认走，不追问。"""
    internal: bool = False
    """内部槽位：**给派生计算当原料，不进提示词、不进工具参数。**

    它和"普通槽位"的区别只有一处，但那处很重要：`prompt_block()` 不渲染它。
    为什么要这样分——因为提示词是给模型看的，模型看『入职日期 2021-03-01』
    只会多占几个 token，还可能拿它去瞎算；它真正该看到的是已经算好的
    『工作年限 55 个月』。**给模型的应该是结论，不是原料。**"""


SLOT_SPECS: dict[str, SlotSpec] = {
    s.name: s
    for s in (
        SlotSpec(
            name="monthly_wage",
            label="月工资",
            kind="money",
            why="经济补偿的基数（第47条）、加班费的基数、试用期工资下限的对照值。"
            "**这是全项目最重要的一个数**——它错了后面全错。",
            targets=(
                ("calc_severance", "avg_monthly_wage"),
                ("calc_overtime", "monthly_wage"),
                ("check_probation", "agreed_contract_wage"),
            ),
        ),
        SlotSpec(
            name="months_of_service",
            label="工作年限（月）",
            kind="int",
            why="经济补偿的年限系数（第47条：每满一年一个月）。存月不存年，"
            "是因为**法条的分档是按月的**（『六个月以上不满一年按一年』），"
            "只存『3 年』会把『3 年 5 个月』这个信息丢掉。",
            targets=(("calc_severance", "months_of_service"),),
        ),
        SlotSpec(
            name="termination_cause",
            label="解除情形",
            kind="enum",
            why="决定**该不该给补偿**（第46条列了七类），不是决定给多少。"
            "顺序错了会给出方向性错误的结论——比数值错严重。",
            targets=(("calc_severance", "cause"),),
        ),
        SlotSpec(
            name="local_avg_monthly_wage",
            label="当地上年度职工月平均工资",
            kind="money",
            why="判断有没有触发『三倍封顶 + 12 年封顶』（第47条第2款）。"
            "不给的话高薪场景会算高。",
            targets=(("calc_severance", "local_avg_monthly_wage"),),
            required=False,
        ),
        SlotSpec(
            name="notice_given",
            label="是否提前 30 日通知",
            kind="bool",
            why="第40条情形下没提前通知要加一个月代通知金（第40条 + 实施条例第20条）。"
            "**用户没提这件事时必须留空，不能猜**——猜『没通知』会凭空多算一个月工资。",
            targets=(("calc_severance", "notice_given"),),
            required=False,
        ),
        SlotSpec(
            name="last_month_wage",
            label="上一个月工资",
            kind="money",
            why="代通知金的基数（实施条例第20条按『上一个月』，不是 12 个月平均）。"
            "只在 `notice_given=false` 时才会用到。",
            targets=(("calc_severance", "last_month_wage"),),
            required=False,
        ),
        SlotSpec(
            name="contract_term_months",
            label="合同期限（月）",
            kind="int",
            why="试用期上限的直接判据（第19条：三个月以上不满一年→试用期≤1个月，"
            "一年以上不满三年→≤2个月，三年以上→≤6个月）。",
            targets=(("check_probation", "term_months"),),
        ),
        SlotSpec(
            name="probation_months",
            label="约定试用期（月）",
            kind="int",
            why="和合同期限查表对照，超过上限就是违法约定，还要算第83条赔偿金。",
            targets=(("check_probation", "agreed_probation_months"),),
        ),
        SlotSpec(
            name="know_date",
            label="知道权利被侵害之日",
            kind="date",
            why="仲裁时效一年的起算点（调解仲裁法第27条）。"
            "这个日期差一天，结论就从『来得及』变成『过期了』。",
            targets=(("calc_limitation", "know_date"),),
        ),
        SlotSpec(
            name="relationship_ended",
            label="劳动关系终止之日",
            kind="date",
            why="劳动关系终止的，仲裁时效从终止之日起算一年（第27条第4款）；"
            "**仍在职的话不受一年限制**——这是最容易答错的一类问题。",
            targets=(("calc_limitation", "relationship_ended"),),
            required=False,
        ),
        SlotSpec(
            name="is_wage_arrears",
            label="是否拖欠劳动报酬",
            kind="bool",
            why="拖欠劳动报酬的争议不受一年仲裁时效限制（第27条第4款）。"
            "这一条能直接推翻『过期了』的结论。",
            targets=(("calc_limitation", "is_wage_arrears"),),
            required=False,
        ),
        SlotSpec(
            name="overtime_kind",
            label="加班类型",
            kind="enum",
            why="倍率由它决定：工作日 150% / 休息日 200% / 法定节假日 300%。"
            "**而且休息日可以补休替代、法定节假日不可以**——差的不只是倍数。",
            targets=(("calc_overtime", "kind"),),
        ),
        SlotSpec(
            name="overtime_hours",
            label="加班小时数",
            kind="float",
            why="工作日延时加班按小时算（150%）。"
            "**小时数不能和天数混**——折算系数是 21.75 天/月、8 小时/天，混了就错。",
            targets=(("calc_overtime", "hours"),),
            required=False,
        ),
        SlotSpec(
            name="overtime_days",
            label="加班天数",
            kind="float",
            why="休息日 / 法定节假日加班按天算（200% / 300%）。",
            targets=(("calc_overtime", "days"),),
            required=False,
        ),
        SlotSpec(
            name="city",
            label="城市",
            kind="str",
            why="**唯一一个不直接进工具参数的槽位。**它的用途是去查当地社平工资，"
            "从而填上 `local_avg_monthly_wage`。它是槽位表里少数的『派生来源』，"
            "留着是因为『三倍封顶』这个问题在真实咨询里出现频率极高。",
            targets=(),
            required=False,
        ),
        # ---- 下面两个是**内部槽位**：给派生计算当原料，不直接进提示词 ----
        SlotSpec(
            name="work_start",
            label="入职日期",
            kind="date",
            why="用户给了入职日期、但没说干了多久时，用『今天 - 入职日期』算出工作月数。"
            "**它是槽位表里唯一一个『会被消费掉』的槽位**——算完 `months_of_service` 之后，"
            "它自己就不再出现在提示词里了（用户看『入职日期 2021-03-01』没意义，"
            "他要知道的是『4 年 7 个月』）。",
            targets=(),
            required=False,
            internal=True,
        ),
        SlotSpec(
            name="other_amounts",
            label="句中的其它金额",
            kind="json",
            why="一句话里出现两个以上金额时的**第二个及以后**（『月薪 10000，年终奖 3 万』）。"
            "抽出来是为了**避免错用**——如果它被丢掉，模型就看不到『年终奖 3 万』这回事；"
            "如果它被当成月薪，答案直接错。留在这里给调试用："
            "『为什么算出来基数不对』的第一眼看的就是它。",
            targets=(),
            required=False,
            internal=True,
        ),
    )
}

# 按"哪个工具要哪些槽位"反查，由 targets 自动推出来。
# **不要手写这张表**——手写就会和 targets 漂移，而漂移的表现是
# "某个槽位明明填了，但没送去工具"，静默且难查。
TOOL_SLOTS: dict[str, list[str]] = {}
for _spec in SLOT_SPECS.values():
    for _tool, _fld in _spec.targets:
        TOOL_SLOTS.setdefault(_tool, []).append(_spec.name)


# ============================================================================
# 值 + 出处
# ============================================================================


class SlotValue(BaseModel):
    """槽位里的一个值。**值只是一半，另一半是"它是怎么来的"。**"""

    value: Any
    turn: int
    """第几轮说的。**注意轮次从 1 开始**，跟人说话的习惯一致（"你第 1 轮说过"）。"""
    quote: str = ""
    """命中的原文片段。用户说"我没说过"的时候，这行字就是证据。"""
    source: str = "rule"
    """rule=正则抽的 / llm=模型抽的 / manual=程序填的（比如从城市查出来的社平工资）。
    `manual` 的精度默认最高——它是程序算的，不是从人话里猜的。"""
    precision: int = 2
    """2=明确数字，1=模糊表述（"一万出头"），0=猜的。
    冲突消解只看这一个数，见模块 docstring 的第 3 条。"""

    def __str__(self) -> str:
        return f"{self.value}（第{self.turn}轮）"


class Slots(BaseModel):
    """槽位表本体。`values` 用 dict 而不是一堆字段——因为要能遍历、能按名字查。"""

    values: dict[str, SlotValue] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)
    """被拒绝的更新（模糊值想覆盖精确值）。**留着是为了能回答
    "为什么第 9 轮说的『一万出头』没生效"**——不然这看起来就像个 bug。"""

    # -- 读 --------------------------------------------------------------- #

    def get(self, name: str, default: Any = None) -> Any:
        v = self.values.get(name)
        return v.value if v is not None else default

    def entry(self, name: str) -> SlotValue | None:
        return self.values.get(name)

    def has(self, name: str) -> bool:
        return name in self.values

    def to_dict(self, *, include_internal: bool = False) -> dict[str, Any]:
        """只要值，不要出处。给工具预填参数用。

        **默认排除内部槽位**——它们不是给模型/工具看的，混进去只会多传一个
        没人会用的字段。要调试时显式 `include_internal=True`。
        """
        return {
            k: v.value
            for k, v in self.values.items()
            if include_internal or not SLOT_SPECS[k].internal
        }

    def provenance(self) -> dict[str, str]:
        """值 → 出处的一行说明。给 trace 和前端用（"月薪 12000（第1轮：『月薪 12000』）"）。"""
        return {
            k: f"{v.value}（第{v.turn}轮：『{v.quote}』）" if v.quote else f"{v.value}（第{v.turn}轮）"
            for k, v in self.values.items()
        }

    # -- 写 --------------------------------------------------------------- #

    def put(
        self,
        name: str,
        value: Any,
        *,
        turn: int,
        quote: str = "",
        source: str = "rule",
        precision: int | None = None,
    ) -> bool:
        """写入一个槽位。**返回 True 表示真的写进去了**（调用方/测试靠它判断）。

        ## 冲突消解只有三条规则，按顺序判

        1. **槽位是空的** → 写。哪怕值是模糊的，也比没有强。
        2. **新值比旧值更精确（或一样精确）** → 覆盖。用户改口了就走这条。
        3. **新值比旧值模糊** → **拒绝**，但记进 `history`。

        ⚠️ 规则 2 里"一样精确就覆盖"是刻意的：用户第 1 轮说 12000、
        第 12 轮说 15000，两个都是明确数字，**必须认新的**——不能说
        "已经有值了就不许改"，那用户就永远改不过来。
        """
        spec = SLOT_SPECS.get(name)
        if spec is None:
            raise KeyError(f"未定义的槽位 {name!r}；可用：{sorted(SLOT_SPECS)}")
        if value is None:
            return False
        new_p = spec.default_precision if precision is None else precision

        old = self.values.get(name)
        if old is not None and new_p < old.precision:
            # 模糊覆盖精确 → 拒绝。**这是本模块最该被问的一条逻辑。**
            self.history.append(
                {
                    "slot": name,
                    "rejected": value,
                    "rejected_precision": new_p,
                    "kept": old.value,
                    "kept_precision": old.precision,
                    "turn": turn,
                    "reason": "新值是模糊表述，不能覆盖已有的精确值",
                }
            )
            return False

        self.values[name] = SlotValue(
            value=value, turn=turn, quote=quote, source=source, precision=new_p
        )
        return True

    # -- 给工具用 ---------------------------------------------------------- #

    def fill_args(self, tool: str) -> dict[str, Any]:
        """**把槽位里的值直接变成某个工具的参数。**

        这是"第 20 轮丢信息"这件事的正解：参数是**程序填的，不经过模型**，
        所以模型没有机会"忘了"。模型只需要补槽位里没有的那几个。

        ⚠️ 只填槽位表里定义的、且这个工具真的要的字段。不做任何猜测、
        不做类型转换——类型转换归 pydantic 校验（`SeveranceArgs` 那一层），
        在这一层转会让"传了个字符串给 int 字段"这种错**静默通过**。
        """
        return {
            fld: self.values[name].value
            for name in TOOL_SLOTS.get(tool, [])
            for (t, fld) in SLOT_SPECS[name].targets
            if t == tool and name in self.values
        }

    def missing_for(self, tool: str) -> list[str]:
        """这个工具**算不出结果**时，还缺哪些槽位。给"主动追问"用。

        只报 `required=True` 的。像 `notice_given`（有没有提前 30 天通知）
        这种字段，用户没提就按法条默认走，**追问它是打扰**——
        用户会觉得"我哪知道要不要通知"。
        """
        return [
            n
            for n in TOOL_SLOTS.get(tool, [])
            if SLOT_SPECS[n].required and n not in self.values
        ]

    def missing_labels(self, tool: str) -> list[str]:
        """缺的那几个槽位的中文名。**只有这个能拿去问用户**——
        不能让用户看 `months_of_service` 这种字段名。"""
        return [SLOT_SPECS[n].label for n in self.missing_for(tool)]

    # -- 渲染 -------------------------------------------------------------- #

    def prompt_block(self) -> str:
        """渲染成给模型看的一段话。

        两条规则：
        - **空槽位不出现**。写"月薪：未知"只会让模型倾向于追问一个
          用户根本没提过的东西。
        - **内部槽位不出现**。见 `SlotSpec.internal` 的说明：
          该给模型的是结论（工作年限），不是原料（入职日期）。
        """
        shown = {
            k: v for k, v in self.values.items() if not SLOT_SPECS[k].internal
        }
        if not shown:
            return ""
        lines = ["【已核实的事实（用户明确说过的，请直接使用，不要再问一遍）】"]
        for name, v in shown.items():
            spec = SLOT_SPECS[name]
            mark = "" if v.precision >= 2 else "（用户表述模糊，注意核对）"
            lines.append(f"- {spec.label}：{v.value}{mark}")
        return "\n".join(lines)


# ============================================================================
# 抽取：正则优先
# ============================================================================

# 中文数字。**只处理到"万"**——真有人写"十万零三千"的话正则抽不出来，
# 那种交给模型兜底路径，不值得为它写一个中文数字解析器。
_CN_DIGIT = {
    "零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}

_MONEY_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>万|w|W|k|K|千)?\s*(?:元|块钱|块)?"
)
_CN_YUAN_RE = re.compile(
    r"(?P<cn>[一二两三四五六七八九十]+)\s*(?P<unit>万|千)?"
    r"(?P<tail>[一二两三四五六七八九十百千零]*)\s*(?:元|块钱|块)?"
)
_MONTHS_RE = re.compile(r"(?P<num>\d+)\s*(?:个)?月")
_YEARS_RE = re.compile(r"(?P<num>\d+)\s*年")
_CN_YEARS_RE = re.compile(r"(?P<cn>[一二两三四五六七八九十半]+)\s*年")
_DATE_RE = re.compile(r"(?P<y>\d{4})\s*[-/年.]\s*(?P<m>\d{1,2})(?:\s*[-/月.]\s*(?P<d>\d{1,2}))?")
# 中文数字 + 时间单位。**只用来"占坑"**（让金额正则跳过），本身不产出槽位——
# 年限归 `_extract_durations` 管，天/小时归加班那一段管。
_CN_TIME_RE = re.compile(r"[一二两三四五六七八九十半]+\s*(?:个)?(?:月|年|天|小时|周|星期|礼拜)")

# "劳动关系已经结束"的说法。**要配上一个日期才会真的写进槽位**，
# 理由见 `SlotExtractor.extract` 里那一段注释。
_ENDED_CUES = (
    "离职", "走了", "辞退", "开除", "解除", "终止", "不干了", "离开公司", "被裁",
)


def cn_to_int(s: str) -> int | None:
    """中文数字 → int。"三" → 3，"十二" → 12，"三十" → 30。

    ## 三条规则（中文数字就这点东西）

    1. 遇到**数字字符**先记着（`cur`），不急着加进总数。
    2. 遇到**单位**（十/百/千）才结算：`(cur or 1) * 单位`。
       `cur or 1` 是为了"十五"这种省略写法——"十"前面没有数字，按 1 算。
    3. 结尾把剩下的 `cur` 加上："三十五" = 30 + 5。

    ⚠️ **"一万二" 这类省略式不在这个函数里处理**（它是 12000 不是 102，
    省略的那一位是"千"）。那种要**结合量级词**才知道省略的是哪一位，
    所以放在 `_money_from()` 里——那里才看得到"万"这个单位。
    **函数边界按"需要什么信息才能算对"来划，不按"看起来像一类"来划。**
    """
    if not s:
        return None
    total, cur = 0, 0
    for ch in s:
        if ch in _CN_DIGIT:
            cur = _CN_DIGIT[ch]
        elif ch in _CN_UNIT:
            total += (cur or 1) * _CN_UNIT[ch]
            cur = 0
        else:
            return None
    return total + cur


# "这个数字是工资吗"的上下文线索。**只看匹配位置**前面 6 个字内有没有这些词。
#
# ⚠️ **没有这条线索，"裸数字"根本不该被当成金额。** 踩过的坑：
# `_MONEY_RE` 的存在是为了让"月薪 12000"这种没写"元"的说法能被抽到，
# 但它的副产品是**任何数字都能匹配**——"公司有 200 人"会被抽成月薪 200。
# 这种错抽的后果比漏抽严重得多：漏抽只是模型多问一句，
# 错抽会让计算基数**静默地**变成一个用户从没提过的数。
_WAGE_CUES = (
    "月薪", "工资", "薪水", "薪资", "薪酬", "收入", "到手", "税前", "月收入",
    "月均", "平均工资", "底薪", "基本工资", "上个月", "每个月", "发我", "发我",
    "赔", "补偿", "给我", "拿到", "月入",
    # 口语里最常出现的几个说法。**注意这些都是"钱在句子里的角色"**，
    # 不是"钱的单位"——"挣""欠""一个月"都能确定前后那个数字是工钱。
    "挣", "欠", "一个月", "日薪", "时薪", "年薪", "到手",
)

# "我刚才说错了"的说法。**它也是一种工资线索**，而且是很强的一种：
#
# 踩过的坑：用户第 1 轮说"月薪12000"，第 2 轮说"不对，是15000"。
# "15000"是个裸数字，前后都没有"月薪/工资"这类词，按裸数字的规矩
# 只能判成**精度 0**（不敢认）——于是**用户的更正被整条丢掉了**，
# 后面所有计算都还在用 12000。而"改口"恰恰是长对话里最常见的一件事。
#
# 判据：句子里有"不对/应该是"这类词，**且这个数字看着就像一笔工钱**
# （四位数以上）。"改口"这个动作本身就说明"这个数字指的是刚才那个东西"，
# 所以它和"月薪"一样能点名。
#
# ⚠️ 刻意**不收"不是""其实"**：它们太常见，而且常常是否定句
# （"不是我不想签"），收进来会把很多普通数字误判成金额。
_CORRECTION_CUES = ("不对", "应该是", "改成", "更正", "说错", "记错", "打错")


def _money_from(match: re.Match[str], text: str) -> tuple[float, int, str, bool]:
    """把一次金额匹配转成 `(元, 精度, 原文, 有没有工资线索)`。
    **精度 0 表示"这不是工资"，调用方要丢掉它。**

    ## 精度怎么判

    | 写法 | 值 | 精度 | 为什么 |
    |---|---|---|---|
    | `12000元` / `1.2万` / `12k` | 12000 | 2 | 有明确数字 |
    | `12000` + 附近有"月薪" | 12000 | 2 | 数字明确，上下文也明确 |
    | `12000` 但没有上下文线索 | — | **0** | **不敢认**，见 `_WAGE_CUES` |
    | `一万` / `三万` | 10000/30000 | 1 | 只有一个量级，用户是估的 |
    | `五千` | 5000 | 1 | 连量级词都没有，更不敢认 |

    **"宁可漏抽也不要错抽"** 是这一段的唯一原则：漏抽的下场是模型多问一句
    （用户顶多嫌啰嗦），错抽的下场是算出一个错的金额而且**没人看得出来**。
    """
    raw = match.group(0)
    unit = match.group("unit") or ""
    ctx = text[max(0, match.start() - 6) : match.start()]
    cued = any(c in ctx for c in _WAGE_CUES)
    # "改口"也算点名，但门槛更高——见 `_CORRECTION_CUES`。
    corrected = any(c in ctx for c in _CORRECTION_CUES)

    if match.groupdict().get("num"):
        num = float(match.group("num"))
        if unit in ("万", "w", "W"):
            return num * 10000, 2, raw, cued   # "1.2万" —— 用户说得很准
        if unit in ("k", "K", "千"):
            return num * 1000, 2, raw, cued
        if "元" in raw or "块" in raw:
            return num, 2, raw, cued
        # 裸数字：只有上下文线索能救它
        if cued and num >= 100:
            return num, 2, raw, True
        if corrected and num >= 1000:
            # "不对，是15000"。门槛取 1000 而不是 100：
            # "不对，公司有200人" 里的 200 落在门槛下面，不会被误认。
            return num, 2, raw, True
        return num, 0, raw, cued

    cn = match.group("cn")
    unit = match.groupdict().get("unit") or ""
    tail = match.groupdict().get("tail") or ""
    # 量级词带来的系数。"万" 是最常见的一个。
    factor = 10000 if unit == "万" else (1000 if unit == "千" else 1)

    # ⚠️ **中文数字如果没有"元/块"，也没有量级词，那它多半根本不是钱。**
    # "公司有一千多人"里的"一千"、任何一句话里的"一个"，都会匹配到这个正则。
    # 所以这里的规矩和上面裸数字那条**完全一样**：没有工资线索就返回精度 0，
    # 让调用方丢掉它。**两边共用同一条原则，不搞两套。**
    explicit_yuan = "元" in raw or "块" in raw

    # ⚠️ **"一万二" 这种省略式**：口语里它是 12000，省略掉的那一位是"千"。
    #    判据：匹配里带了量级词（万/千），**且紧随其后还有一个数字字**。
    #    为什么"十二"不算：它没有量级词，12 = 十 + 二 本来就是
    #    `cn_to_int` 的正常算法，不需要特判。
    #    为什么这条必须单独处理、不能让正则分两次匹配出来：
    #    "月薪一万二" 拆成"一万"和"二"两次匹配之后，**"二"会拿到自己的
    #    上下文线索**（往前 6 个字里有"月薪"），于是被当成第二个金额
    #    记进 `other_amounts`——用户只说了 12000，账上却多出一笔 2 元。
    #    这是实测出来的坑，所以"一万二"必须在正则层面就吃成一个整体。
    if tail and unit:
        rest = cn_to_int(tail) or 0
        if not any(u in tail for u in "十百千万零"):
            # 尾巴上是个**光板数字**（"一万二"的"二"）→ 它说的是低一位的量级。
            # 万后面的光板数字是"千"，千后面的光板数字是"百"。
            rest *= 1000 if unit == "万" else 100
        val = (cn_to_int(cn) or 0) * factor + rest
        if explicit_yuan or cued:
            return float(val), 2, raw, True
        return float(val), 0, raw, False

    val = (cn_to_int(cn) or 0) * factor
    # 中文数字**一律按模糊（精度 1）处理**，哪怕有量级词。
    # 理由：用户说"三万多"和"三万"是同一个写法，正则分不出来；
    # 而"三万"在口语里本来就常常是估值（真给 30000 的人会说"三万整"或写数字）。
    # 精度低一点的后果只是"不许它覆盖已有的精确值"——这正是我们想要的。
    if explicit_yuan or cued:
        # "我那点工资，一万出头吧" 走的就是这条路：精度 1，
        # 能用（能算），但**不许它盖掉用户后面说的精确数字**。
        return float(val), 1, raw, True
    return float(val), 0, raw, False


class SlotExtractor:
    """从一句话里抽槽位。**纯函数式、无状态**。

    因为无状态，它**不需要被实例化**——`SlotStore` 里直接用类方法调，
    不持有实例。少一个对象就少一处"这个对象到底带没带配置"的疑问。

    ## 抽取顺序有讲究

    1. **先抽日期**。因为"2021 年 3 月入职"里的 "2021" 会被金额正则误当成数字，
       而日期抽走之后，剩下的文本里就不会再有这个数。
       （实现上是先记下日期 span，再抽金额时跳过它们。）
    2. 再抽年限（`3 年` / `36 个月`）。
    3. 再抽金额——**跳过已经被日期/年限吃掉的数字**。
    4. 最后抽情形、布尔这些关键词。

    ## 为什么金额要放在年限后面

    "我干了 3 年月薪 10000" —— 如果金额先抽，那个 `3` 可能被当成 3 块钱
    （没有单位、小于 100）而丢弃；但 "3 年" 被吃掉之后，`10000` 是唯一的大数，
    判断起来干净得多。**顺序本身就是一种消歧手段。**
    """

    CAUSE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("illegal_dismissal", ("违法解除", "违法辞退", "没给理由就辞", "没有理由开除")),
        ("fault_dismissal", ("严重违纪", "严重违反", "失职", "被追究刑事责任",
                             "试用期不符合录用条件", "不符合录用条件")),
        ("employee_forced", ("被迫解除", "被迫离职", "不缴社保", "不给交社保",
                             "拖欠工资", "不缴社保", "强迫")),
        ("mutual_employer_proposed", ("公司提出协商", "单位提出的协商", "协商解除",
                                      "公司跟我协商")),
        ("mutual_employee_proposed", ("我提出协商", "我主动协商")),
        ("economic_layoff", ("裁员", "经济性裁员", "优化", "结构性调整", "裁撤")),
        ("no_fault_dismissal", ("无过失性辞退", "不胜任", "医疗期满", "客观情况",
                                "岗位取消", "公司解散")),
        ("fixed_term_expiry", ("合同到期", "合同期满", "不续签", "到期终止")),
        ("employee_resignation", ("主动辞职", "自己辞职", "我辞职", "提出离职")),
    )

    OVERTIME_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("statutory_holiday", ("法定节假日", "春节加班", "国庆加班", "节假日加班")),
        ("rest_day", ("休息日", "周末加班", "双休日加班", "周六加班", "周日加班")),
        ("extended", ("工作日加班", "平时加班", "延时加班", "每天加班", "晚上加班")),
    )

    @classmethod
    def _extract_durations(
        cls, text: str, *, turn: int, taken: list[tuple[int, int]]
    ) -> dict[str, SlotValue]:
        """抽"时长"类槽位（已工作时长 / 合同期限）。**要分两步，少一步都错。**

        ## 第一步：把挨着的数字先拼成"一个时长"

        "3年零6个月"是**一个** 42 个月，不是 36 和 6 两个数，
        更不是"3 年"和"6 个月"两件事。所以同一个位置附近匹配到的几个数
        要先合并成一组。

        合并的判据是"**中间只隔了几个虚字**"（不超过 3 个字、且没有逗号句号）。
        ⚠️ **绝不能用"整句求和"代替**：`"我干了3年，签了3年合同"` 里
        两个"3年"是**两件完全不同的事**，求和会得到一个 72 个月的数——
        用户从来没提过这个数，而且他看不出来。

        ## 第二步：每一组按**它自己周围的字**判归属

        "3年"这句话里到底指什么，看的是它前后那几个字：

        - 后面紧跟着「合同 / 期限 / 协议」→ **合同期限**（`contract_term_months`）
        - 前面几个字里有「签了 / 签约 / 约定 / 合同」→ 也是**合同期限**
        - 都不是 → **已工作时长**（`months_of_service`）

        ⚠️ **判据的窗口必须小（前 5 字、后 4 字），不能用整句。**
        整句里只要有"合同"两个字，就会把句子里**所有**年限都算成合同期限，
        `"我干了3年，签了3年合同"` 于是丢掉了"干了 3 年"这个数——
        而"干了多久"恰恰是算补偿金的第一参数。
        踩过这个坑，所以这里写死了窗口宽度。

        ## 一个上限保护

        时长超过 50 年（600 个月）一律丢掉。理由很实际：`"我2021年来的"`
        这种句子里，`2021年` 会被年份正则匹配到，算出 24252 个月。
        **这不是"年限"，这是把年份当成了时长**，属于必须挡掉的错抽。
        """

        def clash(a: int, b: int) -> bool:
            """`(a, b)` 和已用区间冲突吗。"""
            return any(a < d and c < b for c, d in taken)

        def is_probation(a: int) -> bool:
            """这个数字是不是在说**试用期**那几个字。

            ⚠️ 踩过的坑：`"试用期3个月"` 里的 "3个月" 会被算成**已工作时长**，
            于是补偿金按 3 个月工龄算。而"试用期 3 个月"说的是**约定**，
            和"实际干了多久"完全是两个东西。`试用期` 三个字就在数字前面，
            看一眼就能区分——**这种"看一眼就知道"的判据不要省**。
            """
            return "试用期" in text[max(0, a - 5) : a]

        # ---- 收集所有匹配，先不合并 ----
        raw: list[tuple[int, int, int]] = []  # (起点, 终点, 折合月数)
        for m in _MONTHS_RE.finditer(text):
            if not clash(*m.span()) and not is_probation(m.start()):
                raw.append((m.start(), m.end(), int(m.group("num"))))
        for m in _YEARS_RE.finditer(text):
            if not clash(*m.span()):
                raw.append((m.start(), m.end(), int(m.group("num")) * 12))
        for m in _CN_YEARS_RE.finditer(text):
            if clash(*m.span()):
                continue
            n = cn_to_int(m.group("cn"))
            if n is None and m.group("cn") == "半":
                raw.append((m.start(), m.end(), 6))  # "半年" = 6 个月
            elif n:
                # ⚠️ "两年半" 里的 "半" 必须补上。`_CN_YEARS_RE` 的字符类里
                # 虽然有"半"，但"半"永远出现在"年"的**后面**（"两年半"），
                # 而正则要求字符在"年"前面，所以它只会匹配到"两年"，
                # 剩下的"半"就丢了 —— 24 个月和 30 个月对补偿金是两个数。
                months_here = n * 12
                if text[m.end() : m.end() + 1] == "半":
                    months_here += 6
                raw.append((m.start(), m.end(), months_here))

        # ---- 合并挨着的匹配 ----
        raw.sort()
        groups: list[list[tuple[int, int, int]]] = []
        for hit in raw:
            if groups:
                prev_end = groups[-1][-1][1]
                gap = text[prev_end : hit[0]]
                if hit[0] - prev_end <= 3 and not any(c in gap for c in "，。；、,.;"):
                    groups[-1].append(hit)
                    continue
            groups.append([hit])

        # ---- 每组各判各的 ----
        out: dict[str, SlotValue] = {}
        for g in groups:
            start, end = g[0][0], g[-1][1]
            months = sum(h[2] for h in g)
            if months <= 0 or months > 600:
                continue  # 见上面「一个上限保护」
            for h in g:
                taken.append((h[0], h[1]))

            after = text[end : end + 4]
            before = text[max(0, start - 5) : start]
            is_contract = any(w in after for w in ("合同", "期限", "协议", "聘期")) or any(
                w in before for w in ("合同", "签约", "签了", "签的", "约定", "期限")
            )
            name = "contract_term_months" if is_contract else "months_of_service"
            # 同一种时长在一句话里出现多次时，**后说的覆盖先说的**：
            # "我干了3年……哦不对，是4年" 里后者才是用户的最终说法。
            # （"两段工作经历要相加"那种情况这里处理不了，也不会瞎猜——
            #  加错比问一句严重得多。见 `_money_from` 里同样的取舍。）
            out[name] = SlotValue(
                value=months, turn=turn, quote=text[start:end], source="rule", precision=2
            )
        return out

    @classmethod
    def extract(cls, text: str, *, turn: int) -> dict[str, SlotValue]:
        """返回 `{槽位名: SlotValue}`。**没抽到的槽位不出现在结果里**，
        而不是给个 None——调用方 `Slots.put()` 本来就跳过 None，
        但返回干净的字典型更省事，也更好测。"""
        if not text:
            return {}
        out: dict[str, SlotValue] = {}
        used: list[tuple[int, int]] = []  # 已被吃掉的字符区间

        def take(m: re.Match[str]) -> None:
            used.append(m.span())

        def free(*spans: tuple[int, int]) -> bool:
            """`spans` 是否和已用区间不重叠。"""
            for a, b in spans:
                for c, d in used:
                    if a < d and c < b:
                        return False
            return True

        # ---- 1. 日期（必须先抽，见类 docstring）----
        dates: list[str] = []
        for m in _DATE_RE.finditer(text):
            take(m)
            y, mo, d = m.group("y"), m.group("m"), m.group("d") or "1"
            dates.append(f"{int(y):04d}-{int(mo):02d}-{int(d):02d}")

        # ---- 2. 年限 / 月数 ----
        out.update(cls._extract_durations(text, turn=turn, taken=used))

        # 入职日期 → 工作月数（有日期、又没明说年限时才算）
        if len(dates) >= 1 and "months_of_service" not in out:
            if "入职" in text or "开始工作" in text or "参加工作" in text:
                out["work_start"] = SlotValue(
                    value=dates[0], turn=turn, quote="入职日期", source="rule"
                )

        # ---- 3. 金额（跳过已被日期/年限吃掉的数字）----
        # ⚠️ 先把"中文数字 + 时间单位"的坑占掉。
        # 踩过的坑：`"赔我两个月工资"` 里的"两"被当成金额（2 元）——
        # `_MONTHS_RE` 只认阿拉伯数字，"两个月"从它手里漏了过去，
        # 然后金额正则看到"两"、又看到后面的"工资"线索，就认了。
        # 这类"数字后面跟着时间单位"的地方一律不是钱，先占掉最省事。
        for m in _CN_TIME_RE.finditer(text):
            take(m)

        money_hits: list[tuple[float, int, str, bool]] = []
        for rx in (_MONEY_RE, _CN_YUAN_RE):
            for m in rx.finditer(text):
                if not free(m.span()):
                    continue
                hit = _money_from(m, text)
                if hit[1] == 0 or hit[0] <= 0:
                    # 精度 0 = "这个数字没有工资线索，不敢认它"。见 `_money_from`。
                    continue
                take(m)
                money_hits.append(hit)

        if money_hits:
            # **取第一个"被工资线索点过名"的**，没有就取第一个。
            #
            # 为什么不是"取最大的"：一句话里可能同时有"年终奖 3 万，月薪 10000"，
            # 取最大会拿到年终奖，基数直接错三倍。
            # 为什么不是"取最后一个"："月薪 10000，年终奖 3 万"里最后一个也是年终奖。
            # 线索词（"月薪"就在匹配位置前面 6 个字内）比位置和大小都可靠——
            # **它直接编码了"这个数字是什么"这个信息**，而那正是我们要判断的东西。
            picked = next((i for i, h in enumerate(money_hits) if h[3]), 0)
            val, prec, raw, _ = money_hits[picked]
            out["monthly_wage"] = SlotValue(
                value=val, turn=turn, quote=raw, source="rule", precision=prec
            )
            rest = [h for i, h in enumerate(money_hits) if i != picked]
            if rest:
                # 落选的金额**不猜用途**，但必须记下来：
                # "为什么基数不对"这个问题的第一手证据就是它们。
                out["other_amounts"] = SlotValue(
                    value=[h[0] for h in rest],
                    turn=turn,
                    quote="、".join(h[2] for h in rest),
                    source="rule",
                    precision=0,  # **精度 0**：不许它覆盖任何东西，它只是个记录
                )

        # ---- 4. 情形 / 加班类型 ----
        for name, words in cls.CAUSE_PATTERNS:
            hit = next((w for w in words if w in text), None)
            if hit:
                out["termination_cause"] = SlotValue(
                    value=name, turn=turn, quote=hit, source="rule"
                )
                break
        for name, words in cls.OVERTIME_PATTERNS:
            hit = next((w for w in words if w in text), None)
            if hit:
                out["overtime_kind"] = SlotValue(
                    value=name, turn=turn, quote=hit, source="rule"
                )
                break

        # ---- 5. 布尔 ----
        if "提前通知" in text or "提前30天" in text or "提前三十天" in text:
            neg = any(w in text for w in ("没有提前", "没提前", "未提前", "没通知"))
            out["notice_given"] = SlotValue(
                value=not neg, turn=turn, quote="提前通知", source="rule"
            )
        if any(w in text for w in ("拖欠工资", "欠薪", "不发工资", "没发工资")):
            out["is_wage_arrears"] = SlotValue(
                value=True, turn=turn, quote="拖欠工资", source="rule"
            )

        # 劳动关系终止之日。⚠️ **它是个日期，不是"结束了没有"。**
        #
        # 这里踩过一个坑，值得记下来：一开始写的是"还在职 → False / 已离职 → True"，
        # 看起来更符合直觉，但 `calc_limitation` 的那个参数类型是 `date | None`
        # ——**一个布尔值塞进去会直接报参数校验错误**，工具根本调不动。
        # 教训：槽位的类型必须**以它对端的工具参数为准**，不能按语感定。
        # （`tests/test_memory.py` 里有一条测试专门盯着这件事。）
        #
        # 只有"词 + 日期"都齐了才填。理由是：
        # - "还在职" → **什么都不填**。工具那边"不填"本来就表示关系存续
        #   （见 `LimitationArgs.relationship_ended` 的说明），填 False 反而是多此一举
        # - "已经走了"但没说哪天 → **也什么都不填**。硬拿"今天"顶上，
        #   会让"过没过时效"这个结论凭空变松，比缺参数严重得多
        if dates and any(w in text for w in _ENDED_CUES):
            out["relationship_ended"] = SlotValue(
                value=dates[-1], turn=turn, quote="离职日期", source="rule"
            )

        # 试用期几个月。**必须在这里单独抽，不能靠"时长"那一段**——
        # "试用期3个月" 里的 "3个月" 看起来就是一段时长，但它是**约定**，
        # 不是"实际干了多久"。混在一起会让补偿金按 3 个月算。
        if "试用期" in text:
            tail = text[text.index("试用期") + 3 :][:8]
            m = re.search(r"([一二两三四五六七八九十半\d]+)\s*(?:个)?月", tail)
            if m:
                raw_n = m.group(1)
                n = (
                    int(raw_n) if raw_n.isdigit()
                    else (6 if raw_n == "半" else (cn_to_int(raw_n) or 0))
                )
                if n:
                    out["probation_months"] = SlotValue(
                        value=int(n), turn=turn, quote="试用期" + m.group(0), source="rule"
                    )

        # ---- 6. 加班时长：小时 / 天 ----
        # ⚠️ **"几个小时""几天"这种说法到处都是，光看数字必错。**
        # 踩过的坑：`"没提前30天通知我"` 里的 "30天" 被抽成了"加班 30 天"。
        # 于是算加班费时凭空多出一大笔，**用户看不出来**。
        #
        # 判据：**数字前后 6 个字里必须真的出现"加班"这类词**，
        # 而且窗口要小——用整句判的话，`"我不加班，每天工作8小时"`
        # 会把"8小时"当成加班时长（"加班"就在前面几个字）。
        # 窗口收到 6 个字，"不加班"的"加"落在窗口外，就不会误判。
        #
        # 代价是 `"平时每天加3小时"`（只说"加"没说"加班"）抽不到，
        # 那就让模型多问一句。**漏抽只是啰嗦，错抽是算错钱。**
        for key, rx, cues in (
            ("overtime_hours", r"(\d+(?:\.\d+)?)\s*(?:个)?小时", ("加班", "加点", "延时", "延长工作时间")),
            ("overtime_days", r"(\d+(?:\.\d+)?)\s*(?:个)?(?:整天|天)", ("加班", "加点", "值班", "休息日", "法定节假日")),
        ):
            m = re.search(rx, text)
            if not m:
                continue
            ctx = text[max(0, m.start() - 6) : m.end() + 6]
            if any(c in ctx for c in cues):
                out[key] = SlotValue(
                    value=float(m.group(1)), turn=turn, quote=m.group(0), source="rule"
                )

        # 城市：从一个固定表里认，**不猜**（认不出就不记）
        for city in _KNOWN_CITIES:
            if city in text:
                out["city"] = SlotValue(value=city, turn=turn, quote=city, source="rule")
                break

        return out


_KNOWN_CITIES = (
    "北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "武汉", "西安",
    "重庆", "天津", "苏州", "长沙", "郑州", "青岛", "合肥", "济南", "福州",
    "厦门", "东莞", "佛山", "宁波", "无锡", "大连", "沈阳", "昆明", "南昌",
)


# ============================================================================
# 会话级的槽位管理器
# ============================================================================


class SlotStore(BaseModel):
    """一次会话的槽位。**跨轮持久**——这就是它和"把事实塞进 prompt"的区别。"""

    slots: Slots = Field(default_factory=Slots)

    def observe(self, text: str, *, turn: int) -> list[str]:
        """看一句用户话，更新槽位。**返回这一次真正写进去的槽位名**。

        返回列表而不是 bool，是因为"第 12 轮更新了哪几个槽位"这件事
        要进 trace——排查"为什么第 20 轮少了一个数"时，第一眼看的就是它。
        """
        changed: list[str] = []
        # ⚠️ **内部槽位也照存不误，别在这里过滤。**
        # `work_start`（入职日期）正是 `derive()` 的原料——滤掉它，
        # "用户只给了入职日期、没给年限"这条最常见的路径就直接断了，
        # 而且**不报错**（`derive()` 只是什么都不做）。
        # 内部槽位"不给模型看"这件事由 `prompt_block()` 负责，
        # 不该由"存不存"来管。**职责放错了地方，就会出现这种静默失效。**
        for name, sv in SlotExtractor.extract(text, turn=turn).items():
            if self.slots.put(
                name, sv.value, turn=sv.turn, quote=sv.quote,
                source=sv.source, precision=sv.precision,
            ):
                changed.append(name)
        return changed

    def derive(self, *, turn: int) -> list[str]:
        """**派生槽位**：用已经抽到的槽位算出新的槽位。

        目前只有两条，都是真实咨询里反复出现的：

        - 入职日期 + 今天 → 工作月数（用户常给日期不给年限）
        - 城市 → 当地社平工资（查表；查不到就不填，**不编**）

        放在一个单独的方法里而不是塞进 `observe()`：派生是**纯计算**，
        幂等、可重放；抽取是**读人话**，会漏会错。两者混在一起以后没法分别调。
        """
        import datetime as _dt

        changed: list[str] = []

        # ---- 入职日期 → 工作月数 ----
        start = self.slots.get("work_start")
        if start and not self.slots.has("months_of_service"):
            try:
                d0 = _dt.date.fromisoformat(str(start))
                today = _dt.date.today()
                months = (today.year - d0.year) * 12 + (today.month - d0.month)
                if today.day < d0.day:
                    months -= 1
                if months >= 0 and self.slots.put(
                    "months_of_service", months, turn=turn,
                    quote=f"由入职日期 {d0} 推算", source="manual", precision=2,
                ):
                    changed.append("months_of_service")
            except ValueError:
                pass

        # ---- 城市 → 社平工资 ----
        city = self.slots.get("city")
        if city and not self.slots.has("local_avg_monthly_wage"):
            wage = CITY_AVG_WAGE.get(str(city))
            if wage and self.slots.put(
                "local_avg_monthly_wage", wage, turn=turn,
                quote=f"{city} 上年度职工月平均工资（内置表，需人工复核）",
                source="manual", precision=1,  # **精度只给 1**：这是表里的值，会过期
            ):
                changed.append("local_avg_monthly_wage")

        return changed


CITY_AVG_WAGE: dict[str, float] = {
    # ⚠️ **这是一张会过期的表，精度故意只给 1（模糊）**。
    # 用途是让"三倍封顶"这个判断能跑起来，不是给出最终数字——
    # 真正的数字要去当地人社局当年公布的文件里查（那是用户的离线任务）。
    # 数值是近几年公开口径的大致量级，**上线前必须替换**。
    "北京": 11761.0,
    "上海": 12183.0,
    "广州": 11411.0,
    "深圳": 13730.0,
    "杭州": 10000.0,
    "南京": 9500.0,
    "成都": 8600.0,
    "武汉": 8200.0,
    "西安": 7900.0,
    "重庆": 7900.0,
}


__all__ = [
    "CITY_AVG_WAGE",
    "SLOT_SPECS",
    "TOOL_SLOTS",
    "SlotExtractor",
    "SlotSpec",
    "SlotStore",
    "SlotValue",
    "Slots",
    "cn_to_int",
]
