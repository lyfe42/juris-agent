"""工作记忆：滑动窗口 + 分段摘要（F20）。

## 要解决的问题

20 轮对话的原文大约 6000～10000 token。全塞进 prompt 有三个后果：

1. **贵**。每一轮都要把那 8000 token 重发一遍，20 轮就是 16 万 token 的输入。
2. **慢**。输入越长首字延迟越高。
3. **反而更差**。这是最容易忽略的一点——上下文里塞满"第 7 轮聊过社保"这种
   无关内容，会稀释掉"这一轮到底在问什么"。**长上下文不等于好上下文。**

所以要做减法：**近期原文 + 远期摘要**。

## 两个参数，两个不同的理由

- `window`（保留几轮原文）：**近期对话不压缩。**因为"这一轮的问题"几乎总是
  依赖"上一轮说了什么"（"那我不签呢？"——"那"指什么，只有上一轮知道）。
  压缩近期 = 直接丢掉指代关系。
- `segment`（几轮并成一段摘要）：**摘要本身也有成本**（要调模型）。
  一轮一摘要 = 20 次模型调用；六轮一摘要 = 3 次。而且零碎摘要读起来没有上下文。

## ⚠️ 摘要一定是有损的，这件事必须写在代码里而不是嘴上说

摘要是模型写的，它会**自然地**把"月薪 12000"写成"月薪一万多"——
因为它的目标是写一段通顺的话，不是保真。

所以设计上有一条硬分工：

| 层 | 负责什么 | 能不能有损 |
|---|---|---|
| **槽位**（`slots.py`） | 精确的事实（金额、日期、年限） | ❌ 一个字都不能错 |
| **摘要**（本文件） | 发生过什么、聊到了哪些话题 | ✅ 有损没关系 |
| **窗口**（本文件） | 近几轮的原文与指代关系 | ❌ 原样保留 |

**"精确的事实绝不交给有损压缩去承载"**——这是整个记忆层的核心结论，
也是面试里"第 20 轮丢信息怎么排查"那个问题的答案（先查槽位，再查窗口，
最后才怀疑摘要）。

## 摘要怎么生成

两条路，**默认走规则路**：

- `rule`：不调模型，把这几轮的**用户提问按顺序列出来**，附上提到过的条级键。
  免费、可复现、零延迟。信息量当然不如模型写的，但**它不会编**。
- `llm`：调模型写一段中文摘要。更连贯，代价是钱 + 延迟 + 不可复现
  （`temperature=0` 也不保证稳定，这是实测过的，见 `reports/diagnosis.md`）。

**默认 rule 的理由**：摘要层是三层里唯一"可以有损"的一层，它能提供的
边际价值最小。为最不关键的一层付模型调用费，顺序错了。
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from juris_agent.memory.tokens import count_messages, count_tokens


class Message(BaseModel):
    """一轮里的一条消息。**刻意不直接用 dict**——dict 在跨模块传递时
    很容易多塞一个键出来，而这里每多一个键都会被算进 token 预算。"""

    role: str  # user / assistant
    content: str
    turn: int


class Summary(BaseModel):
    """一段被压缩掉的历史。**带 `from_turn`/`to_turn` 是关键**——

    没有覆盖范围的话，出了"第 9 轮说过的话丢了"这种事，你连
    "它到底有没有被压进摘要"都判断不了，只能靠猜。
    """

    from_turn: int
    to_turn: int
    text: str
    method: str = "rule"  # rule / llm

    @property
    def span(self) -> int:
        return self.to_turn - self.from_turn + 1

    def render(self) -> str:
        return f"[第{self.from_turn}-{self.to_turn}轮摘要] {self.text}"


class Summarizer(Protocol):
    """摘要器的接口：给一串消息，返回一段话。

    用 Protocol 而不是抽象基类：实现方（`rule_summarize`、真要接模型时的
    LLM 版本）**不需要继承任何东西**，只要是"能拿消息列表、返回字符串"的
    可调用对象就行。测试里塞一个 lambda 也能用。
    """

    def __call__(self, msgs: list[Message]) -> str: ...


class RuleSummarizer:
    """规则摘要器的**可调用对象形态**。

    ⚠️ 为什么要有这个类：`WorkingMemory.summarize` 是个 pydantic 字段，
    而 pydantic 不接受"裸函数"当字段类型（它没法给函数生成 schema）。
    所以默认值得是一个**能被 pydantic 接受的对象**，而不是函数本身。

    封装一下还有个附带好处：`__name__` 稳定（`rule_summarize` 函数被
    pydantic 包一层之后名字会变），而摘要的 `method` 字段要写进报表，
    名字必须是稳的。
    """

    __name__ = "rule"

    def __call__(self, msgs: list[Message]) -> str:
        return rule_summarize(msgs)


def rule_summarize(msgs: list[Message]) -> str:
    """不调模型的摘要：**把用户问过的问题列出来**。

    为什么只抄问题不抄回答：回答通常很长（判断 + 条文 + 算式），
    全抄就失去压缩的意义了；而"用户问过什么"恰好是判断
    "这一段在聊什么话题"最省字的表示。

    长度兜底：一段摘要最多 3 条问题，超了就只保留**最后 3 条**——
    越近的越相关，这条原则在摘要层和召回层都成立。
    """
    qs = [m.content.strip().replace("\n", " ") for m in msgs if m.role == "user"]
    if not qs:
        # 全是助手消息（少见，但真会出现：用户连发几条之后模型回答被切成两段）
        return "（这一段没有新的用户提问）"
    kept = qs[-3:]
    prefix = "用户先后问过：" if len(kept) > 1 else "用户问过："
    body = "；".join(q[:40] for q in kept)
    return prefix + body


class WorkingMemory(BaseModel):
    """滑动窗口 + 分段摘要。**它是纯函数式的**：给同样的历史，永远得到同样的结果。

    这一点是刻意的——记忆层如果带随机性，"第 20 轮丢了信息"就变成了
    一个无法复现的 bug，只能靠重试去撞。
    """

    window: int = 6
    """保留最近几轮原文。**6 是估出来的**：一次法律咨询里，"这一轮"
    和"决定答案的那一轮"之间平均隔 2～3 轮，6 轮给了 2 倍余量。"""
    segment: int = 6
    """几轮并成一段摘要。和 window 取一样大是为了让分界干净：
    「最近 6 轮原文 + 再往前每 6 轮一段摘要」。"""
    summarize: Any = Field(default_factory=RuleSummarizer)
    """摘要器。**默认规则版**（不调模型）——理由见模块 docstring。
    换成 LLM 版只要传一个 `__call__(msgs) -> str` 的对象进来。

    ⚠️ 这里注解写的是 `Any` 而不是 `Summarizer`，**不是偷懒**：
    pydantic 会给字段声明生成校验 schema，而 `Protocol` 类是不能
    被实例化去校验的，声明成 `Summarizer` 会让整个模型**在导入时**就报错。
    真正的类型约束由 `Summarizer` 协议 + 类型检查器负责，
    运行期的鸭子类型在这里够用（传错了会在第一次调用时炸，不会静默）。
    """
    summaries: list[Summary] = Field(default_factory=list)
    """**已经压好的摘要，跨轮累积。**它必须存下来：每轮重新压一遍
    前面所有历史的话，第 20 轮要压 14 轮，成本随轮次线性上涨，
    而压出来的东西和第 19 轮几乎一样——纯浪费。"""

    def ingest(self, msgs: list[Message]) -> dict[str, Any]:
        """喂进完整的对话历史，产出"该给模型看的三段"。

        返回：
            recent      近期原文（保序）
            summaries   摘要（旧的在前）
            compressed  这一次新压出来的轮次区间（没有新的就为空）

        实现要点：**只有"超出窗口且还没被压过"的轮次才会触发摘要**。
        判断依据是 `self.summaries` 里已经覆盖到第几轮了，
        不是"轮次总数 % segment"——后者在历史被截断时会重复压缩。
        """
        turns = sorted({m.turn for m in msgs})
        if not turns:
            return {"recent": [], "summaries": list(self.summaries), "compressed": None}

        cutoff = turns[-1] - self.window + 1  # 这一轮及以后保留原文
        # 已经压到第几轮了
        covered = self.summaries[-1].to_turn if self.summaries else 0

        pending = [t for t in turns if t < cutoff and t > covered]
        compressed = None
        # 攒够一整段才压，不足一段就留着——**半段摘要比不摘要更糟**：
        # 它会让"这一段覆盖了第 3-8 轮"这个信息变得不准。
        while len(pending) >= self.segment:
            chunk_turns = pending[: self.segment]
            chunk = [m for m in msgs if m.turn in set(chunk_turns)]
            self.summaries.append(
                Summary(
                    from_turn=chunk_turns[0],
                    to_turn=chunk_turns[-1],
                    text=self.summarize(chunk),
                    method=getattr(self.summarize, "__name__", "custom"),
                )
            )
            compressed = (chunk_turns[0], chunk_turns[-1])
            pending = pending[self.segment:]

        recent = [m for m in msgs if m.turn >= cutoff]
        return {"recent": recent, "summaries": list(self.summaries), "compressed": compressed}

    def render_summaries(self, *, newest_first: bool = True) -> str:
        """把摘要渲染成一段文本。**默认新的在前**——

        和人类读聊天记录一样：越近的越重要，模型读到长上下文时
        对**开头**最敏感（中间容易漏）。所以重要的东西放前面。
        """
        if not self.summaries:
            return ""
        items = list(reversed(self.summaries)) if newest_first else list(self.summaries)
        return "\n".join(s.render() for s in items)


def window_tokens(msgs: list[Message]) -> int:
    """这一段消息要花多少 token。**走 `count_messages` 统一口径**——
    记忆层各处如果各算各的，预算分配就会前后对不上，多出来的那部分
    会把 prompt 挤爆，而且没人知道是哪里多出来的。"""
    return count_messages([{"role": m.role, "content": m.content} for m in msgs])


def text_tokens(text: str) -> int:
    return count_tokens(text)


__all__ = [
    "Message",
    "RuleSummarizer",
    "Summary",
    "Summarizer",
    "WorkingMemory",
    "rule_summarize",
    "text_tokens",
    "window_tokens",
]
