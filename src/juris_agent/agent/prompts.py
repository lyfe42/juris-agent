"""提示词与上下文拼装。**集中放一处，不散在节点里。**

## 为什么要单独一个文件

因为提示词是**会被反复改**的东西，而改它必须能回答"我改的这一版比上一版好在哪"。
如果提示词散在 `nodes.py` 的五个函数里，你没法把它整体拎出来做 A/B。

## 三条本项目验证过、值得记住的提示词原则

**1. 引用编号由我们给，不让模型自己数。**
注入的每条法条前面我们写好 `[1]`、`[2]`……模型只需要**照抄**。
如果让它"引用时自己编号"，它就会编——而编出来的编号我们查得出来
（见 `agent/verify.py`），但用户体验已经坏了。**把可以替模型做的事替它做掉，
是防幻觉最便宜的手段。**

**2. 明确给出"不知道时怎么办"。**
提示词里必须写清楚"如果检索结果不足以回答，就调用 `refuse_to_answer`"，
并且说明**这是被鼓励的**。不写这句，模型的默认行为是"硬答"——
它会用通用法律知识补出一段听起来合理、却没有出处的话。
**拒答必须有明确出口，否则模型一定会编。**

**3. 把结构要求写成"能对着检查"的形式。**
比如"每条结论后面必须跟 `[编号]`"比"请给出依据"可执行得多——
前者写完能被程序校验，后者只能靠人看。
"""

from __future__ import annotations

from typing import Any, Sequence

from juris_agent.agent.state import InjectedArticle

# --------------------------------------------------------------------------- #
# 角色设定
# --------------------------------------------------------------------------- #

AGENT_SYSTEM = """你是一名严谨的劳动法咨询助手，服务于中国内地的劳动者。

## 你的工作方式

你可以调用工具来获取依据和做计算。**先用工具查证，再下结论**，不要凭记忆回答。

- 问「法律怎么规定的」「有没有依据」「能不能要」→ 先调 `search_law` 查条文。
- 问「能拿多少钱」「加班费多少」「试用期合法吗」「仲裁还来得及吗」→
  调对应的计算工具（`calc_severance` / `calc_overtime` / `check_probation` /
  `calc_limitation`）。
- 两件事都要（先问规定、再问金额）→ **两个工具都要调**，先把法条查出来，
  再计算。可以分多轮调用。

## 什么时候该拒答

遇到下面任何一种情况，**调用 `refuse_to_answer`**，不要硬答：

1. 问题不属于劳动法（婚姻、刑事、股权、知识产权、消费者权益等）。
2. 缺少关键事实（不知道金额、日期、解除原因），**而且这些事实只有用户能提供**——
   这种情况先向用户提问澄清；但如果用户已经明确表示无法提供，就拒答。
3. 需要对具体案情做法庭级别的实质判断（"我这场官司能不能赢"）。

**拒答不是失败，编一个听起来合理的答案是。** 用户会按你的答案去和公司谈判、
去仲裁，一个没有出处的结论可能让他损失几万块。

## 输出要求

- 结论必须来自工具返回的内容。工具没说的事，不要说。
- 引用法条时写编号，格式为 `[1]`（编号由工具返回的上下文给出，照抄即可）。
- 一次性把该调的工具都调了，不要一件事问一轮。
"""

ANSWER_SYSTEM = """你是一名严谨的劳动法咨询助手。请**只根据下面提供的法条与计算结果**回答用户的问题。

## 硬性要求

1. **每条法律结论后面必须给出引用编号**，格式 `[1]`、`[2]`。
   编号就是下面「可引用的法条」里每一条前面的方括号数字。
   **只允许使用其中出现过的编号**，写别的编号等于编造依据。
2. 如果提供的法条不足以回答问题，**明确说"没有找到足够依据"**，
   并说明还缺什么信息。**不要用你自己的法律知识补充**。
3. 计算题：把工具给出的**算式和代入过程**完整转述出来，不要只给最终数字。
   用户需要能自己核对。
4. 用口语化的中文，不要用"兹""本条款项"这类公文腔。
   给普通劳动者看，不是给法务看。
5. 结尾用一句话给出**可操作的建议**（下一步该做什么、去哪、注意什么期限）。
6. 如果下面列出了「还缺哪些事实」，**先说清楚缺什么、请用户补充**，
   不要用假设值（"假设你工作了三年"）代替。**假设出来的数字会让用户
   按一个错的数去谈判**，那比不回答更糟。

## 回答结构

- 先给一句话结论。
- 再说依据（带引用编号）。
- 涉及金额的把算式写出来。
- 最后给建议。
"""

CHITCHAT_SYSTEM = """你是一名劳动法咨询助手。用户在和你说一句客套话或问你能做什么。

用一两句话友好回应，并**主动介绍你能帮什么**：
- 解释劳动法条文（劳动合同、解除、加班、社保、试用期……）
- 算经济补偿金、加班费、仲裁时效
- 看一份劳动合同有没有违法条款

不要长篇大论。不要在这种情况下编造法律结论。
"""


# --------------------------------------------------------------------------- #
# 上下文的拼装
# --------------------------------------------------------------------------- #


def build_context_block(injected: Sequence[InjectedArticle]) -> str:
    """把注入的法条拼成给模型看的一段上下文。**`[i]` 编号就是在这里产生的。**

    ⚠️ 两个细节，改的时候别丢：

    - **`source="neighbor"` 的条目标出来**。邻条扩展带出来的条文没有分数、
      和问题的相关度可能不高。标出来让模型知道"这条是顺带捞的上下文"，
      它会更谨慎地引用。不标的话模型会把它当正式命中，引用一堆无关条文。
    - **空集合也要返回一句人话**，不能返回空串。返回空串时模型看到的上下文是
      "什么都没有"，它可能理解成"没有法律依据"而不是"我们没检索到"。
      （不过正常流程里，空上下文根本走不到生成节点——`INV-01` 会拦住并转去拒答。）
    """
    if not injected:
        return "（本次没有检索到任何法条。不要凭记忆作答。）"
    lines: list[str] = ["## 可引用的法条", ""]
    for a in injected:
        tag = "（相邻条文，供参考）" if a.source == "neighbor" else ""
        lines.append(f"[{a.index}] 《{a.law}》{a.article_no} {tag}")
        lines.append(a.text.strip())
        lines.append("")
    return "\n".join(lines).rstrip()


def build_tool_results_block(tool_results: Sequence[Any]) -> str:
    """把工具执行结果拼成给模型看的一段。

    `tool_results` 里每一项是 `ToolCallRecord`（`state.ToolCallRecord`）。
    失败项**也拼进去**，而且明确写"调用失败"——理由：模型需要知道自己刚才
    那次调用没成，否则它会以为"查过了，就是这个结果"。
    静默丢掉失败记录是本项目最不能接受的一类 bug。
    """
    if not tool_results:
        return "（本次没有调用任何工具。）"
    lines: list[str] = ["## 工具调用结果", ""]
    for i, r in enumerate(tool_results, start=1):
        head = f"### {i}. {r.tool}"
        if r.status != "ok":
            head += f"（调用失败：{r.status}）"
        lines.append(head)
        if r.error:
            lines.append(f"错误：{r.error.get('message', '')}")
        if r.payload is not None:
            import json

            lines.append(json.dumps(r.payload, ensure_ascii=False, indent=1, default=str))
        lines.append("")
    return "\n".join(lines).rstrip()


def build_missing_block(missing: Sequence[str]) -> str:
    """「还缺哪些事实」那一段。**这是把"工具报错"翻译成"该问用户什么"。**

    工具因为缺参数失败时，返回给模型的是 `{"field_errors": [{"loc":
    "months_of_service", "type": "missing"}]}`——**字段名是英文的、
    是给程序看的**。照着它追问，用户会收到一句"请提供 months_of_service"。

    所以这里做一层翻译：拿槽位表里的中文标签写成人话。
    翻译的依据是 `SLOTS.required`（哪些字段是"没有就算不出来"的），
    不是"工具报了什么"——**用户没提过、但法条有默认值的字段
    （比如"有没有提前 30 天通知"）不该问**，问了是打扰。
    """
    if not missing:
        return ""
    items = "、".join(dict.fromkeys(str(x) for x in missing))
    return (
        "## 还缺哪些事实\n"
        f"用户还没有提供这些信息：**{items}**。\n"
        "**先向用户提问澄清这几项，不要用假设值算下去。**"
        "可以顺带说明「为什么要这几项」（例如算经济补偿必须先知道工作了多久），"
        "用户才知道这些信息是干什么用的、值不值得去查。"
    )


def build_answer_prompt(
    question: str,
    injected: Sequence[InjectedArticle],
    tool_results: Sequence[Any] = (),
    *,
    history_block: str = "",
    missing: Sequence[str] = (),
) -> str:
    """生成节点用的 user 消息。

    ⚠️ 顺序是「记忆 → 法条 → 工具结果 → 缺什么 → 问题」，**问题放最后**。
    理由：模型对**末尾**内容的注意力最高，问题放最后能让它在写完之前的
    上下文后立刻对准要回答的事。反过来（问题放最前）实测更容易出现
    "答了一半开始总结上下文"。
    """
    parts: list[str] = []
    if history_block:
        parts.append(history_block)
    parts.append(build_context_block(injected))
    if tool_results:
        parts.append(build_tool_results_block(tool_results))
    missing_block = build_missing_block(missing)
    if missing_block:
        parts.append(missing_block)
    parts.append("## 用户的问题")
    parts.append(question)
    parts.append("")
    parts.append("请按系统提示的要求回答。每条法律结论后面给出 `[编号]`。")
    return "\n\n".join(parts)


def build_refusal(
    *,
    category: str,
    reason: str,
    suggestion: str,
    violated: Sequence[str] = (),
) -> dict[str, Any]:
    """拒答的结构。**三个字段一个都不能少**（`INV-06` 会检查）。

    刻意做成普通字典而不是调 `refuse_to_answer` 工具：
    这是**程序决定的拒答**（路由判越界、检索为空、工具全挂），
    不是"模型选择拒答"。让程序自己拒答**比让模型拒**可靠得多——
    模型拒答要花一次调用，还可能拒错对象（把 in-scope 的问题拒掉）。
    """
    return {
        "reason": reason,
        "category": category,
        "suggestion": suggestion,
        "violated": list(violated),
    }


__all__ = [
    "AGENT_SYSTEM",
    "ANSWER_SYSTEM",
    "CHITCHAT_SYSTEM",
    "build_answer_prompt",
    "build_context_block",
    "build_missing_block",
    "build_refusal",
    "build_tool_results_block",
]
