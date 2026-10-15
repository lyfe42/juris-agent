"""四道防死循环的限制（F17）。★

## 先说清楚"死循环"在这个项目里长什么样

Agent 转不出来不是因为代码写错了（那是会直接抛异常的 bug），而是因为
**模型在"合理地"重复自己**。三种真实形态：

1. **原地打转**：查「试用期最长多久」→ 没查到满意的 → 换个说法再查 → 还是那些结果
   → 再换 → ……每一步都"有道理"，但整条轨迹是停的。**它烧的是钱和时间。**
2. **同一句话说两遍**：模型给出工具调用 → 工具返回 → 模型又给出**一模一样**的调用
   （参数一字不差）。这通常是因为它不满意结果，但**再调一次结果不会变**。
3. **没完没了地问**：模型判断"信息不够"，追问用户 → 用户答了 → 它又问另一个 →
   用户又答 → ……**对话永远不收敛**，用户被问烦了直接关掉。

所以只堵一条是不够的：堵了步数，形态 1 会先用光预算；堵了重复，形态 3 根本
不重复调用工具。**必须四个维度各堵一道。**

## 四道闸门（顺序即优先级，先撞上的先报）

| # | 闸门 | 判据 | 拦住什么 |
|---|---|---|---|
| ① | `max_steps` | 总步数 ≥ 12 | 兜底的兜底。所有其他闸门失效时还有它 |
| ② | `repeat_call_limit` | **连续** 2 次 `(工具名, 参数哈希)` 完全相同 | 形态 2 |
| ③ | `no_progress_limit` | **连续** 3 步没产出任何新东西 | 形态 1、3 |
| ④ | `token_budget` | 累计 token > 60000 | 钱。**也是唯一能防"单步很贵"的那道** |

**为什么②在③前面**：重复调用**必然**是"没有进展"，所以③也能拦它。
但②能给出**更精确的诊断**——"你连续两次用相同参数调了 search_law"，
而③只能说"你三步没进展"。**报错要报得能直接定位问题**，
不然线上看日志的人还得自己去比对参数。

## ⚠️ 两个容易做错的地方（面试官爱问的就是这两个）

**1. "连续"是连续，不是累计。** 中间只要夹着一次别的成功调用，计数就归零。
理由：`search_law` 第 1 步调过、第 5 步再调一次，这**完全正常**（那是用户追问了
一个新的点）。把它算成"重复"会让 Agent 在第 5 步就被掐断。
所以计数器看的是**末尾连续的一串**，不是"这轮里一共调了几次"。

**2. 无进展的"进展"要定义得够宽。** 如果只把"检索到新条文"算进展，
那么"调用工具、工具成功、在回答"这条完全正常的路径会被判成没进展。
所以凡是**产出了新的可观察结果**（新的条级键、新的工具结果、产出了答案）
都算进展。判据在 `progress_signature()`。

## 用法：闸门是"查"不是"改"

这个模块**只读状态、不改状态**。检查结果由节点自己决定怎么用——
目前的用法是：主循环每步之后 `check(state)`，不通过就跳到收尾节点，
把 `finish_reason` 设成对应值并把 `GuardHit` 记进 trace。
**把"判断"和"处理"分开**，是为了让闸门本身能被单测穷举（构造 50 条对抗输入
只调 `check`，不用真跑 Agent）。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field

from juris_agent.agent.state import AgentState, GuardHit, ToolCallRecord
from juris_agent.config import GuardSettings, Settings, get_settings

# 闸门名字。**这四个字符串会出现在 trace、日志、报表里**，
# 所以它们是稳定的对外标识，不要随手改（改了报表里的历史数据就对不上了）。
LIMIT_STEPS = "max_steps"
LIMIT_REPEAT = "repeat_call"
LIMIT_NO_PROGRESS = "no_progress"
LIMIT_BUDGET = "token_budget"


class GuardDecision(BaseModel):
    """闸门的判定结果。"""

    ok: bool
    limit: str | None = None
    message: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)

    def as_hit(self, step: int) -> GuardHit:
        """转成可写进 trace 的记录。**撞闸门一定要留证据**——
        只报"被掐了"而不报"为什么被掐"，报表里就只剩一个数字，没法排查。"""
        return GuardHit(
            limit=self.limit or "unknown",
            message=self.message,
            step=step,
            detail=self.detail,
        )


def args_hash(args: dict[str, Any] | None) -> str:
    """参数指纹。**先排序键再序列化**——否则 `{"a":1,"b":2}` 和 `{"b":2,"a":1}`
    会被算成两个不同的调用，闸门就形同虚设（模型换个键序就绕过去了）。
    `sort_keys=True` + `separators` 是经典写法：紧凑且确定。"""
    payload = json.dumps(args or {}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def call_signature(rec: ToolCallRecord) -> str:
    """一次调用的完整签名：`工具名@参数哈希`。"""
    return f"{rec.tool}@{args_hash(rec.args)}"


def trailing_run(items: Sequence[str]) -> int:
    """末尾连续**完全相同**的项有几个。`["a","b","b","b"]` → 3；空列表 → 0。

    写成"从后往前数，遇到不同的就停"——一行就够，别用 `groupby`。

    ⚠️ **它不够用来判"无进展"**，原因见 `no_progress_run()` 的说明。
    """
    if not items:
        return 0
    last = items[-1]
    n = 0
    for x in reversed(items):
        if x != last:
            break
        n += 1
    return n


NO_PROGRESS = "-"
"""`progress_signature()` 在"这一步什么也没产出"时返回的哨兵串。

它被 `no_progress_run()` 单独认出来，因为**"什么都没产出"这件事，
哪怕头一回发生也是没进展**——不需要"以前见过"这个条件。
"""


def no_progress_run(items: Sequence[str]) -> int:
    """③号闸门用的读数：**末尾连续几步"没产出新东西"。**

    一步一步往回数，只要这一步满足下面任一条，它就计一格：

    - **以前出现过**：`items[i] in items[:i]`。拿回来的东西是老面孔。
    - **是个空产出**：值等于哨兵 `NO_PROGRESS`（这一步什么也没拿到）。

    一旦碰到"这一步真的产出了以前没见过的东西"就停。

    ## 为什么不是直接数"末尾连续相同"（`trailing_run`）

    因为真实的原地打转**不是**重复同一步，而是**在两个状态之间来回跳**：

    ```
    A 查「经济补偿 计算标准」 → 检索器返回第 46、47 条
    B 查「解除劳动合同 补偿」 → 检索器返回第 46、47 条   ← 一条不差
    A 再查一遍                → 还是那两条
    B 再查一遍                → 还是那两条
    ```

    每一步的**参数都不同**（所以②重复调用闸门抓不到），相邻两步的产出也
    **不相同**（所以"末尾连续相同"永远只数到 1，③也抓不到）。
    但整条轨迹是**停的**——模型在那两个查询之间循环，一点新信息都没拿到。
    用户付了四次检索的钱。

    正确的判据只能是"这一步拿到的东西，是不是以前就拿过了"，
    跟"用什么话去问"无关。

    ## 为什么也不直接数"末尾连续相同"（第二个理由）

    拿 `[a, a, a]` 举例：

    - "末尾连续相同" → 3
    - 这里的口径 → **2**（第一个 `a` 是新的，**那一步是进展**；后两个才是白跑）

    ③号闸门要的是后者。多算一步的后果是：一段"每步都拿到新东西、
    只是恰好末尾两项相同"的正常轨迹会被提前掐断——
    **把正常轨迹误判成原地打转，比漏判一次原地打转严重得多**
    （前者用户直接拿不到答案，后者只是多花一次钱）。

    代价：它要看整段历史（`items[:i]`），不是只看尾巴。列表最长也就
    `max_steps`（12）项，代价可以忽略。
    """
    n = 0
    for i in range(len(items) - 1, -1, -1):
        if items[i] != NO_PROGRESS and items[i] not in items[:i]:
            break
        n += 1
    return n


ECHO = "<回声>"
"""被抹掉的"回声字段"的占位符。用中文是因为它会出现在调试输出里，
能直接看出来"这里原本是模型自己那句话，不算新信息"。"""


def strip_echo(payload: Any, args: dict[str, Any] | None) -> Any:
    """把出参里**原样抄回入参**的字符串字段换成 `ECHO`。

    ## 为什么需要这个（这是真的踩出来的坑）

    `search_law` 的出参里有 `query` 字段，就是把模型问的那句话抄回来
    （方便模型对账）。于是：

    ```
    第 1 次：args={"query":"经济补偿 计算标准"} → 出参 {"query":"经济补偿 计算标准", "hits":[46,47]}
    第 2 次：args={"query":"解除劳动合同 补偿"} → 出参 {"query":"解除劳动合同 补偿", "hits":[46,47]}
    ```

    两次的 `hits` **一条不差**，可是整个出参的摘要不同（因为 `query` 不同），
    于是"无进展"闸门看到的是一条"一直在产出新东西"的轨迹，**一路放行到步数上限**。
    用户花六次检索的钱，拿回同一批法条三次。

    修法不是"针对 `search_law` 写个特例"——那等于让闸门认识每个工具的 schema，
    以后每加一个会回声入参的工具就得记得来改闸门（一定会忘）。
    用一条**与工具无关**的规则：**出参里哪个字符串字段的值和入参里同名字段一模一样，
    那它就是回声，不是新信息。**

    方向是安全的：这条规则**只会让指纹更容易撞上，不会凭空造出差异**。
    万一某个工具的回声是变形过的（比如加了前缀），这条规则不生效，
    行为退回"整包摘要"——也就是不够灵敏，而不是误报。
    """
    if isinstance(payload, dict):
        return {
            k: (
                ECHO
                if isinstance(v, str) and (args or {}).get(k) == v
                else strip_echo(v, args)
            )
            for k, v in payload.items()
        }
    if isinstance(payload, list):
        return [strip_echo(v, args) for v in payload]
    return payload


def result_digest(payload: Any, args: dict[str, Any] | None = None) -> str:
    """工具**返回了什么**的指纹。**注意：不看参数，只看返回。**

    入参 `args` 只用来抹掉回声字段（见 `strip_echo`），
    它**不参与**摘要计算——否则"换个说法问同一件事"就又变成"有进展"了。
    """
    try:
        blob = json.dumps(
            strip_echo(payload, args), sort_keys=True, ensure_ascii=False, default=str
        )
    except (TypeError, ValueError):  # 兜底：实在序列化不了就用 repr
        blob = repr(payload)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def progress_signature(
    *,
    new_keys: Iterable[str] = (),
    tool_results: Sequence[ToolCallRecord] = (),
    produced_answer: bool = False,
) -> str:
    """这一步"产出了什么"的指纹。**没产出任何东西时返回固定串 `"-"`。**

    ⚠️ 三种产出都要算，缺一个就会出现"正常路径被判成没进展"的误伤：

    - `new_keys`：新检索到的条级键。**要用排序后的集合**，
      否则"同样 8 条但顺序不同"会被算成新产出——模型每次重新检索都能骗过闸门。
    - `tool_results`：工具结果。
    - `produced_answer`：这一步生成了答案/拒答。**必须有这一项**，
      否则最后的收尾步会被判成没进展。

    ## ⚠️ 工具结果的指纹**只看返回，绝不看参数**（这里踩过坑）

    第一版把"工具名 + 参数哈希 + 成功失败"当指纹，想的是"同样的调用才算重复"。
    **那是错的，而且错得很典型**：

    ```
    模型查「经济补偿 计算标准」 → 检索器返回第 46、47 条
    模型查「解除劳动合同 补偿」 → 检索器返回第 46、47 条   ← 一字不差
    模型又查「裁员 补偿金」      → 检索器返回第 46、47 条
    ```

    参数每次都不一样，所以指纹每次都不同——**闸门看到的是一条"一直在产出新东西"
    的轨迹，于是一路放行到步数上限**。可用户看到的是：这个 Agent 花了六次检索
    的钱，查回来同一批法条三次。

    "有没有进展"这件事的判据只能是**拿到的信息新不新**，跟"用什么话去问"无关。
    所以指纹换成**返回值的摘要**。上面的例子三次返回一样 → 同一个指纹 → ③号闸门
    在第 3 步就拦下来了。

    顺带一个副作用是好的：同名同参的调用如果返回了不同结果（比如熔断从"拒绝"
    变成"放行"），指纹不同 → 算进展。**那确实是进展。**
    """
    if produced_answer:
        return "answer"
    if new_keys:
        return "keys:" + hashlib.sha256(
            "|".join(sorted(set(new_keys))).encode("utf-8")
        ).hexdigest()[:12]
    if tool_results:
        parts = [
            f"{r.tool}:{r.status}:{result_digest(r.payload, r.args)}" for r in tool_results
        ]
        return "tool:" + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return NO_PROGRESS


def check(
    state: AgentState,
    settings: Settings | GuardSettings | None = None,
) -> GuardDecision:
    """按①②③④的顺序查一遍。**返回第一个撞上的，不是全部。**"""
    if isinstance(settings, Settings):
        g = settings.guard
    else:
        g = settings or get_settings().guard

    # ① 最大步数。**用 `>=` 而不是 `>`**：`max_steps=12` 的语义是
    #    "最多走 12 步"，第 12 步走完就该停，不是"第 13 步才停"。
    if state.steps >= g.max_steps:
        return GuardDecision(
            ok=False,
            limit=LIMIT_STEPS,
            message=f"已达最大步数 {g.max_steps}，停止并给出当前最好的结果",
            detail={"steps": state.steps, "max_steps": g.max_steps},
        )

    # ② 重复调用。看**末尾连续**的调用签名。
    sigs = [call_signature(r) for r in state.tool_results]
    run = trailing_run(sigs)
    if run >= g.repeat_call_limit:
        return GuardDecision(
            ok=False,
            limit=LIMIT_REPEAT,
            message=(
                f"连续 {run} 次用完全相同的参数调用了同一个工具"
                f"（{state.tool_results[-1].tool}），再调一次结果也不会变，停止"
            ),
            detail={
                "tool": state.tool_results[-1].tool,
                "args_hash": args_hash(state.tool_results[-1].args),
                "consecutive": run,
                "limit": g.repeat_call_limit,
            },
        )

    # ③ 无进展。**注意这里用的是 `no_progress_run` 不是 `trailing_run`**——
    #    用后者会漏掉"两个查询之间来回跳"这种最常见的原地打转形态。
    run = no_progress_run(state.progress_keys)
    if run >= g.no_progress_limit:
        return GuardDecision(
            ok=False,
            limit=LIMIT_NO_PROGRESS,
            message=(
                f"连续 {run} 步没有产出任何新结果（没检索到新条文、没得到新工具结果），"
                "判定为原地打转，停止"
            ),
            detail={
                "consecutive": run,
                "limit": g.no_progress_limit,
                # 把末尾那几步的指纹一起带上，好定位是从哪一步开始打转的
                "tail": list(state.progress_keys[-run:]),
            },
        )

    # ④ token 预算。
    if state.tokens_used > g.token_budget:
        return GuardDecision(
            ok=False,
            limit=LIMIT_BUDGET,
            message=(
                f"累计消耗 {state.tokens_used} token，超过预算 {g.token_budget}，停止"
            ),
            detail={"tokens_used": state.tokens_used, "budget": g.token_budget},
        )

    return GuardDecision(ok=True, detail={"steps": state.steps})


def snapshot(state: AgentState, settings: Settings | GuardSettings | None = None) -> dict[str, Any]:
    """四个闸门各自的当前读数。**给 Web 界面和报表用**——

    "还剩几步""预算用了多少"这类信息，在产品上能直接安抚用户
    （"正在处理，已用 3/12 步"），在报表上能看出一条轨迹是怎么走到尽头的。
    """
    if isinstance(settings, Settings):
        g = settings.guard
    else:
        g = settings or get_settings().guard
    sigs = [call_signature(r) for r in state.tool_results]
    return {
        "steps": state.steps,
        "max_steps": g.max_steps,
        "steps_left": max(0, g.max_steps - state.steps),
        "consecutive_same_call": trailing_run(sigs),
        "repeat_call_limit": g.repeat_call_limit,
        "consecutive_no_progress": no_progress_run(state.progress_keys),
        "no_progress_limit": g.no_progress_limit,
        "tokens_used": state.tokens_used,
        "token_budget": g.token_budget,
    }


__all__ = [
    "LIMIT_BUDGET",
    "LIMIT_NO_PROGRESS",
    "LIMIT_REPEAT",
    "LIMIT_STEPS",
    "NO_PROGRESS",
    "GuardDecision",
    "args_hash",
    "call_signature",
    "check",
    "no_progress_run",
    "progress_signature",
    "result_digest",
    "snapshot",
    "strip_echo",
    "trailing_run",
]
