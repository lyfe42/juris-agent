"""跑几个真问题，把"系统里到底发生了什么"打在终端上（F35 的演示脚本）。

    uv run python scripts/demo.py                    # 真调模型（要 key，要花钱）
    uv run python scripts/demo.py --mode record      # 真调模型，顺手录下来
    uv run python scripts/demo.py --mode replay      # 不调模型，逐字节复现上次的结果
    uv run python scripts/demo.py --question "..."   # 只问一个问题

## 这个脚本和网页版是什么关系

**它们走的是同一段代码**：都调 `AgentRunner.stream`、
都把增量喂给 `api/events.py` 翻译成事件。区别只在于事件最后去哪儿——
网页版发 SSE 给浏览器，这里打印在终端。

**刻意不共用 HTTP**：演示脚本要能在没有服务、没有端口、
没有前端的情况下跑；而网页要能证明"真的能对话"。
两者共同的**内核**是 `events.py` 那一层，
所以"终端看到的事件名"和"浏览器看到的事件名"必然一致——
这比"两边各写一份、靠人保证它们一样"可靠得多。

## 为什么默认只问三个问题

三个问题分别对应三种**完全不同的链路**：

| 问题 | 走哪条路 | 证明什么 |
|---|---|---|
| 试用期最长多久 | calc（工具循环） | 工具调用、法条检索、引用校验都活着 |
| 经济补偿怎么算 | calc + 缺参数追问 | 记忆槽位、主动追问、算钱 |
| 帮我写一首诗 | out_of_scope → 拒答 | **它知道自己不会什么** |

**第三个才是这个项目最该演示的。** 一个什么都肯答的模型看起来更强，
但一个会说"这个我不该答、你去找律师"的系统才有资格被用在工作场景里。
只演示前两个问题，等于把最值钱的那部分藏起来了。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from juris_agent.agent.graph import make_runner  # noqa: E402
from juris_agent.api.events import events_from_update  # noqa: E402
from juris_agent.config import Settings  # noqa: E402
from juris_agent.observability import describe_cost, estimate_cost  # noqa: E402

# 三个问题的顺序是**刻意**的：从"最像一个普通问题"到"最该被拒绝"。
# 反过来排的话，观众看到拒答就走了，看不到这个系统真正会做什么。
DEFAULT_QUESTIONS = [
    "试用期最长可以约定多久？",
    "我在公司干了 3 年 2 个月，月薪 12000，被无故辞退能拿多少经济补偿？",
    "帮我写一首关于秋天的诗。",
]

# 事件名 → 终端上打什么颜色。用 ANSI 而不是引第三方库：
# 演示脚本多一个依赖就多一个"现场装不上"的可能。
_COLOR = {
    "memory": "\033[35m",      # 紫
    "route": "\033[36m",       # 青
    "retrieval": "\033[32m",   # 绿
    "tool_call": "\033[33m",   # 黄
    "tool_result": "\033[33m",
    "missing": "\033[33m",
    "citations": "\033[32m",
    "guard": "\033[31m",       # 红
    "refusal": "\033[31m",
    "error": "\033[31m",
    "answer": "\033[37m",      # 白
    "done": "\033[90m",        # 灰
}
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def _c(name: str, text: str) -> str:
    return f"{_COLOR.get(name, '')}{text}{_RESET}"


def _oneline(name: str, data: dict) -> str:
    """把一个事件压成一行。**只挑真正值得看的信息**，不 dump 整个 JSON。

    为什么不直接 `json.dumps(data)`：事件里带着法条正文（一条几百字），
    全打出来会刷十几屏，把"系统做了什么"这个主线彻底淹掉。
    终端演示和网页的取舍正好相反——网页有折叠、有滚动，可以全给；
    终端只该给一眼能扫完的东西。
    """
    if name == "memory":
        slots = [s for s in (data.get("slots") or []) if s.get("value")]
        bits = [f"策略 {data.get('policy')}", f"{data.get('tokens', 0)} token"]
        for s in slots:
            bits.append(f"{s['label']}={s['value']}")
        if data.get("missing"):
            bits.append(f"还缺 {data['missing']}")
        return " · ".join(bits) or "本轮无历史可用"

    if name == "route":
        return f"{data.get('route')}　{data.get('reason', '')}"

    if name == "retrieval":
        n = data.get("count", 0)
        cites = [a.get("citation", "") for a in (data.get("articles") or [])][:4]
        more = "…" if n > len(cites) else ""
        return f"命中 {n} 条：{'、'.join(cites)}{more}　{data.get('note', '')}"

    if name == "tool_call":
        return f"{data.get('tool')}　{json.dumps(data.get('args') or {}, ensure_ascii=False)}"

    if name == "tool_result":
        bits = [f"{data.get('tool')} → {data.get('status')}"]
        if data.get("attempts", 1) > 1:
            bits.append(f"重试 {data['attempts'] - 1} 次")
        if data.get("idempotency_hit"):
            bits.append("幂等命中")
        if data.get("error"):
            err = data["error"]
            bits.append(f"原因：{err.get('message') or err.get('code')}")
        bits.append(f"{data.get('elapsed_ms', 0):.0f}ms")
        return " · ".join(bits)

    if name == "missing":
        return f"还需要用户补充：{'、'.join(data.get('labels') or [])}"

    if name == "citations":
        bits = [f"有效 {data.get('valid')}"]
        if data.get("invalid"):
            bits.append(f"{_c('guard', '编造并已删除 ' + str(data['invalid']))}")
        if data.get("uncited"):
            bits.append(f"没被引用 {data['uncited']}")
        return " · ".join(bits)

    if name == "guard":
        return f"触发闸门 {data.get('message') or data.get('limit')}"

    if name == "refusal":
        return (
            f"[{data.get('category')}] {data.get('reason')}\n"
            f"    {_DIM}可以试试：{data.get('suggestion', '')}{_RESET}"
        )

    if name == "error":
        return data.get("message", "")

    return json.dumps(data, ensure_ascii=False)[:160]


def _fmt_cost(cost) -> str:
    """成本一行。**判断全在 `describe_cost` 里，这里只负责上色。**

    ⚠️ 这条分工是刻意的。**别把"这笔钱怎么说"的逻辑复制到这个脚本里**——
    判断（下面那三种 0 元）写在两个地方，迟早会出现两种说法，
    而错的通常是没人细看的那一处（比如以后生成的 `reports/cost.md`）。
    脚本和报表共用同一句话，才能保证它们不会互相打架。

    这里只加一件事：**这一轮没花真钱时，整行压暗**。
    终端上一眼就能区分"这行是账单"和"这行是重放"。
    """
    text = describe_cost(cost)
    if cost is not None and not cost.billable:
        return f"{_DIM}{text}{_RESET}"
    return text


def run_one(runner, question: str, *, settings: Settings, llm=None, verbose: bool = True) -> dict:
    """问一个问题，把事件打在终端上，返回一份摘要。

    ⚠️ `llm` 是**显式传进来的**，不从 `runner` 内部掏。
    掏内部属性（`runner._deps.llm`）看着省事，但那样一来
    **"成本从哪儿读"就依赖了 `AgentRunner` 的私有结构**——
    哪天它改成把 llm 存在别处，这个脚本会静默读到 `None`、
    然后每一行都退回"按总 token 粗估"，而那个数看着是合理的。
    """
    if verbose:
        print(f"\n{_BOLD}{'─' * 78}{_RESET}")
        print(f"{_BOLD}问：{question}{_RESET}\n")

    # 开账本。**必须在这个线程里开**——账本是线程本地的。
    # 这里是单线程脚本，所以不那么要紧；但保持和 `api/app.py` 一样，
    # 是为了让"两边跑出同一个数"这件事不用再验证一次。
    if llm is not None and hasattr(llm, "begin_turn"):
        llm.begin_turn()

    t0 = time.perf_counter()
    final = None

    for mode, payload in runner.stream(question, with_values=True):
        if mode == "updates":
            for node, delta in (payload or {}).items():
                for ev in events_from_update(node, delta or {}):
                    # 逐字事件在终端上不能一行一个（一段答案几百行）。
                    # **直接跳过**：整段答案在下面用最终状态打一次，
                    # 而那个版本才是权威的（引用校验之后的样子）。
                    if ev.name == "answer":
                        continue
                    if verbose:
                        print(f"  {_c(ev.name, '▪ ' + ev.name):<24} {_oneline(ev.name, ev.data)}")
        elif mode == "values":
            final = payload

    elapsed = (time.perf_counter() - t0) * 1000

    if final is None:
        print(f"  {_c('error', '这一轮没有产出状态')}")
        return {"question": question, "ok": False}

    answer = (final.get("answer") or "") if isinstance(final, dict) else getattr(final, "answer", "")
    finish = final.get("finish_reason") if isinstance(final, dict) else final.finish_reason
    route = final.get("route") if isinstance(final, dict) else final.route
    cites = final.get("citations") if isinstance(final, dict) else final.citations

    if verbose:
        print(f"\n  {_c('answer', '● answer')}")
        for line in (answer or "").splitlines():
            print(f"    {line}")
        if not (answer or "").strip():
            print(f"    {_DIM}（空）{_RESET}")

    # 成本：优先用本线程这一轮的账，没有就退回状态里的总数——**和 `api/app.py`
    # 用同一套优先级**，否则终端和网页会报出两个不同的金额，
    # 而那种不一致会让人怀疑两边都有问题。
    usage = getattr(llm, "turn_usage", None) if llm is not None else None
    model = settings.model_gen
    if usage is not None and (usage.total_tokens or usage.prompt_tokens):
        cost = estimate_cost(
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cache_hit_tokens=usage.prompt_cache_hit_tokens,
            cache_miss_tokens=usage.prompt_cache_miss_tokens,
        )
    else:
        total = int((final.get("tokens_used") if isinstance(final, dict) else final.tokens_used) or 0)
        cost = estimate_cost(model=model, prompt_tokens=0, completion_tokens=total)
        # 走到这条退路 = 这一轮的账本里一次计费调用都没有（replay 命中缓存时
        # 客户端**刻意不记账**，见 `llm.py`）。**和 `api/app.py` 一样标 False**，
        # 否则 replay 演示会打出一个"看着很合理"的金额。
        cost.billable = False
        cost.basis = (
            "按状态里记录的 token 总数估算（未取到分项）"
            if total
            else "这一轮没有调用模型"
        )

    if verbose:
        print(f"\n  {_c('done', '● done')}　{route} / {finish}　{elapsed:.0f}ms")
        print(f"    成本 {_fmt_cost(cost)}")
        if cites:
            print(f"    引用 {cites}")

    return {
        "question": question,
        "ok": bool((answer or "").strip()),
        "route": route,
        "finish": finish,
        "citations": list(cites or []),
        "tokens": cost.prompt_tokens + cost.completion_tokens,
        "cny": cost.total_cny,
        # ⚠️ `billable` 必须一路带到汇总表。**不能汇总时只看 `cny`**——
        # replay 跑一遍的 `cny` 加起来是 0.27 元，而那个 0.27 是
        # "如果真调要花多少"，不是"这次花了多少"。汇总表是整场演示
        # 最容易被截图引用的一行，报错一个数就是报错全部结论。
        "billable": cost.billable,
        "ms": round(elapsed),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="跑几个真问题，把过程打在终端上")
    ap.add_argument("--mode", default="off", choices=["off", "record", "replay"],
                    help="LLM 缓存模式。off=每次都真调；record=真调并录下来；"
                         "replay=不联网，重放录制（要那份录制里正好有这几个问题）")
    ap.add_argument("--question", action="append", default=None,
                    help="只问这一个问题（可重复）。默认问三个预设问题。")
    ap.add_argument("--quiet", action="store_true", help="只打答案和成本，不打事件流")
    args = ap.parse_args(argv)

    settings = Settings(llm_mode=args.mode)
    questions = args.question or DEFAULT_QUESTIONS

    print(f"{_BOLD}juris-agent 演示{_RESET}　模式 {settings.llm_mode}"
          f"　模型 {settings.model_gen}　检索 {settings.retrieval.chunker}")

    # 装配。**这一步会加载嵌入模型**（几十秒、1GB 内存），
    # 所以整个脚本只装配一次，然后连着问好几个问题——
    # 每个问题重新装配一次的话，演示现场会有一半时间在等模型加载。
    from juris_agent.llm import LLMClient
    from juris_agent.retrieval.build import build_default_retriever

    retriever = build_default_retriever(settings=settings)
    # ⚠️ `LLMClient` **自己从 `get_settings()` 读配置**，不接 `Settings` 对象。
    # 所以 `--mode` 得同时告诉两边：这里用 `mode=` 显式传，
    # 否则会出现"脚本以为在 replay、客户端其实在 off"这种
    # **两边不一致却都不报错**的情况（off 模式下缓存写了但不读，
    # 表现是每次仍然真的联网调模型、账单照付）。
    llm = LLMClient(mode=args.mode)
    runner = make_runner(retriever=retriever, llm=llm, settings=settings)

    rows = []
    for q in questions:
        rows.append(run_one(runner, q, settings=settings, llm=llm, verbose=not args.quiet))
    llm.close()

    # 汇总表。**演示的最后一定要有一张表**——零散的输出看着热闹，
    # 但没人能从里面读出"一共花了多少、拒答了几次"。
    print(f"\n{_BOLD}{'═' * 78}{_RESET}")
    print(f"{_BOLD}汇总{_RESET}")
    total_tokens = total_cny = billed_cny = 0
    for r in rows:
        if not r.get("ok") and r.get("finish") not in {"refused"}:
            flag = "✗"
        elif r.get("finish") == "refused":
            flag = "⊘"   # 拒答**不是失败**，用另一个符号
        else:
            flag = "✓"
        total_tokens += r.get("tokens", 0)
        total_cny += r.get("cny", 0)
        billed = r.get("billable", True)
        if billed:
            billed_cny += r.get("cny", 0)
        # ⚠️ **行里必须标出"这笔钱没真花"**，否则表格会出现
        # "每一行都有 ¥ 金额、合计却是 ¥0"这种自相矛盾的样子，
        # 看的人只会觉得这张表算错了——而算错的是**读法**，不是算术。
        # 加个 `*` 比什么都省事：它就在那个可疑的数字旁边。
        mark = "" if billed else "*"
        print(f"  {flag} {r['question'][:34]:<36} {r.get('route', ''):<12} "
              f"{r.get('finish', ''):<10} {r.get('tokens', 0):>7} tok "
              f"¥{r.get('cny', 0):.6f}{mark}  {r.get('ms', 0):>6}ms")
    print(f"  {'─' * 76}")
    if billed_cny == total_cny:
        print(f"  合计 {total_tokens} token · ¥{total_cny:.6f}　"
              f"{_DIM}（{len(rows)} 个问题；拒答 ⊘ 也是预期行为，不是错误）{_RESET}")
    else:
        # 有重放/缓存的轮次。**两个数都打**：一个是这次真花的，
        # 一个是"要是全部真调会花多少"。只打后者等于虚报开销，
        # 只打前者则看不出这套流程跑一遍的实际预算。
        print(f"  合计 {total_tokens} token · {_BOLD}本次实付 ¥{billed_cny:.6f}{_RESET}"
              f"　{_DIM}（若全部真调约 ¥{total_cny:.6f}）{_RESET}")
        print(f"  {_DIM}{len(rows)} 个问题；拒答 ⊘ 也是预期行为，不是错误；"
              f"带 * 的行是重放/缓存，那个 ¥ 是'当初真调会花多少'，没有真的花掉{_RESET}")
    if settings.llm_mode == "record":
        print(f"\n  {_DIM}已录制到 fixtures/replay/llm/。下次可以 `--mode replay` 离线复现。{_RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
