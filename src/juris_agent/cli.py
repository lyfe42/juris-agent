"""`juris` 命令行。**装完之后真正的入口。**

## 它解决的是哪一件事

项目里原来已经有一堆能跑的脚本（`scripts/demo.py`、`scripts/build_index.py`…），
但它们是**开发视角**的：知道项目内部长什么样的人，才知道该先跑哪个、再跑哪个。
一个拿到这个仓库的人，第一条命令该敲什么？

```bash
juris doctor                      # 我这儿环境齐了吗？
juris ask "被辞退有补偿吗？"       # 问一个
juris chat                        # 连着聊
juris serve                       # 起 Web 界面
```

四条。**这就是这个文件的全部理由。**

## 和 `scripts/` 的分工（别搞混）

| | 给谁用 | 会不会花 API 的钱 |
|---|---|---|
| `juris`（本文件） | **用它的人** | `ask`/`chat` 会（真调模型），`doctor`/`serve` 不会 |
| `scripts/*.py` | **改它的人 / 评测** | 大部分会（就是要花钱跑数字） |

⚠️ **评测脚本不要走 `juris`。** 它们要的是"原始状态 + 可复现 + 只跑被测的那一臂"，
而 CLI 面向的是"给人看的一句话"，中间隔着一层排版。
混着用会出现"报表上的数字和 CLI 打出来的回答对不上"，
而两边其实都没错——**只是量表不同**。这类对不上最费时间。

## 为什么用 `argparse` 而不是 `typer` / `click`

因为 `argparse` 是标准库。**这个 CLI 存在的理由就是"别让第一次用的人先装东西"**，
为它多引一个依赖是自相矛盾的。它的四个子命令也没有复杂到需要 `click`
（没有嵌套命令组、没有需要自动补全的参数）。真到了需要的时候再换，
换的成本就是这一个文件。

## 输出分流：**回答走 stdout，诊断走 stderr**

这样 `juris ask "..." > answer.txt` 拿到的是一个干净的答案，
`juris ask "..." 2>/dev/null` 拿到的是"只有答案、没有成本那一行"。
**不这么分的话，「把答案存下来」这件事就变成了「再把文件里的
`route=law steps=3 ¥0.0123` 那几行删掉」。**
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# --------------------------------------------------------------------------- #
# 让中文输出在任何终端下都不乱码
# --------------------------------------------------------------------------- #


def _force_utf8() -> None:
    """把 stdout / stderr 切到 UTF-8。

    ## 这不是洁癖，是 Windows 上的必经之路

    本机（Windows）控制台的默认编码是 GBK（`cp936`）。答案里必然有中文，
    于是一旦输出被**重定向或管道**（`juris ask ... | tee log.txt`、
    被别的程序当子进程调用），Python 就用 `locale.getpreferredencoding()`
    去编码，而中文字符不一定都在 GBK 里——最典型的是弯引号 `“”`、
    不换行空格，以及**这个项目里到处在用的方括号 `「」`**。
    症状是 `UnicodeEncodeError: 'gbk' codec can't encode character ...`，
    而且它**只在重定向时出现**：直接在终端里敲是好的（终端用的是控制台 API，
    不走编码器）。**"换个方式跑就炸"是最容易被误判成玄学的一类问题。**

    `errors="replace"`：真遇到编不出来的字符，打成一个 `?` 而不是崩掉。
    一个问号难看，但它至少让**剩下的九行**还能看——
    而崩掉的话，这一条命令的输出**整个没有了**。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # 已经被别的东西包过一层（比如 pytest 的 capture）时会失败。
                # **静默放过**：这不是"出错了"，只是"这个流不归我们管"。
                pass


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #


def _eprint(*parts: Any) -> None:
    """诊断信息。**统一走 stderr**，理由见模块 docstring 的「输出分流」。"""
    print(*parts, file=sys.stderr)


# 这些是**别人的**日志器，不是我们的（前缀就是它们的包名）。
#
# ## 为什么要在 CLI 里按下去（这件事比看起来重要）
#
# 因为 `observability/logs.py` 里那句 `logging.basicConfig(stream=sys.stdout)`——
# 它把**标准库根日志**的输出重定向到了 **stdout**。这本身是对的
# （服务端要的就是日志和业务日志混在一个流里给日志平台收），
# 但它带来一个在**命令行里很致命**的副作用：
#
#     httpx 每打一次模型接口都会 INFO 一行 `HTTP Request: POST ...`，
#     这行会跟着答案一起进 stdout。
#
# 于是 `juris ask "问题" > answer.txt` 存下来的文件里，
# **混着十几行 HTTP 请求日志**；而 `juris ask ... | jq` 直接解析失败
# （`--json` 那一路尤其明显：输出不再是"一整段合法 JSON"）。
# 模块 docstring 里承诺的"回答走 stdout、诊断走 stderr"被这一行日志破坏掉了——
# **而且它是在别处的日志配置里破坏的，从 cli.py 里看不出来。**
#
# ⚠️ 所以这里**必须**在装配之前按住它们，且**不能**改
# `configure_logging` 的默认行为：服务端那边这些日志是有用的
# （排查"请求到底发出去没有"就靠它）。要看的用 `--verbose`。
_NOISY_LOGGERS = (
    "httpx",  # 每次模型调用一行 `HTTP Request: ...`
    "httpcore",
    "urllib3",
    "filelock",
    "jieba",  # `Building prefix dict from the default dictionary ...`
    "sentence_transformers",  # 加载模型时的进度和提示
    "transformers",
    "huggingface_hub",  # `Warning: You are sending unauthenticated requests ...`
    "asyncio",
)


def _setup_logging(*, verbose: bool) -> None:
    """命令行的日志装配：**日志走 stderr，第三方刷屏按下去。**

    ## 顺序很关键，而且是实测踩出来的

    ① **先`configure_logging`**。`observability/logs.py` 的默认流是
       **stdout**（服务端要的就是日志和业务输出混在一起给日志平台收）。
       命令行反过来：stdout 只能有答案。所以这里显式传 `stream=sys.stderr`。

       ⚠️ **这一步必须最早调。** `configure_logging` 里有个
       "配过就不再配"的开关（见 `logs.py` 的第 ② 条），而
       `from juris_agent.client import JurisAgent` 会**顺手**
       把日志配成默认的 stdout 版——那之后再调就是空操作了，
       stderr 这个诉求被静默忽略，stdout 继续被日志污染。
       **症状：`--json | jq` 报 `Invalid numeric literal`，
       而 `cli.py` 里看上去到处都对。**

    ② **再按第三方**。见 `_quiet_third_party` 里那段"先 import 再 setLevel"，
       jieba 会在 import 时把自己的级别改回去。

    `verbose=True` 时只做 ①。**日志照样走 stderr**——那是"输出分流"，
    和"要不要看日志"是两件事：想把它存进文件的人，
    该做的是 `2> log.txt`，而不是让答案和日志混在一个流里。
    """
    from juris_agent.observability import configure_logging

    level = "INFO" if verbose else "WARNING"
    configure_logging(level=level, stream=sys.stderr)
    if not verbose:
        _quiet_third_party()


def _quiet_third_party() -> None:
    """把上面那些日志器压到 WARNING。**必须在装配之前调。**

    ## ⚠️ 顺序是"先 import、再 setLevel"，不能反过来

    第一版写的就是"先 setLevel"。跑出来 jieba 那四行**照打**，
    而它明明在 `_NOISY_LOGGERS` 里。实测原因：

    ```python
    _quiet_third_party()                     # jieba 级别 = 30 (WARNING)
    import jieba                             # 这一句把级别**自己改回 10 (DEBUG)**
    logging.getLogger("jieba").level         # → 10，而且多了个 stderr 的 handler
    ```

    **jieba 在 import 时会把自己的 logger 级别设成 DEBUG**（它就是这么写的，
    为了让"首次加载词典"那几行能打出来）。所以"先按住"等于没按——
    按住的是一个还没被 import 的 logger，import 一发生就被覆盖了。

    这个坑的形态值得记一笔：**它不是"我设错了值"，
    而是"我设的那个东西在之后被别人改了"。**
    排查时盯着自己的代码看多久都没用——**得去看那个库的 import 期副作用**。
    判断"谁改的"最直接的办法就是在 import 前后各打一次 `.level`。

    ## 用 `setLevel` 而不是 `logging.disable(INFO)`

    后者是**全局**的，会连我们自己的 `log.info("服务就绪")` 一起按掉——
    那样出问题时用户手里什么线索都没有了，而 `--verbose` 也救不回来
    （被 disable 的日志根本不会走到处理器）。
    **只按别人、不按自己**，这样 `--verbose` 的语义才是干净的。
    """
    import importlib
    import logging

    # 先 import。**失败就跳过，不报错**——没装检索依赖（`--extra ml`）时
    # `sentence_transformers` 根本不存在，而 `juris doctor` 恰恰
    # 就是用来告诉你这件事的，它不能因为这个自己先崩掉。
    for mod in ("jieba", "httpx", "huggingface_hub", "sentence_transformers", "transformers"):
        try:
            importlib.import_module(mod)
        except Exception:  # noqa: BLE001
            # 捕获所有异常而不是只捕获 ImportError：这些库在 import 期
            # 还可能因为缺 DLL、版本不兼容而抛别的（`OSError`、`RuntimeError`）。
            # 对"我想把日志按下去"这件事来说，它们的处理方式完全一样。
            pass

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _answer_of(agent: Any, question: str, args: argparse.Namespace) -> tuple[Any, bool]:
    """跑一个问题。**`--stream` 时把答案边生成边打出来。**

    返回 `(答案, 正文是不是已经流式打过了)`。

    ⚠️ **第二个返回值不是多余的**，它是踩出来的：见 `_print_answer` 里
    "一个增量都没收到"那段。

    ⚠️ 流式那个分支里有个细节值得说：**回调打印的是"增量"，不是"现状"**。
    `on_delta` 每次给一小段（一到几个 token），直接写出去就是
    打字机效果。想攒着、想加时间戳、想让终端不闪，都在这一个回调里做——
    原因是**图那边不知道终端长什么样**（它也可能在给浏览器推 SSE），
    所以它只负责"有一小段新的了"，怎么显示是调用方的事。
    """
    if not args.stream:
        return agent.ask(question, session_id=args.session), False

    pieces: list[str] = []

    def on_delta(piece: str) -> None:
        pieces.append(piece)
        sys.stdout.write(piece)
        # ⚠️ **必须 flush。** 不 flush 的话，这一小段会先在 Python 的用户态
        # 缓冲里攒着（管道下默认块缓冲），于是"流式"变成"等全部生成完
        # 再一次性打出来"——**功能和没开一样，但代码里明明写着 on_delta**。
        # 这种"看起来实现了其实没实现"是最难发现的一类。
        sys.stdout.flush()

    ans = agent.ask(question, session_id=args.session, on_delta=on_delta)
    if not pieces:
        # ⚠️ **`on_delta` 不是每轮都会被调。** 只有"生成节点调模型"那一条路会推增量；
        # 拒答、闲聊、超纲、以及没有模型时，正文是节点**直接写进状态**的。
        # 所以"没收到任何增量"不是异常，而是另一种正常的路径。
        #
        # 第一版没管这件事：`--stream` 且走到上面任一路由时，
        # 正文一个字都不会被打出来，**只打出了一行"引用法条："**——
        # 用户看到的是一个空回答加一堆引用，而且 **`juris ask --stream`
        # 和 `juris ask` 对同一个问题会给出不同的东西**。
        # 所以这里把 `False` 交回去，让 `_print_answer` 整段打。
        return ans, False
    print()  # 正文已经打完了，补一个换行，让下面的引用从新的一行开始
    return ans, True


def _print_answer(ans: Any, args: argparse.Namespace, streamed: bool = False) -> None:
    """人类可读的输出。**答案区域走 stdout，其余走 stderr。**

    `streamed=True` 表示正文已经在 `_answer_of` 里流式打过了，
    这里只补引用那一块——**不补的话同一段答案会被打两遍**
    （打两遍会让人以为模型答了两遍，而这在终端上很难看出是排版 bug）。

    ⚠️ 这里**不按 `route` 分叉**。写这一句是因为很容易想当然地加一个
    `if ans.route == "calc": 另外打一遍算式`——**不要加**：
    `calc` 的算式已经在 `answer` 正文里了（模型自己写的），
    再打一遍会出现"同一个数出现两次，而且第二次是我拼的"。
    更要紧的是，一旦这里按路由加工输出，**CLI 打出来的东西
    就不再等于 `Answer.answer`**，而评测脚本读的是后者——
    两边从此可以静默地不一致。
    """
    if streamed:
        if ans.citations:
            print("引用法条：")
            for c in ans.citations:
                print(f"[{c.index}] {c.text}")
    else:
        # 整段（答案 + 引用）一次打出来。**非流式、以及流式但没收到增量时，
        # 走的都是这一支**——这也是它必须是"整段"而不是"只打答案"的原因。
        print(ans.text)

    # --- 下面全是诊断，走 stderr ------------------------------------------ #
    if ans.warnings:
        for w in ans.warnings:
            _eprint(f"⚠️  {w}")

    if args.trace:
        _print_trace(ans)

    parts = [
        f"route={ans.route or '?'}",
        f"finish={ans.finish_reason or '?'}",
        f"steps={ans.steps}",
        f"tokens={ans.tokens}",
    ]
    if not args.no_cost:
        parts.append(ans.cost_text)
    _eprint("[" + "  ".join(parts) + "]")

    if ans.finish_reason.startswith("guard_"):
        # ⚠️ **撞闸门必须说出来。** 它是"这一轮是被硬停下来的"，
        # 不是"正常答完了"。混在一起报的话，用户会把一条
        # 半截的答案当成完整答案用。理由见 `reports/guard_ablation.md`。
        _eprint(
            f"⚠️  这一轮被防死循环的闸门截断了（{ans.finish_reason}），"
            "答案可能不完整。"
        )


def _print_trace(ans: Any) -> None:
    """把 `state.trace` 打成人话。**"它到底走了哪几步"的唯一入口。**"""
    trace = getattr(ans.state, "trace", None) or []
    if not trace:
        _eprint("(没有 trace)")
        return
    _eprint("执行轨迹：")
    for t in trace:
        step = getattr(t, "step", 0)
        node = getattr(t, "node", "")
        ms = getattr(t, "elapsed_ms", 0.0)
        note = getattr(t, "note", "")
        line = f"  {step:>2}. {node:<12} {ms:>8.1f}ms"
        if note:
            line += f"  {note}"
        _eprint(line)


def _emit_json(ans: Any, *, trace: bool = False) -> None:
    """`--json` 的输出。**只往 stdout 写一行合法的 JSON，别的什么都不写。**

    为什么不是"人话和 JSON 混着打"：那样调用方得先做字符串切分才知道
    从哪开始是 JSON，而**切分逻辑迟早会跟不上格式变化**。
    要么全人话，要么全 JSON，中间没有第三种。
    """
    data = ans.to_dict()
    if trace:
        data["trace"] = [
            {
                "step": getattr(t, "step", 0),
                "node": getattr(t, "node", ""),
                "elapsed_ms": getattr(t, "elapsed_ms", 0.0),
                "note": getattr(t, "note", ""),
            }
            for t in (getattr(ans.state, "trace", None) or [])
        ]
    # `ensure_ascii=False`：让输出里是中文而不是 `\uXXXX`。
    # **这是给人看的 JSON**（`cite` 字段要是转义的就没人愿意看了），
    # 而它已经是合法 UTF-8，机器读也完全没问题。
    print(json.dumps(data, ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------- #
# 子命令
# --------------------------------------------------------------------------- #


def cmd_doctor(args: argparse.Namespace) -> int:
    """体检。**这个命令永远不抛异常，返回值就是结论。**

    退出码是给脚本用的：`juris doctor || echo "环境不对"`。
    **`0` = 能答问题，`1` = 不能**，没有第三种。
    """
    from juris_agent.client import JurisAgent

    agent = JurisAgent()
    try:
        health = agent.health()
    finally:
        # 体检不需要留着那个连接池。**不 close 的话，脚本跑完
        # 连接池要等进程退出才释放**——在 CI 里就是"跑完卡住一会儿"。
        agent.close()

    if not health.get("ready"):
        print("❌ 装不起来，现在答不了问题。")
        print()
        for p in health.get("problems", []):
            print(f"  - {p}")
        print()
        print(_FIX_HINTS)
        return 1

    print("✅ 装好了，可以答问题。")
    print()
    settings = health.get("settings", {})
    print(f"  对话模型    {settings.get('model_gen', '?')}")
    print(f"  裁判模型    {settings.get('model_judge', '?')}")
    print(f"  嵌入模型    {settings.get('embedding_model', '?')}")
    print(f"  运行环境    {settings.get('env', '?')}")
    print(f"  LLM 缓存模式 {settings.get('llm_mode', '?')}")
    print()
    retrieval = health.get("retrieval", {})
    print(f"  切分方式    {retrieval.get('chunker', '?')}")
    print(f"  查询改写    {retrieval.get('rewrite', '?')}")
    print(f"  精排开关    {retrieval.get('use_rerank', '?')}")
    print(f"  注入条数    {retrieval.get('inject_k', '?')}")
    print()
    print(f"  已有会话数  {health.get('sessions', {}).get('sessions', 0)}")
    return 0


_FIX_HINTS = """排查顺序（从最常见的开始）：

  1. 有没有 .env？       cp .env.example .env
  2. .env 里有没有 key？  查 DEEPSEEK_API_KEY 这一行
  3. 装没装检索的依赖？    uv sync --extra ml
  4. 语料建了没？          uv run python scripts/build_index.py
  5. huggingface 下不动？  .env 里设 HF_ENDPOINT=https://hf-mirror.com

⚠️ 改完配置后要重新跑一次 `juris doctor`（每次都是新进程，自动重装）。
   如果在同一个 Python 进程里改（REPL / 交互式），要 new 一个 JurisAgent
   或者调 agent.close()——失败的装配结果会被记住，理由见 client.py。"""


def cmd_ask(args: argparse.Namespace) -> int:
    """问一句就走。**这是最常用的那个。**"""
    from juris_agent.client import JurisAgent, NotReady

    question = args.question
    if question is None or question == "-":
        # 从标准输入读。**这样管道就能用了**：
        #   echo "被辞退有补偿吗" | juris ask
        # 也是给"问题很长、里面带引号"的场景留的退路——
        # 在 shell 里转义带引号的长中文参数很烦。
        question = sys.stdin.read().strip()
    if not question:
        _eprint("没有给问题。用法：juris ask \"被辞退有补偿吗？\"")
        return 2

    agent = JurisAgent()
    try:
        ans, streamed = _answer_of(agent, question, args)
    except NotReady as e:
        _eprint(str(e))
        return 1
    finally:
        agent.close()

    if args.json:
        _emit_json(ans, trace=args.trace)
    else:
        _print_answer(ans, args, streamed)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """连着聊。

    ## ⚠️ `--session` 不给也要**自己造一个**，不能走"无状态"

    这是想清楚了才这么写的。`client.ask(session_id=None)` 是**一问一答、
    完全无状态**的；如果 `chat` 也这么跑，那它就不是聊天，
    只是"把 `juris ask` 套在一个 while 里"——
    用户说「我在西安，月薪 12000」再问「那能赔多少」，
    第二句会得到一个"您没说月薪"的回答（因为**它真不知道**）。

    而"记账"这件事必须**看得见**：开场把会话 id 打出来。
    悄悄开始的记忆比没有记忆更危险——尤其这是法条咨询，
    记错一个月薪会让后面每一轮都算错，而用户从界面上看不出
    "它开始记我的事了"。所以：**打出来，并且告诉他能用它接着聊。**
    """
    from juris_agent.client import JurisAgent, NotReady

    agent = JurisAgent()

    def _new_session() -> str:
        """造一段新会话 id。

        直接自己生成、不用 `SessionStore.new_id()`：那个方法要先把
        runtime 装起来（可能几十秒）才能拿到，而在开场就打 id
        这件事**不该等模型加载**。`client.ask` 拿到一个不存在的 id 时
        会 `get_or_create` 建出来，所以自己造一个完全等价。
        """
        import secrets

        return f"s-cli-{secrets.token_hex(4)}"

    session_id = args.session or _new_session()
    # 把它写回 args，因为 `_answer_of` 是从传进去的 Namespace 上读 `session` 的。
    # **只有这一处赋值**，所以不会出现"界面上显示的 id 和实际用的 id 不是一个"。
    args.session = session_id

    try:
        health = agent.health()
        if not health.get("ready"):
            _eprint("❌ 装不起来：")
            for p in health.get("problems", []):
                _eprint(f"  - {p}")
            _eprint("")
            _eprint(_FIX_HINTS)
            return 1

        _eprint("劳动法咨询（/exit 退出，/new 新会话，/help 帮助）")
        _eprint(f"会话 id：{session_id}   （想接着聊：juris chat --session {session_id}）")
        while True:
            try:
                # ⚠️ **提示符写到 stderr，不用 `input("> ")`。**
                # `input` 的 prompt 参数是往 **stdout** 写的，于是
                # `juris chat < 问题列表.txt > 答案.txt` 的时候，
                # 存下来的答案文件里会夹着一串孤零零的 `> `——
                # 而答案是**按行对应**的，多出来的行会让整个对不上。
                # 写到 stderr，在终端里看起来一模一样（两个流都到屏幕），
                # 重定向时 stdout 就干净了。
                #
                # `\n` 单独先写：让每轮之间空一行，和人读对话的习惯一致。
                sys.stderr.write("\n> ")
                sys.stderr.flush()
                line = input().strip()
            except EOFError:
                # Ctrl-D / 管道读完。**当成正常退出，不是错误**——
                # `echo "问题" | juris chat` 就是靠这个才走得通。
                _eprint("")
                break
            except KeyboardInterrupt:
                # Ctrl-C 在读输入的位置。打一个新行、正常退出。
                _eprint("")
                break

            if not line:
                continue
            if line in ("/exit", "/quit", "/q"):
                break
            if line == "/help":
                _eprint(_CHAT_HELP)
                continue
            if line == "/new":
                session_id = _new_session()
                args.session = session_id
                _eprint(f"（开了一段新会话：{session_id}）")
                continue
            if line == "/session":
                _eprint(f"当前会话：{session_id}")
                continue

            try:
                ans, streamed = _answer_of(agent, line, args)
            except KeyboardInterrupt:
                # ⚠️ 这一句要**接着循环**，不能 break。
                # 用户按 Ctrl-C 想打断的是"这一轮正在生成"，不是"退出程序"。
                # 打断之后会话里的历史是**这一轮的上一轮**（因为写回没发生），
                # 所以问他下一句时上下文仍然是干净的——不会留下半截对话。
                _eprint("\n（打断了这一轮）")
                continue
            except NotReady as e:
                _eprint(str(e))
                break

            _print_answer(ans, args, streamed)
    finally:
        # **必须关**。REPL 里连着问十几句，每一句都可能新建过连接，
        # 不关的话这个进程退出前都占着。
        agent.close()
    return 0


_CHAT_HELP = """/exit     退出
/new      开一段新会话（清掉上下文和记住的月薪/工龄/地点）
/session  看当前会话 id
/help     这一屏

直接输入问题即可。同一个会话里它会记住你前面说过的信息，
所以「我在西安，月薪 12000」之后直接问「那能赔多少」就够了。"""


def cmd_serve(args: argparse.Namespace) -> int:
    """起 Web 服务（含前端）。**演示时用这个。**"""
    try:
        import uvicorn
    except ImportError:
        _eprint("没装 uvicorn。跑 `uv sync`（它在核心依赖里，不用 --extra ml）。")
        return 1

    # ⚠️ **reload 只能配 import string，不能配 app 对象。**
    # uvicorn 的 reload 是靠"起一个子进程、监视文件变化后重启它"实现的，
    # 而重启是**重新 import 一遍**——它必须知道去 import 什么。
    # 直接传 `create_app()` 造好的对象，子进程重启时拿到的还是父进程里那个
    # （已经被新代码之前的旧代码 import 过了），**改代码不生效，且不报错**。
    # 症状是"我明明改了，怎么没变化"——这是 reload 最容易踩的一个坑。
    _eprint(f"起服务：http://{args.host}:{args.port}/")
    _eprint("首次启动要加载嵌入模型（几秒到几分钟），浏览器里会先转圈。")
    uvicorn.run(
        "juris_agent.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


# --------------------------------------------------------------------------- #
# 参数解析
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """造解析器。**抽出来是为了让测试能直接测参数，不用起进程。**"""
    parser = argparse.ArgumentParser(
        prog="juris",
        description="可溯源劳动法咨询 Agent。",
        epilog="例：juris ask \"被辞退有补偿吗？\"   /   juris doctor",
        # ⚠️ **中文 argparse 的一个小坑**：`formatter_class` 默认会对帮助文本
        # 做换行，而它按**字符**数算，一个中文字算一个——所以中文帮助看起来
        # 比英文短很多就折行了，或者反过来很宽。用 RawDescriptionHelpFormatter
        # 保住自己在 epilog 里写好的换行。这里没做复杂的，只是记一笔：
        # 真嫌难看再去动 `TextWrapper` 的 width。
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=_version())
    sub = parser.add_subparsers(dest="cmd", metavar="<命令>")

    # -- ask --------------------------------------------------------------- #
    p_ask = sub.add_parser(
        "ask", help="问一个问题，拿到答案就走", description="问一个问题，拿到答案就走。"
    )
    p_ask.add_argument(
        "question",
        nargs="?",
        help="要问的问题。**省略或写 `-` 就从标准输入读**（方便管道和长问题）。",
    )
    _add_common(p_ask)
    p_ask.add_argument("--json", action="store_true", help="输出一行完整的 JSON，不打人话。")
    p_ask.set_defaults(func=cmd_ask)

    # -- chat -------------------------------------------------------------- #
    p_chat = sub.add_parser(
        "chat", help="连着聊（交互式）", description="连着聊。留空行忽略，/exit 退出。"
    )
    _add_common(p_chat, session=False)
    p_chat.add_argument(
        "--session",
        default=None,
        help=(
            "会话 id。**不给我会自动建一段**，并在开场打给你看——"
            "记住它就能 `juris chat --session <那个 id>` 接着上次聊。"
        ),
    )
    p_chat.set_defaults(func=cmd_chat)

    # -- serve ------------------------------------------------------------- #
    p_serve = sub.add_parser("serve", help="起 Web 服务（含前端界面）")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument(
        "--reload", action="store_true", help="改代码自动重启（开发时开，演示时别开）。"
    )
    p_serve.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    p_serve.set_defaults(func=cmd_serve)

    # -- doctor ------------------------------------------------------------ #
    p_doctor = sub.add_parser(
        "doctor",
        help="体检环境：缺什么、缺在哪一步",
        description="体检环境。退出码 0 = 能答问题，1 = 不能。",
    )
    # 体检最需要看日志（"装不起来"的原因往往就在模型加载那几行里），
    # 所以这个旗对 doctor 格外有用。但**默认还是关**：
    # 装好的人看到的应该是一屏干净的结论，而不是几百行 HTTP 重定向。
    p_doctor.add_argument(
        "--verbose",
        action="store_true",
        help="放出第三方库的日志（模型加载、HTTP 请求）。装不起来时加它。",
    )
    p_doctor.set_defaults(func=cmd_doctor)

    return parser


def _add_common(p: argparse.ArgumentParser, *, session: bool = True) -> None:
    """`ask` / `chat` 共用的那几面旗。**抽出来免得两边漂移。**

    ⚠️ `session=False` 那个参数不是多余的分支：`chat` 自己声明 `--session`
    （它的 help 要说的事不一样——"不给就是每问独立"对 `chat` 是错的，
    它会自动建一段会话）。**同一个 `--session` 在两个子解析器上都注册的话，
    argparse 不报错**，而是在**解析用户输入的时候**才炸
    （`argument --session: conflicting option string`）——
    所以这个坑的位置很靠后：单元测试里造出来的对象没事，
    跑到 `juris chat --session x` 才炸。
    """
    if session:
        p.add_argument(
            "--session",
            default=None,
            help="会话 id（多轮）。不传就是无状态一问一答。",
        )
    p.add_argument(
        "--stream",
        action="store_true",
        help="边生成边打（打字机效果）。关掉的话是等全部算完再一次性打。",
    )
    p.add_argument("--trace", action="store_true", help="打执行轨迹：走了哪几个节点、各花多久。")
    p.add_argument(
        "--no-cost",
        action="store_true",
        help="不打成本那一行。**它本来就在 stderr 上**，重定向 stdout 就看不到了。",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "放出第三方库的日志（httpx 的每次请求、模型加载过程）。"
            "**它们默认是关的**——理由见 cli.py 里 `_NOISY_LOGGERS` 那段注释："
            "不关的话它们会混进 stdout，把 `> answer.txt` 存下来的答案弄脏。"
            "报 bug 时附上 `--verbose` 的输出。"
        ),
    )


def _version() -> str:
    """版本号。**拿不到就给 `?`，不抛异常。**

    `--version` 是个"人随手敲一下"的命令，为了打一行版本号而抛
    `PackageNotFoundError`（包以非安装方式被 import 时会这样）是件很蠢的事。
    """
    try:
        from importlib.metadata import version

        return f"juris {version('juris-agent')}"
    except Exception:  # noqa: BLE001
        return "juris ?"


def main(argv: list[str] | None = None) -> int:
    """入口。**返回退出码**（`__main__` 那边会把它交给 `sys.exit`）。

    ⚠️ **不在这里 `sys.exit()`。** 那样做的后果是：`main()` 变得没法被测试调用
    （`SystemExit` 会从任何断言中间穿出来），而且 `[project.scripts]` 那个入口
    需要的也只是"一个返回 int 的函数"——console_scripts 生成的包装器
    会自己把返回值当退出码用（`sys.exit(main())`）。
    """
    _force_utf8()

    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "cmd", None):
        # 一个子命令都不给：打帮助，**退出码 2**（argparse 对"用法错误"用的就是 2）。
        # 用 0 的话 `juris || echo 失败` 就是哑的。
        parser.print_help(sys.stderr)
        return 2

    # ⚠️ 位置要在**所有**子命令真正干活之前，而且**顺序是"先分流、再按住"**：
    #
    # ① 分流：所有日志改走 **stderr**。这一步不在这里做的话，日志会跟着
    #    答案一起进 stdout（`observability/logs.py` 的默认值是 stdout，
    #    那对服务端是对的），`--json | jq` 当场就废；
    # ② 按住：第三方库那些刷屏的日志压到 WARNING。
    #
    # 两步都**必须**赶在第一批日志产生之前，因为流一旦被写脏就收不回来了。
    _setup_logging(verbose=getattr(args, "verbose", False))

    # ⚠️ **在解析器这一层兜住 Ctrl-C。** 放在各子命令里的话，
    # 每个命令都要抄一遍这个 try，而漏掉一个的症状是
    # "在那一屏按 Ctrl-C 会打出一大坨 traceback"——尤其难看，
    # 因为 Ctrl-C 是**正常操作**，不是错误。
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        _eprint("\n（已中断）")
        return 130  # 130 = 128 + SIGINT，shell 的惯例


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
