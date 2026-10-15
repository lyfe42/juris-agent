"""F31 成本报表：钱花在哪、一次问答多少钱、少注入几条值不值。

产出 `reports/cost.md`（人看的）+ `reports/cost.json`（机器看的）。

## 四张表，各回答一个问题

| 表 | 回答的问题 | 谁在意 |
|---|---|---|
| 一、按阶段拆 | **钱花在哪一步** | 想省钱的人（改哪一步才有用） |
| 二、缓存命中 | prompt 缓存到底省了没有 | 被「开缓存能省 90%」这句话坑过的人 |
| 三、单次问答成本 | **一次问答几毛钱**、波动多大 | 拍预算的人 |
| 四、成本 vs 召回 | 少注入几条，省下的钱值不值掉的分 | 做取舍的人 |

## 为什么这张表必须真调模型（不能用 replay）

压测（`bench_qps.py`）可以离线跑，这张表不行。原因是 `LLMClient`
**在缓存命中时刻意不记账**（见 `llm.py` 里那段「不要累加 `hit.usage`」）——
那是对的（读磁盘确实没花钱），但代价是**重放模式下拿不到分项 token**，
而这张表的每一个格子都要靠分项 token 算出来。

所以命令行只开放 `--mode off`（真调）。跑一次二十几个问题，花几块钱。
**别用 replay 省这点钱**——那样报表里会是一堆 0，
而 0 在这张报表里有三种互不相同的含义，读者分不出来。

## 阶段是怎么分出来的

`LLMClient` 自己不知道「这次调用是干什么的」——它只收到 messages 和 tools。
所以这里包一层，**按调用特征反推阶段**（见 `classify`）：

- 带了 `tools=` → **工具循环**（`nodes.py` 的 `tool` 节点）
- system 提示等于 `ANSWER_SYSTEM` → **生成答案**
- system 提示等于 `CHITCHAT_SYSTEM` → **闲聊**
- 走 `complete_json` → **查询改写**（只有 `rewrite=llm` 才发生，默认不开）

这是**约定而非契约**：哪天有人改掉那几个 system 提示的开头，
分类会静默变成「其他/未分类」。所以报表里单列一行，**它不为空就是报警**——
而不是让它悄悄并进别的阶段里。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from juris_agent.agent import prompts  # noqa: E402
from juris_agent.agent.graph import make_runner  # noqa: E402
from juris_agent.config import ensure_dirs, get_settings  # noqa: E402
from juris_agent.observability import configure_logging, load_pricing  # noqa: E402

# 混合题：**只用一种题型测成本会得出错误的「平均成本」**——
# 闲聊题一次调用就完（而且短），计算题要跑工具循环（两三次调用、
# 每次输出都被思维链撑满），两者差好几倍。
#
# ⚠️ **下面每条后面的注释是「这题想覆盖什么」，不是「它会走哪条路」。**
# 走哪条路由由 `configs/router.yaml` 的词表决定，不归这张表管——
# 第一次跑的时候「试用期最长可以约定多久」就被判成了 `calc` 而不是 `law`。
# 报表里印的是**实际路由**，所以以那张表为准；这里写期望只是为了说明选题意图。
QUESTIONS = [
    "试用期最长可以约定多久",         # 短事实题
    "经济补偿怎么算",                 # 计算题（要跑工具循环）
    "被违法辞退能拿多少赔偿",         # 计算题 + 多法条
    "竞业限制没约定补偿怎么办",       # 法条问答
    "劳动合同到期不续签有补偿吗",     # 法条问答
    "加班费怎么算",                   # 计算题
    "帮我写一首关于秋天的诗",         # 超纲 → 本该在规则层拒答，一次模型都不调
    "你好",                           # 闲聊
]

# 成本-召回曲线的横轴。8 是生产默认值（`config.py` 的 `inject_k`）。
DEFAULT_KS = [2, 4, 8, 12]

# 曲线用前几条题。**同一批题在几个 k 上各跑一遍**，才有「同题对照」的意义；
# 换一批题跑，token 的变化就分不清是"多注入了法条"还是"这题本来就更长"。
SWEEP_N = 3


# --------------------------------------------------------------------------- #
# 记录
# --------------------------------------------------------------------------- #


@dataclass
class CallRecord:
    """一次模型调用。**所有字段都直接从 `Usage` 抄，不做任何推算**——
    报表里的数必须能和日志对上，否则没人能验证它。"""

    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cache_hit_tokens: int
    cache_miss_tokens: int
    ms: float
    got_usage: bool = True
    """**这次调用拿没拿到 usage。**

    缓存命中/重放时供应商那边不会回 usage，客户端也刻意不记账
    （读磁盘本来就没花钱）。那种情况下 token 全是 0——
    而 0 是「这一次没花 token」还是「没拿到数」，是两件事，
    报表里必须分开说，所以留这个字段。
    """


@dataclass
class TurnRecord:
    """一个问题的完整一轮。"""

    question: str
    calls: list[CallRecord] = field(default_factory=list)
    ms: float = 0.0
    route: str = ""
    finish: str = ""
    answer_chars: int = 0
    ok: bool = True

    @property
    def tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for c in self.calls)


def classify(messages, kw: dict, method: str) -> str:
    """反推这次调用属于哪个阶段。**顺序不能换**，理由逐条在下面。

    ① `complete_json` 先判：它压根不走 `chat`，特征最明确。
    ② 带 `tools` 的一律是工具循环：只有 `tool` 节点会传 tools。
       按 system 提示判也行（它用的是 `AGENT_SYSTEM`），但「有没有 tools」
       不会因为有人改文案而失效。
    ③ 再按 system 提示判生成 / 闲聊。
    ④ 都不匹配 → 「其他/未分类」。**不猜**——猜错会把钱记到别的阶段头上，
       比「有一行没分类」难查得多（前者是错的数，后者是缺的数）。
    """
    if method == "complete_json":
        return "查询改写"
    if kw.get("tools"):
        return "工具循环"
    system = ""
    for msg in messages or []:
        if msg.get("role") == "system":
            system = str(msg.get("content") or "")
            break
    if system == prompts.ANSWER_SYSTEM:
        return "生成答案"
    if system == prompts.CHITCHAT_SYSTEM:
        return "闲聊"
    return "其他/未分类"


class RecordingLLM:
    """包一层 `LLMClient`，把每次调用的 token 和耗时记下来。

    ## 为什么用包装，而不是去改 `LLMClient`

    改客户端意味着「报表要用的东西」进了在线路径——它会跟着上线、跟着出 bug，
    而且改坏了是**服务**坏，不是**报表**坏。包一层的话，报表有问题就只是报表有问题。

    ## 每次调用前后各开一次账

    `begin_turn()` + `turn_usage` 这套是**按线程**记的（见 `LLMClient.begin_turn`）。
    这里在每次调用前后各做一次，拿到的正好是「这一次调用花了多少」，
    而不是「这一轮花了多少」。比调用前后做差可靠——做差在并发下会把
    别人的花销算进来（`begin_turn` 的 docstring 里记着这个坑），
    **同一个坑不该因为「这次是单线程」就再踩一遍**。

    ## `__getattr__` 转发

    除了 `chat` / `complete_json`，别的方法（`begin_turn`、`mode`、`close`……）
    原样转发给里面的真客户端。不转发的话，`runner` 里任何一处用法都会
    `AttributeError`——报错还好，**可怕的是 `getattr(llm, "turn_usage", None)`
    这种带默认值的调用**，它会静默拿到 `None` 然后一路降级。
    """

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[CallRecord] = []

    def __getattr__(self, name):
        # `inner` 自己要走**普通查找**，不能落到 `__getattr__` 里，否则
        # 万一 `inner` 还没被赋值（反序列化、`copy` 之类的路径），
        # 这里会 `getattr(self.inner...)` → 又进 `__getattr__` → 递归到爆栈。
        # 这种情况下报「没有 inner」比无限递归好查得多。
        if name == "inner":
            raise AttributeError("RecordingLLM.inner 尚未初始化")
        return getattr(self.inner, name)

    def _record(self, stage: str, t0: float) -> None:
        usage = self.inner.turn_usage
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        self.calls.append(
            CallRecord(
                stage=stage,
                model=str(getattr(self.inner, "model_gen", "") or ""),
                prompt_tokens=pt,
                completion_tokens=ct,
                reasoning_tokens=int(getattr(usage, "reasoning_tokens", 0) or 0),
                cache_hit_tokens=int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0),
                cache_miss_tokens=int(getattr(usage, "prompt_cache_miss_tokens", 0) or 0),
                ms=(time.perf_counter() - t0) * 1000,
                got_usage=bool(pt or ct),
            )
        )

    def chat(self, messages, **kw):
        stage = classify(messages, kw, "chat")
        self.inner.begin_turn()
        t0 = time.perf_counter()
        try:
            return self.inner.chat(messages, **kw)
        finally:
            # **放 `finally` 里**：调用抛异常（限流、超时）时也要记一笔。
            # 失败调用**也可能已经计费**（供应商扣了钱、我们没拿到 usage），
            # 那种情况这里会记成「没拿到 usage」——所以报表里必须写明
            # 「只统计拿得到 usage 的调用」。
            self._record(stage, t0)

    def complete_json(self, *a, **kw):
        stage = classify([], kw, "complete_json")
        self.inner.begin_turn()
        t0 = time.perf_counter()
        try:
            return self.inner.complete_json(*a, **kw)
        finally:
            self._record(stage, t0)


# --------------------------------------------------------------------------- #
# 跑
# --------------------------------------------------------------------------- #


def _g(state, key: str, default=None):
    """状态取值。图的状态在不同路径上可能是 dict 也可能是对象，两边都兜住。"""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def run_questions(runner, llm: RecordingLLM, questions: list[str], *,
                  model: str, pricing, label: str = "") -> list[TurnRecord]:
    """把几个问题各跑一遍，返回逐轮的记录，顺便把每轮打在终端上。

    **一轮的边界靠 `llm.calls = []` 划出来**，不是靠"记下调用时的计数器"——
    后者要维护一个跨层的序号，而且失败重入时容易算漏。
    """
    out: list[TurnRecord] = []
    for q in questions:
        llm.calls = []
        t0 = time.perf_counter()
        final = None
        try:
            for mode, payload in runner.stream(q, with_values=True):
                if mode == "values":
                    final = payload
        except Exception as e:  # noqa: BLE001 — 一轮挂了不该毁掉整张报表
            rec = TurnRecord(question=q, calls=list(llm.calls),
                             ms=(time.perf_counter() - t0) * 1000, ok=False,
                             finish=f"{type(e).__name__}")
            out.append(rec)
            print(f"  [{label}] {q[:14]:<16} 失败：{type(e).__name__}: {e}")
            continue

        rec = TurnRecord(question=q, calls=list(llm.calls),
                         ms=(time.perf_counter() - t0) * 1000)
        rec.route = str(_g(final, "route", "") or "")
        rec.finish = str(_g(final, "finish_reason", "") or "")
        rec.answer_chars = len(str(_g(final, "answer", "") or ""))
        out.append(rec)
        money = money_of(rec.calls, model, pricing)
        print(f"  [{label}] {q[:14]:<16} {len(rec.calls)} 次调用 "
              f"{rec.tokens:>6} token  ¥{money:.6f}  {rec.ms / 1000:>5.1f}s  {rec.finish}")
    return out


def money_of(calls: list[CallRecord], model: str, pricing) -> float:
    """一组调用一共多少钱。

    ⚠️ **这里自己乘，没有走 `observability.estimate_cost`**：
    那个函数是**在线路径**用的，要处理「拿不到分项」的各种退路
    （见它的 docstring），而这里手上握着完整分项，再走一遍退路逻辑
    只会把「我确实有这个数」和「我猜的」混成一锅。

    但**单价和折扣系数仍然抄同一张牌价表、按同一套算法**（命中缓存的输入
    便宜 `cache_hit_discount` 倍），所以报表上的钱和日志里 `describe_cost`
    报的钱是同一个口径——**两个地方报出两种价钱，是最容易让人不信任报表的做法**。
    """
    total = 0.0
    for c in calls:
        price = pricing.price_for(c.model or model)
        hit, miss = c.cache_hit_tokens, c.cache_miss_tokens
        if hit or miss:
            billable_in = hit * pricing.cache_hit_discount + miss
        else:
            # 供应商没回缓存分项（或这是重放/假模型）→ 只能按全价算输入。
            # **这是"可能高估"，不是"可能低估"**：真命中过缓存的话我们多算了钱。
            billable_in = float(c.prompt_tokens)
        total += (billable_in * price.input + c.completion_tokens * price.output) / 1e6
    return total


def recall_by_k(pipeline, gold, ks: list[int]) -> dict[int, dict]:
    """每个 k 的「注入 k 条时，gold 在里面吗」。

    ## 为什么自己算，不拿 `reports/ablation.md` 里的 hit@5

    消融表那几列是**固定的 k**（1/5/10），而这根曲线的横轴是 `inject_k` 本身——
    **我要的是「注入 3 条时的召回」，消融表里没有这一列**。
    硬拿 hit@5 当 hit@3 用，就是把横轴编出来的。

    ## 为什么只检索一遍

    `inject_k` 在生产里是**把已经排好序的结果切前 k 条**（见 `nodes.py` 的
    `retrieve`：`r.search(question, top_k=inject_k)`），管道本身一直用宽的
    `top_k/fused_k`。所以「k=2 的结果」就是「k=12 的结果的前 2 条」——
    一遍检索切五次，比检索五遍快 5 倍，而且**切出来的东西逐字节相同**。

    口径和消融表一致：同一个 `expand_article_keys`（条级展开、去重保序）、
    同一份 gold。不一致的话，两张表上的同一个名字会指两件不同的事。
    """
    ranked: dict[str, list[str]] = {}
    for i, g in enumerate(gold):
        if i == 0:
            pipeline.search(g.query)  # 首次含模型预热，不计也没关系，这里只算召回
        res = pipeline.search(g.query)
        keys = res.expand_article_keys() if hasattr(res, "expand_article_keys") else []
        ranked[g.id] = list(keys)

    out: dict[int, dict] = {}
    for k in ks:
        hit = 0
        art = 0.0
        for g in gold:
            top = ranked.get(g.id, [])[:k]
            gold_set = set(g.gold)
            if any(x in gold_set for x in top):
                hit += 1
            art += (len(set(top) & gold_set) / len(gold_set)) if gold_set else 0.0
        n = len(gold) or 1
        out[k] = {"n": len(gold), "hit": hit / n, "art": art / n}
    return out


# --------------------------------------------------------------------------- #
# 报表
# --------------------------------------------------------------------------- #


def pct(x: float, total: float) -> str:
    return f"{x / total * 100:.1f}%" if total else "—"


def p_at(vals: list[float], p: float) -> float:
    """分位数。**样本只有个位数**，所以用"最近秩"而不是插值——
    插值会给出一个不存在的第 95 百分位成本，看着更精确，其实是编的。"""
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


# 阶段的**固定顺序**。不按 token 多少排序：这份报表会被前后对比，
# 行的顺序一变，两次对比就得靠眼睛找行。
STAGE_ORDER = ["工具循环", "生成答案", "闲聊", "查询改写", "其他/未分类"]

STAGE_NOTE = {
    "工具循环": "模型挑工具、填参数（只有计算题会走）",
    "生成答案": "拿着检索到的法条写答案（**产出的主体在这里**）",
    "闲聊": "寒暄（固定文案也能兜，有模型就用模型回）",
    "查询改写": "把问题改写成几个检索式（默认关，开了很贵）",
    "其他/未分类": "⚠️ 分类规则没覆盖到的调用——**这一行不为零就是报警**",
}


def stage_table(turns: list[TurnRecord], model: str, pricing) -> dict:
    """表一的**数据**部分：按阶段拆钱。

    ⚠️ **只返回数据，不返回 markdown。** 报表的正文由 `render_md(payload)`
    从**存下来的 JSON** 生成——这样改一处措辞就不用再调一遍模型。
    第一次写这个脚本时两者是混在一起的，代价是「改个错别字要再花一次 API 钱」。
    """
    buckets: dict[str, list[CallRecord]] = {}
    for t in turns:
        for c in t.calls:
            buckets.setdefault(c.stage, []).append(c)

    rows: dict[str, dict] = {}
    total = 0.0
    for stage, calls in buckets.items():
        money = money_of(calls, model, pricing)
        total += money
        rows[stage] = {
            "calls": len(calls),
            "prompt": sum(c.prompt_tokens for c in calls),
            "completion": sum(c.completion_tokens for c in calls),
            "reasoning": sum(c.reasoning_tokens for c in calls),
            "cache_hit": sum(c.cache_hit_tokens for c in calls),
            "ms_p50": round(statistics.median([c.ms for c in calls]), 1),
            "cny": round(money, 6),
        }
    return {"total_cny": round(total, 6), "by_stage": rows}


def stage_lines(stages: dict) -> list[str]:
    """表一的 **markdown 行**（表格头 + 各行 + 合计）。"""
    rows: dict = stages.get("by_stage", {})
    total: float = stages.get("total_cny", 0.0)
    lines = [
        "| 阶段 | 干什么的 | 调用 | 输入 token | 输出 token | 其中思考 | 缓存命中输入 | 金额 | 占成本 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    # 先按固定顺序列出已知阶段，再把没见过的阶段追加在后面。
    # **固定顺序是为了让两次运行的报表能上下对齐**——不排的话，
    # 顺序随字典插入顺序变，对比就得靠眼睛找行。
    for stage in STAGE_ORDER + [s for s in sorted(rows) if s not in STAGE_ORDER]:
        r = rows.get(stage)
        if r is None:
            continue
        lines.append(
            f"| {stage} | {STAGE_NOTE.get(stage, '—')} | {r['calls']} | {r['prompt']:,} | "
            f"{r['completion']:,} | {r['reasoning']:,} | {r['cache_hit']:,} | "
            f"¥{r['cny']:.6f} | {pct(r['cny'], total)} |"
        )
    if rows:
        lines.append(
            f"| **合计** | | **{sum(r['calls'] for r in rows.values())}** | "
            f"**{sum(r['prompt'] for r in rows.values()):,}** | "
            f"**{sum(r['completion'] for r in rows.values()):,}** | "
            f"**{sum(r['reasoning'] for r in rows.values()):,}** | "
            f"**{sum(r['cache_hit'] for r in rows.values()):,}** | "
            f"**¥{total:.6f}** | 100% |"
        )
    return lines


def render_md(payload: dict) -> str:
    """**从存下来的 JSON 生成整份 markdown。**

    这样拆开的好处很实际：措辞改十遍也不用再调一次模型。
    （代价是 payload 必须自带正文要用到的一切——所以下面每处取数
    都用 `.get(..., 默认值)`，宁可印出一个明显的占位符，
    也不要因为缺一个键就整份报表崩掉。）
    """
    p = payload
    pr = p.get("pricing", {})
    model = p.get("model", "?")
    stages = p.get("stages", {})
    cache = p.get("cache", {})
    per_q = p.get("per_question", [])
    sweep = p.get("sweep", {}) or {}
    ks = sweep.get("per_k", {}) or {}
    rec = sweep.get("recall", {}) or {}

    in_tokens = cache.get("prompt_tokens", 0)
    hit_tokens = cache.get("hit", 0)
    miss_tokens = cache.get("miss", 0)
    no_usage = cache.get("no_usage_calls", 0)
    in_cost_full = cache.get("cost_full", 0.0)
    in_cost_real = cache.get("cost_real", 0.0)
    cache_saved = cache.get("saved", 0.0)
    hit_rate = pct(hit_tokens, in_tokens)

    # 「命中发生在哪一步」按阶段列出来。**一次都没命中时也要写一行**，
    # 否则报表里会出现一个只有标题、底下空着的段落——读者分不清
    # 是「没有命中」还是「生成器漏了」。
    hit_lines = [
        f"- **{s.strip()}**：{r['cache_hit']:,} token"
        for s, r in stages.get("by_stage", {}).items()
        if r.get("cache_hit")
    ] or ["- **一次都没命中。**（那这张表里的命中率就是 0，不是缺数。）"]

    # **「最贵 / 最便宜」只在真花过钱的题之间比。**
    # 拒答题是 0 元，把它算进来的话「倍数」会变成除零（第一次跑就印出了
    # 「差了 nan 倍」这种一眼假的东西）。0 元那一档单独一段说，
    # 因为它不是一个「价格」，是「没有发生交易」。
    paid = [q for q in per_q if q.get("cny", 0.0) > 0]
    zero_qs = [q for q in per_q if not q.get("cny", 0.0)]
    richest = max(paid, key=lambda q: q.get("cny", 0.0), default=None)
    cheapest = min(paid, key=lambda q: q.get("cny", 0.0), default=None)
    # 路由名是 `Route` 那个 Literal 里的 `out_of_scope`——**不是 `refuse`**。
    # 第一次写成 `== "refuse"` 时，报表明明有一行 0 次调用的超纲题，
    # 却印出「本次没有走到 refuse 的题」。两个名字都认，免得以后再踩。
    refuse_qs = [q for q in per_q if q.get("route") in ("out_of_scope", "refuse")]
    n_calls = sum(q.get("calls", 0) for q in per_q)

    md = [
        "# F31 成本报表",
        "",
        "> **一句话**：这份报表回答「一次问答要花多少钱、钱花在哪一步、"
        "少注入几条法条值不值」。",
        "",
        "> ⚠️ **先说清楚三件事，否则这张表的数会被误用：**",
        "> ① **这是估算，不是账单。** 算的是「把这段 prompt 发过去要花多少」；",
        ">    不含重试、不含失败但已计费的调用、不含供应商侧的口径差异。",
        "> ② **每个数字都绑着一天的牌价**（见「测量条件」里的 `as_of`）。",
        ">    没有日期的价格数字，三个月后就是一段没法验证的历史。",
        "> ③ **这里的模型是推理模型，思考过程按输出计费**——所以「答案短」",
        ">    不等于「便宜」。表里专门有一列「其中思考」，",
        ">    它就是「看着没说什么，钱却花掉了」的那部分。",
        "",
        "## 测量条件",
        "",
        f"- 生成模型：`{model}`；牌价表 `{pr.get('file', 'pricing.yaml')}`，"
        f"**as_of = {pr.get('as_of', '?')}**"
        f"（输入 {pr.get('input')}/百万、输出 {pr.get('output')}/百万，"
        f"{pr.get('currency', 'CNY')}）",
        f"- 主报表：**{p.get('n_ok', 0)}/{p.get('n_questions', 0)} 题成功**，"
        f"`inject_k = {p.get('inject_k', '?')}`（生产默认）",
        f"- 题目是 **{p.get('n_questions', 0)} 条混合题**"
        "（法条问答 / 计算题 / 超纲拒答 / 闲聊）——"
        "只用一种题型会得出错误的「平均成本」",
        f"- 缓存折算系数 `hit_discount = {pr.get('cache_hit_discount', 1.0)}`"
        "（**只影响钱那一列，不影响任何指标**）",
        "- 模式：**真调 API**。理由见 `scripts/cost_report.py` 开头——",
        "  重放模式下客户端刻意不记 token，分项数字会全是 0。",
        "",
        "## 一、钱花在哪：按阶段拆",
        "",
        *stage_lines(stages),
        "",
        "怎么读这张表：",
        "",
        "- **阶段是按调用特征反推的**，不是模型自己报的——`LLMClient` 只收到",
        "  messages，它不知道这次调用是「挑工具」还是「写答案」。",
        "  判定规则写在 `cost_report.py` 的 `classify()` 里，很短，值得看一眼。",
        "- **「其他/未分类」必须是 0 行。** 它不是「其他开销」，而是",
        "  **「分类规则破了」的报警灯**——真出现非零，说明有人改了那几个 system 提示，",
        "  而报表不会报错，只会安静地把钱记到这一行。",
        "- **「其中思考」是输出的一部分**（供应商就是这么计的），不额外加钱。",
        "  单独列出来是为了回答「答案很短为什么还这么贵」。",
        "- **`生成答案` 那一行是省钱的靶心。** 它的输入 token 由两部分拼成：",
        "  固定的提示词 + 检索出来注入的法条。后者是唯一能大改的旋钮，见第四章。",
        "",
        "## 二、prompt 缓存到底省了没有",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| 输入 token 合计 | {in_tokens:,} |",
        f"| 其中命中缓存 | {hit_tokens:,} |",
        f"| 其中未命中 | {miss_tokens:,} |",
        f"| **缓存命中率** | **{hit_rate}** |",
        f"| 输入这一项的钱：命中部分不打折 | ¥{in_cost_full:.6f} |",
        f"| 输入这一项的钱：实际（命中部分打 "
        f"{pr.get('cache_hit_discount', 1.0)} 折） | ¥{in_cost_real:.6f} |",
        f"| **缓存省掉的输入开销** | **{cache_saved * 100:.1f}%** |",
        f"| 没拿到 usage 的调用 | {no_usage} 次 |",
        "",
        "命中都发生在哪一步（也对应第一章那张表的「缓存命中输入」列）：",
        "",
        *hit_lines,
        "",
        "> ### 这里要修正一个很常见的直觉",
        ">",
        "> 本节初稿写的是「命中率应该很低，缓存省不了钱」。",
        f"> **实测把它推翻了**：命中率 {hit_rate}，"
        f"输入开销省掉 {cache_saved * 100:.1f}%。",
        "> 「开了 prompt 缓存能省一大截」这句话**方向是对的**——",
        "> 但**理由几乎总是说错的**，而理由决定了它**能不能扩**。",
        ">",
        "> 缓存只对「**前缀一模一样**」的请求有效。本项目真正命中的是两种前缀：",
        ">",
        "> ① **同一题的工具循环里，后一次调用把前一次的整段对话又发了一遍**——",
        ">    工具定义、系统提示、前面几轮的问答，全都是上一次请求的**前缀**。",
        ">    命中的大头在这里（看上面那份分步清单）。",
        "> ② **不同题之间**，工具定义 + 系统提示这一段前缀是相同的，",
        ">    所以从第二题起，第一段就命中。",
        ">",
        "> **这两种都不是「用户越多越省」。** 更准确的说法是",
        "> **「同一段前缀被反复发」**。这带来三个不能忽略的后果：",
        ">",
        "> - **一轮里的第一次调用必定是冷的**（没有任何前缀可命中），",
        ">   省下来的是后面几次。所以**单题成本里那笔「冷启动」是躲不掉的**。",
        "> - **命中率会被「轮数」影响，而不是被「用户数」影响。**",
        ">   工具循环绕的圈越多，重复发的前缀越多，命中率越高——",
        ">   换句话说**这一列会因为模型爱多绕两圈而变好看**，",
        ">   别把它当成「我们优化得好」的证据。要看省钱，看钱那一列。",
        "> - **缓存有存活时间，这里没测。** 两次问答隔久了缓存就过期，",
        ">   命中率会掉。本表是**连续跑**出来的，所以它是一个**偏乐观的上界**。",
        "",
        "## 三、一次问答多少钱",
        "",
        "| 题 | 路由 | 调用次数 | 输入 token | 输出 token | 其中思考 | 金额 | 耗时 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for q in per_q:
        flag = "" if q.get("ok", True) else " ⚠️失败"
        md.append(
            f"| {q.get('question', '?')} | {q.get('route') or '—'}{flag} "
            f"| {q.get('calls', 0)} | {q.get('prompt_tokens', 0):,} "
            f"| {q.get('completion_tokens', 0):,} | {q.get('reasoning_tokens', 0):,} "
            f"| ¥{q.get('cny', 0.0):.6f} | {q.get('ms', 0) / 1000:.1f}s |"
        )

    md += [
        "",
        f"- **中位数（P50）：¥{p.get('cost_p50', 0.0):.6f}**",
        f"- **P95：¥{p.get('cost_p95', 0.0):.6f}**",
    ]
    if richest and cheapest:
        ratio = richest.get("cny", 0.0) / cheapest.get("cny", 1.0)
        md.append(
            f"- **最贵 / 最便宜：¥{richest.get('cny', 0.0):.6f}"
            f"（{richest.get('question', '—')}） / ¥{cheapest.get('cny', 0.0):.6f}"
            f"（{cheapest.get('question', '—')}），差了 {ratio:.1f} 倍**"
            f"——**注意这个倍数是拿最贵那条跟最便宜那条比，不是「平均值上下浮动」**"
        )
    if zero_qs:
        md.append(
            f"- **另有 {len(zero_qs)} 条题是 ¥0**"
            f"（{', '.join(str(q.get('question', '?')) for q in zero_qs)}）——"
            "**它们不参与上面那个倍数**：0 元不是一个「价格」，是「这笔交易没发生」。"
            "硬拿它做分母会得到一个假的「无穷倍」"
        )
    md += [
        f"- 合计：¥{p.get('cost_total', 0.0):.6f}"
        f"（{p.get('n_questions', 0)} 题，{n_calls} 次模型调用）",
        "",
        "怎么读这张表：",
        "",
        "- **题与题之间差好几倍，别拿一个「平均价」去拍预算。**",
        "  闲聊题一次调用就完，计算题要跑工具循环（多轮调用，每轮输出都被思考撑满）。",
        "  预算要按**最贵那一档**拍，不是按中位数拍。",
        "- **P95 在这点样本下基本没有统计意义**，它的用途是给出量级，",
        "  不是拿来当 SLO。样本量写在「测量条件」里，别忽略。",
    ]
    # 「拒答到底花不花钱」这一条**三种情况必须分开写**，不能挤成一句三元表达式——
    # 挤成一句的后果就是这里原来那个 bug：一条 refuse 题都没有的时候，
    # `all()` 对空列表返回 True，于是报表同时打出「超纲题 0 次调用」
    # 和「没有 refuse 题」两句**互相矛盾**的话。
    if not refuse_qs:
        md.append(
            "- ⚠️ **本次没有走到 `refuse` 路由的题**，所以「拒答省不省钱」"
            "**这一条没有被验证**。这是本表唯一一处「本该有数、却没有」的地方——"
            "**留着这个空，比编一句「拒答零成本」诚实。**"
        )
    elif all(q.get("calls", 0) == 0 for q in refuse_qs):
        md.append(
            f"- **{len(refuse_qs)} 条超纲题是 0 次调用**——它们在规则层就被挡掉了，"
            "一次模型都没调。这就是「拒答」该有的成本：**0**。"
            "（省的不只是提示词那点钱，是**一整次问答**。）"
        )
    else:
        loose = sum(1 for q in refuse_qs if q.get("calls", 0))
        md.append(
            f"- ⚠️ **{loose}/{len(refuse_qs)} 条超纲题调了模型**——"
            "「拒答」本该在**规则层**完成、一次模型都不调。这里不为零说明"
            "该分的路没分干净，值得改（先查 `configs/router.yaml` 的"
            "`out_of_scope` 词表有没有覆盖到那几句）。"
        )

    if ks:
        order = sorted(ks, key=lambda x: int(x))
        # 拿「最小那个 k」和「生产值」对比——因为实际要做的决定就是
        # 「要不要把 inject_k 从生产值往下调」。生产值不在横轴里时退化成
        # 「最小 vs 最大」，那也是个合法的方向性对比。
        k_prod = p.get("inject_k")
        k_hi = str(k_prod) if k_prod is not None and str(k_prod) in ks else order[-1]
        k_lo = order[0]
        trade_lines: list[str] = []
        if k_lo != k_hi:
            lo, hi = ks[k_lo], ks[k_hi]
            r_lo, r_hi = rec.get(str(k_lo), {}), rec.get(str(k_hi), {})
            c_lo, c_hi = lo.get("cny_per_question", 0.0), hi.get("cny_per_question", 0.0)
            t_lo, t_hi = (lo.get("prompt_tokens_answer_avg", 0.0),
                          hi.get("prompt_tokens_answer_avg", 0.0))
            cost_cut = (1 - c_lo / c_hi) if c_hi else 0.0
            tok_cut = (1 - t_lo / t_hi) if t_hi else 0.0
            hit_lo, hit_hi = r_lo.get("hit"), r_hi.get("hit")
            # 工具循环那一档的钱**完全不随 inject_k 变**，所以它是这条曲线
            # 「省钱天花板」的直接解释。算出来而不是写死，是因为改一次
            # 提示词这个比例就会变。
            tool_cny = stages.get("by_stage", {}).get("工具循环", {}).get("cny", 0.0)
            tool_share = tool_cny / stages["total_cny"] if stages.get("total_cny") else 0.0
            trade_lines = [
                "- **本次实测的取舍（这就是这条曲线的结论）：**",
                f"- 注入条数从 `{k_hi}` 降到 `{k_lo}`：生成阶段的输入 token "
                f"{t_hi:.0f} → {t_lo:.0f}（降 {tok_cut * 100:.1f}%），",
                f"  **每题成本 ¥{c_hi:.6f} → ¥{c_lo:.6f}，只降 {cost_cut * 100:.1f}%**。",
            ]
            if hit_lo is not None and hit_hi is not None:
                art_lo, art_hi = r_lo.get("art", 0.0), r_hi.get("art", 0.0)
                hit_drop = hit_hi - hit_lo
                trade_lines.append(
                    f"- 但 `hit@k` 从 {hit_hi:.3f} 掉到 {hit_lo:.3f}"
                    f"（**掉 {hit_drop * 100:.1f} 个百分点**），"
                    f"`art@k` 从 {art_hi:.3f} 掉到 {art_lo:.3f}。"
                )
                # 判据写成「省钱比例 vs 掉分比例」而不是拍一个直觉数。
                # 两边单位不同（一个是钱的比例，一个是百分点的绝对值），
                # 严格说没法直接比大小——**所以下面这句话只用来定方向，
                # 明细数字都在上面，读者可以自己按别的判据重判。**
                verdict = "代价大于收益：省下的钱比掉掉的召回还少，**这一档不划算**" \
                    if cost_cut < hit_drop else "收益大于代价：省下的钱多过掉掉的召回，**这一档可以谈**"
                trade_lines.append(f"- → {verdict}。")
            trade_lines += [
                f"- **为什么省钱这么有限？看第一章：`工具循环` 那一档占了全场成本的 "
                f"{tool_share * 100:.1f}%，而它完全不随 `inject_k` 变**"
                "（工具循环里塞的是工具定义和前几轮的对话，不是检索结果）。",
                "  **所以「靠少注入法条来省钱」在这个系统里天花板很低。**",
                "  真正有空间的省钱方向在别处：少绕一轮工具循环、把工具定义压短——",
                "  **那两件事改的是占大头的那一档，不是这一条曲线。**",
            ]
        md += [
            "",
            "## 四、成本 vs 召回：少注入几条值不值",
            "",
            "| `inject_k` | 生成阶段平均输入 token | 每题平均成本 | "
            "hit@k（gold 在里面吗） | art@k（gold 覆盖了多少） | 召回样本 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for k in order:
            r = ks[k]
            rr = rec.get(str(k), {})
            md.append(
                f"| {k} | {r.get('prompt_tokens_answer_avg', 0):.0f} | "
                f"¥{r.get('cny_per_question', 0.0):.6f} | "
                f"{rr.get('hit', float('nan')):.3f} | {rr.get('art', float('nan')):.3f} | "
                f"{rr.get('n', 0)} 题 |"
            )
        md += [
            "",
            "怎么读这条曲线：",
            "",
            "- **横轴是「注入几条法条」**，也就是发出去的 prompt 里塞多少上下文。",
            "  这是本项目里**最直接的省钱旋钮**——它同时决定 token 和召回，",
            "  所以「省了多少」和「掉了多少」必须在同一张表里看。",
            "- **两半的样本量不一样，表里分开标了。**",
            "  召回那半边**不调模型、免费**，跑的是全量 gold"
            f"（本次 {sweep.get('n_gold', 0)} 题）；",
            f"  成本那半边要真调模型，只跑了 {sweep.get('sweep_n', 0)} 题。",
            "  **两边不同源**，所以下面的对比只能当量级判断，不是会计。",
            "- **`art@k` 比 `hit@k` 更能说明问题。** 组合题（一问要引好几条）",
            "  少注入时 `hit` 可能还撑着（碰到一条就算命中），",
            "  而 `art` 会开始掉——那才是「答不全」的信号。",
            *trade_lines,
            "- ⚠️ **这条曲线能定方向，不能定小数点。**",
            "  真正的用法是「选一个点，并说清代价」：",
            "  「注入 N 条，成本降到 X%，召回掉 Y 个点」。",
            "  只说「我们做了成本优化」等于什么都没说。",
            "- ⚠️ **hit@k 的分母是全部 gold，而成本只测了几题。**",
            "  想拿这条曲线去定案，先把成本那半边扩到同一批题上（`--sweep-n`）——",
            "  **那要花钱，但只有那样两个数才是同一个总体的。**",
        ]
    else:
        md += [
            "",
            "## 四、成本 vs 召回：少注入几条值不值",
            "",
            "**本次没跑（`--skip-sweep`）。** 这一章要回答的是本项目",
            "最直接的省钱问题——注入几条法条是性价比最高的一档。",
            "**这一栏空着不等于「没有结论」，只等于「这次没测」**，",
            "别把缺数读成好数。",
        ]

    md += [
        "",
        "## 五、这份报表**测不到**什么",
        "",
        "1. **不含重试与失败调用。** 只能统计「我们拿得到 usage」的调用。",
        f"   本次有 {no_usage} 次调用没拿到 usage。",
        "   供应商扣了钱而我们没拿到数的那种，这张表看不见。",
        "2. **不是账单。** 真实花销要看供应商控制台。",
        "3. **一次运行的量，不是「长期平均」。** 题换了、注入条数变了，数就变。",
        "4. **缓存那一章偏乐观。** 同一次运行内部，前缀比线上更容易重复；",
        "   而且这里没测缓存过期。所以命中率是**上界**，不是线上实测值。",
        "",
        "---",
        "",
        f"> 生成命令：`python scripts/cost_report.py`（真调 API；"
        f"本次共 {n_calls} 次模型调用"
        + (f"，其中第四章那 {sweep.get('sweep_n', 0)} 条题各跑了 "
           f"{len(ks)} 个 `inject_k`" if ks else "（本次未跑第四章的曲线）")
        + "）",
        "> 详细数据见 `reports/cost.json`；"
        "**改措辞不用重新调模型**：`--render-only` 会拿那份 JSON 重新生成这份 md。",
        "> 服务吞吐见 `reports/qps.md`；检索各段耗时见 `reports/ablation.md`。",
    ]
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F31 成本报表（真调模型）")
    ap.add_argument("--n", type=int, default=len(QUESTIONS),
                    help=f"主报表跑几个问题（默认 {len(QUESTIONS)}，也就是全部）")
    ap.add_argument("--ks", type=int, nargs="*", default=DEFAULT_KS,
                    help=f"成本-召回曲线的横轴（默认 {DEFAULT_KS}）")
    ap.add_argument("--sweep-n", type=int, default=SWEEP_N,
                    help="曲线上每个 k 跑几个问题（**同一批题**，才有同题对照）")
    ap.add_argument("--skip-sweep", action="store_true",
                    help="跳过第四章的成本-召回曲线（省一半钱和时间）")
    ap.add_argument("--log-level", default="WARNING",
                    help="默认 WARNING：一次问答会打几十行日志，把报表结果淹没")
    ap.add_argument("--out", default="reports/cost.md")
    ap.add_argument("--json-out", default="reports/cost.json")
    ap.add_argument("--render-only", action="store_true",
                    help="**不调模型**，只拿已存在的 --json-out 重新生成 markdown。"
                         "改措辞、改表格样式时用这个，别再花一次 API 钱。")
    args = ap.parse_args(argv)

    # `--render-only` 要在**任何模型/检索加载之前**返回——
    # 它是"只改文字"的路径，不该为此加载 1GB 嵌入模型。
    if args.render_only:
        payload = json.loads(Path(args.json_out).read_text(encoding="utf-8"))
        Path(args.out).write_text(render_md(payload), encoding="utf-8")
        print(f"（--render-only，未调用任何模型）报告已写入 {args.out}")
        return 0

    configure_logging(level=args.log_level)
    import logging

    for noisy in ("httpx", "httpcore", "asyncio", "urllib3", "sentence_transformers",
                  "transformers", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    s = get_settings()
    ensure_dirs(s)
    pricing = load_pricing(s.paths.pricing)
    model = s.model_gen
    price = pricing.price_for(model)

    if not s.paths.laws.exists():
        print("[FATAL] 没有 data/laws.jsonl，先跑: python scripts/ingest_corpus.py",
              file=sys.stderr)
        return 2
    if not s.deepseek_api_key:
        print("[FATAL] 这张报表必须真调模型（理由见文件头）：请先配 DEEPSEEK_API_KEY。",
              file=sys.stderr)
        return 2

    from juris_agent.llm import LLMClient
    from juris_agent.retrieval.build import (ArmDeps, build_default_retriever,
                                             build_pipeline, production_arm)

    print("=" * 78)
    print("F31 成本报表")
    print("=" * 78)
    print(f"  生成模型    : {model}")
    print(f"  牌价表      : {s.paths.pricing.name}（as_of={pricing.as_of}，"
          f"输入 {price.input}/百万、输出 {price.output}/百万）")
    print(f"  注入条数    : {s.retrieval.inject_k}（生产默认）")
    print(f"  主报表题数  : {min(args.n, len(QUESTIONS))}")
    if not args.skip_sweep:
        print(f"  曲线横轴    : inject_k={args.ks} × 每档 {args.sweep_n} 题")
    print("  ⚠️ 这是**真调模型、真花钱**的一次运行。")

    t0 = time.perf_counter()
    retriever = build_default_retriever(settings=s)
    print(f"  检索器装配  : {time.perf_counter() - t0:.1f}s（含模型加载）")

    llm = RecordingLLM(LLMClient())
    runner = make_runner(retriever=retriever, llm=llm, settings=s)

    # ---- 一/二/三：主报表 ---------------------------------------------- #
    qs = QUESTIONS[: args.n]
    print(f"\n--- 主报表（inject_k={s.retrieval.inject_k}）---")
    turns = run_questions(runner, llm, qs, model=model, pricing=pricing, label="主")
    stages = stage_table(turns, model, pricing)

    all_calls = [c for t in turns for c in t.calls]
    in_tokens = sum(c.prompt_tokens for c in all_calls)
    hit_tokens = sum(c.cache_hit_tokens for c in all_calls)
    miss_tokens = sum(c.cache_miss_tokens for c in all_calls)
    # 缓存到底省了多少钱：**同一分项、两种算法，只差一个折扣系数**。
    # 两个数都建立在同一份 token 统计上，所以「省了多少」是同一次运行内部的对比，
    # 不是拿两次不同的运行去比（后者会把题目的随机性也算进来）。
    base_in = hit_tokens + miss_tokens or in_tokens
    in_cost_full = base_in * price.input / 1e6
    in_cost_real = (hit_tokens * pricing.cache_hit_discount + miss_tokens) * price.input / 1e6
    cache_saved = (1 - in_cost_real / in_cost_full) if in_cost_full else 0.0

    # ---- 四：成本-召回曲线 --------------------------------------------- #
    # **先把生产值存下来**：第四章要把 `inject_k` 临时改成各个 k 值，
    # 改完必须还原成「原来那个」而不是「硬编码的 8」——
    # 生产值是可以从环境变量改的（`JURIS_AGENT_RETRIEVAL_INJECT_K`），
    # 写死 8 的话，会悄无声息地污染后面还要用这个 Settings 的代码。
    orig_inject_k = s.retrieval.inject_k
    sweep: dict = {}
    if not args.skip_sweep:
        sweep_qs = QUESTIONS[: args.sweep_n]
        per_k: dict[str, dict] = {}
        for k in args.ks:
            print(f"\n--- 曲线：inject_k={k} ---")
            # 改的是 `Settings` 上的值，所以**必须重建 runner**：
            # runner 持有 deps、deps 持有 settings，改完不重建的话
            # 图读到的还是旧值，而报表上会写着新值——**表和实际不一致**。
            # （副作用：重建 runner 会顺带换掉工具执行器，清掉幂等缓存。
            #   这里每道题只跑一次，所以不影响结果。）
            s.retrieval.inject_k = k
            k_runner = make_runner(retriever=retriever, llm=llm, settings=s)
            recs = run_questions(k_runner, llm, sweep_qs, model=model,
                                 pricing=pricing, label=f"k={k}")
            calls = [c for r in recs for c in r.calls]
            # **只统计「生成答案」那一段的输入 token。**
            # 曲线要回答的是「多注入几条会让 prompt 长多少」，而工具循环那几次
            # 调用的 prompt 里也带着法条工具的定义，那部分**不随 inject_k 变**，
            # 混进来会把注入量带来的变化稀释掉。
            ans = [c for c in calls if c.stage == "生成答案"]
            prompt_avg = statistics.mean([c.prompt_tokens for c in ans]) if ans else 0.0
            per_k[str(k)] = {
                "prompt_tokens_answer_avg": round(prompt_avg, 1),
                "calls": len(calls),
                "cny_total": round(money_of(calls, model, pricing), 6),
                "cny_per_question": round(money_of(calls, model, pricing) / len(sweep_qs), 6)
                if sweep_qs else 0.0,
            }
            print(f"  → 生成阶段平均输入 {prompt_avg:.0f} token")

        # 召回那半边**不调模型、免费**，能跑全量 gold；
        # 成本那半边要真花钱，只跑了几题。**两边不同源**，
        # 表里分别标 n —— 不标的话读者会以为它们同源。
        from juris_agent.eval.retrieval_eval import gold_hash, load_gold

        gold = []
        for p in (s.paths.eval / "retrieval_gold.jsonl",
                  s.paths.eval / "retrieval_gold_machine.jsonl"):
            if p.exists():
                gold.extend(load_gold(p))
        pipeline = build_pipeline(production_arm(s), ArmDeps())
        rec_curve = recall_by_k(pipeline, gold, args.ks) if gold else {}
        s.retrieval.inject_k = orig_inject_k  # 还原（见上面那段注释）

        corpus_hash = "unknown"
        if s.paths.corpus_meta.exists():
            corpus_hash = json.loads(
                s.paths.corpus_meta.read_text(encoding="utf-8")
            ).get("corpus_hash", "unknown")

        sweep = {
            "per_k": per_k,
            "recall": {str(k): v for k, v in rec_curve.items()},
            "n_gold": len(gold),
            # 召回率**必须连语料一起记**：换了语料，同一批题的 hit 会变，
            # 两个数放一起比就是错的。这和 F34 门禁里「基线过期拒绝比较」是同一件事。
            "gold_hash": gold_hash(gold) if gold else "",
            "corpus_hash": corpus_hash,
            "sweep_n": len(sweep_qs),
        }

    # ---- 存数据 + 出报表 ----------------------------------------------- #
    per_question = [
        {
            "question": t.question, "route": t.route, "finish": t.finish,
            "ok": t.ok, "calls": len(t.calls),
            "prompt_tokens": sum(c.prompt_tokens for c in t.calls),
            "completion_tokens": sum(c.completion_tokens for c in t.calls),
            "reasoning_tokens": sum(c.reasoning_tokens for c in t.calls),
            "cny": round(money_of(t.calls, model, pricing), 6),
            "ms": round(t.ms, 1),
        }
        for t in turns
    ]
    money_list = [q["cny"] for q in per_question]
    payload = {
        "model": model,
        "mode": "off",
        "pricing": {
            "file": s.paths.pricing.name,
            "as_of": pricing.as_of,
            "currency": pricing.currency,
            "input": price.input,
            "output": price.output,
            "cache_hit_discount": pricing.cache_hit_discount,
        },
        "inject_k": orig_inject_k,
        "n_questions": len(turns),
        "n_ok": sum(1 for t in turns if t.ok),
        "stages": stages,
        "cache": {
            "prompt_tokens": in_tokens,
            "hit": hit_tokens,
            "miss": miss_tokens,
            "no_usage_calls": sum(1 for c in all_calls if not c.got_usage),
            "cost_full": round(in_cost_full, 6),
            "cost_real": round(in_cost_real, 6),
            "saved": round(cache_saved, 4),
        },
        "per_question": per_question,
        "cost_p50": round(p_at(money_list, 0.5), 6),
        "cost_p95": round(p_at(money_list, 0.95), 6),
        "cost_total": round(sum(money_list), 6),
        "sweep": sweep,
    }

    Path(args.json_out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n数据已写入 {args.json_out}")
    Path(args.out).write_text(render_md(payload), encoding="utf-8")
    print(f"报告已写入 {args.out}（改措辞可用 --render-only 重新生成，不花钱）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
