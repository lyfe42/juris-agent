"""把 token 用量换算成钱（F31 的一半）。

## 三条必须说在前面的事

**① 单价不写在代码里。** 报价会变，代码不该因为涨价而改。
所有单价读 `configs/pricing.yaml`，代码里只有"怎么算"。

**② 这个数是估算，不是账单。** 我们算的是「把这段 prompt 发过去要花多少」，
它不含：重试多花的钱、失败但已计费的调用、供应商侧的缓存折算口径差异、
以及**最容易被忽略的一项——被计费但你没看到的思维链 token**
（本项目用的两个模型都是推理模型，思考过程按输出计费）。
**把估算当账单报，是面试里的硬伤**，所以这个模块返回的每个数都带
`basis` 字段，写明它按哪张表、哪一天的价格算的。

**③ 缓存命中的折扣是估算里误差最大的一块。** DeepSeek 把 prompt tokens
分成「命中缓存」和「未命中」两档分别计价，而折扣系数各家口径差别很大。
`pricing.yaml` 里的 `cache.hit_discount` 只影响成本报表，**不影响任何指标**——
所以它取错也不会让评测数字变得不可信，只会让钱那一栏不准。
这个区分很重要：**要让有误差的东西落在不影响结论的地方。**

## 为什么单独放一个模块而不是塞进 API 层

成本报表（F31）是**离线脚本**要的，API 只是**顺带**在 `done` 事件里报一个数。
把算法放在 API 层，离线脚本就得反过来 import API；
放在这里，两边都依赖这一个文件，算法只有一份。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from juris_agent import PROJECT_ROOT


class ModelPrice(BaseModel):
    """一个模型的单价（元 / 百万 token）。"""

    input: float = 0.0
    output: float = 0.0
    note: str = ""


class Pricing(BaseModel):
    """整张牌价表。**它同时是"算法要用的数据"和"报表要引用的出处"。**"""

    currency: str = "CNY"
    unit: str = "per_million_tokens"
    as_of: str = "unknown"
    """价格快照日期。**报表里必须原样带出去**——
    没有日期的价格数字，三个月后就是一段没法验证的历史。"""

    models: dict[str, ModelPrice] = Field(default_factory=dict)
    cache_hit_discount: float = 1.0
    """缓存命中的输入 token 按这个系数折算。1.0 = 不折（保守估计）。"""

    def price_for(self, model: str) -> ModelPrice:
        """查单价。**查不到时不编一个数，而是返回 0 并让调用方看见。**

        为什么不给个"平均价"兜底：拿一个假单价算出来的钱，
        在报表上和一串真实数字混在一起，**没有任何人看得出来哪一行是假的**。
        返回 0 的话，成本列会明显是 0，一眼就知道"这个模型没配价"。
        """
        return self.models.get(model, ModelPrice())


class CostBreakdown(BaseModel):
    """一次调用的成本拆解。**拆开是为了回答"钱花在哪了"。**

    只看总额是没用的：输入贵还是输出贵，决定了优化方向完全不同
    （前者要靠精简 prompt / 少注入法条，后者要靠限制答案长度 / 关掉思维链）。
    """

    model: str = ""
    prompt_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    input_cny: float = 0.0
    output_cny: float = 0.0
    total_cny: float = 0.0

    as_of: str = "unknown"
    """这份金额是按哪天的牌价算的。见 `Pricing.as_of`。"""
    basis: str = ""
    """一句话说明这笔钱包含什么、不包含什么。**报表里要原样打出来。**"""

    billable: bool = True
    """**这一轮的钱（如果有）是不是真的花掉了。**

    分三种情况看：

    - `True`（默认）：这一轮真的打了模型接口，`total_cny` 是我们实付的。
    - `False` + token 数不为 0：**读的是磁盘/缓存里的旧记录**（`replay` 模式、
      或者命中 LLM 缓存）。这里必须把两件事同时说清楚，因为它们是两个数：
      这次调用**本来会花多少**（`total_cny`，做对照实验要的就是它），
      以及**这次实际花了多少**（0）。混为一谈就会写出
      "重放一遍花了 0.13 元"这种自相矛盾的话。
    - `False` + token 数也是 0：这一轮压根没调模型（拒答、闲聊、纯路由）。

    ⚠️ 这个字段是**踩过坑才加的**：`replay` 模式下前端的成本那栏
    打出了 `¥0.138040`。钱的来源是"状态里记着上一轮真调时用了 17255 token"，
    可这一轮一个接口都没打。**它看着是个完全合理的数**，
    所以光靠肉眼是发现不了的——这正是要把它做成一个显式字段、
    而不是靠调用方"记得别算错"的原因。
    """

    @property
    def is_free(self) -> bool:
        """是不是**一次调用都没发生过**（token 数也是 0）。

        用途：`replay` 模式和缓存命中时**成本是 0**，
        而报表里 `0.00 元` 和 `没算` 长得一样。用这个属性把两种情况分开，
        免得写出「我们每次问答只要 0 元」这种明显错的结论。

        ⚠️ 和 `billable` 的分工：这个说的是"发生没发生"，
        `billable` 说的是"计没计费"。`replay` 模式下
        **`is_free` 为假而 `billable` 也为假**（token 记着 17255，钱是 0），
        报表里要显示成"重放 17255 token，本次 0 元"。两个字段都要看。
        """
        return self.total_cny == 0.0 and self.prompt_tokens == 0 and self.completion_tokens == 0


def _default_path() -> Path:
    return PROJECT_ROOT / "configs" / "pricing.yaml"


@lru_cache(maxsize=8)
def _load_cached(path_str: str) -> Pricing:
    """按路径缓存。**缓存键是路径字符串**（`Path` 不可哈希成稳定键，
    而且不同测试用不同 tmp 路径，用 `str` 才能各缓存各的）。"""
    raw: dict[str, Any] = {}
    p = Path(path_str)
    if p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    cache = raw.get("cache") or {}
    models = {
        name: ModelPrice(
            input=float((spec or {}).get("input", 0.0)),
            output=float((spec or {}).get("output", 0.0)),
            note=str((spec or {}).get("note", "")),
        )
        for name, spec in (raw.get("models") or {}).items()
    }
    return Pricing(
        currency=raw.get("currency", "CNY"),
        unit=raw.get("unit", "per_million_tokens"),
        as_of=str(raw.get("as_of", "unknown")),
        models=models,
        cache_hit_discount=float(cache.get("hit_discount", 1.0)),
    )


def load_pricing(path: Path | str | None = None) -> Pricing:
    """读牌价表。文件不存在时返回一份**全 0 的空表**，而不是抛异常。

    为什么不抛：缺了这张表，服务应该**照常回答用户**，只是成本那一栏报 0。
    「一个报表用的配置文件缺失导致整个服务起不来」，是把辅助功能的
    失败升级成了主功能的失败——**这种耦合在项目里积累起来，
    就是"改一个报表把线上搞挂"的那种事故**。
    """
    path = Path(path) if path is not None else _default_path()
    return _load_cached(str(path.resolve()))


# 每百万 token 的除数。**写成常量而不是散在算式里**，
# 免得有人改了 `unit` 却忘了改算式（那会让金额差 6 个数量级）。
_PER_MILLION = 1_000_000.0


def estimate_cost(
    *,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_tokens: int = 0,
    cache_hit_tokens: int = 0,
    cache_miss_tokens: int = 0,
    pricing: Pricing | None = None,
) -> CostBreakdown:
    """算一次调用的钱。

    ## 输入的三种情形，必须分开处理

    供应商返回的 usage 里，输入 token 有三种可能的形态：

    1. **给了命中/未命中两个数**（DeepSeek 就是这种）→ 分别计价。
    2. **一个都没给**，只有一个总输入数 → 全部按未命中算（**保守，会高估**）。
    3. **只给了一个**（比如只给命中数）→ 缺的那个按 `总输入 - 命中` 补。

    第 2 种是刻意的保守：宁可报贵了，不要报便宜了。
    报贵了会被复查的人发现"你算多了"，报便宜了则会让人以为
    "成本没超"而继续加功能——**两个方向的错误代价不对称，所以选保守的那个。**
    """
    p = pricing or load_pricing()
    price = p.price_for(model)

    hit = max(int(cache_hit_tokens), 0)
    miss = max(int(cache_miss_tokens), 0)
    prompt = max(int(prompt_tokens), 0)

    if hit == 0 and miss == 0:
        # 情形 2：完全没给拆分。全部按未命中算。
        miss = prompt
    elif hit + miss < prompt:
        # 情形 3：补上缺口。用差值而不是直接用给的数，
        # 免得"总和小于 prompt"时算漏了一部分输入。
        miss = prompt - hit

    input_cny = (hit * p.cache_hit_discount + miss) * price.input / _PER_MILLION
    # ⚠️ 思维链**已经包含在 completion_tokens 里**（供应商是这么计的），
    # 所以这里**不能**再加上 reasoning_tokens，否则输出成本会被重复计算。
    # 把它单独记下来只是为了让报表能回答"答案很短但很贵，钱花哪了"。
    output_cny = int(completion_tokens) * price.output / _PER_MILLION

    return CostBreakdown(
        model=model,
        prompt_tokens=prompt,
        cache_hit_tokens=hit,
        cache_miss_tokens=miss,
        completion_tokens=int(completion_tokens),
        reasoning_tokens=int(reasoning_tokens),
        input_cny=round(input_cny, 8),
        output_cny=round(output_cny, 8),
        total_cny=round(input_cny + output_cny, 8),
        as_of=p.as_of,
        basis=(
            f"按 {p.as_of} 牌价估算（输入 {price.input}/百万、输出 {price.output}/百万，"
            f"{p.currency}）；不含重试与失败调用的开销，非账单"
        ),
    )


def sum_costs(items: list[CostBreakdown], *, model: str = "合计") -> CostBreakdown:
    """把多次调用的成本加总。**报表按会话/按天聚合时要用它**，
    而不是调用方自己写一个 `sum(...)`——那样 `basis` / `as_of`
    就会在聚合里丢掉，而丢掉的正是"这个数字怎么来的"。
    """
    if not items:
        # ⚠️ `billable=False` 必须显式写。空汇总的 `total_cny` 是 0，
        # 但它**不是**"花过钱、恰好花了 0 元"，而是"一次调用都没有"。
        # 让它取默认值 `True` 的话，报表里"本次实付"会把一份空汇总
        # 算进"有真实开销"的那一堆——**汇总表上一个多出来的 0 元行，
        # 会让人怀疑整张表的加总口径**。
        return CostBreakdown(model=model, basis="无调用", billable=False)
    return CostBreakdown(
        model=model,
        prompt_tokens=sum(i.prompt_tokens for i in items),
        cache_hit_tokens=sum(i.cache_hit_tokens for i in items),
        cache_miss_tokens=sum(i.cache_miss_tokens for i in items),
        completion_tokens=sum(i.completion_tokens for i in items),
        reasoning_tokens=sum(i.reasoning_tokens for i in items),
        input_cny=round(sum(i.input_cny for i in items), 8),
        output_cny=round(sum(i.output_cny for i in items), 8),
        total_cny=round(sum(i.total_cny for i in items), 8),
        as_of=items[0].as_of,
        # 各次的 basis 一字不差（同一张表算出来的），取第一条就够。
        # 万一它们不一致（中途换了牌价表），这里会暴露出来——
        # 拼在一起比只留一条更容易被发现。
        basis="；".join(sorted({i.basis for i in items})),
        # ⚠️ **必须显式传，不能让它取默认值 `True`。** 汇总里哪怕只有一次
        # 真调用，这份合计就是"有花过钱的"；但如果整份汇总都是重放/缓存来的，
        # 它必须跟着是 `False`——否则"离线跑一遍评测要多少钱"这个问题，
        # 会被汇总表上一个默认值答成"要钱"。
        # 用 `any` 而不是 `all`：混着来的时候（一部分真调、一部分重放），
        # 保守地当成"这份合计里有真金白银"，别把花掉的钱说成没花。
        billable=any(i.billable for i in items),
    )


def describe_cost(cost: CostBreakdown | None) -> str:
    """把一笔成本写成人话。**报表、演示脚本、日志都该用这一份。**

    为什么要抽成一个函数、而不是各自在打印的地方拼字符串：
    "这笔钱怎么说"是一个**判断**（下面那三种情况），判断写在两个地方
    就会出现两种说法，而两种说法里必然有一种是错的。
    更糟的是错的通常是**没被仔细看的那一处**（比如生成的报表里）。

    ## 三种"0 元"，必须分开说

    这三件事在报表上都是 `0`，含义完全不同：

    1. **真调了，但没记账**（牌价表里缺这个模型）→ 说"未记账"。
       这是"我不知道花了多少"，是个**缺口**，得去补牌价表。
    2. **压根没调模型**（拒答、闲聊、纯路由）→ 说"没有调用模型"。
       这是真的 0，没什么可补的。
    3. **读了旧记录**（`replay` 模式、命中缓存）→ **金额不是 0**
       （它是"当初真调那一次花了多少"），但这一轮一分没花。

    ⚠️ 第 3 种要**两个数都给**：`0`（这次花的）和 `约 ¥X`（本来会花的）。
    只给前者，就答不了"离线跑一遍评测要多少钱"；
    只给后者，那是**虚报开销**——而且是最危险的一种错：
    `¥0.138040` 这种数看着太正常了，谁都不会去怀疑它。

    ⚠️ 顺序：**先看 `billable`，再看金额**。倒过来第 3 种会被当成第 1 种
    甚至被当成真实开销。
    """
    if cost is None:
        return "（没有成本信息）"

    total = cost.prompt_tokens + cost.completion_tokens

    if not cost.billable:
        if not total:
            return "0 元（这一轮没有调用模型）"
        return (
            f"{total} token · 重放/缓存，本次 0 元"
            f"（当初真调这一次约 ¥{cost.total_cny:.6f}，按 {cost.as_of} 价格）"
        )

    if not cost.total_cny:
        # 这里已经排除了"重放"（上面那段）和"没调模型"（下面这半句）。
        # ⚠️ **剩下的两种情况要说不同的话**：
        #   - 一个 token 都没产出 → 金额当然是 0，牌价表没问题；
        #   - 有 token 但算出来 0 元 → **牌价表里缺这个模型**。
        # 都笼统写成"未记账"的话，第一种会被当成配置缺失去排查，
        # 而真正缺配置的第二种就藏在这句"已经查过了"里没人再看。
        if not total:
            return "0 元（这一轮没有产生 token）"
        return f"{total} token · 未记账（{cost.basis}）"

    cache = ""
    if cost.cache_hit_tokens or cost.cache_miss_tokens:
        cache = f" · 缓存命中 {cost.cache_hit_tokens}/{cost.cache_hit_tokens + cost.cache_miss_tokens}"
    return (
        f"{total} token（入 {cost.prompt_tokens} / 出 {cost.completion_tokens}"
        f"，其中思考 {cost.reasoning_tokens}）{cache} · 约 ¥{cost.total_cny:.6f}"
        f"（按 {cost.as_of} 价格）"
    )


__all__ = [
    "describe_cost",
    "CostBreakdown",
    "ModelPrice",
    "Pricing",
    "estimate_cost",
    "load_pricing",
    "sum_costs",
]
