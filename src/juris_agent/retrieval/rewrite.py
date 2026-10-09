"""F9 查询改写：补上口语与法条用语之间的词汇鸿沟。

## 这个模块是被实测数据逼出来的

F23 的第一轮基线里，下面三条**全部失败**（gold 完全没被召回）：

| 用户会这么说 | 法条里这么写 | 结果 |
|---|---|---|
| 「我要**辞职**，提前多久说」 | 「劳动者**解除劳动合同**，应当提前三十日……」 | 召回的是民法典的拾得遗失物 |
| 「劳动仲裁要**交钱**吗」 | 「劳动争议仲裁**不收费**」 | **一条都没召回** |
| 「每个月最多能**加班**几小时」 | 「**延长工作时间**每日不得超过……」 | 只捞回一堆无关条文 |

三条都是同一个病：**用户用生活词，法条用术语，BM25 要求字面重合**。
这不是"检索算法不够好"，是**输入没对齐**——改检索器是治错了地方，要改的是查询。

## 两条改写路径，都要有

1. **同义词表**（`configs/synonyms.yaml`）：手工维护，**零成本、确定性、可解释**。
   上面那三条鸿沟都是高频、固定的，写进表里就够了。它是**默认路径**。
2. **LLM 改写**：把整句交给模型，让它产出若干"法条风格"的变体。
   能用上你没想到的说法（比如「经济性裁员」↔「公司裁员」），
   但**要花钱、有采样噪声、不可解释**。

所以 F24 里它们是两条独立的对照轴——面试时能回答
"廉价确定的方法能吃掉多少收益，剩下的才值得上 LLM"，这比"我用了 LLM 改写"具体得多。

## 一条纪律：原始 query 永远保留

改写是**扩召回**，不是**替换**。原始 query 排在变体列表第一位，
所以即使改写跑偏，最坏情况也只是退化成"没改写"，**不会比不改更差**。
（实际消融里 +改写 那行如果有变差，那一定不是这个原因，得去别处找。）
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel

from juris_agent.llm import LLMClient


class SynonymTable(BaseModel):
    """口语 → 法条用语的映射表。

    `to` 可以有多个（一个口语词可能对到几个术语），但**不要塞近义堆**——
    每加一个词就是给 BM25 增加一份噪声。表里只放确实造成过召回失败的映射。
    """

    version: str = "1"
    # 键是口语说法，值是法条用语列表
    map: dict[str, list[str]] = {}

    @classmethod
    def load(cls, path: Path) -> "SynonymTable":
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(version=str(raw.get("version", "1")), map=raw.get("map", {}) or {})


class Rewriter(Protocol):
    name: str

    def rewrite(self, query: str) -> list[str]: ...


class IdentityRewriter:
    """不改写。消融的对照组——没有它，"改写提升了 X 点"就没有参照。"""

    name = "none"

    def rewrite(self, query: str) -> list[str]:
        return [query]


class SynonymRewriter:
    """按同义词表把口语词替换成法条用语，生成若干变体。

    变体是**组合**出来的：一句里有 "辞"/"老板" 两个口语词，就产生
    `原句` / `只换辞` / `只换老板` / `两个都换`。不直接做全笛卡尔积——
    变体数随命中词数指数增长，而第 8 个变体的边际收益接近零，
    却实打实地多花一次检索。

    上限 `max_variants` 控制这个增长；超出时按"命中词在查询里出现的先后"截断，
    保证**同样的输入产生同样的变体**（顺序敏感，不用集合）。
    """

    name = "synonym"

    def __init__(self, table: SynonymTable, max_variants: int = 5) -> None:
        self.table = table
        self.max_variants = max_variants

    def match(self, query: str) -> list[tuple[str, list[str]]]:
        """返回 `[(命中的口语词, 对应术语)]`，按在 query 里出现的位置排序。"""
        hits = [(w, terms) for w, terms in self.table.map.items() if w in query]
        hits.sort(key=lambda wt: query.index(wt[0]))
        return hits

    def rewrite(self, query: str) -> list[str]:
        hits = self.match(query)
        if not hits:
            return [query]
        variants: list[str] = [query]  # 第一位永远是原始 query
        for word, terms in hits:
            for term in terms:
                variants.append(query.replace(word, term))
        # 去重保序
        seen: set[str] = set()
        out: list[str] = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out[: self.max_variants]


_LLM_SYSTEM = """你是中国劳动法领域的信息检索助手。

用户会用**生活化口语**提问，而法条用的是**法律术语**。字面不重合会导致检索失败。
例如：「老板不给我交社保」→ 法条里写的是「用人单位未依法为劳动者缴纳社会保险费」。

把用户的问题改写成 {n} 个**法条风格的检索式**。要求：
1. 用劳动法/社会保险法/劳动争议调解仲裁法里的**标准术语**替换口语词。
2. 每个变体**保留原问题的检索意图**，不要引入原问题没有的法律概念。
3. 变体之间**要有实质差异**（换不同术语角度），不要只是同义词微调。
4. 不要写完整句子，写成**关键词组合**更利于检索。

只输出 JSON：{{"variants": ["...", "..."]}}"""


class LLMRewriter:
    """让 LLM 产出法条风格的检索式。

    ## `temperature=0` **不等于**可复现（实测，别信文档）

    常识是"`temperature=0` 就是确定性的"，实测**不是**：同一道题连着问两次，
    **20/20 都给了不同的改写**（用 `scripts/` 里的对照脚本测的，样本 20 题，
    两次调用间隔数秒，模型与参数完全相同）。

    变体在语义上很接近（都是"试用期 期限 上限 六个月"这类关键词组合），
    但**措辞不同**——而 BM25 匹配的是字面，所以检索结果真的会变。
    实测这种抖动让这一臂的 hit@5 在 0.937 ~ 0.952 之间浮动。

    所以**缓存买了"稳定"，没买"无偏"**：`replay` 模式下两次运行逐位一致
    （缓存返回同一份输出），但**建立缓存的那一次采样是不可复现的**。
    要量化这个抖动，只能跑 N 次报均值±std——原计划写的验收标准是对的，
    是"改成缓存就不需要跑多次"这个想法错了。

    ## 解析失败静默降级，但**必须**记账

    失败**不抛错**，退化成 `[query]`（相当于这一题没改写）。理由：改写是召回增强，
    模型偶发回一段非 JSON 不该让整条评测链断掉。

    但"记账"这件事原先只做到了计数器，**没有传到报表里**——实测跑满 269 题
    （含一次预热）共 **270 次调用，三次重跑分别失败 21 / 22 / 32 次
    （7.8% ~ 11.8%）**，原因几乎全是 `LLMError`。也就是说这一臂的数字里混着
    约一成根本没改写的题，而看表的人无从知道。**这个次数每次重跑都会变**
    （失败调用不入缓存——存了等于把一次网络抖动永久钉进 fixture），本身不是定值。
    现在 `failure_reasons` 记下原因，`scripts/ablation.py` 会把它打进报告。
    """

    name = "llm"

    def __init__(self, client: LLMClient, n_variants: int = 3, max_variants: int = 5) -> None:
        self.client = client
        self.n_variants = n_variants
        self.max_variants = max_variants
        self.failures = 0
        self.calls = 0
        # 失败原因分桶。只记个数不够——"JSON 解析失败"和"被限流"的修法完全不同。
        self.failure_reasons: Counter[str] = Counter()

    @property
    def failure_rate(self) -> float:
        return self.failures / self.calls if self.calls else 0.0

    def rewrite(self, query: str) -> list[str]:
        system = _LLM_SYSTEM.format(n=self.n_variants)
        self.calls += 1
        try:
            data = self.client.complete_json(
                query, system=system, max_tokens=512, temperature=0.0
            )
            variants = data.get("variants") or []
            if not isinstance(variants, list):
                raise ValueError("variants 不是列表")
        except Exception as e:
            self.failures += 1
            self.failure_reasons[type(e).__name__] += 1
            return [query]
        out = [query] + [str(v).strip() for v in variants if str(v).strip()]
        seen: set[str] = set()
        uniq: list[str] = []
        for v in out:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        return uniq[: self.max_variants]


REWRITERS = {"none": IdentityRewriter, "synonym": SynonymRewriter, "llm": LLMRewriter}
"""消融里可选的改写路径。注意它们构造参数不同，runner 里按名字分别实例化。"""


__all__ = [
    "REWRITERS",
    "IdentityRewriter",
    "LLMRewriter",
    "Rewriter",
    "SynonymRewriter",
    "SynonymTable",
]
