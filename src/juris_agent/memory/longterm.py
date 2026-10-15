"""长期层：跨越滑动窗口的历史，按需召回（F21 的第四层）。

## 它和前三层的区别

前三层都是"**全都给模型看**"：槽位全给、近期原文全给、摘要全给。
这一层是"**按这一轮的问题，捞几条相关的给**"。

为什么有的层能全给、有的层必须捞：**因为前三层有天然的界**。
槽位十几个字段、近期 6 轮、摘要几段——再长也就几百 token。
而"第 1 到第 14 轮的用户原话"是所有历史里最大的一块，
全给就把预算吃光了。所以它必须被检索。

> 一句话总结四层：**槽位是查表、窗口是保序、摘要是压缩、长期是检索。**
> 四种不同的信息获取方式，对应四种不同的记忆需求，不是同一件事做四遍。

## 为什么不用 `retrieval/bm25.py`（看起来该复用它）

因为**它们的打分目标不一样，硬套会算错**：

- 检索层的 BM25 跑在**几千条法条**上，IDF（逆文档频率）是它最有价值的部分——
  一个词在语料里越罕见越说明它重要，这是 BM25 的核心。
- 长期记忆里只有**几十条历史片段**。样本这么少，IDF 会被单条记录带偏：
  某条历史里独有出现的一个词（比如用户自己的公司名）会拿到极高的 IDF，
  让那条记录在任何问题下都被召回。**样本量不够的时候，IDF 是个陷阱。**

所以这里用一个更朴素的办法：**关键词重叠 + 时间衰减**。它不试图变聪明，
因为在几十条记录上"聪明"的打分方式只会引入方差。

## 关键词怎么切

用 jieba。**不引 BM25 那套自定义词典**——那本词典是为法条术语（"经济补偿金"
必须是一个词）准备的，历史片段是**大白话**（"公司让我下周走人"），
用法条词典切大白话会切得很碎。**词典要跟着语料走。**
"""

from __future__ import annotations

import math
from typing import Any

import jieba
from pydantic import BaseModel, Field, PrivateAttr

# 停用词。**列表故意短**——只去掉那些"在任何中文句子里都出现、且不携带信息"的词。
# 长停用词表在几十条记录的规模上是负收益：删掉的词越多，两条记录越容易"看起来相同"。
_STOP = frozenset(
    "的 了 是 我 你 他 她 它 们 这 那 有 在 和 与 就 都 也 还 要 会 能 可以 "
    "什么 怎么 为什么 一个 我们 你们 他们 一下 的话 然后 因为 所以 但是 "
    "吗 呢 吧 啊 呀 哦 嗯".split()
)


def tokenize(text: str) -> list[str]:
    """切词 + 去停用词 + 去单字。

    **去单字**是因为单字在中文里几乎必然是停用词或量词（"个""份""第"），
    留着它们会让所有记录都互相"相关"。
    """
    return [w for w in jieba.lcut(text) if len(w) > 1 and w not in _STOP]


class MemoryItem(BaseModel):
    """长期层里的一条。**存"一轮"而不是"一句"**——

    单独一句用户提问（"那要是我不签呢？"）脱离上下文没有意义，
    检索出来也没法用。一轮 = 用户问 + 助手答，它自带上下文。
    """

    turn: int
    question: str
    answer: str = ""
    keys: list[str] = Field(default_factory=list)
    """这一轮引用过的法条键。召回时用得上——同一部法条被反复提到的轮次，
    多半是用户真正在纠结的那个问题。"""

    def render(self) -> str:
        """渲染成给模型看的形态。**助手那一段截断**——

        它通常几百字，全带回 prompt 就不叫"按需召回"了。
        截断位置选 200 字：够说清结论，不够说完论证过程（论证过程属于那一轮，
        要论证细节的话应该去问那一轮，而不是指望长期层把它背下来）。
        """
        head = f"[第{self.turn}轮] 用户：{self.question}"
        if self.answer:
            body = self.answer.replace("\n", " ").strip()
            if len(body) > 200:
                body = body[:200] + "…"
            head += f"\n           助手：{body}"
        return head


class LongTermStore(BaseModel):
    """按需召回的记忆库。

    `half_life` 是**时间衰减的半衰期**（以轮为单位）：某一轮的分数乘上
    `0.5 ** (相隔轮数 / half_life)`。默认 10 轮——20 轮会话里，
    第 1 轮的话到第 20 轮只剩约 1/4 的权重。

    为什么要有衰减：**在长对话里，"相关"和"近"高度相关**。用户在第 19 轮
    重提"月薪"时，他要的几乎总是第 14 轮那句补充说明，而不是第 1 轮的原始数字
    （那个数在槽位里，不在检索里）。**槽位管"最初的数"，检索管"最近的补充"。**
    """

    items: list[MemoryItem] = Field(default_factory=list)
    half_life: float = 10.0
    # 切词结果缓存。**用 `PrivateAttr` 而不是普通字段**：pydantic 只把
    # 下划线开头的名字当私有属性，写成普通字段的话它会被算进
    # `model_dump()`，然后序列化出一个几十 KB 的词表进 trace。
    _index: dict[int, set[str]] = PrivateAttr(default_factory=dict)

    def add(self, item: MemoryItem) -> None:
        """加一轮。**被截断/被摘要掉的那一轮才走这里**——
        近期那几轮已经在窗口里了，重复召回等于把同一段话给模型看两遍，
        会浪费预算还会让模型以为那件事更重要。"""
        self.items.append(item)
        self._index[item.turn] = set(tokenize(item.question + " " + item.answer))

    def search(self, query: str, *, current_turn: int, k: int = 3) -> list[MemoryItem]:
        """捞最相关的 k 条。**没有任何词重叠时返回空**，不返回"最不相关的那几条"——

        那是这一层最容易被做错的地方：硬塞 k 条进去，模型会去从一堆无关历史里
        找因果关系，然后编出一个用户没问过的联系。**宁可空手而归。**
        """
        if not self.items:
            return []
        q = set(tokenize(query))
        if not q:
            return []

        scored: list[tuple[float, MemoryItem]] = []
        for item in self.items:
            keys = self._index.get(item.turn) or set(tokenize(item.question + " " + item.answer))
            self._index[item.turn] = keys
            overlap = q & keys
            if not overlap:
                continue
            # 重叠词数。**不除以长度**（不像 Jaccard）：我们关心的是
            # "命中了几个用户关心的词"，不是"两条记录像不像"。
            # 除以长度会惩罚长回答，而长回答往往信息更多。
            base = float(len(overlap))
            gap = max(0, current_turn - item.turn)
            scored.append((base * math.pow(0.5, gap / self.half_life), item))

        scored.sort(key=lambda p: (-p[0], p[1].turn))
        return [item for _, item in scored[:k]]

    def __len__(self) -> int:
        return len(self.items)


__all__ = ["LongTermStore", "MemoryItem", "tokenize"]
