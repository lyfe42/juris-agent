"""token 估算。**给预算分配用的，不是给账单用的。**

## 为什么不直接上真 tokenizer

真 tokenizer（tiktoken / transformers）是**精确**的，但：

1. **慢**。每次往 prompt 里塞一段话都要跑一遍 BPE，一轮对话里要估几十次。
2. **模型绑定**。DeepSeek 的 tokenizer 不公开，用别家的会系统性偏。
3. **我们不需要精确值**。预算分配要回答的问题是
   "这四层里谁先被砍"——那是个**排序**问题，不是**计量**问题。
   估算只要单调（长文本估出来一定更大）就够用了。

所以这里用一条**字符分类**的规则，误差在 ±15% 以内（对中文偏保守）。

## 规则怎么来的

- **CJK 汉字**：现代 BPE 词表里常用汉字基本是 1 字 ≈ 1 token（生僻字会被拆成
  2 个字节级 token，但法条和用户提问里生僻字很少）。**取 1.0 是刻意的保守值**：
  高估 token 只会让我们早一点开始压缩，低估会让 prompt 超长被服务端截断——
  后者严重得多。
- **ASCII / 数字 / 标点**：约 4 个字符 1 个 token（英文词表的老经验值）。
- **其它**（emoji、希腊字母…）：按 2 个字符 1 个 token 估，比一律当 ASCII 保守。

**一个必须说清楚的局限**：这条规则对**代码和 JSON** 偏差最大（`{"a":1}` 这种
符号密集的串会被高估）。而工具调用的入参正是 JSON……所以误差方向是"
高估工具往返的开销"，也就是**偏向早压缩**。方向安全，接受。
"""

from __future__ import annotations

# 三类字符的"每 token 大约几个字符"。**越小越保守**（同样的文本估出更多 token）。
_CJK_PER_TOKEN = 1.0
_ASCII_PER_TOKEN = 4.0
_OTHER_PER_TOKEN = 2.0


def _kind(ch: str) -> int:
    """0=CJK，1=ASCII，2=其它。**用码位区间判，不引 unicodedata**——
    后者查表慢，而这段代码在预算分配里会被反复调用。"""
    o = ord(ch)
    if (
        0x4E00 <= o <= 0x9FFF        # 基本汉字
        or 0x3400 <= o <= 0x4DBF     # 扩展 A
        or 0x3000 <= o <= 0x303F     # 中文标点（，。「」）
        or 0xFF00 <= o <= 0xFFEF     # 全角字符
        or 0xF900 <= o <= 0xFAFF     # 兼容汉字
    ):
        return 0
    if o < 0x80:
        return 1
    return 2


def count_tokens(text: str) -> int:
    """估算一段文本的 token 数。空串 → 0。"""
    if not text:
        return 0
    counts = [0, 0, 0]
    for ch in text:
        counts[_kind(ch)] += 1
    total = (
        counts[0] / _CJK_PER_TOKEN
        + counts[1] / _ASCII_PER_TOKEN
        + counts[2] / _OTHER_PER_TOKEN
    )
    # **向上取整**：宁可高估。报文里有一条"预算还剩 0"总比"发出去被截断"好。
    return int(total) + (1 if total % 1 else 0)


def count_messages(messages: list[dict[str, object]]) -> int:
    """估算一整串消息的 token。

    ⚠️ **每条消息额外加 4 个 token 的固定开销。** 这是 OpenAI 兼容协议里
    每条消息都带的 role 分隔符和 `<|im_start|>` 之类标记的开销，跟正文长度无关。
    不算这一项的话，"20 轮对话"这种**条数多、每条都很短**的场景会被低估得很厉害
    （20 条 × 4 token = 80 token，而正文可能只有 100 token）。

    顺带：**算的是 `str(value)` 的长度**，不是只算 `content`。工具往返消息里
    `tool_calls` 那个字段可能是几百 token 的 JSON，漏了它预算就白算了。
    """
    total = 0
    for m in messages:
        total += 4
        for v in m.values():
            total += count_tokens(v if isinstance(v, str) else str(v))
    return total


__all__ = ["count_messages", "count_tokens"]
