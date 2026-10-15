"""引用校验：把"答案里每句话都有出处"这句话变成可以算出来的数（F25）。

## 为什么这件事值得单独一个模块

其他所有回答质量维度（有没有答到点子上、法条用得对不对、语气合适不合适）
**都要靠模型来评**——要花钱、会漂移、不可复现。而引用校验不一样：

> 注入的时候我们给每条法条打了编号 `[1]..[8]`；答案里写了 `[3]`。
> 我们只要查一下"3 在不在 1..8 里"，就知道这个引用是真是假。

**这是字符串查找，不是语义理解。** 所以它：

- 零 LLM 成本（不用调用任何模型）
- 零延迟（微秒级）
- **100% 可复现**（同样的输入永远同样的结果）

而且它抓的是一种**真实且高频的失败**：模型为了显得有依据，会**编一个编号**出来
（`[9]`），或者在上下文只给了 8 条时引用 `[12]`。用户看着像有引用，点开是空的。
**这类幻觉用任何"看起来像不像"的判断都抓不住，但用范围检查一抓一个准。**

## 三种引用质量指标（指标名和算法都在下面，别只报一个）

给定注入集合 `I = {1..n}`，答案里出现的编号集合 `C`：

| 指标 | 公式 | 它涨说明什么 | 它掉的典型原因 |
|---|---|---|---|
| **引用有效率** precision | `|C ∩ I| / |C|` | 引用都是真的 | 模型编编号 / 引用被截断 |
| **引用覆盖率** coverage | `|C ∩ I| / |I|` | 注入了 8 条，用上了几条 | 检索给了无关条文（噪声） |
| **零引用率** no_citation | 一条都没引用的答案占比 | — | 模型没在用检索结果，凭记忆答 |

⚠️ **必须成对报**，理由和召回率一样：只报有效率，模型可以"只引用它最有把握的
那一条"，有效率 100% 但覆盖率 5%——**它其实在用记忆答题**，报表上却很好看。

## 一个真实的坑：`[1]` 和 `[10]` 的写法

模型偶尔会写成 `[1,2]` 或 `[1][2]` 或 `【1】`（中文方括号）。三种都得认，
否则会把真实引用当成"没引用"，把覆盖率打低。但也不能认太宽——
`[2021]` 这种年份不能当成编号。所以正则限制在**1~2 位数字**，
并且要求整个括号里只有数字和分隔符。
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from juris_agent.agent.state import CitationReport, InjectedArticle

# 认三种写法：`[1]`、`【1】`、`[1,2]`（逗号/顿号/空格分隔）。
# **上限 2 位数字**：注入的条文最多几十条，3 位以上一定是别的东西（年份、金额）。
_CITE_RE = re.compile(r"[\[【]\s*(\d{1,2}(?:\s*[,，、]\s*\d{1,2})*)\s*[\]】]")
_INNER_SPLIT_RE = re.compile(r"[,，、]")


def parse_citations(answer: str) -> list[int]:
    """从答案里抽出编号，**去重且保序**。

    保序是有用的：`[3][1][3]` 抽成 `[3, 1]`——前端按这个顺序列出处，
    和用户读答案时遇到引用的顺序一致。用 `set` 就丢了这个信息。
    """
    out: list[int] = []
    for m in _CITE_RE.finditer(answer or ""):
        for part in _INNER_SPLIT_RE.split(m.group(1)):
            part = part.strip()
            if not part.isdigit():
                continue
            n = int(part)
            if n not in out:
                out.append(n)
    return out


def verify_citations(
    answer: str,
    injected: Sequence[InjectedArticle],
) -> CitationReport:
    """算出引用报告。**纯函数**——同样的输入永远同样的输出。"""
    valid_range = {a.index for a in injected}
    cited = parse_citations(answer)
    valid = [i for i in cited if i in valid_range]
    invalid = [i for i in cited if i not in valid_range]
    uncited = sorted(valid_range - set(valid))
    return CitationReport(cited=cited, valid=valid, invalid=invalid, uncited=uncited)


def strip_invalid(answer: str, report: CitationReport) -> str:
    """把**编造的编号**从答案里删掉。

    这是"修复"，和上面的"检查"分开（`invariants.py` 负责让开发期知道它在发生）。

    ⚠️ 只删编号、**不删那句话**。理由：那句话本身可能是对的，只是编号编错了；
    整句删掉会让答案变得莫名其妙（"……，所以你可以要求赔偿。" 突然少半句）。
    删掉 `[9]` 之后用户看到的是"没有出处的一句陈述"——**这是诚实的状态**：
    我们确实没法证明它。

    中文方括号写成 `【9】` 时也要能删对，所以用同一个正则。
    """
    if not report.invalid:
        return answer
    bad = {str(i) for i in report.invalid}

    def _sub(m: re.Match[str]) -> str:
        nums = [p.strip() for p in _INNER_SPLIT_RE.split(m.group(1))]
        keep = [n for n in nums if n not in bad]
        if not keep:
            return ""
        return "[" + ",".join(keep) + "]"

    # 注意：`_CITE_RE` 不认 `[1,9]` 里混着合法和非法的情况吗？认的——
    # `(\d{1,2}(?:[,，、]\s*\d{1,2})*)` 会整段匹配，所以 `[1,9]` → 保留 `[1]`。
    return _CITE_RE.sub(_sub, answer)


def resolve_keys(report: CitationReport, injected: Sequence[InjectedArticle]) -> list[str]:
    """把有效编号翻译成法条键（`lcl#47`），给评测层和前端用。

    **只翻译有效的**——无效编号翻译不出任何东西，它们已经被 `strip_invalid` 删掉了。
    """
    by_index = {a.index: a for a in injected}
    return [by_index[i].article_key for i in report.valid if i in by_index]


def injected_to_citations(injected: Iterable[InjectedArticle]) -> list[str]:
    """注入集合全部译成法条键。**评测层的 Recall 拿这个和 gold 比。**"""
    return [a.article_key for a in injected]


def summarize(reports: Sequence[CitationReport]) -> dict[str, float]:
    """把一批报告聚成三个报表数字。

    ⚠️ 聚合方式要说清楚，否则三个数会被加错：
    - `precision` / `coverage` 是**逐题算完再平均**（macro average），
      不是"把所有人的有效引用数加起来除以总引用数"（micro average）。
      差别在**权重**：micro 会让"引用了 20 条的题"压倒"引用了 1 条的题"。
      **法律问答里每题等权更合理**，所以用 macro。
    - `no_citation_rate` 是"一条都没引用的题占多少"。
    """
    n = len(reports)
    if not n:
        return {"n": 0.0, "precision": 0.0, "coverage": 0.0, "no_citation_rate": 0.0}
    return {
        "n": float(n),
        "precision": sum(r.precision for r in reports) / n,
        "coverage": sum(r.coverage for r in reports) / n,
        "no_citation_rate": sum(1 for r in reports if not r.cited) / n,
    }


__all__ = [
    "injected_to_citations",
    "parse_citations",
    "resolve_keys",
    "strip_invalid",
    "summarize",
    "verify_citations",
]
