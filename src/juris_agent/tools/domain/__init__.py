"""纯领域逻辑层：劳动法计算器。

**这一层刻意不 import 任何 LLM / 网络 / 向量库**，全部是确定性纯函数。三个理由：

1. 可测。计算题的正确答案是唯一确定的，写成纯函数才能用单元测试钉死边界值
   （试用期正好 6 个月、经济补偿正好 12 年、月工资正好等于 3 倍社平……）。
   这些边界恰恰是面试官最爱挑的地方，也是 naive 实现必然出错的地方。
2. 可评测。检索层可以被 LLM 采样扰动，计算层不能——评测时它必须是基准真值。
3. 可复用。API 层、评测层、以及"给模型看的结果"都调用同一份实现，避免
   工具描述里的算法和实际算法漂移。

金额一律用 `Decimal`，不用 `float`。经济补偿这类金额要精确到分，float 累加会
出 0.01 级误差，在"算加班费"这种要跟工资条对账的场景里就是 bug。
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def money(value: Decimal | int | float | str) -> Decimal:
    """转成精确到分的 Decimal，四舍五入（银行家舍入不适合工资场景）。

    用 `str(value)` 而不是 `Decimal(value)`：`Decimal(0.1)` 会得到
    0.1000000000000000055511151231257827021181583404541015625。
    """
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


__all__ = ["CENT", "money"]
