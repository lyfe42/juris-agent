"""工具层：领域计算器 + 可靠性执行层。

分三个子块，依赖方向是单向的：

```
domain/     纯函数，无 IO、无 LLM、无框架依赖（金额用 Decimal）
  ↑
registry.py 把领域函数包装成 ToolSpec（schema 单一来源 = pydantic 模型）
executor.py 幂等 → 校验 → 熔断 → 重试退避 → 降级（可靠性全部在这层）
  ↑
builtin.py  实例化七个工具并组装成 ToolRegistry
```
"""

from juris_agent.tools.builtin import build_registry
from juris_agent.tools.executor import ToolExecutor, ToolOutcome, ToolStatus
from juris_agent.tools.registry import ToolRegistry, ToolSpec

__all__ = [
    "ToolExecutor",
    "ToolOutcome",
    "ToolRegistry",
    "ToolSpec",
    "ToolStatus",
    "build_registry",
]
