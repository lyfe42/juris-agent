"""可观测性（F28 + F31）：出问题时**查得到**，花钱时**算得清**。

这个包很小，只有三件事，但它们是「像是跑过的服务」和「像个测试框架」
之间最便宜的分界线——**加起来不到 400 行**：

| 文件 | 干什么 | 一句话 |
|---|---|---|
| `context.py` | 给每条请求发一个 id，写进它产生的每一行日志 | `grep <id>` 还原一条请求的一生 |
| `logs.py` | 结构化日志（字段可筛、可聚合）| 开发看彩色，生产出 JSON |
| `cost.py` | token 用量 → 金额，单价读配置文件 | **估算，不是账单** |

## 为什么这三件事要单独成包，而不是塞进 `api/`

因为**离线脚本也要用它们**。`scripts/memory_ablation.py` 要算成本、
`build_gold.py` 要打结构化日志——如果这些东西长在 API 层里，
离线脚本就得反过来 `import` 一个 Web 服务。

正确方向是：**API 依赖可观测性，不是反过来。**
以后要接别的入口（CLI、定时任务）也一样适用。

## 关于「观测」这个词被用滥了

这个词在这几年的技术讨论里被赋予了很多含义（分布式追踪、指标、
仪表盘、告警……）。本项目里它**只有一个意思**：

> **出了事，你能不能在两分钟内说清楚"这条请求走了哪几步、哪一步慢、
> 哪一步降级了、花了多少钱"。**

能不能做到，和用不用某个框架无关。所以这里没有引入任何可观测性框架，
就是 contextvars + structlog + 一张价目表。
"""

from juris_agent.observability.context import (
    bind,
    current_request_id,
    current_session_id,
    new_request_id,
)
from juris_agent.observability.cost import (
    CostBreakdown,
    Pricing,
    describe_cost,
    estimate_cost,
    load_pricing,
    sum_costs,
)
from juris_agent.observability.logs import configure_logging, get_logger

__all__ = [
    "CostBreakdown",
    "Pricing",
    "bind",
    "configure_logging",
    "current_request_id",
    "current_session_id",
    "describe_cost",
    "estimate_cost",
    "get_logger",
    "load_pricing",
    "new_request_id",
    "sum_costs",
]
