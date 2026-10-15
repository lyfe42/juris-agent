"""Agent 层：把检索、工具、生成串成一个会分岔、会回退、防死循环的图。

模块分工（按读代码的顺序排的）：

| 文件 | 干什么 | 一句话记住它 |
|---|---|---|
| `state.py` | 定义"白板"上有哪些字段 | 所有节点都能读写的那份状态 |
| `nodes.remember` | 记忆层接进主链路的那一步 | 组装记忆块 + 补全工具参数 |
| `router.py` | 决定走哪条路（规则，不用模型） | 最不可挽回的一步，不押在模型上 |
| `guard.py` | 四道防死循环闸门 | 步数 / 重复调用 / 无进展 / token 预算 |
| `invariants.py` | 节点级断言 | 开发期炸、生产期软降级 |
| `prompts.py` | 提示词与上下文拼装 | 编号由我们给，不让模型自己数 |
| `verify.py` | 引用校验 | 唯一零成本、可复现的回答质量指标 |
| `nodes.py` | 每个节点真正做的事 | 只返回增量，不改别的字段 |
| `graph.py` | 把节点连成图 | 唯一出口 + 唯一闸门 |

外部只需要 `make_runner()` 和 `AgentRunner.ask()` / `.stream()`。
"""

from juris_agent.agent.graph import AgentRunner, build_graph, make_runner
from juris_agent.agent.nodes import AgentDeps, Nodes
from juris_agent.agent.state import AgentState, CitationReport

__all__ = [
    "AgentDeps",
    "AgentRunner",
    "AgentState",
    "CitationReport",
    "Nodes",
    "build_graph",
    "make_runner",
]
