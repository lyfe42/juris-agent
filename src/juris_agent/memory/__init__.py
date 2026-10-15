"""记忆层（F19–F22）。

四层，四个不同的信息获取方式——**不是同一件事做四遍**：

| 层 | 一句话 | 文件 |
|---|---|---|
| 槽位 | **查表**。精确事实（金额/日期/年限），不参与压缩 | `slots.py` |
| 工作记忆 | **保序**。最近几轮原文，保住指代关系 | `working.py` |
| 摘要 | **压缩**。远期历史，可以有损 | `working.py` |
| 长期 | **检索**。窗口外的历史，按需捞 | `longterm.py` |

分配规则在 `manager.py`，判据是"哪层错了后果最严重"，
不是"哪层更常用"。

配套：
- `tokens.py`   token 估算（预算分配专用，不是账单）
- 评测在 `eval/memory_eval.py`，报告在 `reports/memory_ablation.md`
"""

from juris_agent.memory.longterm import LongTermStore, MemoryItem
from juris_agent.memory.manager import (
    LayerUse,
    MemoryBudget,
    MemoryBundle,
    MemoryManager,
    MemoryPolicy,
    manager_from_settings,
)
from juris_agent.memory.slots import (
    SLOT_SPECS,
    TOOL_SLOTS,
    SlotExtractor,
    SlotSpec,
    SlotStore,
    SlotValue,
    Slots,
)
from juris_agent.memory.tokens import count_messages, count_tokens
from juris_agent.memory.working import (
    Message,
    Summary,
    WorkingMemory,
    rule_summarize,
)

__all__ = [
    "SLOT_SPECS",
    "TOOL_SLOTS",
    "LayerUse",
    "LongTermStore",
    "MemoryBudget",
    "MemoryBundle",
    "MemoryItem",
    "MemoryManager",
    "MemoryPolicy",
    "Message",
    "SlotExtractor",
    "SlotSpec",
    "SlotStore",
    "SlotValue",
    "Slots",
    "Summary",
    "WorkingMemory",
    "count_messages",
    "count_tokens",
    "manager_from_settings",
    "rule_summarize",
]
