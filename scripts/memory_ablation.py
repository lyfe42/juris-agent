"""跑长对话记忆消融，生成 `reports/memory_ablation.md`（F22）。

    uv run python scripts/memory_ablation.py

**这个脚本不调模型，一分钱不花。** 量的全是"记忆层组装出来的那段提示词里
有没有那个值"——纯字符串判定，可复现，几秒钟跑完。

真实逻辑在 `juris_agent/eval/memory_eval.py`，这里只是个别名。
**刻意不做成 `python -m juris_agent.eval.memory_eval`**：
这个项目里所有要跑的东西都从 `scripts/` 起手，一个入口比两个好记，
也不会出现"文档里写的是这个、README 里写的是那个"。

常用参数（都能直接透传给 `memory_eval.main`）：

    --policies layered                      只跑一个策略，调试用
    --no-audit                              跳过数据集体检（确认数据干净后才用）
    --out reports/xxx.md --json reports/xxx.json
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许 `uv run python scripts/xxx.py` 直接跑——把那层 src/ 挂上 sys.path。
# （`uv run python -m` 不需要这一步，但我们让两种跑法都能用。）
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from juris_agent.eval.memory_eval import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
