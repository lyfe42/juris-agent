"""lexguard —— 可溯源劳动法咨询 Agent。

⚠️ 本模块顶部的副作用是**刻意设计**的，不是随手写的样板：

`HF_ENDPOINT` 必须在 `transformers` / `huggingface_hub` 被首次 import **之前**
就存在于 `os.environ` 中。一旦 huggingface_hub 先加载，它会把默认 endpoint 固化
到模块级常量里，之后再改环境变量也不会生效——模型下载会去 huggingface.co 并超时。

因为任何 `from lexguard.xxx import ...` 都会先执行本文件，把 env 引导放在这里，
就得到"无论谁先 import 什么都安全"的保证。这也是为什么这里不 import
pydantic-settings（避免引导阶段引入重依赖）。
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 需要"越早越好"地注入的变量；其余配置由 config.Settings 负责
_BOOTSTRAP_KEYS = ("HF_ENDPOINT", "HF_HOME", "DEEPSEEK_API_KEY")


def _load_dotenv_early() -> None:
    """极简 .env 解析，故意不依赖 python-dotenv。

    引导代码不该依赖一个尚未确认安装成功的第三方包。解析规则只覆盖本项目
    实际用到的形态：KEY=VALUE、# 注释、可选引号。已存在的真实环境变量优先，
    这样 shell `export` / CI secrets 可以覆盖 .env（部署时的常见诉求）。
    """
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key not in _BOOTSTRAP_KEYS:
            continue
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


_load_dotenv_early()

__all__ = ["PROJECT_ROOT"]
