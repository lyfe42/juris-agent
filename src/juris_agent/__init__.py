"""juris_agent —— 可溯源劳动法咨询 Agent。

⚠️ 本模块顶部的副作用是**刻意设计**的，不是随手写的样板：

`HF_ENDPOINT` 必须在 `transformers` / `huggingface_hub` 被首次 import **之前**
就存在于 `os.environ` 中。一旦 huggingface_hub 先加载，它会把默认 endpoint 固化
到模块级常量里，之后再改环境变量也不会生效——模型下载会去 huggingface.co 并超时。

因为任何 `from juris_agent.xxx import ...` 都会先执行本文件，把 env 引导放在这里，
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

# --------------------------------------------------------------------------- #
# 对外门面：`from juris_agent import ask, JurisAgent`
# --------------------------------------------------------------------------- #
#
# ## ⚠️ 为什么这里是 `__getattr__` 而不是顶部一句 `from .client import ask`
#
# 因为那一句会**违背本模块存在的理由**。`client.py` 要 import
# `config`（pydantic-settings，会去读 .env 和所有 JURIS_AGENT_* 环境变量）
# 和 `api.runtime`。在本文件顶部 import 它们，就意味着：
#
#   **任何** `from juris_agent.anything import x`
#   ——包括 `from juris_agent.ingest.clean import Article` 这种只想拿一个
#   纯数据类的用法——都要先把 pydantic-settings 加载完。
#
# 加载它本身不慢，但它会**在引导阶段就去解析配置**。而引导阶段的契约是
# "只往 os.environ 里塞几个最关键的变量，别的什么都不干"（见模块 docstring）：
# 一旦这里把 `Settings` 拉进来并实例化，`HF_ENDPOINT` 生效的时机就
# 从"import 的第一行"变成"某些配置类初始化之后"——**一个只在
# 某些 import 顺序下才复现的下载超时**。这类 bug 最贵。
#
# PEP 562 的模块级 `__getattr__` 正好解决它：**名字照样能用，
# 但真正 import 发生在第一次访问的那一刻**（那时引导早已跑完）。
#
# 代价只有一个，值得说清楚：`from juris_agent import ask` **不再是编译期
# 就能确定的**，所以 IDE 的自动补全和"未定义名称"检查可能看不到它。
# `__all__` 是给它们的提示——把门面的名字都列进去。
_LAZY_EXPORTS = {
    "Answer": "client",
    "Citation": "client",
    "JurisAgent": "client",
    "NotReady": "client",
    "ask": "client",
    "get_agent": "client",
    "reset": "client",
}


def __getattr__(name: str) -> object:
    """第一次访问 `juris_agent.JurisAgent` 之类的名字时，才真的去 import。

    ⚠️ **不要在这里加"fallback 到别处"的逻辑。** 这个函数只该做一件事：
    把 `_LAZY_EXPORTS` 里写着的那几个名字从对应子模块取出来。
    一旦它开始"猜"（比如"没有 `ask` 就去 `client` 里翻同名函数"），
    打错名字得到的就是一个**看起来正常、其实是别的东西**的对象——
    而打错名字本来就该是一条 `AttributeError`。
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    # 塞回模块字典，这样第二次访问走的是正常的属性查找，
    # 不再经过这里。行为上没区别，但省掉每次访问一次字典查找 + 分支。
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """让 `dir(juris_agent)` 也列出那几个懒导出的名字。**交互式使用时很实用。**"""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = ["PROJECT_ROOT", *sorted(_LAZY_EXPORTS)]
