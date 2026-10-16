"""工具注册表：把领域函数包装成可供 LLM 调用的工具。

**工具 description 的写法直接决定模型选工具的准确率**，所以每个 `ToolSpec` 的
description 必须写清三件事：适用场景、**不适用场景**、返回什么。只写"计算加班费"
的 description 会让模型在"公司拖欠工资该赔多少"这种问题上错误地调用加班费工具。
`intent_eval.py` 会量化这件事（三方案对比 + 混淆矩阵）。

schema 的唯一来源是 pydantic 模型：`args_model.model_json_schema()` 同时用于
（a）发给模型的 function 定义，（b）执行时的入参校验。一份定义两处用，避免
"发给模型的 schema" 和"实际校验的 schema"漂移——那是最难查的一类 bug。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    result_model: type[BaseModel]
    fn: Callable[[BaseModel], Any]
    """接收**已校验**的 args_model 实例，返回 result_model 或其可校验的字典。"""
    fallback: Callable[[BaseModel, BaseException], Any] | None = None
    """所有重试耗尽后的降级实现。签名 `(args, 最后的异常)`，返回值同样要能
    通过 result_model 校验。没有降级路径就留 None——**不要为了"有降级"而编一个
    不准确的降级结果**，返回错误信息让模型换策略更诚实。"""
    idempotent: bool = True
    timeout_seconds: float | None = None
    """None 表示用全局默认（ToolSettings.call_timeout_seconds）。纯计算工具可设 0.0
    表示不套超时——给一个纯函数套线程池超时只会徒增延迟。"""

    # —— 下面两个字段**不发给模型**，只给人看（界面/文档）------------------- #
    #
    # 为什么要有它们：这个注册表现在有**两个读者**了。
    #
    #   读者一：模型。它读 `description`，那段写得越长越准（含"不适用场景"）。
    #   读者二：人。他要在界面上看见一个按钮，上面写着"算经济补偿"。
    #
    # 一开始只有一个读者（模型），所以 `description` 身兼两职。加界面的时候就露馅了：
    # 把 `description` 当标题显示，页面上会挂着一大段
    # 「【用途】…【不适用】…【返回】…」；反过来把短标题发给模型，
    # 模型就不知道什么时候**不该**用这个工具（而那正是选错工具的主因）。
    #
    # ⚠️ **这两个字段必须留在注册表里，不能在界面那边另写一张 name → 中文名 的表。**
    # 那种表会漂移：某天有人加了一个工具、忘了改界面那张表，
    # 界面上就会出现一个叫 `calc_xxxx` 的按钮。放在这里的话，
    # 新增工具时它和 description 在同一处，忘不了。
    label: str = ""
    """人看的短名（「算经济补偿」）。空串表示没起，界面退回显示 `name`。"""

    group: str = "其他"
    """界面上的分组。**按"用户在什么场景下会用到"分，不按代码模块分。**"""

    def args_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()

    def to_openai_tool(self) -> dict[str, Any]:
        """OpenAI / DeepSeek function-calling 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema(),
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._specs:
            raise ValueError(f"工具名重复注册: {spec.name}")
        self._specs[spec.name] = spec
        return spec

    def tool(self, **kwargs: Any) -> Callable[[type[BaseModel]], Any]:
        """装饰器形式：把 pydantic 结果模型的方法注册成工具。

        用法（见 `builtin.py`）：`@registry.tool(name=..., args_model=..., ...)`
        加在某个 `def impl(cls, args) -> ResultModel` 上。
        """

        def deco(target: Callable[..., Any]) -> Callable[..., Any]:
            spec = ToolSpec(fn=target, **kwargs)
            self.register(spec)
            return target

        return deco

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(
                f"未注册的工具 {name!r}；已注册: {sorted(self._specs)}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._specs)

    def all(self) -> list[ToolSpec]:
        return [self._specs[n] for n in self.names()]

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [s.to_openai_tool() for s in self.all()]

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._specs
