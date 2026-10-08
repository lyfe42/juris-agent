"""集中配置。所有可调参数只有这一处来源。

设计约束：
1. import 本模块即保证 HF 镜像已生效（靠 `lexguard/__init__.py` 的引导）。
2. 单价、阈值这类会变的数字**不进代码**，放 configs/pricing.yaml 与 configs/*.yaml，
   这里只放"指向它们的路径"。硬编码单价是面试里的减分项。
3. 所有默认值都是可覆盖的，`Settings()` 在测试里可以直接构造而不读 .env。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from lexguard import PROJECT_ROOT


class Paths(BaseSettings):
    """目录布局。集中在嵌套模型里，避免 Settings 被几十个路径字段淹没。"""

    model_config = SettingsConfigDict(env_prefix="LEXGUARD_", extra="ignore")

    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    raw: Path = PROJECT_ROOT / "data" / "raw"
    eval: Path = PROJECT_ROOT / "data" / "eval"
    index: Path = PROJECT_ROOT / "data" / "index"
    reports: Path = PROJECT_ROOT / "reports"
    configs: Path = PROJECT_ROOT / "configs"
    fixtures: Path = PROJECT_ROOT / "fixtures"
    web: Path = PROJECT_ROOT / "web"

    @property
    def laws(self) -> Path:
        """清洗后的法规 JSONL（ingest 的产物）。"""
        return self.data / "laws.jsonl"

    @property
    def corpus_meta(self) -> Path:
        """语料指纹，用于指标元信息里的 corpus_hash。"""
        return self.index / "corpus_meta.json"

    @property
    def pricing(self) -> Path:
        return self.configs / "pricing.yaml"


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEXGUARD_RETRIEVAL_", extra="ignore")

    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    # 检索臂的候选数。dense/sparse 各取 top_k，融合后交给 rerank 的是 fused_k
    top_k: int = 20
    fused_k: int = 20
    # 最终注入 prompt 的条数（成本-召回取舍曲线的横轴就是它）
    inject_k: int = 8
    # RRF 的 K。消融时会扫 1/10/30/60/200，60 是文献常用默认
    rrf_k: int = 60
    # 精排超时则降级回 RRF 排序（有测试断言）
    rerank_timeout_ms: int = 800
    rerank_batch_size: int = 32
    # 邻条扩展：命中的法条向前后各取 N 条并入上下文
    neighbor_window: int = 1


class ToolSettings(BaseSettings):
    """工具执行层的全部旋钮。故障注入实验就是改这里的值重跑。"""

    model_config = SettingsConfigDict(env_prefix="LEXGUARD_TOOL_", extra="ignore")

    max_retries: int = 3
    backoff_base_ms: int = 200
    backoff_jitter: float = 0.2  # ±20%
    backoff_max_ms: int = 5_000

    # 滑动窗口熔断
    breaker_window: int = 20
    # 窗口内样本数不足时不熔断——否则"第一个请求就失败"会把工具直接打死
    breaker_min_samples: int = 5
    breaker_failure_ratio: float = 0.5
    breaker_open_seconds: float = 30.0
    breaker_half_open_probes: int = 2

    # 幂等键 TTL
    idempotency_ttl_seconds: float = 300.0
    idempotency_max_entries: int = 2_048

    # 单次工具调用超时
    call_timeout_seconds: float = 10.0


class GuardSettings(BaseSettings):
    """四重防死循环。"""

    model_config = SettingsConfigDict(env_prefix="LEXGUARD_GUARD_", extra="ignore")

    max_steps: int = 12
    # 连续重复调用同一个 (tool, args) 达到该次数即中止
    repeat_call_limit: int = 2
    # 连续 N 步没有产生新的检索结果/工具结果即判为无进展
    no_progress_limit: int = 3
    token_budget: int = 60_000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LEXGUARD_", env_file=".env", extra="ignore"
    )

    # --- LLM ---
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    model_gen: str = "deepseek-chat"
    model_judge: str = "deepseek-reasoner"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3

    # --- 运行模式 ---
    # record: 真调 API 并把 (prompt_hash, response) 录进 fixtures/replay
    # replay: 只读录制，命中不到就报错——离线调参零成本且完全可复现
    # off:    不录制不重放，直接真调
    llm_mode: str = Field(default="off", pattern="^(record|replay|off)$")
    env: str = Field(default="dev", pattern="^(dev|ci|prod)$")

    paths: Paths = Field(default_factory=Paths)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    tool: ToolSettings = Field(default_factory=ToolSettings)
    guard: GuardSettings = Field(default_factory=GuardSettings)

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程级单例。测试里需要不同配置时直接 `Settings(...)` 构造，别清这个缓存。"""
    return Settings()


def ensure_dirs(settings: Settings | None = None) -> None:
    """建出运行期需要的目录。入口脚本（scripts/*）开头调一次。"""
    s = settings or get_settings()
    for path in (s.paths.index, s.paths.reports, s.paths.fixtures, s.paths.eval):
        path.mkdir(parents=True, exist_ok=True)
