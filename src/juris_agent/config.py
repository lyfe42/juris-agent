"""集中配置。所有可调参数只有这一处来源。

设计约束：
1. import 本模块即保证 HF 镜像已生效（靠 `juris_agent/__init__.py` 的引导）。
2. 单价、阈值这类会变的数字**不进代码**，放 configs/pricing.yaml 与 configs/*.yaml，
   这里只放"指向它们的路径"。硬编码单价是面试里的减分项。
3. 所有默认值都是可覆盖的，`Settings()` 在测试里可以直接构造而不读 .env。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from juris_agent import PROJECT_ROOT


class Paths(BaseSettings):
    """目录布局。集中在嵌套模型里，避免 Settings 被几十个路径字段淹没。"""

    model_config = SettingsConfigDict(env_prefix="JURIS_AGENT_", extra="ignore")

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
    model_config = SettingsConfigDict(env_prefix="JURIS_AGENT_RETRIEVAL_", extra="ignore")

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

    # --- 生产臂的选型（下面这几个不是"默认值"，是**消融矩阵挑出来的结果**）----
    # 全部有实测依据，见 reports/ablation.md：
    #   chunker  : by_article（固定 512 字切分的 hit@5 只有 0.394，法条切分是 0.606 起）
    #   weights  : bm25=0.25 / dense=1.0。等权 RRF 是 0.866，把 BM25 压到 0.25 是 0.903；
    #              纯 dense 是 0.907——两者差 0.4 个点，在评测噪声范围内。
    #              **留一路低权重 BM25 是为了法条编号/原文措辞能精确命中**，
    #              那是 dense 的弱项（问「第四十七条」时稠密模型未必给它最高分）。
    #   rerank   : 单开关收益最大（0.866 → 0.944），必开。代价是 p50 从 20ms 到 221ms。
    #   rewrite  : **默认 none**。LLM 改写实测能到 0.952（最高），但有三个代价：
    #              ① p50 从 221ms 涨到约 6.8s；② 约一成的调用会静默失败退回原文；
    #              ③ temperature=0 下不可复现（同一题连问两次结果不同，20/20）。
    #              所以它是**可选项**，由 `JURIS_AGENT_RETRIEVAL_REWRITE=llm` 打开。
    chunker: str = "by_article"
    sparse_weight: float = 0.25
    dense_weight: float = 1.0
    use_rerank: bool = True
    rewrite: str = Field(default="none", pattern="^(none|synonym|llm)$")


class ToolSettings(BaseSettings):
    """工具执行层的全部旋钮。故障注入实验就是改这里的值重跑。"""

    model_config = SettingsConfigDict(env_prefix="JURIS_AGENT_TOOL_", extra="ignore")

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

    model_config = SettingsConfigDict(env_prefix="JURIS_AGENT_GUARD_", extra="ignore")

    max_steps: int = 12
    # 连续重复调用同一个 (tool, args) 达到该次数即中止
    repeat_call_limit: int = 2
    # 连续 N 步没有产生新的检索结果/工具结果即判为无进展
    no_progress_limit: int = 3
    token_budget: int = 60_000


class MemorySettings(BaseSettings):
    """记忆层的旋钮（F19–F22）。

    ⚠️ **这里放的是"怎么装记忆层"，不是"记忆层的实现在哪"。**
    层的实现全在 `juris_agent/memory/`，这里只是把它的开关暴露成配置——
    理由和项目里其它配置一样：**调参不该改代码**，
    因为改了代码就分不清"这次变好是参数还是逻辑"。

    `policy` 的四个取值就是 F22 消融报告里那四种策略。
    **默认 `layered`**（槽位 + 窗口 + 摘要 + 长期召回），依据在
    `reports/memory_ablation.md`：同样 300 多 token 的提示词预算下，
    它的照抄型保持率是 1.000，滑动窗口是 0.500，无记忆是 0.000。
    """

    model_config = SettingsConfigDict(env_prefix="JURIS_AGENT_MEMORY_", extra="ignore")

    enabled: bool = True
    """关掉它，图里那个 `remember` 节点就什么都不做。
    **留这个开关是为了能做对照实验**：要证明"记忆层有用"，
    必须能把它整个摘掉跑一遍，而不是嘴上说它有用。"""

    policy: str = Field(
        default="layered",
        pattern="^(no_memory|sliding_window|sliding_window_summary|layered)$",
    )

    total_tokens: int = 2048
    """记忆层的总预算。**这个数不是"上下文塞得下多少"，是"够用就好"。**
    prompt 里还有 8 条法条正文（大头），记忆只是其中一段。"""

    slots_max: int = 400
    window: int = 6
    """保留最近几轮**原文**。6 轮的依据见 `memory/working.py`。"""
    segment: int = 6
    """几轮并成一段摘要。默认和 `window` 一样大，
    让分界线干净：「最近 6 轮原文 + 再往前每 6 轮一段摘要」。"""

    prefill_tool_args: bool = True
    """**用槽位补全模型漏填的工具参数**（这是"第 20 轮还记得月薪"的正解）。

    开了之后，凡是被槽位表覆盖的字段，模型没填的就由程序补上；
    模型填了的**不覆盖**，只在两边对不上时记一条冲突（见 `nodes._prefill_args`）。
    关掉它就能对比出"补参数"这一步到底值多少——**能关的开关才好说服人**。"""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="JURIS_AGENT_",
        env_file=".env",
        extra="ignore",
        # 下面 `deepseek_api_key` 用了 validation_alias，字段名本身也要能用于构造，
        # 否则测试里 `Settings(deepseek_api_key="x")` 会静默变成 None。
        populate_by_name=True,
    )

    # --- LLM ---
    # ⚠️ 这个字段**必须显式写 alias**，不能靠 `env_prefix`。
    #
    # 踩过的坑：`env_prefix="JURIS_AGENT_"` 会把所有字段映射成
    # `JURIS_AGENT_<字段名>`，于是这个字段实际去找的是
    # `JURIS_AGENT_DEEPSEEK_API_KEY`。而 `.env.example`（以及所有人的肌肉记忆、
    # CI secrets 的惯例）写的都是 `DEEPSEEK_API_KEY`——**两者对不上**。
    #
    # 症状特别难查：`.env` 读进去了、`os.environ` 里也有
    # （`juris_agent/__init__.py` 的引导会把它塞进去），但 `Settings` 就是拿到 None，
    # 最后表现为"明明配了 key，LLMClient 却报没有 key"。
    #
    # `AliasChoices` 让两种写法都认，省掉"到底该加不加前缀"的反复。
    deepseek_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "JURIS_AGENT_DEEPSEEK_API_KEY"),
    )

    # ⚠️ 模型名是**实测出来的**，不是猜的：`GET https://api.deepseek.com/models`
    # 当前只返回两个 id：`deepseek-flash`(V4.1-Flash) 与 `deepseek-v4-pro`。
    # 老文档里的 `deepseek-chat` / `deepseek-reasoner` 在这个账号上**不存在**，
    # 用它们会直接 400。换账号/换时间点之前，先跑一次 scripts/check_api.py 核对。
    deepseek_base_url: str = "https://api.deepseek.com"
    # 生成与判分**用不同模型**是刻意的：LLM-as-Judge 有自我偏好偏差，
    # 让同一个模型给自己生成的答案打分，分数会系统性偏高（F26 要讲这件事）。
    # 生成用 flash（快、便宜、1M 上下文，造 180 条题的活它够用）；
    # 判分用 pro（更强，且与生成模型不同 → 至少不是一个"自己"）。
    model_gen: str = "deepseek-flash"
    model_judge: str = "deepseek-v4-pro"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3

    # --- 生成预算 ---
    # ⚠️ 这两个数**不能小气**。DeepSeek 这两个模型都是推理模型，
    # **思维链也计入 max_tokens**：预算给小了会拿到"HTTP 200、正文为空、
    # token 照扣"的结果（`LLMTruncated` 描述的就是这个）。`chat()` 里会自动
    # 翻 4 倍重试一次，但那要**多花一次请求的钱**——不如一开始就给够。
    # 实测 2048 对工具调用够用，3072 对"引用 + 算式 + 建议"的完整答案够用。
    tool_max_tokens: int = 2048
    answer_max_tokens: int = 3072

    # --- 运行模式 ---
    # record: 真调 API 并把 (prompt_hash, response) 录进 fixtures/replay
    # replay: 只读录制，命中不到就报错——离线调参零成本且完全可复现
    # off:    不录制不重放，直接真调
    llm_mode: str = Field(default="off", pattern="^(record|replay|off)$")
    env: str = Field(default="dev", pattern="^(dev|ci|prod)$")

    # 日志级别。**原先是硬编码在 `api/app.py` 的 lifespan 里的（写死 INFO）**，
    # 改成配置项是因为两件真事：
    #   ① 压测时每个请求都会打一条「一轮结束」，几百行日志把结果淹没了；
    #   ② 反过来，排查线上问题时又想把级别调到 DEBUG。
    # 两件事都不该改代码。环境变量：`JURIS_AGENT_LOG_LEVEL=WARNING`。
    #
    # ⚠️ **不用 `Literal[...]` 钉死取值**，因为 structlog/stdlib 认的级别名
    # 有好几种写法（`WARN` / `WARNING` / 数字 / 小写）。这里只存字符串，
    # 由 `configure_logging` 去解析——**在错误的层做校验只会把别的写法挡在外面**。
    log_level: str = "INFO"

    paths: Paths = Field(default_factory=Paths)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    tool: ToolSettings = Field(default_factory=ToolSettings)
    guard: GuardSettings = Field(default_factory=GuardSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)

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
