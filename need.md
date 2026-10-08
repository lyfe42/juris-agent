# need.md — 需求 ↔ 功能对照表

**这份文件回答一个问题：我要写的下一个函数，是为了满足什么需求？**

用法：**从上往下做，一次只做一条。** 每条都有「验收标准」——能跑通验收就算完成，
不要提前看后面的。做完一条来这里把 `[ ]` 改成 `[x]`。

进度追踪看 `ROADMAP.md`，完整设计看 `规划.md`。

---

## 怎么读每一条

```
F7  RRF 融合
需求     面试官问「为什么用 RRF 而不是加权求和」，我要有实测数字回答
做什么   src/lexguard/retrieval/fusion.py 的 rrf_fuse()
验收     同一个 query 两次检索结果喂进去，输出排名的顺序可预测；
         tests/test_fusion.py 全绿
依赖     F5, F6
预估     3h
产出     reports/ablation.md 的「+RRF」一行
```

**「需求」那一栏不是装饰。** 如果一条功能你找不到它对应哪个需求，就说明它不该写。
每条需求都指向一个具体的面试问题或评测指标——这是防止写出一堆没人问的功能。

---

# 阶段 0：前置（不是代码，但卡着所有事）

这两件只有你能做。**在他俩完成前，阶段 1 之后全部无法开始。**

- [ ] **P1 法规原文入库**
  - 需求：法律有修订，用错版本的条文回答是真实的法律风险；且「生效日期」是评测集 `temporal` 切片的依据
  - 做什么：从国家法律法规数据库 / 北大法宝导出 7 部法规；**逐条核对最新修订版本与生效日期**；存 `data/raw/`，每条带 frontmatter（法规名 / 条文号 / 版本 / 生效日期）
  - 验收：`data/raw/` 下 7 个文件，每条都有一个能说清楚来源的生效日期
  - 预估：4-8h（核对比下载慢得多）

- [ ] **P2 旧项目真实数字**
  - 需求：简历上 RoboMaster 与脑肿瘤两个项目**目前零量化**，面试官会问「提升了多少」
  - 做什么：整理数据量、耗时、指标（见简历里标 `【待补】` 的地方）
  - 验收：简历上每处 `【待补】` 都能填上真实数字，或删掉
  - 预估：2-3h

---

# 阶段 1：数据层 —— 让法条变成可检索的东西

- [ ] **F1 法条清洗与条级聚合**
  - 需求：法条必须**按条切**，因为 gold 标注的单位就是「条」——切分粒度与评测粒度不一致，Recall 就没法算
  - 做什么：`ingest/clean.py`。条款号正则（`第X条`、`第X条之Y`）、把跨行的条文正文合并成一条、剥离页眉页脚、输出 `data/laws.jsonl`
  - 验收：随机抽 10 条，条文正文完整无截断；条数与你导入的法规对得上；`第X条之Y` 这类不丢
  - 依赖：P1
  - 预估：4h
  - 产出：`data/laws.jsonl`

- [ ] **F2 切分器（4 个臂）**
  - 需求：面试官问「你的切分策略怎么定的」——必须有对照数字，不能只说「我用了按条切」
  - 做什么：`ingest/chunkers.py`，4 个函数：`fixed_512` / `by_article` / `sentence_window` / `by_article_plus_context`（按条 + 注入所属章节的语境）
  - 验收：同一部法规过 4 个切分器，产出 4 份不同粒度的 chunk 列表；每个 chunk 带 `article_key` 便于对齐 gold
  - 依赖：F1
  - 预估：4h
  - 产出：`reports/ablation_chunker.md`（W2 才有数字）

- [ ] **F3 建索引**
  - 需求：检索要有东西可查；且索引必须带 `corpus_hash`，否则指标无法复现
  - 做什么：`ingest/build_index.py`。bge-base-zh-v1.5 编码（GPU、`normalize_embeddings=True`）→ 存 Chroma 持久化目录；写 `corpus_hash` 到索引元数据
  - 验收：建完能按一条 query 查回 top-k；改了语料重新建，`corpus_hash` 会变
  - 依赖：F2
  - 预估：3h（**含装 CUDA torch**，现有环境是 CPU 版）
  - 产出：`data/index/`

- [ ] **F4 评测 gold 集（反向构造）**
  - 需求：没有 gold 就没有 Recall。反向构造 = 拿法条本身生成问题，答案（法条编号）天然已知，**不需要人工标注**
  - 做什么：`scripts/build_gold.py`。从法条正文抽关键句→让 LLM 改写成「用户口吻的提问」→gold = 该条 + 它引用的条；标注 `difficulty` 与 `qtype`（single_fact / multi_article / calculation / temporal）；**dev/test 严格隔离**
  - 验收：120 条 dev + 60 条 test；每条 gold 是法条编号列表；抽检 20 条确认问题问得通、gold 对得上
  - 依赖：F1
  - 预估：6h
  - 产出：`data/eval/retrieval_gold.jsonl`

> ⚠️ **F4 是阶段 1 里最要紧的。** 没有它，阶段 2 的所有数字都无处可测。

---

# 阶段 2：检索层 —— 简历数字的主产地

- [ ] **F5 稠密检索基线**
  - 需求：要有对比基线，「混合检索提升了 X 点」的 X 才有意义
  - 做什么：`retrieval/dense.py`。抽象 `VectorBackend` 接口（Chroma 实现 + FAISS 实现），查 top-k
  - 验收：能跑出 dense-only 的 Recall@1/5/10 —— **这是项目第一个真实数字**
  - 依赖：F3, F4
  - 预估：4h
  - 产出：`reports/ablation.md` 第一行

- [ ] **F6 BM25 关键词检索**
  - 需求：法条检索里「第47条」「经济补偿」这类词，稠密向量经常抓不住；且面试必问「为什么还要 BM25」
  - 做什么：`retrieval/bm25.py`。`jieba` 分词 + **法律领域词典**（「经济补偿金 vs 经济补偿」这类是真实坑）+ 同义词表；`rank_bm25`
  - 验收：查「经济补偿」能召回第47条；有无领域词典的差异能观察出来
  - 依赖：F1
  - 预估：4h
  - 产出：`reports/ablation.md` 的「+BM25」一行

- [ ] **F7 RRF 融合** ★
  - 需求：面试官最爱问「多路召回怎么合并」——RRF 只用排名不用分数，能回答「为什么不用加权求和」
  - 做什么：`retrieval/fusion.py`。`rrf_fuse(rankings, k=60)` + `weighted_sum_fuse`（**两个都实现**，消融要对比）
  - 验收：`tests/test_fusion.py` 全绿；两路结果融合后顺序可预测
  - 依赖：F5, F6
  - 预估：3h
  - 产出：`reports/ablation.md` 的「+RRF」与「RRF vs 加权求和」两行

- [ ] **F8 精排（含超时降级）**
  - 需求：① 提升排序质量；② **精排是慢的，超时必须能降级回 RRF 排序**——这是「工程容错」的真实例子
  - 做什么：`retrieval/rerank.py`。`bge-reranker-base` 批处理；800ms 超时 → 回落到融合排序，`warnings` 里说明已降级
  - 验收：正常情况召回质量提升；**手动让精排超时，系统仍能返回结果**（有测试断言）
  - 依赖：F7
  - 预估：4h
  - 产出：`reports/ablation.md` 的「+Rerank」一行（含 P95 延迟代价）

- [ ] **F9 查询改写**
  - 需求：用户口语（「老板拖欠工资怎么办」）与法条用语（「劳动报酬」「支付令」）之间有词汇鸿沟
  - 做什么：`retrieval/rewrite.py`。多个 query 变体 → 各查一遍 → 合并
  - 验收：口语 query 的 Recall 提升；**注意有 LLM 采样，指标要跑 3 次报均值±std**
  - 依赖：F7
  - 预估：3h
  - 产出：`reports/ablation.md` 的「+改写」一行

- [ ] **F10 七段编排 + 埋点**
  - 需求：面试官问「一次检索耗时花在哪」——必须能分段报出来
  - 做什么：`retrieval/pipeline.py`。改写→BM25→Dense→RRF→精排→邻条扩展→去重；**每段记耗时与候选数**
  - 验收：一次调用返回结果 + 每段的 `elapsed_ms` 与 `candidate_count`
  - 依赖：F5-F9
  - 预估：4h
  - 产出：SSE 面板的检索事件 + `reports/ablation.md` 的延迟列

---

# 阶段 3：工具层 —— 我写的那两个文件，你来重写

> 这一阶段的状态：**功能已经能跑，但代码不是你写的。** 债务清单见 `ROADMAP.md`。
> 这两条不是「再实现一遍」，是**面试前必须能自己从零写出来**的部分。

- [ ] **F11 重写 executor.py** ★★
  - 需求：面试官问「工具调用失败怎么办」——这是全项目追问最密集的一个文件
  - 做什么：**关掉 `src/lexguard/tools/executor.py`，从零重写。** 流水线：幂等键 → 入参校验 → 熔断放行 → 指数退避重试 → 出参校验 → 降级
  - 验收：`tests/test_executor.py` 全绿（测试可复用，会告诉你哪里不对）
  - 依赖：无
  - 预估：6-8h
  - 产出：能讲清楚的代码 + `reports/fault_injection.md` 可重跑

- [ ] **F12 重写 breaker.py** ★★
  - 需求：面试官问「你的熔断器怎么实现的」——这里有三个经典实现缺陷，踩过和没踩过差别很大
  - 做什么：**关掉 `src/lexguard/tools/breaker.py`，从零重写。** 三态机 + `min_samples` + 半开探测额度 + 探测成功清窗口 + `trips`/`rejections` 计数器
  - 验收：`tests/test_breaker.py` 全绿
  - 依赖：无
  - 预估：3-4h
  - 产出：同上

- [ ] **F13 search_law 工具接检索层**
  - 需求：现在 `search_law` 是空的（等语料），接上检索层后 Agent 才真能查法条
  - 做什么：`tools/builtin.py` 里把 `search_law` 的 retriever 接到 F10 的 pipeline；返回结果带 `article_key` 供引用校验
  - 验收：`build_registry(retriever=pipeline)` 后能查到法条；返回体里的引用编号可反查
  - 依赖：F10
  - 预估：2h
  - 产出：Agent 能引用法条

---

# 阶段 4：Agent 层

- [ ] **F14 状态定义与路由**
  - 需求：面试官问「为什么用状态机而不是线性 Chain」——要有结构答案
  - 做什么：`agent/state.py`（pydantic 状态）+ `agent/router.py`（判断走检索 / 工具 / 直接回答 / 拒答）
  - 验收：4 类输入（法条问题 / 计算题 / 越界问题 / 闲聊）能路由到正确的下一步
  - 依赖：F13
  - 预估：4h

- [ ] **F15 Agent 节点**
  - 需求：把路由的决定真正执行掉
  - 做什么：`agent/nodes.py`。检索节点 / 工具节点 / 生成节点 / 拒答节点 / 引用校验节点
  - 验收：单跑每个节点有确定输出
  - 依赖：F14
  - 预估：5h

- [ ] **F16 LangGraph 图与条件边**
  - 需求：多步任务要能回退（工具失败→改参数重试；检索不足→改 query 重查）
  - 做什么：`agent/graph.py`。节点 + 条件边 + 最大步数；开 checkpointer 以便分支恢复
  - 验收：端到端跑通一条混合问答（先查法条再算钱）；trace 里能看到每一步
  - 依赖：F15
  - 预估：6h

- [ ] **F17 四重防死循环** ★
  - 需求：面试官问「Agent 会不会转不出来」——必须四个维度都堵上，且**有对抗集上的数字**
  - 做什么：`agent/guard.py`。① 最大步数 ② 重复 `(tool, args_hash)` 连续 ≥2 次 ③ 无进展检测 ④ token 预算
  - 验收：构造 50 条对抗 query（诱导重复调用 / 无限追问），**死循环占比从 X% 降到 0**，X 要实测
  - 依赖：F16
  - 预估：5h
  - 产出：`reports/` 里防死循环的对照数字

- [ ] **F18 节点级断言**
  - 需求：面试官问「怎么保证不出错」——答案是「开发期抛异常，生产软降级」，同一套断言两种模式
  - 做什么：`agent/invariants.py`。如「进入生成节点时检索结果非空」「答案里的引用编号必须都在注入集合内」
  - 验收：故意违反一条断言，dev 模式抛异常、prod 模式降级并打日志
  - 依赖：F16
  - 预估：3h

---

# 阶段 5：记忆层

- [ ] **F19 结构化槽位** ★
  - 需求：第 20 轮丢信息的根因之一是**摘要把「12000」写成「一万左右」**。槽位抽出来单独存，不参与摘要压缩
  - 做什么：`memory/slots.py`。pydantic 槽位（金额、日期、解除方式、工作年限…），从对话里抽取并更新
  - 验收：20 轮对话后，第 1 轮给的金额仍精确保留
  - 依赖：F16
  - 预估：4h
  - 产出：`reports/memory_ablation.md`

- [ ] **F20 工作记忆**
  - 需求：token 有限，不可能全塞进 prompt
  - 做什么：`memory/working.py`。滑动窗口 + 分段摘要
  - 验收：超长对话不爆 token；近期对话完整保留
  - 依赖：F19
  - 预估：3h

- [ ] **F21 记忆管理器 + 预算分配**
  - 需求：四层记忆抢同一份 token 预算，要有明确分配规则
  - 做什么：`memory/manager.py`。预算分配；**槽位锚点优先级最高，不被裁剪**
  - 验收：第 20 轮的 prompt token 相比全量塞入显著下降，且保持率不降
  - 依赖：F20
  - 预估：5h
  - 产出：`reports/memory_ablation.md` 的 token 与成本列

- [ ] **F22 多轮会话评测集 + 四策略对照** ★
  - 需求：面试官问「你的记忆方案好在哪」——要有四策略对照的保持率曲线
  - 做什么：15 个脚本化 20 轮会话 YAML；跑 `no_memory` / `sliding_window` / `sliding_window_summary` / `layered_memory` 四策略
  - 验收：**轮次-保持率曲线**出图；关键事实保持率与任务成功率两组数字
  - 依赖：F21
  - 预估：6h
  - 产出：`reports/memory_ablation.md`

> 📌 面试原题「第 20 轮丢信息怎么排查」的四步路径，就是 F19→F20→F21 的逆向：
> 查槽位是否丢失 → 查窗口是否被截断 → 查摘要是否模糊化精确值 → 查长期召回是否失效。

---

# 阶段 6：评测层 —— 项目最不可替代的资产

- [ ] **F23 检索指标（双口径）** ★
  - 需求：**只报「命中即算」的 Recall 会让多条款组合题完全不可见**（3 条 gold 召回 1 条 → 命中口径满分，覆盖率口径 0.33）
  - 做什么：`eval/retrieval_eval.py`。`Recall@k_hit` + `Recall@k_art` 双口径、`MRR@10`（未命中记 0，不跳过）、`nDCG@5`
  - 验收：同一份结果算出两套 Recall 且数值不同；按 `difficulty` 与 `qtype` 分组切片
  - 依赖：F4, F10
  - 预估：5h
  - 产出：`reports/ablation.md` 的表头

- [ ] **F24 消融矩阵 runner**
  - 需求：面试官问「你的每个技术选型有多少收益」——要有一张能一次跑出来的表
  - 做什么：`eval/ablate.py` + `configs/ablation.yaml`。逐项开关（切分 / +BM25 / +RRF / +精排 / +改写 / +邻条扩展 / RRF K 值扫描 / Chroma vs FAISS）
  - 验收：一条命令产出完整表格；**每行带元信息**（n / eval_set_hash / corpus_hash / config）
  - 依赖：F23
  - 预估：5h
  - 产出：`reports/ablation.md`、`reports/rrf_k_curve.png`

- [ ] **F25 引用校验 + 拒答指标**
  - 需求：引用准确率是**唯一能程序化算出来、零 LLM 成本**的质量指标；拒答率必须与误拒率**成对报**，否则会鼓励系统乱拒答
  - 做什么：`eval/answer_eval.py`。注入时给片段打 `[1]..[n]` 并绑定 `article_key`；答案解析 `[i]` 反查 → 引用有效率 / 覆盖率；30 条越界问题的正确拒答率 + 120 条 in-scope 的误拒率
  - 验收：能报出引用有效率；拒答率与误拒率两个数一起出现
  - 依赖：F16
  - 预估：5h
  - 产出：`reports/eval_report.md`

- [ ] **F26 LLM-as-Judge**
  - 需求：主观维度（忠实度）需要判分，但**必须主动讲清它的局限**，否则被问出来就是减分
  - 做什么：`eval/judge.py`。rubric + **先引证据后打分** + temperature=0 + pydantic 校验 + 3 次自一致性 + **用 `deepseek-reasoner` 而非生成模型**（规避 self-preference）
  - 验收：30 条人工校准，报一致率与 κ；能量化长度偏差与位置偏差
  - 依赖：F25
  - 预估：6h
  - 产出：`reports/judge_calibration.md`

- [ ] **F27 六类失败归因** ★
  - 需求：面试官原题「失败怎么归因」。**badcase 收敛曲线比任何单个数字都更像「被生产验证过」**
  - 做什么：`eval/attribute.py`。六类（F1 检索 / F2 工具 / F3 Prompt / F4 模型能力 / F5 状态管理 / F6 输入歧义），用四问从 trace 程序化归位
  - 验收：能自动归位大部分 badcase；输出 Pareto 分布；**修 top 类后重跑，报收敛曲线**
  - 依赖：F25
  - 预估：6h
  - 产出：`reports/badcase_trend.md`

---

# 阶段 7：工程层 —— 让它看起来像真跑过的服务

- [ ] **F28 可观测性**
  - 需求：面试官问「线上出问题你怎么查」——`request_id` 全链路可 grep 是答案
  - 做什么：`observability/`。`structlog` + `contextvars` request_id、trace 快照、cost 记账
  - 验收：一次请求的所有日志能用一个 request_id 串起来
  - 依赖：F16
  - 预估：3h

- [ ] **F29 FastAPI + SSE**
  - 需求：面试官要**直接看到**检索/工具/token/成本，不能只听你说
  - 做什么：`api/`。SSE 事件流：`route` / `memory` / `tool_call` / `tool_result` / `retrieval` / `token` / `done`；`done` 带 usage、`cost_cny`、`trace_id`、引用列表
  - 验收：`curl` 能看到事件按序流出
  - 依赖：F28
  - 预估：5h

- [ ] **F30 单页前端**
  - 需求：**这是对抗「这只是个测试框架」的唯一办法**。产品壳必须是真能对话的，不是 demo 截图
  - 做什么：`web/index.html`。真能对话；**引用可点开看到法条原文**；计算题显示算式与代入过程；越界时显示拒答提示
  - 验收：走完三个用例——一次带引用问答 / 一次计算题 / 一次越界拒答
  - 依赖：F29
  - 预估：6h

- [ ] **F31 成本报表**
  - 需求：面试官必问「token 成本怎么控」。单价**不硬编码**，读配置文件
  - 做什么：`observability/cost.py`。三张报表：按阶段 token 分布 / 缓存命中率 / 单请求成本 P50-P95；优化要「**用指标换指标**」（top-8→top-5 使 token 降 X% 但 Recall 只降 Y 点）
  - 验收：报出单次问答成本，并给出成本-Recall 取舍曲线
  - 依赖：F29
  - 预估：4h
  - 产出：`reports/cost.md`

- [ ] **F32 QPS 压测**
  - 需求：面试官问「并发怎么样」。**必须区分 mock 与 real 两个数字**，只报一个会被追问
  - 做什么：`scripts/bench_qps.py`。mock LLM 与服务层 QPS；real 端到端 P95；含同 session 写入的 `asyncio.Lock` 分片 + CAS 版本号
  - 验收：两组数字分开报
  - 依赖：F29
  - 预估：4h
  - 产出：`reports/qps.md`

- [ ] **F33 record-replay**
  - 需求：离线调参不该烧 API 额度，且要完全可复现
  - 做什么：`scripts/replay.py`。录制 `(prompt_hash, response)` 与 `(args_hash, result)`；重放时优先命中
  - 验收：录一次后，离线重跑同一会话零 API 调用
  - 依赖：F16
  - 预估：4h

- [ ] **F34 CI + Recall 回归门禁**
  - 需求：面试官问「怎么防止改坏」——`Recall@5` 低于阈值直接红
  - 做什么：`.github/workflows/`。跑 `pytest -m "not network"`（mock LLM，免 key）+ Recall 门禁
  - 验收：CI 绿；故意让 Recall 掉下去，CI 变红
  - 依赖：F23
  - 预估：3h

- [ ] **F35 README + 演示脚本**
  - 需求：面试官第一眼看的是 README。**标题与第一条 bullet 就要亮出「可靠性/评测」**，别让他从技术栈里挖
  - 做什么：README（架构图 / 消融表 / 复现命令 / **追问速查表**）+ 5 分钟演示脚本（固定走完三个用例）
  - 验收：一个没看过项目的人能照 README 跑起来
  - 依赖：全部
  - 预估：5h

---

## 汇总

| 阶段 | 条目 | 预估总计 |
|---|---|---|
| 0 前置（你，非代码） | P1-P2 | 6-11h |
| 1 数据层 | F1-F4 | 17h |
| 2 检索层 | F5-F10 | 22h |
| 3 工具层（重写） | F11-F13 | 11-14h |
| 4 Agent 层 | F14-F18 | 23h |
| 5 记忆层 | F19-F22 | 18h |
| 6 评测层 | F23-F27 | 27h |
| 7 工程层 | F28-F35 | 34h |
| **合计** | **35 条** | **约 155-165h** |

按兼职 15-20h/周，正好是 8-11 周。**6 周排期意味着要砍掉阶段 7 的一半**（`规划.md` 里写了砍哪些）。

---

## 别一次看完整张表

你现在**只需要看这三条**：

1. **P1** —— 导出法规原文。不做完，阶段 1 之后全部无法开始。
2. **F11** —— 重写 `executor.py`。不依赖语料，现在就能做，而且是面试追问最密集的文件。
3. **F12** —— 重写 `breaker.py`。同上，3-4h 能完成。

其余 `[ ]` 先别读。做完一条再回来往下看一条。
