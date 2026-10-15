# 长对话记忆消融（自动生成，勿手改）

生成命令：`python scripts/memory_ablation.py`

- 评测集：`data/eval/memory_sessions.yaml`，15 段 × 20 轮
- 探针 45 处、期望值 52 条（一处探针可以同时查好几个槽位）
- 记忆预算：`MemoryBudget()` 默认值（total=2048 token，窗口 6 轮）
- 成本按 `configs/pricing.yaml` 的牌价换算（输入 2.0 元/百万 token），**是估算不是账单**

## 先说清楚这个指标是什么

**测的是「提示词里有没有那个值」，不是「模型答得对不对」。**

| | |
|---|---|
| 测什么 | 组装出来的提示词里，关键槽位值还在不在 |
| 怎么判 | 期望值写成 `月工资：12000` 这样的完整片段，做子串匹配 |
| 为什么这么测 | 便宜、可复现、失败可归因（「没进去」和「进去了但没用上」是两种 bug） |
| ⚠️ 局限 | 提示词里有 ≠ 模型会用。所以这是**上界**，不是端到端正确率 |

「隔了多少轮」不是手填的，是从每个槽位自己的 `turn` 记录里算出来的——
**能算出来的数不要手写**，手写的一定会和轮次文本漂移，而漂移了也不报错。

## 事实分成两类，**这条比主表重要**

同一个「记住」，对两类事实是完全不同的难度：

| 类型 | 例子 | 用户原话里长什么样 | 谁承载得住 |
|---|---|---|---|
| **照抄型** | 月工资 `12000`、城市 `西安`、离职日 `2024-05-10` | 原话里**就是**这个值 | 窗口 / 摘要 / 槽位都行 |
| **换算型** | 工作年限 `37`、是否提前通知 `False` | 「三年零一个月」「没提前通知」——**原文里没有这个值** | **只有槽位层** |

**为什么必须分开报：** 换算型的事实，原文里根本不存在那个写法，
所以对它们来说「四种策略对比」是个假命题——只有槽位层答得出来，
剩下三个必然是 0。**把两类混在一起，你会得到一个好像是「插了槽位层就赢麻了」的表，其实那个差距是结构性的，不是测出来的。**

分类是**程序判的**（`classify_quotable`：拿值的骨架去它的原文那一轮里找），
不是人在 YAML 里标的——标错了没人看得出来，算出来的不会骗人。

## 主表

| 策略 | **照抄型保持率** | 照抄型 通过/总数 | 换算型保持率 | 换算型 通过/总数 | 平均提示词 token | 估算成本（元/千次咨询） |
|---|---|---|---|---|---|---|
| `no_memory` | **0.000** | 0/32 | 0.000 | 0/20 | 49 | 0.10 |
| `sliding_window` | **0.500** | 16/32 | 0.100 | 2/20 | 330 | 0.66 |
| `sliding_window_summary` | **0.188** | 6/32 | 0.150 | 3/20 | 265 | 0.53 |
| `layered` | **1.000** | 32/32 | 1.000 | 20/20 | 360 | 0.72 |

（成本列 = 平均提示词 token × 输入牌价 ÷ 1e6 × 1000）

**读表的方式：**「照抄型」那一列才是四种策略真正的区别所在——
它问的是「用户几十轮前说的那句话，还在不在上下文里」。
「换算型」那一列**不是对比**，它是在确认槽位层有没有正常工作；
除 `layered` 外全为 0 是设计使然，不是缺陷。

## 保持率曲线：隔得越久，还记得吗

横轴是「这个事实是多少轮之前说的」。**这一栏才是这份报告的重点**——
主表只有一个总数，它会把「最近的记得牢」和「远的全忘光」平均掉。

| 策略 | 0-5 轮前 | 6-10 轮前 | 11+ 轮前 |
|---|---|---|---|
| `no_memory` | 0.00 | 0.00 | 0.00 |
| `sliding_window` | 1.00 | 1.00 | 0.00 |
| `sliding_window_summary` | 1.00 | 0.20 | 0.12 |
| `layered` | 1.00 | 1.00 | 1.00 |

## 逐条明细（只列失败的那些）

「哪一段、第几轮、缺了哪个槽位、隔了多少轮、**谁挂了**」——
这几个信息齐了，才能判断是「设计如此」还是「实现有 bug」。

⚠️ 按「探针」聚合成一行，不是每条失败一行。第一版一条一行的写法
一共 130 行，读的人第一条就划走了——**明细表的作用是让人能扫一眼看出模式，不是把原始数据倒出来**（原始数据在 `memory_ablation.json` 里，谁要谁去拿）。

| 对话 | 第几轮 | 期望 | 隔了多少轮 | 挂了的策略 |
|---|---|---|---|---|
| s01_layoff | 14 | `城市：西安` | 12 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s01_layoff | 20 | `工作年限（月）：37` | 1 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s02_illegal | 14 | `工作年限（月）：27` | 12 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s02_illegal | 14 | `是否拖欠劳动报酬：True` | 4 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s02_illegal | 18 | `城市：杭州` | 17 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s03_probation | 14 | `合同期限（月）：24` | 13 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s04_overtime | 14 | `加班小时数：3` | 12 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s04_overtime | 20 | `月工资：7200` | 19 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s05_no_written_contract | 20 | `城市：武汉` | 17 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s06_wage_arrears | 14 | `工作年限（月）：50` | 11 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s06_wage_arrears | 20 | `城市：郑州` | 18 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s06_wage_arrears | 20 | `是否拖欠劳动报酬：True` | 4 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s07_limitation | 14 | `工作年限（月）：66` | 11 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s07_limitation | 20 | `劳动关系终止之日：2024-05-10` | 19 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s08_noncompete | 14 | `合同期限（月）：24` | 12 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s08_noncompete | 20 | `城市：合肥` | 17 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s09_notice_pay | 14 | `工作年限（月）：43` | 12 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s09_notice_pay | 20 | `是否提前30日通知：False` | 16 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s09_notice_pay | 20 | `城市：天津` | 17 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s10_expiry | 14 | `合同期限（月）：24` | 11 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s10_expiry | 18 | `城市：青岛` | 16 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s10_expiry | 18 | `工作年限（月）：24` | 14 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s11_social_insurance | 14 | `工作年限（月）：82` | 11 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s11_social_insurance | 20 | `城市：福州` | 18 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s12_maternity | 14 | `工作年限（月）：29` | 12 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s12_maternity | 20 | `城市：南京` | 17 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s13_part_time | 20 | `城市：东莞` | 17 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s14_dispatch | 14 | `工作年限（月）：40` | 11 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s14_dispatch | 20 | `城市：苏州` | 18 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s15_commission | 14 | `工作年限（月）：57` | 1 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s15_commission | 20 | `城市：上海` | 17 | `no_memory`、`sliding_window`、`sliding_window_summary` |
| s01_layoff | 8 | `月工资：12000` | 7 | `no_memory`、`sliding_window_summary` |
| s02_illegal | 8 | `月工资：9800` | 7 | `no_memory`、`sliding_window_summary` |
| s03_probation | 8 | `月工资：15600` | 7 | `no_memory`、`sliding_window_summary` |
| s03_probation | 20 | `约定试用期（月）：6` | 11 | `no_memory`、`sliding_window` |
| s03_probation | 20 | `城市：成都` | 15 | `no_memory`、`sliding_window` |
| s04_overtime | 8 | `月工资：7200` | 7 | `no_memory`、`sliding_window_summary` |
| s05_no_written_contract | 8 | `月工资：8500` | 7 | `no_memory`、`sliding_window_summary` |
| s06_wage_arrears | 8 | `月工资：11000` | 7 | `no_memory`、`sliding_window_summary` |
| s07_limitation | 20 | `城市：长沙` | 16 | `no_memory`、`sliding_window` |
| s08_noncompete | 8 | `月工资：18000` | 7 | `no_memory`、`sliding_window_summary` |
| s09_notice_pay | 8 | `月工资：10500` | 7 | `no_memory`、`sliding_window_summary` |
| s11_social_insurance | 8 | `月工资：12800` | 7 | `no_memory`、`sliding_window_summary` |
| s13_part_time | 8 | `月工资：3000` | 7 | `no_memory`、`sliding_window_summary` |
| s14_dispatch | 8 | `月工资：6600` | 7 | `no_memory`、`sliding_window_summary` |
| s15_commission | 8 | `月工资：14200` | 7 | `no_memory`、`sliding_window_summary` |
| s04_overtime | 14 | `城市：深圳` | 9 | `no_memory` |
| s05_no_written_contract | 14 | `工作年限（月）：12` | 5 | `no_memory` |
| s07_limitation | 8 | `月工资：13500` | 5 | `no_memory` |
| s10_expiry | 8 | `月工资：7800` | 6 | `no_memory` |
| s12_maternity | 8 | `月工资：9500` | 6 | `no_memory` |
| s13_part_time | 14 | `工作年限（月）：12` | 10 | `no_memory` |

**看最后一列。** 挂满四个策略的，多半是「这个事实本来就该丢」；
只挂在某几个策略上的，才是有信息量的那一行——
**同一个事实、同一段对话，差别只在「记忆策略怎么组装」，那才是策略的功劳。**

## 逐段保持率

| 对话 | `no_memory` | `sliding_window` | `sliding_window_summary` | `layered` |
|---|---|---|---|---|
| s01_layoff | 0/3 | 1/3 | 0/3 | 3/3 |
| s02_illegal | 0/4 | 1/4 | 0/4 | 4/4 |
| s03_probation | 0/4 | 1/4 | 2/4 | 4/4 |
| s04_overtime | 0/4 | 2/4 | 1/4 | 4/4 |
| s05_no_written_contract | 0/3 | 2/3 | 1/3 | 3/3 |
| s06_wage_arrears | 0/4 | 1/4 | 0/4 | 4/4 |
| s07_limitation | 0/4 | 1/4 | 2/4 | 4/4 |
| s08_noncompete | 0/3 | 1/3 | 0/3 | 3/3 |
| s09_notice_pay | 0/4 | 1/4 | 0/4 | 4/4 |
| s10_expiry | 0/4 | 1/4 | 1/4 | 4/4 |
| s11_social_insurance | 0/3 | 1/3 | 0/3 | 3/3 |
| s12_maternity | 0/3 | 1/3 | 1/3 | 3/3 |
| s13_part_time | 0/3 | 2/3 | 1/3 | 3/3 |
| s14_dispatch | 0/3 | 1/3 | 0/3 | 3/3 |
| s15_commission | 0/3 | 1/3 | 0/3 | 3/3 |

## layered 那 1.000 分，到底是谁挣的

下表统计：在**所有探针轮次**里，每一层实际放进去几条。
0 = 这一层那一轮什么都没帮上。

| 层 | 有内容的探针轮次数 | 占比 |
|---|---|---|
| 探针轮次总数 | 45 | 100% |
| `slots` | 45 | 100% |
| `recent` | 45 | 100% |
| `summaries` | 30 | 67% |
| `recall` | 10 | 22% |

**这张表是用来泼冷水的，别跳过。**「四层架构拿了满分」这句话是错的——
分数几乎全部来自**槽位层**，它是唯一一个「只要开了就必然命中」的层。
长期召回层在相当多的探针上**一条都没召回**（检索词和问题没有重叠），
所以它在这份评测里的贡献远小于它的名气。
**把「我插了一个结构化的槽位表」讲成「我的四层记忆架构很强」，
被追问一句就散了；自己先把这行数字摆出来，反而是最稳的。**

## 这份评测**测不到**什么

写结论之前先把边界划清楚——下面这几条都是真的，不是免责声明：

1. **槽位覆盖不到的事实，这里一条都没测。** 探针查的全是槽位表里的字段。
   用户说过的「我下周三要去仲裁」这种**没有对应槽位**的信息，
   本评测的保持率**不含它**——那部分只能靠摘要和原文，
   而这两层在题目里正好是弱项。
2. **`layered` 在换算型上恒为 1.000 是同义反复。** 槽位层就是为这些事实
   设计的，它不渲染出来才是 bug。**这个 1.000 不代表任何「效果」。**
3. **提示词里有 ≠ 模型会用。** 这是全文最要紧的一条局限，见开头那张表。
4. **助手回复是假的**（一句不含数字的套话），所以真实对话里
   助手引用过的事实，在真实的窗口里占的位置和这里不一样。
5. **摘要是规则拼的，不是模型写的。** 换成模型摘要，这一列大概率会变——
   但那就换来了一个不可复现的数（`temperature=0` 也不可复现，
   见 `reports/diagnosis.md` 的实测）。**这是刻意的取舍，不是没做。**

## 从这张表里读出来的事

1. **基线先摆出来：`no_memory` 的照抄型保持率是 0.000（0/32）。**
   ——它只带最后一轮的原文，所以它记住的永远是「刚刚那句话」。**它不是「没用」，它是标尺**：后面每个策略的收益，都要先减掉这个数。

2. **两个方案花的 token 差不多（`sliding_window` 330 vs `layered` 360，少 8%），但**曲线的形状完全不一样**。**
   - `sliding_window`：0-5 轮前 1.00、6-10 轮前 1.00、11+ 轮前 0.00
   - `layered`：0-5 轮前 1.00、6-10 轮前 1.00、11+ 轮前 1.00
   滑动窗口是**硬边界**——窗口盖得住的记得住，盖不住的一律归零，没有中间地带。这就是它危险的地方：**它在第 10 轮还是满分，第 11 轮就变成 0 分，而用户不会知道正在跨过那条线。**
   `layered` 是平的，因为它把「必须记住的东西」从窗口里搬出来、放进一个不吃压缩的槽位表——**花同样的钱，把硬边界抹掉了。**

3. **摘要（`sliding_window_summary` = 0.188）对比纯滑动窗口（`sliding_window` = 0.500）：低 0.312。**
   这一栏的数字**是反直觉的**，得说清楚为什么：两者**总预算一样**，区别在怎么花——`sliding_window` 把预算全砸在原文上（窗口被拉到 10 轮），`sliding_window_summary` 保持 6 轮原文 + 拿剩下的预算放摘要。
   而这里的摘要是**规则拼出来的**：它从一段历史里挑几句用户原话串起来。挑中的句子就是全部——没被挑中的，连数字带上下文一起没了。**摘要能把「聊过什么」带过去，带不过去精确的值。这就是为什么槽位层必须单独存在：精确的事实绝不交给有损压缩承载。**

4. **`layered` 在换算型上是 1.000。别把这行当成成绩。**
   换算型的事实（「三年零一个月」→ 工作年限 37、「没提前通知」→ False）**在原话里根本找不到那个写法**，只有槽位层能给出。所以对它们来说「四种策略对比」是假命题——**1.000 的意思是「槽位层正常工作」，不是「效果好」。**
