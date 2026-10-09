# 检索差异诊断（自动生成）

- 评测集 n=269　source=all
- 命中口径：gold 中任一条出现在 top-5

## 逐臂总览

| 臂 | 命中题数 | 平均首位名次（仅命中题） |
|---|---:|---:|
| `B/dense-only` | 263/269 | 2.06 |
| `B/bm25-only` | 205/269 | 3.51 |
| `B/bm25+dense RRF` | 260/269 | 2.42 |
| `B/bm25+dense 加权求和` | 260/269 | 2.23 |

## 独赢题（只有这一臂命中）

这类题最能说明某个臂的**不可替代性**。如果所有臂都没有独赢题，
说明它们高度重叠——多跑一路只是在烧算力。

### 只有 `B/dense-only` 命中（3 题）

- **g019**（human/hard）公司一直没给我交社保，我能辞职并要求补偿吗？
  - gold: ['lcl#38', 'lcl#46']　该臂首位名次: 13
  - 该臂 top-5: ['js2#19', 'lcl_tiaoli#5', 'ldf#100', 'lcl_tiaoli#34', 'js1#38']
  - 其他臂名次: {'B/bm25-only': None, 'B/bm25+dense RRF': None, 'B/bm25+dense 加权求和': None}
- **mlcl0074**（machine/normal）公司拖欠工资、不给缴社保、还老让我们超时加班，这些事劳动监察部门都能管吗？他们平时还查哪些方面？
  - gold: ['lcl#74']　该臂首位名次: 19
  - 该臂 top-5: ['ldf#101', 'tjzc#9', 'lcl#75', 'lcl#95', 'ldf#103']
  - 其他臂名次: {'B/bm25-only': None, 'B/bm25+dense RRF': None, 'B/bm25+dense 加权求和': None}
- **mlcl0090**（machine/easy）我没提前跟公司说就突然离职了，公司说因为我走了造成了损失，要我赔钱，这有法律依据吗？
  - gold: ['lcl#90']　该臂首位名次: 16
  - 该臂 top-5: ['js2#12', 'js1#35', 'ldf#28', 'lcl_tiaoli#5', 'tjzc#6']
  - 其他臂名次: {'B/bm25-only': None, 'B/bm25+dense RRF': None, 'B/bm25+dense 加权求和': None}

## 分歧题（任两臂名次差 ≥ 5，或一命中一未命中）

这里要**人眼看谁对**：是融合把正确答案挤下去了，还是另一臂召回了
评测集漏标的平行条款（那属于 gold 的问题，不是系统的问题）。

### g005　公司不跟我签书面劳动合同，要赔钱吗？

- gold: ['lcl#82']　(human/single_fact/normal)

| 臂 | 首位命中名次 | top-3 条级排名 |
|---|---:|---|
| `B/dense-only` | 16 | ['lcl_tiaoli#5', 'ldf#97', 'js2#7'] |
| `B/bm25-only` | 12 | ['lcl#10', 'lcl_tiaoli#6', 'js2#8'] |
| `B/bm25+dense RRF` | 9 | ['lcl_tiaoli#7', 'lcl_tiaoli#5', 'js2#9'] |
| `B/bm25+dense 加权求和` | 16 | ['lcl_tiaoli#5', 'lcl#10', 'lcl_tiaoli#7'] |

### g007　公司每个月最多能安排我加班多少小时？

- gold: ['ldf#41']　(human/single_fact/normal)

| 臂 | 首位命中名次 | top-3 条级排名 |
|---|---:|---|
| `B/dense-only` | 1 | ['ldf#41', 'lcl#31', 'ldf#44'] |
| `B/bm25-only` | 8 | ['lcl#31', 'mfd#200', 'ldf#61'] |
| `B/bm25+dense RRF` | 2 | ['lcl#31', 'ldf#41', 'lcl#72'] |
| `B/bm25+dense 加权求和` | 2 | ['lcl#31', 'ldf#41', 'js1#42'] |

### g010　我要辞职，得提前多久跟公司说？

- gold: ['lcl#37']　(human/single_fact/easy)

| 臂 | 首位命中名次 | top-3 条级排名 |
|---|---:|---|
| `B/dense-only` | 2 | ['ldf#31', 'lcl#37', 'lcl#40'] |
| `B/bm25-only` | 6 | ['mfd#319', 'mfd#71', 'mfd#314'] |
| `B/bm25+dense RRF` | 2 | ['ldf#31', 'lcl#37', 'mfd#319'] |
| `B/bm25+dense 加权求和` | 1 | ['lcl#37', 'ldf#31', 'mfd#319'] |

### g011　经济补偿金是怎么算的？

- gold: ['lcl#47']　(human/single_fact/normal)

| 臂 | 首位命中名次 | top-3 条级排名 |
|---|---:|---|
| `B/dense-only` | 2 | ['lcl_tiaoli#27', 'lcl#47', 'lcl_tiaoli#22'] |
| `B/bm25-only` | 9 | ['mfd#390', 'js1#53', 'mfd#206'] |
| `B/bm25+dense RRF` | 1 | ['lcl#47', 'lcl_tiaoli#22', 'lcl_tiaoli#31'] |
| `B/bm25+dense 加权求和` | 4 | ['lcl_tiaoli#27', 'mfd#390', 'js1#53'] |

### g015　劳务派遣工能和正式员工拿一样的工资吗？

- gold: ['lcl#63']　(human/single_fact/normal)

| 臂 | 首位命中名次 | top-3 条级排名 |
|---|---:|---|
| `B/dense-only` | 1 | ['lcl#63', 'lcl#62', 'lcl_tiaoli#30'] |
| `B/bm25-only` | 17 | ['lcl#60', 'lcl#59', 'lcl#65'] |
| `B/bm25+dense RRF` | 7 | ['lcl#58', 'lcl_tiaoli#30', 'lcl#65'] |
| `B/bm25+dense 加权求和` | 1 | ['lcl#63', 'lcl#60', 'lcl#58'] |

### g017　公司违法把我开除，我能拿到多少赔偿？

- gold: ['lcl#87', 'lcl#47']　(human/multi_article/normal)

| 臂 | 首位命中名次 | top-3 条级排名 |
|---|---:|---|
| `B/dense-only` | 4 | ['js2#18', 'lcl_tiaoli#25', 'ldf#28'] |
| `B/bm25-only` | **未命中** | ['js1#44', 'mfd#71', 'ldf#67'] |
| `B/bm25+dense RRF` | 8 | ['js2#18', 'js1#53', 'js1#44'] |
| `B/bm25+dense 加权求和` | 13 | ['js2#18', 'js1#44', 'lcl_tiaoli#25'] |

### g019　公司一直没给我交社保，我能辞职并要求补偿吗？

- gold: ['lcl#38', 'lcl#46']　(human/multi_article/hard)

| 臂 | 首位命中名次 | top-3 条级排名 |
|---|---:|---|
| `B/dense-only` | 13 | ['js2#19', 'lcl_tiaoli#5', 'ldf#100'] |
| `B/bm25-only` | **未命中** | ['mfd#76', 'js1#37', 'mfd#1118'] |
| `B/bm25+dense RRF` | **未命中** | ['js1#38', 'js2#19', 'mfd#76'] |
| `B/bm25+dense 加权求和` | **未命中** | ['js2#19', 'mfd#76', 'js1#37'] |

### g021　老板拖欠工资，我能直接去法院告他吗？

- gold: ['lcl#30', 'tjzc#5']　(human/multi_article/hard)

| 臂 | 首位命中名次 | top-3 条级排名 |
|---|---:|---|
| `B/dense-only` | 4 | ['js1#15', 'tjzc#16', 'tjzc#9'] |
| `B/bm25-only` | **未命中** | ['mfd#114', 'mfd#1166', 'mfd#1194'] |
| `B/bm25+dense RRF` | 8 | ['js1#15', 'js1#53', 'mfd#114'] |
| `B/bm25+dense 加权求和` | 8 | ['js1#15', 'mfd#114', 'tjzc#16'] |
