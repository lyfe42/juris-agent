"""F4 评测 gold 集（反向构造）。

## 为什么是"反向"

正常流程是"先有问题，再找答案"——但在法律领域这行不通：我不懂法律，编不出
靠谱的问题，更没法判断"这道题的正确答案是哪几条"。

反向构造把因果掉过来：**从法条出发造问题**。

    法条 → 让 LLM 写成"用户会怎么问" → 问题的答案（这条法条）天然已知

于是 gold 标注**不需要任何法律知识**，也不需要一个法律专家来复核——
因为答案不是"人判断出来的"，是从构造过程中**继承**来的。
这是选法律领域当载体的第二个收益（第一个是法条编号可以直接当 gold）。

## 两个必须诚实处理的偏差

### 偏差一：机器造的题措辞贴着原文，指标会虚高

LLM 改写时难免复用原文的词（"延长工作时间"），于是这条题在 BM25 上天然好召回。
**这不是模型的错，是构造方式的固有性质。** 处理办法不是掩盖，是三重：

1. prompt 里明确要求"像普通打工人那样说话，不要用法条术语"；
2. 标注 `source="machine"`，指标**按 source 分切片报**——
   `human` 那 21 条是人工写的口语题，两个切片一比就知道偏差有多大；
3. 脚本末尾**量化**这个偏差：算每条 query 与它 gold 原文的 token 重合度（Jaccard），
   机器题 vs 人工题对比。这个数直接写进报告，不猜。

### 偏差二：gold 可能漏掉并行条款

同一个规则常常在多部法规里各写一遍（试用期上限在《劳动合同法》19 条和
《劳动法》21 条都有）。LLM 只看到一条，造出的题只会指向这一条。
所以这里做了**程序化的 gold 扩展**：扫条文里的「第X条」引用，
把被引用的条并进 gold。但**引用扫不出来"并行条款"**——那是语义关系，
不是字面引用。这个偏差无法在生成阶段消除，只能：
- 在报告里如实标注"机器题的 gold 召回覆盖口径偏乐观"；
- 靠人工抽查去补（已有的 21 条 human 里就这样补过两处）。

## dev/test 隔离

按 `sha256(id)` 取模切分，**不用随机数**——随机切分每次跑出的集合都不一样，
"上次 test 上 0.61"这句话就失去了意义。确定性切分保证：
只要生成参数不变，集合就不变，历史数字才可比。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from juris_agent.config import ensure_dirs, get_settings  # noqa: E402
from juris_agent.ingest.clean import Article, cn_to_int, load_articles  # noqa: E402
from juris_agent.llm import LLMClient  # noqa: E402
from juris_agent.retrieval.bm25 import Tokenizer  # noqa: E402

# 扫条文里的「第X条」引用，用于扩展 gold。
# 刻意**不匹配**「第X条之一」这类——它们在本语料里没出现，加了反而会误配。
_REF_RE = re.compile(r"第([一二三四五六七八九十百千零〇两]+)条")

_SYSTEM = """你是中国劳动法领域的评测集构造助手。给定一条法条正文，你要造出一道\
**普通打工人会问的问题**，且这道题的正确答案就是这条法条。

要求：
1. **用生活化口语提问**，不要照抄法条里的术语。
   反例（照抄，不合格）：「延长工作时间每月不得超过多少小时？」
   正例（口语，合格）：「公司每个月最多能让我加多久的班？」
2. 问题必须能被**这一条法条**回答，不要引入这条法条没涉及的其他法律概念。
3. 不要提及"第X条""本法""根据规定"这类字样——真人提问不会这么说。
4. 如果这条法条**没法**造出一道自然的问题（例如纯定义、纯程序性条款、
   单纯说"依照X法执行"），就把 askable 设为 false。**宁可少造，不要硬造**——
   硬造出来的题 gold 是错的，会污染整个评测集。

输出 JSON：
{
  "askable": true/false,
  "question": "问题文本（askable=false 时为空串）",
  "qtype": "single_fact | multi_article | calculation | temporal",
  "difficulty": "easy | normal | hard",
  "reason": "一句话说明为什么这道题切题（或为什么不可造）"
}"""


def _strip_prompt_leak(q: str) -> str:
    """防呆：把模型偶尔漏出来的元话术清掉。"""
    q = q.strip().strip("\"'「」")
    for junk in ("问题：", "问：", "答案：", "提问："):
        if q.startswith(junk):
            q = q[len(junk):].strip()
    return q


def referenced_keys(article: Article, valid: set[str]) -> list[str]:
    """扫出条文里引用的其他条（同法内），用于扩展 gold。"""
    out: list[str] = []
    for m in _REF_RE.finditer(article.text):
        try:
            num = cn_to_int(m.group(1))
        except (ValueError, KeyError):
            continue
        key = f"{article.law_id}#{num}"
        if key != article.article_key and key in valid and key not in out:
            out.append(key)
    return out


def candidate_articles(articles: list[Article], min_chars: int) -> list[Article]:
    """粗筛可造题的条。

    只是**粗筛**，真正的"能不能造题"由 LLM 判断（见 prompt 第 4 条）。
    这里滤掉：太短的（"依照本法执行。"）、没有实际规范的。
    长度阈值故意设得低——把判断权交给模型，粗筛只负责省 token。
    """
    out = []
    for a in articles:
        if a.char_count < min_chars:
            continue
        # 纯程序性/转授权条款：一条只讲"依照X办理"的造不出好题
        if re.fullmatch(r"[^。]*依照[^。]*[办理|执行|规定|实施]。?", a.text):
            continue
        out.append(a)
    return out


def stratified_sample(articles: list[Article], n: int, seed: str) -> list[Article]:
    """按法规分层抽样，按语料规模配额，每部法至少 1 个位子。

    排序用 `sha256(seed + article_key)` 而不是 `random`——**确定性**，
    同样的 seed 永远抽出同一批题，评测集才可复现。

    ⚠️ 这个函数**必须在剔除干扰语料之前调用**（`--exclude-law` 在 main 里生效），
    否则配额会按剩下的规模重算。
    """
    from collections import defaultdict

    by_law: dict[str, list[Article]] = defaultdict(list)
    for a in articles:
        by_law[a.law_id].append(a)

    total = sum(len(v) for v in by_law.values())
    picked: list[Article] = []
    for law_id, arts in sorted(by_law.items()):
        quota = max(1, round(n * len(arts) / total))
        arts_sorted = sorted(arts, key=lambda a: hashlib.sha256(f"{seed}{a.article_key}".encode()).hexdigest())
        picked.extend(arts_sorted[:quota])

    # 配额四舍五入会多出几个，按全局确定性顺序裁到 n
    picked.sort(key=lambda a: hashlib.sha256(f"{seed}{a.article_key}".encode()).hexdigest())
    return picked[:n]


def split_of(item_id: str, test_ratio: float = 0.33) -> str:
    """确定性切分。`sha256(id)` 的整数前缀 mod 1000 落在 test 区间就进 test。"""
    h = int(hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:8], 16)
    return "test" if (h % 1000) < test_ratio * 1000 else "dev"


def token_jaccard(tok: Tokenizer, a: str, b: str) -> float:
    """两条文本的 token 集合重合度。用来量化"机器题贴着原文"这个偏差。"""
    sa, sb = set(tok(a)), set(tok(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="反向构造检索评测集（F4）")
    ap.add_argument("--n", type=int, default=180, help="机器题总数（默认 180 = 120 dev + 60 test 附近）")
    ap.add_argument("--min-chars", type=int, default=25)
    ap.add_argument("--seed", default="juris-v1", help="抽样种子；改它 = 换一批题，历史数字作废")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--mode", default=None, choices=["off", "record", "replay"])
    ap.add_argument("--out", default="data/eval/retrieval_gold_machine.jsonl")
    # 只抽不调 API：先看清将要花多少 token 再决定跑不跑
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（试跑用）")
    ap.add_argument("--exclude-law", action="append", default=None,
                    help="不从这些法规出题（默认 mfd，见下方说明）")
    args = ap.parse_args(argv)

    s = get_settings()
    ensure_dirs(s)
    if not s.paths.laws.exists():
        print("[FATAL] 没有 data/laws.jsonl，先跑 ingest_corpus.py", file=sys.stderr)
        return 2

    # ⚠️ 默认把民法典排除出**出题来源**，但**不**把它从语料里删掉——
    # 它在检索侧继续当干扰项（README 里"民法典整部保留当干扰"就是这个意思），
    # 只是不让它来出题。
    #
    # 为什么必须排除：民法典占语料 1260/1730 = 73%，按规模配额抽样的话
    # 评测集会有 73% 是民法典题，而用户问的是劳动法。
    # 第一版就是这么抽的（179 条里 131 条 mfd），指标会被民法典主导，
    # "劳动法检索行不行"这个问题根本答不上来。
    exclude = set(args.exclude_law) if args.exclude_law is not None else {"mfd"}

    articles = load_articles(s.paths.laws)
    valid = {a.article_key for a in articles}
    cands = [a for a in candidate_articles(articles, args.min_chars) if a.law_id not in exclude]
    sample = stratified_sample(cands, args.n, args.seed)

    from collections import Counter

    print(f"语料 {len(articles)} 条 → 粗筛后 {len(cands)} 条 → 抽样 {len(sample)} 条")
    print("抽样分布:", dict(Counter(a.law_id for a in sample)))

    if args.limit:
        sample = sample[: args.limit]
        print(f"--limit 生效，只跑 {len(sample)} 条")

    if args.dry_run:
        est_in = sum(a.char_count for a in sample) + 600 * len(sample)
        print(f"[dry-run] 预计输入约 {est_in // 4} tokens（粗估），未调用 API")
        return 0

    out_path = s.paths.root / args.out
    dict_path = s.paths.configs / "legal_dict.txt"
    tok = Tokenizer(dict_path=dict_path, use_dict=False)

    with LLMClient(mode=args.mode) as client:
        requests = [
            {
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": f"法条（{a.law_name} {a.article_no}）：\n{a.text}"},
                ],
                "model": s.model_gen,
                "temperature": 0.0,
                # 推理模型的思维链**也计入 max_tokens**，给 512 会被思考吃光、
                # 正文返回空串（实测 deepseek-flash 在 32 时就是这样）。
                # 给足 2048；万一还不够，`LLMClient.chat` 会自动翻 4 倍重试。
                "max_tokens": 2048,
                "response_format": {"type": "json_object"},
            }
            for a in sample
        ]
        responses = client.complete_many(requests, concurrency=args.concurrency)

        items: list[dict] = []
        skipped = 0
        for art, resp in zip(sample, responses):
            try:
                d = json.loads(resp.text)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not d.get("askable"):
                skipped += 1
                continue
            q = _strip_prompt_leak(str(d.get("question", "")))
            if len(q) < 6:
                skipped += 1
                continue
            item_id = f"m{art.law_id}{art.article_num:04d}"
            gold = [art.article_key] + referenced_keys(art, valid)
            items.append(
                {
                    "id": item_id,
                    "query": q,
                    "gold": gold,
                    "qtype": d.get("qtype", "single_fact"),
                    "difficulty": d.get("difficulty", "normal"),
                    "source": "machine",
                    "split": split_of(item_id),
                    "note": f"反向构造自 {art.article_key}｜{d.get('reason', '')[:80]}",
                }
            )

        usage = client.usage_total

    # 去重（同一法条不可能重复，但 id 撞车要防）
    seen: set[str] = set()
    uniq: list[dict] = []
    for it in items:
        if it["id"] in seen:
            continue
        seen.add(it["id"])
        uniq.append(it)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in uniq:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    n_dev = sum(1 for i in uniq if i["split"] == "dev")
    n_test = len(uniq) - n_dev
    print(f"\n产物 {out_path}")
    print(f"  机器题 {len(uniq)} 条（dev {n_dev} / test {n_test}）；模型判为不可造 {skipped} 条")
    print(f"  调用 {len(sample)} 次；tokens: in={usage.prompt_tokens} out={usage.completion_tokens} "
          f"(cache hit {usage.prompt_cache_hit_tokens})")

    # --- 量化"机器题贴着原文"这个偏差 ---------------------------------- #
    art_by_key = {a.article_key: a for a in articles}
    machine_j = [
        token_jaccard(tok, i["query"], art_by_key[i["gold"][0]].text)
        for i in uniq
        if i["gold"] and i["gold"][0] in art_by_key
    ]
    human_path = s.paths.eval / "retrieval_gold.jsonl"
    human_j: list[float] = []
    if human_path.exists():
        for line in human_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            h = json.loads(line)
            if h["gold"] and h["gold"][0] in art_by_key:
                human_j.append(token_jaccard(tok, h["query"], art_by_key[h["gold"][0]].text))

    if machine_j:
        m_mean = sum(machine_j) / len(machine_j)
        print(f"\n[偏差量化] query 与 gold 原文的 token Jaccard 重合度")
        print(f"  机器题 (n={len(machine_j)}): 均值 {m_mean:.3f}")
        if human_j:
            h_mean = sum(human_j) / len(human_j)
            print(f"  人工题 (n={len(human_j)}):   均值 {h_mean:.3f}")
            # ⚠️ 只在样本足够时才下结论。
            # 第一版这里无论 n 多小都打印"机器题明显更高"，在 n=3 时会
            # 拿 0.088 vs 0.069 去断言一个统计上毫无意义的差异——
            # 这正是我在别处反复警告的"用数据支撑不了的话"。宁可不说。
            if len(machine_j) >= 30 and len(human_j) >= 10:
                diff = m_mean - h_mean
                verdict = "明显更高" if diff > 0.05 else "接近" if abs(diff) <= 0.05 else "反而更低"
                print(f"  差值 {diff:+.3f} → 机器题{verdict}。"
                      "措辞越贴原文，它的 Recall 越偏乐观；"
                      "所以指标必须按 source 分切片报，不能混在一起当总体成绩。")
            else:
                print("  （样本太少，不足以据此下结论——只记录数字，不做判断）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
