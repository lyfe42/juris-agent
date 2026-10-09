"""F1：把 data/raw/ 下的法规原文清洗成条级的 data/laws.jsonl。

产出三样东西：
  - `data/laws.jsonl`              一条一行，`article_key` 是跨模块主键
  - `data/index/corpus_meta.json`  `corpus_hash` + 条数 + 清单核对状态
  - `reports/corpus_summary.md`    人读的按法规汇总（终端是 GBK，中文只能进文件）

**这个脚本不猜生效日期。** 日期以 `configs/laws_manifest.yaml` 为准；
清单里没写的才回退到正文正则，并标 `text-regex`。未经核对的条目会打警告——
这是 P1 的验收点，警告会一直出现，直到你把 `verified` 改成 true。

用法：
    python scripts/ingest_corpus.py
    python scripts/ingest_corpus.py --strict   # 有未核对项就退出码非 0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from juris_agent.config import get_settings  # noqa: E402
from juris_agent.ingest.clean import (  # noqa: E402
    clean_corpus,
    load_manifest,
    summarize,
    write_jsonl,
)


def _write_summary_md(rows: list[dict], meta: dict, out_path: Path) -> None:
    lines = [
        "# 语料汇总（ingest 自动生成，勿手改）",
        "",
        f"- `corpus_hash`: `{meta['corpus_hash']}`",
        f"- 条数合计: **{meta['n_articles']}**",
        f"- 法规数: {meta['n_laws']}",
        f"- 生成时间: {meta['built_at']}",
        "",
        "| 法规 | 条数 | 字符数 | 生效日期 | 日期来源 | 有章的条数 |",
        "|---|---:|---:|---|---|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['law_short']} | {r['n_articles']} | {r['n_chars']} | "
            f"{r['effective_date'] or '—'} | {r['effective_date_source']} | "
            f"{r['with_chapter']} |"
        )
    lines.append("")
    lines.append(
        "> 生效日期来源 `manifest` = 来自人工清单（权威）；`text-regex` = 从正文抓的备选，"
        "需人工核对；清单里 `verified: false` 的条目全部待核。"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F1 法条清洗与条级聚合")
    ap.add_argument("--strict", action="store_true", help="存在未核对/缺文件的警告时退出码非 0")
    args = ap.parse_args(argv)

    s = get_settings()
    manifest_path = s.paths.configs / "laws_manifest.yaml"
    if not manifest_path.exists():
        print(f"[FATAL] 找不到法规清单: {manifest_path}", file=sys.stderr)
        return 2

    manifest = load_manifest(manifest_path)
    articles, warnings = clean_corpus(s.paths.raw, manifest)
    if not articles:
        print("[FATAL] 没有切出任何条文，检查 data/raw/ 与清单是否对应", file=sys.stderr)
        return 2

    digest = write_jsonl(articles, s.paths.laws)
    rows = summarize(articles)

    meta = {
        "corpus_hash": digest,
        "n_articles": len(articles),
        "n_laws": len(rows),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_verified_all": all(m.verified for m in manifest),
        "laws": rows,
    }
    s.paths.corpus_meta.parent.mkdir(parents=True, exist_ok=True)
    s.paths.corpus_meta.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_summary_md(rows, meta, s.paths.reports / "corpus_summary.md")

    # 终端是 GBK：只打 ASCII，中文进文件
    print(f"corpus_hash = {digest}")
    print(f"articles    = {len(articles)}  across {len(rows)} laws")
    print(f"{'law_id':<12}{'articles':>9}{'chars':>9}  effective")
    for r in rows:
        print(
            f"{r['law_id']:<12}{r['n_articles']:>9}{r['n_chars']:>9}  "
            f"{r['effective_date'] or '-'} ({r['effective_date_source']})"
        )
    print(f"\nwrote {s.paths.laws}")
    print(f"wrote {s.paths.corpus_meta}")
    print(f"wrote {s.paths.reports / 'corpus_summary.md'}")

    if warnings:
        print(f"\n[WARN] {len(warnings)} 条：")
        for w in warnings:
            # 警告可能含中文法规简称，转义成 ascii 以免 GBK 终端抛 UnicodeEncodeError
            print("  - " + w.encode("ascii", "backslashreplace").decode("ascii"))
    if args.strict and warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
