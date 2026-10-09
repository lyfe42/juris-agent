"""数据层：法条清洗（F1）→ 切分（F2）→ 建索引（F3）。

这一层是「语料 → 可检索」的入口，所有下游（检索、评测、Agent）都建立在
`clean.Article` 的 `article_key` 之上。改 `article_key` 的格式等于改全套接口。
"""

from __future__ import annotations

from juris_agent.ingest.clean import (
    Article,
    LawMeta,
    clean_corpus,
    corpus_hash,
    load_articles,
    load_manifest,
    parse_articles,
    read_source,
    summarize,
    write_jsonl,
)

__all__ = [
    "Article",
    "LawMeta",
    "clean_corpus",
    "corpus_hash",
    "load_articles",
    "load_manifest",
    "parse_articles",
    "read_source",
    "summarize",
    "write_jsonl",
]
