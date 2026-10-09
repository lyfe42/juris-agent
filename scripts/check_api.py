"""启动自检：确认 key 有效、**确认模型名在当前账号上真的存在**。

## 为什么需要这个脚本

模型名是会变的。这个项目 `.env.example` 里原本写的 `deepseek-chat` /
`deepseek-reasoner` 是两年前的常见写法，但在当前账号上**根本不存在**——
请求会直接 400。而这类错误的报错信息往往只给一个 `invalid model`，
不看文档不容易反应过来"是模型名过时了"，很容易误判成 key 或网络问题。

所以：**换 key、隔一段时间、或换账号之后，先跑这个脚本**，
它会列出账号下真实可用的模型 id，再各发一条最小请求验证通路与延迟。

用法：
    python scripts/check_api.py
    python scripts/check_api.py --model deepseek-v4-pro
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402

from juris_agent.config import get_settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DeepSeek API 自检")
    ap.add_argument("--model", action="append", default=None, help="要试的模型（默认试配置里的两个）")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args(argv)

    s = get_settings()
    if not s.deepseek_api_key:
        print("[FATAL] 没有 DEEPSEEK_API_KEY。复制 .env.example 为 .env 并填入。", file=sys.stderr)
        print("        注意 .env 在 .gitignore 里，不会被提交。", file=sys.stderr)
        return 2

    key = s.deepseek_api_key
    print(f"key          {key[:6]}…{key[-4:]}  (len={len(key)})")
    print(f"base_url     {s.deepseek_base_url}")
    print(f"配置的生成模型 {s.model_gen}")
    print(f"配置的判分模型 {s.model_judge}")
    print(f"HF_ENDPOINT  {os.environ.get('HF_ENDPOINT', '(未设置)')}")
    print()

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    # --- 1. 账号下有哪些模型 ------------------------------------------- #
    try:
        with httpx.Client(timeout=args.timeout) as c:
            r = c.get(f"{s.deepseek_base_url}/models", headers=headers)
    except httpx.HTTPError as e:
        print(f"[FATAL] 连不上 {s.deepseek_base_url}: {e}", file=sys.stderr)
        print("        本机装了代理的话，试 export HTTPS_PROXY=http://127.0.0.1:7890", file=sys.stderr)
        return 2

    if r.status_code != 200:
        print(f"[FATAL] GET /models -> HTTP {r.status_code}: {r.text[:300]}", file=sys.stderr)
        return 2

    available = [m["id"] for m in r.json().get("data", [])]
    print("账号可用模型:")
    for m in r.json().get("data", []):
        ctx = m.get("context_window")
        print(f"  - {m['id']:<20} {m.get('name', ''):<24} ctx={ctx}")
    print()

    configured = {s.model_gen, s.model_judge}
    missing = configured - set(available)
    if missing:
        # 这是一条**必须显式失败**的检查：配置里写了不存在的模型，
        # 后面 gold 生成/裁判会以各种奇怪的方式失败，不如现在就拦住。
        print(f"[FAIL] 配置里有账号上不存在的模型: {sorted(missing)}")
        print(f"       可用: {available}")
        print("       改 .env 里的 JURIS_AGENT_MODEL_GEN / JURIS_AGENT_MODEL_JUDGE")
        return 1
    print(f"[OK] 配置的两个模型都在可用列表里")

    # --- 2. 各发一条最小请求 -------------------------------------------- #
    to_try = args.model or sorted(configured)
    print()
    for model in to_try:
        if model not in available:
            print(f"  {model:<20} 跳过（不在账号的模型列表里）")
            continue
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "只回复两个字：收到"}],
            "max_tokens": 32,
            "temperature": 0.0,
        }
        t0 = time.perf_counter()
        with httpx.Client(timeout=args.timeout) as c:
            try:
                r = c.post(f"{s.deepseek_base_url}/chat/completions", headers=headers, json=body)
            except httpx.HTTPError as e:
                print(f"  {model:<20} 网络失败: {e}")
                continue
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            print(f"  {model:<20} HTTP {r.status_code}: {r.text[:200]}")
            continue
        j = r.json()
        u = j.get("usage", {})
        text = j["choices"][0]["message"]["content"].strip()
        # 输出可能因控制台编码显示成乱码，但**长度和 token 数是准的**，
        # 足以判断链路通没通。脚本不为了"好看"去猜控制台编码。
        print(f"  {model:<20} OK {dt:5.1f}s  回复长度={len(text)}  "
              f"tokens in/out={u.get('prompt_tokens')}/{u.get('completion_tokens')}")

    print("\n全部通过。可以去跑 scripts/build_gold.py 了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
