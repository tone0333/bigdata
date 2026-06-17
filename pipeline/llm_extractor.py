"""
LLM 特征提取器 — 独立运行的异步高并发管道。

功能：
  - 调用 SiliconFlow API (OpenAI 兼容) 对电商评论文本进行结构化特征抽取
  - 提取 sentiment（情感）、category（类别）、summary（摘要）
  - 指数退避自动重试（tenacity），Semaphore 并发控制
  - 结果与原始数据水平拼接，输出到 data/ 目录

用法：
  python pipeline/llm_extractor.py
  python pipeline/llm_extractor.py --input D:/path/to/reviews.csv --limit 500 --concurrency 10

前置条件：
  - 设置环境变量 SILICONFLOW_API_KEY
"""
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from openai import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    APITimeoutError,
    AsyncOpenAI,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm.asyncio import tqdm_asyncio

# ── 把项目根目录加入路径 ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ── 颜色输出 ───────────────────────────────────────────
YELLOW = "\033[93m"
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

# ── LLM Prompt ──────────────────────────────────────────
SYSTEM_PROMPT = """你是一个电商评论分析助手。请分析用户给出的网购评论，提取结构化特征。

要求：
1. sentiment（情感倾向）：正面 / 负面 / 中性
2. category（评论类别）：从评论内容中归纳一个最合适的类别标签，如 产品质量、物流服务、性价比、使用体验、书籍内容、售后服务 等
3. summary（一句话摘要）：用不超过 30 个字概括评论核心观点

严格只返回如下 JSON 格式，不要包含任何其他文字：
{"sentiment": "...", "category": "...", "summary": "..."}"""

RETRYABLE_ERRORS = (
    RateLimitError,
    APIConnectionError,
    InternalServerError,
    APITimeoutError,
)


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
)
async def extract_features(text: str, sem: asyncio.Semaphore, client: AsyncOpenAI) -> dict:
    """调用 LLM 提取单条评论的特征，带并发控制和自动重试。"""
    async with sem:
        response = await client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=200,
            extra_body={"enable_thinking": False},
        )

    content = response.choices[0].message.content.strip()
    # 剥离可能的 markdown 代码围栏
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "sentiment": "解析失败",
            "category": "解析失败",
            "summary": content[:100] if content else "空响应",
        }


async def run_extraction(
    input_path: str,
    output_path: str,
    limit: int,
    concurrency: int,
):
    """执行批量 LLM 特征提取。"""
    # ── 0. API Key 守卫 ──
    api_key = config.SILICONFLOW_API_KEY
    if not api_key:
        print()
        print(RED + "=" * 58 + RESET)
        print(RED + "  错误: 未设置 SILICONFLOW_API_KEY 环境变量" + RESET)
        print(RED + "  请设置后重试:" + RESET)
        print(RED + "    Windows: set SILICONFLOW_API_KEY=sk-xxx" + RESET)
        print(RED + "    Linux/Mac: export SILICONFLOW_API_KEY=sk-xxx" + RESET)
        print(RED + "=" * 58 + RESET)
        print()
        sys.exit(1)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=config.SILICONFLOW_BASE_URL,
        timeout=30.0,
    )

    # ── 1. 读取数据 ──
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"{RED}输入文件不存在: {input_path}{RESET}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    if limit:
        df = df.head(limit)
    texts = df["review"].tolist()
    print(f"读取 {len(texts)} 条评论")

    # ── 2. 并发抽取 ──
    sem = asyncio.Semaphore(concurrency)
    print(f"开始并发处理（并发数: {concurrency}）...")
    tasks = [extract_features(text, sem, client) for text in texts]
    results = await tqdm_asyncio.gather(*tasks, desc="特征抽取")

    # ── 3. 保存 ──
    results_df = pd.DataFrame(results)
    final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"{GREEN}完成！已保存 {len(final_df)} 条记录到 {output_file}{RESET}")


def main():
    parser = argparse.ArgumentParser(description="LLM 特征提取器")
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.RAW_CSV),
        help=f"输入 CSV 路径（默认: {config.RAW_CSV}）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.FEATURES_CSV),
        help=f"输出 CSV 路径（默认: {config.FEATURES_CSV}）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="处理条数上限（default: 1000）",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=config.LLM_CONCURRENCY,
        help=f"并发数（default: {config.LLM_CONCURRENCY}）",
    )
    args = parser.parse_args()

    # 检查 API Key
    if not config.SILICONFLOW_API_KEY:
        print(YELLOW + "⚠ SILICONFLOW_API_KEY 未设置，将无法调用 LLM API" + RESET)

    asyncio.run(run_extraction(args.input, args.output, args.limit, args.concurrency))


if __name__ == "__main__":
    main()
