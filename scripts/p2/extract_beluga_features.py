import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles
from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from tqdm.asyncio import tqdm

# =================配置区域=================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

print(API_KEY)
exit()

# 路径配置
INPUT_ROOT = Path("data/BelugaID")
OUTPUT_ROOT = Path("annotations/BelugaID/p2/single_image_prompts")
PROMPT_FILE = Path("scripts/p2/prompts/single_image/BelugaID.txt")
ERROR_LOG_FILE = OUTPUT_ROOT / "error_log.jsonl"

# 并发与重试配置
MAX_CONCURRENT_REQUESTS = 10
MAX_RETRIES = 3
MODEL_NAME = "gemini-1.5-flash"

# 支持的图片扩展名
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
# =========================================

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_client():
    if not API_KEY:
        raise ValueError("未找到 GEMINI_API_KEY，请检查 .env 文件或环境变量。")
    # 新版 SDK 初始化
    return genai.Client(api_key=API_KEY)

def load_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt 文件未找到: {prompt_path}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

def is_valid_image(file_path: Path) -> bool:
    """检查文件是否为有效的图像文件（非 corrupted）。"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, Exception):
        return False

def robust_json_parser(response_text: str) -> Dict[str, Any]:
    """强力 JSON 提取器 (Phase 3)"""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # 正则回退
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"无法从回复中解析 JSON: {response_text[:100]}...")

async def log_error(image_id: str, error_reason: str):
    """记录错误到死信队列"""
    error_entry = {"image": image_id, "error": error_reason}
    async with aiofiles.open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
        await f.write(json.dumps(error_entry) + "\n")

async def process_single_image(
    sem: asyncio.Semaphore,
    client: genai.Client,
    image_path: Path,
    prompt_text: str,
    relative_path: Path
):
    """处理单张图片的异步工作流 (Phase 4)"""
    output_json_path = OUTPUT_ROOT / relative_path.with_suffix('.json')
    
    # 1. 状态检测 (断点续传)
    if output_json_path.exists() and output_json_path.stat().st_size > 0:
        return "SKIPPED"

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    if not is_valid_image(image_path):
        await log_error(str(relative_path), "Corrupted Image File")
        return "CORRUPTED"

    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                img = Image.open(image_path)
                
                # 新版 SDK 异步调用方式
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=[prompt_text, img],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                
                json_data = robust_json_parser(response.text)
                
                # 原子写入
                temp_path = output_json_path.with_suffix('.tmp')
                async with aiofiles.open(temp_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(json_data, indent=2, ensure_ascii=False))
                
                os.rename(temp_path, output_json_path)
                return "SUCCESS"

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Resource has been exhausted" in error_msg:
                    wait_time = 2 ** (attempt + 1)
                    await asyncio.sleep(wait_time)
                    continue
                
                if attempt == MAX_RETRIES - 1:
                    await log_error(str(relative_path), error_msg)
                    return "FAILED"
                
                await asyncio.sleep(1)

async def main():
    client = setup_client()
    
    if not PROMPT_FILE.exists():
        logger.error(f"请先创建 Prompt 文件: {PROMPT_FILE}")
        return

    prompt_text = load_prompt(PROMPT_FILE)
    
    logger.info(f"正在扫描 {INPUT_ROOT} ...")
    image_files = []
    for root, dirs, files in os.walk(INPUT_ROOT):
        for file in files:
            file_path = Path(root) / file
            if "IDs" not in file_path.parts:
                continue
            if file_path.suffix.lower() in VALID_EXTENSIONS:
                image_files.append(file_path)

    logger.info(f"找到 {len(image_files)} 张待处理图片。")

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []
    for img_path in image_files:
        relative_path = img_path.relative_to(INPUT_ROOT)
        tasks.append(process_single_image(sem, client, img_path, prompt_text, relative_path))

    results = await tqdm.gather(*tasks, desc="Processing Images")
    
    summary = {"SUCCESS": 0, "SKIPPED": 0, "FAILED": 0, "CORRUPTED": 0}
    for res in results:
        if res in summary:
            summary[res] += 1
            
    logger.info(f"任务完成。统计结果: {summary}")

if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    asyncio.run(main())