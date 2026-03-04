import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, UnidentifiedImageError
from tqdm.asyncio import tqdm

# =================Configuration=================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

INPUT_ROOT = Path("data/BelugaID")
OUTPUT_ROOT = Path("annotations/BelugaID/p2/single_image_prompts")
PROMPT_FILE = Path("scripts/p2/prompts/single_image/BelugaID.txt")
ERROR_LOG_FILE = OUTPUT_ROOT / "error_log.jsonl"

MAX_CONCURRENT_REQUESTS = 10
MAX_RETRIES = 3
MODEL_NAME = "gemini-2.0-flash"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
# =========================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_client():
    if not API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not found. Please check your .env file or environment variables."
        )
    return genai.Client(api_key=API_KEY)


def load_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def is_valid_image(file_path: Path) -> bool:
    """Check if the file is a valid (non-corrupted) image file."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, Exception):
        return False


def robust_json_parser(response_text: str) -> Dict[str, Any]:
    """Robust JSON extractor (Phase 3)"""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Regex fallback
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Unable to parse JSON from response: {response_text[:100]}...")


async def log_error(image_id: str, error_reason: str):
    """Log error to the dead-letter queue."""
    error_entry = {"image": image_id, "error": error_reason}
    async with aiofiles.open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        await f.write(json.dumps(error_entry) + "\n")
