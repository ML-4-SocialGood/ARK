import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict

import aiofiles
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, UnidentifiedImageError
from tqdm.asyncio import tqdm

# =================Configuration=================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

MAX_CONCURRENT_REQUESTS = 10
MAX_RETRIES = 3
# MODEL_NAME = "gemini-2.0-flash"
# MODEL_NAME = "gemini-2.5-flash-lite"
MODEL_NAME = "gemini-3-flash-preview"

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


async def log_error(image_id: str, error_reason: str, error_log_file: Path):
    """Log error to the dead-letter queue."""
    error_entry = {"image": image_id, "error": error_reason}
    async with aiofiles.open(error_log_file, "a", encoding="utf-8") as f:
        await f.write(json.dumps(error_entry) + "\n")


async def process_single_image(
    sem: asyncio.Semaphore,
    client: genai.Client,
    image_path: Path,
    prompt_text: str,
    relative_path: Path,
    output_root: Path,
    error_log_file: Path,
) -> str:
    """
    Asynchronous workflow for processing a single image (Phase 4)
    """
    output_json_path = output_root / relative_path.with_suffix(".json")

    # 1. State check (Resume from breakpoint)
    if output_json_path.exists() and output_json_path.stat().st_size > 0:
        return "SKIPPED"

    # 2. Ensure output directory exists
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Image integrity check
    if not is_valid_image(image_path):
        await log_error(str(relative_path), "Corrupted Image File", error_log_file)
        return "CORRUPTED"

    async with sem:  # Limit concurrency
        for attempt in range(MAX_RETRIES):
            try:
                # Prepare image data with context manager to ensure file is closed
                with Image.open(image_path) as img:
                    # Send request (Gemini 2.0 Flash)
                    response = await client.aio.models.generate_content(
                        model=MODEL_NAME,
                        contents=[prompt_text, img],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json"
                        ),
                    )

                if response.text is None:
                    raise ValueError(
                        "Model response is None (possibly blocked by safety filters)."
                    )

                # Parse result
                json_data = robust_json_parser(response.text)

                # Atomic write
                temp_path = output_json_path.with_suffix(".tmp")
                async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(json_data, indent=2, ensure_ascii=False))

                os.rename(temp_path, output_json_path)
                return "SUCCESS"

            except Exception as e:
                error_msg = str(e)
                # Handle 429 Too Many Requests
                if "429" in error_msg or "Resource has been exhausted" in error_msg:
                    wait_time = 2 ** (attempt + 1)  # Exponential backoff: 2s, 4s, 8s...
                    await asyncio.sleep(wait_time)
                    continue

                # For other errors, log if it's the last retry
                if attempt == MAX_RETRIES - 1:
                    await log_error(
                        str(relative_path),
                        f"Failed after {MAX_RETRIES} retries: {error_msg}",
                        error_log_file,
                    )
                    return "FAILED"

                await asyncio.sleep(1)  # Default wait

    return "FAILED"


async def main():
    parser = argparse.ArgumentParser(
        description="Extract features from images using Gemini."
    )
    parser.add_argument(
        "--species",
        type=str,
        required=True,
        help="Species name (e.g., BelugaID) to determine input/output paths.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of images to process (useful for testing).",
    )
    args = parser.parse_args()

    species_name = args.species
    input_root = Path(f"data/{species_name}")
    output_root = Path(f"annotations/{species_name}/p2/single_image_prompts")
    prompt_file = Path(f"scripts/p2/prompts/single_image/{species_name}.txt")
    error_log_file = output_root / "error_log.jsonl"

    try:
        client = setup_client()
    except ValueError as e:
        logger.error(e)
        return

    if not prompt_file.exists():
        logger.error(f"Prompt file not found: {prompt_file}")
        return

    prompt_text = load_prompt(prompt_file)

    logger.info(f"Scanning {input_root}...")

    if not input_root.exists():
        logger.error(f"Input directory not found: {input_root}")
        return

    # Scan for tasks
    image_files = []
    for root, _, files in os.walk(input_root):
        for file in files:
            file_path = Path(root) / file

            # Filter 1: Only process contents under 'IDs' directories
            if "IDs" not in file_path.parts:
                continue

            # Filter 2: Extension check
            if file_path.suffix.lower() in VALID_EXTENSIONS:
                image_files.append(file_path)

    logger.info(f"Found {len(image_files)} images pending processing.")

    if args.limit is not None:
        image_files = image_files[: args.limit]
        logger.info(
            f"Limit applied: Processing only the first {len(image_files)} images."
        )

    if not image_files:
        return

    # Create a list of asynchronous tasks
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []
    for img_path in image_files:
        relative_path = img_path.relative_to(input_root)
        task = process_single_image(
            sem,
            client,
            img_path,
            prompt_text,
            relative_path,
            output_root,
            error_log_file,
        )
        tasks.append(task)

    # Execute tasks and display progress bar
    results = await tqdm.gather(*tasks, desc="Processing Images")

    # Summarize results
    summary = {"SUCCESS": 0, "SKIPPED": 0, "FAILED": 0, "CORRUPTED": 0}
    for res in results:
        if res in summary:
            summary[res] += 1

    logger.info(f"Tasks completed. Summary: {summary}")
    if summary["FAILED"] > 0 or summary["CORRUPTED"] > 0:
        logger.info(f"Error logs saved to: {error_log_file}")


if __name__ == "__main__":
    asyncio.run(main())
