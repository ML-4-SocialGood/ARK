"""
/home/dzha866/Projects/ARK/scripts_evaluate/proprietary.py
Main inference script for batch processing Re-ID tasks with proprietary API models (e.g., GPT-4o).
Restricted to evaluate ONLY the first 50 annotations per file to control budget.
"""

import argparse
import json
import logging
import os
import sys
import time
import re
import base64
import io

from PIL import Image
from tqdm import tqdm
from openai import OpenAI

# 尝试加载 .env 文件中的环境变量
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Add project root to path to ensure imports work
sys.path.append(os.getcwd())

from scripts_evaluate.prompts import PromptGenerator
from scripts_evaluate.utils import ensure_directories, setup_logging


def encode_image_for_openai(image_path: str, crop_watermarks: bool = False) -> str:
    """
    Reads an image, resizes it to a maximum of 1024x1024 to save tokens,
    converts to JPEG, and returns the base64 data URI expected by OpenAI.
    """
    with Image.open(image_path) as img:
        if crop_watermarks:
            width, height = img.size
            # Crop top 10% and bottom 10% to remove camera trap watermarks/timestamps
            img = img.crop((0, int(height * 0.10), width, int(height * 0.90)))
            
        img.thumbnail((1024, 1024))
        if img.mode != "RGB":
            img = img.convert("RGB")

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_string}"


def main():
    parser = argparse.ArgumentParser(
        description="Run Re-ID inference using OpenAI API (Limited to 100 per run)."
    )
    parser.add_argument(
        "--species", type=str, required=True, help="Species name (e.g., BelugaID)"
    )
    parser.add_argument(
        "--protocol", type=str, required=True, help="Protocol (e.g., P1)"
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        required=True,
        help="Path to annotation JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4",
        help="OpenAI model name (e.g., gpt-5.4-pro, gpt-4o)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Limit the number of tasks to evaluate to control budget (default: 50, use 0 for no limit).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run without hitting the actual API (useful for testing script logic).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results file if present",
    )
    parser.add_argument(
        "--crop_watermarks",
        action="store_true",
        help="Crop the top 10% and bottom 10% of images to remove camera trap watermarks/timestamps.",
    )

    args = parser.parse_args()

    # Ensure API Key is set
    api_key = os.environ.get(
        "OPENAI_API_KEY", "dummy_key_for_dry_run" if args.dry_run else None
    )
    if not api_key and not args.dry_run:
        logging.error(
            "OPENAI_API_KEY environment variable not set. Please export it or use --dry_run."
        )
        return

    # >>> 针对 P3 协议的特殊约束：强制只允许跑 N4_M2.json 结尾的文件 <<<
    if args.protocol.upper() == "P3" and not args.annotation_file.endswith("N4_M2.json"):
        logging.error(f"Protocol P3 strictly requires annotation files ending with 'N4_M2.json'. Provided: {args.annotation_file}")
        sys.exit(1)

    # >>> 针对 P6 协议的特殊约束：强制只允许跑 N4.json 结尾的文件 <<<
    if args.protocol.upper() == "P6" and not args.annotation_file.endswith("N4.json"):
        logging.error(f"Protocol P6 strictly requires annotation files ending with 'N4.json'. Provided: {args.annotation_file}")
        sys.exit(1)

    # 1. Setup Directories and Logging
    paths = ensure_directories(args.species, args.protocol)
    model_safe_name = args.model.replace(":", "_")
    anno_basename = os.path.splitext(os.path.basename(args.annotation_file))[0]

    output_dir = paths["predictions"] / model_safe_name / anno_basename
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = (
        paths["logs"] / f"inference_proprietary_{model_safe_name}_{anno_basename}.log"
    )

    setup_logging(log_file)
    logging.info("=== Starting Proprietary Inference (BUDGET CONSTRAINED) ===")
    if args.dry_run:
        logging.info("*** DRY RUN MODE ENABLED: No real API requests will be made ***")
    logging.info(f"Species: {args.species}, Protocol: {args.protocol}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Output Directory: {output_dir}")

    # Initialize OpenAI Client
    client = OpenAI(api_key=api_key)
    prompt_gen = PromptGenerator(species=args.species)

    # 2. Load Annotations and STRICTLY LIMIT to 100
    if not os.path.exists(args.annotation_file):
        logging.error(f"Annotation file not found: {args.annotation_file}")
        return

    with open(args.annotation_file, "r") as f:
        tasks = json.load(f)

    if args.limit > 0:
        # >>> BUDGET CONTROL ENFORCEMENT <<<
        original_len = len(tasks)
        tasks = tasks[:args.limit]
        logging.info(
            f"Loaded {original_len} tasks. Restricted to first {len(tasks)} tasks to control budget."
        )
    else:
        logging.info(f"Loaded {len(tasks)} tasks. No budget limit applied.")

    # 3. Resume Logic
    processed_task_ids = set()
    if args.resume:
        logging.info("Resume flag set. Checking existing results in directory...")
        for file_path in output_dir.glob("*.json"):
            processed_task_ids.add(file_path.stem)
        logging.info(f"Found {len(processed_task_ids)} already processed tasks.")

    # 4. Main Inference Loop
    for task in tqdm(tasks, desc="Inference"):
        task_id = task.get("task_id")

        if task_id in processed_task_ids:
            continue

        # Construct Prompts based on protocol
        if args.protocol.upper() == "P7":
            prompt_n, prompt_c, image_paths = prompt_gen.construct_p7_prompts(task)
            if not prompt_n or not prompt_c:
                continue
            prompts_to_run = {"neutral": prompt_n, "counterfactual": prompt_c}
        else:
            prompt_text, image_paths = prompt_gen.construct_mcq_prompt(
                task, protocol=args.protocol
            )
            if not prompt_text:
                continue
            prompts_to_run = {"default": prompt_text}

        # Verify images exist locally
        missing_images = [img for img in image_paths if not os.path.exists(img)]
        if missing_images:
            logging.warning(
                f"Skipping task {task_id}: Missing image files: {missing_images}"
            )
            continue

        logging.info(f"--- Task: {task_id} ---")
        task_results = {}

        for prompt_name, current_prompt_text in prompts_to_run.items():
            if len(prompts_to_run) > 1:
                logging.info(f"--- Running Sub-Task: [{prompt_name}] ---")

            logging.info("[Generated Prompt (Snippet)]:")
            logging.info(
                current_prompt_text[:300] + "..."
                if len(current_prompt_text) > 300
                else current_prompt_text
            )

            # Build OpenAI messages format by interleaving text and images based on <image> placeholders
            messages_content = []
            parts = current_prompt_text.split("<image>")
            img_idx = 0

            for i, part in enumerate(parts):
                if part:
                    messages_content.append({"type": "text", "text": part})
                
                # 如果不是最后一部分，说明原文本这里有一个 <image> 占位符
                if i < len(parts) - 1:
                    if img_idx < len(image_paths):
                        img_path = image_paths[img_idx]
                        try:
                            b64_url = encode_image_for_openai(img_path, crop_watermarks=args.crop_watermarks)
                            messages_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": b64_url, "detail": "high"},
                                }
                            )
                        except Exception as e:
                            logging.error(f"Failed to encode image {img_path}: {e}")
                        img_idx += 1
                    else:
                        logging.warning(f"Not enough images ({len(image_paths)}) for placeholders in prompt.")

            # 容错处理：拼接剩余未能对齐占位符的图片（以防某些 prompt 格式异常）
            while img_idx < len(image_paths):
                img_path = image_paths[img_idx]
                try:
                    b64_url = encode_image_for_openai(img_path, crop_watermarks=args.crop_watermarks)
                    messages_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": b64_url, "detail": "high"},
                        }
                    )
                except Exception as e:
                    logging.error(f"Failed to encode image {img_path}: {e}")
                img_idx += 1

            max_retries = 3
            extracted_answer = None
            model_output = ""
            duration = 0

            for attempt in range(max_retries):
                logging.info("Sending request to OpenAI...")
                try:
                    start_time = time.time()

                    if args.dry_run:
                        # 模拟网络延迟并返回虚拟响应（故意包含 Answer 和 Conclusion 以测试正则提取）
                        time.sleep(0.5)
                        duration = time.time() - start_time
                        model_output = "DRY RUN MOCK RESPONSE. Based on the analysis, Option A is correct. Conclusion: Yes."
                        mock_tokens = 123
                    else:
                        api_params = {
                            "model": args.model,
                            "messages": [{"role": "user", "content": messages_content}],
                            "temperature": 0.0,
                            "seed": 42,
                        }
                        
                        if "gpt-5" in args.model.lower() or "o1" in args.model.lower() or "o3" in args.model.lower():
                            api_params["max_completion_tokens"] = 2048
                        else:
                            api_params["max_tokens"] = 2048

                        response = client.chat.completions.create(**api_params)

                        duration = time.time() - start_time
                        model_output = response.choices[0].message.content
                        mock_tokens = (
                            response.usage.total_tokens if response.usage else "N/A"
                        )

                    # EXACT SAME EXTRACTION LOGIC AS run_inference.py
                    extracted_answer = None
                    valid_options = []
                    opts_pattern = ""

                    if args.protocol.upper() != "P7":
                        valid_options = [
                            str(opt.get("option")).upper()
                            for opt in task.get("gallery", [])
                        ]
                        opts_pattern = "|".join(
                            [re.escape(opt) for opt in valid_options]
                        )

                    if model_output:
                        if args.protocol.upper() == "P7":
                            match = re.search(
                                r"(?:Answer|Conclusion|Verification)\s*[:\-\s]*\s*(Yes|No)\b",
                                model_output,
                                re.IGNORECASE,
                            )
                            if match:
                                extracted_answer = match.group(1).upper()
                            if not extracted_answer:
                                match = re.search(
                                    r"^\s*(Yes|No)\b", model_output, re.IGNORECASE
                                )
                                if match:
                                    extracted_answer = match.group(1).upper()
                            if not extracted_answer:
                                match = re.search(
                                    r"\b(Yes|No)[,\.]", model_output, re.IGNORECASE
                                )
                                if match:
                                    extracted_answer = match.group(1).upper()
                            if not extracted_answer:
                                match = re.search(
                                    r"\b(Yes|No)\b", model_output, re.IGNORECASE
                                )
                                if match:
                                    extracted_answer = match.group(1).upper()
                            if not extracted_answer:
                                if re.search(
                                    r"\b(different)\b", model_output, re.IGNORECASE
                                ):
                                    extracted_answer = "NO"
                                elif re.search(
                                    r"\b(same)\b", model_output, re.IGNORECASE
                                ):
                                    extracted_answer = "YES"

                        elif args.protocol.upper() == "P3":
                            found_opts = []
                            match = re.search(
                                rf"(?:Answer|Option|Choice)s?\s*[:\-\s]*\s*((?:{opts_pattern})(?:\s*(?:,|and|&)\s*(?:{opts_pattern}))*)",
                                model_output,
                                re.IGNORECASE,
                            )
                            if match:
                                raw_ans = match.group(1).upper()
                                found_opts = re.findall(rf"({opts_pattern})", raw_ans)
                            if not found_opts:
                                match = re.search(r"\[(.*?)\]", model_output)
                                if match:
                                    found_opts = re.findall(
                                        rf"({opts_pattern})", match.group(1).upper()
                                    )
                            if not found_opts and len(model_output.strip()) <= 40:
                                found_opts = re.findall(
                                    rf"({opts_pattern})", model_output.upper()
                                )

                            if found_opts:
                                extracted_answer = ", ".join(
                                    sorted(list(set(found_opts)))
                                )

                        else:
                            match = re.search(
                                rf"(?:Answer|Option|Choice)\s*[:\-\s]*\s*({opts_pattern})(?!\w)",
                                model_output,
                                re.IGNORECASE,
                            )
                            if match:
                                extracted_answer = match.group(1).upper()
                            if not extracted_answer:
                                match = re.search(
                                    rf"\b(?:is|be|are)\s*:?\s*({opts_pattern})(?!\w)",
                                    model_output,
                                    re.IGNORECASE,
                                )
                                if match:
                                    extracted_answer = match.group(1).upper()
                            if not extracted_answer:
                                match = re.search(
                                    rf"(?:Image|Candidate)\s+({opts_pattern}|\d+)",
                                    model_output,
                                    re.IGNORECASE,
                                )
                                if match:
                                    try:
                                        val = match.group(1).upper()
                                        if val.isdigit():
                                            idx = int(val) - 2
                                            if 0 <= idx < len(valid_options):
                                                extracted_answer = valid_options[idx]
                                        elif val in valid_options:
                                            extracted_answer = val
                                    except (ValueError, IndexError):
                                        pass
                            if not extracted_answer:
                                match = re.search(
                                    rf"(?:^|\s)({opts_pattern})[\.\)]?\s*$",
                                    model_output.strip(),
                                    re.IGNORECASE,
                                )
                                if match:
                                    extracted_answer = match.group(1).upper()
                            if not extracted_answer:
                                match = re.search(
                                    rf"^\s*({opts_pattern})[\.\)]",
                                    model_output.strip(),
                                    re.IGNORECASE,
                                )
                                if match:
                                    extracted_answer = match.group(1).upper()
                            if (
                                not extracted_answer
                                and model_output.strip().upper() in valid_options
                            ):
                                extracted_answer = model_output.strip().upper()

                    logging.info(
                        "==================== Model Response ===================="
                    )
                    logging.info(f"\n{model_output}")
                    logging.info(
                        "========================================================"
                    )
                    logging.info(
                        f"[Debug Info] Duration: {duration:.2f}s | Tokens used: {mock_tokens}"
                    )
                    logging.info("-" * 60)

                    if extracted_answer:
                        break

                    logging.warning(
                        f"Task {task_id} [{prompt_name}]: Failed to extract answer (Attempt {attempt + 1}/{max_retries})."
                    )

                except Exception as e:
                    logging.error(
                        f"OpenAI API Error processing task {task_id} (Attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(2**attempt)  # Exponential backoff for API limits
                    continue

            task_results[prompt_name] = {
                "prompt": current_prompt_text,
                "prediction_text": model_output,
                "extracted_answer": extracted_answer,
                "duration_seconds": duration,
            }

        # Save Result exactly like local models
        if args.protocol.upper() == "P7":
            result_entry = {
                "task_id": task_id,
                "ground_truth": task.get("ground_truth"),
                "model": args.model,
                "image_paths": image_paths,
                "neutral": task_results.get("neutral", {}),
                "counterfactual": task_results.get("counterfactual", {}),
            }
        else:
            default_res = task_results.get("default", {})
            result_entry = {
                "task_id": task_id,
                "ground_truth": task.get("ground_truth", task.get("answer")),
                "extracted_answer": default_res.get("extracted_answer"),
                "model": args.model,
                "prompt": default_res.get("prompt"),
                "image_paths": image_paths,
                "prediction_text": default_res.get("prediction_text"),
                "duration_seconds": default_res.get("duration_seconds"),
            }

        task_file = output_dir / f"{task_id}.json"
        with open(task_file, "w") as f:
            json.dump(result_entry, f, indent=4)

        # 增加2秒的间隔，避免调用API频率过高被封禁
        time.sleep(2)

    logging.info("Proprietary Inference finished.")


if __name__ == "__main__":
    main()
