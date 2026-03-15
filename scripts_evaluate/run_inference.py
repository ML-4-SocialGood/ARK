"""
/home/dzha866/Projects/ARK/scripts_evaluate/run_inference.py
Main inference script for batch processing Re-ID tasks with Ollama.
Supports resume functionality and robust error handling.
"""

import argparse
import json
import logging
import os
import sys
import time
import re

from tqdm import tqdm

# Add project root to path to ensure imports work
sys.path.append(os.getcwd())

from scripts_evaluate.llm_client import OllamaClient
from scripts_evaluate.prompts import PromptGenerator
from scripts_evaluate.utils import ensure_directories, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run Re-ID inference using Ollama.")
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
        default="qwen3.5:4b",
        help="Ollama model name (must support vision)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:11434",
        help="Ollama server host (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results file if present",
    )

    args = parser.parse_args()

    # 1. Setup Directories and Logging
    paths = ensure_directories(args.species, args.protocol)

    # Define output file name based on model to avoid overwriting different model results
    # Replace colons in model name (e.g. qwen:7b) to avoid filesystem issues
    model_safe_name = args.model.replace(":", "_")
    
    # Extract annotation filename (e.g., "BelugaID_I2I_P5_occlusion_S1_N4") to separate results
    anno_basename = os.path.splitext(os.path.basename(args.annotation_file))[0]
    
    # Create a directory for this model's predictions grouped by annotation file
    output_dir = paths["predictions"] / model_safe_name / anno_basename
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = paths["logs"] / f"inference_{model_safe_name}_{anno_basename}.log"

    setup_logging(log_file)
    logging.info("=== Starting Inference ===")
    logging.info(f"Species: {args.species}, Protocol: {args.protocol}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Annotation File: {args.annotation_file}")
    logging.info(f"Output Directory: {output_dir}")

    # 2. Load Annotations
    if not os.path.exists(args.annotation_file):
        logging.error(f"Annotation file not found: {args.annotation_file}")
        return

    with open(args.annotation_file, "r") as f:
        tasks = json.load(f)

    logging.info(f"Loaded {len(tasks)} tasks.")

    # 3. Initialize Client
    # Timeout set to 300s (5 mins) to handle large images/slow generation
    client = OllamaClient(host=args.host, model=args.model, timeout=300)
    if not client.check_connection():
        logging.error(
            "Could not connect to Ollama server. Please check if 'ollama serve' is running."
        )
        return

    prompt_gen = PromptGenerator(species=args.species)

    # 4. Resume Logic
    processed_task_ids = set()
    if args.resume:
        logging.info("Resume flag set. Checking existing results in directory...")
        # List all json files in the output directory
        for file_path in output_dir.glob("*.json"):
            # Assuming filename is {task_id}.json
            processed_task_ids.add(file_path.stem)
        logging.info(f"Found {len(processed_task_ids)} already processed tasks.")

    # 5. Main Inference Loop
    
    # Use tqdm for progress bar
    for task in tqdm(tasks, desc="Inference"):
        task_id = task.get("task_id")

        if task_id in processed_task_ids:
            continue

        # Construct Prompts based on protocol
        if args.protocol.upper() == "P7":
            prompt_n, prompt_c, image_paths = prompt_gen.construct_p7_prompts(task)
            if not prompt_n or not prompt_c:
                logging.warning(f"Skipping task {task_id}: Could not generate P7 prompts.")
                continue
            prompts_to_run = {"neutral": prompt_n, "counterfactual": prompt_c}
        else:
            prompt_text, image_paths = prompt_gen.construct_mcq_prompt(
                task, protocol=args.protocol
            )
            if not prompt_text:
                logging.warning(
                    f"Skipping task {task_id}: Could not generate prompt (missing images?)."
                )
                continue
            prompts_to_run = {"default": prompt_text}

        # Verify images exist locally
        missing_images = [img for img in image_paths if not os.path.exists(img)]
        if missing_images:
            logging.warning(
                f"Skipping task {task_id}: Missing image files: {missing_images}"
            )
            continue

        # Verbose output for monitoring (like test_vl.py)
        logging.info(f"--- Task: {task_id} ---")

        logging.info(f"[Image Paths] ({len(image_paths)} images):")
        for p in image_paths:
            if os.path.exists(p):
                logging.info(f"  [OK] {p}")
            else:
                logging.info(f"  [MISSING] {p}")

        task_results = {}

        for prompt_name, prompt_text in prompts_to_run.items():
            if len(prompts_to_run) > 1:
                logging.info(f"--- Running Sub-Task: [{prompt_name}] ---")
                
            logging.info("[Generated Prompt (Snippet)]:")
            logging.info(prompt_text[:300] + "..." if len(prompt_text) > 300 else prompt_text)

            # Retry variables
            max_retries = 3
            extracted_answer = None
            model_output = ""
            duration = 0
            response = {}

            for attempt in range(max_retries):
                if attempt > 0:
                    logging.info(f"[Retry] Attempt {attempt + 1}/{max_retries}...")

                logging.info("Sending request to Ollama... (This may take 10-30 seconds)")

                try:
                    start_time = time.time()

                    # API Call
                    response = client.generate(prompt=prompt_text, images=image_paths)

                    duration = time.time() - start_time

                    # Extract text response
                    model_output = response.get("response", "")

                    # Extract answer using Regex
                    extracted_answer = None
                    valid_options = []
                    opts_pattern = ""
                    
                    # Extract valid options dynamically for P1-P6
                    if args.protocol.upper() != "P7":
                        valid_options = [str(opt.get("option")).upper() for opt in task.get("gallery", [])]
                        opts_pattern = "|".join([re.escape(opt) for opt in valid_options])
                    
                    if model_output:
                        if args.protocol.upper() == "P7":
                            # Extraction Logic for P7 (Yes/No expected)
                            # Strategy 1: Look for explicit "Answer: Yes/No" or "Conclusion: Yes/No"
                            match = re.search(r'(?:Answer|Conclusion|Verification)\s*[:\-\s]*\s*(Yes|No)\b', model_output, re.IGNORECASE)
                            if match:
                                extracted_answer = match.group(1).upper()
                                
                            # Strategy 2: Look for Yes/No at the very beginning of the response
                            if not extracted_answer:
                                match = re.search(r'^\s*(Yes|No)\b', model_output, re.IGNORECASE)
                                if match:
                                    extracted_answer = match.group(1).upper()
                                    
                            # Strategy 3: Look for Yes/No with punctuation (e.g., "Yes.", "No,") early in the text
                            if not extracted_answer:
                                match = re.search(r'\b(Yes|No)[,\.]', model_output, re.IGNORECASE)
                                if match:
                                    extracted_answer = match.group(1).upper()
                                    
                            # Strategy 4: Fallback to the first occurrence anywhere
                            if not extracted_answer:
                                match = re.search(r'\b(Yes|No)\b', model_output, re.IGNORECASE)
                                if match:
                                    extracted_answer = match.group(1).upper()
                                    
                            # Strategy 5: Semantic fallback if model completely ignored "Yes/No" instruction
                            if not extracted_answer:
                                if re.search(r'\b(different)\b', model_output, re.IGNORECASE):
                                    extracted_answer = "NO"
                                elif re.search(r'\b(same)\b', model_output, re.IGNORECASE):
                                    extracted_answer = "YES"
                        
                        elif args.protocol.upper() == "P3":
                            # Extraction Logic for P3 (Multiple correct options expected)
                            found_opts = []
                            
                            # Strategy 1: Look for explicit multiple answers "Answer: A, C, D" or "Options: A and B"
                            match = re.search(rf'(?:Answer|Option|Choice)s?\s*[:\-\s]*\s*((?:{opts_pattern})(?:\s*(?:,|and|&)\s*(?:{opts_pattern}))*)', model_output, re.IGNORECASE)
                            if match:
                                raw_ans = match.group(1).upper()
                                found_opts = re.findall(rf'({opts_pattern})', raw_ans)
                                
                            # Strategy 2: Look for bracketed combinations like [A, C, D]
                            if not found_opts:
                                match = re.search(r'\[(.*?)\]', model_output)
                                if match:
                                    found_opts = re.findall(rf'({opts_pattern})', match.group(1).upper())
                                    
                            # Strategy 3: Direct match if output is very short (Fallback)
                            if not found_opts and len(model_output.strip()) <= 40:
                                found_opts = re.findall(rf'({opts_pattern})', model_output.upper())
                                
                            if found_opts:
                                # Remove duplicates, sort alphabetically, and format as "A, C, D"
                                extracted_answer = ", ".join(sorted(list(set(found_opts))))
                                
                        elif args.protocol.upper() in ["P1", "P2", "P4", "P5", "P6"]:
                            # Extraction Logic for P1 / P2 / P4 / P5 / P6 (Single correct option)
                            # Strategy 1: Look for explicit "Answer: X", "Option X" pattern anywhere
                            match = re.search(rf'(?:Answer|Option|Choice)\s*[:\-\s]*\s*({opts_pattern})(?!\w)', model_output, re.IGNORECASE)
                            if match:
                                extracted_answer = match.group(1).upper()

                            # Strategy 2: Look for conversational patterns like "is A", "should be B"
                            if not extracted_answer:
                                match = re.search(r'\b(?:is|be|are)\s*:?\s*({opts_pattern})(?!\w)', model_output, re.IGNORECASE)
                                if match:
                                    extracted_answer = match.group(1).upper()

                            # Strategy 3: Look for "Image X", "Candidate X" or "Image A", "Candidate B"
                            if not extracted_answer:
                                # This handles "Image 1", "Candidate A", etc.
                                match = re.search(r'(?:Image|Candidate)\s+({opts_pattern}|\d+)', model_output, re.IGNORECASE)
                                if match:
                                    try:
                                        val = match.group(1).upper()
                                        if val.isdigit():
                                            # Image 1 is Query, Image 2 is Option A (idx 0), Image 3 is Option B (idx 1), etc.
                                            idx = int(val) - 2
                                            if 0 <= idx < len(valid_options):
                                                extracted_answer = valid_options[idx]
                                        elif val in valid_options:
                                            extracted_answer = val
                                    except (ValueError, IndexError):
                                        pass # Ignore if mapping fails

                            # Strategy 4: Look for a valid option character at the very end of the string
                            if not extracted_answer:
                                match = re.search(rf'(?:^|\s)({opts_pattern})[\.\)]?\s*$', model_output.strip(), re.IGNORECASE)
                                if match:
                                    extracted_answer = match.group(1).upper()

                            # Strategy 5: Look for valid option character at the start (e.g. "A. The answer is...")
                            if not extracted_answer:
                                match = re.search(rf'^\s*({opts_pattern})[\.\)]', model_output.strip(), re.IGNORECASE)
                                if match:
                                    extracted_answer = match.group(1).upper()
                                    
                            # Strategy 6: Direct match if output is extremely short (Fallback)
                            if not extracted_answer and model_output.strip().upper() in valid_options:
                                extracted_answer = model_output.strip().upper()

                    # Verbose output for response
                    logging.info("==================== Model Response ====================")
                    if model_output:
                        logging.info(f"\n{model_output}")
                    elif response.get("thinking"):
                        logging.info("[Thinking Process (Response was empty)]:")
                        logging.info(response.get("thinking"))
                    else:
                        logging.info("[No response or thinking field found.]")
                    logging.info("========================================================")

                    logging.info("[Debug Info]:")
                    logging.info(f"  Done: {response.get('done')}")
                    logging.info(f"  Eval Count: {response.get('eval_count')} tokens")
                    total_duration = response.get("total_duration")
                    if total_duration:
                        logging.info(f"  Total Duration: {total_duration / 1e9:.2f}s")
                    logging.info("-" * 60)

                    if extracted_answer:
                        break
                    
                    logging.warning(f"Task {task_id} [{prompt_name}]: Failed to extract answer (Attempt {attempt + 1}/{max_retries}).")

                except Exception as e:
                    logging.error(f"Error processing task {task_id} [{prompt_name}] (Attempt {attempt + 1}/{max_retries}): {e}")
                    continue

            task_results[prompt_name] = {
                "prompt": prompt_text,
                "prediction_text": model_output,
                "extracted_answer": extracted_answer,
                "duration_seconds": duration
            }

        # Save Result
        if args.protocol.upper() == "P7":
            result_entry = {
                "task_id": task_id,
                "ground_truth": task.get("ground_truth"),
                "model": args.model,
                "image_paths": image_paths,
                "neutral": task_results.get("neutral", {}),
                "counterfactual": task_results.get("counterfactual", {})
            }
        else:
            default_res = task_results.get("default", {})
            result_entry = {
                "task_id": task_id,
                "ground_truth": task.get("answer"),
                "extracted_answer": default_res.get("extracted_answer"),
                "model": args.model,
                "prompt": default_res.get("prompt"),
                "image_paths": image_paths,
                "prediction_text": default_res.get("prediction_text"),
                "duration_seconds": default_res.get("duration_seconds"),
            }

        # Save to individual JSON file
        task_file = output_dir / f"{task_id}.json"
        with open(task_file, "w") as f:
            json.dump(result_entry, f, indent=4)

    logging.info("Inference finished.")


if __name__ == "__main__":
    main()
