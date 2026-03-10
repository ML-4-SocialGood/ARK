"""
/home/dzha866/Projects/ARK/scripts_eval/run_inference.py
Main inference script for batch processing Re-ID tasks with Ollama.
Supports resume functionality and robust error handling.
"""

import argparse
import json
import logging
import os
import sys
import time

from tqdm import tqdm

# Add project root to path to ensure imports work
sys.path.append(os.getcwd())

from scripts_eval.llm_client import OllamaClient
from scripts_eval.prompts import PromptGenerator
from scripts_eval.utils import ensure_directories, setup_logging


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
        default="qwen3-vl:8b",
        help="Ollama model name (must support vision)",
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
    
    # Create a directory for this model's predictions
    output_dir = paths["predictions"] / model_safe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = paths["base"] / f"inference_{model_safe_name}.log"

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
    client = OllamaClient(model=args.model, timeout=300)
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

        # Construct Prompt
        prompt_text, image_paths = prompt_gen.construct_mcq_prompt(
            task, protocol=args.protocol
        )

        if not prompt_text:
            logging.warning(
                f"Skipping task {task_id}: Could not generate prompt (missing images?)."
            )
            continue

        # Verify images exist locally
        missing_images = [img for img in image_paths if not os.path.exists(img)]
        if missing_images:
            logging.warning(
                f"Skipping task {task_id}: Missing image files: {missing_images}"
            )
            continue

        # Verbose output for monitoring (like test_vl.py)
        print(f"\n--- Task: {task_id} ---")
        print("[Generated Prompt (Snippet)]:")
        print(prompt_text[:300] + "..." if len(prompt_text) > 300 else prompt_text)

        print(f"\n[Image Paths] ({len(image_paths)} images):")
        for p in image_paths:
            if os.path.exists(p):
                print(f"  [OK] {p}")
            else:
                print(f"  [MISSING] {p}")

        print("\nSending request to Ollama... (This may take 10-30 seconds)")

        try:
            start_time = time.time()

            # API Call
            response = client.generate(prompt=prompt_text, images=image_paths)

            duration = time.time() - start_time

            # Extract text response
            model_output = response.get("response", "")

            # Verbose output for response
            print("\n" + "=" * 20 + " Model Response " + "=" * 20)
            if model_output:
                print(model_output)
            elif response.get("thinking"):
                print("[Thinking Process (Response was empty)]:")
                print(response.get("thinking"))
            else:
                print("[No response or thinking field found.]")
            print("=" * 56)

            print("\n[Debug Info]:")
            print(f"  Done: {response.get('done')}")
            print(f"  Eval Count: {response.get('eval_count')} tokens")
            if response.get("total_duration"):
                print(
                    f"  Total Duration: {response.get('total_duration') / 1e9:.2f}s"
                )
            print("-" * 60)

            # Save Result
            result_entry = {
                "task_id": task_id,
                "ground_truth": task.get("answer"),
                "model": args.model,
                "prompt": prompt_text,
                "image_paths": image_paths,
                "prediction_text": model_output,
                "full_response": response,  # Save full metadata (tokens, time, etc.)
                "duration_seconds": duration,
            }

            # Save to individual JSON file
            task_file = output_dir / f"{task_id}.json"
            with open(task_file, "w") as f:
                json.dump(result_entry, f, indent=4)

        except Exception as e:
            logging.error(f"Error processing task {task_id}: {e}")
            # We continue to the next task instead of crashing
            continue

    logging.info("Inference finished.")


if __name__ == "__main__":
    main()
