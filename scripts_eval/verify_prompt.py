"""
/home/dzha866/Projects/ARK/scripts_eval/verify_prompt.py
Script to verify the generated prompts from annotation files.
"""

import json
import os
import sys

# Ensure the project root is in sys.path so we can import from scripts_eval
sys.path.append(os.getcwd())

from scripts_eval.prompts import PromptGenerator


def main():
    # Path to your annotation file
    annotation_file = "annotations/BelugaID/p1/BelugaID_I2I_P1_N4.json"

    if not os.path.exists(annotation_file):
        print(f"Error: Annotation file not found at {annotation_file}")
        print("Please run this script from the project root directory.")
        return

    print(f"Loading annotations from: {annotation_file}")
    with open(annotation_file, "r") as f:
        tasks = json.load(f)

    # Initialize PromptGenerator
    generator = PromptGenerator(species="BelugaID")

    print(f"\nVerifying prompts for {len(tasks)} tasks (showing first 3 samples)...\n")

    for i, task in enumerate(tasks[:3]):
        print(f"--- Sample {i + 1}: Task ID {task.get('task_id', 'Unknown')} ---")

        # Generate the prompt
        prompt_text, image_paths = generator.construct_mcq_prompt(task, protocol="P1")

        print("\n[Generated Prompt Text]:")
        print(prompt_text)

        print("\n[Image Paths Sequence (Query -> Options)]:")
        for idx, img_path in enumerate(image_paths):
            role = "Query" if idx == 0 else f"Option {chr(64 + idx)}"  # A, B, C...
            print(f"  {idx + 1}. [{role}] {img_path}")

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
