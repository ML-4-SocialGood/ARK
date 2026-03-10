"""
/home/dzha866/Projects/ARK/scripts_eval/test_vl.py
Test script to verify if the VLM receives images correctly from the generated prompts.
"""
import json
import os
import sys

# Ensure the project root is in sys.path
sys.path.append(os.getcwd())

from scripts_eval.llm_client import OllamaClient
from scripts_eval.prompts import PromptGenerator

def main():
    # 1. Configuration
    # 注意：必须使用支持视觉的模型 (如 qwen3-vl:8b)，纯文本模型 (qwen3.5:4b) 无法看到图片
    model_name = "qwen3.5:4b" 
    annotation_file = "annotations/BelugaID/p1/BelugaID_I2I_P1_N4.json"
    
    print(f"=== Testing Image Transmission to {model_name} ===")

    # 2. Load Annotations
    if not os.path.exists(annotation_file):
        print(f"Error: Annotation file not found: {annotation_file}")
        print("Please run this script from the project root directory.")
        return

    with open(annotation_file, "r") as f:
        tasks = json.load(f)
        if not tasks:
            print("Error: Annotation file is empty.")
            return
        task = tasks[0] # Test with the first task

    print(f"Loaded Task ID: {task.get('task_id')}")
    print(f"Ground Truth: {task.get('answer')}")

    # 3. Generate Prompt & Images
    generator = PromptGenerator(species="BelugaID")
    prompt_text, image_paths = generator.construct_mcq_prompt(task, protocol="P1")

    print("\n[Generated Prompt (Snippet)]:")
    print(prompt_text[:300] + "..." if len(prompt_text) > 300 else prompt_text)
    
    print(f"\n[Image Paths] ({len(image_paths)} images):")
    valid_images = []
    for p in image_paths:
        if os.path.exists(p):
            print(f"  [OK] {p}")
            valid_images.append(p)
        else:
            print(f"  [MISSING] {p}")
    
    if not valid_images:
        print("Error: No valid images found to send.")
        return

    # 4. Initialize Client
    # 显式指定 model，防止使用默认的纯文本模型
    client = OllamaClient(model=model_name, timeout=300)
    
    # 5. Send Request
    print("\nSending request to Ollama... (This may take 10-30 seconds)")
    
    try:
        # OllamaClient 会自动处理 base64 编码
        response = client.generate(
            prompt=prompt_text,
            images=valid_images
        )
        
        print("\n" + "="*20 + " Model Response " + "="*20)
        
        if response.get("response"):
            print(response.get("response"))
        elif response.get("thinking"):
            print("[Thinking Process (Response was empty)]:")
            print(response.get("thinking"))
        else:
            print("[No response or thinking field found. Full JSON below:]")
            print(json.dumps(response, indent=2))
        print("="*56)
        
        print("\n[Debug Info]:")
        print(f"  Done: {response.get('done')}")
        print(f"  Eval Count: {response.get('eval_count')} tokens")
        if response.get('total_duration'):
            print(f"  Total Duration: {response.get('total_duration') / 1e9:.2f}s")

    except Exception as e:
        print(f"Error calling Ollama: {e}")

if __name__ == "__main__":
    main()
