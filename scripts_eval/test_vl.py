from llm_client import OllamaClient
import json

client = OllamaClient(model="qwen3.5:4b", timeout=300)
image_path = "scripts_eval/0_-1_0_test0596_6498.jpg"  # 替换为你的实际图片路径

print(f"Testing with image: {image_path}")
try:
    response = client.generate(
        prompt="Describe this image in detail.", images=[image_path]
    )

    content = response.get("response")
    if content:
        print("Response:", content)
    else:
        print("Warning: Received empty response text.")
        print("Full API Response:", json.dumps(response, indent=2))
except Exception as e:
    print(f"Error: {e}")
