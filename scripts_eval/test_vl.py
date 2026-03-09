from llm_client import OllamaClient

client = OllamaClient(model="qwen3.5:4b")
image_path = "scripts_eval/0_-1_0_test0596_6498.jpg"  # 替换为你的实际图片路径

print(f"Testing with image: {image_path}")
try:
    response = client.generate(
        prompt="Describe this image in detail.",
        images=[image_path]
    )
    print("Response:", response.get("response"))
except Exception as e:
    print(f"Error: {e}")
