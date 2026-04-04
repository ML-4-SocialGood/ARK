import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def list_available_gpt_models():
    try:
        models = client.models.list()
        # 筛选并打印所有 GPT 系列模型
        gpt_models = [m.id for m in models.data if "gpt" in m.id]
        
        print("当前 API Key 可用的 GPT 模型列表:")
        for model_id in sorted(gpt_models):
            print(f"- {model_id}")
            
    except Exception as e:
        print(f"查询失败，请检查 API Key 或网络连通性: {e}")

if __name__ == "__main__":
    list_available_gpt_models()