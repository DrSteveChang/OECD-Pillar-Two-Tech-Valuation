from google import genai
import os

# 使用你之前配置好的环境
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

print("--- 正在列出你当前账号可用的所有模型 ID ---")
for model in client.models.list():
    print(f"模型名称 (name): {model.name}")
    print(f"显示名称 (display_name): {model.display_name}")
    print("-" * 30)