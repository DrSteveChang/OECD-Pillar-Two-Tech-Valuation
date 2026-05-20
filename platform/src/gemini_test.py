import os
from google import genai
from google.genai.types import GenerateContentConfig, Tool

def test_gemini_connection():
    # 从环境变量安全读取密钥
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("错误：未检测到 GOOGLE_API_KEY 环境变量")
        return

    client = genai.Client(api_key=api_key)
    model_id = "gemini-3.5-flash"

    # 配置联网搜索工具
    tools = [Tool(google_search_retrieval=None)]

    print(f"正在尝试连接 Gemini ({model_id})...")
    
    try:
        response = client.models.generate_content(
            model=model_id,
            contents="What are the top 3 recent announcements from the Gemini API?",
            config=GenerateContentConfig(tools=tools)
        )
        print("\n--- 连接成功！返回内容如下：---\n")
        print(response.candidates[0].content.parts[0].text)
    except Exception as e:
        print(f"\n连接失败，错误信息：{e}")

if __name__ == "__main__":
    test_gemini_connection()