"""
模型工厂模块
统一管理各厂商大模型客户端的创建，支持：
  - deepseek : 纯文本推理模型（text_agent 使用）
  - doubao   : 豆包多模态模型（image_agent / pdf_agent 使用）
  - qwen     : 通义千问
  - openai   : OpenAI GPT 系列
  - kimi     : Moonshot Kimi
  - google   : Google Gemini
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


class GetModelByVendor:
    """根据厂商名称创建对应的 LangChain 模型客户端。"""

    def __init__(self, vendor: str = "deepseek"):
        self.vendor = vendor

    def generate_model_client(self):
        """
        工厂方法：返回对应厂商的 LangChain Chat 模型实例。

        Returns:
            LangChain BaseChatModel 子类实例
        Raises:
            ValueError: 当 vendor 未知时抛出
        """
        if self.vendor == "deepseek" or self.vendor is None or self.vendor == "":
            from langchain_deepseek import ChatDeepSeek
            return ChatDeepSeek(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                model=os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat"),
            )

        elif self.vendor == "doubao":
            # 豆包多模态模型，支持图片/视频输入
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                api_key=os.getenv("DOUBAO_API_KEY"),
                model=os.getenv("DOUBAO_MODEL_NAME", "doubao-seed-2-0-lite-260215"),
                base_url=os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
            )

        elif self.vendor == "qwen":
            from langchain_community.chat_models import ChatTongyi
            return ChatTongyi(
                api_key=os.getenv("QWEN_API_KEY"),
                model=os.getenv("QWEN_MODEL_NAME", "qwen-turbo"),
            )

        elif self.vendor == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o-mini",
            )

        elif self.vendor == "kimi":
            from langchain_community.chat_models import MoonshotChat
            return MoonshotChat(
                api_key=os.getenv("MOONSHORT_API_KEY"),
                base_url=os.getenv("MOONSHORT_BASE_URL", "https://api.moonshot.cn/v1"),
                model=os.getenv("MOONSHORT_MODEL_NAME", "kimi-k2.5"),
                temperature=1,
            )

        elif self.vendor == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                api_key=os.getenv("GOOGLE_API_KEY"),
                model=os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash"),
            )

        elif self.vendor == "xiaomi":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                api_key=os.getenv("XIAOMI_API_KEY"),
                model=os.getenv("XIAOMI_MODEL_NAME", "mimo-v2-pro"),
                base_url=os.getenv("XIAOMI_BASE_URL", "https://api.xiaomimimo.com/v1/"),
            )

        elif self.vendor == "local":
            # from langchain_community.chat_models import ChatOllama
            # return ChatOllama(
            #     model=os.getenv("LOCAL_MODEL_NAME", "qwen3.5:0.8b"),
            #     base_url=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:11434"),
            # )
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=os.getenv("LOCAL_MODEL_NAME", "qwen3.5:0.8b"),
                base_url=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:11434"),
            )
        elif self.vendor == "zhipu":
            # from zai import ZhipuAiClient
            # return ZhipuAiClient(
            #     api_key=os.getenv("ZHIPU_API_KEY"),
            #     model=os.getenv("ZHIPU_MODEL_NAME", "glm-4.7"),
            # )
            from langchain_community.chat_models import ChatZhipuAI
            return ChatZhipuAI(
                api_key=os.getenv("ZHIPU_API_KEY"),
                model=os.getenv("ZHIPU_MODEL_NAME", "glm-4.7"),
            )
        else:
            raise ValueError(
                f"未知的模型厂商: '{self.vendor}'。"
                f"支持: deepseek, doubao, qwen, openai, kimi, google 以及本地部署的大模型"
            )


if __name__ == "__main__":
    # client = GetModelByVendor("local").generate_model_client()
    # client = GetModelByVendor().generate_model_client()
    # client = GetModelByVendor("doubao").generate_model_client()
    client = GetModelByVendor("xiaomi").generate_model_client()
    for token in client.stream("你是谁？你的上下文长度是多少？你擅长做什么？"):
        print(token.content, end="", flush=True)
    print()