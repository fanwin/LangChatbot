import os
from typing import Literal

from dotenv import load_dotenv

from deepagents import create_deep_agent
from tavily import TavilyClient

from utils.model_factory import GetModelByVendor

load_dotenv()

def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False):
    """Run a web search"""

    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

def get_weather_tool(city:str):
    """
    获取城市的天气情况
    :param city:
    :return:
    """
    return city + " 万里无云."


research_instruction = """你是一位专家研究员。你的工作是进行深入的研究，然后撰写一份完善的报告。
            你可以使用互联网搜索工具作为获取信息的主要手段。
            ## `internet_search`
            使用这个工具对给定的查询进行互联网搜索。你可以指定返回结果的最大数量、主题，以及是否包含原始内容。"""

deep_agent =create_deep_agent(
    model=GetModelByVendor("doubao").generate_model_client(),
    tools=[get_weather_tool, internet_search],
    system_prompt="你是一名多角色的人工智能助手，请根据用户的具体需求或指令出色完成任务。\n" + research_instruction
)

# result = deep_agent.invoke({"messages": [{"role": "user", "content": "今天上海的天气如何。"}]})
# print(result["messages"][-1].content)
