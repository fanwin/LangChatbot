import os
from dotenv import load_dotenv
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent
from utils.model_factory import GetModelByVendor

load_dotenv()
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

text_model_client = GetModelByVendor().generate_model_client()
vision_model_client = GetModelByVendor("doubao").generate_model_client()

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
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

research_subagent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions",
    "system_prompt": "You are a great researcher",
    "tools": [get_weather_tool, internet_search],
    "model": text_model_client,  # Optional override, defaults to main agent model
}
subagents = [research_subagent]

sub_agent = create_deep_agent(
    model=text_model_client,
    subagents=subagents,
    # tools=[get_weather_tool]
)