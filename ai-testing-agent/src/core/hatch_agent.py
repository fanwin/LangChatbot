"""
Hatch Agent - 入口模块（精简版）

这是 Hatch Agent 的唯一入口文件，职责仅限于：
  1. 导出 Agent 实例
  2. 提供工具函数注册

所有核心逻辑已拆分到子模块：
  examples.cache              → LRU 缓存（避免重复处理）
  examples.file_utils          → Base64 解码 / 文件保存
  examples.image_analyzer      → Vision 模型图片分析
  examples.pdf_analyzer        → PDF 文档解析
  examples.message_transformer → 多模态消息→纯文本转换
  examples.middleware           → before_model / after_model 中间件

使用方式：
    from examples.hatch_agent import agent

    response = agent.invoke({
        "messages": [{"role": "user", "content": "你好"}]
    })
"""

from langchain.agents import create_agent

from utils.model_factory import GetModelByVendor

from src.core.middleware import check_message_flow, log_response


# ============================================================
# 工具定义
# ============================================================

def get_weather_tool(city: str) -> str:
    """
    获取城市天气（示例工具）。

    Args:
        city: 城市名称

    Returns:
        天气描述字符串
    """
    return f"{city} 万里无云."


# ============================================================
# Agent 实例
# ============================================================

agent = create_agent(
    model=GetModelByVendor().generate_model_client(),
    middleware=[check_message_flow, log_response],
    tools=[get_weather_tool],
    system_prompt="你是一名智能助手，请根据用户输入的指令，给出专业、准确、有帮助的回复。你可以帮助用户完成各种任务，包括但不限于：回答问题、创作文本（如诗歌、文章等）、分析内容、提供建议等。请始终用中文回复。",
)
