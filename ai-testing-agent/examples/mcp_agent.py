import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from model_factory import GetModelByVendor

def get_mcp_rag_tools():
    mcp_client = MultiServerMCPClient(
        {
            "mcp_service_rag": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        }
    )
    tools = asyncio.run(mcp_client.get_tools())
    return tools

mcp_rag_agent = create_agent(
    model=GetModelByVendor().generate_model_client(),
    tools=get_mcp_rag_tools(),
    system_prompt="""你是一名MCP服务器智能Agent，请正确感知用户的问题，合理决策使用的工具。""",
)