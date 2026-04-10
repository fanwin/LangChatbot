import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent
from utils.model_factory import GetModelByVendor

# 定义一个工具，用来调用本地部署的mcp 服务
# https://github.com/docling-project/docling-mcp
def create_mcp_client() -> MultiServerMCPClient:
    """
    创建一个本地mcp客户端
    :return:
    """

    server_config = {
        "docling-server": {
            "url": "http://localhost:8888/sse",  # SSE 服务端点
            "transport": "sse",                   # 传输协议
        }
    }

    return MultiServerMCPClient(server_config)

mcp_client = create_mcp_client()
tools = asyncio.run(mcp_client.get_tools())

agent = create_deep_agent(
    model=GetModelByVendor("doubao").generate_model_client(),
    tools=tools,
    system_prompt="你是一个pdf解析专家，请解析pdf文档",
)

# result = agent.invoke({"messages": [{"role": "user", "content": "请解析pdf文档"}]})
# print(result["messages"][-1].content)