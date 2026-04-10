from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from utils.model_factory import GetModelByVendor


class AgentSuites():
    def __init__(self):
        """
        初始化方法
        这是类的构造函数，当创建类的实例时会被自动调用
        在这里可以初始化对象的属性
        """
        pass  # 使用pass语句作为占位符，表示该方法暂时不做任何操作

    def _generate_model_client(self, vendor: str = None):
        """
        根据指定的供应商生成模型客户端
        参数:
            vendor (str, optional): 模型供应商的名称，默认为None
        返回:
            生成的模型客户端对象
        异常:
            当获取模型客户端过程中发生错误时，会抛出包含原始错误信息的异常
        """
        try:
            # 根据供应商获取模型客户端
            model_client = GetModelByVendor(vendor).generate_model_client()
            return model_client
        except Exception as e:
            # 捕获并重新抛出异常，保持原始错误信息
            raise Exception(e)

    def _get_tools(self, city: str = None):
        from tool_factory import MyToolFactory
        tool_factory = MyToolFactory().get_weather_tool

        return tool_factory

    def generate_agent(self, mode_name: str = None):
        """
        生成一个智能代理(Agent)的方法

        该方法尝试使用指定的模型客户端和工具创建一个智能代理，如果创建过程中出现任何异常，将抛出异常。
        :return:  成功时返回创建的智能代理实例，失败时抛出异常
        """
        try:
            # 使用create_agent函数创建智能代理
            # 参数包括模型客户端、工具列表和系统提示
            agent = create_agent(
                model=self._generate_model_client(mode_name),  # 指定使用的模型客户端
                tools=[self._get_tools()],  # 工具列表，当前为空列表
                system_prompt=""  # 系统提示，当前为空字符串
            )

            return agent
        except Exception as e:
            # 捕获所有异常并重新抛出
            raise Exception(e)


@before_model(can_jump_to=["end"])
def check_message_flow(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("中间件的使用，大模型运作前....")
    print("当前对话的完整记录长度（包含HumanMessage和AIMessage）: ", str(len(state["messages"])))
    print("HumanMessage 参与的次数：", str(len([_ for _ in state["messages"] if isinstance(_, HumanMessage)])))
    print("AIMessage 参与的次数：", str(len([_ for _ in state["messages"] if isinstance(_, AIMessage)])))
    print(state["messages"])
    print("用户现在提到了的问题: \n\t", state["messages"][-1].content)
    if len(state["messages"]) > 2:
        print("上一个问题的回答: \n\t", state["messages"][-2].content)

    last_msg = state["messages"][-1]
    if isinstance(last_msg, HumanMessage) and last_msg.content[-1]['type'] == 'image_url':
        print("用户上传了图片，把前端传递的base64编码的图片转换为图片文件，并保存到本地")
    return None


@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"Model returned: {state['messages'][-1].content}")
    return None


def get_weather_tool(city: str):
    """
    获取城市的天气
    :param city:
    :return:
    """
    return city + " 万里无云."


agent = create_agent(
    model=GetModelByVendor('doubao').generate_model_client(),
    middleware=[check_message_flow],
    tools=[get_weather_tool],
    system_prompt=""
)

# response = agent.invoke({"messages": [{"role": "user", "content": "今天上海的天气如何。"}]})


# if __name__ == '__main__':
#     # 测试代码
#     agent = AgentSuites().generate_agent()
#     message = {"messages":[
#         {
#             'role': 'user',
#             'content': '今天上海的天气如何。'
#         },
#         {
#             'role': 'assistant',
#             'content': '你是一个很智能的人工智能助手，可以帮助做许多高复杂性的任务？'
#         }
#     ]
#     }
#     ret = agent.invoke(message)
#     for msg in ret['messages']:
#         print(msg.content)
