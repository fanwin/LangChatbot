from langchain.tools import tool

class MyToolFactory:

    @tool
    def get_weather_tool(self, city:str):
        """
        获取天气d工具
        :param city:
        :return:
        """
        return city + " 万里无云."