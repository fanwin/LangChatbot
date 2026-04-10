"""
Hatch Agent Core - 多模态附件处理核心模块

提供非多模态大模型处理图片/PDF 文档的完整能力，
包括：Vision 模型分析、PDF 文档解析、LRU 缓存、消息转换、Agent 中间件。

模块结构：
  - cache:             LRU 缓存（避免同一文件重复处理）
  - file_utils:       Base64 解码与本地文件保存
  - image_analyzer:   Vision 模型图片分析（豆包/OpenAI）
  - pdf_analyzer:      PDF 文档解析（PyMuPDF4LLM + LLMImageBlobParser）
  - message_transformer: 多模态消息→纯文本转换
  - middleware:        Agent 中间件（before_model / after_model）
  - hatch_agent:       Agent 实例创建入口
"""

from src.core.hatch_agent import agent

__all__ = ["agent"]
