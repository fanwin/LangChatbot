import base64
import tempfile
import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyMuPDF4LLMLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_openai import ChatOpenAI


class Base64PDFProcessor:
    """
    PDF文档处理器，用于处理base64编码的PDF数据
    """
    
    def __init__(self, openai_api_key: str = None, model: str = "gpt-4o-mini", max_tokens: int = 1024):
        """
        初始化处理器
        
        Args:
            openai_api_key: OpenAI API密钥，如果为None则从环境变量读取
            model: 使用的模型名称
            max_tokens: 最大token数
        """
        self.model = model
        self.max_tokens = max_tokens
        self.openai_api_key = openai_api_key
        
    def process(self, base64_data: str) -> List[Dict[str, Any]]:
        """
        处理base64编码的PDF数据
        
        Args:
            base64_data: base64编码的PDF数据字符串
            
        Returns:
            包含PDF完整内容的文档列表，每个文档包含page_content和metadata
        """
        # 创建临时文件保存PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            try:
                # 解码base64数据并写入临时文件
                pdf_bytes = base64.b64decode(base64_data)
                temp_file.write(pdf_bytes)
                temp_file.flush()
                temp_path = temp_file.name
                
                # 使用PyMuPDF4LLMLoader加载PDF
                loader = PyMuPDF4LLMLoader(
                    temp_path,
                    mode="page",
                    extract_images=True,
                    images_parser=LLMImageBlobParser(
                        model=ChatOpenAI(
                            model=self.model,
                            max_tokens=self.max_tokens,
                            api_key=self.openai_api_key
                        )
                    ),
                )
                
                # 加载文档
                docs = loader.load()
                
                # 转换为字典格式
                result = []
                for doc in docs:
                    result.append({
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    })
                
                return result
                
            finally:
                # 清理临时文件
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    def get_full_content(self, base64_data: str) -> str:
        """
        获取PDF的完整文本内容（所有页面合并）
        
        Args:
            base64_data: base64编码的PDF数据字符串
            
        Returns:
            PDF的完整文本内容
        """
        docs = self.process(base64_data)
        return "\n\n".join([doc["page_content"] for doc in docs])
    
    def get_page_content(self, base64_data: str, page_number: int) -> str:
        """
        获取指定页面的内容
        
        Args:
            base64_data: base64编码的PDF数据字符串
            page_number: 页码（从1开始）
            
        Returns:
            指定页面的文本内容
        """
        docs = self.process(base64_data)
        if 0 < page_number <= len(docs):
            return docs[page_number - 1]["page_content"]
        else:
            raise ValueError(f"页码 {page_number} 超出范围，PDF共有 {len(docs)} 页")


# 便捷函数
def process_base64_pdf(base64_data: str, openai_api_key: str = None) -> List[Dict[str, Any]]:
    """
    处理base64编码的PDF数据的便捷函数
    
    Args:
        base64_data: base64编码的PDF数据字符串
        openai_api_key: OpenAI API密钥
        
    Returns:
        包含PDF完整内容的文档列表
    """
    processor = Base64PDFProcessor(openai_api_key=openai_api_key)
    return processor.process(base64_data)


def get_pdf_full_content(base64_data: str, openai_api_key: str = None) -> str:
    """
    获取PDF完整内容的便捷函数
    
    Args:
        base64_data: base64编码的PDF数据字符串
        openai_api_key: OpenAI API密钥
        
    Returns:
        PDF的完整文本内容
    """
    processor = Base64PDFProcessor(openai_api_key=openai_api_key)
    return processor.get_full_content(base64_data)

