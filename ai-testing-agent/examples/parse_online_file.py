import os

import pytesseract
from langchain_community.document_loaders import OnlinePDFLoader, UnstructuredPDFLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_core.document_loaders import BaseLoader

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from utils.model_factory import GetModelByVendor

class LoadingPdfFactory:
    def __init__(self, provider: str = ""):
        self.vision_model_client = GetModelByVendor(provider).generate_model_client()

    def generate_loader_by_file_path(self, file_path: str):
        """
        使用 OnlinePDFLoader
        :param file_path:
        :return:
        """

        # 加载在线文档 1   ok
        loader = OnlinePDFLoader(file_path)
        return loader

    def generate_loader_use_pymupdf4llm(self, file_path: str):
        """
        使用 PyMuPDF4LLMLoader
        :return:
        """
        # 加载在线文档 2   ok
        loader = PyMuPDF4LLMLoader(
            file_path=file_path,
            mode="single",
        )
        return loader

    def generate_loader_use_docling(self, file_path: str):
        """
        使用 Docling
        :return:
        """
        from langchain_docling.loader import DoclingLoader

        FILE_PATH = "https://arxiv.org/pdf/2408.09869"

        loader = DoclingLoader(file_path=file_path)
        return loader

if __name__ == '__main__':
    file_path = "https://www.who.int/docs/default-source/coronaviruse/situation-reports/20200121-sitrep-1-2019-ncov.pdf"
    file_path = "https://arxiv.org/pdf/2408.09869"
    loader = LoadingPdfFactory("").generate_loader_by_file_path(file_path=file_path)
    print(loader.load())