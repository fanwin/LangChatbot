"""
Base64 PDF 文档处理器

接收 base64 编码的 PDF 数据，使用 PyMuPDF4LLM + LLMImageBlobParser
提取完整文本内容（含内嵌图片的 Vision 模型识别），返回结构化的文档内容。

依赖：
  - langchain-pymupdf4llm
  - langchain-community
  - langchain-openai (或通过 GetModelByVendor 获取多模态模型)

用法示例：
    from processors.base64_processor import Base64PDFProcessor

    processor = Base64PDFProcessor()
    result = processor.process(base64_data, filename="doc.pdf")
    print(result.content)   # 完整文本
    print(result.page_count) # 页数
"""

import os
import re
import base64
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PDFProcessResult:
    """PDF 处理结果数据类。"""
    success: bool
    content: str = ""
    page_count: int = 0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class Base64PDFProcessor:
    """
    Base64 PDF 文档处理器。

    核心流程：
      1. 接收 base64 编码的 PDF 原始数据（或完整 data URL）
      2. 解码并保存为本地临时文件
      3. 使用 PyMuPDF4LLMLoader 按页提取文本
      4. 通过 LLMImageBlobParser + Vision 模型识别内嵌图片
      5. 组装完整的文档内容并返回
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        save_dir: Optional[str] = None,
        extract_images: bool = True,
    ):
        """
        初始化处理器。

        Args:
            model_name: 用于解析 PDF 内嵌图片的 Vision 模型名称
            max_tokens: Vision 模型的最大输出 token 数
            save_dir: 临时文件保存目录（默认系统临时目录）
            extract_images: 是否启用 PDF 内嵌图片的 LLM 解析
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.save_dir = save_dir or os.path.join(tempfile.gettempdir(), "pdf_processor_tmp")
        self.extract_images = extract_images
        os.makedirs(self.save_dir, exist_ok=True)

    # ================================================================
    # 公共接口
    # ================================================================

    def process(
        self,
        base64_data: str,
        filename: Optional[str] = None,
    ) -> PDFProcessResult:
        """
        处理 base64 编码的 PDF 数据，返回完整文档内容。

        支持两种输入格式：
          - 纯 base64 字符串（原始编码数据）
          - Data URL 格式（data:application/pdf;base64,xxxxx）

        Args:
            base64_data: PDF 的 base64 编码数据
            filename: 可选文件名（用于日志和元信息）

        Returns:
            PDFProcessResult，包含解析结果或错误信息
        """
        print(f"[Base64PDFProcessor] 开始处理 PDF 文档...")

        # ---- Step 1: 解码 base64 并保存为临时文件 ----
        pdf_path = self._decode_and_save(base64_data, filename)
        if pdf_path is None:
            return PDFProcessResult(
                success=False,
                error="base64 数据解码失败，请检查输入格式是否正确",
            )

        try:
            # ---- Step 2: 使用 PyMuPDF4LLM + LLMImageBlobParser 解析 ----
            documents = self._load_pdf(pdf_path)

            if not documents:
                return PDFProcessResult(
                    success=True,
                    content="",
                    page_count=0,
                    error="PDF 解析完成但未提取到任何内容（可能是扫描版或空文档）",
                    metadata={"source": pdf_path},
                )

            # ---- Step 3: 组装输出内容 ----
            result = self._assemble_output(documents, pdf_path)

            print(f"[Base64PDFProcessor] ✅ 处理完成！"
                  f"共 {result.page_count} 页，内容长度 {len(result.content)} 字符")
            return result

        except ImportError as e:
            missing_pkg = str(e).split("'")[-2] if "'" in str(e) else "未知包"
            print(f"[Base64PDFProcessor] ❌ 缺少依赖: {missing_pkg}")
            return PDFProcessResult(
                success=False,
                error=f"缺少 Python 依赖包: {missing_pkg}。"
                       f"请执行: pip install langchain-pymupdf4llm langchain-community langchain-openai",
            )
        except Exception as e:
            print(f"[Base64PDFProcessor] ❌ 处理异常: {e}")
            import traceback
            traceback.print_exc()
            return PDFProcessResult(
                success=False,
                error=str(e),
            )
        finally:
            # 清理临时文件
            self._cleanup_temp_file(pdf_path)

    # ================================================================
    # 内部方法：解码与保存
    # ================================================================

    def _decode_and_save(
        self,
        base64_data: str,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """
        将 base64 数据解码为二进制并保存为临时 PDF 文件。

        Args:
            base64_data: 纯 base64 或 data URL 格式字符串
            filename: 可选文件名

        Returns:
            保存后的文件绝对路径；失败返回 None
        """
        raw_bytes = self._extract_raw_bytes(base64_data)
        if raw_bytes is None:
            return None

        # 确定文件名
        if filename and filename.lower().endswith(".pdf"):
            safe_name = re.sub(r'[^\w\-.]', '_', filename)
        else:
            safe_name = f"{uuid.uuid4().hex}.pdf"

        filepath = os.path.join(self.save_dir, safe_name)

        try:
            with open(filepath, "wb") as f:
                f.write(raw_bytes)
            print(f"[Base64PDFProcessor] 📄 PDF 已保存: {filepath} "
                  f"({len(raw_bytes)} bytes)")
            return filepath
        except Exception as e:
            print(f"[Base64PDFProcessor] 保存文件失败: {e}")
            return None

    @staticmethod
    def _extract_raw_bytes(base64_data: str) -> Optional[bytes]:
        """
        从 base64 数据中提取原始字节。

        支持：
          - 纯 base64 字符串
          - data:application/pdf;base64,xxxxx 格式
        """
        if not base64_data or not base64_data.strip():
            print("[Base64PDFProcessor] 输入数据为空")
            return None

        try:
            data = base64_data.strip()

            # 检测 data URL 格式
            if data.startswith("data:"):
                match = re.match(r"data:[^;]+;base64,(.+)", data, re.DOTALL)
                if not match:
                    print("[Base64PDFProcessor] data URL 格式无效")
                    return None
                b64_str = match.group(1)
            else:
                b64_str = data

            # 标准化 base64 字符串
            b64_str = b64_str.replace("-", "+").replace("_", "/")
            padding = len(b64_str) % 4
            if padding:
                b64_str += "=" * (4 - padding)

            return base64.b64decode(b64_str)
        except Exception as e:
            print(f"[Base64PDFProcessor] base64 解码失败: {e}")
            return None

    # ================================================================
    # 内部方法：PDF 加载与解析
    # ================================================================

    def _load_pdf(self, pdf_path: str):
        """
        使用 PyMuPDF4LLMLoader 加载并解析 PDF 文件。

        核心配置：
          - mode="page": 按页返回文档，保留页面结构
          - extract_images=True: 提取 PDF 内嵌图片
          - images_parser=LLMImageBlobParser(...): 用 Vision 模型理解图片内容

        Args:
            pdf_path: PDF 文件的本地路径

        Returns:
            Document 对象列表
        """
        from langchain_community.document_loaders.parsers import LLMImageBlobParser
        from langchain_pymupdf4llm import PyMuPDF4LLMLoader

        # 获取 Vision 模型客户端
        vision_model = self._get_vision_model()

        loader = PyMuPDF4LLMLoader(
            pdf_path,
            mode="page",
            extract_images=self.extract_images,
            images_parser=LLMImageBlobParser(model=vision_model),
        )

        documents = loader.load()
        print(f"[Base64PDFProcessor] 📖 PDF 加载完成，共 {len(documents)} 页")
        return documents

    def _get_vision_model(self):
        """
        获取用于解析 PDF 内嵌图片的 Vision 模型客户端。

        优先级策略：
          1. 尝试使用项目已有的 GetModelByVendor 工厂
          2. 回退到 OpenAI ChatOpenAI (gpt-4o-mini)
        """
        # 策略 1: 尝试项目内的 model_factory
        try:
            from utils.model_factory import GetModelByVendor
            return GetModelByVendor("openai").generate_model_client()
        except Exception as e:
            print(f"[Base64PDFProcessor] GetModelByVendor 不可用: {e}")

        # 策略 2: 直接使用 ChatOpenAI
        from dotenv import load_dotenv
        load_dotenv()

        from langchain_openai import ChatOpenAI
        import os

        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL")

        if not api_key:
            raise ValueError(
                "无法创建 Vision 模型客户端。"
                "请设置 OPENAI_API_KEY 环境变量，或在项目中配置 GetModelByVendor。"
            )

        model_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "model": self.model_name,
            "max_tokens": self.max_tokens,
        }
        if base_url:
            model_kwargs["base_url"] = base_url

        print(f"[Base64PDFProcessor] 使用 ChatOpenAI ({self.model_name}) 作为 Vision 模型")
        return ChatOpenAI(**model_kwargs)

    # ================================================================
    # 内部方法：结果组装
    # ================================================================

    def _assemble_output(self, documents: list, source_path: str) -> PDFProcessResult:
        """
        将 PyMuPDF4LLM 返回的 Document 列表组装为结构化输出。

        Args:
            documents: Document 对象列表
            source_path: 源 PDF 文件路径

        Returns:
            PDFProcessResult
        """
        output_parts: list[str] = []
        all_metadata: dict[str, Any] = {}

        filename = os.path.basename(source_path)
        output_parts.append(f"📕 **PDF 文档**: {filename}")
        output_parts.append(f"总页数：{len(documents)} 页\n")

        for idx, doc in enumerate(documents, start=1):
            page_content = doc.page_content.strip() if hasattr(doc, 'page_content') else ""
            page_meta = doc.metadata if hasattr(doc, 'metadata') else {}

            output_parts.append(f"--- 第 {idx} 页 ---")

            if page_content:
                output_parts.append(page_content)
            else:
                output_parts.append("（本页无文字内容）")

            # 收集页面元信息
            if page_meta:
                page_label = page_meta.get("page", idx)
                output_parts.append(f"[页码: {page_label}]")
                all_metadata[f"page_{idx}"] = page_meta

            output_parts.append("")  # 空行分隔

        final_content = "\n".join(output_parts).strip()

        all_metadata.update({
            "source": source_path,
            "filename": filename,
            "total_pages": len(documents),
        })

        return PDFProcessResult(
            success=True,
            content=final_content,
            page_count=len(documents),
            metadata=all_metadata,
        )

    # ================================================================
    # 工具方法
    # ================================================================

    @staticmethod
    def _cleanup_temp_file(filepath: str):
        """安全删除临时文件。"""
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                print(f"[Base64PDFProcessor] 🧹 临时文件已清理: {filepath}")
        except Exception as e:
            print(f"[Base64PDFProcessor] 清理临时文件失败: {e}")

    def __repr__(self) -> str:
        return (
            f"Base64PDFProcessor(model={self.model_name}, "
            f"max_tokens={self.max_tokens}, "
            f"extract_images={self.extract_images})"
        )


# ================================================================
# 便捷函数（可直接调用，无需实例化）
# ================================================================

def process_base64_pdf(
    base64_data: str,
    filename: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
    max_tokens: int = 1024,
) -> PDFProcessResult:
    """
    一站式处理 base64 PDF 数据的便捷函数。

    Args:
        base64_data: PDF 的 base64 编码（纯 base64 或 data URL 格式）
        filename: 可选文件名
        model_name: Vision 模型名称
        max_tokens: Vision 模型最大 token 数

    Returns:
        PDFProcessResult

    示例：
        result = process_base64_pdf(b64_string, "report.pdf")
        if result.success:
            print(result.content)
        else:
            print(f"处理失败: {result.error}")
    """
    processor = Base64PDFProcessor(
        model_name=model_name,
        max_tokens=max_tokens,
    )
    return processor.process(base64_data, filename=filename)


if __name__ == "__main__":
    # 简单测试入口
    import sys

    print("=" * 50)
    print("  Base64 PDF Processor - 测试模式")
    print("=" * 50)

    if len(sys.argv) < 2:
        print("\n用法:")
        print('  python base64_processor.py "<base64_data>" [filename]')
        print('\n或传入 data URL 格式:')
        print('  python base64_processor.py "data:application/pdf;base64,..."')
        sys.exit(1)

    b64_input = sys.argv[1]
    fname = sys.argv[2] if len(sys.argv) > 2 else "test.pdf"

    result = process_base64_pdf(b64_input, filename=fname)

    if result.success:
        print("\n" + "=" * 50)
        print("✅ 处理成功！")
        print("=" * 50)
        print(result.content)
        print(f"\n--- 元信息 ---")
        for k, v in result.metadata.items():
            print(f"  {k}: {v}")
    else:
        print(f"\n❌ 处理失败: {result.error}")
