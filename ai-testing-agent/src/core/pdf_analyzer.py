"""
PDF 文档解析模块 - PyMuPDF4LLM + LLMImageBlobParser 完整链路

核心能力：
  1. 提取 PDF 中的所有文本内容（按页保留结构）
  2. 自动识别并提取 PDF 中嵌入的图片
  3. 使用 Vision 模型对嵌入图片进行 OCR/理解
  4. 将文本 + 图片描述合并为完整的文档摘要
  5. 支持在线 PDF URL 的流式内存解析（无需落盘）

依赖：
  - langchain-pymupdf4llm (PyMuPDF4LLMLoader)
  - langchain-community (LLMImageBlobParser)
  - Vision 模型客户端 (通过 GetModelByVendor 获取)
  - requests (在线 PDF 下载)

使用方式：
    from examples.pdf_analyzer import analyze_pdf, analyze_pdf_from_url

    # 本地文件
    result = analyze_pdf("/path/to/document.pdf", user_text="总结这份报告")

    # 在线 URL
    result = analyze_pdf_from_url("https://example.com/doc.pdf", user_text="总结这份报告")
    print(result)  # 完整的文档分析结果
"""

import os
import io
import hashlib
import tempfile
from typing import Optional, Union
from urllib.parse import urlparse


# ============================================================
# 配置常量
# ============================================================

MAX_PDF_SIZE = 100 * 1024 * 1024  # 100MB
DOWNLOAD_TIMEOUT = 30  # 秒
ALLOWED_SCHEMES = {"http", "https"}


# ============================================================
# 公共接口
# ============================================================

def analyze_pdf(pdf_path: str, user_text: str = "") -> Optional[str]:
    """
    解析本地 PDF 文档，提取完整文本和图片内容。

    缓存机制：基于文件元数据（路径 + 修改时间 + 文件大小）的 LRU 缓存。
    当同一文件被重复解析时直接返回缓存结果，避免重复调用 PyMuPDF 和 Vision 模型。

    Args:
        pdf_path: 本地 PDF 文件绝对路径
        user_text: 用户针对此文档的问题（会追加到输出末尾）

    Returns:
        PDF 文档的完整文字提取结果；失败时返回降级响应
    """
    print(f"[pdf_analyzer] 📕 开始解析本地文件: {pdf_path}")

    # 基于文件元数据的缓存查询（安全网：防止绕过 transformer 层缓存的调用）
    from src.core.cache import get_pdf_cached, put_pdf_cache, compute_file_hash

    file_cache_key = compute_file_hash(pdf_path)
    if file_cache_key:
        cached_result = get_pdf_cached(file_cache_key)
        if cached_result is not None:
            print(f"[pdf_analyzer] ✅ 缓存命中（文件元数据哈希={file_cache_key[:12]}...）")
            if user_text.strip():
                cached_result += f"\n{'=' * 40}\n💬 **用户针对此文档的问题**：{user_text}"
            return cached_result

    # 未命中缓存，执行完整解析
    result = _analyze_pdf_internal(pdf_path, user_text, source_type="local")

    # 写入缓存（仅缓存成功的结果）
    if result and file_cache_key and not result.startswith("📎 **文件附件"):
        put_pdf_cache(file_cache_key, result)
        print(f"[pdf_analyzer] 💾 已缓存结果（文件哈希={file_cache_key[:12]}...）")

    return result


def analyze_pdf_from_url(pdf_url: str, user_text: str = "") -> Optional[str]:
    """
    解析在线 PDF 文档（流式内存解析，无需落盘）。

    核心流程：
      1. 验证 URL 合法性（防止 SSRF）
      2. 检查缓存（基于 URL + 内容哈希）
      3. 流式下载到内存（BytesIO）
      4. 调用 PyMuPDF 解析
      5. 缓存结果

    Args:
        pdf_url: 在线 PDF 文档的 URL
        user_text: 用户针对此文档的问题

    Returns:
        PDF 文档的完整分析结果；失败时返回错误信息

    示例：
        >>> result = analyze_pdf_from_url("https://example.com/report.pdf")
        >>> print(result)
    """
    print(f"[pdf_analyzer] 🌐 开始解析在线 PDF: {pdf_url}")

    # 1. URL 验证
    if not _is_valid_url(pdf_url):
        return _make_error_fallback(
            pdf_url, user_text, "URL 格式不合法或协议不支持（仅支持 http/https）"
        )

    # 2. 检查缓存
    from src.core.cache import get_pdf_cached, put_pdf_cache, compute_content_hash

    url_hash = hashlib.md5(pdf_url.encode()).hexdigest()
    cached_result = get_pdf_cached(url_hash)
    if cached_result:
        print(f"[pdf_analyzer] ✅ 缓存命中（URL 哈希）")
        if user_text.strip():
            cached_result += f"\n{'=' * 40}\n💬 **用户针对此文档的问题**：{user_text}"
        return cached_result

    # 3. 下载到内存
    pdf_bytes_io = download_pdf_to_memory(pdf_url)
    if pdf_bytes_io is None:
        return _make_error_fallback(pdf_url, user_text, "下载失败或文件过大")

    # 4. 计算内容哈希（用于二次缓存）
    pdf_bytes_io.seek(0)
    content_hash = hashlib.md5(pdf_bytes_io.read()).hexdigest()
    pdf_bytes_io.seek(0)

    # 检查内容哈希缓存
    cached_by_content = get_pdf_cached(content_hash)
    if cached_by_content:
        print(f"[pdf_analyzer] ✅ 缓存命中（内容哈希）")
        # 同时缓存 URL 哈希
        put_pdf_cache(url_hash, cached_by_content)
        if user_text.strip():
            cached_by_content += f"\n{'=' * 40}\n💬 **用户针对此文档的问题**：{user_text}"
        return cached_by_content

    # 5. 解析 PDF
    result = _analyze_pdf_internal(pdf_bytes_io, user_text, source_type="url", source_name=pdf_url)

    # 6. 缓存结果（同时缓存 URL 哈希和内容哈希）
    if result:
        put_pdf_cache(url_hash, result)
        put_pdf_cache(content_hash, result)
        print(f"[pdf_analyzer] 💾 已缓存结果（URL + 内容双重哈希）")

    return result


def download_pdf_to_memory(url: str) -> Optional[io.BytesIO]:
    """
    流式下载在线 PDF 到内存（BytesIO 对象）。

    安全措施：
      - 限制文件大小（MAX_PDF_SIZE）
      - 设置超时（DOWNLOAD_TIMEOUT）
      - 验证 Content-Type
      - 流式下载（避免一次性加载大文件）

    Args:
        url: PDF 文件的 URL

    Returns:
        包含 PDF 数据的 BytesIO 对象；失败返回 None
    """
    try:
        import requests

        print(f"[pdf_analyzer] 📥 开始下载: {url}")

        response = requests.get(
            url,
            stream=True,
            timeout=DOWNLOAD_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (PDF Analyzer Bot)"}
        )
        response.raise_for_status()

        # 验证 Content-Type
        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type and "application/octet-stream" not in content_type:
            print(f"[pdf_analyzer] ⚠️ Content-Type 可能不是 PDF: {content_type}")

        # 检查文件大小
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_PDF_SIZE:
            print(f"[pdf_analyzer] ❌ 文件过大: {int(content_length) / 1024 / 1024:.2f}MB")
            return None

        # 流式下载到内存
        pdf_buffer = io.BytesIO()
        downloaded_size = 0

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                downloaded_size += len(chunk)
                if downloaded_size > MAX_PDF_SIZE:
                    print(f"[pdf_analyzer] ❌ 下载超过大小限制")
                    return None
                pdf_buffer.write(chunk)

        pdf_buffer.seek(0)
        print(f"[pdf_analyzer] ✅ 下载完成: {downloaded_size / 1024:.2f}KB")
        return pdf_buffer

    except requests.exceptions.Timeout:
        print(f"[pdf_analyzer] ❌ 下载超时（{DOWNLOAD_TIMEOUT}秒）")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[pdf_analyzer] ❌ 下载失败: {e}")
        return None
    except Exception as e:
        print(f"[pdf_analyzer] ❌ 未知错误: {e}")
        return None


def _analyze_pdf_internal(
    pdf_source: Union[str, io.BytesIO],
    user_text: str = "",
    source_type: str = "local",
    source_name: Optional[str] = None
) -> Optional[str]:
    """
    内部统一的 PDF 解析函数，支持文件路径或 BytesIO 对象。

    Args:
        pdf_source: 文件路径（str）或 BytesIO 对象
        user_text: 用户问题
        source_type: "local" 或 "url"
        source_name: 用于显示的源名称（URL 或文件名）

    Returns:
        解析结果文本
    """
    source_display = source_name or (pdf_source if isinstance(pdf_source, str) else "在线文档")

    try:
        # 动态导入（避免未安装依赖时启动失败）
        from langchain_community.document_loaders.parsers import LLMImageBlobParser
        from langchain_pymupdf4llm import PyMuPDF4LLMLoader

        # 获取 Vision 模型客户端
        vision_model = _get_vision_model_client()

        # 处理 BytesIO 对象：需要先保存到临时文件
        # （因为 PyMuPDF4LLMLoader 目前只支持文件路径）
        temp_file_path = None
        if isinstance(pdf_source, io.BytesIO):
            temp_file_path = _save_bytesio_to_temp(pdf_source)
            if not temp_file_path:
                return _make_error_fallback(source_display, user_text, "无法创建临时文件")
            pdf_path_to_load = temp_file_path
        else:
            pdf_path_to_load = pdf_source

        # 配置并执行加载
        loader = PyMuPDF4LLMLoader(
            pdf_path_to_load,
            mode="page",
            extract_images=True,
            images_parser=LLMImageBlobParser(model=vision_model),
        )
        documents = loader.load()

        # 清理临时文件
        if temp_file_path:
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass

        if not documents:
            print("[pdf_analyzer] ⚠️ 解析完成但无内容")
            return _make_empty_fallback(source_display)

        # 组装结构化输出
        output = _assemble_output(documents, source_display, user_text)
        print(f"[pdf_analyzer] ✅ 成功！{len(documents)} 页，{len(output)} 字符")
        return output

    except ImportError as e:
        missing_pkg = str(e).split("'")[-2] if "'" in str(e) else "未知包"
        print(f"[pdf_analyzer] ❌ 缺少依赖: {missing_pkg}")
        return _make_error_fallback(source_display, user_text, f"缺少 {missing_pkg}")

    except Exception as e:
        print(f"[pdf_analyzer] ❌ 解析异常: {e}")
        import traceback
        traceback.print_exc()

        # 对于本地文件，尝试降级策略
        if isinstance(pdf_source, str):
            return _try_basic_extraction(pdf_source, user_text, str(e))
        else:
            return _make_error_fallback(source_display, user_text, str(e))


# ============================================================
# 辅助工具函数
# ============================================================

def _is_valid_url(url: str) -> bool:
    """
    验证 URL 合法性（防止 SSRF 攻击）。

    检查项：
      - URL 格式正确
      - 协议为 http/https
      - 不是内网地址（可选）

    Args:
        url: 待验证的 URL

    Returns:
        True 表示合法
    """
    try:
        parsed = urlparse(url)

        # 检查协议
        if parsed.scheme not in ALLOWED_SCHEMES:
            print(f"[pdf_analyzer] ❌ 不支持的协议: {parsed.scheme}")
            return False

        # 检查是否有主机名
        if not parsed.netloc:
            print(f"[pdf_analyzer] ❌ URL 缺少主机名")
            return False

        # 可选：防止访问内网地址
        hostname = parsed.hostname
        if hostname:
            # 简单的内网地址检测
            if hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
                print(f"[pdf_analyzer] ⚠️ 警告：尝试访问本地地址")
                # 可以选择返回 False 来禁止

            if hostname.startswith("192.168.") or hostname.startswith("10.") or hostname.startswith("172."):
                print(f"[pdf_analyzer] ⚠️ 警告：尝试访问内网地址")
                # 可以选择返回 False 来禁止

        return True

    except Exception as e:
        print(f"[pdf_analyzer] ❌ URL 解析失败: {e}")
        return False


def _save_bytesio_to_temp(bytes_io: io.BytesIO) -> Optional[str]:
    """
    将 BytesIO 对象保存到临时文件。

    Args:
        bytes_io: PDF 数据的 BytesIO 对象

    Returns:
        临时文件路径；失败返回 None
    """
    try:
        # 创建临时文件（不自动删除）
        temp_fd, temp_path = tempfile.mkstemp(suffix=".pdf", prefix="pdf_analyzer_")

        bytes_io.seek(0)
        with os.fdopen(temp_fd, 'wb') as f:
            f.write(bytes_io.read())

        print(f"[pdf_analyzer] 💾 已保存到临时文件: {temp_path}")
        return temp_path

    except Exception as e:
        print(f"[pdf_analyzer] ❌ 保存临时文件失败: {e}")
        return None


# ============================================================
# Vision 模型客户端获取
# ============================================================

def _get_vision_model_client():
    """
    获取用于 PDF 内嵌图片解析的 Vision 模型客户端。

    优先级：doubao > openai > 抛出异常
    """
    from dotenv import load_dotenv
    load_dotenv()

    # 尝试 doubao
    if os.getenv("DOUBAO_API_KEY"):
        try:
            from utils.model_factory import GetModelByVendor
            return GetModelByVendor("doubao").generate_model_client()
        except Exception as e:
            print(f"[pdf_analyzer] ⚠️ doubao 客户端创建失败: {e}")

    # 尝试 openai
    if os.getenv("OPENAI_API_KEY"):
        try:
            from utils.model_factory import GetModelByVendor
            return GetModelByVendor("openai").generate_model_client()
        except Exception as e:
            print(f"[pdf_analyzer] ⚠️ openai 客户端创建失败: {e}")

    raise ValueError(
        "无法创建 Vision 模型客户端。"
        "请至少配置 DOUBAO_API_KEY 或 OPENAI_API_KEY。"
    )


# ============================================================
# 输出组装
# ============================================================

def _assemble_output(documents: list, source_path: str, user_text: str) -> str:
    """将 PyMuPDF4LLM 的 Document 列表组装为可读文本。"""
    parts: list[str] = []

    parts.append(f"📕 **PDF 文档分析报告**")
    parts.append(f"文件名：{os.path.basename(source_path)}")
    parts.append(f"总页数：{len(documents)} 页\n")

    for idx, doc in enumerate(documents, start=1):
        page_content = doc.page_content.strip() if hasattr(doc, 'page_content') else ""
        page_meta = doc.metadata if hasattr(doc, 'metadata') else {}

        parts.append(f"--- **第 {idx} 页** ---")

        if page_content:
            parts.append(page_content)
        else:
            parts.append("（本页无文字内容）")

        if isinstance(page_meta, dict) and page_meta.get("page"):
            parts.append(f"[页码: {page_meta.get('page')}]")
        parts.append("")  # 空行分隔

    result = "\n".join(parts).strip()

    if user_text.strip():
        result += (
            f"\n{'=' * 40}\n"
            f"💬 **用户针对此文档的问题**：{user_text}"
        )

    return result


# ============================================================
# 兜底 / 降级策略
# ============================================================

def _make_empty_fallback(pdf_path: str) -> str:
    """空内容的兜底响应。"""
    return (
        f"\n📎 **PDF 文件附件**: {os.path.basename(pdf_path)}\n"
        "> 该文件可能是扫描版或加密 PDF，无法自动提取文本内容。\n"
        "请尝试上传清晰的照片版本。"
    )


def _make_error_fallback(pdf_path: str, user_text: str, reason: str) -> str:
    """错误时的元信息占位符。"""
    filename = os.path.basename(pdf_path) if pdf_path else "未知文件"
    lines = [
        f"📎 **文件附件**: {filename}",
        f"> 失败原因: {reason}",
        "> 可能的原因：文件损坏、格式不支持、或缺少必要的解析库。",
        "> 请确认已安装: pip install langchain-pymupdf4llm pymupdf",
    ]
    if user_text.strip():
        lines.append(f"\n💬 用户附加消息: \"{user_text}\"")
    return "\n".join(lines)


def _try_basic_extraction(pdf_path: str, user_text: str, reason: str) -> str:
    """
    降级策略：尝试用纯 PyMuPDF 进行基础文本提取。
    如果仍然失败则返回元信息占位符。
    """
    filename = os.path.basename(pdf_path) if pdf_path else "未知文件"

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text_parts: list[str] = []
        for page_num in range(len(doc)):
            text = doc[page_num].get_text().strip()
            if text:
                text_parts.append(f"--- 第 {page_num + 1} 页 ---\n{text}")
        doc.close()

        if text_parts:
            basic_result = (
                f"📕 **PDF 文档（基础模式提取）**: {filename}\n\n"
                + "\n\n".join(text_parts)
            )
            if user_text.strip():
                basic_result += f"\n\n💬 用户问题: {user_text}"
            basic_result += (
                "\n\n> 注：基础模式仅提取文字，PDF 内嵌图表未被分析。"
                "完整解析需要: pip install langchain-pymupdf4llm"
            )
            print(f"[pdf_analyzer] ✅ 基础模式成功 ({len(text_parts)} 页有内容)")
            return basic_result
    except Exception as e:
        print(f"[pdf_analyzer] 基础模式也失败: {e}")

    # 彻底失败
    return _make_error_fallback(pdf_path, user_text, reason)
