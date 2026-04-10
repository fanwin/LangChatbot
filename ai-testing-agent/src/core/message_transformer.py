"""
消息转换模块 - 多模态 HumanMessage → 纯文本 HumanMessage

核心职责：
  将包含图片/PDF 等多模态附件的 HumanMessage 转换为纯文本格式，
  使非多模态大模型也能"理解"附件内容。

新增功能：
  - 自动识别纯文本中的 PDF URL 并解析（支持在线文档）

输出结构（两层分离，与原版 core/hatch_agent.py 保持一致）：
  ┌─ content（前端可见）── 用户文字
  │  + 标记包裹的模型专用数据（前端自动隐藏）
  │
  └─ additional_kwargs.attachments —— 供前端渲染缩略图/文件卡片

使用方式：
    from src.core.message_transformer import transform_multimodal_message

    new_msg = transform_multimodal_message(original_humanmessage)
    # new_msg.content: "用户文字\n<!-- __HATCH_AGENT_INTERNAL_START__ -->\n图片分析结果\n<!-- __HATCH_AGENT_INTERNAL_END__ -->"
"""

import re
from typing import Any

from langchain_core.messages import HumanMessage

from src.core.cache import compute_content_hash, get_image_cached, put_image_cache, get_pdf_cached, put_pdf_cache
from src.core.file_utils import save_base64_to_local, save_base64_image_to_local
from src.core.image_analyzer import analyze_image
from src.core.pdf_analyzer import analyze_pdf, analyze_pdf_from_url


# ============================================================
# 前后端约定的标记：用于分隔「用户可见文字」和「模型专用分析数据」
# 前端渲染时会自动剥离此标记之间的内容（utils.ts -> stripModelInternalData）
# ============================================================

_MODEL_DATA_MARKER_START = "\n<!-- __HATCH_AGENT_INTERNAL_START__ -->\n"
_MODEL_DATA_MARKER_END = "\n<!-- __HATCH_AGENT_INTERNAL_END__ -->\n"


# ============================================================
# URL 提取正则表达式
# ============================================================

# 匹配 http/https 开头的 PDF URL
PDF_URL_PATTERN = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+\.pdf(?:\?[^\s<>"{}|\\^`\[\]]*)?',
    re.IGNORECASE
)


# ============================================================
# MIME 类型 → 中文标签 映射
# ============================================================

MEDIA_TYPE_LABELS = {
    "application/pdf": "PDF 文档",
    "audio/mpeg": "音频",
    "audio/wav": "音频",
    "video/mp4": "视频",
}


# ============================================================
# 核心转换函数
# ============================================================

def transform_multimodal_message(message: HumanMessage) -> HumanMessage:
    """
    将多模态 HumanMessage 转换为纯文本格式。

    处理流程：
      1. 遍历 content 数组中的每个 part
      2. 文本块 → 直接收集
      3. 图片块 → Vision 模型分析（带 LRU 缓存）
      4. PDF 块 → PyMuPDF4LLM 解析（带 LRU 缓存）
      5. 其他文件 → 记录元信息
      6. 纯文本中的 PDF URL → 自动识别并解析
      7. 组装：content = 用户可见文字 + 标记包裹的模型专用数据

    Args:
        message: 可能包含多模态 content 的 HumanMessage

    Returns:
        转换后的新 HumanMessage，content 中包含标记包裹的附件分析数据
    """
    content = message.content

    # 已经是纯文本，检查是否包含 PDF URL
    if isinstance(content, str):
        return _handle_plain_text_with_urls(message, content)

    if not isinstance(content, list):
        return message

    # ---- 各类内容收集器 ----
    text_parts: list[str] = []
    image_descriptions: list[str] = []
    document_contents: list[str] = []
    attachment_summary_parts: list[str] = []
    attachment_metadata: list[dict] = []

    for part in content:
        _process_part(
            part, text_parts, image_descriptions, document_contents,
            attachment_summary_parts, attachment_metadata,
        )

    # ---- 检查文本中是否包含 PDF URL ----
    user_text = " ".join(text_parts)
    pdf_urls = extract_pdf_urls(user_text)
    if pdf_urls:
        _process_pdf_urls(pdf_urls, user_text, document_contents, attachment_summary_parts, attachment_metadata)

    # ---- 组装最终消息 ----
    visible_text = _build_visible_text(text_parts, attachment_summary_parts)
    model_context = _build_model_context(image_descriptions, document_contents)

    new_msg = _assemble_message(
        message, visible_text, model_context, attachment_metadata,
    )

    print(f"[transformer] ✅ 转换完成 | 可见: {len(visible_text)} 字符"
          f" | 模型上下文: {len(model_context)} 字符")

    return new_msg


def _handle_plain_text_with_urls(message: HumanMessage, content: str) -> HumanMessage:
    """
    处理纯文本消息，检查是否包含 PDF URL。

    Args:
        message: 原始消息
        content: 纯文本内容

    Returns:
        如果包含 PDF URL，返回转换后的消息；否则返回原消息
    """
    pdf_urls = extract_pdf_urls(content)

    if not pdf_urls:
        return message

    print(f"[transformer] 🔗 检测到纯文本中的 PDF URL: {len(pdf_urls)} 个")

    # 收集器
    document_contents: list[str] = []
    attachment_summary_parts: list[str] = []
    attachment_metadata: list[dict] = []

    # 处理所有 PDF URL
    _process_pdf_urls(pdf_urls, content, document_contents, attachment_summary_parts, attachment_metadata)

    # 如果没有成功解析任何 PDF，返回原消息
    if not document_contents:
        return message

    # 组装消息
    model_context = "\n".join(document_contents).strip()

    final_content = content
    if model_context:
        final_content += (
            _MODEL_DATA_MARKER_START
            + model_context
            + _MODEL_DATA_MARKER_END
        )

    new_msg = HumanMessage(content=final_content)

    # 保留原始元数据
    if hasattr(message, "id") and message.id:
        new_msg.id = message.id
    if hasattr(message, "name") and message.name:
        new_msg.name = message.name
    if hasattr(message, "response_metadata") and message.response_metadata:
        new_msg.response_metadata = message.response_metadata

    # 附件元信息
    if attachment_metadata:
        new_msg.additional_kwargs["attachments"] = attachment_metadata

    print(f"[transformer] ✅ 纯文本 URL 转换完成 | 模型上下文: {len(model_context)} 字符")

    return new_msg


def extract_pdf_urls(text: str) -> list[str]:
    """
    从文本中提取所有 PDF URL。

    Args:
        text: 待检测的文本

    Returns:
        PDF URL 列表
    """
    if not text:
        return []

    urls = PDF_URL_PATTERN.findall(text)
    return list(set(urls))  # 去重


def _process_pdf_urls(
    pdf_urls: list[str],
    user_text: str,
    document_contents: list[str],
    attachment_summary_parts: list[str],
    attachment_metadata: list[dict],
) -> None:
    """
    处理提取到的 PDF URL 列表。

    Args:
        pdf_urls: PDF URL 列表
        user_text: 用户输入的文本
        document_contents: 文档内容收集器
        attachment_summary_parts: 附件摘要收集器
        attachment_metadata: 附件元数据收集器
    """
    for url in pdf_urls:
        print(f"[transformer] 🌐 解析在线 PDF: {url}")

        try:
            # 调用在线 PDF 解析
            doc_content = analyze_pdf_from_url(url, user_text)

            if doc_content:
                document_contents.append(doc_content)

                # 提取文件名
                filename = url.split('/')[-1].split('?')[0] or "在线文档.pdf"
                attachment_summary_parts.append(f"📕 {filename}")

                # 添加元数据
                attachment_metadata.append({
                    "type": "file",
                    "mimeType": "application/pdf",
                    "filename": filename,
                    "url": url,
                })

                print(f"[transformer] ✅ 在线 PDF 解析成功: {filename}")
            else:
                print(f"[transformer] ⚠️ 在线 PDF 解析失败: {url}")

        except Exception as e:
            print(f"[transformer] ❌ 在线 PDF 解析异常: {e}")
            import traceback
            traceback.print_exc()


def _handle_plain_text_with_urls(message: HumanMessage, content: str) -> HumanMessage:
    """
    处理纯文本消息，检查是否包含 PDF URL。

    Args:
        message: 原始消息
        content: 纯文本内容

    Returns:
        如果包含 PDF URL，返回转换后的消息；否则返回原消息
    """
    pdf_urls = extract_pdf_urls(content)

    if not pdf_urls:
        return message

    print(f"[transformer] 🔗 检测到纯文本中的 PDF URL: {len(pdf_urls)} 个")

    # 收集器
    document_contents: list[str] = []
    attachment_summary_parts: list[str] = []
    attachment_metadata: list[dict] = []

    # 处理所有 PDF URL
    _process_pdf_urls(pdf_urls, content, document_contents, attachment_summary_parts, attachment_metadata)

    # 如果没有成功解析任何 PDF，返回原消息
    if not document_contents:
        return message

    # 组装消息
    model_context = "\n".join(document_contents).strip()

    final_content = content
    if model_context:
        final_content += (
            _MODEL_DATA_MARKER_START
            + model_context
            + _MODEL_DATA_MARKER_END
        )

    new_msg = HumanMessage(content=final_content)

    # 保留原始元数据
    if hasattr(message, "id") and message.id:
        new_msg.id = message.id
    if hasattr(message, "name") and message.name:
        new_msg.name = message.name
    if hasattr(message, "response_metadata") and message.response_metadata:
        new_msg.response_metadata = message.response_metadata

    # 附件元信息
    if attachment_metadata:
        new_msg.additional_kwargs["attachments"] = attachment_metadata

    print(f"[transformer] ✅ 纯文本 URL 转换完成 | 模型上下文: {len(model_context)} 字符")

    return new_msg


def extract_pdf_urls(text: str) -> list[str]:
    """
    从文本中提取所有 PDF URL。

    Args:
        text: 待检测的文本

    Returns:
        PDF URL 列表
    """
    if not text:
        return []

    urls = PDF_URL_PATTERN.findall(text)
    return list(set(urls))  # 去重


def _process_pdf_urls(
    pdf_urls: list[str],
    user_text: str,
    document_contents: list[str],
    attachment_summary_parts: list[str],
    attachment_metadata: list[dict],
) -> None:
    """
    处理提取到的 PDF URL 列表。

    Args:
        pdf_urls: PDF URL 列表
        user_text: 用户输入的文本
        document_contents: 文档内容收集器
        attachment_summary_parts: 附件摘要收集器
        attachment_metadata: 附件元数据收集器
    """
    for url in pdf_urls:
        print(f"[transformer] 🌐 解析在线 PDF: {url}")

        try:
            # 调用在线 PDF 解析
            doc_content = analyze_pdf_from_url(url, user_text)

            if doc_content:
                document_contents.append(doc_content)

                # 提取文件名
                filename = url.split('/')[-1].split('?')[0] or "在线文档.pdf"
                attachment_summary_parts.append(f"📕 {filename}")

                # 添加元数据
                attachment_metadata.append({
                    "type": "file",
                    "mimeType": "application/pdf",
                    "filename": filename,
                    "url": url,
                })

                print(f"[transformer] ✅ 在线 PDF 解析成功: {filename}")
            else:
                print(f"[transformer] ⚠️ 在线 PDF 解析失败: {url}")

        except Exception as e:
            print(f"[transformer] ❌ 在线 PDF 解析异常: {e}")
            import traceback
            traceback.print_exc()


# ============================================================
# 分块处理逻辑
# ============================================================

def _process_part(
    part,
    text_parts: list[str],
    image_descriptions: list[str],
    document_contents: list[str],
    attachment_summary_parts: list[str],
    attachment_metadata: list[dict],
) -> None:
    """处理 content 数组中的单个 part。"""
    # 纯字符串文本
    if isinstance(part, str):
        text_parts.append(part)
        return

    if not isinstance(part, dict) or "type" not in part:
        return

    part_type = part["type"]

    # --- 文本块 ---
    if part_type == "text" and part.get("text"):
        text_parts.append(str(part["text"]))
        return

    # --- 图片块 ---
    if part_type == "image_url":
        _handle_image_part(
            part, text_parts, image_descriptions, attachment_summary_parts, attachment_metadata,
        )
        return

    # --- 文件块 ---
    if part_type == "file":
        _handle_file_part(
            part, text_parts, document_contents, attachment_summary_parts, attachment_metadata,
        )
        return


def _handle_image_part(
    part: dict,
    text_parts: list[str],
    image_descriptions: list[str],
    attachment_summary_parts: list[str],
    attachment_metadata: list[dict],
) -> None:
    """处理图片类型的 part。"""
    image_url = part.get("image_url", {})
    url_data = image_url.get("url", "") if isinstance(image_url, dict) else ""

    if not url_data:
        return

    print("[transformer] 📷 检测到图片")
    attachment_summary_parts.append("📷 图片")

    # 收集元信息
    attachment_metadata.append({"type": "image", "url": url_data})

    # 查询缓存
    cache_key = compute_content_hash(url_data)
    cached_desc = get_image_cached(cache_key) if cache_key else None

    if cached_desc is not None:
        print(f"[transformer] 📷 图片命中缓存 (hash={cache_key[:12]}...)")
        image_descriptions.append(f"\n📷 **图片内容分析**：\n{cached_desc}")
    else:
        local_path = save_base64_image_to_local(url_data)
        if local_path:
            user_text = " ".join(text_parts)
            desc = analyze_image(local_path, user_text)
            if desc and cache_key:
                put_image_cache(cache_key, desc)
                print(f"[transformer] 📷 已缓存 (hash={cache_key[:12]}...)")
            if desc:
                image_descriptions.append(f"\n📷 **图片内容分析**：\n{desc}")


def _handle_file_part(
    part: dict,
    text_parts: list[str],
    document_contents: list[str],
    attachment_summary_parts: list[str],
    attachment_metadata: list[dict],
) -> None:
    """处理文件类型的 part。"""
    filename = part.get("filename", "未知文件")
    media_type = (part.get("source_media_type") or "").lower()
    source_data = part.get("source_data", "")

    print(f"[transformer] 📎 检测到文件: {filename} (MIME: {media_type})")

    is_pdf = (
        media_type == "application/pdf"
        or filename.lower().endswith(".pdf")
    )

    if is_pdf:
        attachment_summary_parts.append(f"📕 {filename}")
    else:
        attachment_summary_parts.append(f"📎 {filename}")

    attachment_metadata.append({
        "type": "file",
        "mimeType": part.get("source_media_type", "application/octet-stream"),
        "filename": filename,
    })

    # PDF 完整解析链路
    if is_pdf and source_data:
        _handle_pdf_file(source_data, filename, text_parts, document_contents)
        return

    # 非 PDF 或 PDF 解析失败的兜底
    type_label = MEDIA_TYPE_LABELS.get(media_type, "文件")
    file_info = f"\n📎 **{type_label}附件**: {filename}"
    if media_type:
        file_info += f" (类型: {media_type})"
    file_info += "\n> 注：该文件未能被自动解析内容。"
    document_contents.append(file_info)


def _handle_pdf_file(
    source_data: str,
    filename: str,
    text_parts: list[str],
    document_contents: list[str],
) -> None:
    """处理 PDF 文件的完整解析（含缓存查询）。"""
    print("[transformer] 📕 启动 PDF 解析...")

    pdf_cache_key = compute_content_hash(source_data)
    cached_pdf = get_pdf_cached(pdf_cache_key) if pdf_cache_key else None

    if cached_pdf is not None:
        print(f"[transformer] 📕 PDF 命中缓存 (hash={pdf_cache_key[:12]}...)")
        document_contents.append(cached_pdf)
        return

    # 未命中，执行完整解析
    try:
        pdf_data_url = f"data:application/pdf;base64,{source_data}"
        save_result = save_base64_to_local(pdf_data_url, preferred_filename=filename)
        if save_result:
            pdf_local_path, _ = save_result
            user_text = " ".join(text_parts)
            doc_content = analyze_pdf(pdf_local_path, user_text)
            if doc_content and pdf_cache_key:
                put_pdf_cache(pdf_cache_key, doc_content)
                print(f"[transformer] 📕 PDF 已缓存 (hash={pdf_cache_key[:12]}...)")
            if doc_content:
                document_contents.append(doc_content)
    except Exception as e:
        print(f"[transformer] ⚠️ PDF 解析失败: {e}")


# ============================================================
# 组装辅助函数
# ============================================================

def _build_visible_text(text_parts: list[str], summary_parts: list[str]) -> str:
    """组装前端可见的纯文本内容。"""
    parts: list[str] = []

    if text_parts:
        parts.append(" ".join(t for t in text_parts if t.strip()))
    elif summary_parts:
        parts.append("[已上传: " + ", ".join(summary_parts) + "]")

    return "\n".join(parts)


def _build_model_context(image_descriptions: list[str], document_contents: list[str]) -> str:
    """组装模型专用的上下文内容（将通过标记注入 content）。"""
    all_parts = list(image_descriptions) + list(document_contents)
    return "\n".join(all_parts).strip()


def _assemble_message(
    original_msg: HumanMessage,
    visible_text: str,
    model_context: str,
    attachment_metadata: list[dict],
) -> HumanMessage:
    """
    组装最终的 HumanMessage 对象（与原版 core/hatch_agent.py 行为完全一致）。

    输出结构：
      final_content = visible_text + [标记包裹的model_context]

    前端渲染时：
      - 显示 visible_text 部分
      - 自动剥离 <!-- __HATCH_AGENT_INTERNAL_START__ --> ... <!-- END --> 之间的内容
      - 从 additional_kwargs.attachments 渲染缩略图/文件卡片
    """
    # ---- 组装最终内容：用户可见 + 模型专用（标记包裹） ----
    final_parts: list[str] = []

    if visible_text:
        final_parts.append(visible_text)

    if model_context:
        # 用特殊标记包裹模型专用数据，前端会自动隐藏
        final_parts.append(
            _MODEL_DATA_MARKER_START
            + model_context
            + _MODEL_DATA_MARKER_END
        )

    final_content = "\n".join(final_parts)

    new_msg = HumanMessage(content=final_content)

    # 保留原始元数据
    if hasattr(original_msg, "id") and original_msg.id:
        new_msg.id = original_msg.id
    if hasattr(original_msg, "name") and original_msg.name:
        new_msg.name = original_msg.name
    if hasattr(original_msg, "response_metadata") and original_msg.response_metadata:
        new_msg.response_metadata = original_msg.response_metadata

    # 附件元信息 → additional_kwargs（供前端渲染缩略图/文件卡片）
    # 注：模型上下文已通过 _MODEL_DATA_MARKER 标记注入到 content 中
    if attachment_metadata:
        new_msg.additional_kwargs["attachments"] = attachment_metadata

    return new_msg
