"""
Hatch Agent - 支持非多模态大模型处理图片/PDF文档的核心 Agent 模块

核心机制：
  在 before_model 中间件中拦截带图片/PDF的 HumanMessage，
  将 base64 编码的数据解析并转换为文字描述，
  替换原消息中的多模态内容为纯文本，使任何文本模型都能"理解"附件。

支持的附件类型：
  - 图片 (image/png, image/jpeg, image/gif, image/webp ...) → Vision模型分析
  - PDF文档 (application/pdf) → PyMuPDF4LLM + LLMImageBlobParser 解析（含图片识别）
  - 其他文件 (音频、视频等) → 记录元信息

转换链路：
  前端上传图片/PDF → base64 data URL → 后端中间件检测 image_url / file block
    → 解码 base64 → 保存本地临时文件
      → 图片：调用 Vision 模型分析
      → PDF：PyMuPDF4LLMLoader 提取文字 + LLMImageBlobParser 识别内嵌图片
        → 用描述文字替换原内容 → 纯文本消息进入非多模态模型
"""

import os
import re
import base64
import tempfile
import uuid
import hashlib
from collections import OrderedDict
from typing import Any, Optional

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from utils.model_factory import GetModelByVendor


# ============================================================
# 附件处理结果缓存（避免同一文件重复调用 Vision 模型）
# ============================================================

# 最大缓存条目数（LRU 淘汰策略）
_MAX_CACHE_SIZE = 128

# LRU 缓存字典：{hash_key: cached_result}
_image_cache: OrderedDict[str, str] = OrderedDict()
_pdf_cache: OrderedDict[str, str] = OrderedDict()


def _compute_content_hash(data_url: str) -> Optional[str]:
    """
    从 data URL 或纯 base64 字符串中提取 base64 payload 并计算 MD5 哈希。

    同一文件无论文件名如何变化，只要内容一致就会产生相同的哈希值，
    从而命中缓存。

    Args:
        data_url: data URL (data:xxx;base64,...) 或纯 base64 字符串

    Returns:
        MD5 哈希字符串；解析失败返回 None
    """
    if not data_url or not data_url.strip():
        return None

    try:
        # 提取 base64 payload 部分
        if data_url.startswith("data:"):
            match = re.match(r"data:[^;]+;base64,(.+)", data_url, re.DOTALL)
            if not match:
                return None
            b64_payload = match.group(1)
        else:
            b64_payload = data_url.strip()

        # 标准化（处理 URL 安全变体）
        b64_payload = b64_payload.replace("-", "+").replace("_", "/")
        padding = len(b64_payload) % 4
        if padding:
            b64_payload += "=" * (4 - padding)

        return hashlib.md5(b64_payload.encode()).hexdigest()
    except Exception as e:
        print(f"[hatch_agent] 计算内容哈希失败: {e}")
        return None


def _lru_get(cache: OrderedDict[str, str], key: str) -> Optional[str]:
    """从 LRU 缓存中获取值。命中时将条目移到末尾（最近使用）。"""
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


def _lru_put(cache: OrderedDict[str, str], key: str, value: str) -> None:
    """写入 LRU 缓存，超出容量时淘汰最久未使用的条目。"""
    if key in cache:
        cache.move_to_end(key)
        cache[key] = value
    else:
        cache[key] = value
        while len(cache) > _MAX_CACHE_SIZE:
            evicted_key, _ = cache.popitem(last=False)
            print(f"[hatch_agent] 🗑️ 缓存已满，淘汰旧条目: {evicted_key[:12]}...")


def get_cache_stats() -> dict[str, int]:
    """返回当前缓存统计信息（用于调试/监控）。"""
    return {
        "image_cache_size": len(_image_cache),
        "pdf_cache_size": len(_pdf_cache),
        "max_cache_size": _MAX_CACHE_SIZE,
    }


def clear_attachment_cache() -> None:
    """清空所有附件处理缓存。"""
    _image_cache.clear()
    _pdf_cache.clear()
    print("[hatch_agent] 🧹 附件处理缓存已全部清空")


# ============================================================
# 图片处理工具函数
# ============================================================

def extract_base64_from_data_url(data_url: str) -> Optional[tuple[str, bytes]]:
    """
    从 data URL (data:image/png;base64,xxxxx) 中提取 MIME 类型和原始二进制数据。

    Args:
        data_url: 完整的 data URL 字符串

    Returns:
        (mime_type, binary_data) 元组；解析失败返回 None
    """
    if not data_url or not data_url.startswith("data:"):
        return None

    try:
        # 分离 header 和 base64 数据部分
        # 格式: data:image/png;base64,iVBORw0KGgo...
        match = re.match(r"data:([^;]+);base64,(.+)", data_url, re.DOTALL)
        if not match:
            return None

        mime_type = match.group(1).strip()
        b64_data = match.group(2)

        # 处理 URL 安全的 base64 变体（替换 - 和 +）
        b64_data = b64_data.replace("-", "+").replace("_", "/")
        # 补齐 padding
        padding = len(b64_data) % 4
        if padding:
            b64_data += "=" * (4 - padding)

        raw_bytes = base64.b64decode(b64_data)
        return mime_type, raw_bytes
    except Exception as e:
        print(f"[hatch_agent] 解析 data URL 失败: {e}")
        return None


def save_base64_to_local(
    data_url: str,
    save_dir: Optional[str] = None,
    preferred_filename: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """
    将 base64 编码的任意文件数据（图片/PDF/其他）解码并保存为本地文件。

    通用版文件保存函数，支持所有 MIME 类型：
      - 图片：自动推断扩展名
      - PDF：使用 .pdf 扩展名
      - 其他：根据 MIME 类型或自定义文件名确定扩展名

    Args:
        data_url: data URL 格式的字符串 (data:mime/type;base64,xxxx)
        save_dir: 保存目录（默认为系统临时目录下的 agent_files）
        preferred_filename: 优先使用的文件名（如前端传入的 filename）

    Returns:
        (filepath, mime_type) 元组；失败返回 None
    """
    result = extract_base64_from_data_url(data_url)
    if result is None:
        return None

    mime_type, raw_bytes = result

    # 完整的 MIME → 扩展名映射表（覆盖常见图片 + 文档类型）
    mime_ext_map: dict[str, str] = {
        # 图片类
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/svg+xml": ".svg",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        # 文档类
        "application/pdf": ".pdf",
        # 其他常见格式
        "text/plain": ".txt",
        "application/json": ".json",
        "text/csv": ".csv",
        "application/msword": ".doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "video/mp4": ".mp4",
    }
    ext = mime_ext_map.get(mime_type.lower(), ".bin")

    # 如果提供了优先文件名，尝试从中提取扩展名
    if preferred_filename:
        name_ext = os.path.splitext(preferred_filename)[1].lower()
        if name_ext:
            ext = name_ext

    # 确定保存目录
    if save_dir is None:
        save_dir = os.path.join(tempfile.gettempdir(), "agent_files")
    os.makedirs(save_dir, exist_ok=True)

    # 使用 UUID 生成唯一文件名，避免冲突
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(save_dir, filename)

    try:
        with open(filepath, "wb") as f:
            f.write(raw_bytes)
        print(f"[hatch_agent] 📄 文件已保存到本地: {filepath} "
              f"(MIME: {mime_type}, 大小: {len(raw_bytes)} bytes)")
        return filepath, mime_type
    except Exception as e:
        print(f"[hatch_agent] 保存文件失败: {e}")
        return None


# 向后兼容的别名（旧代码仍可用）
def save_base64_image_to_local(data_url: str, **kw: Any) -> Optional[str]:
    """向后兼容包装：仅返回 filepath（丢弃 mime_type）。"""
    result = save_base64_to_local(data_url, **kw)
    return result[0] if result else None


def analyze_image_with_vision_model(image_path: str, user_text: str = "") -> Optional[str]:
    """
    使用视觉模型分析图片内容，生成文字描述。

    这是让非多模态模型"理解"图片的关键降级方案：
    调用一个支持图片输入的轻量模型（如豆包 doubao / 通义千问 VL / GPT-4o-mini），
    将图片内容转为详细文字描述，然后注入到消息中供纯文本模型使用。

    优先级策略：
      1. 尝试调用 DOUBAO（豆包，默认配置的多模态模型）
      2. 若豆包未配置，回退到 OPENAI GPT-4o-mini
      3. 若都不可用，返回基础占位提示

    Args:
        image_path: 本地图片文件的绝对路径
        user_text: 用户同时发送的文字消息（如有）

    Returns:
        图片内容的文字描述；失败返回 None
    """
    # ---- 构建用户提示词 ----
    prompt_template = """请仔细分析这张图片，用中文输出以下信息：

1. **图片类型**: （照片、截图、文档扫描、图表、手绘图等）
2. **主要内容**: 详细描述图片中的所有可见元素
3. **关键细节**: 文字内容、数字、颜色等可识别的具体信息
4. **上下文推断**: 如果是截图或文档，说明其可能的用途

{user_context}

请直接输出分析结果，不需要额外格式。"""

    user_context = f"\n用户的附加问题或说明：\"{user_text}\"" if user_text.strip() else ""
    prompt = prompt_template.format(user_context=user_context)

    # ---- 尝试不同的 Vision 模型提供商 ----
    vision_providers = [
        ("doubao", _analyze_with_doubao),
        ("openai", _analyze_with_openai),
    ]

    for provider_name, analyze_fn in vision_providers:
        try:
            print(f"[hatch_agent] 尝试使用 {provider_name} Vision 模型分析图片...")
            description = analyze_fn(image_path, prompt)
            if description and description.strip():
                print(f"[hatch_agent] ✅ {provider_name} 图片分析成功，描述长度: {len(description)} 字符")
                return description
        except Exception as e:
            print(f"[hatch_agent] ⚠️ {provider_name} 分析失败: {e}")
            continue

    # 所有 Vision 模型均不可用时的兜底
    print("[hatch_agent] ❌ 所有 Vision 模型均不可用，返回基础占位提示")
    fallback = (
        "[用户上传了一张图片附件]"
        + (f"，附带文字：「{user_text}」" if user_text.strip() else "")
        + "\n（注：当前环境无可用的视觉分析模型，无法提取图片内容。"
        "请配置 DOUBAO 或 OPENAI API Key 以启用图片理解功能。）"
    )
    return fallback


def _analyze_with_doubao(image_path: str, prompt: str) -> Optional[str]:
    """使用豆包（doubao）多模态模型分析图片。"""
    from langchain_openai import ChatOpenAI
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("DOUBAO_API_KEY", "")
    if not api_key:
        raise ValueError("DOUBAO_API_KEY 未配置")

    model = ChatOpenAI(
        api_key=api_key,
        model=os.getenv("DOUBAO_MODEL_NAME", "doubao-seed-2-0-lite-260215"),
        base_url=os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        max_tokens=2048,
    )

    # 读取图片并构建 base64 data URL
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    # 推断 MIME 类型
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/png")

    message_content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{img_b64}"},
        },
    ]

    response = model.invoke([HumanMessage(content=message_content)])
    return response.content


def _analyze_with_openai(image_path: str, prompt: str) -> Optional[str]:
    """使用 OpenAI GPT-4o-mini 分析图片。"""
    from langchain_openai import ChatOpenAI
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 未配置")

    model = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
        max_tokens=2048,
    )

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/png")

    message_content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{img_b64}"},
        },
    ]

    response = model.invoke([HumanMessage(content=message_content)])
    return response.content


# ============================================================
# PDF 文档解析工具函数
# ============================================================

def analyze_pdf_document(pdf_path: str, user_text: str = "") -> Optional[str]:
    """
    使用 PyMuPDF4LLMLoader + LLMImageBlobParser 解析 PDF 文档内容。

    核心能力：
      1. 提取 PDF 中的所有文本内容（保留页面结构）
      2. 自动识别并提取 PDF 中嵌入的图片
      3. 使用 Vision 模型（豆包/OpenAI）对嵌入图片进行 OCR/理解
      4. 将文本+图片描述合并为完整的文档摘要

    依赖：
      - PyMuPDF4LLM (langchain-pymupdf4llm)
      - LLMImageBlobParser (langchain_community.document_loaders.parsers)
      - Vision 模型客户端 (通过 GetModelByVendor 获取)

    Args:
        pdf_path: 本地 PDF 文件的绝对路径
        user_text: 用户同时发送的文字消息（如有）

    Returns:
        PDF 文档的完整文字提取结果；失败返回 None
    """
    print(f"[hatch_agent] 📕 开始解析 PDF 文档: {pdf_path}")

    try:
        # ---- 动态导入（避免未安装依赖时启动失败）----
        from langchain_community.document_loaders.parsers import LLMImageBlobParser
        from langchain_pymupdf4llm import PyMuPDF4LLMLoader
        from utils.model_factory import GetModelByVendor

        # ---- 获取 Vision 模型客户端（用于解析 PDF 内嵌图片）----
        # 优先使用 doubao，其次 openai
        vision_model_client = _get_vision_model_client()

        # ---- 配置 Loader ----
        loader = PyMuPDF4LLMLoader(
            pdf_path,
            mode="page",          # 按页返回，保留页面结构信息
            extract_images=True,  # 启用图片提取
            images_parser=LLMImageBlobParser(model=vision_model_client),
        )

        # ---- 执行加载和解析 ----
        documents = loader.load()

        if not documents:
            print("[hatch_agent] ⚠️ PDF 解析完成但未提取到任何内容")
            fallback = (
                f"\n📎 **PDF 文件附件**: {os.path.basename(pdf_path)}\n"
                "> 注：该文件可能是扫描版或加密 PDF，无法自动提取文本内容。"
                "请尝试上传清晰的照片版本。"
            )
            return fallback

        # ---- 组装最终输出 ----
        output_parts: list[str] = []

        output_parts.append(f"📕 **PDF 文档分析报告**")
        output_parts.append(f"文件名：{os.path.basename(pdf_path)}")
        output_parts.append(f"总页数：{len(documents)} 页\n")

        for idx, doc in enumerate(documents, start=1):
            page_content = doc.page_content.strip()
            page_meta = doc.metadata or {}

            output_parts.append(f"--- **第 {idx} 页** ---")

            if page_content:
                output_parts.append(page_content)
            else:
                output_parts.append("（本页无文字内容）")

            # 输出页码等元信息（如有）
            if page_meta.get("page"):
                output_parts.append(f"[页码: {page_meta.get('page')}]")
            output_parts.append("")  # 空行分隔

        final_output = "\n".join(output_parts)

        # 追加用户问题上下文（如果有）
        if user_text.strip():
            final_output += (
                f"\n{'=' * 40}\n"
                f"💬 **用户针对此文档的问题**：{user_text}"
            )

        print(f"[hatch_agent] ✅ PDF 解析成功！共 {len(documents)} 页，"
              f"总长度 {len(final_output)} 字符")
        return final_output

    except ImportError as e:
        missing_pkg = str(e).split("'")[-2] if "'" in str(e) else "未知包"
        print(f"[hatch_agent] ❌ 缺少依赖包: {missing_pkg}")
        print("[hatch_agent]   请执行: pip install langchain-pymupdf4llm langchain-community")
        return _fallback_pdf_response(pdf_path, user_text, reason=f"缺少依赖 {missing_pkg}")

    except Exception as e:
        print(f"[hatch_agent] ❌ PDF 解析失败: {e}")
        import traceback
        traceback.print_exc()
        return _fallback_pdf_response(pdf_path, user_text, reason=str(e))


def _get_vision_model_client():
    """
    获取用于 PDF 图片解析的 Vision 模型客户端。
    优先级：doubao > openai > 抛出异常
    """
    from utils.model_factory import GetModelByVendor
    import os
    from dotenv import load_dotenv
    load_dotenv()

    # 优先级 1: doubao
    if os.getenv("DOUBAO_API_KEY"):
        try:
            return GetModelByVendor("doubao").generate_model_client()
        except Exception as e:
            print(f"[hatch_agent] ⚠️ doubao 客户端创建失败: {e}")

    # 优先级 2: openai
    if os.getenv("OPENAI_API_KEY"):
        try:
            return GetModelByVendor("openai").generate_model_client()
        except Exception as e:
            print(f"[hatch_agent] ⚠️ openai 客户端创建失败: {e}")

    raise ValueError(
        "无法创建 Vision 模型客户端。"
        "请至少配置 DOUBAO_API_KEY 或 OPENAI_API_KEY 以启用 PDF 内嵌图片解析功能。"
    )


def _fallback_pdf_response(pdf_path: str, user_text: str, reason: str = "") -> str:
    """
    PDF 解析完全失败时的兜底响应。

    尝试使用纯 PyMuPDF（不带 LLM 图片解析）进行基础文本提取，
    如果仍然失败则返回元信息占位符。
    """
    filename = os.path.basename(pdf_path) if pdf_path else "未知文件"

    # 尝试用基础方式读取 PDF 文本
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text_parts: list[str] = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
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
                "\n\n> 注：基础模式仅提取了文字内容，PDF 内嵌的图表/截图未被分析。"
                f"完整解析需要安装: pip install langchain-pymupdf4llm"
            )
            print(f"[hatch_agent] ✅ 基础模式 PDF 提取成功 ({len(text_parts)} 页有内容)")
            return basic_result
    except Exception as e:
        print(f"[hatch_agent] 基础模式 PDF 提取也失败了: {e}")

    # 彻底失败的兜底
    parts = [
        f"📎 **文件附件**: {filename}",
        f"> 该文件未能被自动解析。",
    ]
    if reason:
        parts[0] += f"\n> 失败原因: {reason}"
    parts.append(
        "> 可能的原因：文件损坏、格式不受支持、或缺少必要的解析库。\n"
        "> 请确认已安装: pip install langchain-pymupdf4llm pymupdf"
    )
    if user_text.strip():
        parts.append(f"\n💬 用户附加消息: \"{user_text}\"")

    return "\n".join(parts)


def _call_doubao_multimodal(message: HumanMessage) -> Optional[str]:
    """
    调用豆包多模态模型，直接处理包含图片/文件附件的原始多模态消息。

    与 _analyze_with_doubao 不同：
      - _analyze_with_doubao: 只分析一张图片，返回描述文字（供文本模型用）
      - _call_doubao_multimodal: 处理完整的多模态消息，返回最终回答

    Args:
        message: 包含多模态 content (image_url / file blocks) 的 HumanMessage

    Returns:
        豆包模型的最终回复文字；失败返回 None
    """
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("DOUBAO_API_KEY", "")
    if not api_key:
        raise ValueError("DOUBAO_API_KEY 未配置，无法调用多模态模型")

    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(
        api_key=api_key,
        model=os.getenv("DOUBAO_MODEL_NAME", "doubao-seed-2-0-lite-260215"),
        base_url=os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        max_tokens=4096,
    )

    # 直接将原始多模态 content 传给豆包
    # content 格式: [{type:"text", text:"..."}, {type:"image_url", ...}, {type:"file", ...}]
    response = model.invoke([message])

    result = response.content
    return result if isinstance(result, str) else str(result) if result else None


# ============================================================
# 多模态内容转换器
# ============================================================

# 前后端约定的标记：用于分隔「用户可见文字」和「模型专用分析数据」
# 前端渲染时会自动剥离此标记之间的内容
_MODEL_DATA_MARKER_START = "\n<!-- __HATCH_AGENT_INTERNAL_START__ -->\n"
_MODEL_DATA_MARKER_END = "\n<!-- __HATCH_AGENT_INTERNAL_END__ -->\n"


def transform_multimodal_message(message: HumanMessage) -> HumanMessage:
    """
    核心转换函数：将包含图片/PDF等多模态内容的 HumanMessage
    转换为纯文本格式的 HumanMessage。

    输出格式（两层结构）：
      ┌─ 用户可见层（前端直接展示）── 用户输入的文字 + 简洁附件提示
      │
      └─ 模型专用层（前端自动隐藏）── 图片/PDF 完整分析报告（供模型理解）

    Args:
        message: 可能包含多模态 content 的 HumanMessage

    Returns:
        content 为纯文本字符串的新 HumanMessage，
        其中模型专用数据被 _MODEL_DATA_MARKER 包裹
    """
    content = message.content

    # 情况 1：已经是纯文本，无需处理
    if isinstance(content, str):
        return message

    # 情况 2：数组形式的多模态内容
    if not isinstance(content, list):
        return message

    text_parts: list[str] = []
    image_descriptions: list[str] = []
    document_contents: list[str] = []   # PDF 等文档的提取结果

    # 收集附件元信息（用于生成简洁的用户可见提示）
    attachment_summary_parts: list[str] = []

    for part in content:
        if isinstance(part, str):
            text_parts.append(part)
            continue

        if not isinstance(part, dict) or "type" not in part:
            continue

        part_type = part["type"]

        # --- 文本块 ---
        if part_type == "text" and part.get("text"):
            text_parts.append(str(part["text"]))

        # --- 图片块（Vision 模型分析 + 缓存） ---
        elif part_type == "image_url":
            image_url = part.get("image_url", {})
            url_data = image_url.get("url", "") if isinstance(image_url, dict) else ""

            if url_data:
                print("[hatch_agent] 📷 检测到图片附件，开始解析...")
                attachment_summary_parts.append("📷 图片")

                # ---- 查询图片分析缓存 ----
                cache_key = _compute_content_hash(url_data)
                cached_desc = _lru_get(_image_cache, cache_key) if cache_key else None

                if cached_desc is not None:
                    # 命中缓存，直接复用，跳过 Vision 调用
                    print(f"[hatch_agent] 📷 图片命中缓存 (hash: {cache_key[:12]}...)，"
                          f"跳过重复 Vision 分析")
                    image_descriptions.append(f"\n📷 **图片内容分析**：\n{cached_desc}")
                else:
                    # 未命中，执行 Vision 分析并写入缓存
                    local_path = save_base64_image_to_local(url_data)
                    if local_path:
                        user_text = " ".join(text_parts)
                        desc = analyze_image_with_vision_model(local_path, user_text)
                        if desc and cache_key:
                            _lru_put(_image_cache, cache_key, desc)
                            print(f"[hatch_agent] 📷 图片分析结果已缓存 (hash: {cache_key[:12]}...)")
                        if desc:
                            image_descriptions.append(f"\n📷 **图片内容分析**：\n{desc}")

        # --- 文件块（PDF / 音频 / 其他） ---
        elif part_type == "file":
            filename = part.get("filename", "未知文件")
            media_type = (part.get("source_media_type") or "").lower()
            source_data = part.get("source_data", "")

            print(f"[hatch_agent] 📎 检测到文件附件: {filename} "
                  f"(MIME: {media_type})")

            # ---- PDF 文件：完整解析链路 ----
            is_pdf = (
                media_type == "application/pdf"
                or filename.lower().endswith(".pdf")
            )

            if is_pdf:
                attachment_summary_parts.append(f"📕 {filename}")

            if is_pdf and source_data:
                print("[hatch_agent] 📕 检测到 PDF，启动完整文档解析...")

                # ---- 查询 PDF 解析缓存 ----
                pdf_cache_key = _compute_content_hash(source_data)
                cached_pdf = _lru_get(_pdf_cache, pdf_cache_key) if pdf_cache_key else None

                if cached_pdf is not None:
                    # 命中缓存，直接复用
                    print(f"[hatch_agent] 📕 PDF 命中缓存 (hash: {pdf_cache_key[:12]}...)，"
                          f"跳过重复解析")
                    document_contents.append(cached_pdf)
                    continue

                # 未命中，执行完整解析并写入缓存
                try:
                    pdf_data_url = f"data:application/pdf;base64,{source_data}"
                    save_result = save_base64_to_local(
                        pdf_data_url,
                        preferred_filename=filename,
                    )
                    if save_result:
                        pdf_local_path, _ = save_result
                        user_text = " ".join(text_parts)
                        doc_content = analyze_pdf_document(pdf_local_path, user_text)
                        if doc_content and pdf_cache_key:
                            _lru_put(_pdf_cache, pdf_cache_key, doc_content)
                            print(f"[hatch_agent] 📕 PDF 解析结果已缓存 (hash: {pdf_cache_key[:12]}...)")
                        if doc_content:
                            document_contents.append(doc_content)
                            continue
                except Exception as e:
                    print(f"[hatch_agent] ⚠️ PDF 完整解析失败，降级为元信息模式: {e}")

            # ---- 非PDF 或 PDF 解析失败的兜底 ----
            type_label = {
                "application/pdf": "PDF 文档",
                "audio/mpeg": "音频",
                "audio/wav": "音频",
                "video/mp4": "视频",
            }.get(media_type, "文件")

            if not is_pdf:
                attachment_summary_parts.append(f"📎 {filename}")

            file_info = f"\n📎 **{type_label}附件**: {filename}"
            if media_type:
                file_info += f" (类型: {media_type})"
            file_info += "\n> 注：该文件未能被自动解析内容。"
            document_contents.append(file_info)

    # ===== 组装最终内容：用户可见 + 模型专用（标记包裹） =====
    visible_parts: list[str] = []
    model_only_parts: list[str] = []

    # --- 提取附件元信息（供前端渲染缩略图/文件卡片） ---
    attachment_metadata: list[dict] = []

    for part in content:
        if not isinstance(part, dict) or "type" not in part:
            continue
        pt = part["type"]
        if pt == "image_url":
            img_url = part.get("image_url", {})
            url_val = img_url.get("url", "") if isinstance(img_url, dict) else ""
            if url_val:
                attachment_metadata.append({
                    "type": "image",
                    "url": url_val,
                })
        elif pt == "file":
            attachment_metadata.append({
                "type": "file",
                "mimeType": part.get("source_media_type", "application/octet-stream"),
                "filename": part.get("filename", "attachment"),
                "source_data": part.get("source_data", "")[:200],  # 仅保留前缀用于标识
            })

    # 用户原始文字 → 可见层
    if text_parts:
        combined_text = " ".join(t for t in text_parts if t.strip())
        visible_parts.append(combined_text)

    # 附件简洁提示 → 可见层（仅当没有用户文字时显示）
    if attachment_summary_parts and not text_parts:
        visible_parts.append("[已上传: " + ", ".join(attachment_summary_parts) + "]")

    # 图片/PDF 完整分析 → 模型专用层（标记包裹）
    if image_descriptions:
        model_only_parts.extend(image_descriptions)
    if document_contents:
        model_only_parts.extend(document_contents)

    # 如果没有任何有效内容
    if not visible_parts and not model_only_parts:
        visible_parts.append("[用户发送了一条空消息]")

    # 组装最终字符串
    final_parts: list[str] = []

    if visible_parts:
        final_parts.append("\n".join(visible_parts))

    if model_only_parts:
        # 用特殊标记包裹模型专用数据，前端会自动隐藏
        final_parts.append(
            _MODEL_DATA_MARKER_START
            + "\n".join(model_only_parts).strip()
            + _MODEL_DATA_MARKER_END
        )

    final_content = "\n".join(final_parts)

    print(f"[hatch_agent] ✅ 多模态→纯文本转换完成")
    print(f"[hatch_agent]   用户可见部分: {len(visible_parts)} 段")
    print(f"[hatch_agent]   模型专用部分: {len(model_only_parts)} 段 "
          f"(共 {sum(len(p) for p in model_only_parts)} 字符)")

    # 返回新的 HumanMessage（保持其他 metadata 不变）
    new_msg = HumanMessage(content=final_content)
    if hasattr(message, "id") and message.id:
        new_msg.id = message.id  # type: ignore
    if hasattr(message, "name") and message.name:
        new_msg.name = message.name  # type: ignore
    if hasattr(message, "response_metadata") and message.response_metadata:
        new_msg.response_metadata = message.response_metadata  # type: ignore

    # 将原始附件元信息存入 additional_kwargs，供前端渲染缩略图/文件卡片
    if attachment_metadata:
        new_msg.additional_kwargs["attachments"] = attachment_metadata  # type: ignore

    return new_msg


# ============================================================
# Agent 中间件定义
# ============================================================


def _extract_attachment_metadata(content: list) -> list[dict]:
    """
    从多模态 content 数组中提取附件元信息（用于前端渲染），
    不做任何内容转换。

    仅提取 URL/文件名/MIME 等轻量信息，不读取 base64 数据体。
    """
    metadata: list[dict] = []
    for part in content:
        if not isinstance(part, dict) or "type" not in part:
            continue
        pt = part["type"]
        if pt == "image_url":
            img_url = part.get("image_url", {})
            url_val = img_url.get("url", "") if isinstance(img_url, dict) else ""
            if url_val:
                metadata.append({"type": "image", "url": url_val})
        elif pt == "file":
            metadata.append({
                "type": "file",
                "mimeType": part.get("source_media_type", "application/octet-stream"),
                "filename": part.get("filename", "attachment"),
            })
    return metadata

@before_model(can_jump_to=["end"])
def check_message_flow(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    大模型执行前的中间件 —— 多模态内容转换入口

    核心职责：
      1. 打印调试日志（消息数量、最近对话）
      2. **检测最后一条 HumanMessage 是否含有多模态内容**
      3. 如果有图片/PDF 等附件，根据「多模态模式」决定处理方式：

         ┌─ 多模态模式 OFF（默认）── 调用 transform_multimodal_message()
         │   将图片/Vision分析 → PDF/文本提取 → 替换为纯文本
         │   使非多模态模型（deepseek）也能"理解"附件
         │
         └─ 多模态模式 ON（前端开关打开）── 不做任何转换！
             直接将原始 image_url / file content 原样传给豆包多模态模型
             让模型原生理解图片和文档内容

      4. 将（可能经过转换的）消息写回 state
    """
    print("=" * 60)
    print("🔄 [before_model 中间件启动] 大模型运作前的预处理...")
    print("=" * 60)
    print(f"当前对话完整记录长度: {len(state['messages'])} 条")
    print(f"HumanMessage 数量: {len([_ for _ in state['messages'] if isinstance(_, HumanMessage)])}")
    print(f"AIMessage 数量: {len([_ for _ in state['messages'] if isinstance(_, AIMessage)])}")

    # ---- 从运行时配置中读取前端传递的多模态模式开关 ----
    # 前端通过 stream.submit 的 config.configurable.use_multimodal_model 传入
    use_multimodal_mode = False
    try:
        cfg = getattr(runtime, "config", None) or {}
        configurable = cfg.get("configurable", {}) if isinstance(cfg, dict) else {}
        use_multimodal_mode = bool(configurable.get("use_multimodal_model", False))
    except Exception:
        pass

    mode_label = "✨ 多模态直通模式（原生豆包）" if use_multimodal_mode else "📝 文本降级模式（Vision+deepseek）"
    print(f"🎛️ 当前模式: {mode_label}")

    if state["messages"]:
        last_msg = state["messages"][-1]
        print(f"用户最新输入: \n\t{last_msg.content}")

        if len(state["messages"]) > 1:
            prev_msg = state["messages"][-2]
            print(f"上一轮回复: \n\t{prev_msg.content}")

        # ===== 关键：多模态内容检测与条件分支 =====
        if isinstance(last_msg, HumanMessage):
            content = last_msg.content

            # 判断是否为多模态内容（list 格式）
            is_multimodal = isinstance(content, list) and any(
                isinstance(p, dict) and p.get("type") in ("image_url", "file")
                for p in content if isinstance(p, dict)
            )

            if is_multimodal:
                if use_multimodal_mode:
                    # ===== 分支 A：多模态模式 ON =====
                    # 前端开启了多模态开关 → 调用豆包多模态模型原生处理图片/PDF
                    # 得到结果后直接跳到 end（跳过 deepseek 纯文本模型）
                    print("\n✨ [多模态直通模式] 检测到多模态消息，调用豆包原生处理...\n")

                    attachment_metadata = _extract_attachment_metadata(content)
                    if attachment_metadata:
                        last_msg.additional_kwargs["attachments"] = attachment_metadata  # type: ignore

                    try:
                        # 调用豆包多模态模型直接处理原始多模态消息
                        multimodal_response = _call_doubao_multimodal(last_msg)

                        if multimodal_response:
                            print(f"✨ [豆包响应成功] 长度: {len(multimodal_response)} 字符")
                            # 将豆包响应写入 state 作为 AI 回复，并跳转到 end
                            ai_msg = AIMessage(content=multimodal_response)
                            state["messages"].append(ai_msg)
                            # 返回特殊标记通知框架跳过后续模型调用，直接结束
                            return {"command": {"goto": "__end__", "update": None}}
                        else:
                            print("⚠️ [豆包响应为空] 降级走文本转换流程...")
                            # 豆包失败时降级：回退到分支 B 的文本转换逻辑
                            transformed = transform_multimodal_message(last_msg)
                            state["messages"][-1] = transformed
                            print("✅ [降级完成] 已转为纯文本，继续传递给 deepseek\n")

                    except Exception as e:
                        print(f"⚠️ [豆包处理异常] {e}，降级走文本转换流程...")
                        # 异常时同样降级到文本转换
                        transformed = transform_multimodal_message(last_msg)
                        state["messages"][-1] = transformed
                        print("✅ [降级完成] 已转为纯文本，继续传递给 deepseek\n")
                else:
                    # ===== 分支 B：多模态模式 OFF（原有逻辑，完全不变）=====
                    print("\n🔍 [检测到多模态消息] 开始执行 图片→文本 转换...")
                    transformed = transform_multimodal_message(last_msg)
                    # 用转换后的纯文本消息替换原始消息
                    state["messages"][-1] = transformed
                    print("✅ [多模态转换完成] 消息已替换为纯文本，继续传递给模型\n")
            else:
                print("\nℹ️ [纯文本消息] 无需转换，直接传递给模型\n")

    return None


@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """大模型执行后打印响应内容（用于调试）。"""
    print("-" * 40)
    print(f"[after_model] 模型响应: {state['messages'][-1].content[:200]}...")
    print("-" * 40)
    return None


# ============================================================
# 工具定义
# ============================================================

def get_weather_tool(city: str):
    """
    获取城市的天气

    :param city: 城市名称
    :return: 天气描述字符串
    """
    return city + " 万里无云."


# ============================================================
# Agent 实例创建
# ============================================================

agent = create_agent(
    model=GetModelByVendor().generate_model_client(),
    middleware=[check_message_flow],
    tools=[get_weather_tool],
    system_prompt="你是一名智能助手，请根据用户输入的指令，给出专业、准确、有帮助的回复。"
                  "你可以帮助用户完成各种任务，包括但不限于：回答问题、创作文本（如诗歌、文章等）、分析内容、提供建议等。"
                  "注意：如果问题和工具不匹配，请不要调用。"
)

# response = agent.invoke({"messages": [{"role": "user", "content": "今天上海的天气如何。"}]})

