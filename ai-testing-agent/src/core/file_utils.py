"""
文件工具模块 - Base64 解码与本地文件保存

提供从前端上传的 base64 编码数据中提取并保存为本地文件的通用能力。
支持所有常见 MIME 类型的自动扩展名推断。

使用方式：
    from examples.file_utils import save_base64_to_local, extract_base64_from_data_url

    # 保存 data URL 到本地
    filepath, mime_type = save_base64_to_local(data_url, preferred_filename="photo.png")

    # 仅提取（不保存）
    mime, raw_bytes = extract_base64_from_data_url(data_url)
"""

import os
import re
import base64
import tempfile
import uuid
from typing import Any, Optional


# ============================================================
# MIME 类型 → 文件扩展名 映射表
# ============================================================

MIME_EXT_MAP: dict[str, str] = {
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
    # 纯文本 / 结构化数据
    "text/plain": ".txt",
    "application/json": ".json",
    "text/csv": ".csv",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    # 音频
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    # 视频
    "video/mp4": ".mp4",
}

# 默认保存目录
_DEFAULT_SAVE_DIR = os.path.join(tempfile.gettempdir(), "agent_files")


# ============================================================
# Data URL 解析
# ============================================================

def extract_base64_from_data_url(data_url: str) -> Optional[tuple[str, bytes]]:
    """
    从 Data URL (data:image/png;base64,xxxxx) 中提取 MIME 类型和原始二进制数据。

    Args:
        data_url: 完整的 Data URL 字符串

    Returns:
        (mime_type, binary_data) 元组；解析失败返回 None

    示例：
        >>> result = extract_base64_from_data_url("data:image/png;base64,iVBOR...")
        >>> mime, data = result
        >>> print(mime)  # "image/png"
    """
    if not data_url or not data_url.startswith("data:"):
        return None

    try:
        match = re.match(r"data:([^;]+);base64,(.+)", data_url, re.DOTALL)
        if not match:
            return None

        mime_type = match.group(1).strip()
        b64_data = match.group(2)

        # 处理 URL 安全的 base64 变体
        b64_data = b64_data.replace("-", "+").replace("_", "/")
        padding = len(b64_data) % 4
        if padding:
            b64_data += "=" * (4 - padding)

        raw_bytes = base64.b64decode(b64_data)
        return mime_type, raw_bytes
    except Exception as e:
        print(f"[file_utils] 解析 Data URL 失败: {e}")
        return None


# ============================================================
# 本地文件保存
# ============================================================

def save_base64_to_local(
    data_url: str,
    save_dir: Optional[str] = None,
    preferred_filename: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """
    将 base64 编码的数据解码并保存为本地文件。

    通用版文件保存函数，支持所有 MIME 类型：
      - 自动根据 MIME 推断扩展名
      - 支持通过 preferred_filename 覆盖扩展名推断
      - 使用 UUID 生成唯一文件名，避免冲突

    Args:
        data_url: Data URL 格式字符串 (data:mime/type;base64,xxxx)
        save_dir: 保存目录（默认系统临时目录下的 agent_files）
        preferred_filename: 优先使用的文件名（用于推断扩展名）

    Returns:
        (filepath, mime_type) 元组；失败返回 None
    """
    result = extract_base64_from_data_url(data_url)
    if result is None:
        return None

    mime_type, raw_bytes = result

    ext = MIME_EXT_MAP.get(mime_type.lower(), ".bin")

    # 如果提供了优先文件名，尝试从中提取扩展名
    if preferred_filename:
        name_ext = os.path.splitext(preferred_filename)[1].lower()
        if name_ext:
            ext = name_ext

    # 确定保存目录
    target_dir = save_dir or _DEFAULT_SAVE_DIR
    os.makedirs(target_dir, exist_ok=True)

    # UUID 唯一文件名
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(target_dir, filename)

    try:
        with open(filepath, "wb") as f:
            f.write(raw_bytes)
        print(f"[file_utils] 📄 已保存: {filepath} "
              f"(MIME: {mime_type}, {len(raw_bytes)} bytes)")
        return filepath, mime_type
    except Exception as e:
        print(f"[file_utils] 保存文件失败: {e}")
        return None


def save_base64_image_to_local(data_url: str, **kw: Any) -> Optional[str]:
    """
    向后兼容的便捷函数：仅返回文件路径。

    Args:
        data_url: Data URL 字符串
        **kw: 传递给 save_base64_to_local 的额外参数

    Returns:
        文件绝对路径；失败返回 None
    """
    result = save_base64_to_local(data_url, **kw)
    return result[0] if result else None


def ensure_save_dir(save_dir: Optional[str] = None) -> str:
    """
    确保保存目录存在，返回目录路径。

    Args:
        save_dir: 指定目录路径（None 则使用默认值）

    Returns:
        目录绝对路径
    """
    target = save_dir or _DEFAULT_SAVE_DIR
    os.makedirs(target, exist_ok=True)
    return target
