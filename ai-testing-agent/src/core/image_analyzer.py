"""
图片分析模块 - Vision 模型图片内容理解

使用多模态 Vision 模型（豆包 doubao / OpenAI GPT-4o-mini）分析图片内容，
将图片转为详细文字描述，使纯文本大模型也能"理解"图片。

模型优先级：
  1. DOUBAO (火山引擎豆包) — 默认配置的多模态模型
  2. OPENAI (GPT-4o-mini)     — 回退方案

使用方式：
    from examples.image_analyzer import analyze_image

    description = analyze_image("/path/to/image.png", user_text="这张图是什么？")
    print(description)
"""

import os
import base64
from typing import Optional

from langchain_core.messages import HumanMessage


# ============================================================
# 分析提示词模板
# ============================================================

DEFAULT_ANALYSIS_PROMPT = """请仔细分析这张图片，用中文输出以下信息：

1. **图片类型**: （照片、截图、文档扫描、图表、手绘图等）
2. **主要内容**: 详细描述图片中的所有可见元素
3. **关键细节**: 文字内容、数字、颜色等可识别的具体信息
4. **上下文推断**: 如果是截图或文档，说明其可能的用途

{user_context}

请直接输出分析结果，不需要额外格式。"""


# ============================================================
# 公共接口
# ============================================================

def analyze_image(image_path: str, user_text: str = "") -> Optional[str]:
    """
    使用 Vision 模型分析图片内容。

    按优先级尝试不同的 Vision 提供商，全部失败时返回占位提示。

    Args:
        image_path: 本地图片文件的绝对路径
        user_text: 用户同时发送的文字消息（会追加到提示词中）

    Returns:
        图片内容的文字描述；所有模型不可用时返回兜底提示
    """
    # 构建提示词
    user_context = (
        f"\n用户的附加问题或说明：\"{user_text}\""
        if user_text.strip() else ""
    )
    prompt = DEFAULT_ANALYSIS_PROMPT.format(user_context=user_context)

    # 按优先级尝试各提供商
    providers = [
        ("doubao", _analyze_with_doubao),
        ("openai", _analyze_with_openai),
    ]

    for provider_name, analyze_fn in providers:
        try:
            print(f"[image_analyzer] 尝试使用 {provider_name} Vision 模型...")
            description = analyze_fn(image_path, prompt)
            if description and description.strip():
                print(f"[image_analyzer] ✅ {provider_name} 分析成功 "
                      f"(长度: {len(description)} 字符)")
                return description
        except Exception as e:
            print(f"[image_analyzer] ⚠️ {provider_name} 失败: {e}")
            continue

    # 所有模型均不可用
    print("[image_analyzer] ❌ 所有 Vision 模型不可用")
    return _build_fallback(user_text)


# ============================================================
# 各提供商实现
# ============================================================

def _analyze_with_doubao(image_path: str, prompt: str) -> Optional[str]:
    """使用豆包（doubao）多模态模型分析图片。"""
    from langchain_openai import ChatOpenAI
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

    response = model.invoke([
        HumanMessage(content=_build_image_message_content(image_path, prompt))
    ])
    return response.content


def _analyze_with_openai(image_path: str, prompt: str) -> Optional[str]:
    """使用 OpenAI GPT-4o-mini 分析图片。"""
    from langchain_openai import ChatOpenAI
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

    response = model.invoke([
        HumanMessage(content=_build_image_message_content(image_path, prompt))
    ])
    return response.content


# ============================================================
# 内部工具函数
# ============================================================

def _build_image_message_content(image_path: str, prompt: str) -> list[dict]:
    """
    构建 LangChain 多模态消息 content（包含图片 + 文字）。

    Args:
        image_path: 本地图片文件路径
        prompt: 文字提示词

    Returns:
        LangChain 格式的多模态 content 列表
    """
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    # 根据扩展名推断 MIME 类型
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(ext, "image/png")

    return [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{img_b64}"},
        },
    ]


def _build_fallback(user_text: str) -> str:
    """构建所有 Vision 模型不可用时的兜底响应。"""
    return (
        "[用户上传了一张图片附件]"
        + (f"，附带文字：「{user_text}」" if user_text.strip() else "")
        + "\n（注：当前环境无可用的视觉分析模型。"
        "请配置 DOUBAO 或 OPENAI API Key 以启用图片理解功能。）"
    )
