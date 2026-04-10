"""
Agent 中间件模块 - before_model / after_model 钩子

提供大模型执行前后的拦截能力：
  - check_message_flow (before_model): 多模态内容检测与转换入口
  - log_response (after_model): 模型响应日志记录

多模态模式切换：
  ┌─ OFF（默认）→ 调用 transform_multimodal_message → 纯文本进入 deepseek
  └─ ON          → 直接传给豆包多模态模型 → 原生理解图片/PDF

使用方式：
    from examples.middleware import check_message_flow, log_response
"""

from typing import Any

from langchain.agents.middleware import before_model, after_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from src.core.message_transformer import transform_multimodal_message


# ============================================================
# 多模态直通调用（豆包原生处理）
# ============================================================

def _call_doubao_multimodal(message: HumanMessage) -> Any | None:
    """
    调用豆包多模态模型，直接处理包含图片/文件附件的原始消息。

    与 image_analyzer 的区别：
      - image_analyzer: 分析单张图 → 返回描述文字（供文本模型）
      - 此函数: 处理完整多模态消息 → 返回最终回答
    """
    from dotenv import load_dotenv
    load_dotenv()

    import os
    api_key = os.getenv("DOUBAO_API_KEY", "")
    if not api_key:
        raise ValueError("DOUBAO_API_KEY 未配置")

    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(
        api_key=api_key,
        model=os.getenv("DOUBAO_MODEL_NAME", "doubao-seed-2-0-pro-260215"),
        base_url=os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        max_tokens=4096,
    )

    response = model.invoke([message])
    result = response.content
    return result if isinstance(result, str) else str(result) if result else None


# ============================================================
# 附件元信息提取（轻量级，不读取数据体）
# ============================================================

def extract_attachment_metadata(content: list) -> list[dict]:
    """从多模态 content 中提取附件元信息（URL/文件名/MIME）。"""
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


# ============================================================
# 核心中间件：before_model
# ============================================================

@before_model(can_jump_to=["end"])
def check_message_flow(state: dict, runtime: Runtime) -> dict[str, Any] | None:
    """
    大模型执行前的中间件 —— 多模态内容检测与路由。

    决策逻辑：
      1. 检测最后一条 HumanMessage 是否含有多模态 content (list)
      2. 如果是：
         - 多模态模式 ON  → _call_doubao_multimodal() → 写入 AI 回复 → goto end
         - 多模态模式 OFF → transform_multimodal_message() → 替换为纯文本 → 继续传递
      3. 如果否 → 直接传递给模型
    """
    print("=" * 60)
    print("🔄 [before_model] 大模型运作前的预处理...")
    print("=" * 60)
    print(f"消息总数: {len(state['messages'])}")

    # ---- 读取多模态模式开关 ----
    use_multimodal_mode = False
    try:
        cfg = getattr(runtime, "config", None) or {}
        configurable = cfg.get("configurable", {}) if isinstance(cfg, dict) else {}
        use_multimodal_mode = bool(configurable.get("use_multimodal_model", False))
    except Exception:
        pass

    mode_label = (
        "✨ 多模态直通（豆包原生）"
        if use_multimodal_mode
        else "📝 文本降级（Vision + deepseek）"
    )
    print(f"🎛️ 模式: {mode_label}")

    # ---- 检测最新用户消息 ----
    if state["messages"]:
        last_msg = state["messages"][-1]
        print(f"用户输入: {str(last_msg.content)[:200]}...")

        if isinstance(last_msg, HumanMessage):
            content = last_msg.content

            is_multimodal = (
                isinstance(content, list)
                and any(
                    isinstance(p, dict) and p.get("type") in ("image_url", "file")
                    for p in content if isinstance(p, dict)
                )
            )

            if is_multimodal:
                if use_multimodal_mode:
                    return _handle_multimodal_mode_on(last_msg, content, state)
                else:
                    _handle_multimodal_mode_off(state, last_msg)
            else:
                print("\nℹ️ [纯文本] 无需转换\n")

    return None


def _handle_multimodal_mode_on(last_msg: HumanMessage, content: list, state: dict) -> dict:
    """分支 A：多模态模式 ON — 豆包原生处理。"""
    print("\n✨ [多模态直通] 调用豆包原生处理...\n")

    attachment_metadata = extract_attachment_metadata(content)
    if attachment_metadata:
        last_msg.additional_kwargs["attachments"] = attachment_metadata

    try:
        response = _call_doubao_multimodal(last_msg)
        if response:
            print(f"✨ [豆包响应成功] 长度: {len(response)} 字符")
            # 将豆包响应写入 state 作为 AI 回复
            ai_msg = AIMessage(content=response)
            state["messages"].append(ai_msg)
            return {"command": {"goto": "__end__", "update": None}}
        else:
            print("⚠️ [豆包为空] 降级...")
            raise RuntimeError("empty response")
    except Exception as e:
        print(f"⚠️ [豆包异常] {e}，降级...")
        # 降级到文本转换流程
        transformed = transform_multimodal_message(last_msg)
        state["messages"][-1] = transformed
        print("✅ [降级完成] 已转为纯文本，继续传递给模型\n")


def _handle_multimodal_mode_off(state: dict, last_msg: HumanMessage) -> None:
    """分支 B：多模态模式 OFF — Vision + 文本降级。"""
    print("\n🔍 [检测到多模态] 开始 图片→文本 转换...")
    transformed = transform_multimodal_message(last_msg)
    state["messages"][-1] = transformed
    print("✅ [转换完成] 已替换为纯文本\n")


# ============================================================
# 辅助中间件：after_model
# ============================================================

@after_model
def log_response(state: dict, runtime: Runtime) -> dict[str, Any] | None:
    """大模型执行后打印响应摘要。"""
    print("-" * 40)
    last_content = state["messages"][-1].content if state["messages"] else ""
    print(f"[after_model] 响应: {str(last_content)[:200]}...")
    print("-" * 40)
    return None
