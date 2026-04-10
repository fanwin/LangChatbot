import { Message } from "@langchain/langgraph-sdk";
import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
// FIXME  MC80OmFIVnBZMlhvaklQb3RvVTZlbFZPY1E9PTo3Y2RiZWM1OQ==

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** 图片/文件附件内容块（用于多模态消息渲染） */
export interface FileAttachment {
  type: "image" | "file";
  mimeType: string;
  url?: string;           // image 的 data URL
  base64Data?: string;    // file 的 base64 数据
  filename?: string;      // 文件名
}

/**
 * 后端 hatch_agent 中间件用于包裹模型专用数据的 HTML 注释标记。
 * 前端渲染时需自动剥离此标记之间的内容，避免泄露中间分析数据到 UI。
 */
const MODEL_INTERNAL_MARKER_START = /<!--\s*__HATCH_AGENT_INTERNAL_START__\s*-->\s*/g;
const MODEL_INTERNAL_MARKER_END = /<!--\s*__HATCH_AGENT_INTERNAL_END__\s*-->/g;

/**
 * 从消息 content 中剥离后端注入的模型专用数据（PDF 分析报告、图片描述等），
 * 仅返回用户可见的原始输入文字。
 *
 * 后端用 HTML 注释标记包裹中间数据：
 *   用户文字 <!-- __HATCH_AGENT_INTERNAL_START__ --> 模型专用数据 <!-- __HATCH_AGENT_INTERNAL_END__ -->
 *
 * 此函数检测并移除标记之间的所有内容。
 */
function stripModelInternalData(text: string): string {
  // 使用正则直接匹配并移除整个标记块（含标记本身）
  const markerPattern = /<!--\s*__HATCH_AGENT_INTERNAL_START__\s*-->[\s\S]*?<!--\s*__HATCH_AGENT_INTERNAL_END__\s*-->/g;
  return text.replace(markerPattern, "").trim();
}

export function extractStringFromMessageContent(message: Message): string {
  const raw = typeof message.content === "string"
    ? message.content
    : Array.isArray(message.content)
    ? message.content
        .filter(
          (c: unknown) =>
            (typeof c === "object" &&
              c !== null &&
              "type" in c &&
              (c as { type: string }).type === "text") ||
            typeof c === "string"
        )
        .map((c: unknown) =>
          typeof c === "string"
            ? c
            : typeof c === "object" && c !== null && "text" in c
            ? (c as { text?: string }).text || ""
            : ""
        )
        .join("")
    : "";

  // 剥离后端注入的模型专用中间数据（仅保留用户原始输入）
  return stripModelInternalData(raw);
}
// NOTE  MS80OmFIVnBZMlhvaklQb3RvVTZlbFZPY1E9PTo3Y2RiZWM1OQ==

/**
 * 从消息内容中提取图片和文件附件（用于 UI 渲染）
 * 支持:
 *   - { type: "image_url", image_url: { url: "data:image/..." } }
 *   - { type: "file", source_media_type: "...", source_data: "...", filename: "..." }
 */
export function extractFileAttachments(message: Message): FileAttachment[] {
  const attachments: FileAttachment[] = [];

  // 优先从原始 content 数组中提取（消息尚未被后端转换时）
  if (Array.isArray(message.content)) {
    for (const part of message.content) {
      if (typeof part !== "object" || part === null || !("type" in part)) continue;

      const block = part as Record<string, any>;

      if (block.type === "image_url" && block.image_url?.url) {
        attachments.push({
          type: "image",
          mimeType: "",
          url: block.image_url.url,
        });
      }

      if (block.type === "file") {
        attachments.push({
          type: "file",
          mimeType: block.source_media_type || "application/octet-stream",
          base64Data: block.source_data,
          filename: block.filename || "attachment",
        });
      }
    }

    return attachments;
  }

  // content 已被后端转换为纯字符串时，
  // 从 additional_kwargs.attachments（后端中间件注入的元数据）中恢复
  const rawAttachments = (message as any).additional_kwargs?.attachments;
  if (Array.isArray(rawAttachments)) {
    for (const item of rawAttachments) {
      if (!item || typeof item !== "object") continue;
      if (item.type === "image" && item.url) {
        attachments.push({
          type: "image",
          mimeType: item.mimeType || "",
          url: item.url,
        });
      } else if (item.type === "file") {
        attachments.push({
          type: "file",
          mimeType: item.mimeType || "application/octet-stream",
          filename: item.filename || "attachment",
        });
      }
    }
  }

  return attachments;
}

export function extractSubAgentContent(data: unknown): string {
  if (typeof data === "string") {
    return data;
  }

  if (data && typeof data === "object") {
    const dataObj = data as Record<string, unknown>;

    // Try to extract description first
    if (dataObj.description && typeof dataObj.description === "string") {
      return dataObj.description;
    }

    // Then try prompt
    if (dataObj.prompt && typeof dataObj.prompt === "string") {
      return dataObj.prompt;
    }

    // For output objects, try result
    if (dataObj.result && typeof dataObj.result === "string") {
      return dataObj.result;
    }

    // Fallback to JSON stringification
    return JSON.stringify(data, null, 2);
  }

  // Fallback for any other type
  return JSON.stringify(data, null, 2);
}
// NOTE  Mi80OmFIVnBZMlhvaklQb3RvVTZlbFZPY1E9PTo3Y2RiZWM1OQ==

export function isPreparingToCallTaskTool(messages: Message[]): boolean {
  const lastMessage = messages[messages.length - 1];
  return (
    (lastMessage.type === "ai" &&
      lastMessage.tool_calls?.some(
        (call: { name?: string }) => call.name === "task"
      )) ||
    false
  );
}

export function formatMessageForLLM(message: Message): string {
  let role: string;
  if (message.type === "human") {
    role = "Human";
  } else if (message.type === "ai") {
    role = "Assistant";
  } else if (message.type === "tool") {
    role = `Tool Result`;
  } else {
    role = message.type || "Unknown";
  }

  const timestamp = message.id ? ` (${message.id.slice(0, 8)})` : "";

  let contentText = "";

  // Extract content text
  if (typeof message.content === "string") {
    contentText = message.content;
  } else if (Array.isArray(message.content)) {
    const textParts: string[] = [];

    message.content.forEach((part: any) => {
      if (typeof part === "string") {
        textParts.push(part);
      } else if (part && typeof part === "object" && part.type === "text") {
        textParts.push(part.text || "");
      }
      // Ignore other types like tool_use in content - we handle tool calls separately
    });

    contentText = textParts.join("\n\n").trim();
  }

  // For tool messages, include additional tool metadata
  if (message.type === "tool") {
    const toolName = (message as any).name || "unknown_tool";
    const toolCallId = (message as any).tool_call_id || "";
    role = `Tool Result [${toolName}]`;
    if (toolCallId) {
      role += ` (call_id: ${toolCallId.slice(0, 8)})`;
    }
  }

  // Handle tool calls from .tool_calls property (for AI messages)
  const toolCallsText: string[] = [];
  if (
    message.type === "ai" &&
    message.tool_calls &&
    Array.isArray(message.tool_calls) &&
    message.tool_calls.length > 0
  ) {
    message.tool_calls.forEach((call: any) => {
      const toolName = call.name || "unknown_tool";
      const toolArgs = call.args ? JSON.stringify(call.args, null, 2) : "{}";
      toolCallsText.push(`[Tool Call: ${toolName}]\nArguments: ${toolArgs}`);
    });
  }

  // Combine content and tool calls
  const parts: string[] = [];
  if (contentText) {
    parts.push(contentText);
  }
  if (toolCallsText.length > 0) {
    parts.push(...toolCallsText);
  }

  if (parts.length === 0) {
    return `${role}${timestamp}: [Empty message]`;
  }

  if (parts.length === 1) {
    return `${role}${timestamp}: ${parts[0]}`;
  }

  return `${role}${timestamp}:\n${parts.join("\n\n")}`;
}

export function formatConversationForLLM(messages: Message[]): string {
  const formattedMessages = messages.map(formatMessageForLLM);
  return formattedMessages.join("\n\n---\n\n");
}
// NOTE  My80OmFIVnBZMlhvaklQb3RvVTZlbFZPY1E9PTo3Y2RiZWM1OQ==
