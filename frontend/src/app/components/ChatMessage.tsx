"use client";
// TODO  MC80OmFIVnBZMlhvaklQb3RvVTZVRk5rTWc9PTphNjZhZWJhZQ==

import React, { useMemo, useState, useCallback } from "react";
import { SubAgentIndicator } from "@/app/components/SubAgentIndicator";
import { ToolCallBox } from "@/app/components/ToolCallBox";
import { MarkdownContent } from "@/app/components/MarkdownContent";
import type {
  SubAgent,
  ToolCall,
  ActionRequest,
  ReviewConfig,
} from "@/app/types/types";
import { Message } from "@langchain/langgraph-sdk";
import {
  extractSubAgentContent,
  extractStringFromMessageContent,
  extractFileAttachments,
} from "@/app/utils/utils";
import { cn } from "@/lib/utils";
import { User, Bot, FileText, Image as ImageIcon } from "lucide-react";

interface ChatMessageProps {
  message: Message;
  toolCalls: ToolCall[];
  isLoading?: boolean;
  actionRequestsMap?: Map<string, ActionRequest>;
  reviewConfigsMap?: Map<string, ReviewConfig>;
  ui?: any[];
  stream?: any;
  onResumeInterrupt?: (value: any) => void;
  graphId?: string;
}
// NOTE  Mi80OmFIVnBZMlhvaklQb3RvVTZVRk5rTWc9PTphNjZhZWJhZQ==

export const ChatMessage = React.memo<ChatMessageProps>(
  ({
    message,
    toolCalls,
    isLoading,
    actionRequestsMap,
    reviewConfigsMap,
    ui,
    stream,
    onResumeInterrupt,
    graphId,
  }) => {
    const isUser = message.type === "human";
    const messageContent = extractStringFromMessageContent(message);
    const hasContent = messageContent && messageContent.trim() !== "";
    const hasToolCalls = toolCalls.length > 0;

    // 提取用户消息中的图片和文件附件
    const fileAttachments = useMemo(() =>
      isUser ? extractFileAttachments(message) : [],
      [message, isUser]
    );
    const hasAttachments = fileAttachments.length > 0;
    const subAgents = useMemo(() => {
      return toolCalls
        .filter((toolCall: ToolCall) => {
          return (
            toolCall.name === "task" &&
            toolCall.args["subagent_type"] &&
            toolCall.args["subagent_type"] !== "" &&
            toolCall.args["subagent_type"] !== null
          );
        })
        .map((toolCall: ToolCall) => {
          const subagentType = (toolCall.args as Record<string, unknown>)[
            "subagent_type"
          ] as string;
          return {
            id: toolCall.id,
            name: toolCall.name,
            subAgentName: subagentType,
            input: toolCall.args,
            output: toolCall.result ? { result: toolCall.result } : undefined,
            status: toolCall.status,
          } as SubAgent;
        });
    }, [toolCalls]);

    const [expandedSubAgents, setExpandedSubAgents] = useState<
      Record<string, boolean>
    >({});
    const isSubAgentExpanded = useCallback(
      (id: string) => expandedSubAgents[id] ?? true,
      [expandedSubAgents]
    );
    const toggleSubAgent = useCallback((id: string) => {
      setExpandedSubAgents((prev) => ({
        ...prev,
        [id]: prev[id] === undefined ? false : !prev[id],
      }));
    }, []);

    return (
      <div
        className={cn(
          "flex w-full max-w-full overflow-x-hidden",
          isUser ? "flex-row-reverse" : "flex-row"
        )}
      >
        {/* Avatar */}
        <div className={cn(
          "mt-4 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full",
          isUser
            ? "ml-3 bg-gradient-to-br from-[#076699] to-[#054d73]"
            : "mr-3 bg-gradient-to-br from-[#2F6868] to-[#1c3c3c]"
        )}>
          {isUser ? (
            <User className="h-4 w-4 text-white" />
          ) : (
            <Bot className="h-4 w-4 text-white" />
          )}
        </div>

        <div
          className={cn(
            "min-w-0 max-w-full",
            isUser ? "max-w-[70%]" : "w-full"
          )}
        >
          {/* 用户消息：先显示附件预览，再显示文字 */}
          {isUser && hasAttachments && (
            <div className="mt-3 flex flex-col gap-2">
              {/* 图片类附件：直接展示缩略图 */}
              {fileAttachments.filter(a => a.type === "image").map((attachment, idx) => (
                <div key={`img-${idx}`} className="inline-block max-w-[260px] overflow-hidden rounded-lg border border-border/40 bg-accent/20">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={attachment.url}
                    alt="上传的图片"
                    className="max-h-[220px] w-full object-contain"
                    loading="lazy"
                  />
                </div>
              ))}
              {/* 文件类附件（PDF等）：显示文件卡片 */}
              {fileAttachments.filter(a => a.type === "file").map((attachment, idx) => {
                const isPdf = attachment.mimeType === "application/pdf"
                  || (attachment.filename || "").endsWith(".pdf");
                const isAudio = (attachment.mimeType || "").startsWith("audio/");
                return (
                  <div
                    key={`file-${idx}`}
                    className="flex items-center gap-2.5 rounded-lg border border-border/40 bg-accent/30 px-3 py-2"
                  >
                    <div className={cn(
                      "flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-md",
                      isPdf ? "bg-red-500/10 text-red-600" :
                      isAudio ? "bg-purple-500/10 text-purple-600" :
                      "bg-blue-500/10 text-blue-600"
                    )}>
                      {isPdf ? (
                        <FileText size={16} />
                      ) : isAudio ? (
                        <span className="text-xs font-bold">♪</span>
                      ) : (
                        <ImageIcon size={16} />
                      )}
                    </div>
                    <div className="min-w-0">
                      <p className="truncate text-xs font-medium text-foreground">
                        {attachment.filename || "附件"}
                      </p>
                      <p className="text-[11px] text-muted-foreground">
                        {isPdf ? "PDF 文档" : isAudio ? "音频文件" : attachment.mimeType || "文件"}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {hasContent && (
            <div className={cn("relative flex items-end gap-0")}>
              <div
                className={cn(
                  "mt-4 overflow-hidden break-words text-sm font-normal leading-[150%]",
                  isUser
                    ? "rounded-xl rounded-br-none border border-border px-3 py-2 text-foreground"
                    : "text-primary"
                )}
                style={
                  isUser
                    ? { backgroundColor: "var(--color-user-message-bg)" }
                    : undefined
                }
              >
                {isUser ? (
                  <p className="m-0 whitespace-pre-wrap break-words text-sm leading-relaxed">
                    {messageContent}
                  </p>
                ) : hasContent ? (
                  <MarkdownContent content={messageContent} />
                ) : null}
              </div>
            </div>
          )}
          {hasToolCalls && (
            <div className="mt-4 flex w-full flex-col">
              {toolCalls.map((toolCall: ToolCall) => {
                if (toolCall.name === "task") return null;
                const toolCallGenUiComponent = ui?.find(
                  (u) => u.metadata?.tool_call_id === toolCall.id
                );
                const actionRequest = actionRequestsMap?.get(toolCall.name);
                const reviewConfig = reviewConfigsMap?.get(toolCall.name);
                return (
                  <ToolCallBox
                    key={toolCall.id}
                    toolCall={toolCall}
                    uiComponent={toolCallGenUiComponent}
                    stream={stream}
                    graphId={graphId}
                    actionRequest={actionRequest}
                    reviewConfig={reviewConfig}
                    onResume={onResumeInterrupt}
                    isLoading={isLoading}
                  />
                );
              })}
            </div>
          )}
          {!isUser && subAgents.length > 0 && (
            <div className="flex w-fit max-w-full flex-col gap-4">
              {subAgents.map((subAgent) => (
                <div
                  key={subAgent.id}
                  className="flex w-full flex-col gap-2"
                >
                  <div className="flex items-end gap-2">
                    <div className="w-[calc(100%-100px)]">
                      <SubAgentIndicator
                        subAgent={subAgent}
                        onClick={() => toggleSubAgent(subAgent.id)}
                        isExpanded={isSubAgentExpanded(subAgent.id)}
                      />
                    </div>
                  </div>
                  {isSubAgentExpanded(subAgent.id) && (
                    <div className="w-full max-w-full">
                      <div className="bg-surface border-border-light rounded-md border p-4">
                        <h4 className="text-primary/70 mb-2 text-xs font-semibold uppercase tracking-wider">
                          输入
                        </h4>
                        <div className="mb-4">
                          <MarkdownContent
                            content={extractSubAgentContent(subAgent.input)}
                          />
                        </div>
                        {subAgent.output && (
                          <>
                            <h4 className="text-primary/70 mb-2 text-xs font-semibold uppercase tracking-wider">
                              输出
                            </h4>
                            <MarkdownContent
                              content={extractSubAgentContent(subAgent.output)}
                            />
                          </>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }
);

ChatMessage.displayName = "ChatMessage";
// NOTE  My80OmFIVnBZMlhvaklQb3RvVTZVRk5rTWc9PTphNjZhZWJhZQ==
