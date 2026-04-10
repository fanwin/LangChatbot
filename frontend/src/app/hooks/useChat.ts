"use client";
// TODO  MC80OmFIVnBZMlhvaklQb3RvVTZVbUYyU1E9PTpkMTNkMGU5NA==

import { useCallback } from "react";
import { useStream } from "@langchain/langgraph-sdk/react";
import {
  type Message,
  type Assistant,
  type Checkpoint,
} from "@langchain/langgraph-sdk";
import { v4 as uuidv4 } from "uuid";
import type { UseStreamThread } from "@langchain/langgraph-sdk/react";
import type { TodoItem } from "@/app/types/types";
import { useClient } from "@/providers/ClientProvider";
import { useQueryState } from "nuqs";
// eslint-disable  MS80OmFIVnBZMlhvaklQb3RvVTZVbUYyU1E9PTpkMTNkMGU5NA==

export type StateType = {
  messages: Message[];
  todos: TodoItem[];
  files: Record<string, string>;
  email?: {
    id?: string;
    subject?: string;
    page_content?: string;
  };
  ui?: any;
};
// FIXME  Mi80OmFIVnBZMlhvaklQb3RvVTZVbUYyU1E9PTpkMTNkMGU5NA==

export function useChat({
  activeAssistant,
  onHistoryRevalidate,
  thread,
  useMultimodalModel = false,
}: {
  activeAssistant: Assistant | null;
  onHistoryRevalidate?: () => void;
  thread?: UseStreamThread<StateType>;
  useMultimodalModel?: boolean;
}) {
  // Helper: convert File to base64 data URL
  const fileToBase64 = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });

  const [threadId, setThreadId] = useQueryState("threadId");
  const client = useClient();

  const stream = useStream<StateType>({
    assistantId: activeAssistant?.assistant_id || "",
    client: client ?? undefined,
    reconnectOnMount: true,
    threadId: threadId ?? null,
    onThreadId: setThreadId,
    defaultHeaders: { "x-auth-scheme": "langsmith" },
    // Enable fetching state history when switching to existing threads
    fetchStateHistory: true,
    // Revalidate thread list when stream finishes, errors, or creates new thread
    onFinish: onHistoryRevalidate,
    onError: onHistoryRevalidate,
    onCreated: onHistoryRevalidate,
    experimental_thread: thread,
  });

  const sendMessage = useCallback(
    async (content: string, attachedFiles?: File[]) => {
      // Build multimodal content if files are attached
      let messageContent: Message["content"];

      if (attachedFiles && attachedFiles.length > 0) {
        const contentParts: Array<Record<string, unknown>> = [];

        // 始终添加文本部分（即使为空）
        if (content.trim()) {
          contentParts.push({ type: "text", text: content });
        }

        for (const file of attachedFiles) {
          try {
            const dataUrl = await fileToBase64(file);
            const mimeType = file.type || "application/octet-stream";

            if (mimeType.startsWith("image/")) {
              // 图片：使用 OpenAI 风格的 image_url 格式
              contentParts.push({
                type: "image_url" as const,
                image_url: { url: dataUrl },
              });
            } else {
              // PDF / 音频等文件：使用 base64 数据块格式
              const base64Data = dataUrl.includes(",") ? dataUrl.split(",")[1] : dataUrl;
              contentParts.push({
                type: "file" as const,
                source_media_type: mimeType,
                source_data: base64Data,
                filename: file.name || "attachment",
              });
            }
          } catch (err) {
            console.error(`Failed to process file "${file.name}":`, err);
            // 文件处理失败时添加文本提示
            contentParts.push({
              type: "text" as const,
              text: `[附件加载失败: ${file.name}]`,
            });
          }
        }

        messageContent = contentParts as Message["content"];
      } else {
        messageContent = content;
      }

      const newMessage: Message = {
        id: uuidv4(),
        type: "human",
        content: messageContent,
      };

      stream.submit(
        { messages: [newMessage] },
        {
          optimisticValues: (prev) => ({
            messages: [...(prev.messages ?? []), newMessage],
          }),
          config: {
            ...(activeAssistant?.config ?? {}),
            recursion_limit: 100,
            // 将多模态开关状态传给后端，后端中间件据此决定是否跳过多模态→文本转换
            configurable: {
              use_multimodal_model: useMultimodalModel,
            },
          },
        }
      );
      // 发送消息后立即刷新线程列表
      onHistoryRevalidate?.();
    },
    [stream, activeAssistant?.config, onHistoryRevalidate, fileToBase64]
  );

  const runSingleStep = useCallback(
    (
      messages: Message[],
      checkpoint?: Checkpoint,
      isRerunningSubagent?: boolean,
      optimisticMessages?: Message[]
    ) => {
      if (checkpoint) {
        stream.submit(undefined, {
          ...(optimisticMessages
            ? { optimisticValues: { messages: optimisticMessages } }
            : {}),
          config: activeAssistant?.config,
          checkpoint: checkpoint,
          ...(isRerunningSubagent
            ? { interruptAfter: ["tools"] }
            : { interruptBefore: ["tools"] }),
        });
      } else {
        stream.submit(
          { messages },
          { config: activeAssistant?.config, interruptBefore: ["tools"] }
        );
      }
    },
    [stream, activeAssistant?.config]
  );

  const setFiles = useCallback(
    async (files: Record<string, string>) => {
      if (!threadId) return;
      // TODO: missing a way how to revalidate the internal state
      // I think we do want to have the ability to externally manage the state
      await client.threads.updateState(threadId, { values: { files } });
    },
    [client, threadId]
  );

  const continueStream = useCallback(
    (hasTaskToolCall?: boolean) => {
      stream.submit(undefined, {
        config: {
          ...(activeAssistant?.config || {}),
          recursion_limit: 100,
        },
        ...(hasTaskToolCall
          ? { interruptAfter: ["tools"] }
          : { interruptBefore: ["tools"] }),
      });
      // Update thread list when continuing stream
      onHistoryRevalidate?.();
    },
    [stream, activeAssistant?.config, onHistoryRevalidate]
  );

  const markCurrentThreadAsResolved = useCallback(() => {
    stream.submit(null, { command: { goto: "__end__", update: null } });
    // Update thread list when marking thread as resolved
    onHistoryRevalidate?.();
  }, [stream, onHistoryRevalidate]);

  const resumeInterrupt = useCallback(
    (value: any) => {
      stream.submit(null, { command: { resume: value } });
      // Update thread list when resuming from interrupt
      onHistoryRevalidate?.();
    },
    [stream, onHistoryRevalidate]
  );

  const stopStream = useCallback(() => {
    stream.stop();
  }, [stream]);

  return {
    stream,
    todos: stream.values.todos ?? [],
    files: stream.values.files ?? {},
    email: stream.values.email,
    ui: stream.values.ui,
    setFiles,
    messages: stream.messages,
    isLoading: stream.isLoading,
    isThreadLoading: stream.isThreadLoading,
    interrupt: stream.interrupt,
    getMessagesMetadata: stream.getMessagesMetadata,
    sendMessage,
    runSingleStep,
    continueStream,
    stopStream,
    markCurrentThreadAsResolved,
    resumeInterrupt,
    // 多模态模型状态
    useMultimodalModel,
  };
}
// eslint-disable  My80OmFIVnBZMlhvaklQb3RvVTZVbUYyU1E9PTpkMTNkMGU5NA==
