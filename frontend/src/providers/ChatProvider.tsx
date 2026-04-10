"use client";

import { ReactNode, createContext, useContext, useState } from "react";
import { Assistant } from "@langchain/langgraph-sdk";
import { type StateType, useChat } from "@/app/hooks/useChat";
import type { UseStreamThread } from "@langchain/langgraph-sdk/react";

interface ChatProviderProps {
  children: ReactNode;
  activeAssistant: Assistant | null;
  onHistoryRevalidate?: () => void;
  thread?: UseStreamThread<StateType>;
}
// TODO  MC8yOmFIVnBZMlhvaklQb3RvVTZla1Y0Ymc9PTpmZjk4YjdiMA==

export function ChatProvider({
  children,
  activeAssistant,
  onHistoryRevalidate,
  thread,
}: ChatProviderProps) {
  // 多模态大模型状态（全局共享）
  const [useMultimodalModel, setUseMultimodalModel] = useState(() => {
    if (typeof window === "undefined") return false;
    try {
      const config = JSON.parse(localStorage.getItem("deep-agent-config") || "{}");
      return config.useMultimodalModel || false;
    } catch {
      return false;
    }
  });

  const chat = useChat({ 
    activeAssistant, 
    onHistoryRevalidate, 
    thread,
    useMultimodalModel,
  });
  
  return (
    <ChatContext.Provider value={{ ...chat, useMultimodalModel, setUseMultimodalModel }}>
      {children}
    </ChatContext.Provider>
  );
}

export type ChatContextType = ReturnType<typeof useChat> & {
  useMultimodalModel: boolean;
  setUseMultimodalModel: (value: boolean) => void;
};

export const ChatContext = createContext<ChatContextType | undefined>(
  undefined
);

export function useChatContext() {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error("useChatContext must be used within a ChatProvider");
  }
  return context;
}
// TODO  MS8yOmFIVnBZMlhvaklQb3RvVTZla1Y0Ymc9PTpmZjk4YjdiMA==
