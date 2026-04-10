export interface StandaloneConfig {
  deploymentUrl: string;
  assistantId: string;
  langsmithApiKey?: string;
  // 多模态模型配置
  useMultimodalModel?: boolean;
  multimodalModelName?: string;
}

// 预设的多模态大模型列表
export const MULTIModal_MODELS = [
  { id: "gpt-4o", name: "GPT-4o", description: "OpenAI 多模态旗舰模型" },
  { id: "gpt-4o-mini", name: "GPT-4o Mini", description: "OpenAI 轻量多模态模型" },
  { id: "gemini-2.0-flash", name: "Gemini 2.0 Flash", description: "Google 多模态快速模型" },
  { id: "claude-sonnet-4-20250514", name: "Claude Sonnet 4", description: "Anthropic 多模态模型" },
  { id: "qwen-vl-max", name: "通义千问 VL Max", description: "阿里多模态视觉语言模型" },
] as const;

export type MultimodalModelId = (typeof MULTIModal_MODELS)[number]["id"];
//  MC8yOmFIVnBZMlhvaklQb3RvVTZhVlpFTWc9PTo3YTQ1NzQzYw==

const CONFIG_KEY = "deep-agent-config";

export function getConfig(): StandaloneConfig | null {
  if (typeof window === "undefined") return null;

  const stored = localStorage.getItem(CONFIG_KEY);
  if (!stored) return null;

  try {
    return JSON.parse(stored);
  } catch {
    return null;
  }
}

export function saveConfig(config: StandaloneConfig): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(CONFIG_KEY, JSON.stringify(config));
}
// FIXME  MS8yOmFIVnBZMlhvaklQb3RvVTZhVlpFTWc9PTo3YTQ1NzQzYw==
