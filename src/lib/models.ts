import type { AgentMode, ModeModelSettings, ModelRef, ThinkingLevel } from "../types";

export type ModelId =
  | "anthropic:claude-opus-4-7"
  | "anthropic:claude-opus-4-6"
  | "anthropic:claude-sonnet-4-6"
  | "anthropic:claude-haiku-4-5"
  | "openai:gpt-5.5"
  | "openai:gpt-5.4"
  | "openai:gpt-5.4-mini"
  | "openai:gpt-5.3-codex"
  | "openai:gpt-5.3-codex-spark"
  | "openai:gpt-5.2"
  | "google:gemini-3.1-pro-preview"
  | "kimi:kimi-for-coding";
export type ProviderId = "anthropic" | "openai" | "google" | "kimi";
export type ModeModelSelection = { model: ModelId; thinking: ThinkingLevel };
export type ModeModelSelections = Record<AgentMode, ModeModelSelection>;

export const PROVIDERS: {
  value: ProviderId;
  label: string;
  icon: string;
}[] = [
  {
    value: "anthropic",
    label: "Anthropic",
    icon: "simple-icons:anthropic",
  },
  {
    value: "openai",
    label: "OpenAI",
    icon: "simple-icons:openai",
  },
  {
    value: "google",
    label: "Google",
    icon: "simple-icons:google",
  },
  {
    value: "kimi",
    label: "Kimi",
    icon: "local:kimi",
  },
];

export const THINKING_LEVELS: { value: ThinkingLevel; label: string }[] = [
  { value: "off", label: "Off" },
  { value: "low", label: "Low" },
  { value: "medium", label: "Medium" },
  { value: "high", label: "High" },
  { value: "xhigh", label: "XHigh" },
  { value: "max", label: "Max" },
];

export const MODELS: {
  value: ModelId;
  provider: ProviderId;
  label: string;
  thinking: readonly ThinkingLevel[];
  defaultThinking: ThinkingLevel;
}[] = [
  {
    value: "anthropic:claude-opus-4-7",
    provider: "anthropic",
    label: "Opus 4.7",
    thinking: ["off", "low", "medium", "high", "xhigh", "max"],
    defaultThinking: "medium",
  },
  {
    value: "anthropic:claude-opus-4-6",
    provider: "anthropic",
    label: "Opus 4.6",
    thinking: ["off", "low", "medium", "high", "max"],
    defaultThinking: "medium",
  },
  {
    value: "anthropic:claude-sonnet-4-6",
    provider: "anthropic",
    label: "Sonnet 4.6",
    thinking: ["off", "low", "medium", "high", "max"],
    defaultThinking: "medium",
  },
  {
    value: "anthropic:claude-haiku-4-5",
    provider: "anthropic",
    label: "Haiku 4.5",
    thinking: ["off", "low", "medium", "high"],
    defaultThinking: "medium",
  },
  {
    value: "openai:gpt-5.5",
    provider: "openai",
    label: "GPT-5.5",
    thinking: ["off", "low", "medium", "high", "xhigh"],
    defaultThinking: "medium",
  },
  {
    value: "openai:gpt-5.4",
    provider: "openai",
    label: "GPT-5.4",
    thinking: ["off", "low", "medium", "high", "xhigh"],
    defaultThinking: "medium",
  },
  {
    value: "openai:gpt-5.4-mini",
    provider: "openai",
    label: "GPT-5.4 Mini",
    thinking: ["off", "low", "medium", "high", "xhigh"],
    defaultThinking: "medium",
  },
  {
    value: "openai:gpt-5.3-codex",
    provider: "openai",
    label: "GPT-5.3 Codex",
    thinking: ["off", "low", "medium", "high", "xhigh"],
    defaultThinking: "medium",
  },
  {
    value: "openai:gpt-5.3-codex-spark",
    provider: "openai",
    label: "GPT-5.3 Codex Spark",
    thinking: ["low", "medium", "high", "xhigh"],
    defaultThinking: "low",
  },
  {
    value: "openai:gpt-5.2",
    provider: "openai",
    label: "GPT-5.2",
    thinking: ["off", "low", "medium", "high", "xhigh"],
    defaultThinking: "medium",
  },
  {
    value: "google:gemini-3.1-pro-preview",
    provider: "google",
    label: "Gemini 3.1 Pro",
    thinking: ["low", "medium", "high"],
    defaultThinking: "medium",
  },
  {
    value: "kimi:kimi-for-coding",
    provider: "kimi",
    label: "Kimi 2.6",
    thinking: ["off", "high"],
    defaultThinking: "high",
  },
];

export type ModelEntry = (typeof MODELS)[number];

function isModelId(value: string): value is ModelId {
  return MODELS.some((model) => model.value === value);
}

export function modelIdFromRef(model: ModelRef | null | undefined): ModelId {
  if (model) {
    const id = `${model.provider}:${model.name}`;
    if (isModelId(id)) return id;
  }
  if (model?.provider === "openai") {
    return "openai:gpt-5.5";
  }
  if (model?.provider === "google") {
    return "google:gemini-3.1-pro-preview";
  }
  if (model?.provider === "kimi") {
    return "kimi:kimi-for-coding";
  }
  return "anthropic:claude-opus-4-7";
}

export function modelRefFromId(model: ModelId): ModelRef {
  const [provider, name] = model.split(":");
  return { provider, name };
}

export function thinkingFromRef(
  model: ModelRef | null | undefined,
): ThinkingLevel {
  if (model?.provider === "google") {
    if (
      model.effort === "low" ||
      model.effort === "medium" ||
      model.effort === "high"
    ) {
      return model.effort;
    }
    return "medium";
  }
  if (model?.provider === "kimi") {
    if (model.effort === "none") return "off";
    return "high";
  }
  if (
    model?.provider === "openai" &&
    model.name === "gpt-5.3-codex-spark" &&
    model.effort === "none"
  ) {
    return "low";
  }
  if (model?.effort === "none") return "off";
  if (model?.effort === "xhigh") return "xhigh";
  if (model?.provider === "openai" && model.effort === "max") return "xhigh";
  if (
    model?.effort === "low" ||
    model?.effort === "medium" ||
    model?.effort === "high" ||
    model?.effort === "max"
  ) {
    return model.effort;
  }
  return "medium";
}

export function modelRefWithThinking(
  model: ModelRef,
  thinking: ThinkingLevel,
): ModelRef {
  if (
    model.provider === "openai" &&
    model.name === "gpt-5.3-codex-spark" &&
    thinking === "off"
  ) {
    return { ...model, effort: "low" };
  }
  if (thinking === "off") return { ...model, effort: "none" };
  if (model.provider === "kimi") return { ...model, effort: "high" };
  return { ...model, effort: thinking };
}

export function selectionFromRef(
  model: ModelRef | null | undefined,
): ModeModelSelection {
  return {
    model: modelIdFromRef(model),
    thinking: thinkingFromRef(model),
  };
}

export function selectionsFromSettings(
  settings: ModeModelSettings | null | undefined,
  fallback: ModelRef,
): ModeModelSelections {
  return {
    act: selectionFromRef(settings?.act ?? fallback),
    plan: selectionFromRef(settings?.plan ?? fallback),
    goal: selectionFromRef(settings?.goal ?? settings?.act ?? fallback),
  };
}
