export const APPEARANCE_CHANGED_EVENT = "sinew:appearance-changed";

const APPEARANCE_STORAGE_KEY = "sinew.appearance.settings";
const LEGACY_MESSAGE_FONT_SIZE_STORAGE_KEY = "sinew.appearance.messageFontSize";

export const MIN_MESSAGE_FONT_SIZE = 11;
export const MAX_MESSAGE_FONT_SIZE = 20;
export const MIN_CODE_FONT_SIZE = 11;
export const MAX_CODE_FONT_SIZE = 17;
export const MIN_UI_SCALE = 12;
export const MAX_UI_SCALE = 15;

export type ThemeMode = "system" | "dark" | "light";
export type AccentColor = "blue" | "violet" | "green" | "orange" | "rose";
export type ChatDensity = "compact" | "standard" | "comfortable";
export type MessageLineHeight = "compact" | "normal" | "comfortable";
export type MessageWidth = "narrow" | "standard" | "wide" | "full";
export type MessageBubbleStyle = "minimal" | "soft" | "solid";
export type MessageMeta = "hidden" | "roles";
export type MarkdownDensity = "compact" | "normal" | "comfortable";
export type CodeFontFamily = "geist" | "system" | "jetbrains" | "fira";
export type LinkUnderline = "subtle" | "hover" | "always";
export type CornerRadius = "compact" | "default" | "round";

export type AppearanceSettings = {
  messageFontSize: number;
  lineHeight: MessageLineHeight;
  messageWidth: MessageWidth;
  chatDensity: ChatDensity;
  bubbleStyle: MessageBubbleStyle;
  messageMeta: MessageMeta;
  markdownDensity: MarkdownDensity;
  codeFontSize: number;
  codeFontFamily: CodeFontFamily;
  codeWrap: boolean;
  linkUnderline: LinkUnderline;
  themeMode: ThemeMode;
  accentColor: AccentColor;
  highContrast: boolean;
  reduceMotion: boolean;
  uiScale: number;
  cornerRadius: CornerRadius;
};

export const DEFAULT_APPEARANCE_SETTINGS: AppearanceSettings = {
  messageFontSize: 12,
  lineHeight: "normal",
  messageWidth: "full",
  chatDensity: "standard",
  bubbleStyle: "minimal",
  messageMeta: "hidden",
  markdownDensity: "normal",
  codeFontSize: 12,
  codeFontFamily: "geist",
  codeWrap: true,
  linkUnderline: "subtle",
  themeMode: "dark",
  accentColor: "blue",
  highContrast: false,
  reduceMotion: false,
  uiScale: 13,
  cornerRadius: "default",
};

const LINE_HEIGHT_VALUE: Record<MessageLineHeight, string> = {
  compact: "1.38",
  normal: "1.5",
  comfortable: "1.66",
};

const MESSAGE_WIDTH_VALUE: Record<MessageWidth, string> = {
  narrow: "62ch",
  standard: "78ch",
  wide: "96ch",
  full: "100%",
};

const CHAT_DENSITY_VALUE: Record<
  ChatDensity,
  {
    bodyPadding: string;
    contentGap: string;
    messageGap: string;
    bubblePaddingY: string;
    bubblePaddingX: string;
  }
> = {
  compact: {
    bodyPadding: "8px 10px 12px",
    contentGap: "6px",
    messageGap: "4px",
    bubblePaddingY: "6px",
    bubblePaddingX: "9px",
  },
  standard: {
    bodyPadding: "10px 12px 16px",
    contentGap: "10px",
    messageGap: "6px",
    bubblePaddingY: "8px",
    bubblePaddingX: "11px",
  },
  comfortable: {
    bodyPadding: "14px 16px 22px",
    contentGap: "15px",
    messageGap: "8px",
    bubblePaddingY: "11px",
    bubblePaddingX: "14px",
  },
};

const MARKDOWN_DENSITY_VALUE: Record<
  MarkdownDensity,
  {
    paragraphGap: string;
    listGap: string;
    listItemGap: string;
    headingMargin: string;
    codeBlockMargin: string;
    codeBlockPadding: string;
    quotePadding: string;
  }
> = {
  compact: {
    paragraphGap: "4px",
    listGap: "5px",
    listItemGap: "1px",
    headingMargin: "8px 0 4px",
    codeBlockMargin: "6px 0",
    codeBlockPadding: "8px 10px",
    quotePadding: "5px 10px",
  },
  normal: {
    paragraphGap: "6px",
    listGap: "8px",
    listItemGap: "2px",
    headingMargin: "12px 0 5px",
    codeBlockMargin: "8px 0",
    codeBlockPadding: "10px 12px",
    quotePadding: "6px 12px",
  },
  comfortable: {
    paragraphGap: "10px",
    listGap: "12px",
    listItemGap: "4px",
    headingMargin: "16px 0 7px",
    codeBlockMargin: "12px 0",
    codeBlockPadding: "13px 15px",
    quotePadding: "9px 14px",
  },
};

const RADIUS_VALUE: Record<
  CornerRadius,
  {
    micro: string;
    small: string;
    med: string;
    card: string;
    feature: string;
    large: string;
  }
> = {
  compact: {
    micro: "2px",
    small: "3px",
    med: "5px",
    card: "7px",
    feature: "10px",
    large: "14px",
  },
  default: {
    micro: "3px",
    small: "5px",
    med: "7px",
    card: "10px",
    feature: "14px",
    large: "20px",
  },
  round: {
    micro: "5px",
    small: "8px",
    med: "10px",
    card: "14px",
    feature: "18px",
    large: "24px",
  },
};

const CODE_FONT_FAMILY_VALUE: Record<CodeFontFamily, string> = {
  geist:
    '"Geist Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
  system:
    'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
  jetbrains:
    '"JetBrains Mono", "Geist Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
  fira:
    '"Fira Code", "Geist Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
};

export function loadAppearanceSettings(): AppearanceSettings {
  const fallback = {
    ...DEFAULT_APPEARANCE_SETTINGS,
    messageFontSize: loadLegacyMessageFontSize() ?? DEFAULT_APPEARANCE_SETTINGS.messageFontSize,
  };

  try {
    if (typeof window === "undefined") return fallback;
    const raw = window.localStorage.getItem(APPEARANCE_STORAGE_KEY);
    if (!raw) return fallback;
    return normalizeAppearanceSettings(JSON.parse(raw), fallback);
  } catch {
    return fallback;
  }
}

export function saveAppearanceSettings(settings: AppearanceSettings): void {
  const normalized = normalizeAppearanceSettings(settings);
  try {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(APPEARANCE_STORAGE_KEY, JSON.stringify(normalized));
    window.localStorage.setItem(
      LEGACY_MESSAGE_FONT_SIZE_STORAGE_KEY,
      String(normalized.messageFontSize),
    );
  } catch {
    // Storage can fail in private mode or when quota is exhausted. The live
    // CSS application still succeeds for the current session.
  }
}

export function applyStoredAppearanceSettings(): void {
  applyAppearanceSettings(loadAppearanceSettings());
}

export function applyAppearanceSettings(settings: unknown): void {
  if (typeof document === "undefined") return;
  const normalized = normalizeAppearanceSettings(settings);
  const root = document.documentElement;
  const style = root.style;
  const density = CHAT_DENSITY_VALUE[normalized.chatDensity];
  const markdown = MARKDOWN_DENSITY_VALUE[normalized.markdownDensity];
  const radius = RADIUS_VALUE[normalized.cornerRadius];

  root.dataset.themeMode = resolveThemeMode(normalized.themeMode);
  root.dataset.themePreference = normalized.themeMode;
  root.dataset.accentColor = normalized.accentColor;
  root.dataset.chatDensity = normalized.chatDensity;
  root.dataset.bubbleStyle = normalized.bubbleStyle;
  root.dataset.messageMeta = normalized.messageMeta;
  root.dataset.markdownDensity = normalized.markdownDensity;
  root.dataset.codeWrap = normalized.codeWrap ? "true" : "false";
  root.dataset.linkUnderline = normalized.linkUnderline;
  root.dataset.highContrast = normalized.highContrast ? "true" : "false";
  root.dataset.reduceMotion = normalized.reduceMotion ? "true" : "false";

  style.setProperty("--message-font-size", `${normalized.messageFontSize}px`);
  style.setProperty("--message-line-height", LINE_HEIGHT_VALUE[normalized.lineHeight]);
  style.setProperty("--message-max-width", MESSAGE_WIDTH_VALUE[normalized.messageWidth]);
  style.setProperty("--chat-body-padding", density.bodyPadding);
  style.setProperty("--chat-content-gap", density.contentGap);
  style.setProperty("--chat-message-gap", density.messageGap);
  style.setProperty("--message-bubble-padding-y", density.bubblePaddingY);
  style.setProperty("--message-bubble-padding-x", density.bubblePaddingX);
  style.setProperty("--code-font-size", `${normalized.codeFontSize}px`);
  style.setProperty("--font-mono", CODE_FONT_FAMILY_VALUE[normalized.codeFontFamily]);
  style.setProperty("--markdown-paragraph-gap", markdown.paragraphGap);
  style.setProperty("--markdown-list-gap", markdown.listGap);
  style.setProperty("--markdown-list-item-gap", markdown.listItemGap);
  style.setProperty("--markdown-heading-margin", markdown.headingMargin);
  style.setProperty("--markdown-code-block-margin", markdown.codeBlockMargin);
  style.setProperty("--markdown-code-block-padding", markdown.codeBlockPadding);
  style.setProperty("--markdown-quote-padding", markdown.quotePadding);
  style.setProperty("--fs-base", `${normalized.uiScale}px`);
  style.setProperty("--fs-sm", `${Math.max(11, normalized.uiScale - 1)}px`);
  style.setProperty("--fs-xs", `${Math.max(10, normalized.uiScale - 2)}px`);
  style.setProperty("--fs-mono", `${Math.max(11, normalized.uiScale - 1)}px`);
  style.setProperty("--fs-mono-sm", `${Math.max(10.5, normalized.uiScale - 1.5)}px`);
  style.setProperty("--r-micro", radius.micro);
  style.setProperty("--r-small", radius.small);
  style.setProperty("--r-med", radius.med);
  style.setProperty("--r-card", radius.card);
  style.setProperty("--r-feature", radius.feature);
  style.setProperty("--r-large", radius.large);
}

export function normalizeAppearanceSettings(
  value: unknown,
  fallback: AppearanceSettings = DEFAULT_APPEARANCE_SETTINGS,
): AppearanceSettings {
  const record = isRecord(value) ? value : {};
  return {
    messageFontSize: clampInteger(
      record.messageFontSize,
      MIN_MESSAGE_FONT_SIZE,
      MAX_MESSAGE_FONT_SIZE,
      fallback.messageFontSize,
    ),
    lineHeight: pick(
      record.lineHeight,
      ["compact", "normal", "comfortable"],
      fallback.lineHeight,
    ),
    messageWidth: pick(
      record.messageWidth,
      ["narrow", "standard", "wide", "full"],
      fallback.messageWidth,
    ),
    chatDensity: pick(
      record.chatDensity,
      ["compact", "standard", "comfortable"],
      fallback.chatDensity,
    ),
    bubbleStyle: pick(
      record.bubbleStyle,
      ["minimal", "soft", "solid"],
      fallback.bubbleStyle,
    ),
    messageMeta: pick(record.messageMeta, ["hidden", "roles"], fallback.messageMeta),
    markdownDensity: pick(
      record.markdownDensity,
      ["compact", "normal", "comfortable"],
      fallback.markdownDensity,
    ),
    codeFontSize: clampInteger(
      record.codeFontSize,
      MIN_CODE_FONT_SIZE,
      MAX_CODE_FONT_SIZE,
      fallback.codeFontSize,
    ),
    codeFontFamily: pick(
      record.codeFontFamily,
      ["geist", "system", "jetbrains", "fira"],
      fallback.codeFontFamily,
    ),
    codeWrap: booleanValue(record.codeWrap, fallback.codeWrap),
    linkUnderline: pick(
      record.linkUnderline,
      ["subtle", "hover", "always"],
      fallback.linkUnderline,
    ),
    themeMode: pick(record.themeMode, ["system", "dark", "light"], fallback.themeMode),
    accentColor: pick(
      record.accentColor,
      ["blue", "violet", "green", "orange", "rose"],
      fallback.accentColor,
    ),
    highContrast: booleanValue(record.highContrast, fallback.highContrast),
    reduceMotion: booleanValue(record.reduceMotion, fallback.reduceMotion),
    uiScale: clampInteger(record.uiScale, MIN_UI_SCALE, MAX_UI_SCALE, fallback.uiScale),
    cornerRadius: pick(
      record.cornerRadius,
      ["compact", "default", "round"],
      fallback.cornerRadius,
    ),
  };
}

export function resolveThemeMode(mode: ThemeMode): "dark" | "light" {
  if (mode !== "system") return mode;
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return DEFAULT_APPEARANCE_SETTINGS.themeMode === "light" ? "light" : "dark";
  }
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export function watchSystemAppearance(onChange: () => void): () => void {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return noop;
  }
  const media = window.matchMedia("(prefers-color-scheme: light)");
  const handler = () => onChange();
  if (typeof media.addEventListener === "function") {
    media.addEventListener("change", handler);
    return () => media.removeEventListener("change", handler);
  }
  media.addListener(handler);
  return () => media.removeListener(handler);
}

function loadLegacyMessageFontSize(): number | null {
  try {
    if (typeof window === "undefined") return null;
    const raw = window.localStorage.getItem(LEGACY_MESSAGE_FONT_SIZE_STORAGE_KEY);
    if (raw === null) return null;
    return clampInteger(
      raw,
      MIN_MESSAGE_FONT_SIZE,
      MAX_MESSAGE_FONT_SIZE,
      DEFAULT_APPEARANCE_SETTINGS.messageFontSize,
    );
  } catch {
    return null;
  }
}

function pick<T extends string>(value: unknown, options: readonly T[], fallback: T): T {
  return options.includes(value as T) ? (value as T) : fallback;
}

function booleanValue(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function clampInteger(
  value: unknown,
  min: number,
  max: number,
  fallback: number,
): number {
  const parsed = typeof value === "number" ? value : Number.parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, Math.round(parsed)));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function noop(): void {}
