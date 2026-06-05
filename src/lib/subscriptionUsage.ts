import type { AgentEvent, ModelRef, StreamTokenUsage } from "../types";

const STORE_KEY = "sinew.subscriptionUsage.v1";
const FIVE_HOURS_MS = 5 * 60 * 60 * 1000;
const WEEK_MS = 7 * 24 * 60 * 60 * 1000;
const MAX_RECORD_AGE_MS = 15 * 24 * 60 * 60 * 1000;
const MAX_RECORDS = 2_000;

export type SubscriptionUsageRecord = {
  id: string;
  provider: string;
  model: string;
  atMs: number;
  units: number;
  inputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  cacheReadTokens: number;
  cacheCreationTokens: number;
  totalTokens: number;
};

export type SubscriptionUsageSnapshot = {
  provider: string;
  model: string;
  windowUsed: number;
  weeklyUsed: number;
  windowLimit: number;
  weeklyLimit: number;
  windowRatio: number;
  weeklyRatio: number;
  displayRatio: number;
  windowRemaining: number;
  weeklyRemaining: number;
  nextWindowResetAtMs: number | null;
  weeklyResetAtMs: number | null;
  recordCount: number;
  source: "local_estimate";
};

type StoredUsage = {
  records: SubscriptionUsageRecord[];
};

export function readSubscriptionUsageSnapshot(
  model: ModelRef,
  nowMs = Date.now(),
): SubscriptionUsageSnapshot {
  const store = readStore(nowMs);
  return snapshotForModel(store.records, model, nowMs);
}

export function appendSubscriptionUsageFromEvent(
  event: AgentEvent,
  nowMs = Date.now(),
): SubscriptionUsageSnapshot | null {
  if (event.type !== "token_usage") return null;
  if (!isSubscriptionTrackedProvider(event.provider)) return null;

  const units = usageUnits(event.provider, event.model, event.usage);
  if (units <= 0) return null;

  const store = readStore(nowMs);
  const record: SubscriptionUsageRecord = {
    id: usageRecordId(event, nowMs),
    provider: event.provider,
    model: event.model,
    atMs: nowMs,
    units,
    inputTokens: safeNumber(event.usage.input_tokens),
    outputTokens: safeNumber(event.usage.output_tokens),
    reasoningTokens: safeNumber(event.usage.reasoning_tokens),
    cacheReadTokens: safeNumber(event.usage.cache_read_tokens),
    cacheCreationTokens: safeNumber(event.usage.cache_creation_tokens),
    totalTokens: totalTokens(event.usage),
  };

  const records = pruneRecords([...store.records, record], nowMs);
  writeStore({ records });
  return snapshotForModel(records, { provider: event.provider, name: event.model }, nowMs);
}

function snapshotForModel(
  records: SubscriptionUsageRecord[],
  model: ModelRef,
  nowMs: number,
): SubscriptionUsageSnapshot {
  const providerRecords = records.filter((record) => record.provider === model.provider);
  const modelRecords = providerRecords.filter((record) => record.model === model.name);
  const windowStartMs = nowMs - FIVE_HOURS_MS;
  const weeklyStartMs = nowMs - WEEK_MS;
  const windowUsed = sumUnits(modelRecords, windowStartMs);
  const weeklyUsed = sumUnits(modelRecords, weeklyStartMs);
  const limits = modelLimits(model.provider, model.name);
  const windowRatio = ratio(windowUsed, limits.windowLimit);
  const weeklyRatio = ratio(weeklyUsed, limits.weeklyLimit);

  return {
    provider: model.provider,
    model: model.name,
    windowUsed,
    weeklyUsed,
    windowLimit: limits.windowLimit,
    weeklyLimit: limits.weeklyLimit,
    windowRatio,
    weeklyRatio,
    displayRatio: Math.max(windowRatio, weeklyRatio),
    windowRemaining: Math.max(0, limits.windowLimit - windowUsed),
    weeklyRemaining: Math.max(0, limits.weeklyLimit - weeklyUsed),
    nextWindowResetAtMs: nextRollingResetAtMs(modelRecords, windowStartMs, nowMs),
    weeklyResetAtMs: nextRollingResetAtMs(modelRecords, weeklyStartMs, nowMs),
    recordCount: modelRecords.length,
    source: "local_estimate",
  };
}

export function isSubscriptionTrackedProvider(provider: string): boolean {
  return provider === "openai" || provider === "anthropic";
}

function usageUnits(provider: string, model: string, usage: StreamTokenUsage): number {
  const tokens = totalTokens(usage);
  if (tokens <= 0) return 0;
  return tokens * modelUsageWeight(provider, model);
}

function modelUsageWeight(provider: string, model: string): number {
  if (provider === "openai") {
    if (model.includes("mini")) return 0.35;
    if (model.includes("spark")) return 0.5;
    if (model.includes("gpt-5.5")) return 1.35;
    if (model.includes("gpt-5.4")) return 1;
    if (model.includes("codex")) return 0.85;
    return 1;
  }

  if (provider === "anthropic") {
    if (model.includes("haiku")) return 0.35;
    if (model.includes("sonnet")) return 1;
    if (model.includes("opus")) return 2.5;
  }

  return 1;
}

function modelLimits(provider: string, _model: string): {
  windowLimit: number;
  weeklyLimit: number;
} {
  if (provider === "openai") {
    return {
      windowLimit: 900_000,
      weeklyLimit: 6_000_000,
    };
  }
  if (provider === "anthropic") {
    return {
      windowLimit: 650_000,
      weeklyLimit: 4_500_000,
    };
  }
  return {
    windowLimit: 1_000_000,
    weeklyLimit: 7_000_000,
  };
}

function sumUnits(records: SubscriptionUsageRecord[], sinceMs: number): number {
  return records.reduce(
    (total, record) => (record.atMs >= sinceMs ? total + record.units : total),
    0,
  );
}

function nextRollingResetAtMs(
  records: SubscriptionUsageRecord[],
  sinceMs: number,
  nowMs: number,
): number | null {
  const active = records
    .filter((record) => record.atMs >= sinceMs)
    .sort((a, b) => a.atMs - b.atMs);
  const first = active[0];
  if (!first) return null;
  const windowMs = nowMs - sinceMs;
  return first.atMs + windowMs;
}

function readStore(nowMs: number): StoredUsage {
  if (typeof window === "undefined") return { records: [] };
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    if (!raw) return { records: [] };
    const parsed = JSON.parse(raw) as Partial<StoredUsage>;
    return { records: pruneRecords(parseRecords(parsed.records), nowMs) };
  } catch {
    return { records: [] };
  }
}

function writeStore(store: StoredUsage) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORE_KEY, JSON.stringify(store));
  } catch {
    // Local usage is a best-effort estimate; storage failure should never block chat.
  }
}

function pruneRecords(
  records: SubscriptionUsageRecord[],
  nowMs: number,
): SubscriptionUsageRecord[] {
  const minAtMs = nowMs - MAX_RECORD_AGE_MS;
  const pruned = records.filter(
    (record) => record.atMs >= minAtMs && record.units > 0,
  );
  if (pruned.length <= MAX_RECORDS) return pruned;
  return pruned.slice(pruned.length - MAX_RECORDS);
}

function parseRecords(value: unknown): SubscriptionUsageRecord[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const record = item as Partial<SubscriptionUsageRecord>;
    if (typeof record.provider !== "string" || typeof record.model !== "string") {
      return [];
    }
    const atMs = safeNumber(record.atMs);
    const units = safeNumber(record.units);
    if (atMs <= 0 || units <= 0) return [];
    return [
      {
        id: typeof record.id === "string" ? record.id : `${record.provider}:${record.model}:${atMs}`,
        provider: record.provider,
        model: record.model,
        atMs,
        units,
        inputTokens: safeNumber(record.inputTokens),
        outputTokens: safeNumber(record.outputTokens),
        reasoningTokens: safeNumber(record.reasoningTokens),
        cacheReadTokens: safeNumber(record.cacheReadTokens),
        cacheCreationTokens: safeNumber(record.cacheCreationTokens),
        totalTokens: safeNumber(record.totalTokens),
      },
    ];
  });
}

function usageRecordId(event: Extract<AgentEvent, { type: "token_usage" }>, nowMs: number): string {
  const usage = event.usage;
  return [
    event.provider,
    event.model,
    nowMs,
    usage.input_tokens,
    usage.output_tokens,
    usage.reasoning_tokens,
    usage.cache_read_tokens,
    usage.cache_creation_tokens,
  ].join(":");
}

function totalTokens(usage: StreamTokenUsage): number {
  const explicit = safeNumber(usage.total_tokens);
  if (explicit > 0) return explicit;
  return (
    safeNumber(usage.input_tokens) +
    safeNumber(usage.output_tokens) +
    safeNumber(usage.reasoning_tokens) +
    safeNumber(usage.cache_read_tokens) +
    safeNumber(usage.cache_creation_tokens)
  );
}

function ratio(value: number, limit: number): number {
  if (limit <= 0) return 0;
  return Math.max(0, Math.min(1, value / limit));
}

function safeNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, Math.round(value))
    : 0;
}
