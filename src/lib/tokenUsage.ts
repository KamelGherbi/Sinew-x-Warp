import type {
  StreamTokenUsage,
  TokenUsageScopeSummary,
  TokenUsageSummary,
  TokenUsageTotals,
} from "../types";

// =============================================================================
// Token usage + estimated cost
// -----------------------------------------------------------------------------
// The backend (`token_usage_summary`) returns authoritative, persisted token
// totals split into a "conversation" scope (the active conversation) and a
// "global" scope (every conversation in the local store), each broken down by
// provider and model.
//
// This module adds two things on top of that raw data:
//   1. An *estimated* dollar cost derived from coarse, family-level catalogue
//      prices. Sinew ships renamed / forward-dated model identifiers, so these
//      are deliberately approximate and are always surfaced as estimates. When
//      a model can't be priced we report `costKnown: false` instead of guessing.
//   2. A live overlay: `token_usage` stream events received while a turn is in
//      flight are merged on top of the persisted summary so the indicator moves
//      immediately, without waiting for the next backend reload.
// =============================================================================

// ---------------------------------------------------------------------------
// Pricing model
// ---------------------------------------------------------------------------

/** Catalogue prices in US dollars per 1,000,000 tokens. */
export type TokenRates = {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
};

type ProviderBilling = {
  /**
   * Whether the provider folds reasoning/thinking tokens into `outputTokens`.
   * When false (e.g. Gemini), reasoning is reported separately and must be
   * billed on top of output. When true, reasoning is already counted in output.
   */
  outputIncludesReasoning: boolean;
  /**
   * Whether `inputTokens` already includes the cache-read tokens (OpenAI-style
   * `prompt_tokens`). When true we bill only the non-cached remainder at the
   * full input rate and the cached portion at the cheaper cache-read rate.
   */
  inputIncludesCacheRead: boolean;
};

const DEFAULT_BILLING: ProviderBilling = {
  outputIncludesReasoning: true,
  inputIncludesCacheRead: true,
};

const PROVIDER_BILLING: Record<string, ProviderBilling> = {
  // OpenAI Responses API: completion tokens include reasoning, prompt tokens
  // include the cached portion, no separate cache-creation charge.
  openai: { outputIncludesReasoning: true, inputIncludesCacheRead: true },
  // Kimi is OpenAI-compatible.
  kimi: { outputIncludesReasoning: true, inputIncludesCacheRead: true },
  // Anthropic: thinking tokens are counted inside output; cache read/creation
  // tokens are reported as fields separate from `input_tokens`.
  anthropic: { outputIncludesReasoning: true, inputIncludesCacheRead: false },
  // Gemini: candidate (output) tokens exclude "thoughts", which arrive in the
  // separate reasoning field; cache-read is not surfaced (always 0).
  google: { outputIncludesReasoning: false, inputIncludesCacheRead: true },
};

function billingFor(provider: string): ProviderBilling {
  return PROVIDER_BILLING[provider.toLowerCase()] ?? DEFAULT_BILLING;
}

function rates(
  input: number,
  output: number,
  cacheReadFactor = 0.1,
  cacheWriteFactor = 1,
): TokenRates {
  return {
    input,
    output,
    cacheRead: round4(input * cacheReadFactor),
    cacheWrite: round4(input * cacheWriteFactor),
  };
}

// Anthropic charges ~1.25x input for cache writes and ~0.1x for cache reads.
const anthropicRates = (input: number, output: number) =>
  rates(input, output, 0.1, 1.25);

/**
 * Estimated per-million-token rates for a provider/model pair. Matching is done
 * on coarse model-family keywords so new point releases stay covered. Returns
 * `null` when the provider/model can't be priced (e.g. OpenRouter passthrough),
 * which the UI renders as an unknown cost rather than a fabricated number.
 */
export function ratesFor(provider: string, model: string): TokenRates | null {
  const p = provider.toLowerCase();
  const m = model.toLowerCase();
  switch (p) {
    case "anthropic":
      if (m.includes("opus")) return anthropicRates(15, 75);
      if (m.includes("haiku")) return anthropicRates(0.8, 4);
      // sonnet, fable, and any other mid-tier Claude.
      return anthropicRates(3, 15);
    case "openai":
      if (m.includes("mini")) return rates(0.25, 2);
      if (m.includes("spark")) return rates(0.5, 4);
      // gpt-5.x family + codex share the flagship tier.
      return rates(1.25, 10);
    case "google":
      if (m.includes("flash")) return rates(0.3, 2.5);
      // pro and anything else.
      return rates(1.25, 10);
    case "kimi":
      return rates(0.6, 2.5);
    default:
      // OpenRouter and unknown providers: pricing varies per underlying model.
      return null;
  }
}

/** Estimated dollar cost for a single provider/model's accumulated totals. */
export function estimateCost(
  provider: string,
  model: string,
  totals: TokenUsageTotals,
): { usd: number; known: boolean } {
  const r = ratesFor(provider, model);
  if (!r) return { usd: 0, known: false };

  const billing = billingFor(provider);
  const input = safe(totals.inputTokens);
  const output = safe(totals.outputTokens);
  const reasoning = safe(totals.reasoningTokens);
  const cacheRead = safe(totals.cacheReadTokens);
  const cacheWrite = safe(totals.cacheCreationTokens);

  const nonCachedInput = billing.inputIncludesCacheRead
    ? Math.max(0, input - cacheRead)
    : input;
  const generatedOutput = billing.outputIncludesReasoning
    ? output
    : output + reasoning;

  const usd =
    perMillion(nonCachedInput, r.input) +
    perMillion(generatedOutput, r.output) +
    perMillion(cacheRead, r.cacheRead) +
    perMillion(cacheWrite, r.cacheWrite);

  return { usd, known: true };
}

// ---------------------------------------------------------------------------
// Live overlay accumulation
// ---------------------------------------------------------------------------

// A live overlay keyed by `${provider}\u0000${model}`. Each `token_usage`
// stream event represents one request (one model round-trip), mirroring how the
// backend persists a `requests: 1` record per assistant message.
export type LiveUsageMap = Record<string, TokenUsageTotals>;

const KEY_SEP = "\u0000";

function modelKey(provider: string, model: string): string {
  return `${provider}${KEY_SEP}${model}`;
}

function splitModelKey(key: string): { provider: string; model: string } {
  const index = key.indexOf(KEY_SEP);
  if (index < 0) return { provider: key, model: key };
  return { provider: key.slice(0, index), model: key.slice(index + 1) };
}

export function emptyTotals(): TokenUsageTotals {
  return {
    requests: 0,
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
    reasoningTokens: 0,
    cacheReadTokens: 0,
    cacheCreationTokens: 0,
  };
}

function totalsFromStreamUsage(usage: StreamTokenUsage): TokenUsageTotals {
  const inputTokens = safe(usage.input_tokens);
  const outputTokens = safe(usage.output_tokens);
  const reasoningTokens = safe(usage.reasoning_tokens);
  const cacheReadTokens = safe(usage.cache_read_tokens);
  const cacheCreationTokens = safe(usage.cache_creation_tokens);
  const explicitTotal = safe(usage.total_tokens);
  const summedTotal =
    inputTokens +
    outputTokens +
    reasoningTokens +
    cacheReadTokens +
    cacheCreationTokens;
  return {
    requests: 1,
    inputTokens,
    outputTokens,
    totalTokens: explicitTotal > 0 ? explicitTotal : summedTotal,
    reasoningTokens,
    cacheReadTokens,
    cacheCreationTokens,
  };
}

function addTotals(a: TokenUsageTotals, b: TokenUsageTotals): TokenUsageTotals {
  return {
    requests: a.requests + b.requests,
    inputTokens: a.inputTokens + b.inputTokens,
    outputTokens: a.outputTokens + b.outputTokens,
    totalTokens: a.totalTokens + b.totalTokens,
    reasoningTokens: a.reasoningTokens + b.reasoningTokens,
    cacheReadTokens: a.cacheReadTokens + b.cacheReadTokens,
    cacheCreationTokens: a.cacheCreationTokens + b.cacheCreationTokens,
  };
}

/**
 * Returns a new overlay with one `token_usage` event folded in. The original
 * map is never mutated so it stays React-state friendly. Events that carry no
 * tokens are ignored.
 */
export function addLiveUsage(
  map: LiveUsageMap,
  provider: string,
  model: string,
  usage: StreamTokenUsage,
): LiveUsageMap {
  const cleanProvider = provider.trim();
  const cleanModel = model.trim();
  if (!cleanProvider || !cleanModel) return map;
  const totals = totalsFromStreamUsage(usage);
  if (totals.totalTokens <= 0) return map;
  const key = modelKey(cleanProvider, cleanModel);
  const next = { ...map };
  next[key] = addTotals(next[key] ?? emptyTotals(), totals);
  return next;
}

/** Merge every conversation's live overlay into a single global overlay. */
export function mergeLiveMaps(maps: LiveUsageMap[]): LiveUsageMap {
  const merged: LiveUsageMap = {};
  for (const map of maps) {
    for (const [key, totals] of Object.entries(map)) {
      merged[key] = addTotals(merged[key] ?? emptyTotals(), totals);
    }
  }
  return merged;
}

// ---------------------------------------------------------------------------
// View model
// ---------------------------------------------------------------------------

export type TotalsView = TokenUsageTotals & {
  /** Estimated cost in USD for the priced portion of these totals. */
  costUsd: number;
  /** True only when every token-bearing model in this group could be priced. */
  costKnown: boolean;
};

export type ModelUsageView = {
  provider: string;
  model: string;
  totals: TotalsView;
};

export type ProviderUsageView = {
  provider: string;
  totals: TotalsView;
  models: ModelUsageView[];
};

export type ScopeUsageView = {
  totals: TotalsView;
  providers: ProviderUsageView[];
};

export type TokenUsageView = {
  conversation: ScopeUsageView;
  global: ScopeUsageView;
  /** False when no tokens have been recorded anywhere yet (empty state). */
  hasAny: boolean;
};

export function buildTokenUsageView(
  summary: TokenUsageSummary | null,
  liveByConversation: Record<string, LiveUsageMap>,
  activeConversationId: string,
): TokenUsageView {
  const conversationLive = liveByConversation[activeConversationId] ?? {};
  const globalLive = mergeLiveMaps(Object.values(liveByConversation));
  const conversation = buildScope(summary?.conversation ?? null, conversationLive);
  const global = buildScope(summary?.global ?? null, globalLive);
  return {
    conversation,
    global,
    hasAny: global.totals.totalTokens > 0,
  };
}

function buildScope(
  scope: TokenUsageScopeSummary | null,
  live: LiveUsageMap,
): ScopeUsageView {
  // Merge backend + live at model granularity. The backend's provider/scope
  // totals are exactly the sum of their models, so rebuilding from the model
  // level is lossless and keeps the live overlay easy to fold in.
  const byModel = new Map<string, { provider: string; model: string; totals: TokenUsageTotals }>();

  const addModel = (provider: string, model: string, totals: TokenUsageTotals) => {
    const key = modelKey(provider, model);
    const existing = byModel.get(key);
    if (existing) existing.totals = addTotals(existing.totals, totals);
    else byModel.set(key, { provider, model, totals: { ...totals } });
  };

  if (scope) {
    for (const provider of scope.providers) {
      for (const model of provider.models) {
        addModel(model.provider, model.model, model.totals);
      }
    }
  }
  for (const [key, totals] of Object.entries(live)) {
    const { provider, model } = splitModelKey(key);
    addModel(provider, model, totals);
  }

  // Group models under providers.
  const providerMap = new Map<string, ModelUsageView[]>();
  for (const { provider, model, totals } of byModel.values()) {
    const view: ModelUsageView = {
      provider,
      model,
      totals: withCost(provider, model, totals),
    };
    const list = providerMap.get(provider);
    if (list) list.push(view);
    else providerMap.set(provider, [view]);
  }

  const providers: ProviderUsageView[] = [];
  for (const [provider, models] of providerMap.entries()) {
    sortModels(models);
    providers.push({
      provider,
      totals: aggregateTotals(models.map((entry) => entry.totals)),
      models,
    });
  }
  sortProviders(providers);

  return {
    totals: aggregateTotals(providers.map((entry) => entry.totals)),
    providers,
  };
}

function withCost(
  provider: string,
  model: string,
  totals: TokenUsageTotals,
): TotalsView {
  const hasTokens = totals.totalTokens > 0;
  const cost = estimateCost(provider, model, totals);
  return {
    ...totals,
    costUsd: cost.usd,
    // A zero-token model never blocks the "known" flag.
    costKnown: hasTokens ? cost.known : true,
  };
}

function aggregateTotals(children: TotalsView[]): TotalsView {
  let totals = emptyTotals();
  let costUsd = 0;
  let costKnown = true;
  for (const child of children) {
    totals = addTotals(totals, child);
    costUsd += child.costUsd;
    if (child.totalTokens > 0 && !child.costKnown) costKnown = false;
  }
  return { ...totals, costUsd, costKnown };
}

function sortModels(models: ModelUsageView[]): void {
  models.sort(
    (a, b) =>
      b.totals.totalTokens - a.totals.totalTokens ||
      a.model.localeCompare(b.model),
  );
}

function sortProviders(providers: ProviderUsageView[]): void {
  providers.sort(
    (a, b) =>
      b.totals.totalTokens - a.totals.totalTokens ||
      a.provider.localeCompare(b.provider),
  );
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/** Compact token count, e.g. 1234 -> "1.2K", 4_500_000 -> "4.5M". */
export function formatTokenCount(value: number): string {
  const tokens = safe(value);
  if (tokens < 1000) return String(tokens);
  if (tokens < 1_000_000) return `${trimZero(tokens / 1000)}K`;
  if (tokens < 1_000_000_000) return `${trimZero(tokens / 1_000_000)}M`;
  return `${trimZero(tokens / 1_000_000_000)}B`;
}

/** Estimated cost label. Always prefixed to read as an approximation. */
export function formatCostUsd(usd: number): string {
  if (!Number.isFinite(usd) || usd <= 0) return "$0.00";
  if (usd < 0.01) return "<$0.01";
  if (usd < 100) return `$${usd.toFixed(2)}`;
  return `$${Math.round(usd).toLocaleString()}`;
}

function trimZero(value: number): string {
  const fixed = value.toFixed(1);
  return fixed.endsWith(".0") ? fixed.slice(0, -2) : fixed;
}

function perMillion(tokens: number, rate: number): number {
  return (safe(tokens) / 1_000_000) * rate;
}

function round4(value: number): number {
  return Math.round(value * 10_000) / 10_000;
}

function safe(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, value)
    : 0;
}
