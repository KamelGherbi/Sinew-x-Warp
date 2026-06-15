import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "@iconify/react";
import { api } from "../lib/ipc";
import { PROVIDERS, type ProviderId } from "../lib/models";
import {
  buildTokenUsageView,
  formatCostUsd,
  formatTokenCount,
  type LiveUsageMap,
  type ScopeUsageView,
  type TotalsView,
} from "../lib/tokenUsage";
import type {
  ProviderConnectionState,
  ProviderUsageBalance,
  ProviderUsageSpend,
  ProviderUsageStatus,
  ProviderUsageSummary,
  ProviderUsageWindow,
  TokenUsageSummary,
} from "../types";

// =============================================================================
// StatusWidget — shell-level AI quota + usage overview
// -----------------------------------------------------------------------------
// A compact, always-visible titlebar affordance (CodexBar-like) that leads with
// the *exact, provider-side* quota each provider reports (session/weekly/rate
// limits, credit balances, spend) and keeps the *local* token/cost estimate as
// a clearly-labelled secondary detail.
//
//   * Provider-side quota comes from `provider_usage_summary` and is read live
//     from each provider when connected — it is the authoritative "how much is
//     left" figure, never an estimate.
//   * Local token usage + estimated cost (per conversation and global) is the
//     fallback we always have, framed as an approximation from catalogue pricing
//     so it is never confused with real subscription remaining.
//
// The backend `provider_usage_summary` command centralizes the external quota
// probes so the UI stays focused on rendering and clear labelling.
// =============================================================================

const PROVIDERS_CHANGED_EVENT = "sinew:providers-changed";
// Provider-side quota calls hit external APIs, so keep the shell dynamic without
// hammering providers. Focus/provider-change events still refresh immediately.
const REFRESH_INTERVAL_MS = 60_000;
// Stable empty live overlay: the shell widget reads persisted totals only.
const NO_LIVE_USAGE: Record<string, LiveUsageMap> = {};
// Zero-filled per-provider totals for providers with no recorded usage yet, so
// every provider row can still render its (empty) conversation/global figures.
const ZERO_TOTALS_VIEW: TotalsView = {
  requests: 0,
  inputTokens: 0,
  outputTokens: 0,
  totalTokens: 0,
  reasoningTokens: 0,
  cacheReadTokens: 0,
  cacheCreationTokens: 0,
  costUsd: 0,
  costKnown: true,
};

type Tone = "ok" | "pending" | "error" | "off";
// Severity of a quota bar, driven by how much headroom is left.
type BarTone = "ok" | "warn" | "low" | "unknown";

type UsageProviderId = Exclude<ProviderId, "anthropic">;

const USAGE_PROVIDERS = PROVIDERS.filter(
  (provider): provider is (typeof PROVIDERS)[number] & { value: UsageProviderId } =>
    provider.value !== "anthropic",
);

/** Shared shape across the provider status payloads we care about here. */
type ProviderStatusLike = {
  connected: boolean;
  connectionState: ProviderConnectionState;
  error?: string | null;
};

type StatusMap = Record<UsageProviderId, ProviderStatusLike | null>;

const EMPTY_STATUS: StatusMap = {
  openai: null,
  google: null,
  kimi: null,
  openrouter: null,
};

type Props = {
  workspacePath: string;
  conversationId: string;
};

export function StatusWidget({ workspacePath, conversationId }: Props) {
  const [summary, setSummary] = useState<TokenUsageSummary | null>(null);
  const [configured, setConfigured] = useState<readonly string[]>([]);
  const [statuses, setStatuses] = useState<StatusMap>(EMPTY_STATUS);
  const [usage, setUsage] = useState<ProviderUsageSummary | null>(null);
  const [open, setOpen] = useState(false);

  const rootRef = useRef<HTMLDivElement | null>(null);
  const mountedRef = useRef(true);
  // Monotonic request id: only the most recent refresh is allowed to commit,
  // so a slow response from a previous workspace/conversation never wins.
  const reqIdRef = useRef(0);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    const myId = ++reqIdRef.current;
    const [
      summaryRes,
      configuredRes,
      usageRes,
      openai,
      google,
      kimi,
      openrouter,
    ] = await Promise.all([
      api.tokenUsageSummary(workspacePath, conversationId).catch(() => null),
      api.listConfiguredModelProviders().catch(() => [] as string[]),
      api.providerUsageSummary().catch(() => null),
      api.getOpenAiProviderStatus().catch(() => null),
      api.getGoogleProviderStatus().catch(() => null),
      api.getKimiProviderStatus().catch(() => null),
      api.getOpenRouterProviderStatus().catch(() => null),
    ]);
    // Drop stale or post-unmount responses.
    if (!mountedRef.current || myId !== reqIdRef.current) return;
    setSummary(summaryRes);
    setConfigured(configuredRes.filter((provider) => provider !== "anthropic"));
    setUsage(usageRes);
    setStatuses({ openai, google, kimi, openrouter });
  }, [workspacePath, conversationId]);

  // Mount + workspace/conversation change.
  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Focus, visibility, provider settings changes, and a light poll.
  useEffect(() => {
    const onChange = () => void refresh();
    const onVisible = () => {
      if (document.visibilityState === "visible") void refresh();
    };
    window.addEventListener(PROVIDERS_CHANGED_EVENT, onChange);
    window.addEventListener("focus", onChange);
    document.addEventListener("visibilitychange", onVisible);
    const interval = window.setInterval(() => {
      if (document.visibilityState === "visible") void refresh();
    }, REFRESH_INTERVAL_MS);
    return () => {
      window.removeEventListener(PROVIDERS_CHANGED_EVENT, onChange);
      window.removeEventListener("focus", onChange);
      document.removeEventListener("visibilitychange", onVisible);
      window.clearInterval(interval);
    };
  }, [refresh]);

  // Keep a clicked-open popover stable while users move into it and scroll.
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (target && rootRef.current?.contains(target)) return;
      setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const view = useMemo(
    () => buildTokenUsageView(summary, NO_LIVE_USAGE, conversationId),
    [summary, conversationId],
  );

  const configuredSet = useMemo(() => new Set(configured), [configured]);

  // Index the provider-side quota payloads by lowercased provider id so each
  // row can surface its exact session/weekly/rate-limit windows.
  const usageByProvider = useMemo(() => {
    const map = new Map<string, ProviderUsageStatus>();
    for (const status of usage?.providers ?? []) {
      map.set(status.provider.toLowerCase(), status);
    }
    return map;
  }, [usage]);

  // Index per-provider usage for both scopes so each provider row can surface
  // its own conversation and global token/cost figures alongside its status.
  const providerRows = useMemo(() => {
    const conversationByProvider = indexProviderTotals(view.conversation);
    const globalByProvider = indexProviderTotals(view.global);
    return USAGE_PROVIDERS.map((provider) => {
      const status = statuses[provider.value];
      const descriptor = describeStatus(status, configuredSet.has(provider.value));
      const conversationTotals =
        conversationByProvider.get(provider.value) ?? ZERO_TOTALS_VIEW;
      const globalTotals = globalByProvider.get(provider.value) ?? ZERO_TOTALS_VIEW;
      const quota = usageByProvider.get(provider.value) ?? null;
      return { provider, descriptor, conversationTotals, globalTotals, quota };
    });
  }, [view, statuses, configuredSet, usageByProvider]);

  const connectedCount = providerRows.filter(
    (row) => row.descriptor.tone === "ok",
  ).length;
  const liveQuotaCount = providerRows.filter(
    (row) => row.quota?.state === "available",
  ).length;

  // Overall health for the at-a-glance dot. "Green" whenever at least one
  // provider is usable; otherwise reflect connecting/error/idle state. Computed
  // order-independently so the result is stable regardless of provider order.
  const overallTone = useMemo<Tone>(() => {
    let anyConnecting = false;
    let anyError = false;
    for (const { descriptor } of providerRows) {
      if (descriptor.tone === "ok") return "ok";
      if (descriptor.tone === "pending") anyConnecting = true;
      else if (descriptor.tone === "error") anyError = true;
    }
    return anyConnecting ? "pending" : anyError ? "error" : "off";
  }, [providerRows]);

  const { conversation, global, hasAny } = view;
  const someUnpriced =
    (global.totals.totalTokens > 0 && !global.totals.costKnown) ||
    (conversation.totals.totalTokens > 0 && !conversation.totals.costKnown);

  // Tightest provider-side headroom, if anyone reports one — the most CodexBar-
  // like at-a-glance number. Falls back to the local token estimate otherwise.
  const tightestRemaining = useMemo(
    () => lowestRemainingPercent(providerRows.map((row) => row.quota)),
    [providerRows],
  );

  const triggerLabel = tightestRemaining != null
    ? `AI status \u00b7 ${formatPercent(tightestRemaining)} provider quota remaining \u00b7 ${connectedCount} provider${connectedCount === 1 ? "" : "s"} connected`
    : hasAny
      ? `AI status \u00b7 local estimate ${formatTokenCount(global.totals.totalTokens)} tokens (~${formatCostUsd(global.totals.costUsd)}) \u00b7 ${connectedCount} provider${connectedCount === 1 ? "" : "s"} connected`
      : `AI status \u00b7 ${connectedCount} provider${connectedCount === 1 ? "" : "s"} connected`;

  return (
    <div
      ref={rootRef}
      className="status-widget"
      data-tone={overallTone}
      data-open={open ? "true" : "false"}
    >
      <button
        type="button"
        className="status-widget__trigger"
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls="status-widget-popover"
        aria-label={triggerLabel}
        onClick={() => setOpen((value) => !value)}
      >
        <span className="status-widget__dot" data-tone={overallTone} aria-hidden />
        <Icon
          className="status-widget__glyph"
          icon="solar:chart-2-linear"
          width={13}
          height={13}
          aria-hidden
        />
        {tightestRemaining != null ? (
          <span className="status-widget__value">
            {formatPercent(tightestRemaining)}
            <span className="status-widget__value-unit"> left</span>
          </span>
        ) : (
          <>
            <span className="status-widget__value">
              {hasAny ? formatTokenCount(global.totals.totalTokens) : "0"}
            </span>
            {hasAny && global.totals.costUsd > 0 && (
              <span className="status-widget__cost">
                {"\u2248 "}
                {formatCostUsd(global.totals.costUsd)}
              </span>
            )}
          </>
        )}
      </button>

      <div
        id="status-widget-popover"
        className="status-widget__popover"
        role="dialog"
        aria-label="AI quota and usage"
        onPointerDown={(event) => event.stopPropagation()}
        onWheel={(event) => event.stopPropagation()}
      >
        <section className="status-widget__section">
          <header className="status-widget__section-head">
            <span className="status-widget__section-title">Provider quota</span>
            <span className="status-widget__section-meta">
              {liveQuotaCount > 0
                ? `${liveQuotaCount} live \u00b7 ${connectedCount} connected`
                : `${connectedCount} connected`}
            </span>
          </header>
          <div className="status-widget__providers">
            {providerRows.map(
              ({ provider, descriptor, conversationTotals, globalTotals, quota }) => (
                <div
                  className="status-widget__provider"
                  key={provider.value}
                  data-configured={
                    configuredSet.has(provider.value) ? "true" : "false"
                  }
                >
                  <div className="status-widget__provider-head">
                    <span className="status-widget__provider-mark" aria-hidden>
                      <Icon icon={provider.icon} width={15} height={15} />
                    </span>
                    <span className="status-widget__provider-name">
                      {provider.label}
                    </span>
                    <span
                      className="status-widget__provider-status"
                      data-tone={descriptor.tone}
                    >
                      <span className="status-widget__provider-dot" aria-hidden />
                      {descriptor.label}
                    </span>
                  </div>

                  <ProviderQuotaBlock quota={quota} />

                  {globalTotals.totalTokens > 0 && (
                    <LocalEstimate
                      conversationTotals={conversationTotals}
                      globalTotals={globalTotals}
                    />
                  )}
                </div>
              ),
            )}
          </div>
        </section>

        <section className="status-widget__section">
          <header className="status-widget__section-head">
            <span className="status-widget__section-title">
              Local token estimate
            </span>
            <span className="status-widget__section-meta">Estimated cost</span>
          </header>
          {hasAny ? (
            <div className="status-widget__scopes">
              <ScopeRow label="This conversation" scope={conversation} />
              <ScopeRow label="All conversations" scope={global} />
            </div>
          ) : (
            <div className="status-widget__empty">
              No token usage yet. Local token and cost estimates appear here once
              a model responds.
            </div>
          )}
        </section>

        <div className="status-widget__note">
          Provider quota is read live from each provider when connected. Token
          figures are local estimates from catalogue pricing
          {someUnpriced ? " (some models have no known price)" : ""}
          {usage ? ` \u00b7 updated ${formatUpdatedAt(usage.updatedAtMs)}` : ""}.
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Provider-side quota (primary): exact windows, balances and spend.
// ---------------------------------------------------------------------------

/** Renders the authoritative provider-side quota, or why it is unavailable. */
function ProviderQuotaBlock({ quota }: { quota: ProviderUsageStatus | null }) {
  if (!quota || quota.state === "unavailable") {
    return (
      <div className="status-widget__quota" data-state="unavailable">
        <div className="status-widget__quota-head">
          <span className="status-widget__quota-title">Provider quota</span>
          <span className="status-widget__quota-flag" data-tone="muted">
            Unavailable
          </span>
        </div>
        <div className="status-widget__quota-note">
          {quota?.error ?? "Live provider-side quota is unavailable."}
        </div>
      </div>
    );
  }

  if (quota.state === "error") {
    return (
      <div className="status-widget__quota" data-state="error">
        <div className="status-widget__quota-head">
          <span className="status-widget__quota-title">Provider quota</span>
          <span className="status-widget__quota-flag" data-tone="error">
            Error
          </span>
        </div>
        <div className="status-widget__quota-note">
          {quota.error ?? "Couldn't load provider-side quota."}
        </div>
      </div>
    );
  }

  const hasDetail =
    quota.windows.length > 0 || quota.balance != null || quota.spend != null;

  return (
    <div className="status-widget__quota" data-state="available">
      <div className="status-widget__quota-head">
        <span className="status-widget__quota-title">
          {quota.label ?? "Provider-side quota"}
        </span>
        <span
          className="status-widget__quota-flag"
          data-tone="ok"
          title="Exact, read live from the provider"
        >
          {quota.exact ? "Exact" : "Live"}
        </span>
      </div>
      {quota.windows.map((win) => (
        <UsageWindowRow key={win.id} win={win} />
      ))}
      {quota.balance != null && <BalanceLine balance={quota.balance} />}
      {quota.spend != null && <SpendLine spend={quota.spend} />}
      {!hasDetail && (
        <div className="status-widget__quota-note">
          Connected — no quota details reported.
        </div>
      )}
    </div>
  );
}

/** One quota window: a label, headline figure, headroom bar and detail line. */
function UsageWindowRow({ win }: { win: ProviderUsageWindow }) {
  const { primary, amounts, reset } = describeWindow(win);
  const remaining = remainingFractionOf(win);
  const tone = barToneFor(remaining);
  const fillPercent = remaining != null ? clampPercent(remaining * 100) : null;
  const details = [amounts, reset].filter((part): part is string => Boolean(part));

  return (
    <div className="status-widget__window">
      <div className="status-widget__window-row">
        <span className="status-widget__window-label" title={win.label}>
          {win.label}
        </span>
        <span className="status-widget__window-value">{primary}</span>
      </div>
      {fillPercent != null && (
        <div className="status-widget__bar" data-tone={tone} aria-hidden>
          <span
            className="status-widget__bar-fill"
            style={{ width: `${fillPercent}%` }}
          />
        </div>
      )}
      {details.length > 0 && (
        <div className="status-widget__window-sub">{details.join(" \u00b7 ")}</div>
      )}
    </div>
  );
}

/** Credit / balance figure (e.g. OpenRouter credits, Codex credits). */
function BalanceLine({ balance }: { balance: ProviderUsageBalance }) {
  return (
    <div className="status-widget__quota-line">
      <span className="status-widget__quota-line-label">{balance.label}</span>
      <span className="status-widget__quota-line-value">
        {formatBalanceValue(balance)}
      </span>
    </div>
  );
}

/** Rolling spend figures (today / week / month) when a provider reports them. */
function SpendLine({ spend }: { spend: ProviderUsageSpend }) {
  const parts = formatSpendParts(spend);
  if (parts.length === 0) return null;
  return (
    <div className="status-widget__quota-line">
      <span className="status-widget__quota-line-label">Spend</span>
      <span className="status-widget__quota-line-value">
        {parts.join(" \u00b7 ")}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Local token estimate (secondary): always-available, clearly approximate.
// ---------------------------------------------------------------------------

/** Per-provider local token/cost estimate, framed as a fallback detail. */
function LocalEstimate({
  conversationTotals,
  globalTotals,
}: {
  conversationTotals: TotalsView;
  globalTotals: TotalsView;
}) {
  return (
    <div className="status-widget__local">
      <span className="status-widget__local-label">Local estimate</span>
      <div className="status-widget__provider-usage">
        <ProviderMetric label="This conversation" totals={conversationTotals} />
        <ProviderMetric label="All conversations" totals={globalTotals} />
      </div>
    </div>
  );
}

function ScopeRow({ label, scope }: { label: string; scope: ScopeUsageView }) {
  const { totals } = scope;
  return (
    <div className="status-widget__scope">
      <div className="status-widget__scope-row">
        <span className="status-widget__scope-label">{label}</span>
        <span className="status-widget__scope-cost">{costLabel(totals)}</span>
      </div>
      <div className="status-widget__scope-sub">
        <span>{formatTokenCount(totals.totalTokens)} tokens</span>
        <span>
          {totals.requests} {totals.requests === 1 ? "request" : "requests"}
        </span>
      </div>
    </div>
  );
}

/** Estimated-cost label, always framed as an approximation. */
function costLabel(totals: TotalsView): string {
  if (totals.totalTokens > 0 && !totals.costKnown && totals.costUsd <= 0) {
    return "est. n/a";
  }
  return `\u2248 ${formatCostUsd(totals.costUsd)}`;
}

/**
 * One scope's figures for a single provider: a label, its token count, and the
 * estimated cost. Used to show conversation and global usage per AI provider.
 */
function ProviderMetric({
  label,
  totals,
}: {
  label: string;
  totals: TotalsView;
}) {
  return (
    <div className="status-widget__provider-metric">
      <span className="status-widget__provider-scope">{label}</span>
      <span className="status-widget__provider-figures">
        <span className="status-widget__provider-tokens">
          {formatTokenCount(totals.totalTokens)} tokens
        </span>
        <span className="status-widget__provider-cost">{costLabel(totals)}</span>
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Quota window math + formatting helpers.
// ---------------------------------------------------------------------------

type WindowParts = { primary: string; amounts: string | null; reset: string | null };

/** Builds the headline figure plus optional absolute amounts and reset hint. */
function describeWindow(win: ProviderUsageWindow): WindowParts {
  let primary = "\u2014";
  let primaryUsesPair = false;
  if (win.remainingPercent != null) {
    primary = `${formatPercent(win.remainingPercent)} left`;
  } else if (win.remaining != null) {
    primary = `${formatUsageValue(win.remaining, win.unit)} left`;
  } else if (win.usedPercent != null) {
    primary = `${formatPercent(win.usedPercent)} used`;
  } else if (win.used != null && win.limit != null) {
    primary = `${formatUsageValue(win.used, win.unit)} / ${formatUsageValue(win.limit, win.unit)}`;
    primaryUsesPair = true;
  } else if (win.used != null) {
    primary = `${formatUsageValue(win.used, win.unit)} used`;
  } else if (win.limit != null) {
    primary = `${formatUsageValue(win.limit, win.unit)} limit`;
  }

  let amounts: string | null = null;
  if (!primaryUsesPair && !isPercentUnit(win.unit) && win.limit != null) {
    const usedValue =
      win.used ??
      (win.remaining != null ? win.limit - win.remaining : null);
    if (usedValue != null) {
      amounts = `${formatUsageValue(usedValue, win.unit)} of ${formatUsageValue(win.limit, win.unit)} used`;
    }
  }

  return { primary, amounts, reset: formatResetLabel(win) };
}

/** Fraction (0..1) of headroom remaining, from percents or raw amounts. */
function remainingFractionOf(win: ProviderUsageWindow): number | null {
  if (win.remainingPercent != null) return clamp01(win.remainingPercent / 100);
  if (win.usedPercent != null) return clamp01(1 - win.usedPercent / 100);
  if (win.limit != null && win.limit > 0) {
    if (win.remaining != null) return clamp01(win.remaining / win.limit);
    if (win.used != null) return clamp01(1 - win.used / win.limit);
  }
  return null;
}

function barToneFor(remaining: number | null): BarTone {
  if (remaining == null) return "unknown";
  if (remaining <= 0.08) return "low";
  if (remaining <= 0.2) return "warn";
  return "ok";
}

/** Lowest remaining headroom across all available provider quotas, in percent. */
function lowestRemainingPercent(
  quotas: readonly (ProviderUsageStatus | null)[],
): number | null {
  let lowest: number | null = null;
  for (const quota of quotas) {
    if (!quota || quota.state !== "available") continue;
    for (const win of quota.windows) {
      const remaining = remainingFractionOf(win);
      if (remaining == null) continue;
      const percent = clampPercent(remaining * 100);
      if (lowest == null || percent < lowest) lowest = percent;
    }
  }
  return lowest;
}

function isPercentUnit(unit?: string | null): boolean {
  if (!unit) return false;
  const normalized = unit.trim().toLowerCase();
  return normalized === "percent" || normalized === "%";
}

function isCurrencyCode(currency?: string | null): boolean {
  return Boolean(currency && /^[a-z]{3}$/i.test(currency.trim()));
}

/** Unit-aware value: percents, currency, or compact number + unit suffix. */
function formatUsageValue(value: number, unit?: string | null): string {
  if (isPercentUnit(unit)) return formatPercent(value);
  if (isCurrencyCode(unit)) return formatMoney(value, unit);
  const formatted = formatCompactNumber(value);
  const trimmed = unit?.trim();
  return trimmed ? `${formatted} ${trimmed}` : formatted;
}

function formatBalanceValue(balance: ProviderUsageBalance): string {
  if (isCurrencyCode(balance.currency)) {
    return formatMoney(balance.amount, balance.currency);
  }
  return formatUsageValue(balance.amount, balance.unit);
}

function formatSpendParts(spend: ProviderUsageSpend): string[] {
  const parts: string[] = [];
  if (spend.today != null) parts.push(`Today ${formatMoney(spend.today, spend.currency)}`);
  if (spend.week != null) parts.push(`Week ${formatMoney(spend.week, spend.currency)}`);
  if (spend.month != null) parts.push(`Month ${formatMoney(spend.month, spend.currency)}`);
  return parts;
}

/** Currency-aware money: ISO codes use Intl, anything else falls back gently. */
function formatMoney(amount: number, currency?: string | null): string {
  if (isCurrencyCode(currency)) {
    const fractionDigits = Math.abs(amount) >= 100 ? 0 : 2;
    try {
      return new Intl.NumberFormat(undefined, {
        style: "currency",
        currency: (currency as string).toUpperCase(),
        minimumFractionDigits: fractionDigits,
        maximumFractionDigits: fractionDigits,
      }).format(amount);
    } catch {
      // Fall through to the plain rendering below.
    }
  }
  const formatted = formatCompactNumber(amount);
  const trimmed = currency?.trim();
  return trimmed && !isCurrencyCode(currency) ? `${formatted} ${trimmed}` : `$${formatted}`;
}

function formatPercent(value: number): string {
  const percent = clampPercent(value);
  if (percent > 0 && percent < 1) return "<1%";
  return `${Math.round(percent)}%`;
}

/** Compact number: 1,234 stays exact-ish, larger values collapse to K/M. */
function formatCompactNumber(value: number): string {
  if (!Number.isFinite(value)) return "0";
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${trimOneDecimal(value / 1_000_000)}M`;
  if (abs >= 10_000) return `${trimOneDecimal(value / 1_000)}K`;
  if (Number.isInteger(value)) return value.toLocaleString();
  return value.toFixed(2);
}

function trimOneDecimal(value: number): string {
  const fixed = value.toFixed(1);
  return fixed.endsWith(".0") ? fixed.slice(0, -2) : fixed;
}

/** Resolve a window's reset moment from ms or a parseable timestamp string. */
function resolveResetMs(win: ProviderUsageWindow): number | null {
  if (win.resetAtMs != null && Number.isFinite(win.resetAtMs) && win.resetAtMs > 0) {
    return win.resetAtMs;
  }
  if (win.resetAt) {
    const parsed = Date.parse(win.resetAt);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function formatResetLabel(win: ProviderUsageWindow): string | null {
  const ms = resolveResetMs(win);
  if (ms == null) return null;
  const diff = ms - Date.now();
  if (diff <= 0) return "Resets now";
  return `Resets ${formatRelativeShort(diff)}`;
}

function formatRelativeShort(diffMs: number): string {
  const minutes = Math.round(diffMs / 60_000);
  if (minutes < 1) return "in <1m";
  if (minutes < 60) return `in ${minutes}m`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `in ${hours}h`;
  const days = Math.round(hours / 24);
  return `in ${days}d`;
}

function formatUpdatedAt(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return "just now";
  const diff = Date.now() - ms;
  if (diff < 45_000) return "just now";
  const minutes = Math.round(diff / 60_000);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return `${days}d ago`;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(100, Math.max(0, value));
}

/** Index a scope's per-provider totals by lowercased provider id for lookup. */
function indexProviderTotals(scope: ScopeUsageView): Map<string, TotalsView> {
  const map = new Map<string, TotalsView>();
  for (const provider of scope.providers) {
    map.set(provider.provider.toLowerCase(), provider.totals);
  }
  return map;
}

type StatusDescriptor = { label: string; tone: Tone };

/**
 * Maps a provider status payload to a label + tone, mirroring the language used
 * by the settings provider cards so the shell and settings stay consistent.
 */
function describeStatus(
  status: ProviderStatusLike | null,
  configured: boolean,
): StatusDescriptor {
  const state = status?.connectionState ?? "disconnected";
  if (state === "connecting") return { label: "Connecting", tone: "pending" };
  if (status?.connected) return { label: "Connected", tone: "ok" };
  if (state === "error") return { label: "Needs attention", tone: "error" };
  if (configured) return { label: "Configured", tone: "pending" };
  return { label: "Not connected", tone: "off" };
}
