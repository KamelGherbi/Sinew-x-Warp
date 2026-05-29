import type { ModelEntry } from "./models";

// Lets the user hide individual models from the model selectors (chat composer,
// sub-agents, …) without disconnecting a whole provider.
//
// We persist a *blocklist* of hidden model `value`s (e.g. "openai:gpt-5.2" or
// "openrouter:openai/gpt-4o"). An empty list means every model is visible — so
// the app behaves exactly as before until the user explicitly hides something.
// New models added later (including freshly added OpenRouter models) are visible
// by default because they are simply absent from the blocklist. "Show all" just
// clears the list.

export const MODEL_VISIBILITY_CHANGED_EVENT = "sinew:model-visibility-changed";

const STORAGE_KEY = "sinew.models.visibility";

export type ModelVisibilitySettings = {
  hidden: string[];
};

export const DEFAULT_MODEL_VISIBILITY: ModelVisibilitySettings = { hidden: [] };

export function loadModelVisibility(): ModelVisibilitySettings {
  try {
    if (typeof window === "undefined") return clone(DEFAULT_MODEL_VISIBILITY);
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return clone(DEFAULT_MODEL_VISIBILITY);
    return normalizeModelVisibility(JSON.parse(raw));
  } catch {
    return clone(DEFAULT_MODEL_VISIBILITY);
  }
}

// Persists the settings and notifies the rest of the window so open selectors
// (chat composer, sub-agents) refresh live. localStorage `storage` events don't
// fire in the same document, hence the custom event.
export function saveModelVisibility(settings: ModelVisibilitySettings): void {
  const normalized = normalizeModelVisibility(settings);
  try {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(normalized));
  } catch {
    // Storage can fail in private mode or when quota is exhausted; the live
    // event below still keeps the current session in sync.
  }
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(MODEL_VISIBILITY_CHANGED_EVENT));
  }
}

export function normalizeModelVisibility(value: unknown): ModelVisibilitySettings {
  const record = isRecord(value) ? value : {};
  const raw = Array.isArray(record.hidden) ? record.hidden : [];
  const hidden = Array.from(
    new Set(
      raw.filter((item): item is string => typeof item === "string" && item.length > 0),
    ),
  );
  return { hidden };
}

export function isModelHidden(
  value: string,
  settings: ModelVisibilitySettings,
): boolean {
  return settings.hidden.includes(value);
}

export function isModelVisible(
  value: string,
  settings: ModelVisibilitySettings,
): boolean {
  return !isModelHidden(value, settings);
}

// Returns a new settings object with `value` hidden or shown. Pure — callers are
// responsible for persisting the result via saveModelVisibility.
export function setModelHidden(
  value: string,
  hidden: boolean,
  settings: ModelVisibilitySettings,
): ModelVisibilitySettings {
  return setModelsHidden([value], hidden, settings);
}

// Bulk variant used by the per-provider "show all / hide all" toggles.
export function setModelsHidden(
  values: readonly string[],
  hidden: boolean,
  settings: ModelVisibilitySettings,
): ModelVisibilitySettings {
  const next = new Set(settings.hidden);
  for (const value of values) {
    if (hidden) next.add(value);
    else next.delete(value);
  }
  return { hidden: Array.from(next) };
}

export function resetModelVisibility(): ModelVisibilitySettings {
  return clone(DEFAULT_MODEL_VISIBILITY);
}

// Filters a model list down to the visible ones. With no hidden entries it
// returns a shallow copy of the input untouched, so callers can use it
// unconditionally without changing legacy behaviour.
export function filterVisibleModels(
  models: readonly ModelEntry[],
  settings: ModelVisibilitySettings,
): ModelEntry[] {
  if (settings.hidden.length === 0) return [...models];
  const hidden = new Set(settings.hidden);
  return models.filter((model) => !hidden.has(model.value));
}

// Subscribes to visibility changes (same-window custom event + cross-window
// storage event). Returns an unsubscribe function.
export function watchModelVisibility(onChange: () => void): () => void {
  if (typeof window === "undefined") return noop;
  const handler = () => onChange();
  const storageHandler = (event: StorageEvent) => {
    if (event.key === STORAGE_KEY) onChange();
  };
  window.addEventListener(MODEL_VISIBILITY_CHANGED_EVENT, handler);
  window.addEventListener("storage", storageHandler);
  return () => {
    window.removeEventListener(MODEL_VISIBILITY_CHANGED_EVENT, handler);
    window.removeEventListener("storage", storageHandler);
  };
}

function clone(settings: ModelVisibilitySettings): ModelVisibilitySettings {
  return { hidden: [...settings.hidden] };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function noop(): void {}
