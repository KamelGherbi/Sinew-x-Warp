import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";
import "./lib/customIcons";
import { api } from "./lib/ipc";

const MESSAGE_FONT_SIZE_STORAGE_KEY = "sinew.appearance.messageFontSize";
const DEFAULT_MESSAGE_FONT_SIZE = 12;
const MIN_MESSAGE_FONT_SIZE = 11;
const MAX_MESSAGE_FONT_SIZE = 18;

applyStoredMessageFontSize();

// Suppress the native WebKit context menu everywhere except inside text
// inputs (where the OS-level copy/paste menu is still useful). Components
// that want a context menu must intercept the event themselves and call
// `event.preventDefault()` *before* the listener below — they then render
// their own custom menu (this is how Monaco and our own ImageContextMenu
// behave). This mirrors what VSCode does: WebKit's menu is half-broken
// inside an embedded WKWebView (no download, no "open in new window", no
// share), so we hide it and serve our own actions.
window.addEventListener(
  "contextmenu",
  (event) => {
    if (event.defaultPrevented) return;
    const target = event.target as HTMLElement | null;
    if (!target) {
      event.preventDefault();
      return;
    }
    if (target.closest("input, textarea, [contenteditable=\"true\"], [contenteditable=\"\"]")) {
      return;
    }
    event.preventDefault();
  },
  { capture: false },
);

// Route every left or middle click on an `<a href="http(s)://…">` anchor
// through Tauri's `open_external_url` command. Without this, plain anchors
// silently fail inside the WKWebView/wry shell:
//   • `target="_blank"` needs `webView(_:createWebViewWith:…)` to be wired
//     up on the native side, which wry does not do by default.
//   • a same-window navigation to https:// gets blocked by Tauri's default
//     navigation policy.
// So any component that just writes `<a href="https://…">` (Discord and
// GitHub buttons in Settings, markdown links, …) gets a working handler
// for free instead of having to remember to wire `api.openExternalUrl` by
// hand.
const openAnchorExternally = (event: MouseEvent) => {
  if (event.defaultPrevented) return;
  // `click` only fires for the primary (left) button; `auxclick` is used
  // for the middle button. We deliberately ignore right-click here — that
  // path goes through the contextmenu handler above.
  if (event.type === "auxclick" && event.button !== 1) return;
  const target = event.target;
  if (!(target instanceof Element)) return;
  const anchor = target.closest("a");
  if (!anchor) return;
  const href = anchor.getAttribute("href");
  if (!href) return;
  const trimmed = href.trim();
  if (!/^https?:\/\//i.test(trimmed)) return;
  event.preventDefault();
  void api
    .openExternalUrl(trimmed)
    .catch((err) => console.error("[external-link] failed to open", trimmed, err));
};
window.addEventListener("click", openAnchorExternally);
window.addEventListener("auxclick", openAnchorExternally);

window.addEventListener("sinew:message-font-size-changed", applyStoredMessageFontSize);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

function applyStoredMessageFontSize(): void {
  document.documentElement.style.setProperty(
    "--message-font-size",
    `${loadMessageFontSize()}px`,
  );
}

function loadMessageFontSize(): number {
  try {
    return normalizeMessageFontSize(
      window.localStorage.getItem(MESSAGE_FONT_SIZE_STORAGE_KEY),
    );
  } catch {
    return DEFAULT_MESSAGE_FONT_SIZE;
  }
}

function normalizeMessageFontSize(value: unknown): number {
  const numberValue =
    typeof value === "number" ? value : Number.parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(numberValue)) return DEFAULT_MESSAGE_FONT_SIZE;
  return Math.min(
    MAX_MESSAGE_FONT_SIZE,
    Math.max(MIN_MESSAGE_FONT_SIZE, Math.round(numberValue)),
  );
}
