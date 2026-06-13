# Sinew Remote relay

This folder is the small relay reachable at the configured Sinew Remote URL
(`remote.sinew-ide.com` for official builds). It is the **only piece you host**:
the desktop app runs on your PC, and the mobile PWA is served by this relay.

Responsibilities:

- keep an outbound WebSocket from the desktop app;
- route opaque encrypted frames between a paired phone and that PC;
- serve the mobile PWA (`public/`);
- send generic Web Push notifications when the PC reports a completed turn.

It never decrypts chat content. Pairing responses and all runtime
commands/events are AEAD envelopes between the phone and the PC.

## Run locally

```bash
cd remote
npm install
npm start
# → http://localhost:8787
```

## Deploy on Railway

The clean setup for a fork is a **separate Railway service** for Sinew Remote.
Do not mount this relay inside an unrelated production API (for example a POS / iOS
backend): Remote owns `/`, `/ws`, `/healthz`, `/vapid-public-key` and its PWA
assets, and it keeps pairing/connection state in memory.

The relay is self-contained (`remote/package.json` only pulls `ws` +
`web-push`), so deploy just this folder:

1. **New Service → Deploy from GitHub repo** and pick this repository/fork.
2. **Settings → Root Directory → `remote`.** The included `remote/railway.json`
   configures Nixpacks, `npm install`, `npm start`, and the `/healthz` health check.
3. **Variables** (Settings → Variables):
   - `VAPID_PUBLIC_KEY`
   - `VAPID_PRIVATE_KEY`
   - `VAPID_SUBJECT` (optional, defaults to `mailto:security@sinew-ide.com`)
   - `PORT` is injected by Railway — do **not** set it.
4. **Networking → Public URL** gives you an HTTPS URL such as
   `https://your-sinew-remote.up.railway.app`. A custom domain is optional.
   Railway terminates TLS, so the desktop reaches it over
   `wss://your-sinew-remote.up.railway.app/ws`.
5. **Keep a single replica.** Routing state (connected PCs, pairing codes) is
   in-memory, so horizontal scaling would split a phone from its PC. One
   instance is plenty for a PC ↔ a few devices.

Health check: `GET /healthz` → `{ "ok": true }`.

### VAPID keys (push notifications)

```bash
npx web-push generate-vapid-keys
```

Put the public/private keys in the Railway variables above. Without them the
relay still works — only push notifications are disabled (chat, streaming and
pairing are unaffected).

## Point a forked desktop build at your relay

Official builds fall back to `wss://remote.sinew-ide.com/ws`. Forked builds can
set a compile-time default without editing Rust source:

```bash
SINEW_REMOTE_RELAY_URL="https://your-sinew-remote.up.railway.app" npm run tauri build
# or provide the WebSocket URL directly:
SINEW_REMOTE_RELAY_URL="wss://your-sinew-remote.up.railway.app/ws" npm run tauri build
```

The desktop normalizes `https://...` to `wss://.../ws`. If an existing install
still has the old official default saved, a forked build with
`SINEW_REMOTE_RELAY_URL` migrates that default to your configured relay. Manually
customized relay URLs are preserved.

## No database / no volume

The relay is stateless: nothing is persisted, no DB, no disk. On restart both
the desktop and the phone reconnect automatically and resume streaming.
