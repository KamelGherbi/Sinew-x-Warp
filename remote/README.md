# Sinew Remote relay

This folder contains the small relay used by `remote.sinew-ide.com`.

Responsibilities:

- keep an outbound WebSocket from the desktop app;
- route opaque encrypted frames between a paired phone and that PC;
- serve the mobile PWA;
- send generic Web Push notifications when the PC reports a completed turn.

Run locally:

```bash
npm run remote
```

Production environment variables:

- `PORT` (Railway sets it automatically)
- `VAPID_PUBLIC_KEY`
- `VAPID_PRIVATE_KEY`
- `VAPID_SUBJECT` (defaults to `mailto:security@sinew-ide.com`)

Generate VAPID keys with:

```bash
npx web-push generate-vapid-keys
```

The relay intentionally never decrypts chat content. Pairing responses and all runtime commands/events are AEAD envelopes between the phone and PC.
