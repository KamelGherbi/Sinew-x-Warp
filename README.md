# Sinew

A desktop IDE with built-in AI coding agents. Tauri 2 + React + Rust workspace.

## Features

- Multi-provider AI chat: Anthropic, OpenAI, Google, Kimi 
- Workspace-aware agent tools: read / patch, grep, glob, bash, web fetch, MCP
- Multi-agent teams with a shared task board
- Monaco editor and xterm terminal embedded in the app

## Requirements

- [Rust](https://rustup.rs/) 1.80+
- [Node.js](https://nodejs.org/) 20+
- Platform prerequisites for Tauri 2: see the [Tauri docs](https://tauri.app/start/prerequisites/)

## Development

```bash
bun install
bun run tauri dev
```

## Build

```bash
bun run tauri build
```

## A note on OAuth credentials

Provider OAuth client IDs (and Google's client secret) are embedded in the
source. This follows the standard practice for "installed applications" — the
same approach used by tools like `gcloud`. These credentials are not treated
as secret in this context.

## License

[MIT](./LICENSE)
