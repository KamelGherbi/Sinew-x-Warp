<p align="center">
  <img src=".github/assets/hero.png" alt="Sinew — the coding harness you shape" width="100%" />
</p>

<p align="center">
  <a href="https://github.com/Paseru/sinew/releases/latest"><img alt="Release" src="https://img.shields.io/github/v/release/Paseru/sinew?style=flat-square&labelColor=0b0b0d&color=3b82f6"></a>
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/github/license/Paseru/sinew?style=flat-square&labelColor=0b0b0d&color=c4b5fd"></a>
  <a href="https://github.com/Paseru/sinew/actions"><img alt="Build" src="https://img.shields.io/github/actions/workflow/status/Paseru/sinew/release.yml?style=flat-square&labelColor=0b0b0d&color=3b82f6&label=build"></a>
  <a href="https://github.com/Paseru/sinew/releases"><img alt="Downloads" src="https://img.shields.io/github/downloads/Paseru/sinew/total?style=flat-square&labelColor=0b0b0d&color=c4b5fd"></a>
  <a href="https://discord.gg/MADQNHtZW"><img alt="Discord" src="https://img.shields.io/badge/discord-join-3b82f6?style=flat-square&labelColor=0b0b0d&logo=discord&logoColor=white"></a>
</p>

<p align="center">
  <b>Sinew</b> is a desktop AI coding harness you can actually reshape.<br/>
  Rewrite the description of every tool, turn the ones you don't need off,<br/>
  and the assistant only sees what you keep.
</p>

<p align="center">
  <code>tauri 2</code> · <code>react</code> · <code>rust</code> · <code>monaco</code> · <code>xterm</code> · <code>mcp</code>
</p>

---

## Why Sinew

Most AI coding tools ship a fixed harness: a hard-coded prompt, a hard-coded set
of tools, a hard-coded loop. You get whatever the vendor decided is "right".

Sinew flips that. The harness is the surface area you control.

- **Every tool description is editable.** Rephrase it, scope it down, change the contract.
- **Every tool is toggleable.** Run minimal like Pi, or unlock the full set.
- **Every provider is pluggable.** Same agent loop across Anthropic, OpenAI, Google, Kimi, OpenRouter.
- **Agents can run in swarms.** Multiple sub-agents, one shared task board, message passing.
- **It's a real IDE.** Monaco editor and xterm terminal, not a chat box with a file picker.

---

## Shape the harness

<p align="center">
  <img src=".github/assets/harness.png" alt="Minimal harness vs full harness" width="100%" />
</p>

The same agent loop can behave like a barebones pair-programmer or a full
autonomous coding crew. You choose. Tool descriptions live in settings,
versioned with your config, and the assistant gets exactly the surface area
you decide on.

```
Minimal           →   read · patch · bash
Search-capable    →   + grep · glob
Web-aware         →   + WebSearch · WebFetch
Full              →   + image · mcp · todo · question · skill · subagent · team
```

No magic. No hidden tools. No prompt you can't see.

---

## Agent swarm

<p align="center">
  <img src=".github/assets/swarm.png" alt="Agent swarm with shared task board" width="100%" />
</p>

Launch a team of sub-agents from a single prompt. Each agent gets its own
context, its own role, its own tool subset. They coordinate through a shared
task board and `SendMessage`. The main agent stays in the loop, can watch the
board update live, stop any teammate, or hand work off.

- **Roles.** Architect, frontend, backend, tester — or whatever you define.
- **Task board.** Real dependencies, real ownership, real status.
- **Messaging.** Direct (`@name`) or broadcast (`*`).
- **Live view.** Streamed into the chat as cards, not buried in logs.

---

## Multi-provider, one loop

<p align="center">
  <a href="https://www.anthropic.com"><img alt="Anthropic" src="https://img.shields.io/badge/Anthropic-Claude-c4b5fd?style=for-the-badge&labelColor=0b0b0d&logo=anthropic&logoColor=c4b5fd"></a>
  <a href="https://openai.com"><img alt="OpenAI" src="https://img.shields.io/badge/OpenAI-GPT-e8e9ec?style=for-the-badge&labelColor=0b0b0d&logo=openai&logoColor=e8e9ec"></a>
  <a href="https://ai.google.dev"><img alt="Google" src="https://img.shields.io/badge/Google-Gemini-3b82f6?style=for-the-badge&labelColor=0b0b0d&logo=google&logoColor=3b82f6"></a>
  <a href="https://www.moonshot.cn"><img alt="Kimi" src="https://img.shields.io/badge/Moonshot-Kimi-c4b5fd?style=for-the-badge&labelColor=0b0b0d"></a>
  <a href="https://openrouter.ai"><img alt="OpenRouter" src="https://img.shields.io/badge/OpenRouter-Any%20model-9aa0a8?style=for-the-badge&labelColor=0b0b0d"></a>
</p>

OAuth where possible, API keys where it makes sense. Switch provider mid-project
without changing your tools, your prompt, or your workflow.

---

## Built-in tools

| Tool | Purpose |
|------|---------|
| `read` | Read text files or attach images visually |
| `apply_patch` | Create, update, delete, rename files via a strict patch format |
| `grep` | Workspace-wide text / regex search |
| `Glob` | Find files by glob pattern |
| `bash` | Run shell commands in a managed session |
| `WebSearch` / `WebFetch` | Hit the web, read the page |
| `CreateImage` | Generate images (OpenAI Image, Nano Banana) |
| `ToDoList` | Maintain a structured task list across turns |
| `Question` | Ask the user single / multiple-choice questions |
| `mcp` | Bind any Model Context Protocol server as tools |
| `LoadMcpTool` | Lazy-load MCP tools on demand |
| `Skill` | Long-form, on-disk skills the agent can invoke |
| `subagent_*` | Delegate to a configured sub-agent |
| `TeamRun` / `TeamStatus` / `TeamStop` | Drive the agent swarm |

All of them are listed in Settings, every description is a textarea, every
row has an on/off toggle.

---

## Screenshot

<p align="center">
  <img src=".github/assets/screenshot.png" alt="Sinew IDE" width="100%" />
</p>

---

## Architecture

<p align="center">
  <img src=".github/assets/architecture.png" alt="Sinew architecture" width="100%" />
</p>

- **`src/`** — React UI (Monaco, xterm, chat, settings, file tree).
- **`src-tauri/`** — Tauri 2 shell, IPC commands, workspace I/O, conversation store.
- **`crates/sinew-core`** — Provider-agnostic types: messages, tools, streams.
- **`crates/sinew-app`** — Agent loop, tool implementations, swarm, MCP, compaction.
- **`crates/sinew-{anthropic,openai,google,kimi,openrouter}`** — Provider adapters (auth, wire, streaming).

---

## How it compares

| | **Sinew** | Cursor | Claude Code | Aider | Zed AI |
|---|:---:|:---:|:---:|:---:|:---:|
| Native desktop app | ✓ | ✓ | — (CLI) | — (CLI) | ✓ |
| Open source | ✓ | — | — | ✓ | ✓ |
| Multi-provider | ✓ | ✓ | — | ✓ | ✓ |
| Editable tool descriptions | ✓ | — | — | — | — |
| Toggle individual tools | ✓ | partial | — | — | — |
| Agent swarm + task board | ✓ | — | — | — | — |
| MCP servers | ✓ | ✓ | ✓ | — | partial |
| Embedded terminal | ✓ | ✓ | n/a | n/a | ✓ |

---

## Install

Grab the latest build for your OS from the
[releases page](https://github.com/Paseru/sinew/releases/latest).

- **macOS** — `.dmg`
- **Windows** — `.msi` / `.exe`
- **Linux** — `.AppImage` / `.deb`

The app self-updates from GitHub releases.

---

## Build from source

```bash
# requires Rust 1.80+ and Node 20+
# see https://tauri.app/start/prerequisites/ for platform deps

npm install
npm run tauri dev      # development
npm run tauri build    # release bundle
```

The repo is a Cargo workspace (`crates/*` + `src-tauri/`) plus a Vite + React
frontend (`src/`).

---

## OAuth credentials

Provider OAuth client IDs (and Google's client secret) are embedded in the
source. This follows the standard practice for "installed applications" — the
same approach used by tools like `gcloud`. These credentials are not treated
as secret in this context.

---

## Community

- [Discord](https://discord.gg/MADQNHtZW) — chat, support, share your harness configs
- [Issues](https://github.com/Paseru/sinew/issues) — bugs and feature requests
- [Discussions](https://github.com/Paseru/sinew/discussions) — design, providers, MCP

---

## License

[MIT](./LICENSE)

<p align="center">
  <sub>Built with Tauri, Rust, and a stubborn refusal to ship a black-box harness.</sub>
</p>
