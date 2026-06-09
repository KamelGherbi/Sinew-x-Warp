Code map:
- L'agent doit garder à jour cette carte simple des fichiers à chaque création, suppression, renommage, déplacement ou modification.

.
├── .github
│   ├── assets
│   │   ├── architecture.png
│   │   ├── harness.png
│   │   ├── hero.png
│   │   ├── modes.png
│   │   ├── screenshot.png
│   │   └── swarm.png
│   └── workflows
│       ├── release.yml
│       └── security.yml
├── crates
│   ├── sinew-anthropic
│   │   ├── src
│   │   │   ├── auth.rs
│   │   │   ├── client.rs
│   │   │   ├── lib.rs
│   │   │   ├── model_info.rs
│   │   │   ├── stream.rs
│   │   │   └── wire.rs
│   │   └── Cargo.toml
│   ├── sinew-app
│   │   ├── src
│   │   │   ├── agent
│   │   │   │   ├── assistant_message.rs
│   │   │   │   ├── cancel.rs
│   │   │   │   ├── clean_context.rs
│   │   │   │   ├── compaction.rs
│   │   │   │   ├── context.rs
│   │   │   │   ├── events.rs
│   │   │   │   ├── history.rs
│   │   │   │   ├── mode.rs
│   │   │   │   ├── tests.rs
│   │   │   │   ├── tool_dispatch.rs
│   │   │   │   ├── tool_preflight.rs
│   │   │   │   ├── tool_summary.rs
│   │   │   │   └── turn.rs
│   │   │   ├── team
│   │   │   │   ├── agent_turns.rs
│   │   │   │   ├── context.rs
│   │   │   │   ├── descriptors.rs
│   │   │   │   ├── launch.rs
│   │   │   │   ├── live.rs
│   │   │   │   ├── messaging.rs
│   │   │   │   ├── model.rs
│   │   │   │   ├── render.rs
│   │   │   │   ├── session.rs
│   │   │   │   ├── status_stop.rs
│   │   │   │   ├── task_board.rs
│   │   │   │   └── tests.rs
│   │   │   ├── agent.rs
│   │   │   ├── bash.rs
│   │   │   ├── compact.rs
│   │   │   ├── edit.rs
│   │   │   ├── glob.rs
│   │   │   ├── grep.rs
│   │   │   ├── image.rs
│   │   │   ├── lib.rs
│   │   │   ├── mcp.rs
│   │   │   ├── question.rs
│   │   │   ├── read.rs
│   │   │   ├── ripgrep.rs
│   │   │   ├── skill.rs
│   │   │   ├── store.rs
│   │   │   ├── subagent.rs
│   │   │   ├── team.rs
│   │   │   ├── text.rs
│   │   │   ├── todo.rs
│   │   │   ├── tool_names.rs
│   │   │   ├── tool_run.rs
│   │   │   ├── web.rs
│   │   │   ├── workspace.rs
│   │   │   └── write.rs
│   │   └── Cargo.toml
│   ├── sinew-core
│   │   ├── src
│   │   │   ├── error.rs
│   │   │   ├── lib.rs
│   │   │   ├── message.rs
│   │   │   ├── model.rs
│   │   │   ├── provider.rs
│   │   │   ├── stream.rs
│   │   │   └── tool.rs
│   │   └── Cargo.toml
│   ├── sinew-google
│   │   ├── src
│   │   │   ├── auth.rs
│   │   │   ├── client.rs
│   │   │   ├── lib.rs
│   │   │   ├── model_info.rs
│   │   │   ├── stream.rs
│   │   │   └── wire.rs
│   │   └── Cargo.toml
│   ├── sinew-kimi
│   │   ├── src
│   │   │   ├── auth.rs
│   │   │   ├── client.rs
│   │   │   ├── lib.rs
│   │   │   ├── model_info.rs
│   │   │   ├── stream.rs
│   │   │   └── wire.rs
│   │   └── Cargo.toml
│   ├── sinew-openai
│   │   ├── src
│   │   │   ├── auth.rs
│   │   │   ├── client.rs
│   │   │   ├── lib.rs
│   │   │   ├── model_info.rs
│   │   │   ├── responses_stream.rs
│   │   │   ├── stream.rs
│   │   │   ├── websocket.rs
│   │   │   └── wire.rs
│   │   └── Cargo.toml
│   └── sinew-openrouter
│       ├── src
│       │   ├── auth.rs
│       │   ├── client.rs
│       │   ├── lib.rs
│       │   ├── model_info.rs
│       │   ├── stream.rs
│       │   └── wire.rs
│       └── Cargo.toml
├── resources
│   └── skills
│       ├── apex
│       │   └── SKILL.md
│       ├── prompt-creator
│       │   └── SKILL.md
│       ├── skill-creator
│       │   └── SKILL.md
│       └── subagent-creator
│           └── SKILL.md
├── scripts
│   ├── prepare-sidecars.mjs
│   └── tauri-cli.mjs
├── src
│   ├── components
│   │   ├── chat
│   │   │   ├── AIThinkingBlock.tsx
│   │   │   ├── ChatPane.tsx
│   │   │   ├── dotmatrix-core.tsx
│   │   │   ├── dotmatrix-hooks.ts
│   │   │   ├── DotmSquare2.tsx
│   │   │   ├── DotmSquare5.tsx
│   │   │   ├── FileChangeBlock.tsx
│   │   │   ├── Markdown.tsx
│   │   │   ├── MermaidDiagram.tsx
│   │   │   ├── PlanningNextMoveBlock.tsx
│   │   │   ├── Questionnaire.tsx
│   │   │   ├── stream.ts
│   │   │   ├── TodoStrip.tsx
│   │   │   └── ToolCard.tsx
│   │   ├── ConversationList.tsx
│   │   ├── EditorPane.tsx
│   │   ├── FileTree.tsx
│   │   ├── GitPanel.tsx
│   │   ├── ImageContextMenu.tsx
│   │   ├── SearchPane.tsx
│   │   ├── SessionSwitcher.tsx
│   │   ├── SettingsPane.tsx
│   │   ├── SinewMark.tsx
│   │   ├── Splitter.tsx
│   │   ├── TerminalPanel.tsx
│   │   ├── UpdateBadge.tsx
│   │   ├── UpdaterLockScreen.tsx
│   │   ├── Welcome.tsx
│   │   ├── WindowControls.tsx
│   │   └── Workspace.tsx
│   ├── lib
│   │   ├── appearance.ts
│   │   ├── customIcons.ts
│   │   ├── fileIcon.ts
│   │   ├── ipc.ts
│   │   ├── language.ts
│   │   ├── models.ts
│   │   ├── modelVisibility.ts
│   │   ├── recents.ts
│   │   ├── sessions.ts
│   │   ├── subscriptionUsage.ts
│   │   └── tools.ts
│   ├── App.tsx
│   ├── main.tsx
│   ├── styles.css
│   ├── types.ts
│   └── vite-env.d.ts
├── src-tauri
│   ├── binaries
│   │   ├── .gitkeep
│   │   ├── rg-aarch64-apple-darwin
│   │   ├── rg-universal-apple-darwin
│   │   └── rg-x86_64-apple-darwin
│   ├── capabilities
│   │   └── default.json
│   ├── gen
│   │   └── schemas
│   │       ├── acl-manifests.json
│   │       ├── capabilities.json
│   │       ├── desktop-schema.json
│   │       └── macOS-schema.json
│   ├── icons
│   │   ├── android
│   │   │   ├── mipmap-anydpi-v26
│   │   │   │   └── ic_launcher.xml
│   │   │   ├── mipmap-hdpi
│   │   │   │   ├── ic_launcher.png
│   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   └── ic_launcher_round.png
│   │   │   ├── mipmap-mdpi
│   │   │   │   ├── ic_launcher.png
│   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   └── ic_launcher_round.png
│   │   │   ├── mipmap-xhdpi
│   │   │   │   ├── ic_launcher.png
│   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   └── ic_launcher_round.png
│   │   │   ├── mipmap-xxhdpi
│   │   │   │   ├── ic_launcher.png
│   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   └── ic_launcher_round.png
│   │   │   ├── mipmap-xxxhdpi
│   │   │   │   ├── ic_launcher.png
│   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   └── ic_launcher_round.png
│   │   │   └── values
│   │   │       └── ic_launcher_background.xml
│   │   ├── ios
│   │   │   ├── AppIcon-20x20@1x.png
│   │   │   ├── AppIcon-20x20@2x-1.png
│   │   │   ├── AppIcon-20x20@2x.png
│   │   │   ├── AppIcon-20x20@3x.png
│   │   │   ├── AppIcon-29x29@1x.png
│   │   │   ├── AppIcon-29x29@2x-1.png
│   │   │   ├── AppIcon-29x29@2x.png
│   │   │   ├── AppIcon-29x29@3x.png
│   │   │   ├── AppIcon-40x40@1x.png
│   │   │   ├── AppIcon-40x40@2x-1.png
│   │   │   ├── AppIcon-40x40@2x.png
│   │   │   ├── AppIcon-40x40@3x.png
│   │   │   ├── AppIcon-512@2x.png
│   │   │   ├── AppIcon-60x60@2x.png
│   │   │   ├── AppIcon-60x60@3x.png
│   │   │   ├── AppIcon-76x76@1x.png
│   │   │   ├── AppIcon-76x76@2x.png
│   │   │   └── AppIcon-83.5x83.5@2x.png
│   │   ├── 128x128.png
│   │   ├── 128x128@2x.png
│   │   ├── 32x32.png
│   │   ├── 64x64.png
│   │   ├── icon.icns
│   │   ├── icon.ico
│   │   ├── icon.png
│   │   ├── nsis-sidebar.bmp
│   │   ├── source.svg
│   │   ├── Square107x107Logo.png
│   │   ├── Square142x142Logo.png
│   │   ├── Square150x150Logo.png
│   │   ├── Square284x284Logo.png
│   │   ├── Square30x30Logo.png
│   │   ├── Square310x310Logo.png
│   │   ├── Square44x44Logo.png
│   │   ├── Square71x71Logo.png
│   │   ├── Square89x89Logo.png
│   │   └── StoreLogo.png
│   ├── src
│   │   ├── context.rs
│   │   ├── conversations.rs
│   │   ├── git.rs
│   │   ├── lib.rs
│   │   ├── main.rs
│   │   ├── models.rs
│   │   ├── platform.rs
│   │   ├── providers.rs
│   │   ├── state.rs
│   │   ├── swarm.rs
│   │   ├── terminal.rs
│   │   ├── tests.rs
│   │   ├── turns.rs
│   │   ├── updater.rs
│   │   ├── vibe_island.rs
│   │   ├── workflow.rs
│   │   └── workspace.rs
│   ├── build.rs
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   ├── tauri.sidecars.conf.json
│   └── tauri.windows.conf.json
├── .DS_Store
├── .gitignore
├── AGENTS.md
├── Cargo.lock
├── Cargo.toml
├── index.html
├── LICENSE
├── package-lock.json
├── package.json
├── README.md
├── settings.json
├── test-stop.md
├── tsconfig.json
├── tsconfig.node.json
└── vite.config.ts
