import { useCallback, useEffect, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { Welcome } from "./components/Welcome";
import { Workspace } from "./components/Workspace";
import { UpdaterLockScreen } from "./components/UpdaterLockScreen";
import { loadLastWorkspace, recordRecent, deriveName } from "./lib/recents";
import { api } from "./lib/ipc";
import { workspaceSessionKey } from "./lib/sessions";
import type {
  ConversationSummary,
  RemoteOpenConversationInput,
  UpdateInfo,
  WorkspaceBootstrap,
  WorkspaceSession,
} from "./types";

type AppState =
  | { kind: "boot" }
  | { kind: "update_available"; info: UpdateInfo; autoInstall: boolean }
  | { kind: "welcome" }
  | {
      kind: "workspace";
      sessions: WorkspaceSession[];
      activeSessionKey: string;
    };

const startsEmpty =
  new URLSearchParams(window.location.search).get("newWindow") === "1";

/// Maximum time we wait on the boot updater check before falling through to
/// the normal flow. Keeps the app responsive on flaky networks — if the
/// update endpoint is unreachable we don't trap the user on a black canvas.
const BOOT_CHECK_TIMEOUT_MS = 4000;

async function loadStartupState(): Promise<AppState> {
  if (startsEmpty) {
    return { kind: "welcome" };
  }

  const last = loadLastWorkspace();
  if (!last) {
    return { kind: "welcome" };
  }

  try {
    const bootstrap = await api.openWorkspace(last);
    recordRecent(bootstrap.workspace.path, bootstrap.workspace.name);
    return {
      kind: "workspace",
      sessions: [sessionFromBootstrap(bootstrap)],
      activeSessionKey: sessionKeyFromBootstrap(bootstrap),
    };
  } catch {
    return { kind: "welcome" };
  }
}

export default function App() {
  const [state, setState] = useState<AppState>({ kind: "boot" });
  const [bootError, setBootError] = useState<string | null>(null);
  const stateRef = useRef(state);
  const remoteOpenConversationsSignatureRef = useRef("");

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    if (state.kind === "boot") return;
    const conversations =
      state.kind === "workspace"
        ? remoteOpenConversationsFromSessions(
            state.sessions,
            state.activeSessionKey,
          )
        : [];
    const signature = JSON.stringify(conversations);
    if (signature === remoteOpenConversationsSignatureRef.current) return;
    remoteOpenConversationsSignatureRef.current = signature;
    void api.remoteSetOpenConversations(conversations).catch((err) => {
      console.warn("remote open conversations unavailable", err);
    });
  }, [state]);

  const openWorkspace = useCallback(async (path: string) => {
    setBootError(null);
    try {
      const bootstrap = await api.openWorkspace(path);
      recordRecent(bootstrap.workspace.path, bootstrap.workspace.name);
      const activeSessionKey = sessionKeyFromBootstrap(bootstrap);
      setState((current) => ({
        kind: "workspace",
        sessions: upsertBootstrap(
          current.kind === "workspace" ? current.sessions : [],
          bootstrap,
        ),
        activeSessionKey,
      }));
    } catch (err) {
      setBootError(String(err));
    }
  }, []);

  const continueWithoutUpdate = useCallback(async () => {
    setBootError(null);
    setState({ kind: "boot" });
    const nextState = await loadStartupState();
    setState(nextState);
  }, []);

  const pickAndOpenWorkspace = useCallback(async () => {
    const selected = await open({ directory: true, multiple: false });
    if (typeof selected === "string") {
      await openWorkspace(selected);
    }
  }, [openWorkspace]);

  // Boot sequence, in order:
  //   1. Updater notice — race the check against a short timeout. If an
  //      update is available we render <UpdaterLockScreen /> with a "Later"
  //      path, so the user can keep using the app without installing.
  //   2. Auto-open last workspace (existing behaviour) when no update is
  //      pending, or after the user skips the update. Silent fallback to
  //      Welcome on any failure.
  // The whole thing runs once at mount; the in-session <UpdateBadge />
  // still handles mid-session checks via its own 30 min interval.
  useEffect(() => {
    let cancelled = false;

    (async () => {
      // 1. Updater notice.
      try {
        const info = await Promise.race<UpdateInfo | null>([
          api.checkForUpdate(),
          new Promise<null>((resolve) =>
            window.setTimeout(() => resolve(null), BOOT_CHECK_TIMEOUT_MS),
          ),
        ]);
        if (cancelled) return;
        if (info && info.available && info.version) {
          setState({ kind: "update_available", info, autoInstall: false });
          return;
        }
      } catch {
        // Silent: a failed check (offline, server down, manifest 5xx)
        // shouldn't prevent the app from booting. The mid-session badge
        // will retry later, and the next launch will check again.
      }

      // 2. Auto-open last workspace, falling back to Welcome.
      if (cancelled) return;
      const nextState = await loadStartupState();
      if (!cancelled) setState(nextState);
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // Mid-session install: when the <UpdateBadge /> in Workspace fires
  // "sinew:install-update" (user clicked "Install & restart" in the
  // popover), we swap the whole window to the updater screen with
  // `autoInstall` enabled. From there the screen runs the download
  // → install → auto-restart flow and stays modal until the update is
  // applied or the user quits.
  useEffect(() => {
    const handler = (event: WindowEventMap["sinew:install-update"]) => {
      const info = event.detail?.info;
      if (!info || !info.available || !info.version) return;
      setState({ kind: "update_available", info, autoInstall: true });
    };
    window.addEventListener("sinew:install-update", handler);
    return () => window.removeEventListener("sinew:install-update", handler);
  }, []);

  const backToWelcome = useCallback(() => {
    void api.resetWindowTitle().catch(() => {
      // best-effort; leaving the previous title is harmless
    });
    setState({ kind: "welcome" });
  }, []);

  const replaceBootstrap = useCallback((bootstrap: WorkspaceBootstrap) => {
    const activeSessionKey = sessionKeyFromBootstrap(bootstrap);
    setState((current) => ({
      kind: "workspace",
      sessions: upsertBootstrap(
        current.kind === "workspace" ? current.sessions : [],
        bootstrap,
      ),
      activeSessionKey,
    }));
  }, []);

  const updateWorkspaceConversations = useCallback(
    (workspacePath: string, conversations: ConversationSummary[]) => {
      setState((current) => {
        if (current.kind !== "workspace") return current;
        return {
          ...current,
          sessions: current.sessions.map((session) =>
            session.workspacePath === workspacePath
              ? {
                  ...session,
                  bootstrap: {
                    ...session.bootstrap,
                    conversations,
                  },
                }
              : session,
          ),
        };
      });
    },
    [],
  );

  const selectSession = useCallback(
    async (workspacePath: string, conversationId: string) => {
      const key = workspaceSessionKey(workspacePath, conversationId);
      const current = stateRef.current;
      let base: WorkspaceSession | undefined;
      if (current.kind === "workspace") {
        const existing = current.sessions.find((session) => session.key === key);
        base =
          existing ??
          current.sessions.find((session) => session.workspacePath === workspacePath);
        if (existing) {
          setState({ ...current, activeSessionKey: key });
        }
      }

      try {
        if (!base) {
          const opened = await api.openWorkspace(workspacePath);
          recordRecent(opened.workspace.path, opened.workspace.name);
          base = sessionFromBootstrap(opened);
        }

        const [activeConversation, conversations] = await Promise.all([
          api.loadConversation(workspacePath, conversationId),
          api.listConversations(workspacePath),
        ]);
        if (
          activeConversation.id !== conversationId ||
          activeConversation.workspaceId !== workspacePath
        ) {
          return;
        }

        const bootstrap: WorkspaceBootstrap = {
          ...base.bootstrap,
          conversations,
          activeConversation,
        };
        setState((nextCurrent) => ({
          kind: "workspace",
          sessions: upsertBootstrap(
            nextCurrent.kind === "workspace" ? nextCurrent.sessions : [],
            bootstrap,
          ),
          activeSessionKey: key,
        }));
      } catch (err) {
        console.error(err);
      }
    },
    [],
  );

  const createConversationSession = useCallback(async (workspacePath?: string) => {
    const current = stateRef.current;
    const targetWorkspacePath =
      workspacePath ??
      (current.kind === "workspace"
        ? current.sessions.find(
            (session) => session.key === current.activeSessionKey,
          )?.workspacePath
        : undefined);
    if (!targetWorkspacePath) return;

    try {
      const bootstrap = await api.createConversation(targetWorkspacePath);
      recordRecent(bootstrap.workspace.path, bootstrap.workspace.name);
      replaceBootstrap(bootstrap);
    } catch (err) {
      console.error(err);
    }
  }, [replaceBootstrap]);

  const renameConversationSession = useCallback(
    async (workspacePath: string, conversationId: string, title: string) => {
      try {
        const conversations = await api.renameConversation(
          workspacePath,
          conversationId,
          title,
        );
        setState((current) => {
          if (current.kind !== "workspace") return current;
          return {
            ...current,
            sessions: current.sessions.map((session) =>
              session.workspacePath === workspacePath
                ? {
                    ...session,
                    bootstrap: {
                      ...session.bootstrap,
                      conversations,
                      activeConversation:
                        session.conversationId === conversationId
                          ? { ...session.bootstrap.activeConversation, title }
                          : session.bootstrap.activeConversation,
                    },
                  }
                : session,
            ),
          };
        });
      } catch (err) {
        console.error(err);
      }
    },
    [],
  );

  const deleteConversationSession = useCallback(
    async (workspacePath: string, conversationId: string) => {
      try {
        const conversations = await api.deleteConversation(workspacePath, conversationId);
        const deletedKey = workspaceSessionKey(workspacePath, conversationId);
        setState((current) => {
          const remainingSessions =
            current.kind === "workspace"
              ? current.sessions.filter((session) => session.key !== deletedKey)
              : [];
          const sessions = remainingSessions.map((session) =>
            session.workspacePath === workspacePath
              ? {
                  ...session,
                  bootstrap: {
                    ...session.bootstrap,
                    conversations,
                  },
                }
              : session,
          );
          if (sessions.length === 0) return { kind: "welcome" };
          const activeSessionKey =
            current.kind === "workspace" &&
            sessions.some((session) => session.key === current.activeSessionKey)
              ? current.activeSessionKey
              : sessions[0].key;
          return { kind: "workspace", sessions, activeSessionKey };
        });
      } catch (err) {
        console.error(err);
      }
    },
    [],
  );

  const archiveConversationSession = useCallback(
    async (workspacePath: string, conversationId: string) => {
      try {
        const conversations = await api.archiveConversation(workspacePath, conversationId);
        const archivedKey = workspaceSessionKey(workspacePath, conversationId);
        setState((current) => {
          const remainingSessions =
            current.kind === "workspace"
              ? current.sessions.filter((session) => session.key !== archivedKey)
              : [];
          const sessions = remainingSessions.map((session) =>
            session.workspacePath === workspacePath
              ? {
                  ...session,
                  bootstrap: {
                    ...session.bootstrap,
                    conversations,
                  },
                }
              : session,
          );
          if (sessions.length === 0) return { kind: "welcome" };
          const activeSessionKey =
            current.kind === "workspace" &&
            sessions.some((session) => session.key === current.activeSessionKey)
              ? current.activeSessionKey
              : sessions[0].key;
          return { kind: "workspace", sessions, activeSessionKey };
        });
      } catch (err) {
        console.error(err);
      }
    },
    [],
  );

  const restoreConversationSession = useCallback(
    async (workspacePath: string, conversationId: string) => {
      try {
        const conversations = await api.restoreConversation(workspacePath, conversationId);
        updateWorkspaceConversations(workspacePath, conversations);
      } catch (err) {
        console.error(err);
      }
    },
    [updateWorkspaceConversations],
  );

  const closeProjectSession = useCallback((workspacePath: string) => {
    setState((current) => {
      if (current.kind !== "workspace") return current;
      const sessions = current.sessions.filter(
        (session) => session.workspacePath !== workspacePath,
      );
      if (sessions.length === 0) return { kind: "welcome" };
      const activeStillOpen = sessions.some(
        (session) => session.key === current.activeSessionKey,
      );
      return {
        kind: "workspace",
        sessions,
        activeSessionKey: activeStillOpen
          ? current.activeSessionKey
          : sessions[0].key,
      };
    });
  }, []);

  if (state.kind === "boot") {
    // Minimal splash while the updater check resolves. Pure canvas — the
    // real updater UI (or Welcome) takes over within a few hundred ms on
    // a healthy network, ~4s worst case before the timeout fires.
    return <div className="app-boot" aria-hidden="true" />;
  }

  if (state.kind === "update_available") {
    return (
      <UpdaterLockScreen
        info={state.info}
        autoInstall={state.autoInstall}
        onSkip={state.autoInstall ? undefined : continueWithoutUpdate}
      />
    );
  }

  if (state.kind === "welcome") {
    return (
      <Welcome
        onPick={openWorkspace}
        error={bootError}
        deriveName={deriveName}
      />
    );
  }

  const activeSession = state.sessions.find(
    (session) => session.key === state.activeSessionKey,
  ) ?? state.sessions[0];

  if (!activeSession) {
    return (
      <Welcome
        onPick={openWorkspace}
        error={bootError}
        deriveName={deriveName}
      />
    );
  }

  return (
    <Workspace
      bootstrap={activeSession.bootstrap}
      sessions={state.sessions}
      activeSessionKey={activeSession.key}
      onSwitchWorkspace={pickAndOpenWorkspace}
      onOpenWorkspace={pickAndOpenWorkspace}
      onSelectSession={selectSession}
      onCreateConversationSession={createConversationSession}
      onRenameConversationSession={renameConversationSession}
      onDeleteConversationSession={deleteConversationSession}
      onArchiveConversationSession={archiveConversationSession}
      onRestoreConversationSession={restoreConversationSession}
      onCloseProjectSession={closeProjectSession}
      onWorkspaceConversationsReplace={updateWorkspaceConversations}
      onBootstrapReplace={replaceBootstrap}
    />
  );
}

function remoteOpenConversationsFromSessions(
  sessions: WorkspaceSession[],
  activeSessionKey: string,
): RemoteOpenConversationInput[] {
  return sessions.map((session) => {
    const workspace = session.bootstrap.workspace;
    const conversation = session.bootstrap.activeConversation;
    const summary = session.bootstrap.conversations.find(
      (item) => item.id === session.conversationId,
    );
    return {
      workspaceId: session.workspacePath,
      workspaceName: workspace.name,
      conversationId: session.conversationId,
      title: summary?.title || conversation.title || "New conversation",
      updatedAtMs: summary?.updatedAtMs ?? 0,
      active: session.key === activeSessionKey,
    };
  });
}

function sessionKeyFromBootstrap(bootstrap: WorkspaceBootstrap): string {
  return workspaceSessionKey(
    bootstrap.workspace.path,
    bootstrap.activeConversation.id,
  );
}

function sessionFromBootstrap(bootstrap: WorkspaceBootstrap): WorkspaceSession {
  const conversationId = bootstrap.activeConversation.id;
  const workspacePath = bootstrap.workspace.path;
  return {
    key: workspaceSessionKey(workspacePath, conversationId),
    workspacePath,
    conversationId,
    bootstrap,
  };
}

function upsertBootstrap(
  sessions: WorkspaceSession[],
  bootstrap: WorkspaceBootstrap,
): WorkspaceSession[] {
  const nextSession = sessionFromBootstrap(bootstrap);
  let replaced = false;
  const next = sessions.map((session) => {
    if (session.key === nextSession.key) {
      replaced = true;
      return nextSession;
    }
    if (session.workspacePath === nextSession.workspacePath) {
      return {
        ...session,
        bootstrap: {
          ...session.bootstrap,
          workspace: bootstrap.workspace,
          conversations: bootstrap.conversations,
          modeModelSettings: bootstrap.modeModelSettings,
        },
      };
    }
    return session;
  });
  return replaced ? next : [...next, nextSession];
}
