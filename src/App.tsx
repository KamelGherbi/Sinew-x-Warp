import { useCallback, useEffect, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { Welcome } from "./components/Welcome";
import { Workspace } from "./components/Workspace";
import { loadLastWorkspace, recordRecent, deriveName } from "./lib/recents";
import { api } from "./lib/ipc";
import { workspaceSessionKey } from "./lib/sessions";
import type { ConversationSummary, WorkspaceBootstrap, WorkspaceSession } from "./types";

type AppState =
  | { kind: "welcome" }
  | {
      kind: "workspace";
      sessions: WorkspaceSession[];
      activeSessionKey: string;
    };

const startsEmpty =
  new URLSearchParams(window.location.search).get("newWindow") === "1";

export default function App() {
  const [state, setState] = useState<AppState>({ kind: "welcome" });
  const [bootError, setBootError] = useState<string | null>(null);
  const stateRef = useRef(state);

  useEffect(() => {
    stateRef.current = state;
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

  const pickAndOpenWorkspace = useCallback(async () => {
    const selected = await open({ directory: true, multiple: false });
    if (typeof selected === "string") {
      await openWorkspace(selected);
    }
  }, [openWorkspace]);

  // Try to auto-open last workspace on boot. Silent fallback to the
  // welcome screen if the folder no longer exists or fails to open.
  useEffect(() => {
    if (startsEmpty) return;
    const last = loadLastWorkspace();
    if (!last) return;
    (async () => {
      try {
        const bootstrap = await api.openWorkspace(last);
        recordRecent(bootstrap.workspace.path, bootstrap.workspace.name);
        setState({
          kind: "workspace",
          sessions: [sessionFromBootstrap(bootstrap)],
          activeSessionKey: sessionKeyFromBootstrap(bootstrap),
        });
      } catch {
        // leave on welcome; user can pick again
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
        const bootstrap = await api.deleteConversation(workspacePath, conversationId);
        const deletedKey = workspaceSessionKey(workspacePath, conversationId);
        const replacementKey = sessionKeyFromBootstrap(bootstrap);
        setState((current) => {
          const remainingSessions =
            current.kind === "workspace"
              ? current.sessions.filter((session) => session.key !== deletedKey)
              : [];
          const sessions = upsertBootstrap(remainingSessions, bootstrap);
          const activeSessionKey =
            current.kind === "workspace" &&
            current.activeSessionKey !== deletedKey &&
            sessions.some((session) => session.key === current.activeSessionKey)
              ? current.activeSessionKey
              : replacementKey;
          return { kind: "workspace", sessions, activeSessionKey };
        });
      } catch (err) {
        console.error(err);
      }
    },
    [],
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
      onCloseProjectSession={closeProjectSession}
      onWorkspaceConversationsReplace={updateWorkspaceConversations}
      onBootstrapReplace={replaceBootstrap}
    />
  );
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
