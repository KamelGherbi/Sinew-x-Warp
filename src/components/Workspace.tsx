import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  startTransition,
} from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { Icon } from "@iconify/react";
import { api } from "../lib/ipc";
import { modelRefWithThinking, thinkingFromRef } from "../lib/models";
import { workspaceSessionKey } from "../lib/sessions";
import { recordRecent } from "../lib/recents";
import { Splitter } from "./Splitter";
import { FileTree, type FileTreeHandle } from "./FileTree";
import {
  ConversationList,
  type ConversationListProject,
} from "./ConversationList";
import { GitPanel, GitMark } from "./GitPanel";
import { EditorPane } from "./EditorPane";
import { SettingsPane } from "./SettingsPane";
import { SessionSwitcher } from "./SessionSwitcher";
import { TerminalPanel } from "./TerminalPanel";
import { SearchPane } from "./SearchPane";
import { ChatPane, type ExternalDropFeed } from "./chat/ChatPane";
import { SinewMark } from "./SinewMark";
import { UpdateBadge } from "./UpdateBadge";
import { WindowControls, isWindowsPlatform } from "./WindowControls";
import type {
  ActiveTurnSummary,
  ActiveTurnsChangedPayload,
  AgentEvent,
  AgentMode,
  ConversationEventPayload,
  ConversationSummary,
  EditorRevealTarget,
  EditorTab,
  FileChange,
  MessageVisibility,
  PlanArtifact,
  PlanControl,
  PlanImplementationOptions,
  SavedConversation,
  ServiceTier,
  ThinkingLevel,
  WorkspaceBootstrap,
  WorkspaceEntry,
  WorkspaceFileChangedPayload,
  WorkspaceSession,
} from "../types";

type Props = {
  bootstrap: WorkspaceBootstrap;
  onSwitchWorkspace: () => void;
  onBootstrapReplace: (b: WorkspaceBootstrap) => void;
  sessions?: WorkspaceSession[];
  activeSessionKey?: string | null;
  onSelectSession?: (
    workspacePath: string,
    conversationId: string,
  ) => void | Promise<void>;
  onOpenWorkspace?: () => void;
  onOpenProject?: () => void;
  onBackToWelcome?: () => void;
  onCreateConversationSession?: (workspacePath?: string) => void | Promise<void>;
  onRenameConversationSession?: (
    workspacePath: string,
    conversationId: string,
    title: string,
  ) => void | Promise<void>;
  onDeleteConversationSession?: (
    workspacePath: string,
    conversationId: string,
  ) => void | Promise<void>;
  onArchiveConversationSession?: (
    workspacePath: string,
    conversationId: string,
  ) => void | Promise<void>;
  onRestoreConversationSession?: (
    workspacePath: string,
    conversationId: string,
  ) => void | Promise<void>;
  onCloseProjectSession?: (workspacePath: string) => void;
  onWorkspaceConversationsReplace?: (
    workspacePath: string,
    conversations: ConversationSummary[],
  ) => void;
};

const INITIAL_LEFT = 280;
const INITIAL_RIGHT = 420;
const MIN_COL = 220;
const MAX_COL_RATIO = 0.6;
const INITIAL_SPLIT_TOP = 0.55;
const INITIAL_TERMINAL_HEIGHT = 240;
const MIN_TERMINAL_HEIGHT = 140;
const MAX_TERMINAL_RATIO = 0.92;
const APP_ZOOM_MIN = 1;
const APP_ZOOM_MAX = 2;
const APP_ZOOM_STEP = 0.1;
const TERMINAL_OPEN_EVENT = "terminal-open-requested";
const CLOSE_ACTIVE_TAB_EVENT = "editor-close-active-tab-requested";
const SEND_BUSY_RETRY_DELAYS_MS = [160, 320, 640, 1000, 1400];
const EMPTY_STREAMING_IDS: ReadonlySet<string> = new Set<string>();
const EMPTY_ATTENTION_IDS: ReadonlySet<string> = new Set<string>();
const LAYOUT_PANEL_VISIBILITY_KEY = "sinew.layout.panelVisibility";
const LAYOUT_VIEW_MODE_KEY = "sinew.layout.viewMode";
const PROJECT_TREE_COLLAPSED_KEY = "sinew.workspace.projectTreeCollapsed";
const COMPACTION_CONTINUATION_PROMPT =
  "Continue from the compacted context. Do not repeat completed work. Pick up exactly where you left off and proceed with the next useful step.";
const GOAL_COMPACTION_CONTINUATION_PROMPT =
  "Continue working toward the active goal from the compacted context. Do not repeat completed work. If the goal is now truly complete, audit it and call update_goal with status complete.";
const IS_WINDOWS = isWindowsPlatform();

type LayoutVisibility = {
  folder: boolean;
  editor: boolean;
  chat: boolean;
};

type LayoutPanel = keyof LayoutVisibility;

export function Workspace({
  bootstrap,
  onSwitchWorkspace,
  onBootstrapReplace,
  sessions,
  activeSessionKey,
  onSelectSession,
  onOpenWorkspace,
  onOpenProject,
  onBackToWelcome: _onBackToWelcome,
  onCreateConversationSession,
  onRenameConversationSession,
  onDeleteConversationSession,
  onArchiveConversationSession,
  onRestoreConversationSession,
  onCloseProjectSession,
  onWorkspaceConversationsReplace,
}: Props) {
  const workspacePath = bootstrap.workspace.path;

  const [conversations, setConversations] = useState<ConversationSummary[]>(
    bootstrap.conversations,
  );
  const [activeConv, setActiveConv] = useState<SavedConversation>(
    bootstrap.activeConversation,
  );
  const [, setGlobalModeModelSettings] = useState(
    bootstrap.modeModelSettings,
  );
  const [streamingConversationIdsByWorkspace, setStreamingConversationIdsByWorkspace] =
    useState<Map<string, Set<string>>>(() => new Map());
  const [attentionConversationIdsByWorkspace, setAttentionConversationIdsByWorkspace] =
    useState<Map<string, Set<string>>>(() => new Map());
  const [streamingModelsBySession, setStreamingModelsBySession] =
    useState<Map<string, SavedConversation["model"]>>(() => new Map());
  const lastAgentEventSequenceByConversationRef = useRef<Map<string, number>>(
    new Map(),
  );
  const replayActiveTurnEventsRef = useRef<
    (conversationId: string, afterSequence?: number) => Promise<void>
  >(async () => {});
  const activeConvIdRef = useRef(bootstrap.activeConversation.id);
  const workspacePathRef = useRef(workspacePath);
  const navigationSeqRef = useRef(0);
  const [layoutVisibility, setLayoutVisibility] = useState<LayoutVisibility>(() =>
    loadLayoutVisibility(),
  );
  const [sessionsOpen, setSessionsOpen] = useState(false);
  const [sessionsRefreshToken, setSessionsRefreshToken] = useState(0);
  // Bottom-sidebar tab — "conversations" preserves the existing default.
  // Both panels stay mounted (display:none on the inactive one) so users
  // can flip between them without losing in-progress form input.
  const [bottomTab, setBottomTab] = useState<"conversations" | "git">(
    "conversations",
  );
  const [projectTreeCollapsed, setProjectTreeCollapsed] = useState(() =>
    loadProjectTreeCollapsed(workspacePath),
  );

  useEffect(() => {
    setProjectTreeCollapsed(loadProjectTreeCollapsed(workspacePath));
  }, [workspacePath]);

  useEffect(() => {
    saveLayoutVisibility(layoutVisibility);
  }, [layoutVisibility]);

  useEffect(() => {
    activeConvIdRef.current = activeConv.id;
  }, [activeConv.id]);

  useEffect(() => {
    workspacePathRef.current = workspacePath;
  }, [workspacePath]);

  useEffect(() => {
    navigationSeqRef.current += 1;
    workspacePathRef.current = bootstrap.workspace.path;
    activeConvIdRef.current = bootstrap.activeConversation.id;
    setConversations(bootstrap.conversations);
    setActiveConv(bootstrap.activeConversation);
    setGlobalModeModelSettings(bootstrap.modeModelSettings);
  }, [bootstrap]);

  useEffect(() => {
    navigationSeqRef.current += 1;
  }, [workspacePath]);

  const streamingConversationIds = useMemo<ReadonlySet<string>>(
    () => streamingConversationIdsByWorkspace.get(workspacePath) ?? EMPTY_STREAMING_IDS,
    [streamingConversationIdsByWorkspace, workspacePath],
  );

  const attentionConversationIds = useMemo<ReadonlySet<string>>(
    () => attentionConversationIdsByWorkspace.get(workspacePath) ?? EMPTY_ATTENTION_IDS,
    [attentionConversationIdsByWorkspace, workspacePath],
  );

  const markConversationAttention = useCallback(
    (targetWorkspacePath: string, id: string, needsAttention: boolean) => {
      if (!targetWorkspacePath || !id) return;
      setAttentionConversationIdsByWorkspace((prev) => {
        const current = prev.get(targetWorkspacePath) ?? EMPTY_ATTENTION_IDS;
        if (current.has(id) === needsAttention) return prev;
        const nextIds = new Set(current);
        if (needsAttention) {
          nextIds.add(id);
        } else {
          nextIds.delete(id);
        }
        const next = new Map(prev);
        if (nextIds.size > 0) {
          next.set(targetWorkspacePath, nextIds);
        } else {
          next.delete(targetWorkspacePath);
        }
        return next;
      });
    },
    [],
  );

  const markConversationStreaming = useCallback(
    (workspacePathOrId: string, conversationIdOrActive: string | boolean, maybeActive?: boolean) => {
      const targetWorkspacePath =
        typeof maybeActive === "boolean" ? workspacePathOrId : workspacePathRef.current;
      const id =
        typeof maybeActive === "boolean"
          ? String(conversationIdOrActive)
          : workspacePathOrId;
      const active =
        typeof maybeActive === "boolean" ? maybeActive : Boolean(conversationIdOrActive);
      if (!targetWorkspacePath || !id) return;

      if (active) {
        markConversationAttention(targetWorkspacePath, id, false);
      }

      setStreamingConversationIdsByWorkspace((prev) => {
        const current = prev.get(targetWorkspacePath) ?? EMPTY_STREAMING_IDS;
        if (current.has(id) === active) return prev;
        const nextIds = new Set(current);
        if (active) {
          nextIds.add(id);
        } else {
          nextIds.delete(id);
        }
        const next = new Map(prev);
        if (nextIds.size > 0) {
          next.set(targetWorkspacePath, nextIds);
        } else {
          next.delete(targetWorkspacePath);
        }
        return next;
      });

      if (!active) {
        const sessionKey = workspaceSessionKey(targetWorkspacePath, id);
        setStreamingModelsBySession((prev) => {
          if (!prev.has(sessionKey)) return prev;
          const next = new Map(prev);
          next.delete(sessionKey);
          return next;
        });
      }
    },
    [markConversationAttention],
  );

  const markConversationStreamingModel = useCallback(
    (
      workspacePathOrId: string,
      conversationIdOrModel: string | SavedConversation["model"],
      modelOrThinking: SavedConversation["model"] | ThinkingLevel,
      maybeThinking?: ThinkingLevel,
    ) => {
      const targetWorkspacePath = maybeThinking ? workspacePathOrId : workspacePathRef.current;
      const id = maybeThinking ? String(conversationIdOrModel) : workspacePathOrId;
      const model = (maybeThinking ? modelOrThinking : conversationIdOrModel) as SavedConversation["model"];
      const thinking = (maybeThinking ?? modelOrThinking) as ThinkingLevel;
      if (!targetWorkspacePath || !id) return;
      const selected = modelRefWithThinking(model, thinking);
      const sessionKey = workspaceSessionKey(targetWorkspacePath, id);
      setStreamingModelsBySession((prev) => {
        const next = new Map(prev);
        next.set(sessionKey, selected);
        return next;
      });
    },
    [],
  );

  useEffect(() => {
    markConversationAttention(workspacePath, activeConv.id, false);
  }, [activeConv.id, markConversationAttention, workspacePath]);

  const refreshConversationList = useCallback(async () => {
    const workspaceAtRequest = workspacePath;
    try {
      const summaries = await api.listConversations(workspaceAtRequest);
      if (workspacePathRef.current !== workspaceAtRequest) return;
      setConversations(summaries);
    } catch (err) {
      console.error(err);
    }
  }, [workspacePath]);

  const selectConversation = useCallback(
    async (id: string) => {
      markConversationAttention(workspacePath, id, false);
      if (id === activeConv.id) return;
      const seq = ++navigationSeqRef.current;
      try {
        const loaded = await api.loadConversation(workspacePath, id);
        if (seq !== navigationSeqRef.current) return;
        if (loaded.id !== id || loaded.workspaceId !== workspacePath) return;
        activeConvIdRef.current = loaded.id;
        setActiveConv(loaded);
        onSelectSession?.(workspacePath, id);
        const sequenceKey = workspaceSessionKey(workspacePath, id);
        const last = lastAgentEventSequenceByConversationRef.current.get(sequenceKey) ?? 0;
        if (streamingConversationIds.has(id)) {
          void replayActiveTurnEventsRef.current(id, last).catch((err) =>
            console.error(err),
          );
        }
      } catch (err) {
        console.error(err);
      }
    },
    [workspacePath, activeConv.id, onSelectSession, streamingConversationIds, markConversationAttention],
  );

  const createConversation = useCallback(async (targetWorkspacePath?: string) => {
    const conversationWorkspacePath = targetWorkspacePath ?? workspacePath;
    if (onCreateConversationSession) {
      await onCreateConversationSession(conversationWorkspacePath);
      return;
    }
    const seq = ++navigationSeqRef.current;
    try {
      const next = await api.createConversation(conversationWorkspacePath);
      if (seq !== navigationSeqRef.current) return;
      if (next.workspace.path !== conversationWorkspacePath) return;
      activeConvIdRef.current = next.activeConversation.id;
      setConversations(next.conversations);
      setActiveConv(next.activeConversation);
      setGlobalModeModelSettings(next.modeModelSettings);
      onBootstrapReplace(next);
    } catch (err) {
      console.error(err);
    }
  }, [workspacePath, onBootstrapReplace, onCreateConversationSession]);

  const renameConversation = useCallback(
    async (id: string, title: string) => {
      try {
        const next = await api.renameConversation(workspacePath, id, title);
        setConversations(next);
        onWorkspaceConversationsReplace?.(workspacePath, next);
      } catch (err) {
        console.error(err);
      }
    },
    [workspacePath, onWorkspaceConversationsReplace],
  );

  const refreshConversationAfterMessageStart = useCallback(
    async (workspaceAtRequest: string, conversationId: string) => {
      const [loaded, summaries] = await Promise.all([
        api.loadConversation(workspaceAtRequest, conversationId),
        api.listConversations(workspaceAtRequest),
      ]);
      if (workspacePathRef.current !== workspaceAtRequest) return;
      onWorkspaceConversationsReplace?.(workspaceAtRequest, summaries);

      startTransition(() => {
        if (
          loaded.id === conversationId &&
          loaded.workspaceId === workspaceAtRequest &&
          activeConvIdRef.current === conversationId
        ) {
          setActiveConv((current) =>
            current.id === conversationId ? loaded : current,
          );
        }
        setConversations(summaries);
      });
    },
    [onWorkspaceConversationsReplace],
  );

  const applyOptimisticConversationTitle = useCallback(
    (conversationId: string, title: string) => {
      const updatedAtMs = Date.now();
      setActiveConv((current) =>
        current.id === conversationId ? { ...current, title } : current,
      );
      setConversations((current) =>
        sortConversationSummaries(
          current.map((conversation) =>
            conversation.id === conversationId
              ? {
                  ...conversation,
                  title,
                  updatedAtMs: Math.max(conversation.updatedAtMs, updatedAtMs),
                }
              : conversation,
          ),
        ),
      );
    },
    [],
  );

  const deleteConversation = useCallback(
    async (id: string) => {
      if (streamingConversationIds.has(id)) return;
      markConversationAttention(workspacePath, id, false);
      if (onDeleteConversationSession) {
        await onDeleteConversationSession(workspacePath, id);
        return;
      }
      const seq = ++navigationSeqRef.current;
      try {
        const summaries = await api.deleteConversation(workspacePath, id);
        if (seq !== navigationSeqRef.current) return;
        setConversations(summaries);
        onWorkspaceConversationsReplace?.(workspacePath, summaries);
        if (id !== activeConvIdRef.current) return;
        const nextSummary = summaries[0];
        if (!nextSummary) return;
        const nextConversation = await api.loadConversation(workspacePath, nextSummary.id);
        if (seq !== navigationSeqRef.current) return;
        activeConvIdRef.current = nextConversation.id;
        setActiveConv(nextConversation);
      } catch (err) {
        console.error(err);
        if (seq === navigationSeqRef.current) {
          navigationSeqRef.current += 1;
        }
      }
    },
    [
      workspacePath,
      streamingConversationIds,
      onDeleteConversationSession,
      onWorkspaceConversationsReplace,
      markConversationAttention,
    ],
  );

  const archiveConversation = useCallback(
    async (id: string) => {
      if (streamingConversationIds.has(id)) return;
      markConversationAttention(workspacePath, id, false);
      if (onArchiveConversationSession) {
        await onArchiveConversationSession(workspacePath, id);
        return;
      }
      const seq = ++navigationSeqRef.current;
      try {
        const summaries = await api.archiveConversation(workspacePath, id);
        if (seq !== navigationSeqRef.current) return;
        setConversations(summaries);
        onWorkspaceConversationsReplace?.(workspacePath, summaries);
        if (id !== activeConvIdRef.current) return;
        const nextSummary = summaries[0];
        if (!nextSummary) return;
        const nextConversation = await api.loadConversation(workspacePath, nextSummary.id);
        if (seq !== navigationSeqRef.current) return;
        activeConvIdRef.current = nextConversation.id;
        setActiveConv(nextConversation);
      } catch (err) {
        console.error(err);
        if (seq === navigationSeqRef.current) {
          navigationSeqRef.current += 1;
        }
      }
    },
    [
      workspacePath,
      streamingConversationIds,
      onArchiveConversationSession,
      onWorkspaceConversationsReplace,
      markConversationAttention,
    ],
  );

  // ---------------- Editor tabs ----------------
  const [tabs, setTabs] = useState<EditorTab[]>([]);
  const [activeTabIndex, setActiveTabIndex] = useState<number>(-1);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsActive, setSettingsActive] = useState(false);
  const [fileTreeRefreshToken, setFileTreeRefreshToken] = useState(0);
  const [fileSearchOpen, setFileSearchOpen] = useState(false);
  const [pendingRootCreate, setPendingRootCreate] = useState<
    "file" | "directory" | null
  >(null);
  const fileTreeRef = useRef<FileTreeHandle | null>(null);
  const [editorRevealTarget, setEditorRevealTarget] =
    useState<EditorRevealTarget | null>(null);
  const tabsRef = useRef(tabs);
  const appZoomRef = useRef(1);
  const toggleTerminalRef = useRef<() => void>(() => {});
  const fileTreeRefreshTimerRef = useRef<number | null>(null);
  const revealSeqRef = useRef(0);
  tabsRef.current = tabs;

  useEffect(() => {
    if (!pendingRootCreate || fileSearchOpen) return;
    const handle = fileTreeRef.current;
    if (!handle) return;
    handle.startCreateRoot(pendingRootCreate);
    setPendingRootCreate(null);
  }, [pendingRootCreate, fileSearchOpen]);

  const startRootCreate = useCallback(
    (kind: "file" | "directory") => {
      if (fileSearchOpen) {
        setFileSearchOpen(false);
        setPendingRootCreate(kind);
        return;
      }
      const handle = fileTreeRef.current;
      if (handle) {
        handle.startCreateRoot(kind);
      } else {
        setPendingRootCreate(kind);
      }
    },
    [fileSearchOpen],
  );

  useEffect(() => {
    tabsRef.current = tabs;
  }, [tabs]);

  const refreshFileTree = useCallback(() => {
    setFileTreeRefreshToken((value) => value + 1);
  }, []);

  const refreshFileTreeSoon = useCallback(() => {
    if (fileTreeRefreshTimerRef.current !== null) return;
    fileTreeRefreshTimerRef.current = window.setTimeout(() => {
      fileTreeRefreshTimerRef.current = null;
      refreshFileTree();
    }, 120);
  }, [refreshFileTree]);

  useEffect(() => {
    return () => {
      if (fileTreeRefreshTimerRef.current !== null) {
        window.clearTimeout(fileTreeRefreshTimerRef.current);
      }
    };
  }, []);

  const refreshOpenTabFromDisk = useCallback(async (relativePath: string) => {
    if (!relativePath) return;
    const workspaceAtRequest = workspacePathRef.current;
    if (
      !tabsRef.current.some((tab) => tab.relativePath === relativePath)
    ) {
      return;
    }

    try {
      const doc = await api.readFile(workspaceAtRequest, relativePath);
      if (workspacePathRef.current !== workspaceAtRequest) return;
      setTabs((prev) => {
        const idx = prev.findIndex((tab) => tab.relativePath === relativePath);
        if (idx < 0) return prev;
        const next = prev.slice();
        const tab = next[idx];
        next[idx] = tab.dirty
          ? { ...tab, doc }
          : {
              ...tab,
              doc,
              buffer: doc.content ?? "",
            };
        return next;
      });
    } catch (err) {
      console.error(err);
      if (workspacePathRef.current !== workspaceAtRequest) return;
      setTabs((prev) =>
        prev.filter(
          (tab) => tab.dirty || tab.relativePath !== relativePath,
        ),
      );
    }
  }, []);

  const refreshChangedFiles = useCallback(
    (changes: FileChange[]) => {
      if (changes.length === 0) return;
      refreshFileTreeSoon();
      const seen = new Set<string>();
      for (const change of changes) {
        if (!change.relativePath || seen.has(change.relativePath)) continue;
        seen.add(change.relativePath);
        void refreshOpenTabFromDisk(change.relativePath);
      }
    },
    [refreshFileTreeSoon, refreshOpenTabFromDisk],
  );

  useEffect(() => {
    let disposed = false;
    void api.watchWorkspace(workspacePath).catch((err) => {
      if (!disposed) {
        console.warn("workspace watcher unavailable", err);
      }
    });
    return () => {
      disposed = true;
      void api.unwatchWorkspace(workspacePath).catch((err) => {
        console.warn("workspace watcher cleanup failed", err);
      });
    };
  }, [workspacePath]);

  useEffect(() => {
    const onFocus = () => refreshFileTree();
    const onVisibility = () => {
      if (document.visibilityState === "visible") refreshFileTree();
    };
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [refreshFileTree]);

  const hasStreamingConversation = streamingConversationIds.size > 0;
  useEffect(() => {
    if (!hasStreamingConversation) return;
    const interval = window.setInterval(refreshFileTreeSoon, 1000);
    return () => window.clearInterval(interval);
  }, [hasStreamingConversation, refreshFileTreeSoon]);

  const openFile = useCallback(
    async (
      entry: WorkspaceEntry,
      reveal?: Omit<EditorRevealTarget, "id" | "relativePath">,
    ) => {
      if (entry.kind !== "file") return;
      setLayoutVisibility((current) => ({ ...current, editor: true }));
      const queueReveal = () => {
        if (!reveal) return;
        setEditorRevealTarget({
          ...reveal,
          id: ++revealSeqRef.current,
          relativePath: entry.relativePath,
        });
      };
      const existing = tabs.findIndex(
        (t) => t.relativePath === entry.relativePath,
      );
      if (existing >= 0) {
        setActiveTabIndex(existing);
        setSettingsActive(false);
        queueReveal();
        return;
      }
      try {
        const doc = await api.readFile(workspacePath, entry.relativePath);
        const newTab: EditorTab = {
          relativePath: entry.relativePath,
          doc,
          buffer: doc.content ?? "",
          dirty: false,
        };
        setTabs((prev) => {
          const existingIndex = prev.findIndex(
            (t) => t.relativePath === entry.relativePath,
          );
          if (existingIndex >= 0) {
            setActiveTabIndex(existingIndex);
            setSettingsActive(false);
            return prev;
          }
          const next = [...prev, newTab];
          setActiveTabIndex(next.length - 1);
          setSettingsActive(false);
          return next;
        });
        queueReveal();
      } catch (err) {
        console.error(err);
      }
    },
    [workspacePath, tabs],
  );

  const activateFileTab = useCallback((index: number) => {
    setActiveTabIndex(index);
    setSettingsActive(false);
  }, []);

  const openSettings = useCallback((section?: "providers") => {
    if (settingsActive && !section) {
      setSettingsOpen(false);
      setSettingsActive(false);
      return;
    }
    setSettingsOpen(true);
    setSettingsActive(true);
    setLayoutVisibility((current) => ({ ...current, editor: true }));
    if (section) {
      window.dispatchEvent(
        new CustomEvent("sinew:open-settings-section", {
          detail: { section },
        }),
      );
    }
  }, [settingsActive]);

  const closeSettings = useCallback(() => {
    setSettingsOpen(false);
    setSettingsActive(false);
  }, []);

  const openChatFile = useCallback(
    (rawPath: string) => {
      const relativePath = chatPathToRelative(rawPath, workspacePath);
      if (!relativePath) return;
      void openFile({
        name: basename(relativePath),
        relativePath,
        absolutePath: `${workspacePath}/${relativePath}`,
        kind: "file",
        hasChildren: false,
      });
    },
    [openFile, workspacePath],
  );

  // Open an arbitrary absolute path in a *read-only* Monaco tab. Used when
  // the user cmd+clicks a path in the terminal that points outside of the
  // active workspace.
  const openExternalFile = useCallback(
    async (
      absolutePath: string,
      reveal?: { lineNumber: number; columnStart: number; columnEnd: number },
    ) => {
      const queueReveal = () => {
        if (!reveal) return;
        setEditorRevealTarget({
          ...reveal,
          id: ++revealSeqRef.current,
          relativePath: absolutePath,
          query: "",
        });
      };
      const existing = tabsRef.current.findIndex(
        (t) => t.relativePath === absolutePath,
      );
      if (existing >= 0) {
        setActiveTabIndex(existing);
        queueReveal();
        return;
      }
      try {
        const doc = await api.readExternalFile(absolutePath);
        const newTab: EditorTab = {
          relativePath: absolutePath,
          doc,
          buffer: doc.content ?? "",
          dirty: false,
          external: true,
        };
        setTabs((prev) => {
          const existingIndex = prev.findIndex(
            (t) => t.relativePath === absolutePath,
          );
          if (existingIndex >= 0) {
            setActiveTabIndex(existingIndex);
            return prev;
          }
          const next = [...prev, newTab];
          setActiveTabIndex(next.length - 1);
          return next;
        });
        queueReveal();
      } catch (err) {
        console.error("Unable to open external file", absolutePath, err);
      }
    },
    [],
  );

  // Dispatch a raw path picked from the terminal (cmd+click). Resolves
  // the path on the backend and routes to the right editor / file-tree
  // / Finder helper depending on whether it is a file, a directory, in
  // or out of the active workspace.
  const openTerminalPath = useCallback(
    async (rawPath: string) => {
      const trimmed = rawPath.trim();
      if (!trimmed) return;
      try {
        const resolution = await api.resolveTerminalPath(workspacePath, trimmed);
        if (resolution.kind === "missing") return;

        const buildReveal = () => {
          if (resolution.line == null) return undefined;
          const lineNumber = Math.max(1, resolution.line);
          const columnStart = Math.max(1, resolution.column ?? 1);
          return {
            lineNumber,
            columnStart,
            columnEnd: columnStart + 1,
          };
        };

        if (resolution.kind === "directory") {
          if (!resolution.isOutsideWorkspace && resolution.relativePath != null) {
            void api.revealEntry(workspacePath, resolution.relativePath);
          } else {
            void api.revealAbsolutePath(resolution.absolutePath);
          }
          return;
        }

        // kind === "file"
        if (!resolution.isOutsideWorkspace && resolution.relativePath != null) {
          const reveal = buildReveal();
          await openFile(
            {
              name: basename(resolution.relativePath),
              relativePath: resolution.relativePath,
              absolutePath: resolution.absolutePath,
              kind: "file",
              hasChildren: false,
            },
            reveal ? { ...reveal, query: "" } : undefined,
          );
        } else {
          await openExternalFile(resolution.absolutePath, buildReveal());
        }
      } catch (err) {
        console.error("Unable to resolve terminal path", rawPath, err);
      }
    },
    [openExternalFile, openFile, workspacePath],
  );

  const closeTab = useCallback((index: number) => {
    const tabCount = tabsRef.current.length;
    if (index < 0 || index >= tabCount) return;

    setTabs((prev) => {
      if (index < 0 || index >= prev.length) return prev;
      const next = prev.slice();
      next.splice(index, 1);
      return next;
    });
    setActiveTabIndex((active) => {
      const nextLength = tabCount - 1;
      if (nextLength <= 0) return -1;
      if (active === index) return Math.min(index, nextLength - 1);
      if (active > index) return active - 1;
      return Math.min(active, nextLength - 1);
    });
  }, []);

  const closeActiveEditorTab = useCallback(() => {
    if (settingsActive) {
      closeSettings();
      return;
    }
    if (activeTabIndex >= 0) closeTab(activeTabIndex);
  }, [activeTabIndex, closeSettings, closeTab, settingsActive]);

  const handleTreeEntryRenamed = useCallback(
    (oldRelativePath: string, entry: WorkspaceEntry) => {
      setTabs((prev) =>
        prev.map((tab) => {
          if (tab.external) return tab;
          const nextPath = replaceTreePath(
            tab.relativePath,
            oldRelativePath,
            entry,
          );
          if (!nextPath) return tab;
          return retargetTab(tab, nextPath, workspacePath, entry);
        }),
      );
    },
    [workspacePath],
  );

  const handleTreeEntryDeleted = useCallback((entry: WorkspaceEntry) => {
    setTabs((prev) =>
      prev.filter(
        (tab) =>
          tab.external || tab.dirty || !entryContainsPath(entry, tab.relativePath),
      ),
    );
  }, []);

  const handleTreeEntriesMoved = useCallback(
    (moves: { from: WorkspaceEntry; to: WorkspaceEntry }[]) => {
      setTabs((prev) =>
        prev.map((tab) => {
          if (tab.external) return tab;
          for (const move of moves) {
            const nextPath = replaceTreePath(
              tab.relativePath,
              move.from.relativePath,
              move.to,
            );
            if (nextPath) {
              return retargetTab(tab, nextPath, workspacePath, move.to);
            }
          }
          return tab;
        }),
      );
    },
    [workspacePath],
  );

  useEffect(() => {
    if (activeTabIndex >= tabs.length) {
      setActiveTabIndex(tabs.length - 1);
    }
  }, [tabs.length, activeTabIndex]);

  const updateBuffer = useCallback((index: number, value: string) => {
    setTabs((prev) => {
      const next = prev.slice();
      const tab = next[index];
      if (!tab) return prev;
      // External (read-only) tabs are never dirty and should never have
      // their buffer mutated by Monaco onChange events.
      if (tab.external) return prev;
      next[index] = {
        ...tab,
        buffer: value,
        dirty: value !== (tab.doc.content ?? ""),
      };
      return next;
    });
  }, []);

  const saveTab = useCallback(
    async (index: number) => {
      const tab = tabs[index];
      if (!tab || !tab.dirty || tab.external) return;
      try {
        const updated = await api.writeFile(
          workspacePath,
          tab.relativePath,
          tab.buffer,
        );
        setTabs((prev) => {
          const next = prev.slice();
          if (!next[index]) return prev;
          next[index] = {
            ...next[index],
            doc: updated,
            buffer: updated.content ?? next[index].buffer,
            dirty: false,
          };
          return next;
        });
      } catch (err) {
        console.error(err);
      }
    },
    [workspacePath, tabs],
  );

  const applyAppZoom = useCallback((nextZoom: number) => {
    const zoom = Math.max(
      APP_ZOOM_MIN,
      Math.min(APP_ZOOM_MAX, Math.round(nextZoom * 100) / 100),
    );
    appZoomRef.current = zoom;
    void getCurrentWebview()
      .setZoom(zoom)
      .catch((err) => console.warn("Unable to set app zoom", err));
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const hasPrimaryModifier = event.metaKey || event.ctrlKey;

      if (
        IS_WINDOWS &&
        hasPrimaryModifier &&
        !event.altKey &&
        !event.shiftKey &&
        event.key.toLowerCase() === "w"
      ) {
        event.preventDefault();
        closeActiveEditorTab();
        return;
      }
      if (
        hasPrimaryModifier &&
        !event.altKey &&
        !event.shiftKey &&
        event.key.toLowerCase() === "j"
      ) {
        event.preventDefault();
        toggleTerminalRef.current();
        return;
      }
      if (hasPrimaryModifier && !event.altKey) {
        if (
          event.key === "+" ||
          event.key === "=" ||
          event.code === "NumpadAdd"
        ) {
          event.preventDefault();
          applyAppZoom(appZoomRef.current + APP_ZOOM_STEP);
          return;
        }
        if (
          event.key === "-" ||
          event.code === "Minus" ||
          event.code === "NumpadSubtract"
        ) {
          event.preventDefault();
          applyAppZoom(appZoomRef.current - APP_ZOOM_STEP);
          return;
        }
        if (
          event.key === "0" ||
          event.code === "Digit0" ||
          event.code === "Numpad0"
        ) {
          event.preventDefault();
          applyAppZoom(1);
          return;
        }
      }
      if (hasPrimaryModifier && event.key.toLowerCase() === "s") {
        event.preventDefault();
        if (settingsActive) return;
        if (activeTabIndex >= 0) void saveTab(activeTabIndex);
        return;
      }
      if (
        hasPrimaryModifier &&
        event.shiftKey &&
        event.key.toLowerCase() === "f"
      ) {
        event.preventDefault();
        setFileSearchOpen(true);
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [activeTabIndex, applyAppZoom, closeActiveEditorTab, saveTab, settingsActive]);

  useEffect(() => {
    let disposed = false;
    let unlisten: UnlistenFn | null = null;

    void listen(CLOSE_ACTIVE_TAB_EVENT, () => {
      closeActiveEditorTab();
    }).then((nextUnlisten) => {
      if (disposed) {
        nextUnlisten();
      } else {
        unlisten = nextUnlisten;
      }
    });

    return () => {
      disposed = true;
      if (unlisten) unlisten();
    };
  }, [closeActiveEditorTab]);

  // ---------------- Event subscriptions ----------------

  const agentSubsRef = useRef<
    Set<
      (
        conversationId: string,
        event: AgentEvent,
        workspacePath: string,
        sequence?: number,
      ) => void
    >
  >(new Set());

  useEffect(() => {
    let cancelled = false;
    let unlisten: UnlistenFn | null = null;
    (async () => {
      const u = await listen<ConversationEventPayload>(
        "agent-event",
        (event) => {
          const payload = event.payload;
          const payloadWorkspacePath = payload.workspaceId ?? workspacePathRef.current;
          const sequenceKey = workspaceSessionKey(
            payloadWorkspacePath,
            payload.conversationId,
          );
          if (
            payload.event.type === "turn_started" &&
            payloadWorkspacePath === workspacePathRef.current
          ) {
            lastAgentEventSequenceByConversationRef.current.delete(sequenceKey);
          }
          if (
            typeof payload.sequence === "number" &&
            payloadWorkspacePath === workspacePathRef.current
          ) {
            const last =
              lastAgentEventSequenceByConversationRef.current.get(
                sequenceKey,
              ) ?? 0;
            if (payload.sequence <= last) return;
            lastAgentEventSequenceByConversationRef.current.set(
              sequenceKey,
              payload.sequence,
            );
          }
          for (const handler of agentSubsRef.current) {
            handler(
              payload.conversationId,
              payload.event,
              payloadWorkspacePath,
              payload.sequence,
            );
          }
        },
      );
      if (cancelled) {
        u();
      } else {
        unlisten = u;
      }
    })();
    return () => {
      cancelled = true;
      if (unlisten) unlisten();
    };
  }, []);

  const subscribeEvents = useCallback(
    (
      handler: (
        conversationId: string,
        event: AgentEvent,
        sequence?: number,
      ) => void,
    ) => {
      const scopedHandler = (
        conversationId: string,
        event: AgentEvent,
        eventWorkspacePath: string,
        sequence?: number,
      ) => {
        if (eventWorkspacePath !== workspacePathRef.current) return;
        handler(conversationId, event, sequence);
      };
      agentSubsRef.current.add(scopedHandler);
      return () => {
        agentSubsRef.current.delete(scopedHandler);
      };
    },
    [],
  );

  const replayActiveTurnEvents = useCallback(
    async (conversationId: string, afterSequence = 0) => {
      const workspaceAtRequest = workspacePathRef.current;
      let replay = await api.replayActiveTurnEvents(
        workspaceAtRequest,
        conversationId,
        afterSequence,
      );
      if (workspacePathRef.current !== workspaceAtRequest) return;
      if (!replay.active) {
        markConversationStreaming(workspaceAtRequest, conversationId, false);
        return;
      }
      if (replay.latestSequence < afterSequence) {
        const sequenceKey = workspaceSessionKey(workspaceAtRequest, conversationId);
        lastAgentEventSequenceByConversationRef.current.delete(sequenceKey);
        replay = await api.replayActiveTurnEvents(
          workspaceAtRequest,
          conversationId,
          0,
        );
        if (workspacePathRef.current !== workspaceAtRequest) return;
        if (!replay.active) {
          markConversationStreaming(workspaceAtRequest, conversationId, false);
          return;
        }
      }
      markConversationStreaming(workspaceAtRequest, conversationId, true);
      const sortedEvents = [...replay.events].sort(
        (a, b) => a.sequence - b.sequence,
      );
      for (const entry of sortedEvents) {
        const sequenceKey = workspaceSessionKey(workspaceAtRequest, conversationId);
        const last =
          lastAgentEventSequenceByConversationRef.current.get(sequenceKey) ?? 0;
        if (entry.sequence <= last && last !== 0) continue;
        lastAgentEventSequenceByConversationRef.current.set(
          sequenceKey,
          entry.sequence,
        );
        for (const handler of agentSubsRef.current) {
          handler(conversationId, entry.event, workspaceAtRequest, entry.sequence);
        }
      }
    },
    [markConversationStreaming],
  );

  useEffect(() => {
    replayActiveTurnEventsRef.current = replayActiveTurnEvents;
  }, [replayActiveTurnEvents]);

  const syncActiveTurns = useCallback(
    (activeTurns: ActiveTurnSummary[]) => {
      const activeIdsByWorkspace = new Map<string, Set<string>>();
      const activeSessionKeys = new Set<string>();
      for (const turn of activeTurns) {
        const ids = activeIdsByWorkspace.get(turn.workspaceId) ?? new Set<string>();
        ids.add(turn.conversationId);
        activeIdsByWorkspace.set(turn.workspaceId, ids);
        activeSessionKeys.add(workspaceSessionKey(turn.workspaceId, turn.conversationId));
      }

      setStreamingConversationIdsByWorkspace((prev) => {
        let changed = prev.size !== activeIdsByWorkspace.size;
        if (!changed) {
          for (const [workspaceId, activeIds] of activeIdsByWorkspace) {
            const currentIds = prev.get(workspaceId);
            if (!currentIds || currentIds.size !== activeIds.size) {
              changed = true;
              break;
            }
            for (const id of activeIds) {
              if (!currentIds.has(id)) {
                changed = true;
                break;
              }
            }
            if (changed) break;
          }
        }
        return changed ? activeIdsByWorkspace : prev;
      });

      setStreamingModelsBySession((prev) => {
        let changed = false;
        const next = new Map(prev);
        for (const sessionKey of Array.from(next.keys())) {
          if (!activeSessionKeys.has(sessionKey)) {
            next.delete(sessionKey);
            changed = true;
          }
        }
        return changed ? next : prev;
      });

      setAttentionConversationIdsByWorkspace((prev) => {
        if (prev.size === 0) return prev;
        let changed = false;
        const next = new Map(prev);
        for (const [activeWorkspacePath, activeIds] of activeIdsByWorkspace) {
          const currentIds = next.get(activeWorkspacePath);
          if (!currentIds) continue;
          const nextIds = new Set(currentIds);
          for (const id of activeIds) {
            if (nextIds.delete(id)) changed = true;
          }
          if (nextIds.size > 0) {
            next.set(activeWorkspacePath, nextIds);
          } else {
            next.delete(activeWorkspacePath);
          }
        }
        return changed ? next : prev;
      });

      for (const turn of activeTurns) {
        if (turn.workspaceId !== workspacePathRef.current) continue;
        const sequenceKey = workspaceSessionKey(turn.workspaceId, turn.conversationId);
        let last =
          lastAgentEventSequenceByConversationRef.current.get(sequenceKey) ?? 0;
        if (turn.latestSequence < last) {
          lastAgentEventSequenceByConversationRef.current.delete(sequenceKey);
          last = 0;
        }
        if (turn.latestSequence > last) {
          void replayActiveTurnEvents(turn.conversationId, last).catch((err) => {
            console.error(err);
          });
        }
      }
    },
    [replayActiveTurnEvents],
  );

  useEffect(() => {
    const handler = async (
      conversationId: string,
      event: AgentEvent,
      eventWorkspacePath: string,
    ) => {
      const isActiveWorkspace = eventWorkspacePath === workspacePathRef.current;
      const fileChanges = fileChangesFromAgentEvent(event);
      if (isActiveWorkspace && fileChanges.length > 0) {
        refreshChangedFiles(fileChanges);
      }

      if (event.type === "conversation_title_updated") {
        if (isActiveWorkspace) {
          const { title, updated_at_ms: updatedAtMs } = event;
          startTransition(() => {
            if (workspacePathRef.current !== eventWorkspacePath) return;
            setActiveConv((current) =>
              current.id === conversationId ? { ...current, title } : current,
            );
            setConversations((current) =>
              sortConversationSummaries(
                current.map((conversation) =>
                  conversation.id === conversationId
                    ? {
                        ...conversation,
                        title,
                        updatedAtMs: Math.max(
                          conversation.updatedAtMs,
                          updatedAtMs,
                        ),
                      }
                    : conversation,
                ),
              ),
            );
          });
          void api
            .listConversations(eventWorkspacePath)
            .then((summaries) => {
              onWorkspaceConversationsReplace?.(eventWorkspacePath, summaries);
              if (workspacePathRef.current === eventWorkspacePath) {
                setConversations(summaries);
              }
            })
            .catch((err) => console.error(err));
        }
        return;
      }

      if (event.type === "turn_started") {
        markConversationStreaming(eventWorkspacePath, conversationId, true);
        return;
      }
      if (event.type !== "turn_finished") {
        return;
      }
      markConversationStreaming(eventWorkspacePath, conversationId, false);
      const workspaceAtRequest = eventWorkspacePath;
      const isActiveConversation =
        isActiveWorkspace && conversationId === activeConvIdRef.current;
      markConversationAttention(
        workspaceAtRequest,
        conversationId,
        !isActiveConversation,
      );
      const shouldLoadActive = isActiveConversation;
      try {
        const summariesPromise = api.listConversations(workspaceAtRequest);
        const loadedPromise =
          shouldLoadActive
            ? api.loadConversation(workspaceAtRequest, conversationId)
            : Promise.resolve(null);
        const [loaded, summaries] = await Promise.all([
          loadedPromise,
          summariesPromise,
        ]);
        onWorkspaceConversationsReplace?.(workspaceAtRequest, summaries);
        if (!isActiveWorkspace) return;
        startTransition(() => {
          if (workspacePathRef.current !== workspaceAtRequest) return;
          if (
            loaded &&
            loaded.id === conversationId &&
            loaded.workspaceId === workspaceAtRequest &&
            activeConvIdRef.current === conversationId
          ) {
            setActiveConv(loaded);
          }
          setConversations(summaries);
        });
      } catch (err) {
        console.error(err);
      }
    };
    agentSubsRef.current.add(handler);
    return () => {
      agentSubsRef.current.delete(handler);
    };
  }, [markConversationStreaming, markConversationAttention, onWorkspaceConversationsReplace, refreshChangedFiles]);

  useEffect(() => {
    if (!streamingConversationIds.has(activeConv.id)) return;
    const sequenceKey = workspaceSessionKey(workspacePath, activeConv.id);
    const last = lastAgentEventSequenceByConversationRef.current.get(sequenceKey) ?? 0;
    void replayActiveTurnEvents(activeConv.id, last).catch((err) => {
      console.error(err);
    });
    // Only replay on conversation/workspace switches. A newly submitted prompt is
    // marked streaming optimistically before the backend can replay it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeConv.id, replayActiveTurnEvents, workspacePath]);

  useEffect(() => {
    let cancelled = false;
    void api
      .listActiveTurns()
      .then((activeTurns) => {
        if (!cancelled) syncActiveTurns(activeTurns);
      })
      .catch((err) => {
        if (!cancelled) console.error(err);
      });
    return () => {
      cancelled = true;
    };
  }, [syncActiveTurns, workspacePath]);

  useEffect(() => {
    let cancelled = false;
    let unlisten: UnlistenFn | null = null;
    (async () => {
      const u = await listen<ActiveTurnsChangedPayload>(
        "active-turns-changed",
        (event) => {
          syncActiveTurns(event.payload.activeTurns);
        },
      );
      if (cancelled) {
        u();
      } else {
        unlisten = u;
      }
    })();
    return () => {
      cancelled = true;
      if (unlisten) unlisten();
    };
  }, [syncActiveTurns]);

  useEffect(() => {
    let cancelled = false;
    let unlisten: UnlistenFn | null = null;
    (async () => {
      const u = await listen<WorkspaceFileChangedPayload>(
        "workspace-file-changed",
        async (event) => {
          const payload = event.payload;
          if (payload.workspacePath !== workspacePath) return;
          refreshFileTreeSoon();
          if (!payload.relativePath) return;
          void refreshOpenTabFromDisk(payload.relativePath);
        },
      );
      if (cancelled) {
        u();
      } else {
        unlisten = u;
      }
    })();
    return () => {
      cancelled = true;
      if (unlisten) unlisten();
    };
  }, [refreshFileTreeSoon, refreshOpenTabFromDisk, workspacePath]);

  const externalDropFeed = useMemo<ExternalDropFeed>(
    () => ({
      subscribe(handler) {
        dropSubsRef.current.add(handler);
        return () => {
          dropSubsRef.current.delete(handler);
        };
      },
      subscribeDrag(handler) {
        dragSubsRef.current.add(handler);
        return () => {
          dragSubsRef.current.delete(handler);
        };
      },
    }),
    [],
  );
  const dropSubsRef = useRef<
    Set<
      (attachments: { path: string; name: string; origin: "finder" }[]) => void
    >
  >(new Set());
  const dragSubsRef = useRef<Set<(active: boolean) => void>>(new Set());
  const chatDropZoneRef = useRef<HTMLDivElement | null>(null);
  const fileTreeDropZoneRef = useRef<HTMLDivElement | null>(null);
  const [fileTreeDropState, setFileTreeDropState] = useState<{
    active: boolean;
    targetRelative: string | null;
  }>({ active: false, targetRelative: null });
  const [importError, setImportError] = useState<string | null>(null);

  const findFolderTargetAt = useCallback(
    (x: number, y: number): string | null => {
      const el = document.elementFromPoint(x, y);
      if (!el) return null;
      const row = (el as Element).closest?.(
        ".tree-row[data-kind='directory']",
      ) as HTMLElement | null;
      if (!row) return null;
      return row.dataset.dropPath ?? null;
    },
    [],
  );

  useEffect(() => {
    let cancelled = false;
    let unlisten: UnlistenFn | null = null;
    (async () => {
      try {
        const u = await getCurrentWebview().onDragDropEvent(async (event) => {
          const type = event.payload.type;
          const position =
            "position" in event.payload ? event.payload.position : null;
          const chatRect = chatDropZoneRef.current?.getBoundingClientRect();
          const sidebarRect =
            fileTreeDropZoneRef.current?.getBoundingClientRect();
          const overChat = (() => {
            if (!chatRect || !position) return false;
            return (
              position.x >= chatRect.left &&
              position.x <= chatRect.right &&
              position.y >= chatRect.top &&
              position.y <= chatRect.bottom
            );
          })();
          const overSidebar = (() => {
            if (overChat) return false;
            if (!sidebarRect || !position) return false;
            return (
              position.x >= sidebarRect.left &&
              position.x <= sidebarRect.right &&
              position.y >= sidebarRect.top &&
              position.y <= sidebarRect.bottom
            );
          })();

          if (type === "enter" || type === "over") {
            for (const handler of dragSubsRef.current) handler(overChat);
            if (overSidebar && position) {
              const target = findFolderTargetAt(position.x, position.y);
              setFileTreeDropState({ active: true, targetRelative: target });
            } else {
              setFileTreeDropState((prev) =>
                prev.active ? { active: false, targetRelative: null } : prev,
              );
            }
            return;
          }
          if (type === "leave") {
            for (const handler of dragSubsRef.current) handler(false);
            setFileTreeDropState({ active: false, targetRelative: null });
            return;
          }
          if (type === "drop") {
            for (const handler of dragSubsRef.current) handler(false);
            setFileTreeDropState({ active: false, targetRelative: null });
            const paths = event.payload.paths ?? [];
            if (!paths.length) return;
            if (overChat) {
              const attachments = paths.map((path) => ({
                path,
                name: basename(path),
                origin: "finder" as const,
              }));
              for (const handler of dropSubsRef.current) handler(attachments);
              return;
            }
            if (overSidebar && position) {
              const target = findFolderTargetAt(position.x, position.y);
              try {
                setImportError(null);
                await api.importPaths(workspacePath, paths, target ?? undefined);
                refreshFileTree();
              } catch (err) {
                console.error(err);
                setImportError(String(err));
              }
            }
          }
        });
        if (cancelled) {
          u();
        } else {
          unlisten = u;
        }
      } catch (err) {
        console.warn("webview drag-drop unavailable", err);
      }
    })();
    return () => {
      cancelled = true;
      if (unlisten) unlisten();
    };
  }, [workspacePath, findFolderTargetAt, refreshFileTree]);

  const sendMessage = useCallback(
    async (
      text: string,
      attachments: { path: string; name?: string }[],
      model: SavedConversation["model"],
      thinking: ThinkingLevel,
      mode: AgentMode,
      serviceTier?: ServiceTier | null,
      rewriteFromHistoryIndex?: number,
      planControl?: PlanControl,
      messageVisibility?: MessageVisibility,
      planImplementationOptions?: PlanImplementationOptions,
      revertWorkspaceChanges?: boolean,
    ) => {
      const conversationId = activeConv.id;
      const workspaceAtRequest = workspacePath;
      const optimisticTitle = titleFromOutgoingUserText(text);
      const shouldUpdateTitleFromUserMessage =
        messageVisibility !== "systemReminder" &&
        Boolean(optimisticTitle) &&
        (rewriteFromHistoryIndex === 0 ||
          (rewriteFromHistoryIndex === undefined && activeConv.history.length === 0));

      if (shouldUpdateTitleFromUserMessage && optimisticTitle) {
        applyOptimisticConversationTitle(conversationId, optimisticTitle);
      }

      markConversationStreamingModel(conversationId, model, thinking);
      markConversationStreaming(conversationId, true);
      try {
        await sendMessageWithBusyRetry(
          workspaceAtRequest,
          conversationId,
          text,
          attachments,
          model,
          thinking,
          mode,
          serviceTier,
          rewriteFromHistoryIndex,
          planControl,
          messageVisibility,
          planImplementationOptions,
          revertWorkspaceChanges,
        );
      } catch (err) {
        markConversationStreaming(conversationId, false);
        void refreshConversationAfterMessageStart(
          workspaceAtRequest,
          conversationId,
        ).catch((refreshErr) => console.error(refreshErr));
        throw err;
      }

      void refreshConversationAfterMessageStart(
        workspaceAtRequest,
        conversationId,
      ).catch((err) => console.error(err));
    },
    [
      workspacePath,
      activeConv.id,
      activeConv.history.length,
      applyOptimisticConversationTitle,
      markConversationStreaming,
      markConversationStreamingModel,
      refreshConversationAfterMessageStart,
    ],
  );

  const compactConversation = useCallback(
    async (
      model: SavedConversation["model"],
      thinking: ThinkingLevel,
      serviceTier?: ServiceTier | null,
      options?: { continueAfter?: boolean; instruction?: string },
    ) => {
      const conversationId = activeConv.id;
      const continueAfter = options?.continueAfter ?? true;
      const continuationMode = conversationContinuationMode(activeConv);
      const continuationPrompt =
        continuationMode === "goal"
          ? GOAL_COMPACTION_CONTINUATION_PROMPT
          : COMPACTION_CONTINUATION_PROMPT;
      markConversationStreamingModel(conversationId, model, thinking);
      markConversationStreaming(conversationId, true);
      try {
        await api.compactConversation(
          workspacePath,
          conversationId,
          model,
          thinking,
          serviceTier,
          options?.instruction,
        );

        markConversationStreaming(conversationId, false);

        const [loaded, summaries] = await Promise.all([
          api.loadConversation(workspacePath, conversationId),
          api.listConversations(workspacePath),
        ]);
        if (workspacePathRef.current !== workspacePath) return;

        setConversations(summaries);
        if (activeConvIdRef.current === conversationId) {
          setActiveConv(loaded);
        }

        if (!continueAfter) return;

        await sleep(0);

        markConversationStreamingModel(conversationId, model, thinking);
        markConversationStreaming(conversationId, true);
        await sendMessageWithBusyRetry(
          workspacePath,
          conversationId,
          continuationPrompt,
          [],
          model,
          thinking,
          continuationMode,
          serviceTier,
          undefined,
          undefined,
          "systemReminder",
        );

        const reloaded = await api.loadConversation(workspacePath, conversationId);
        if (
          workspacePathRef.current === workspacePath &&
          activeConvIdRef.current === conversationId
        ) {
          setActiveConv((current) =>
            current.id === conversationId ? reloaded : current,
          );
        }
      } catch (err) {
        markConversationStreaming(conversationId, false);
        throw err;
      }
    },
    [activeConv, markConversationStreaming, markConversationStreamingModel, workspacePath],
  );

  const changeConversationMode = useCallback(
    async (mode: AgentMode) => {
      const conversationId = activeConv.id;
      const updated = await api.setConversationMode(
        workspacePath,
        conversationId,
        mode,
      );
      const summaries = await api.listConversations(workspacePath);
      startTransition(() => {
        setActiveConv((current) =>
          current.id === conversationId ? updated : current,
        );
        setConversations(summaries);
      });
    },
    [activeConv.id, workspacePath],
  );

  const changeConversationModelPreference = useCallback(
    async (
      mode: AgentMode,
      model: SavedConversation["model"],
      thinking: ThinkingLevel,
    ) => {
      const conversationId = activeConv.id;
      const updated = await api.setConversationModelPreference(
        workspacePath,
        conversationId,
        mode,
        model,
        thinking,
      );
      const selected = modelRefWithThinking(model, thinking);
      setGlobalModeModelSettings((current) => ({
        ...current,
        [mode]: selected,
      }));
      startTransition(() => {
        setActiveConv((current) =>
          current.id === conversationId
            ? {
                ...current,
                model: selected,
                modeModelSettings: updated,
              }
            : current,
        );
      });
    },
    [activeConv.id, workspacePath],
  );

  const implementPlanFresh = useCallback(
    async (
      plan: PlanArtifact,
      prompt = "Implement completely this plan. Use the attached markdown plan as the source of truth.",
      planImplementationOptions?: PlanImplementationOptions,
      mode: AgentMode = "act",
    ) => {
      const implementationWorkspacePath =
        planImplementationOptions?.implementationWorkspacePath?.trim() || workspacePath;
      const next = await api.createConversation(implementationWorkspacePath);
      const conversationId = next.activeConversation.id;
      // The new conversation is seeded with the workspace's global default,
      // which represents the most recent model the user picked anywhere. Per
      // the plan, every brand-new conversation must use that seed (not the
      // preference of whatever conversation the user was sitting in when
      // they triggered the action).
      const seedModel = next.activeConversation.modeModelSettings[mode];
      const seedThinking = thinkingFromRef(seedModel);
      const title = titleFromPlanImplementation(plan);
      const titledActiveConversation = {
        ...next.activeConversation,
        title,
      };
      const titledConversations = await api.renameConversation(
        implementationWorkspacePath,
        conversationId,
        title,
      );
      activeConvIdRef.current = conversationId;
      setConversations(titledConversations);
      setActiveConv(titledActiveConversation);
      setGlobalModeModelSettings(next.modeModelSettings);
      onBootstrapReplace({
        ...next,
        conversations: titledConversations,
        activeConversation: titledActiveConversation,
      });
      markConversationStreamingModel(implementationWorkspacePath, conversationId, seedModel, seedThinking);
      markConversationStreaming(implementationWorkspacePath, conversationId, true);
      try {
        await sendMessageWithBusyRetry(
          implementationWorkspacePath,
          conversationId,
          prompt,
          [
            {
              path: plan.absolutePath ?? plan.path,
              name: basename(plan.path),
            },
          ],
          seedModel,
          seedThinking,
          mode,
          null,
          undefined,
          "implementPlan",
          "systemReminder",
          {
            ...planImplementationOptions,
            implementationPath: ".",
          },
        );
        const loaded = await api.loadConversation(implementationWorkspacePath, conversationId);
        startTransition(() => {
          setActiveConv((current) =>
            current.id === conversationId ? loaded : current,
          );
        });
      } catch (err) {
        markConversationStreaming(conversationId, false);
        throw err;
      }
    },
    [workspacePath, markConversationStreaming, markConversationStreamingModel, onBootstrapReplace],
  );

  const stopTurn = useCallback(async () => {
    try {
      await api.cancelTurn(workspacePath, activeConv.id);
    } catch (err) {
      console.error(err);
    }
  }, [workspacePath, activeConv.id]);

  // Switch this window to another workspace path. Used by the Git
  // panel when the user clicks a worktree row or creates a new one.
  // We refuse the switch while any conversation in the current window
  // is still streaming so the in-flight turn isn't orphaned mid-tool.
  // Throwing here lets the caller surface a contextual error notice
  // instead of swallowing the failure silently.
  const switchWorkspace = useCallback(
    async (targetPath: string): Promise<void> => {
      if (streamingConversationIds.size > 0) {
        throw new Error(
          "A conversation is still streaming. Stop active turns before switching workspace.",
        );
      }
      const nextBootstrap = await api.openWorkspace(targetPath);
      recordRecent(
        nextBootstrap.workspace.path,
        nextBootstrap.workspace.name,
      );
      // Drop editor / settings / search state that belongs to the
      // outgoing workspace so the incoming one boots cleanly. The
      // bootstrap-replace effect refreshes conversations, file tree,
      // etc. on its own.
      setTabs([]);
      setActiveTabIndex(-1);
      setSettingsOpen(false);
      setSettingsActive(false);
      setFileSearchOpen(false);
      setPendingRootCreate(null);
      setEditorRevealTarget(null);
      setImportError(null);
      onBootstrapReplace(nextBootstrap);
      refreshFileTree();
    },
    [
      onBootstrapReplace,
      refreshFileTree,
      streamingConversationIds,
    ],
  );

  // ---------------- Layout state ----------------
  const [leftWidth, setLeftWidth] = useState(INITIAL_LEFT);
  const [rightWidth, setRightWidth] = useState(INITIAL_RIGHT);
  const [topSplit, setTopSplit] = useState(INITIAL_SPLIT_TOP);
  const [terminalAvailable, setTerminalAvailable] = useState(false);
  const [terminalOpen, setTerminalOpen] = useState(false);
  const [terminalFullHeight, setTerminalFullHeight] = useState(false);
  const [terminalHeight, setTerminalHeight] = useState(INITIAL_TERMINAL_HEIGHT);

  const clampColumn = useCallback((v: number) => {
    if (typeof window === "undefined") return v;
    const max = window.innerWidth * MAX_COL_RATIO;
    return Math.max(MIN_COL, Math.min(max, v));
  }, []);

  const clampTerminal = useCallback((v: number) => {
    if (typeof window === "undefined") return v;
    const max = Math.max(MIN_TERMINAL_HEIGHT, window.innerHeight * MAX_TERMINAL_RATIO);
    return Math.max(MIN_TERMINAL_HEIGHT, Math.min(max, v));
  }, []);

  const showTerminal = useCallback(() => {
    setTerminalAvailable(true);
    setTerminalOpen(true);
    setTerminalHeight((value) => clampTerminal(value));
  }, [clampTerminal]);

  const hideTerminal = useCallback(() => {
    setTerminalOpen(false);
    setTerminalFullHeight(false);
  }, []);

  const closeTerminalPanel = useCallback(() => {
    setTerminalOpen(false);
    setTerminalFullHeight(false);
    setTerminalAvailable(false);
  }, []);

  const toggleTerminal = useCallback(() => {
    if (terminalOpen) {
      hideTerminal();
    } else {
      showTerminal();
    }
  }, [hideTerminal, showTerminal, terminalOpen]);
  toggleTerminalRef.current = toggleTerminal;

  const toggleTerminalFullHeight = useCallback(() => {
    setTerminalFullHeight((value) => !value);
  }, []);

  useEffect(() => {
    let disposed = false;
    let unlisten: UnlistenFn | null = null;

    void listen(TERMINAL_OPEN_EVENT, () => {
      showTerminal();
    }).then((nextUnlisten) => {
      if (disposed) {
        nextUnlisten();
      } else {
        unlisten = nextUnlisten;
      }
    });

    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [showTerminal]);

  const sidebarHeightRef = useRef<HTMLDivElement | null>(null);
  const applyTopDelta = useCallback((delta: number) => {
    const el = sidebarHeightRef.current;
    if (!el) return;
    const h = el.clientHeight;
    if (h <= 0) return;
    setTopSplit((prev) => {
      const nextPx = Math.max(80, Math.min(h - 80, prev * h + delta));
      return nextPx / h;
    });
  }, []);

  const onDragFile = useCallback(
    (entry: WorkspaceEntry, event: React.DragEvent) => {
      if (entry.kind !== "file") return;
      const payload = JSON.stringify({
        relativePath: entry.relativePath,
        absolutePath: entry.absolutePath,
        name: entry.name,
      });
      event.dataTransfer.setData("application/x-sinew-file", payload);
      event.dataTransfer.setData("text/plain", entry.relativePath);
      event.dataTransfer.effectAllowed = "copy";
    },
    [],
  );

  const activeFilePath =
    !settingsActive && activeTabIndex >= 0 && tabs[activeTabIndex]
      ? tabs[activeTabIndex].relativePath
      : null;
  const terminalVisible = terminalAvailable && terminalOpen;
  const activeConversationIsStreaming = streamingConversationIds.has(
    activeConv.id,
  );
  const activeStreamingModel = activeConversationIsStreaming
    ? streamingModelsBySession.get(workspaceSessionKey(workspacePath, activeConv.id)) ?? activeConv.model
    : null;
  const chatModeModelSettings = activeConv.modeModelSettings;
  const effectiveActiveSessionKey =
    activeSessionKey ?? workspaceSessionKey(workspacePath, activeConv.id);
  const openProjectPicker = onOpenProject ?? onOpenWorkspace ?? onSwitchWorkspace;
  const toggleProjectTreeCollapsed = useCallback(() => {
    setProjectTreeCollapsed((current) => {
      const next = !current;
      saveProjectTreeCollapsed(workspacePathRef.current, next);
      return next;
    });
  }, []);
  const folderVisible = layoutVisibility.folder;
  const editorVisible = layoutVisibility.editor;
  const chatVisible = layoutVisibility.chat;
  const centerVisible = editorVisible;
  const visibleHorizontalPanels = [folderVisible, centerVisible, chatVisible].filter(
    Boolean,
  ).length;
  const chatExpanded = chatVisible && visibleHorizontalPanels === 1;
  const toggleLayoutPanel = useCallback((panel: LayoutPanel) => {
    setLayoutVisibility((current) => ({
      ...current,
      [panel]: !current[panel],
    }));
  }, []);
  const titlebarActionsStyle = {
    left: folderVisible ? leftWidth : 8,
    right: chatVisible && !chatExpanded ? rightWidth : 180,
  } satisfies React.CSSProperties;
  const conversationProjects = useMemo<ConversationListProject[] | undefined>(() => {
    if (!sessions) return undefined;

    const projects = new Map<
      string,
      {
        key: string;
        name: string;
        path: string;
        conversations: ConversationListProject["conversations"];
        streamingIds: Set<string>;
        attentionIds: Set<string>;
      }
    >();

    const seenConversationKeys = new Set<string>();

    const ensureProject = (workspace: WorkspaceBootstrap["workspace"]) => {
      const project =
        projects.get(workspace.path) ??
        {
          key: workspace.path,
          name: workspace.name,
          path: workspace.path,
          conversations: [],
          streamingIds: new Set<string>(),
          attentionIds: new Set<string>(),
        };
      projects.set(workspace.path, project);
      return project;
    };

    for (const session of sessions) {
      const sessionWorkspace = session.bootstrap.workspace;
      const sessionConversation =
        session.key === effectiveActiveSessionKey
          ? activeConv
          : session.bootstrap.activeConversation;
      const sessionConversations =
        sessionWorkspace.path === workspacePath
          ? conversations
          : session.bootstrap.conversations;
      const summary = sessionConversations.find(
        (conversation) => conversation.id === sessionConversation.id,
      );
      const project = ensureProject(sessionWorkspace);
      const conversationKey = workspaceSessionKey(
        sessionWorkspace.path,
        sessionConversation.id,
      );
      seenConversationKeys.add(conversationKey);

      project.conversations.push({
        id: sessionConversation.id,
        title: summary?.title ?? sessionConversation.title,
        updatedAtMs: summary?.updatedAtMs ?? 0,
        sessionKey: session.key,
      });

      for (const id of streamingConversationIdsByWorkspace.get(sessionWorkspace.path) ?? EMPTY_STREAMING_IDS) {
        project.streamingIds.add(id);
      }
      for (const id of attentionConversationIdsByWorkspace.get(sessionWorkspace.path) ?? EMPTY_ATTENTION_IDS) {
        project.attentionIds.add(id);
      }
    }

    for (const session of sessions) {
      const sessionWorkspace = session.bootstrap.workspace;
      const sessionConversations =
        sessionWorkspace.path === workspacePath
          ? conversations
          : session.bootstrap.conversations;
      const project = ensureProject(sessionWorkspace);
      for (const conversation of sessionConversations) {
        const conversationKey = workspaceSessionKey(
          sessionWorkspace.path,
          conversation.id,
        );
        if (seenConversationKeys.has(conversationKey)) continue;
        seenConversationKeys.add(conversationKey);
        project.conversations.push(conversation);
      }
    }

    return Array.from(projects.values()).map((project) => ({
      ...project,
      conversations: sortConversationSummaries(project.conversations),
    }));
  }, [
    attentionConversationIdsByWorkspace,
    activeConv,
    conversations,
    effectiveActiveSessionKey,
    sessions,
    streamingConversationIdsByWorkspace,
    workspacePath,
  ]);

  const selectConversationFromList = useCallback(
    (id: string, targetWorkspacePath?: string, sessionKey?: string) => {
      const resolvedWorkspacePath = targetWorkspacePath || workspacePath;
      markConversationAttention(resolvedWorkspacePath, id, false);
      if (
        sessionKey &&
        sessionKey !== effectiveActiveSessionKey &&
        onSelectSession
      ) {
        void onSelectSession(resolvedWorkspacePath, id);
        return;
      }
      if (resolvedWorkspacePath !== workspacePath) {
        if (onSelectSession) {
          void onSelectSession(resolvedWorkspacePath, id);
        }
        return;
      }
      void selectConversation(id);
    },
    [
      effectiveActiveSessionKey,
      markConversationAttention,
      onSelectSession,
      selectConversation,
      workspacePath,
    ],
  );

  const renameConversationFromList = useCallback(
    (id: string, title: string, targetWorkspacePath?: string) => {
      if (targetWorkspacePath && targetWorkspacePath !== workspacePath) {
        if (onRenameConversationSession) {
          void onRenameConversationSession(targetWorkspacePath, id, title);
        } else {
          void api
            .renameConversation(targetWorkspacePath, id, title)
            .then((next) => onWorkspaceConversationsReplace?.(targetWorkspacePath, next))
            .catch((err) => console.error(err));
        }
        return;
      }
      if (onRenameConversationSession) {
        void onRenameConversationSession(workspacePath, id, title);
        return;
      }
      void renameConversation(id, title);
    },
    [onRenameConversationSession, onWorkspaceConversationsReplace, renameConversation, workspacePath],
  );

  const deleteConversationFromList = useCallback(
    (id: string, targetWorkspacePath?: string) => {
      const resolvedWorkspacePath = targetWorkspacePath || workspacePath;
      markConversationAttention(resolvedWorkspacePath, id, false);
      if (resolvedWorkspacePath !== workspacePath) {
        if (onDeleteConversationSession) {
          void onDeleteConversationSession(resolvedWorkspacePath, id);
        }
        return;
      }
      void deleteConversation(id);
    },
    [deleteConversation, markConversationAttention, onDeleteConversationSession, workspacePath],
  );

  const archiveConversationFromList = useCallback(
    (id: string, targetWorkspacePath?: string) => {
      const resolvedWorkspacePath = targetWorkspacePath || workspacePath;
      markConversationAttention(resolvedWorkspacePath, id, false);
      if (resolvedWorkspacePath !== workspacePath) {
        if (onArchiveConversationSession) {
          void onArchiveConversationSession(resolvedWorkspacePath, id);
        } else {
          void api
            .archiveConversation(resolvedWorkspacePath, id)
            .then((next) => onWorkspaceConversationsReplace?.(resolvedWorkspacePath, next))
            .catch((err) => console.error(err));
        }
        return;
      }
      void archiveConversation(id);
    },
    [
      archiveConversation,
      markConversationAttention,
      onArchiveConversationSession,
      onWorkspaceConversationsReplace,
      workspacePath,
    ],
  );

  const openSessionSwitcher = useCallback(() => {
    setSessionsOpen(true);
  }, []);

  const closeSessionSwitcher = useCallback(() => {
    setSessionsOpen(false);
  }, []);

  const selectSessionFromSwitcher = useCallback(
    (targetWorkspacePath: string, id: string) => {
      markConversationAttention(targetWorkspacePath, id, false);
      setSessionsOpen(false);
      if (onSelectSession) {
        void onSelectSession(targetWorkspacePath, id);
        return;
      }
      selectConversationFromList(
        id,
        targetWorkspacePath,
        workspaceSessionKey(targetWorkspacePath, id),
      );
    },
    [markConversationAttention, onSelectSession, selectConversationFromList],
  );

  const createSessionFromSwitcher = useCallback(() => {
    setSessionsOpen(false);
    void createConversation();
  }, [createConversation]);

  const renameSessionFromSwitcher = useCallback(
    (targetWorkspacePath: string, id: string, title: string) => {
      renameConversationFromList(id, title, targetWorkspacePath);
      setSessionsRefreshToken((value) => value + 1);
    },
    [renameConversationFromList],
  );

  const deleteSessionFromSwitcher = useCallback(
    (targetWorkspacePath: string, id: string) => {
      deleteConversationFromList(id, targetWorkspacePath);
      setSessionsRefreshToken((value) => value + 1);
    },
    [deleteConversationFromList],
  );

  const restoreSessionFromSwitcher = useCallback(
    (targetWorkspacePath: string, id: string) => {
      if (onRestoreConversationSession) {
        void onRestoreConversationSession(targetWorkspacePath, id);
      } else {
        void api
          .restoreConversation(targetWorkspacePath, id)
          .then((next) => onWorkspaceConversationsReplace?.(targetWorkspacePath, next))
          .catch((err) => console.error(err));
      }
      setSessionsRefreshToken((value) => value + 1);
    },
    [onRestoreConversationSession, onWorkspaceConversationsReplace],
  );

  const streamingSessionKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const [streamingWorkspacePath, ids] of streamingConversationIdsByWorkspace) {
      for (const id of ids) {
        keys.add(workspaceSessionKey(streamingWorkspacePath, id));
      }
    }
    return keys;
  }, [streamingConversationIdsByWorkspace]);

  return (
    <div
      className="workspace"
      data-center-visible={centerVisible ? "true" : "false"}
      data-folder-visible={folderVisible ? "true" : "false"}
    >
      <div
        className="titlebar"
        data-tauri-drag-region
        data-platform={IS_WINDOWS ? "windows" : undefined}
      >
        <div
          className="titlebar__actions"
          data-tauri-drag-region
          style={titlebarActionsStyle}
        >
          <button
            className="titlebar__btn"
            data-on={terminalVisible ? "true" : "false"}
            onClick={toggleTerminal}
            title={terminalVisible ? "Hide terminal" : "Show terminal"}
          >
            <Icon
              icon={
                terminalVisible
                  ? "solar:command-bold-duotone"
                  : "solar:command-linear"
              }
              width={12}
              height={12}
            />
            Terminal
          </button>
          <button
            className="titlebar__btn"
            data-on={chatVisible ? "true" : "false"}
            onClick={() => toggleLayoutPanel("chat")}
            title={chatVisible ? "Hide chat" : "Show chat"}
          >
            <Icon icon="solar:chat-round-dots-linear" width={12} height={12} />
            Chat
          </button>
          <button
            className="titlebar__btn"
            data-on={editorVisible ? "true" : "false"}
            onClick={() => toggleLayoutPanel("editor")}
            title={editorVisible ? "Hide editor" : "Show editor"}
          >
            <Icon icon="solar:code-square-linear" width={12} height={12} />
            Editor
          </button>
          <button
            className="titlebar__btn"
            data-on={folderVisible ? "true" : "false"}
            onClick={() => toggleLayoutPanel("folder")}
            title={folderVisible ? "Hide folder" : "Show folder"}
          >
            <Icon icon="solar:folder-with-files-linear" width={12} height={12} />
            Folder
          </button>
          <button
            className="titlebar__btn"
            onClick={openProjectPicker}
            title={onOpenProject || onOpenWorkspace ? "Open project" : "Switch workspace"}
          >
            <Icon icon="solar:folder-with-files-linear" width={12} height={12} />
            {onOpenProject || onOpenWorkspace ? "Open" : "Switch"}
          </button>
        </div>
          <button
            className="titlebar__btn titlebar__settings-right"
            data-on={settingsActive ? "true" : "false"}
            onClick={() => openSettings()}
            title="Settings"
        >
          <Icon icon="solar:settings-linear" width={12} height={12} />
          Settings
        </button>
        <div className="titlebar__brand" data-tauri-drag-region>
          <span className="titlebar__brand-mark">
            <SinewMark size={11} />
          </span>
          <span className="titlebar__brand-name">Sinew</span>
        </div>
        <UpdateBadge />
        <WindowControls />
      </div>

      <div
        className="main"
        data-center-visible={centerVisible ? "true" : "false"}
        data-folder-visible={folderVisible ? "true" : "false"}
      >
        <div
          className="main-panels"
          data-hidden={terminalVisible && terminalFullHeight ? "true" : "false"}
        >
          {folderVisible && (
          <div
            className="sidebar"
            style={
              visibleHorizontalPanels === 1
                ? { flex: "1 1 0", minWidth: 0 }
                : { width: leftWidth, flex: `0 0 ${leftWidth}px` }
            }
            ref={sidebarHeightRef}
          >
            <div
              className="sidebar__section"
              style={
                projectTreeCollapsed
                  ? { flex: "0 0 32px" }
                  : { flex: `0 0 ${topSplit * 100}%` }
              }
              ref={fileTreeDropZoneRef}
              data-collapsed={projectTreeCollapsed ? "true" : "false"}
              data-drop-active={fileTreeDropState.active ? "true" : "false"}
            >
              <div className="sidebar__head">
                <span className="sidebar__head-title">
                  <button
                    type="button"
                    className="sidebar__head-toggle"
                    aria-expanded={!projectTreeCollapsed}
                    aria-label={
                      projectTreeCollapsed
                        ? "Show project files"
                        : "Hide project files"
                    }
                    title={
                      projectTreeCollapsed
                        ? "Show project files"
                        : "Hide project files"
                    }
                    onClick={toggleProjectTreeCollapsed}
                  >
                    <Icon
                      icon={
                        projectTreeCollapsed
                          ? "solar:alt-arrow-right-linear"
                          : "solar:alt-arrow-down-linear"
                      }
                      width={13}
                      height={13}
                    />
                  </button>
                  <Icon icon="solar:folder-bold-duotone" width={16} height={16} />
                  <span>{bootstrap.workspace.name}</span>
                </span>
                <span className="sidebar__head-actions">
                <button
                  type="button"
                  className="sidebar__head-btn"
                  title="New file"
                  onClick={() => startRootCreate("file")}
                >
                  <Icon icon="solar:document-add-linear" width={15} height={15} />
                </button>
                <button
                  type="button"
                  className="sidebar__head-btn"
                  title="New folder"
                  onClick={() => startRootCreate("directory")}
                >
                  <Icon
                    icon="solar:add-folder-linear"
                    width={15}
                    height={15}
                  />
                </button>
                <button
                  type="button"
                  className="sidebar__head-btn"
                  data-active={fileSearchOpen ? "true" : "false"}
                  title={fileSearchOpen ? "Show files" : "Search files"}
                  onClick={() => setFileSearchOpen((value) => !value)}
                >
                  <Icon
                    icon={
                      fileSearchOpen
                        ? "solar:folder-open-linear"
                        : "solar:magnifer-linear"
                    }
                    width={15}
                    height={15}
                  />
                </button>
              </span>
            </div>
              {!projectTreeCollapsed &&
                (fileSearchOpen ? (
                  <SearchPane
                    workspacePath={workspacePath}
                    onOpenFile={openFile}
                    refreshToken={fileTreeRefreshToken}
                  />
                ) : (
                  <FileTree
                    ref={fileTreeRef}
                    workspacePath={workspacePath}
                    activeFile={activeFilePath}
                    onOpenFile={openFile}
                    onDragFile={onDragFile}
                    onEntryRenamed={handleTreeEntryRenamed}
                    onEntryDeleted={handleTreeEntryDeleted}
                    onEntriesMoved={handleTreeEntriesMoved}
                    refreshToken={fileTreeRefreshToken}
                    dropActive={fileTreeDropState.active}
                    dropTargetRelative={fileTreeDropState.targetRelative}
                  />
                ))}
              {!projectTreeCollapsed && importError && (
                <div
                  className="sidebar__import-error"
                  onClick={() => setImportError(null)}
                  title="click to dismiss"
                >
                  {importError}
                </div>
              )}
          </div>
          {!projectTreeCollapsed && (
            <Splitter orientation="horizontal" onDelta={applyTopDelta} />
          )}
          <div
            className="sidebar__section sidebar__section--bottom"
            style={{ flex: "1 1 0" }}
          >
            <div className="sidebar__head sidebar__head--tabs">
              <div className="sidebar-tabs" role="tablist">
                <button
                  type="button"
                  role="tab"
                  className="sidebar-tab"
                  data-active={bottomTab === "conversations" ? "true" : "false"}
                  aria-selected={bottomTab === "conversations"}
                  onClick={() => setBottomTab("conversations")}
                >
                  <Icon
                    icon="solar:chat-round-dots-linear"
                    width={13}
                    height={13}
                  />
                  <span>{conversationProjects ? "Projects" : "Conversations"}</span>
                </button>
                <button
                  type="button"
                  role="tab"
                  className="sidebar-tab"
                  data-active={bottomTab === "git" ? "true" : "false"}
                  aria-selected={bottomTab === "git"}
                  onClick={() => setBottomTab("git")}
                >
                  <GitMark size={13} />
                  <span>Git</span>
                </button>
              </div>
              {bottomTab === "conversations" && (
                <div className="sidebar__head-actions">
                  <button
                    type="button"
                    className="sidebar__head-btn"
                    onClick={openSessionSwitcher}
                    title="Sessions"
                  >
                    <Icon icon="solar:clock-circle-linear" width={15} height={15} />
                  </button>
                  {(onOpenProject || onOpenWorkspace) && (
                    <button
                      type="button"
                      className="sidebar__head-btn"
                      onClick={openProjectPicker}
                      title="Open project"
                    >
                      <Icon icon="solar:add-folder-linear" width={15} height={15} />
                    </button>
                  )}
                  <button
                    type="button"
                    className="sidebar__head-btn"
                    onClick={() => createConversation()}
                    title="New conversation"
                  >
                    <Icon
                      icon="solar:add-square-linear"
                      width={15}
                      height={15}
                    />
                  </button>
                </div>
              )}
            </div>
            {/*
              Both panels stay mounted so swapping tabs preserves any
              transient state (a half-typed commit message, an open
              "New worktree" form, etc). We toggle visibility with
              display:none to avoid double-mounting their effects.
            */}
            <div
              className="sidebar-tab-pane"
              role="tabpanel"
              aria-hidden={bottomTab !== "conversations"}
              style={{
                display: bottomTab === "conversations" ? "flex" : "none",
              }}
            >
              <ConversationList
                conversations={conversations}
                activeId={activeConv.id}
                streamingIds={streamingConversationIds}
                attentionIds={attentionConversationIds}
                projects={conversationProjects}
                activeSessionKey={effectiveActiveSessionKey}
                onSelect={selectConversationFromList}
                onCreate={createConversation}
                onRename={renameConversationFromList}
                onDelete={deleteConversationFromList}
                onArchive={archiveConversationFromList}
                onCloseProject={onCloseProjectSession}
              />
            </div>
            <div
              className="sidebar-tab-pane"
              role="tabpanel"
              aria-hidden={bottomTab !== "git"}
              style={{
                display: bottomTab === "git" ? "flex" : "none",
              }}
            >
              <GitPanel
                workspacePath={workspacePath}
                active={bottomTab === "git"}
                hasStreamingConversation={hasStreamingConversation}
                onSwitchWorkspace={switchWorkspace}
              />
            </div>
          </div>
          </div>
        )}
        {folderVisible && (centerVisible || chatVisible) && (
          <Splitter
            orientation="vertical"
            onDelta={(delta) => setLeftWidth((v) => clampColumn(v + delta))}
          />
        )}
        {centerVisible && (
          <div className="workbench-center">
            <div
              className="editor-shell"
              data-hidden="false"
            >
            <EditorPane
              tabs={tabs}
              activeIndex={activeTabIndex}
              onActivate={activateFileTab}
              onClose={closeTab}
              onChange={updateBuffer}
              onSave={saveTab}
              onOpenFile={openChatFile}
              settingsOpen={settingsOpen}
              settingsActive={settingsActive}
              settingsView={<SettingsPane workspacePath={workspacePath} />}
              revealTarget={editorRevealTarget}
              onSettingsActivate={() => setSettingsActive(true)}
              onSettingsClose={closeSettings}
            />
          </div>

          </div>
        )}
        {centerVisible && chatVisible && (
          <Splitter
            orientation="vertical"
            onDelta={(delta) => setRightWidth((v) => clampColumn(v - delta))}
          />
        )}
        {chatVisible && (
          <div
            className="chat-stack"
            data-expanded={chatExpanded ? "true" : "false"}
            data-visible={chatVisible ? "true" : "false"}
            style={
              chatExpanded
                ? {
                    flex: "1 1 0",
                    minWidth: 0,
                  }
              : centerVisible
                ? {
                    width: rightWidth,
                    flex: `0 0 ${rightWidth}px`,
                    minWidth: 0,
                  }
                : {
                    flex: "1 1 0",
                    minWidth: 0,
                  }
            }
          >
            <div
              className="chat-shell"
              data-expanded={chatExpanded ? "true" : "false"}
              data-hidden="false"
            >
              <ChatPane
                workspacePath={workspacePath}
                conversationId={activeConv.id}
                activeModel={activeConv.model}
                modeModelSettings={chatModeModelSettings}
                streamingModel={activeStreamingModel}
                planWorkflow={activeConv.planWorkflow}
                goalWorkflow={activeConv.goalWorkflow}
                isStreaming={activeConversationIsStreaming}
                history={activeConv.history}
                subscribeEvents={subscribeEvents}
                onSend={sendMessage}
                onCompact={compactConversation}
                onModeChange={changeConversationMode}
                onModelPreferenceChange={changeConversationModelPreference}
                onImplementPlanFresh={implementPlanFresh}
                onStop={stopTurn}
                onOpenFile={openChatFile}
                onOpenSessions={openSessionSwitcher}
                onOpenSettings={openSettings}
                externalDrops={externalDropFeed}
                dropZoneRef={chatDropZoneRef}
              />
            </div>
          </div>
        )}
        </div>
        {terminalVisible && !terminalFullHeight && (
          <Splitter
            orientation="horizontal"
            onDelta={(delta) =>
              setTerminalHeight((value) => clampTerminal(value - delta))
            }
          />
        )}
        <div
          className="terminal-shell"
          data-full-height={terminalFullHeight ? "true" : "false"}
          style={{
            display: terminalVisible ? "block" : "none",
            height: terminalVisible
              ? terminalFullHeight
                ? "auto"
                : terminalHeight
              : 0,
            flex: terminalVisible
              ? terminalFullHeight
                ? "1 1 0"
                : `0 0 ${terminalHeight}px`
              : "0 0 0",
          }}
        >
          {terminalAvailable && (
            <TerminalPanel
              active={terminalVisible}
              fullHeight={terminalFullHeight}
              workspacePath={workspacePath}
              onClose={hideTerminal}
              onCloseLastSession={closeTerminalPanel}
              onToggleFullHeight={toggleTerminalFullHeight}
              onOpenTerminalPath={openTerminalPath}
            />
          )}
        </div>
        {terminalAvailable && !terminalOpen && (
          <div className="terminal-restore">
            <button
              type="button"
              className="terminal-restore__button"
              onClick={showTerminal}
              title="Show terminal"
            >
              <Icon icon="solar:square-alt-arrow-up-linear" width={14} height={14} />
            </button>
          </div>
        )}
      </div>
      {sessionsOpen && (
        <SessionSwitcher
          activeWorkspacePath={workspacePath}
          activeSessionKey={effectiveActiveSessionKey}
          streamingSessionKeys={streamingSessionKeys}
          refreshToken={sessionsRefreshToken}
          onSelect={selectSessionFromSwitcher}
          onCreate={createSessionFromSwitcher}
          onRename={renameSessionFromSwitcher}
          onDelete={deleteSessionFromSwitcher}
          onRestore={restoreSessionFromSwitcher}
          onClose={closeSessionSwitcher}
        />
      )}
    </div>
  );
}

function loadLayoutVisibility(): LayoutVisibility {
  const fallback: LayoutVisibility = { folder: true, editor: true, chat: true };
  try {
    if (typeof window === "undefined") return fallback;
    const raw = window.localStorage.getItem(LAYOUT_PANEL_VISIBILITY_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<LayoutVisibility>;
      const next = {
        folder: typeof parsed.folder === "boolean" ? parsed.folder : fallback.folder,
        editor: typeof parsed.editor === "boolean" ? parsed.editor : fallback.editor,
        chat: typeof parsed.chat === "boolean" ? parsed.chat : fallback.chat,
      };
      return next.folder || next.editor || next.chat ? next : fallback;
    }

    const oldRaw = window.localStorage.getItem(LAYOUT_VIEW_MODE_KEY);
    if (oldRaw === "chat") return { folder: true, editor: false, chat: true };
    if (oldRaw === "editor") return { folder: true, editor: true, chat: false };

    const oldChatFocus = window.localStorage.getItem("sinew.layout.chatFocus") === "true";
    const oldCenterVisible = window.localStorage.getItem("sinew.layout.centerVisible");
    if (oldChatFocus || oldCenterVisible === "false") {
      return { folder: true, editor: false, chat: true };
    }
    return fallback;
  } catch {
    return fallback;
  }
}

function saveLayoutVisibility(value: LayoutVisibility): void {
  try {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(LAYOUT_PANEL_VISIBILITY_KEY, JSON.stringify(value));
  } catch {
    // Ignore storage errors; layout controls still work for the session.
  }
}

function loadProjectTreeCollapsed(workspacePath: string): boolean {
  try {
    if (typeof window === "undefined" || !workspacePath) return false;
    const raw = window.localStorage.getItem(PROJECT_TREE_COLLAPSED_KEY);
    if (!raw) return false;
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    return parsed[workspacePath] === true;
  } catch {
    return false;
  }
}

function saveProjectTreeCollapsed(workspacePath: string, collapsed: boolean): void {
  try {
    if (typeof window === "undefined" || !workspacePath) return;
    const raw = window.localStorage.getItem(PROJECT_TREE_COLLAPSED_KEY);
    const parsed = raw ? (JSON.parse(raw) as Record<string, unknown>) : {};
    const next: Record<string, boolean> = {};
    for (const [path, value] of Object.entries(parsed)) {
      if (value === true) next[path] = true;
    }
    if (collapsed) {
      next[workspacePath] = true;
    } else {
      delete next[workspacePath];
    }
    window.localStorage.setItem(PROJECT_TREE_COLLAPSED_KEY, JSON.stringify(next));
  } catch {
    // Ignore storage errors; the project tree still toggles for the session.
  }
}

async function sendMessageWithBusyRetry(
  ...args: Parameters<typeof api.sendMessage>
): Promise<void> {
  for (let attempt = 0; ; attempt += 1) {
    try {
      await api.sendMessage(...args);
      return;
    } catch (err) {
      const delayMs = SEND_BUSY_RETRY_DELAYS_MS[attempt];
      if (!isConversationBusyError(err) || delayMs === undefined) {
        throw err;
      }
      await sleep(delayMs);
    }
  }
}

function isConversationBusyError(err: unknown): boolean {
  return String(err).includes("a turn is already running for this conversation");
}

function conversationContinuationMode(conversation: SavedConversation): AgentMode {
  if (conversation.planWorkflow.status !== "idle") return "plan";
  if (conversation.goalWorkflow.status === "active") return "goal";
  return "act";
}

const CONVERSATION_TITLE_MAX_WORDS = 6;
const CONVERSATION_TITLE_MAX_CHARS = 48;

function titleFromOutgoingUserText(text: string): string | null {
  const words = text.trim().split(/\s+/).filter(Boolean).slice(0, CONVERSATION_TITLE_MAX_WORDS);
  if (words.length === 0) return null;
  const title = words.join(" ").replace(/[\s"'`“”‘’«»*_—:;.?!-]+$/u, "");
  if (!title) return null;
  const chars = Array.from(title);
  if (chars.length <= CONVERSATION_TITLE_MAX_CHARS) return title;
  return `${chars.slice(0, CONVERSATION_TITLE_MAX_CHARS - 1).join("")}…`;
}

function titleFromPlanImplementation(plan: PlanArtifact): string {
  const planTitle = plan.title?.trim();
  const fileTitle = basename(plan.path).replace(/\.md$/i, "").trim();
  const base = planTitle || fileTitle || "plan";
  return titleFromOutgoingUserText(`Implement: ${base}`) ?? "Implement plan";
}

function sortConversationSummaries(
  conversations: ConversationSummary[],
): ConversationSummary[] {
  return [...conversations].sort((a, b) => b.updatedAtMs - a.updatedAtMs);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function basename(path: string): string {
  const idx = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return idx >= 0 ? path.slice(idx + 1) : path;
}

function fileChangesFromAgentEvent(event: AgentEvent): FileChange[] {
  if (event.type === "tool_finished") return event.file_changes;
  if (event.type === "sub_agent_event") {
    return fileChangesFromAgentEvent(event.event);
  }
  return [];
}

function entryContainsPath(entry: WorkspaceEntry, relativePath: string): boolean {
  return (
    relativePath === entry.relativePath ||
    (entry.kind === "directory" &&
      relativePath.startsWith(`${entry.relativePath}/`))
  );
}

function replaceTreePath(
  relativePath: string,
  oldRelativePath: string,
  entry: WorkspaceEntry,
): string | null {
  if (relativePath === oldRelativePath) return entry.relativePath;
  if (relativePath.startsWith(`${oldRelativePath}/`)) {
    return `${entry.relativePath}${relativePath.slice(oldRelativePath.length)}`;
  }
  return null;
}

function retargetTab(
  tab: EditorTab,
  relativePath: string,
  workspacePath: string,
  entry: WorkspaceEntry,
): EditorTab {
  const exactEntry = relativePath === entry.relativePath && entry.kind === "file";
  const absolutePath = exactEntry
    ? entry.absolutePath
    : `${workspacePath}/${relativePath}`;
  const name = basename(relativePath);
  return {
    ...tab,
    relativePath,
    doc: {
      ...tab.doc,
      name,
      relativePath,
      absolutePath,
    },
  };
}

function chatPathToRelative(rawPath: string, workspacePath: string): string | null {
  let path = rawPath
    .trim()
    .replace(/^['"`<]+|['"`>,.;:]+$/g, "")
    .replace(/#L\d+(?:C\d+)?$/i, "")
    .replace(/:\d+(?::\d+)?$/, "")
    .replace(/\\/g, "/");

  if (!path || path.includes("://")) return null;

  const root = workspacePath.replace(/\\/g, "/").replace(/\/+$/, "");
  if (path === root) return null;
  if (path.startsWith(`${root}/`)) {
    path = path.slice(root.length + 1);
  } else if (path.startsWith("/")) {
    return null;
  }

  path = path.replace(/^\.\//, "");
  if (!path || path.startsWith("../") || path.includes("/../")) return null;

  return path;
}
