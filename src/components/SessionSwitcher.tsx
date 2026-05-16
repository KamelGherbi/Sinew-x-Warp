import { useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "@iconify/react";
import { api } from "../lib/ipc";
import type { SessionSummary } from "../types";

type Props = {
  activeWorkspacePath: string;
  activeSessionKey: string | null;
  streamingSessionKeys: ReadonlySet<string>;
  refreshToken?: number;
  onSelect: (workspacePath: string, id: string) => void;
  onCreate: () => void;
  onRename: (workspacePath: string, id: string, title: string) => void;
  onDelete: (workspacePath: string, id: string) => void;
  onClose: () => void;
};

type GroupedSessions = {
  label: string;
  sessions: SessionSummary[];
};

type SessionScope = "current" | "all";

export function SessionSwitcher({
  activeWorkspacePath,
  activeSessionKey,
  streamingSessionKeys,
  refreshToken,
  onSelect,
  onCreate,
  onRename,
  onDelete,
  onClose,
}: Props) {
  const [query, setQuery] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [scope, setScope] = useState<SessionScope>("current");
  const searchRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    searchRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  useEffect(() => {
    let disposed = false;
    setLoading(true);
    setError(null);
    api
      .listSessions(undefined, 300)
      .then((loaded) => {
        if (disposed) return;
        setSessions(loaded);
      })
      .catch((err) => {
        if (!disposed) setError(String(err));
      })
      .finally(() => {
        if (!disposed) setLoading(false);
      });
    return () => {
      disposed = true;
    };
  }, [refreshToken]);

  const scopedSessions = useMemo(
    () =>
      scope === "current"
        ? sessions.filter((session) => session.workspaceId === activeWorkspacePath)
        : sessions,
    [activeWorkspacePath, scope, sessions],
  );

  const grouped = useMemo(
    () => groupSessions(filterSessions(scopedSessions, query)),
    [scopedSessions, query],
  );

  const resultCount = grouped.reduce(
    (total, group) => total + group.sessions.length,
    0,
  );

  const beginRename = (session: SessionSummary) => {
    setEditingId(sessionKey(session));
    setEditingTitle(session.title || "Untitled");
  };

  const commitRename = () => {
    if (!editingId) return;
    const next = editingTitle.trim();
    const current = sessions.find((session) => sessionKey(session) === editingId);
    setEditingId(null);
    setEditingTitle("");
    if (current && next && next !== current.title) {
      onRename(current.workspaceId, current.id, next);
      setSessions((items) =>
        items.map((session) =>
          sessionKey(session) === editingId ? { ...session, title: next } : session,
        ),
      );
    }
  };

  return (
    <div className="session-switcher" role="dialog" aria-modal="true">
      <button
        type="button"
        className="session-switcher__backdrop"
        aria-label="Close sessions"
        onClick={onClose}
      />
      <div className="session-switcher__panel">
        <div className="session-switcher__head">
          <div>
            <div className="session-switcher__eyebrow">
              {scope === "current" ? "Project sessions" : "Global sessions"}
            </div>
            <div className="session-switcher__title">
              {scope === "current"
                ? activeProjectName(scopedSessions, activeWorkspacePath)
                : "All projects"}
            </div>
          </div>
          <button
            type="button"
            className="session-switcher__close"
            onClick={onClose}
            title="Close"
          >
            <Icon icon="solar:close-circle-linear" width={18} height={18} />
          </button>
        </div>

        <div className="session-switcher__search">
          <Icon icon="solar:magnifer-linear" width={16} height={16} />
          <input
            ref={searchRef}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search sessions..."
          />
          <span>{resultCount}</span>
        </div>

        <div className="session-switcher__toolbar">
          <button type="button" onClick={onCreate}>
            <Icon icon="solar:add-square-linear" width={15} height={15} />
            New session
          </button>
          <div className="session-switcher__scope" role="group" aria-label="Session scope">
            <button
              type="button"
              data-active={scope === "current" ? "true" : "false"}
              onClick={() => setScope("current")}
            >
              Current project
            </button>
            <button
              type="button"
              data-active={scope === "all" ? "true" : "false"}
              onClick={() => setScope("all")}
            >
              All projects
            </button>
          </div>
          <span>Type /sessions, /session, /resume or /continue in chat.</span>
        </div>

        <div className="session-switcher__list">
          {loading && <div className="session-switcher__empty">Loading sessions…</div>}
          {error && <div className="session-switcher__empty">{error}</div>}
          {!loading && !error && grouped.length === 0 && (
            <div className="session-switcher__empty">
              {scope === "current"
                ? "No matching sessions in this project."
                : "No matching sessions."}
            </div>
          )}
          {grouped.map((group) => (
            <section className="session-switcher__group" key={group.label}>
              <div className="session-switcher__group-label">{group.label}</div>
              {group.sessions.map((session) => {
                const key = sessionKey(session);
                const isActive = key === activeSessionKey;
                const isStreaming = streamingSessionKeys.has(key);
                const isEditing = editingId === key;
                return (
                  <div
                    className="session-switcher__row"
                    data-active={isActive ? "true" : "false"}
                    data-streaming={isStreaming ? "true" : "false"}
                    key={key}
                    onClick={() => {
                      if (isEditing) return;
                      onSelect(session.workspaceId, session.id);
                    }}
                  >
                    <span className="session-switcher__row-icon">
                      {isStreaming ? (
                        <span className="session-switcher__spinner" />
                      ) : (
                        <Icon
                          icon={
                            isActive
                              ? "solar:chat-round-dots-bold"
                              : "solar:chat-round-dots-linear"
                          }
                          width={16}
                          height={16}
                        />
                      )}
                    </span>
                    <div className="session-switcher__row-main">
                      {isEditing ? (
                        <input
                          className="session-switcher__rename"
                          value={editingTitle}
                          autoFocus
                          onChange={(event) => setEditingTitle(event.target.value)}
                          onClick={(event) => event.stopPropagation()}
                          onBlur={commitRename}
                          onKeyDown={(event) => {
                            if (event.key === "Enter") {
                              event.preventDefault();
                              commitRename();
                            } else if (event.key === "Escape") {
                              event.preventDefault();
                              setEditingId(null);
                              setEditingTitle("");
                            }
                          }}
                        />
                      ) : (
                        <>
                          <div className="session-switcher__row-title">
                            {session.title || "Untitled"}
                          </div>
                          <div className="session-switcher__row-meta">
                            <span title={session.workspaceId}>{session.workspaceName}</span>
                            {isActive ? "Current session" : formatSessionTime(session.updatedAtMs)}
                            <span>{session.messageCount} messages</span>
                            {session.workspaceId !== activeWorkspacePath && <span>Other project</span>}
                            {isStreaming && <span>Running</span>}
                          </div>
                        </>
                      )}
                    </div>
                    <div className="session-switcher__row-actions">
                      <button
                        type="button"
                        title="Rename"
                        onClick={(event) => {
                          event.stopPropagation();
                          beginRename(session);
                        }}
                      >
                        <Icon icon="solar:pen-linear" width={14} height={14} />
                      </button>
                      <button
                        type="button"
                        title="Delete"
                        className="session-switcher__danger"
                        onClick={(event) => {
                          event.stopPropagation();
                          if (confirm("Delete this session?")) {
                            onDelete(session.workspaceId, session.id);
                            setSessions((items) =>
                              items.filter((item) => sessionKey(item) !== key),
                            );
                          }
                        }}
                      >
                        <Icon icon="solar:trash-bin-minimalistic-linear" width={14} height={14} />
                      </button>
                    </div>
                  </div>
                );
              })}
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}

function filterSessions(
  sessions: SessionSummary[],
  query: string,
): SessionSummary[] {
  const needle = query.trim().toLowerCase();
  const sorted = [...sessions].sort((a, b) => b.updatedAtMs - a.updatedAtMs);
  if (!needle) return sorted;
  return sorted.filter((session) =>
    `${session.title || "Untitled"} ${session.workspaceName} ${session.workspaceId}`
      .toLowerCase()
      .includes(needle),
  );
}

function groupSessions(sessions: SessionSummary[]): GroupedSessions[] {
  const groups = new Map<string, SessionSummary[]>();
  for (const session of sessions) {
    const label = sessionDateGroup(session.updatedAtMs);
    groups.set(label, [...(groups.get(label) ?? []), session]);
  }
  return Array.from(groups, ([label, groupedSessions]) => ({
    label,
    sessions: groupedSessions,
  }));
}

function sessionKey(session: SessionSummary): string {
  return `${session.workspaceId}\u001f${session.id}`;
}

function activeProjectName(
  sessions: SessionSummary[],
  activeWorkspacePath: string,
): string {
  return (
    sessions.find((session) => session.workspaceId === activeWorkspacePath)
      ?.workspaceName ?? workspaceNameFromPath(activeWorkspacePath)
  );
}

function workspaceNameFromPath(path: string): string {
  const trimmed = path.replace(/[\\/]+$/, "");
  const separator = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return separator >= 0 ? trimmed.slice(separator + 1) || trimmed : trimmed;
}

function sessionDateGroup(timestamp: number): string {
  if (!timestamp) return "Older";
  const now = startOfDay(Date.now());
  const day = startOfDay(timestamp);
  const diffDays = Math.floor((now - day) / 86_400_000);
  if (diffDays <= 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return "Previous 7 days";
  return "Older";
}

function startOfDay(timestamp: number): number {
  const date = new Date(timestamp);
  date.setHours(0, 0, 0, 0);
  return date.getTime();
}

function formatSessionTime(timestamp: number): string {
  if (!timestamp) return "Unknown time";
  const date = new Date(timestamp);
  const now = new Date();
  if (date.toDateString() === now.toDateString()) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
}
