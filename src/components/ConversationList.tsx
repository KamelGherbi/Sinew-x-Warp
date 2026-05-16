import { useMemo, useRef, useState } from "react";
import { Icon } from "@iconify/react";
import type { ConversationSummary } from "../types";

export type ConversationListProject = {
  key: string;
  name: string;
  path: string;
  conversations: ConversationListConversation[];
  streamingIds: ReadonlySet<string>;
};

export type ConversationListConversation = ConversationSummary & {
  sessionKey?: string;
};

type Props = {
  conversations: ConversationSummary[];
  activeId: string | null;
  streamingIds: ReadonlySet<string>;
  projects?: ConversationListProject[];
  activeSessionKey?: string | null;
  onSelect: (id: string, workspacePath?: string, sessionKey?: string) => void;
  onCreate: (workspacePath?: string) => void;
  onRename: (id: string, title: string, workspacePath?: string) => void;
  onDelete: (id: string, workspacePath?: string) => void;
  onOpenProject?: () => void;
};

export function ConversationList({
  conversations,
  activeId,
  streamingIds,
  projects,
  activeSessionKey,
  onSelect,
  onCreate,
  onRename,
  onDelete,
  onOpenProject,
}: Props) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [collapsedProjects, setCollapsedProjects] = useState<Set<string>>(
    () => new Set(),
  );
  const editRef = useRef<HTMLSpanElement | null>(null);

  const displayProjects = useMemo<ConversationListProject[]>(() => {
    if (projects) return projects;
    return [
      {
        key: "active-workspace",
        name: "Current project",
        path: "",
        conversations,
        streamingIds,
      },
    ];
  }, [conversations, projects, streamingIds]);

  const hasProjectGroups = Boolean(projects);

  const commitRename = (id: string, workspacePath?: string) => {
    const value = editRef.current?.textContent?.trim() ?? "";
    setEditingId(null);
    if (value) {
      onRename(id, value, workspacePath);
    }
  };

  const toggleProject = (key: string) => {
    setCollapsedProjects((current) => {
      const next = new Set(current);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  return (
    <div className="sidebar__section" style={{ flex: "1 1 0" }}>
      <div className="sidebar__head">
        <span className="sidebar__head-title">
          <Icon icon="solar:widget-5-bold-duotone" width={16} height={16} />
          <span>{hasProjectGroups ? "Projects" : "Conversations"}</span>
        </span>
        <span className="sidebar__head-actions">
          {onOpenProject && (
            <button
              className="sidebar__head-btn"
              onClick={onOpenProject}
              title="Open project"
            >
              <Icon icon="solar:add-folder-linear" width={15} height={15} />
            </button>
          )}
          <button
            className="sidebar__head-btn"
            onClick={() => onCreate()}
            title="New conversation"
          >
            <Icon icon="solar:add-square-linear" width={15} height={15} />
          </button>
        </span>
      </div>
      <div className="sidebar__body">
        <div className="conv-list" data-grouped={hasProjectGroups ? "true" : "false"}>
          {displayProjects.length === 0 && (
            <div className="conv-empty">No projects open.</div>
          )}
          {displayProjects.map((project) => {
            const isCollapsed = collapsedProjects.has(project.key);
            const streamingCount = project.conversations.reduce(
              (count, conv) => count + (project.streamingIds.has(conv.id) ? 1 : 0),
              0,
            );
            return (
              <div className="conv-project" key={project.key}>
                {hasProjectGroups && (
                  <div className="conv-project__head">
                    <button
                      type="button"
                      className="conv-project__toggle"
                      onClick={() => toggleProject(project.key)}
                      title={project.path}
                    >
                      <Icon
                        icon={
                          isCollapsed
                            ? "solar:alt-arrow-right-linear"
                            : "solar:alt-arrow-down-linear"
                        }
                        width={13}
                        height={13}
                      />
                      <span className="conv-project__name">{project.name}</span>
                    </button>
                    <span className="conv-project__meta">
                      {streamingCount > 0 && (
                        <span className="conv-project__streaming" title="Streaming conversations">
                          {streamingCount}
                        </span>
                      )}
                      <button
                        type="button"
                        className="conv-project__action"
                        title={`New conversation in ${project.name}`}
                        aria-label={`New conversation in ${project.name}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          onCreate(project.path);
                        }}
                      >
                        <Icon icon="solar:add-square-linear" width={14} height={14} />
                      </button>
                      <span>{project.conversations.length}</span>
                    </span>
                  </div>
                )}
                {!isCollapsed && project.conversations.length === 0 && (
                  <div className="conv-empty">No conversations yet.</div>
                )}
                {!isCollapsed &&
                  project.conversations.map((conv) => {
                    const rowKey = conv.sessionKey ?? `${project.path}::${conv.id}`;
                    const isEditing = editingId === rowKey;
                    const isActive = activeSessionKey
                      ? activeSessionKey === rowKey
                      : activeId === conv.id;
                    const isStreaming = project.streamingIds.has(conv.id);
                    return (
                      <div
                        key={rowKey}
                        className="conv-row"
                        data-active={isActive ? "true" : "false"}
                        data-streaming={isStreaming ? "true" : "false"}
                        data-grouped={hasProjectGroups ? "true" : "false"}
                        onClick={() =>
                          !isEditing && onSelect(conv.id, project.path, conv.sessionKey)
                        }
                      >
                        <span className="conv-row__icon">
                          {isStreaming ? (
                            <span className="conv-row__spinner" aria-label="Streaming" />
                          ) : (
                            <Icon
                              icon={
                                isActive
                                  ? "solar:chat-round-dots-bold"
                                  : "solar:chat-round-dots-linear"
                              }
                              width={15}
                              height={15}
                            />
                          )}
                        </span>
                        <span
                          ref={isEditing ? editRef : undefined}
                          className="conv-row__title"
                          contentEditable={isEditing}
                          suppressContentEditableWarning
                          onKeyDown={(event) => {
                            if (!isEditing) return;
                            if (event.key === "Enter") {
                              event.preventDefault();
                              commitRename(conv.id, project.path);
                            } else if (event.key === "Escape") {
                              setEditingId(null);
                            }
                          }}
                          onBlur={() => {
                            if (isEditing) commitRename(conv.id, project.path);
                          }}
                        >
                          {conv.title || "Untitled"}
                        </span>
                        <span className="conv-row__actions">
                          <button
                            className="conv-row__btn"
                            title="Rename"
                            onClick={(event) => {
                              event.stopPropagation();
                              setEditingId(rowKey);
                              queueMicrotask(() => {
                                const node = editRef.current;
                                if (node) {
                                  node.focus();
                                  const sel = window.getSelection();
                                  const range = document.createRange();
                                  range.selectNodeContents(node);
                                  sel?.removeAllRanges();
                                  sel?.addRange(range);
                                }
                              });
                            }}
                          >
                            <Icon icon="solar:pen-linear" width={13} height={13} />
                          </button>
                          <button
                            className="conv-row__btn conv-row__btn--danger"
                            title="Delete"
                            onClick={(event) => {
                              event.stopPropagation();
                              if (confirm("Delete this conversation?")) {
                                onDelete(conv.id, project.path);
                              }
                            }}
                          >
                            <Icon
                              icon="solar:trash-bin-minimalistic-linear"
                              width={13}
                              height={13}
                            />
                          </button>
                        </span>
                      </div>
                    );
                  })}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
