import { Fragment, useMemo, useRef, useState } from "react";
import { Icon } from "@iconify/react";
import type { ConversationSummary } from "../types";

const EMPTY_IDS: ReadonlySet<string> = new Set<string>();

export type ConversationListProject = {
  key: string;
  name: string;
  path: string;
  conversations: ConversationListConversation[];
  streamingIds: ReadonlySet<string>;
  // Conversations whose AI turn has finished and should stay easy to notice
  // until the user opens them. Optional so existing callers keep compiling
  // until the parent wires the state.
  attentionIds?: ReadonlySet<string>;
};

export type ConversationListConversation = ConversationSummary & {
  sessionKey?: string;
};

type Props = {
  conversations: ConversationSummary[];
  activeId: string | null;
  streamingIds: ReadonlySet<string>;
  // Finished/attention conversations for the single-project (ungrouped) view.
  attentionIds?: ReadonlySet<string>;
  projects?: ConversationListProject[];
  activeSessionKey?: string | null;
  onSelect: (id: string, workspacePath?: string, sessionKey?: string) => void;
  onCreate: (workspacePath?: string) => void;
  onRename: (id: string, title: string, workspacePath?: string) => void;
  onDelete: (id: string, workspacePath?: string) => void;
  onArchive: (id: string, workspacePath?: string) => void;
  onCloseProject?: (workspacePath: string) => void;
};

// A conversation row decorated with its current status so we can both order
// and render it without recomputing. `surfaced` rows (running + finished) are
// pinned to the top of each project so they stay reachable amid long history.
type DecoratedRow = {
  conv: ConversationListConversation;
  rowKey: string;
  isActive: boolean;
  isStreaming: boolean;
  isAttention: boolean;
};

// Lower rank floats to the top: running first, finished/attention next, then
// the rest of the history in its incoming (recency) order.
function rowRank(row: DecoratedRow): number {
  if (row.isStreaming) return 0;
  if (row.isAttention) return 1;
  return 2;
}

// Renders only the body of the conversations list. The parent owns the
// `sidebar__section` shell and the head (shared with the Git tab), so this
// component focuses on the project/conversation rows.
export function ConversationList({
  conversations,
  activeId,
  streamingIds,
  attentionIds,
  projects,
  activeSessionKey,
  onSelect,
  onCreate,
  onRename,
  onDelete,
  onArchive,
  onCloseProject,
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
        attentionIds: attentionIds ?? EMPTY_IDS,
      },
    ];
  }, [conversations, projects, streamingIds, attentionIds]);

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
    <div className="sidebar__body">
      <div className="conv-list" data-grouped={hasProjectGroups ? "true" : "false"}>
        {displayProjects.length === 0 && (
          <div className="conv-empty">No projects open.</div>
        )}
        {displayProjects.map((project) => {
          const isCollapsed = collapsedProjects.has(project.key);
          const attentionSet = project.attentionIds ?? EMPTY_IDS;

          // Decorate every conversation with its status once, then derive the
          // pinned order and head counts from the same source of truth.
          const decorated: DecoratedRow[] = project.conversations.map((conv) => {
            const rowKey = conv.sessionKey ?? `${project.path}::${conv.id}`;
            const isActive = activeSessionKey
              ? activeSessionKey === rowKey
              : activeId === conv.id;
            const isStreaming = project.streamingIds.has(conv.id);
            // Once a conversation is open (active) or running it no longer
            // needs an attention nudge, so suppress it in both states.
            const isAttention =
              !isStreaming && !isActive && attentionSet.has(conv.id);
            return { conv, rowKey, isActive, isStreaming, isAttention };
          });

          const orderedRows = decorated
            .map((row, index) => ({ row, index }))
            .sort(
              (a, b) =>
                rowRank(a.row) - rowRank(b.row) || a.index - b.index,
            )
            .map((entry) => entry.row);

          const streamingCount = decorated.reduce(
            (count, row) => count + (row.isStreaming ? 1 : 0),
            0,
          );
          const attentionCount = decorated.reduce(
            (count, row) => count + (row.isAttention ? 1 : 0),
            0,
          );
          const surfacedCount = streamingCount + attentionCount;

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
                      <span className="conv-project__streaming" title="Running conversations">
                        {streamingCount}
                      </span>
                    )}
                    {attentionCount > 0 && (
                      <span
                        className="conv-project__attention"
                        title="Finished — open to view"
                      >
                        {attentionCount}
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
                    {onCloseProject && (
                      <button
                        type="button"
                        className="conv-project__action conv-project__action--danger"
                        title={
                          streamingCount > 0
                            ? "Stop running conversations before closing this project"
                            : `Close ${project.name}`
                        }
                        aria-label={`Close ${project.name}`}
                        disabled={streamingCount > 0}
                        onClick={(event) => {
                          event.stopPropagation();
                          if (streamingCount === 0) onCloseProject(project.path);
                        }}
                      >
                        <Icon icon="solar:close-circle-linear" width={14} height={14} />
                      </button>
                    )}
                    <span>{project.conversations.length}</span>
                  </span>
                </div>
              )}
              {!isCollapsed && project.conversations.length === 0 && (
                <div className="conv-empty">No conversations yet.</div>
              )}
              {!isCollapsed &&
                orderedRows.map((row, index) => {
                  const { conv, rowKey, isActive, isStreaming, isAttention } = row;
                  const isEditing = editingId === rowKey;
                  // Separate the pinned (running + finished) rows from the rest
                  // of the history so the surfaced section is obvious.
                  const showDivider =
                    surfacedCount > 0 &&
                    surfacedCount < orderedRows.length &&
                    index === surfacedCount;
                  return (
                    <Fragment key={rowKey}>
                      {showDivider && (
                        <div
                          className="conv-row__divider"
                          role="separator"
                          data-grouped={hasProjectGroups ? "true" : "false"}
                        />
                      )}
                      <div
                        className="conv-row"
                        data-active={isActive ? "true" : "false"}
                        data-streaming={isStreaming ? "true" : "false"}
                        data-attention={isAttention ? "true" : "false"}
                        data-grouped={hasProjectGroups ? "true" : "false"}
                        onClick={() =>
                          !isEditing && onSelect(conv.id, project.path, conv.sessionKey)
                        }
                      >
                        <span className="conv-row__icon">
                          {isStreaming ? (
                            <span className="conv-row__spinner" aria-label="Running" />
                          ) : isAttention ? (
                            <span
                              className="conv-row__notif"
                              role="img"
                              aria-label="Finished — open to view"
                            />
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
                            title="Archive"
                            onClick={(event) => {
                              event.stopPropagation();
                              onArchive(conv.id, project.path);
                            }}
                          >
                            <Icon icon="solar:archive-linear" width={13} height={13} />
                          </button>
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
                    </Fragment>
                  );
                })}
            </div>
          );
        })}
      </div>
    </div>
  );
}
