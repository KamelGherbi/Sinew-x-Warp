export const WORKSPACE_SESSION_SEPARATOR = "\u001f";

export function workspaceSessionKey(
  workspacePath: string,
  conversationId: string,
): string {
  return `${workspacePath}${WORKSPACE_SESSION_SEPARATOR}${conversationId}`;
}
