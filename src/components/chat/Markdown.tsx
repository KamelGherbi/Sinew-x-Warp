import {
  Children,
  cloneElement,
  isValidElement,
  memo,
  type ReactElement,
  type ReactNode,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { api } from "../../lib/ipc";
import { MermaidDiagram } from "./MermaidDiagram";

type Props = {
  text: string;
  onOpenFile: (path: string) => void;
};

type LinkifyOptions = {
  onOpenFile: (path: string) => void;
};

type ColorName =
  | "red"
  | "orange"
  | "yellow"
  | "green"
  | "blue"
  | "purple"
  | "pink"
  | "gray"
  | "danger"
  | "warning"
  | "success"
  | "info"
  | "accent"
  | "muted";

const COLOR_ALIASES: Record<string, ColorName> = {
  accent: "accent",
  amber: "orange",
  blue: "blue",
  danger: "danger",
  error: "danger",
  gray: "gray",
  green: "green",
  grey: "gray",
  important: "danger",
  info: "info",
  muted: "muted",
  ok: "success",
  orange: "orange",
  pink: "pink",
  purple: "purple",
  red: "red",
  rose: "red",
  success: "success",
  violet: "purple",
  warn: "warning",
  warning: "warning",
  yellow: "yellow",
};

const FILE_TOKEN =
  /((?:\.{1,2}\/)?(?:[A-Za-z0-9_.+-]+\/)+[A-Za-z0-9_.+-]+\.[A-Za-z0-9]+(?::\d+(?::\d+)?)?|[A-Za-z0-9_.+-]+\.(?:tsx?|jsx?|rs|toml|json|md|css|scss|html|ya?ml|lock|sh|zsh|bash|py|go|java|kt|swift|sql|env|mjs|cjs|config)(?::\d+(?::\d+)?)?)/g;

function isMermaidLanguage(className?: string): boolean {
  if (!className) return false;
  return className
    .split(/\s+/)
    .some((name) => name === "mermaid" || name === "language-mermaid");
}

function isFileToken(value: string): boolean {
  FILE_TOKEN.lastIndex = 0;
  const match = FILE_TOKEN.exec(value.trim());
  return match?.[0] === value.trim();
}

function FileLink({
  path,
  children,
  variant,
  onOpenFile,
}: {
  path: string;
  children: ReactNode;
  variant?: "code";
  onOpenFile: (path: string) => void;
}) {
  return (
    <button
      type="button"
      className="chat-file-link"
      data-variant={variant}
      title="Open file"
      onClick={(event) => {
        event.stopPropagation();
        onOpenFile(path);
      }}
    >
      {children}
    </button>
  );
}

function linkifyText(
  text: string,
  { onOpenFile }: LinkifyOptions,
  keyPrefix = "file",
): ReactNode[] {
  const nodes: ReactNode[] = [];
  let lastIndex = 0;

  FILE_TOKEN.lastIndex = 0;
  for (const match of text.matchAll(FILE_TOKEN)) {
    const index = match.index ?? 0;
    const value = match[0];
    if (index > lastIndex) {
      nodes.push(text.slice(lastIndex, index));
    }
    nodes.push(
      <FileLink
        key={`${keyPrefix}-${value}-${index}`}
        path={value}
        onOpenFile={onOpenFile}
      >
        {value}
      </FileLink>,
    );
    lastIndex = index + value.length;
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }

  return nodes;
}

function normalizeColorName(value: string): ColorName | null {
  return COLOR_ALIASES[value.toLowerCase()] ?? null;
}

function readColorStart(
  text: string,
  index: number,
): { color: ColorName; contentStart: number } | null {
  if (text[index] !== ":" || text[index + 1] !== ":" || text[index - 1] === "\\") {
    return null;
  }

  let cursor = index + 2;
  const colorStart = cursor;
  while (cursor < text.length && /[A-Za-z-]/.test(text[cursor])) {
    cursor += 1;
  }

  if (cursor === colorStart || text[cursor] !== "[") return null;
  const color = normalizeColorName(text.slice(colorStart, cursor));
  if (!color) return null;

  return { color, contentStart: cursor + 1 };
}

function findClosingColorBracket(text: string, openBracketIndex: number): number {
  let depth = 1;
  for (let cursor = openBracketIndex + 1; cursor < text.length; cursor += 1) {
    const char = text[cursor];
    if (char === "\\" && cursor + 1 < text.length) {
      cursor += 1;
      continue;
    }
    if (char === "[") {
      depth += 1;
    } else if (char === "]") {
      depth -= 1;
      if (depth === 0) return cursor;
    }
  }
  return -1;
}

function renderInlineText(
  text: string,
  options: LinkifyOptions,
  keyPrefix = "text",
): ReactNode[] {
  const nodes: ReactNode[] = [];
  let cursor = 0;
  let plainStart = 0;

  while (cursor < text.length) {
    const colorStart = readColorStart(text, cursor);
    if (!colorStart) {
      cursor += 1;
      continue;
    }

    const openBracketIndex = colorStart.contentStart - 1;
    const closeBracketIndex = findClosingColorBracket(text, openBracketIndex);
    if (closeBracketIndex < 0) {
      cursor += 1;
      continue;
    }

    if (plainStart < cursor) {
      nodes.push(
        ...linkifyText(
          text.slice(plainStart, cursor),
          options,
          `${keyPrefix}-plain-${plainStart}`,
        ),
      );
    }

    nodes.push(
      <span
        key={`${keyPrefix}-color-${cursor}`}
        className="md-color"
        data-color={colorStart.color}
      >
        {renderInlineText(
          text.slice(colorStart.contentStart, closeBracketIndex),
          options,
          `${keyPrefix}-color-${cursor}`,
        )}
      </span>,
    );

    cursor = closeBracketIndex + 1;
    plainStart = cursor;
  }

  if (plainStart < text.length) {
    nodes.push(
      ...linkifyText(
        text.slice(plainStart),
        options,
        `${keyPrefix}-plain-${plainStart}`,
      ),
    );
  }

  return nodes;
}

function childrenToString(children: ReactNode): string {
  return Children.toArray(children)
    .map((child) => {
      if (typeof child === "string") return child;
      if (typeof child === "number") return String(child);
      if (!isValidElement(child)) return "";
      const props = child.props as { children?: ReactNode };
      return childrenToString(props.children);
    })
    .join("");
}

function childText(child: ReactNode): string | null {
  if (typeof child === "string") return child;
  if (typeof child === "number") return String(child);
  return null;
}

function renderInlineNode(
  child: ReactNode,
  options: LinkifyOptions,
  keyPrefix: string,
): ReactNode {
  const text = childText(child);
  if (text !== null) {
    return renderInlineText(text, options, keyPrefix);
  }
  if (!isValidElement(child)) return child;
  if (child.type === "a" || child.type === "code" || child.type === "pre") {
    return child;
  }

  const props = child.props as { children?: ReactNode };
  if (props.children === undefined) return child;

  return cloneElement(
    child as ReactElement<{ children?: ReactNode }>,
    undefined,
    renderInlineChildren(props.children, options, keyPrefix),
  );
}

type ColorStart = { index: number; color: ColorName; contentStart: number };

function findNextColorStart(text: string, from: number): ColorStart | null {
  for (let index = from; index < text.length; index += 1) {
    const start = readColorStart(text, index);
    if (start) return { index, ...start };
  }
  return null;
}

function collectColorContent(
  children: ReactNode[],
  startChildIndex: number,
  startOffset: number,
): { endChildIndex: number; endOffset: number; content: ReactNode[] } | null {
  const content: ReactNode[] = [];
  let depth = 1;

  for (let index = startChildIndex; index < children.length; index += 1) {
    const child = children[index];
    const text = childText(child);
    if (text === null) {
      content.push(child);
      continue;
    }

    const segmentStart = index === startChildIndex ? startOffset : 0;
    let pendingStart = segmentStart;
    for (let cursor = segmentStart; cursor < text.length; cursor += 1) {
      const char = text[cursor];
      if (char === "\\" && cursor + 1 < text.length) {
        cursor += 1;
        continue;
      }
      if (char === "[") {
        depth += 1;
      } else if (char === "]") {
        depth -= 1;
        if (depth === 0) {
          if (pendingStart < cursor) {
            content.push(text.slice(pendingStart, cursor));
          }
          return { endChildIndex: index, endOffset: cursor, content };
        }
      }
    }

    if (pendingStart < text.length) {
      content.push(text.slice(pendingStart));
    }
  }

  return null;
}

function renderInlineChildren(
  children: ReactNode,
  options: LinkifyOptions,
  keyPrefix = "child",
): ReactNode {
  const childArray = Children.toArray(children);
  const nodes: ReactNode[] = [];
  let childIndex = 0;
  let textOffset = 0;

  while (childIndex < childArray.length) {
    const child = childArray[childIndex];
    const text = childText(child);
    if (text === null) {
      nodes.push(renderInlineNode(child, options, `${keyPrefix}-${childIndex}`));
      childIndex += 1;
      textOffset = 0;
      continue;
    }

    const colorStart = findNextColorStart(text, textOffset);
    if (!colorStart) {
      nodes.push(
        ...linkifyText(
          text.slice(textOffset),
          options,
          `${keyPrefix}-${childIndex}-plain-${textOffset}`,
        ),
      );
      childIndex += 1;
      textOffset = 0;
      continue;
    }

    if (textOffset < colorStart.index) {
      nodes.push(
        ...linkifyText(
          text.slice(textOffset, colorStart.index),
          options,
          `${keyPrefix}-${childIndex}-plain-${textOffset}`,
        ),
      );
    }

    const collected = collectColorContent(
      childArray,
      childIndex,
      colorStart.contentStart,
    );
    if (!collected) {
      nodes.push(
        ...linkifyText(
          text.slice(colorStart.index, colorStart.contentStart),
          options,
          `${keyPrefix}-${childIndex}-plain-${colorStart.index}`,
        ),
      );
      textOffset = colorStart.contentStart;
      continue;
    }

    nodes.push(
      <span
        key={`${keyPrefix}-${childIndex}-color-${colorStart.index}`}
        className="md-color"
        data-color={colorStart.color}
      >
        {renderInlineChildren(
          collected.content,
          options,
          `${keyPrefix}-${childIndex}-color-${colorStart.index}`,
        )}
      </span>,
    );

    childIndex = collected.endChildIndex;
    textOffset = collected.endOffset + 1;
    const endText = childText(childArray[childIndex]) ?? "";
    if (textOffset >= endText.length) {
      childIndex += 1;
      textOffset = 0;
    }
  }

  return nodes;
}

export function FileLinkedText({ text, onOpenFile }: Props) {
  return <>{renderInlineText(text, { onOpenFile }, "file-linked")}</>;
}

/**
 * Markdown renderer tuned for chat. GFM + highlight.js. Memoized so
 * streaming token-by-token re-renders don't trash the whole tree.
 */
export const Markdown = memo(function Markdown({ text, onOpenFile }: Props) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
        components={{
          pre({ children }) {
            // `code` already swaps fenced ```mermaid blocks for an
            // interactive <MermaidDiagram />. Unwrap the surrounding
            // <pre> in that case so the diagram isn't boxed by code
            // styling; otherwise render a plain <pre>.
            const only =
              Children.count(children) === 1
                ? Children.toArray(children)[0]
                : null;
            if (isValidElement(only) && only.type === MermaidDiagram) {
              return only;
            }
            if (isValidElement<{ className?: string; children?: ReactNode }>(only)) {
              if (isMermaidLanguage(only.props.className)) {
                return (
                  <MermaidDiagram source={childrenToString(only.props.children)} />
                );
              }
            }
            return <pre>{children}</pre>;
          },
          p({ children }) {
            return <p>{renderInlineChildren(children, { onOpenFile })}</p>;
          },
          li({ children }) {
            return <li>{renderInlineChildren(children, { onOpenFile })}</li>;
          },
          h1({ children }) {
            return <h1>{renderInlineChildren(children, { onOpenFile })}</h1>;
          },
          h2({ children }) {
            return <h2>{renderInlineChildren(children, { onOpenFile })}</h2>;
          },
          h3({ children }) {
            return <h3>{renderInlineChildren(children, { onOpenFile })}</h3>;
          },
          h4({ children }) {
            return <h4>{renderInlineChildren(children, { onOpenFile })}</h4>;
          },
          code({ children, className }) {
            const value = childrenToString(children);
            if (isMermaidLanguage(className)) {
              return <MermaidDiagram source={value} />;
            }
            if (!className && isFileToken(value)) {
              return (
                <FileLink
                  path={value}
                  variant="code"
                  onOpenFile={onOpenFile}
                >
                  {value}
                </FileLink>
              );
            }
            return <code className={className}>{children}</code>;
          },
          table({ children }) {
            // Wrap the table so horizontal scrolling lives on a block-level
            // ancestor. The <table> itself stays a real display: table so
            // the auto column-width algorithm distributes space correctly
            // (otherwise display: block on the table collapses left columns).
            return (
              <div className="md-table-wrap">
                <table>{children}</table>
              </div>
            );
          },
          a({ href, children }) {
            const value = href ?? childrenToString(children);
            if (isFileToken(value)) {
              return (
                <FileLink path={value} onOpenFile={onOpenFile}>
                  {children}
                </FileLink>
              );
            }
            return (
              <a
                href={href}
                target="_blank"
                rel="noreferrer"
                onClick={(event) => {
                  if (!href) return;
                  event.preventDefault();
                  event.stopPropagation();
                  void api.openExternalUrl(href);
                }}
              >
                {children}
              </a>
            );
          },
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
});
