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

type Props = {
  text: string;
  onOpenFile: (path: string) => void;
};

type LinkifyOptions = {
  onOpenFile: (path: string) => void;
};

const FILE_TOKEN =
  /((?:\.{1,2}\/)?(?:[A-Za-z0-9_.+-]+\/)+[A-Za-z0-9_.+-]+\.[A-Za-z0-9]+(?::\d+(?::\d+)?)?|[A-Za-z0-9_.+-]+\.(?:tsx?|jsx?|rs|toml|json|md|css|scss|html|ya?ml|lock|sh|zsh|bash|py|go|java|kt|swift|sql|env|mjs|cjs|config)(?::\d+(?::\d+)?)?)/g;

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

function linkifyText(text: string, { onOpenFile }: LinkifyOptions): ReactNode[] {
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
      <FileLink key={`${value}-${index}`} path={value} onOpenFile={onOpenFile}>
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

function childrenToString(children: ReactNode): string {
  return Children.toArray(children)
    .map((child) => (typeof child === "string" ? child : ""))
    .join("");
}

function linkifyChildren(children: ReactNode, options: LinkifyOptions): ReactNode {
  return Children.map(children, (child) => {
    if (typeof child === "string") return linkifyText(child, options);
    if (!isValidElement(child)) return child;
    if (child.type === "a" || child.type === "code" || child.type === "pre") {
      return child;
    }

    const props = child.props as { children?: ReactNode };
    if (props.children === undefined) return child;

    return cloneElement(
      child as ReactElement<{ children?: ReactNode }>,
      undefined,
      linkifyChildren(props.children, options),
    );
  });
}

export function FileLinkedText({ text, onOpenFile }: Props) {
  return <>{linkifyText(text, { onOpenFile })}</>;
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
          p({ children }) {
            return <p>{linkifyChildren(children, { onOpenFile })}</p>;
          },
          li({ children }) {
            return <li>{linkifyChildren(children, { onOpenFile })}</li>;
          },
          h1({ children }) {
            return <h1>{linkifyChildren(children, { onOpenFile })}</h1>;
          },
          h2({ children }) {
            return <h2>{linkifyChildren(children, { onOpenFile })}</h2>;
          },
          h3({ children }) {
            return <h3>{linkifyChildren(children, { onOpenFile })}</h3>;
          },
          h4({ children }) {
            return <h4>{linkifyChildren(children, { onOpenFile })}</h4>;
          },
          code({ children, className }) {
            const value = childrenToString(children);
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
