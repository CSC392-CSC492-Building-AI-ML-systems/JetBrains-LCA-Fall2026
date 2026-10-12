import { useState } from "react";
import type { TreeNode } from "../api";

function Node({ node, depth, onSelect, selected }: {
  node: TreeNode;
  depth: number;
  onSelect: (node: TreeNode) => void;
  selected?: string;
}) {
  const [open, setOpen] = useState(node.name === "configs");
  const isDir = node.type === "dir";
  const hasChildren = isDir && (node.children?.length ?? 0) > 0;
  return (
    <li>
      <button
        type="button"
        className={`tree-item${selected === node.path ? " selected" : ""}${isDir ? "" : " file"}`}
        style={{ paddingLeft: 4 + depth * 14 }}
        onClick={() => (isDir ? setOpen((o) => !o) : onSelect(node))}
        aria-expanded={isDir ? open : undefined}
      >
        {isDir ? (hasChildren && open ? "▾ " : "▸ ") : ""}
        {node.name}
      </button>
      {isDir && open && hasChildren && (
        <ul>
          {node.children!.map((c) => (
            <Node key={c.path} node={c} depth={depth + 1} onSelect={onSelect} selected={selected} />
          ))}
        </ul>
      )}
    </li>
  );
}

export function ProjectTree({ nodes, onSelect, selected }: {
  nodes: TreeNode[];
  onSelect: (node: TreeNode) => void;
  selected?: string;
}) {
  return (
    <ul className="tree">
      {nodes.map((n) => (
        <Node key={n.path} node={n} depth={0} onSelect={onSelect} selected={selected} />
      ))}
    </ul>
  );
}
