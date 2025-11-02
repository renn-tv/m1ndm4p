import re
from typing import List, Optional, Tuple

from models import MindmapNode


_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET_PATTERN = re.compile(r"^(\s*)-\s+(.*)$")


def to_markdown(root: MindmapNode) -> str:
    """Serialize a mind map to Markdown using heading levels."""
    if root is None:
        raise ValueError("root node must not be None")

    lines: List[str] = []

    def write_node(node: MindmapNode, level: int) -> None:
        heading = f"{'#' * max(level, 1)} {node.title}".rstrip()
        lines.append(heading)

        if node.body:
            lines.append("")
            lines.append(node.body)

        if node.children:
            lines.append("")
            for idx, child in enumerate(node.children):
                write_node(child, level + 1)
                if idx != len(node.children) - 1:
                    lines.append("")

    write_node(root, 1)

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines) + "\n"


def from_markdown(md: str) -> MindmapNode:
    """Parse Markdown headings and bullets into a mind map tree."""
    lines = md.splitlines()
    stack: List[Tuple[int, MindmapNode]] = []
    root: Optional[MindmapNode] = None
    i = 0

    while i < len(lines):
        line = lines[i]
        heading_match = _HEADING_PATTERN.match(line)
        if not heading_match:
            i += 1
            continue

        level = len(heading_match.group(1))
        title = heading_match.group(2).strip()
        node = MindmapNode(title=title)

        while stack and stack[-1][0] >= level:
            stack.pop()

        if stack:
            stack[-1][1].children.append(node)
        else:
            root = node

        stack.append((level, node))
        i += 1

        body_lines: List[str] = []
        while i < len(lines):
            current = lines[i]
            if current.strip() == "":
                if body_lines:
                    i += 1
                    break
                i += 1
                continue
            if _HEADING_PATTERN.match(current) or _BULLET_PATTERN.match(current):
                break
            body_lines.append(current)
            i += 1

        if body_lines:
            node.body = "\n".join(body_lines).strip()

        while i < len(lines) and lines[i].strip() == "":
            i += 1

        bullet_stack: List[Tuple[int, MindmapNode]] = []
        while i < len(lines):
            bullet_match = _BULLET_PATTERN.match(lines[i])
            if not bullet_match:
                break

            indent_str = bullet_match.group(1).replace("\t", "    ")
            indent = len(indent_str)
            bullet_title = bullet_match.group(2).strip()
            bullet_node = MindmapNode(title=bullet_title)

            while bullet_stack and indent <= bullet_stack[-1][0]:
                bullet_stack.pop()

            parent = bullet_stack[-1][1] if bullet_stack else node
            parent.children.append(bullet_node)
            bullet_stack.append((indent, bullet_node))
            i += 1

        while i < len(lines) and lines[i].strip() == "":
            i += 1

    if root is None:
        raise ValueError("No root heading found in Markdown")

    return root
