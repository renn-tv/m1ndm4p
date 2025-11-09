import re
from typing import List, Optional, Tuple

from node_models import MindmapNode


_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET_PATTERN = re.compile(r"^(\s*)([-+*])\s+(.*)$")
_ORDERED_PATTERN = re.compile(r"^(\s*)(\d+)([.\)])\s+(.*)$")


def _format_body_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("* "):
        return stripped
    return f"* {stripped}"


def _strip_body_line_prefix(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith("* "):
        return stripped[2:].lstrip()
    return line


def to_markdown(root: MindmapNode) -> str:
    """Serialize a mind map to Markdown.

    Rules:
    - Heading nodes (kind="heading") are rendered as Markdown headings.
    - List nodes (kind="list") are rendered as list items (bullets / ordered),
      even when they have children, so nested lists round-trip.
    - Body text is emitted as paragraphs under the owning heading or list item.
    """
    if root is None:
        raise ValueError("root node must not be None")

    lines: List[str] = []

    def emit_body(body: str, indent: int = 0) -> None:
        if not body:
            return
        prefix = " " * indent
        for idx, line in enumerate(body.splitlines()):
            # Preserve blank lines inside body
            if line.strip() == "":
                lines.append("")
            else:
                lines.append(f"{prefix}{line}")
        lines.append("")

    def write_node(node: MindmapNode, depth: int) -> None:
        """Emit a node and its children, respecting kind for round-tripping.

        - kind="heading": render as Markdown heading.
        - kind="list": render as list item (with original marker when possible),
          even when the node has children (nested lists).
        """
        indent = "  " * depth

        if node.kind == "list":
            # Render as list item (unordered or ordered) and keep nested children as nested list.
            marker = node.list_marker or "-"
            if marker and marker[0].isdigit():
                # Ordered (e.g. "1." or "2)")
                prefix = f"{marker} "
                if node.title.startswith(prefix):
                    line = node.title.rstrip()
                else:
                    line = f"{marker} {node.title}".rstrip()
            else:
                # Unordered: don't duplicate marker in title
                line = f"- {node.title}".rstrip()
            lines.append(f"{indent}{line}")

            # Body under list item is indented one level
            if node.body:
                emit_body(node.body, indent=len(indent) + 2)

            # Children as nested list items (no extra blank line between list item and its nested list)
            for child in node.children:
                write_node(child, depth + 1)

        else:
            # Default: heading-style node.
            # Clamp heading level between 1 and 6 (depth 0 = #)
            level = min(depth + 1, 6)
            lines.append(f"{'#' * level} {node.title}".rstrip())
            lines.append("")

            if node.body:
                emit_body(node.body, indent=0)

            # Children follow as usual
            for child in node.children:
                write_node(child, depth + 1)

    # Root is treated as depth 0 heading
    write_node(root, 0)

    # Trim trailing blank lines
    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines) + "\n"


def _match_list_item(line: str):
    """Return (indent, marker, text) for a list item, or None."""
    bullet_match = _BULLET_PATTERN.match(line)
    if bullet_match:
        indent_str = bullet_match.group(1).replace("\t", "    ")
        indent = len(indent_str)
        marker = bullet_match.group(2)
        text = bullet_match.group(3).rstrip()
        return indent, marker, text

    ordered_match = _ORDERED_PATTERN.match(line)
    if ordered_match:
        indent_str = ordered_match.group(1).replace("\t", "    ")
        indent = len(indent_str)
        number = ordered_match.group(2)
        sep = ordered_match.group(3)
        text = ordered_match.group(4).rstrip()
        marker = f"{number}{sep}"
        return indent, marker, text

    return None


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
        title = heading_match.group(2).rstrip()
        node = MindmapNode(title=title, kind="heading")

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
            # Stop body when we hit a heading or a list item
            if _HEADING_PATTERN.match(current) or _match_list_item(current):
                break
            body_lines.append(_strip_body_line_prefix(current))
            i += 1

        if body_lines:
            node.body = "\n".join(body_lines).strip()

        while i < len(lines) and lines[i].strip() == "":
            i += 1

        # Parse list items (unordered/ordered) as children
        list_stack: List[Tuple[int, MindmapNode]] = []
        while i < len(lines):
            match = _match_list_item(lines[i])
            if not match:
                break

            indent, marker, text = match

            # For ordered lists, keep the numeric marker visible (e.g. "1)", "2.").
            # For plain bullets (-, *, +), do NOT show the marker in the title.
            if marker and marker[0].isdigit():
                list_node = MindmapNode(
                    title=f"{marker} {text}",
                    list_marker=marker,
                    kind="list",
                )
            else:
                list_node = MindmapNode(
                    title=text,
                    list_marker=marker,
                    kind="list",
                )

            # Maintain nesting by indent
            while list_stack and indent <= list_stack[-1][0]:
                list_stack.pop()

            parent = list_stack[-1][1] if list_stack else node
            parent.children.append(list_node)
            list_stack.append((indent, list_node))
            i += 1

        while i < len(lines) and lines[i].strip() == "":
            i += 1

    if root is None:
        raise ValueError("No root heading found in Markdown")

    return root
