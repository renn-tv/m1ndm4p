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
    """Serialize the current mind map to Markdown.

    Pure, faithful mapping:
    - Do NOT mutate or reinterpret the MindmapNode tree.
    - Use node.kind to decide heading vs list:
      - kind == "heading" (or None) -> Markdown heading (#..###### by depth)
      - kind == "list"             -> bullet using list_marker (default "*")
    - Depth in the tree controls heading level / indent only.
    """
    if root is None:
        raise ValueError("root node must not be None")

    lines: List[str] = []
    first_content_emitted = False

    def append_line(line: str) -> None:
        nonlocal first_content_emitted
        if line != "":
            first_content_emitted = True
        lines.append(line)

    def emit_h1_spacing() -> None:
        """Ensure a single blank line before H1 (except at top)."""
        if not first_content_emitted:
            return
        # Strip trailing blanks, then add exactly one.
        while lines and lines[-1] == "":
            lines.pop()
        append_line("")

    def emit_h2_separator() -> None:
        """Insert Level 2 separator: blank, '---', blank (except at top)."""
        if not first_content_emitted:
            return
        # Normalize trailing blanks.
        while lines and lines[-1] == "":
            lines.pop()
        append_line("")
        append_line("---")
        append_line("")

    def emit_single_blank_before_block() -> None:
        """Ensure exactly one blank line before a new block (L3 or similar)."""
        if not first_content_emitted:
            return
        # Strip trailing blanks, then add one if there is content before.
        while lines and lines[-1] == "":
            lines.pop()
        if lines:
            append_line("")

    def emit_body(body: str, indent: int = 0) -> None:
        """Emit body text exactly as provided, with optional indent."""
        if not body:
            return
        prefix = " " * indent
        for raw in body.splitlines():
            if raw.strip() == "":
                append_line("")
            else:
                append_line(f"{prefix}{raw}")

    def write_heading(node: MindmapNode, depth: int) -> None:
        """Emit node as a heading based on its depth."""
        level = min(depth + 1, 6)

        if level == 1:
            emit_h1_spacing()
        elif level == 2:
            emit_h2_separator()
        elif level == 3:
            emit_single_blank_before_block()

        heading_line = f"{'#' * level} {node.title}".rstrip()
        append_line(heading_line)

        if node.body:
            emit_body(node.body, indent=0)

    def write_list_item(node: MindmapNode, depth: int) -> None:
        """Emit node as a bullet based on its depth and list_marker.

        Indent rule:
        - Exactly two spaces before '*' (or ordered marker), regardless of depth.
        - This keeps output stable and avoids compounding spaces.
        """
        marker = node.list_marker or "*"
        indent = "  "
        title = node.title or ""

        if marker and marker[0].isdigit():
            # Ordered marker like "1." or "2)"
            line = f"{marker} {title}".rstrip()
        else:
            # Normalize unordered bullets to "*"
            line = f"* {title}".rstrip()

        append_line(f"{indent}{line}")

        if node.body:
            emit_body(node.body, indent=len(indent))

    def write_children(parent: MindmapNode, depth: int) -> None:
        """Emit children of parent with per-level rules:

        - Let children = direct MindmapNode children of parent.
        - Compute:
          - internal = any child with children
        - If internal is True:
            - All children at this level are headings (no bullets here).
            - For each child:
                - heading at this depth
                - recurse into its children at depth + 1
        - If internal is False (all leaves):
            - All children at this level are bullets ("deepest level" for this branch).
        """
        children = list(parent.children)
        if not children:
            return

        has_internal = any(bool(child.children) for child in children)

        if has_internal:
            # Mixed or all internal: treat all as headings at this level.
            for child in children:
                write_heading(child, depth)
                if child.children:
                    write_children(child, depth + 1)
        else:
            # All leaves: emit as bullets at this level.
            for child in children:
                write_list_item(child, depth)

    # Root: always start as heading, then apply per-parent leaf bulletization.
    write_heading(root, 0)
    write_children(root, 1)

    # Final cleanup: no trailing blank lines.
    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines) + "\n"


def _match_list_item(line: str):
    """Return (indent, marker, text) for a list item, or None.

    For ordered items, `marker` is the full prefix (e.g. "1." or "2)"),
    and `text` is the content WITHOUT that prefix. This prevents
    duplicating markers on round-trip.
    """
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
        marker = f"{number}{sep}"
        text = ordered_match.group(4).rstrip()
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

            # Normalize marker style for new / generated maps:
            # - Always use "*" for bullets in the model.
            # - Preserve ordered markers (e.g. "1.", "1)") as-is.
            if marker and not marker[0].isdigit():
                normalized_marker = "*"
            else:
                normalized_marker = marker

            list_node = MindmapNode(
                title=text,
                list_marker=normalized_marker,
                kind="list",
            )

            # Maintain nesting by indent:
            # - Indent in Markdown is spaces; treat each 4 spaces (or a tab) as one nesting level.
            # - This results in visual indentation via "  " per depth on output.
            while list_stack and indent <= list_stack[-1][0]:
                list_stack.pop()

            parent = list_stack[-1][1] if list_stack else node
            parent.children.append(list_node)
            list_stack.append((indent, list_node))
            i += 1

            # Capture indented body lines that belong to this list item.
            body_lines: List[str] = []
            while i < len(lines):
                current = lines[i]
                # Stop if we hit another list item or a heading.
                if _match_list_item(current) or _HEADING_PATTERN.match(current):
                    break
                # Stop if indentation decreases below the bullet's indent.
                if len(current) - len(current.lstrip(" ")) < indent:
                    break
                if current.strip() == "":
                    if body_lines:
                        i += 1
                        break
                    i += 1
                    continue
                body_lines.append(_strip_body_line_prefix(current))
                i += 1

            if body_lines:
                list_node.body = "\n".join(body_lines).strip()

        while i < len(lines) and lines[i].strip() == "":
            i += 1

    if root is None:
        raise ValueError("No root heading found in Markdown")

    return root
