from __future__ import annotations

from pathlib import Path
import asyncio
import textwrap
from typing import Optional

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Tree
from textual.widgets._tree import TextType
from rich.text import Text

from md_io import from_markdown, to_markdown
from models import MindmapNode
import ai


class MindmapTree(Tree[MindmapNode]):
    """Tree widget specialised for ``MindmapNode`` data."""

    def process_label(self, label: TextType) -> Text:
        if isinstance(label, str):
            return Text.from_markup(label, justify="left")
        return label


class MindmapApp(App[None]):
    """Textual user interface for the Markdown-backed mind map."""

    TITLE = "m1ndm4p"

    CSS = """
    #mindmap-tree {
        width: 1fr;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=False),
        Binding("s", "save", "Save", show=False),
        Binding("o", "open", "Open", show=False),
        Binding("left", "collapse_cursor", "Collapse", show=False),
        Binding("right", "expand_cursor", "Expand", show=False),
        Binding("up", "cursor_up", "Cursor Up", show=False),
        Binding("down", "cursor_down", "Cursor Down", show=False),
        Binding("0", "delete_node", "(del)"),
        Binding("t", "generate_body", "(text)"),
        Binding("e", "edit_node", "(edit)"),
        Binding("a", "expand_all", "Expand All"),
        Binding("1", "generate_children(1)", "(AI nodes)", key_display="1-9"),
        Binding("?", "auto_generate_children", "(AI auto)"),
    ] + [
        Binding(str(i), f"generate_children({i})", "(AI nodes)", show=False)
        for i in range(2, 10)
    ]

    def __init__(self) -> None:
        super().__init__()
        self.title = "m1ndm4p"
        self._tree_widget: Optional[MindmapTree] = None
        self.mindmap_root = MindmapNode("Central Idea")
        self._edit_state: Optional[dict[str, object]] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        tree = MindmapTree("Mind Map", id="mindmap-tree")
        tree.show_root = True
        self._tree_widget = tree
        yield tree
        yield Footer()

    def on_mount(self) -> None:
        self.rebuild_tree()
        self.show_status("Ready")

    def rebuild_tree(self) -> None:
        tree = self.require_tree()
        tree.clear()
        root_node = tree.root
        root_node.set_label(self._format_node_label(self.mindmap_root))
        root_node.data = self.mindmap_root
        self.populate_tree(root_node, self.mindmap_root)
        root_node.expand_all()
        tree.select_node(root_node)

    def require_tree(self) -> MindmapTree:
        if self._tree_widget is None:
            raise RuntimeError("Tree widget not initialised")
        return self._tree_widget

    def populate_tree(self, tree_node: Tree.Node[MindmapNode], mindmap_node: MindmapNode) -> None:
        tree_node.set_label(self._format_node_label(mindmap_node))
        tree_node.data = mindmap_node
        for child in list(tree_node.children):
            child.remove()
        for child in mindmap_node.children:
            child_tree_node = tree_node.add(self._format_node_label(child), data=child)
            self.populate_tree(child_tree_node, child)
        if mindmap_node.body:
            for index, line in enumerate(self._body_lines(mindmap_node.body)):
                display = f"  {line}" if line else "  "
                text_leaf = tree_node.add_leaf(Text(display, style="dim italic"))
                text_leaf.allow_expand = False
                text_leaf.data = {"kind": "body_line", "node": mindmap_node, "index": index}

    def on_tree_node_selected(self, event: Tree.NodeSelected[MindmapNode]) -> None:  # noqa: D401
        """React to selection changes with a status update."""

        node_data = event.node.data
        if node_data is None:
            return
        level = self._tree_node_level(event.node)
        if isinstance(node_data, dict) and node_data.get("kind") == "body_line":
            self.show_status(f"Selected (level {level}): text line")
        else:
            title = node_data.title if isinstance(node_data, MindmapNode) else str(node_data)
            self.show_status(f"Selected (level {level}): {title}")

    def get_selected_tree_node(self) -> Optional[Tree.Node[MindmapNode]]:
        tree = self.require_tree()
        return tree.cursor_node

    def get_selected_model_node(self) -> Optional[MindmapNode]:
        selected = self.get_selected_tree_node()
        return selected.data if selected else None

    def _format_node_label(self, node: MindmapNode) -> Text:
        return Text(node.title)

    @staticmethod
    def _body_lines(body: str) -> list[str]:
        lines: list[str] = []
        for block in body.splitlines():
            block = block.strip()
            if not block:
                lines.append("")
                continue
            wrapped = textwrap.wrap(block, width=40) or [""]
            lines.extend(wrapped)
        if not lines:
            lines.append("")
        return lines

    def _apply_generated_children(
        self,
        tree_node: Tree.Node[MindmapNode],
        model_node: MindmapNode,
        child_titles: list[str],
    ) -> int:
        tree = self.require_tree()
        model_node.children = []
        for child in list(tree_node.children):
            child.remove()
        child_level = self._tree_node_level(tree_node) + 1
        for index, title in enumerate(child_titles, start=1):
            clean_title = title.strip() or f"Level {child_level} Node {index}"
            new_model = MindmapNode(clean_title)
            model_node.children.append(new_model)
            new_tree_node = tree_node.add(self._format_node_label(new_model), data=new_model)
            new_tree_node.expand()
        tree_node.expand()
        tree.refresh(layout=True)
        return child_level

    def _start_inline_edit(
        self,
        tree_node: Tree.Node,
        initial_text: str,
        *,
        kind: str,
        context: dict[str, object],
    ) -> None:
        buffer = ""
        original_label = tree_node._label.copy()
        self._edit_state = {
            "tree_node": tree_node,
            "buffer": buffer,
            "kind": kind,
            "context": context,
            "original_label": original_label,
            "initial": initial_text,
            "tree_node_id": tree_node.id,
        }
        self._update_edit_label(select_all=True)
        self.show_status("Editing… Enter to save, Esc to cancel.")

    def _update_edit_label(self, *, select_all: bool = False) -> None:
        if self._edit_state is None:
            return
        tree_node = self._edit_state["tree_node"]
        buffer = self._edit_state["buffer"]
        kind = self._edit_state["kind"]
        initial = self._edit_state["initial"]
        caret = "▌"
        if kind == "body_line":
            if select_all and not buffer:
                highlight = Text(f"  {initial}", style="reverse dim italic") if initial else Text("  ", style="reverse dim italic")
                highlight.append(caret, style="dim italic")
                tree_node.set_label(highlight)
            else:
                display_text = buffer if buffer else initial
                display = f"  {display_text}" if display_text else "  "
                tree_node.set_label(Text(display + caret, style="dim italic"))
        else:
            if select_all and not buffer:
                highlight = Text(initial or "", style="reverse")
                highlight.append(caret)
                tree_node.set_label(highlight)
            else:
                tree_node.set_label(Text((buffer or initial) + caret))
        self.require_tree().refresh(layout=True)

    def _handle_edit_key(self, event: events.Key) -> None:
        if self._edit_state is None:
            return
        key = event.key
        if key == "escape":
            self._cancel_inline_edit()
            event.stop()
            return
        if key == "enter":
            self._commit_inline_edit()
            event.stop()
            return
        if key == "backspace":
            buffer = self._edit_state["buffer"]
            if buffer:
                self._edit_state["buffer"] = buffer[:-1]
                self._update_edit_label()
            event.stop()
            return
        character = event.character
        if event.is_printable and character and len(character) == 1:
            self._edit_state["buffer"] += character
            self._update_edit_label()
            event.stop()
            return
        # ignore other keys
        event.stop()

    def _cancel_inline_edit(self) -> None:
        if self._edit_state is None:
            return
        tree_node = self._edit_state["tree_node"]
        original_label = self._edit_state["original_label"]
        tree_node.set_label(original_label)
        self.require_tree().refresh(layout=True)
        self._edit_state = None
        self.show_status("Edit cancelled.")

    def _commit_inline_edit(self) -> None:
        if self._edit_state is None:
            return
        buffer = self._edit_state["buffer"].strip()
        initial = (self._edit_state.get("initial") or "").strip()
        kind = self._edit_state["kind"]
        tree_node = self._edit_state["tree_node"]
        context = self._edit_state["context"]

        if kind == "body_line":
            parent_node: MindmapNode = context["parent_node"]  # type: ignore[assignment]
            parent_tree_node_id = context["parent_tree_node_id"]  # type: ignore[assignment]
            line_index: int = context["line_index"]  # type: ignore[assignment]
            lines = self._body_lines(parent_node.body or "")
            while len(lines) <= line_index:
                lines.append("")
            new_line_value = buffer if buffer else initial
            lines[line_index] = new_line_value
            updated_body = "\n".join(lines).strip()
            parent_node.body = updated_body[:200]
            tree = self.require_tree()
            parent_tree_node = tree.get_node_by_id(parent_tree_node_id) or tree.root
            parent_tree_node.expand()
            tree.refresh(layout=True)
            # reselect appropriate line if possible
            new_lines = self._body_lines(parent_node.body or "")
            new_index = min(line_index, len(new_lines) - 1)
            selection_offset = len(parent_node.children) + new_index
            children_nodes = list(parent_tree_node.children)
            if 0 <= selection_offset < len(children_nodes):
                target_node = children_nodes[selection_offset]
                target_node.data = {
                    "kind": "body_line",
                    "node": parent_node,
                    "index": new_index,
                }
                tree.select_node(target_node)
            else:
                tree.select_node(parent_tree_node)
            self.populate_tree(parent_tree_node, parent_node)
            self.show_status("Text updated.")
        else:
            node: MindmapNode = context["node"]  # type: ignore[assignment]
            new_title = buffer or initial or node.title
            node.title = new_title
            tree_node.set_label(self._format_node_label(node))
            self.require_tree().refresh(layout=True)
            self.show_status("Node title updated.")

        self._edit_state = None

    def _tree_node_level(self, tree_node: Tree.Node[MindmapNode]) -> int:
        level = 0
        current = tree_node
        while current.parent is not None:
            level += 1
            current = current.parent
        return level

    async def action_generate_children(self, count: int) -> None:
        selected_tree_node = self.get_selected_tree_node()
        if selected_tree_node is None or selected_tree_node.data is None:
            self.bell()
            self.show_status("No node selected.")
            return
        if count <= 0:
            self.show_status("Count must be positive.")
            return
        model_node = selected_tree_node.data
        context = to_markdown(self.mindmap_root)
        tree = self.require_tree()
        spinner_frames = ["  .", "  ..", "  ..."]
        spinner_index = 0
        spinner_node = selected_tree_node.add_leaf(Text(spinner_frames[0], style="dim italic"))
        selected_tree_node.expand()
        tree.refresh(layout=True)

        def tick() -> None:
            nonlocal spinner_index
            spinner_index = (spinner_index + 1) % len(spinner_frames)
            spinner_node.set_label(Text(spinner_frames[spinner_index], style="dim italic"))
            tree.refresh(layout=False)

        spinner_timer = self.set_interval(0.25, tick)
        try:
            child_titles = await asyncio.to_thread(
                ai.generate_children, model_node.title, count, context_markdown=context
            )
        except Exception as exc:  # pragma: no cover - defensive programming
            self.handle_ai_error("generate child nodes", exc)
            child_titles = []
        finally:
            spinner_timer.stop()
            spinner_node.remove()
            tree.refresh(layout=True)
        if not child_titles:
            self.show_status("No child titles returned.")
            return
        child_level = self._apply_generated_children(selected_tree_node, model_node, child_titles)
        self.show_status(
            f"Generated {len(child_titles)} child{'ren' if len(child_titles) != 1 else ''} at level {child_level}."
        )

    async def action_auto_generate_children(self) -> None:
        selected_tree_node = self.get_selected_tree_node()
        if selected_tree_node is None or selected_tree_node.data is None:
            self.bell()
            self.show_status("No node selected.")
            return
        model_node = selected_tree_node.data
        context = to_markdown(self.mindmap_root)
        tree = self.require_tree()
        spinner_frames = ["  .", "  ..", "  ..."]
        spinner_index = 0
        spinner_node = selected_tree_node.add_leaf(Text(spinner_frames[0], style="dim italic"))
        selected_tree_node.expand()
        tree.refresh(layout=True)

        def tick() -> None:
            nonlocal spinner_index
            spinner_index = (spinner_index + 1) % len(spinner_frames)
            spinner_node.set_label(Text(spinner_frames[spinner_index], style="dim italic"))
            tree.refresh(layout=False)

        spinner_timer = self.set_interval(0.25, tick)
        try:
            child_titles = await asyncio.to_thread(
                ai.generate_children_auto, model_node.title, context_markdown=context
            )
        except Exception as exc:  # pragma: no cover - defensive programming
            self.handle_ai_error("generate child nodes", exc)
            child_titles = []
        finally:
            spinner_timer.stop()
            spinner_node.remove()
        if not child_titles:
            self.show_status("No child titles returned.")
            return
        child_level = self._apply_generated_children(selected_tree_node, model_node, child_titles)
        self.show_status(
            f"Generated {len(child_titles)} AI-selected child{'ren' if len(child_titles) != 1 else ''} at level {child_level}."
        )

    def action_expand_all(self) -> None:
        tree = self.require_tree()
        tree.root.expand_all()
        tree.refresh(layout=True)
        self.show_status("Expanded entire tree.")

    def action_edit_node(self) -> None:
        if self._edit_state is not None:
            return
        selected_tree_node = self.get_selected_tree_node()
        if selected_tree_node is None:
            self.bell()
            self.show_status("No node selected.")
            return
        data = selected_tree_node.data
        if isinstance(data, dict) and data.get("kind") == "body_line":
            parent_node = data["node"]
            line_index = data["index"]
            lines = self._body_lines(parent_node.body or "")
            current_value = lines[line_index] if line_index < len(lines) else ""
            parent_tree_node = selected_tree_node.parent or self.require_tree().root
            self._start_inline_edit(
                selected_tree_node,
                current_value,
                kind="body_line",
                context={
                    "parent_node": parent_node,
                    "parent_tree_node_id": parent_tree_node.id,
                    "line_index": line_index,
                },
            )
            return
        if not isinstance(data, MindmapNode):
            self.bell()
            self.show_status("Cannot edit this entry.")
            return
        self._start_inline_edit(
            selected_tree_node,
            data.title,
            kind="title",
            context={"node": data},
        )

    async def action_generate_body(self) -> None:
        selected_tree_node = self.get_selected_tree_node()
        if selected_tree_node is None or selected_tree_node.data is None:
            self.bell()
            self.show_status("No node selected.")
            return
        model_node = selected_tree_node.data
        context = to_markdown(self.mindmap_root)
        tree = self.require_tree()
        spinner_frames = ["  .", "  ..", "  ..."]
        spinner_index = 0
        spinner_node = selected_tree_node.add_leaf(Text(spinner_frames[0], style="dim italic"))
        selected_tree_node.expand()
        tree.refresh(layout=True)

        def tick() -> None:
            nonlocal spinner_index
            spinner_index = (spinner_index + 1) % len(spinner_frames)
            spinner_node.set_label(Text(spinner_frames[spinner_index], style="dim italic"))
            tree.refresh(layout=False)

        spinner_timer = self.set_interval(0.25, tick)
        try:
            paragraph = await asyncio.to_thread(
                ai.generate_paragraph, model_node.title, context_markdown=context
            )
        except Exception as exc:  # pragma: no cover - defensive programming
            self.handle_ai_error("generate paragraph", exc)
            return
        finally:
            spinner_timer.stop()
            spinner_node.remove()
            tree.refresh(layout=True)
        paragraph = paragraph.strip()
        if not paragraph:
            self.show_status("No text returned.")
            return
        model_node.body = paragraph
        line_count = len(self._body_lines(model_node.body))
        self.populate_tree(selected_tree_node, model_node)
        selected_tree_node.expand()
        self.require_tree().refresh(layout=True)
        self.show_status(
            f"Stored {line_count} text line{'s' if line_count != 1 else ''} for {model_node.title}."
        )

    async def on_event(self, event: events.Event) -> None:  # noqa: D401
        if isinstance(event, events.Key) and self._edit_state is not None:
            self._handle_edit_key(event)
            return
        await super().on_event(event)

    def action_delete_node(self) -> None:
        selected_tree_node = self.get_selected_tree_node()
        if selected_tree_node is None or selected_tree_node.data is None:
            self.bell()
            self.show_status("No node selected.")
            return
        tree = self.require_tree()
        node_data = selected_tree_node.data
        if isinstance(node_data, dict) and node_data.get("kind") == "body_line":
            parent_node = node_data["node"]
            line_index = node_data["index"]
            lines = self._body_lines(parent_node.body or "")
            if 0 <= line_index < len(lines):
                del lines[line_index]
                parent_node.body = "\n".join(lines).strip()
            parent_tree_node = selected_tree_node.parent or tree.root
            self.populate_tree(parent_tree_node, parent_node)
            parent_tree_node.expand()
            tree.refresh(layout=True)
            children_nodes = list(parent_tree_node.children)
            text_child_index = len(parent_node.children) + min(line_index, len(lines) - 1)
            if 0 <= text_child_index < len(children_nodes):
                tree.select_node(children_nodes[text_child_index])
            else:
                tree.select_node(parent_tree_node)
            self.show_status("Text line deleted.")
            return
        if isinstance(node_data, MindmapNode):
            node_data.body = None
            node_data.children = []
            for child in list(selected_tree_node.children):
                child.remove()
            selected_tree_node.set_label(self._format_node_label(node_data))
            tree.refresh(layout=True)
            tree.select_node(selected_tree_node)
            self.show_status("Node cleared.")
            return
        self.bell()
        self.show_status("Nothing deleted.")

    def action_save(self) -> None:
        markdown = to_markdown(self.mindmap_root)
        Path("mindmap.md").write_text(markdown, encoding="utf-8")
        self.show_status("Saved to mindmap.md")

    def action_open(self) -> None:
        path = Path("mindmap.md")
        if not path.exists():
            self.bell()
            self.show_status("mindmap.md not found.")
            return
        try:
            markdown = path.read_text(encoding="utf-8")
            self.mindmap_root = from_markdown(markdown)
        except Exception as exc:  # pragma: no cover - defensive programming
            self.bell()
            self.show_status(f"Failed to load mindmap.md: {exc}")
            return
        self.rebuild_tree()
        total, text_nodes = self._count_nodes(self.mindmap_root)
        self.show_status(
            f"Loaded mindmap.md ({total} nodes, {text_nodes} text line{'s' if text_nodes != 1 else ''})"
        )

    def action_collapse_cursor(self) -> None:
        node = self.get_selected_tree_node()
        if node:
            node.collapse()

    def action_expand_cursor(self) -> None:
        node = self.get_selected_tree_node()
        if node:
            node.expand()

    def action_cursor_up(self) -> None:
        self.require_tree().action_cursor_up()

    def action_cursor_down(self) -> None:
        self.require_tree().action_cursor_down()

    def show_status(self, message: str) -> None:
        self.sub_title = message

    def find_parent(self, root: MindmapNode, target: MindmapNode) -> Optional[MindmapNode]:
        for child in root.children:
            if child is target:
                return root
            parent = self.find_parent(child, target)
            if parent is not None:
                return parent
        return None

    def _count_nodes(self, node: MindmapNode) -> tuple[int, int]:
        total = 1
        text_lines = len(self._body_lines(node.body)) if node.body else 0
        for child in node.children:
            child_total, child_text = self._count_nodes(child)
            total += child_total
            text_lines += child_text
        return total, text_lines


if __name__ == "__main__":
    MindmapApp().run()
