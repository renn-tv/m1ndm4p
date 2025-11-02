from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Static, Tree

import ai
from md_io import from_markdown, to_markdown
from models import MindmapNode


class MindmapTree(Tree[MindmapNode]):
    """Tree widget specialised for ``MindmapNode`` data."""


class MindmapApp(App[None]):
    """Textual user interface for the Markdown-backed mind map."""

    CSS = """
    #mindmap-tree {
        width: 1fr;
    }
    #body-panel {
        width: 1fr;
        border: heavy $boost;
        padding: 1 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "save", "Save"),
        ("o", "open", "Open"),
        ("0", "delete_node", "Delete Node"),
        ("t", "generate_body", "Generate Body"),
        ("left", "collapse_cursor", "Collapse"),
        ("right", "expand_cursor", "Expand"),
        ("up", "cursor_up", "Cursor Up"),
        ("down", "cursor_down", "Cursor Down"),
    ] + [
        (str(i), f"generate_children({i})", f"Add {i} Child{'ren' if i > 1 else ''}")
        for i in range(1, 10)
    ]

    def __init__(self) -> None:
        super().__init__()
        self.body_panel: Optional[Static] = None
        self.tree: Optional[MindmapTree] = None
        self.mindmap_root = MindmapNode("Central Idea")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            tree = MindmapTree("Mind Map", id="mindmap-tree", show_root=True)
            self.tree = tree
            yield tree
            body_panel = Static(id="body-panel")
            self.body_panel = body_panel
            yield body_panel
        yield Footer()

    def on_mount(self) -> None:
        self.rebuild_tree()
        self.show_status("Ready")

    def rebuild_tree(self) -> None:
        tree = self.require_tree()
        tree.clear()
        root_node = tree.set_root(self.mindmap_root.title, data=self.mindmap_root)
        self.populate_tree(root_node, self.mindmap_root)
        root_node.expand()
        tree.select_node(root_node.id)
        self.update_body_panel(self.mindmap_root)

    def require_tree(self) -> MindmapTree:
        if self.tree is None:
            raise RuntimeError("Tree widget not initialised")
        return self.tree

    def require_body_panel(self) -> Static:
        if self.body_panel is None:
            raise RuntimeError("Body panel not initialised")
        return self.body_panel

    def populate_tree(self, tree_node: Tree.Node[MindmapNode], mindmap_node: MindmapNode) -> None:
        tree_node.set_label(mindmap_node.title)
        tree_node.data = mindmap_node
        for child in list(tree_node.children):
            child.remove()
        for child in mindmap_node.children:
            child_tree_node = tree_node.add(child.title, data=child)
            if child.children:
                self.populate_tree(child_tree_node, child)

    def on_tree_node_selected(self, event: Tree.NodeSelected[MindmapNode]) -> None:  # noqa: D401
        """Update body panel when the tree selection changes."""

        node_data = event.node.data
        if node_data is None:
            return
        self.update_body_panel(node_data)

    def update_body_panel(self, node: MindmapNode) -> None:
        body_panel = self.require_body_panel()
        title_markup = f"[b]{node.title}[/b]"
        if node.body:
            body_panel.update(f"{title_markup}\n\n{node.body}")
        else:
            body_panel.update(f"{title_markup}\n\n[dim]No body text. Press 't' to generate.[/dim]")

    def get_selected_tree_node(self) -> Optional[Tree.Node[MindmapNode]]:
        tree = self.require_tree()
        return tree.cursor_node

    def get_selected_model_node(self) -> Optional[MindmapNode]:
        selected = self.get_selected_tree_node()
        return selected.data if selected else None

    def action_generate_children(self, count: int) -> None:
        selected_tree_node = self.get_selected_tree_node()
        if selected_tree_node is None or selected_tree_node.data is None:
            self.bell()
            self.show_status("No node selected.")
            return
        model_node = selected_tree_node.data
        context = self.safe_to_markdown()
        try:
            child_titles = ai.generate_children(model_node.title, count, context_markdown=context)
        except Exception as exc:  # pragma: no cover - defensive programming
            self.handle_ai_error("generate child nodes", exc)
            child_titles = []
        if not child_titles:
            self.show_status("No child titles returned.")
            return
        for title in child_titles:
            clean_title = title.strip() or "Untitled"
            new_model = MindmapNode(clean_title)
            model_node.children.append(new_model)
            new_tree_node = selected_tree_node.add(clean_title, data=new_model)
            new_tree_node.expand()
        selected_tree_node.expand()
        self.update_body_panel(model_node)
        self.show_status(f"Added {len(child_titles)} child{'ren' if len(child_titles) != 1 else ''}.")

    def action_generate_body(self) -> None:
        selected_tree_node = self.get_selected_tree_node()
        if selected_tree_node is None or selected_tree_node.data is None:
            self.bell()
            self.show_status("No node selected.")
            return
        model_node = selected_tree_node.data
        context = self.safe_to_markdown()
        try:
            paragraph = ai.generate_paragraph(model_node.title, context_markdown=context)
        except Exception as exc:  # pragma: no cover - defensive programming
            self.handle_ai_error("generate paragraph", exc)
            return
        model_node.body = paragraph.strip()
        self.update_body_panel(model_node)
        self.show_status("Body updated.")

    def action_delete_node(self) -> None:
        selected_tree_node = self.get_selected_tree_node()
        if selected_tree_node is None or selected_tree_node.data is None:
            self.bell()
            self.show_status("No node selected.")
            return
        if selected_tree_node.data is self.mindmap_root:
            self.bell()
            self.show_status("Cannot delete the root node.")
            return
        parent_model = self.find_parent(self.mindmap_root, selected_tree_node.data)
        if parent_model is None:
            self.bell()
            self.show_status("Parent not found; nothing deleted.")
            return
        parent_model.children = [child for child in parent_model.children if child is not selected_tree_node.data]
        selected_tree_node.remove()
        self.show_status("Node deleted.")
        self.update_body_panel(parent_model)

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
        self.show_status("Loaded mindmap.md")

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

    def handle_ai_error(self, action: str, exc: Exception) -> None:
        self.bell()
        self.show_status(f"Could not {action}: {exc}")

    def find_parent(self, root: MindmapNode, target: MindmapNode) -> Optional[MindmapNode]:
        for child in root.children:
            if child is target:
                return root
            parent = self.find_parent(child, target)
            if parent is not None:
                return parent
        return None

    def safe_to_markdown(self) -> str:
        with contextlib.suppress(Exception):
            return to_markdown(self.mindmap_root)
        return ""


if __name__ == "__main__":
    MindmapApp().run()
