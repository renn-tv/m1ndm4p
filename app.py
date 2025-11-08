from __future__ import annotations

import asyncio
from contextlib import contextmanager
import html
from html.parser import HTMLParser
from pathlib import Path
import re
try:
    import regex as _regex
except ImportError:  # pragma: no cover
    _regex = None
import subprocess
import sys
import textwrap
from typing import Awaitable, Callable, Literal, Optional
import webbrowser
from urllib import request
from urllib.parse import urlparse

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, OptionList, Static, Tree, TextArea
from textual.widgets.option_list import Option, OptionDoesNotExist
from textual.widgets._tree import TextType, TreeNode, UnknownNodeID
from rich.text import Text

from md_io import from_markdown, to_markdown
from node_models import MindmapNode
import ai


def _key_name_and_modifiers(key_value: str) -> tuple[str, set[str]]:
    parts = key_value.split("+")
    key_name = parts[-1].lower()
    modifiers = {part.lower() for part in parts[:-1] if part}
    return key_name, modifiers


if _regex:
    _EMOJI_PREFIX_RE = _regex.compile(r"^\s*(\X)\s*")

    def _extract_leading_emoji(text: str) -> tuple[str, str]:
        stripped = text.lstrip()
        if not stripped:
            return "", ""
        remainder = stripped
        parts: list[str] = []
        while True:
            match = _EMOJI_PREFIX_RE.match(remainder)
            if not match:
                break
            grapheme = match.group(1)
            if not any(0x2600 <= ord(ch) <= 0x1FAFF for ch in grapheme):
                break
            parts.append(grapheme)
            remainder = remainder[match.end() :].lstrip()
        prefix = "".join(parts)
        return (remainder if prefix else stripped), prefix
else:
    _EMOJI_PREFIX_RE = re.compile(
        r"^\s*(?:(?::[A-Za-z0-9_+\-]+:)|[\u2600-\u27BF\U0001F000-\U0001FFFF])\s*"
    )

    def _extract_leading_emoji(text: str) -> tuple[str, str]:
        stripped = text.lstrip()
        if not stripped:
            return "", ""
        match = _EMOJI_PREFIX_RE.match(stripped)
        if not match:
            return stripped, ""
        emoji = stripped[: match.end()].strip()
        remainder = stripped[match.end() :].lstrip()
        return remainder, emoji


class MindmapTree(Tree[MindmapNode]):
    """Tree widget specialised for ``MindmapNode`` data."""

    def process_label(self, label: TextType) -> Text:
        if isinstance(label, str):
            return Text.from_markup(label, justify="left")
        return label


class ModelSelectorScreen(ModalScreen[str | None]):
    """Modal dialog that lets the user pick an OpenRouter model."""

    DEFAULT_CSS = """
    ModelSelectorScreen {
        align: center middle;
    }

    #model-selector-panel {
        min-width: 50;
        max-width: 80;
        background: $panel;
        border: round $secondary;
        padding: 1 2 2 2;
        box-sizing: border-box;
    }

    #model-selector-title {
        content-align: center middle;
        text-style: bold;
        padding-bottom: 1;
    }

    #model-selector-list {
        border: none;
        background: $surface;
        padding: 0;
    }

    #model-selector-list > .option-list--option,
    #model-selector-list > .option-list--option-highlighted {
        padding: 0 2;
    }

    #model-selector-list > .option-list--option-highlighted {
        background: $accent;
        color: $text;
        text-style: bold;
    }

    #model-selector-list:focus {
        border: none;
        outline: none;
    }
    """

    def __init__(self, models: list[str], current_model: str) -> None:
        super().__init__()
        self._models = models
        self._current_model = current_model

    def compose(self) -> ComposeResult:
        with Vertical(id="model-selector-panel"):
            yield Static("Select OpenRouter model", id="model-selector-title")
            yield OptionList(
                *[Option(model, id=model) for model in self._models],
                id="model-selector-list",
            )

    def on_mount(self) -> None:
        option_list = self.query_one("#model-selector-list", OptionList)
        option_list.focus()
        try:
            option_list.highlighted = option_list.get_option_index(self._current_model)
        except OptionDoesNotExist:
            option_list.highlighted = 0 if option_list.option_count else None
        option_list.scroll_to_highlight()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        selected = event.option_id or str(event.option.prompt)
        self.dismiss(selected)

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            self.dismiss(None)


class ContextTextArea(TextArea):
    """Specialised TextArea that posts a submit message on Enter."""

    class Submitted(Message):
        def __init__(self, textarea: "ContextTextArea") -> None:
            super().__init__()
            self.textarea = textarea
            self.text = textarea.text

    async def on_event(self, event: events.Event) -> None:  # noqa: D401
        if isinstance(event, events.Key):
            key_name, modifiers = _key_name_and_modifiers(event.key)
            if key_name == "enter" and "shift" not in modifiers:
                event.stop()
                self.post_message(self.Submitted(self))
                return
        await super().on_event(event)


class ContextEditorScreen(ModalScreen[dict[str, str] | None]):
    """Modal dialog that lets the user edit lightweight external context."""

    DEFAULT_CSS = """
    ContextEditorScreen {
        align: center middle;
        background: transparent;
    }

    #context-editor-text {
        width: 90;
        height: 16;
        border: round $secondary;
        background: $surface 6%;
        padding: 0;
    }
    """

    def __init__(self, initial_text: str) -> None:
        super().__init__()
        self._initial_text = initial_text

    def compose(self) -> ComposeResult:
        yield ContextTextArea(
            text=self._initial_text,
            id="context-editor-text",
            placeholder="Paste or edit reference text to guide AI responses…",
            soft_wrap=True,
        )

    def on_mount(self) -> None:
        textarea = self.query_one("#context-editor-text", ContextTextArea)
        textarea.focus()
        textarea.action_select_all()

    def _gather_text(self) -> str:
        textarea = self.query_one("#context-editor-text", ContextTextArea)
        return textarea.text

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            self.dismiss(None)

    def on_context_text_area_submitted(self, message: ContextTextArea.Submitted) -> None:
        message.stop()
        self.dismiss({"action": "apply", "text": message.text})


class URLImportScreen(ModalScreen[dict[str, str] | None]):
    """Modal prompt for importing external context from a URL."""

    DEFAULT_CSS = """
    URLImportScreen {
        align: center middle;
        background: transparent;
    }

    #url-import-field {
        width: 60;
        border: round $secondary;
        background: $surface;
    }
    """

    def __init__(self, initial_url: str) -> None:
        super().__init__()
        self._initial_url = initial_url

    def compose(self) -> ComposeResult:
        yield Input(
            value=self._initial_url,
            placeholder="https://example.com/article",
            id="url-import-field",
        )

    def on_mount(self) -> None:
        self.query_one("#url-import-field", Input).focus()

    def on_key(self, event: events.Key) -> None:
        field = self.query_one("#url-import-field", Input)
        if event.key == "escape":
            event.stop()
            self.dismiss(None)
        elif event.key == "enter":
            event.stop()
            self.dismiss({"action": "fetch", "url": field.value})
class MindmapApp(App[None]):
    """Textual user interface for the Markdown-backed mind map."""

    TITLE = "m1ndm4p"

    CSS = """
    #mindmap-tree {
        width: 1fr;
    }
    #mindmap-tree .tree--cursor,
    #mindmap-tree:focus .tree--cursor {
        text-style: none;
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
        Binding("i", "edit_context", "Context"),
        Binding("l", "level_prompt", "Levels"),
        Binding("w", "import_web_context", "(web ctx)"),
        Binding("tab", "add_manual_child", "(manual +)", show=False, priority=True),
        Binding("a", "expand_all", "Expand All"),
        Binding("+", "add_child_suggestion", "(child +)"),
        Binding("-", "remove_child_suggestion", "(child -)"),
        Binding("f", "full_generate", "(full)"),
        Binding(":", "add_emoji", "Emoji"),
        Binding("m", "choose_model", "Model"),
        Binding("p", "preview_markmap", "Markmap"),
        Binding("1", "generate_children(1)", "(AI nodes)", key_display="1-9"),
        Binding("?", "auto_generate_children", "(AI auto)"),
    ] + [
        Binding(str(i), f"generate_children({i})", "(AI nodes)", show=False)
        for i in range(2, 10)
    ]

    def __init__(self, initial_markdown_path: str | Path | None = None) -> None:
        super().__init__()
        self.title = "m1ndm4p"
        self._tree_widget: Optional[MindmapTree] = None
        self.mindmap_root = MindmapNode("Central Idea")
        self._edit_state: Optional[dict[str, object]] = None
        self.model_choices = list(ai.AVAILABLE_MODELS)
        self.selected_model = ai.get_active_model()
        if self.selected_model not in self.model_choices:
            self.model_choices.append(self.selected_model)
        self._markmap_preview_path: Optional[Path] = None
        ai.set_active_model(self.selected_model)
        self._full_pending: Optional[dict[str, object]] = None
        self._full_in_progress = False
        self._external_context = ""
        self._context_url = ""
        self._stop_after_current_step = False
        self._full_task: Optional[asyncio.Task[None]] = None
        self._level_pending = False
        self._level_anchor_id: Optional[str] = None
        self._current_level_limit: int = 2
        self._active_path: Optional[Path] = None
        self._initial_load_path: Optional[Path] = (
            Path(initial_markdown_path).expanduser() if initial_markdown_path else None
        )
        ai.reset_prompt_log()
        ai.reset_connection_log()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        tree = MindmapTree("Mind Map", id="mindmap-tree")
        tree.show_root = True
        self._tree_widget = tree
        yield tree
        yield Footer()

    def on_mount(self) -> None:
        self.rebuild_tree()
        loaded = False
        if self._initial_load_path:
            loaded = self._load_mindmap(self._initial_load_path)
        if not loaded:
            self._apply_current_level_limit()

    def rebuild_tree(self) -> None:
        ai.reset_dummy_counters()
        tree = self.require_tree()
        tree.clear()
        root_node = tree.root
        root_node.set_label(self._format_node_label(self.mindmap_root))
        root_node.data = self.mindmap_root
        self.populate_tree(root_node, self.mindmap_root)
        tree.select_node(root_node)
        tree.focus()
        tree.refresh(layout=True)
        tree.call_after_refresh(lambda: self._apply_current_level_limit(announce=False))

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
        """Selection changes automatically refresh the level indicator."""
        if event.node.data is None:
            return
        # No status message needed—the level indicator already reflects scope.

    def get_selected_tree_node(self) -> Optional[Tree.Node[MindmapNode]]:
        tree = self.require_tree()
        return tree.cursor_node

    def get_selected_model_node(self) -> Optional[MindmapNode]:
        selected = self.get_selected_tree_node()
        return selected.data if selected else None

    def _format_node_label(self, node: MindmapNode) -> Text:
        normalized = self._normalized_emoji_spacing(node.title)
        if normalized != node.title:
            node.title = normalized
        return Text(normalized)

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

    @staticmethod
    def _strip_leading_emoji(text: str) -> tuple[str, str]:
        return _extract_leading_emoji(text)

    def _with_prefixed_emoji(self, text: str, emoji: str) -> str:
        base, _ = self._strip_leading_emoji(text)
        base = base.lstrip()
        nbsp = "\u00A0"
        if base:
            return f"{emoji}{nbsp}{base}"
        return f"{emoji}{nbsp}"

    def _normalized_emoji_spacing(self, text: str) -> str:
        remainder, emoji = self._strip_leading_emoji(text)
        if not emoji:
            return text
        nbsp = "\u00A0"
        remainder = remainder.lstrip()
        prefix = emoji.rstrip()
        return f"{prefix}{nbsp}{remainder}" if remainder else f"{prefix}{nbsp}"

    def _capture_expand_state(self) -> dict[str, bool]:
        tree = self.require_tree()
        state: dict[str, bool] = {}
        stack = [tree.root]
        while stack:
            node = stack.pop()
            state[node.id] = node.is_expanded
            stack.extend(node.children)
        return state

    def _restore_expand_state(self, state: dict[str, bool]) -> None:
        tree = self.require_tree()
        for node_id, expanded in state.items():
            node = tree.get_node_by_id(node_id)
            if node is None:
                continue
            if expanded:
                node.expand()
            else:
                node.collapse()
        tree.refresh(layout=True)

    def _has_external_context(self) -> bool:
        return bool(self._external_context.strip())

    def _context_state_label(self) -> str:
        return "on" if self._has_external_context() else "off"

    def _set_external_context(self, value: str) -> None:
        self._external_context = value.strip()

    def _contextual_markdown(self) -> str:
        base = to_markdown(self.mindmap_root)
        extra = self._external_context.strip()
        if not extra:
            return base
        external_block = f"External knowledge:\n{extra}"
        return f"{base}\n\n{external_block}" if base else external_block

    @contextmanager
    def _prompt_log_session(self, label: str) -> None:
        ai.start_prompt_session(label)
        try:
            yield
        finally:
            ai.finish_prompt_session()

    @staticmethod
    def _strip_html(html_text: str) -> str:
        class _Extractor(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self._chunks: list[str] = []

            def handle_data(self, data: str) -> None:
                if data.strip():
                    self._chunks.append(data.strip())

            def get_text(self) -> str:
                return "\n".join(self._chunks)

        extractor = _Extractor()
        try:
            extractor.feed(html_text)
        except Exception:
            return html_text
        return "\n".join(line for line in extractor.get_text().splitlines() if line.strip())

    def _markmap_markdown(self) -> str:
        tree = self.require_tree()
        root = tree.root
        title = self.mindmap_root.title.strip() or "Mind map"
        escaped_title = self._markmap_escape(title)
        lines: list[str] = [f'# <span class="mm-label mm-root">{escaped_title}</span>', ""]
        for index, child in enumerate(root.children):
            lines.extend(self._markmap_collect_lines(child, level=0))
            if index != len(root.children) - 1:
                lines.append("")
        while lines and not lines[-1].strip():
            lines.pop()
        lines.append("")
        return "\n".join(lines)

    def _body_paragraphs(self, body: str) -> list[str]:
        paragraphs: list[str] = []
        buffer: list[str] = []
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line:
                if buffer:
                    paragraphs.append(" ".join(buffer))
                    buffer = []
                continue
            buffer.append(line)
        if buffer:
            paragraphs.append(" ".join(buffer))
        return paragraphs or ([""] if body.strip() else [])

    @staticmethod
    def _markmap_escape(text_value: str) -> str:
        return html.escape(text_value, quote=False)

    def _markmap_collect_lines(self, tree_node: Tree.Node[MindmapNode], level: int) -> list[str]:
        indent = "  " * level
        data = tree_node.data
        lines: list[str] = []
        if isinstance(data, MindmapNode):
            label = data.title.strip() or "(untitled)"
            escaped_label = self._markmap_escape(label)
            lines.append(f'{indent}- <span class="mm-label">{escaped_label}</span>')
            is_expanded = getattr(tree_node, "is_expanded", True)
            if data.body and is_expanded:
                for paragraph in self._body_paragraphs(data.body):
                    if paragraph.strip():
                        escaped_paragraph = self._markmap_escape(paragraph.strip())
                        lines.append(
                            f'{indent}  - <span class="mm-label mm-body">{escaped_paragraph}</span>'
                        )
            has_children = bool(tree_node.children)
            if has_children:
                if not is_expanded:
                    lines.append(f"{indent}  <!-- markmap: fold -->")
                for child in tree_node.children:
                    lines.extend(self._markmap_collect_lines(child, level + 1))
        return lines

    def _markmap_preview_html(self, markdown: str) -> str:
        safe_markdown = markdown.replace("</script>", "<\\/script>")
        return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>m1ndm4p Markmap preview</title>
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <style>
      :root, html, body {{
        height: 100%;
      }}
      body {{
        margin: 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #0f0f0f;
        color: #f2f2f2;
      }}
      .markmap {{
        position: relative;
        width: 100%;
        height: 100%;
      }}
      .markmap > svg {{
        width: 100%;
        height: 100%;
      }}
      .markmap > svg text {{
        fill: #f2f2f2;
      }}
      .mm-label {{
        color: #f2f2f2 !important;
      }}
      .mm-body {{
        font-style: italic;
      }}
      .mm-root {{
        color: #ffffff !important;
        font-weight: 600;
      }}
    </style>
    <script>
      // Collapse nodes by inserting <!-- markmap: fold --> under the bullet.
      window.markmap = {{ autoLoader: {{ toolbar: true }} }};
      console.log("Markmap preview generated at runtime.");
    </script>
  </head>
  <body>
    <div class="markmap">
      <script type="text/template">
{safe_markdown.strip()}
      </script>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/markmap-autoloader@latest"></script>
    <script>
      (async () => {{
        if (!window.markmap || !window.markmap.ready) {{
          return;
        }}
        await window.markmap.ready;
        document.querySelectorAll(".markmap > svg").forEach((svg) => {{
          const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
          style.textContent = `
            .markmap-fo > .markmap-node-content,
            .markmap-fo > .markmap-node-content * {{
              color: #f2f2f2 !important;
              opacity: 1 !important;
            }}
            text {{
              fill: #f2f2f2 !important;
              opacity: 1 !important;
            }}
            .markmap-node {{
              opacity: 1 !important;
            }}
          `;
          svg.appendChild(style);
        }});
      }})();
    </script>
  </body>
</html>
"""

    def _write_markmap_preview(self) -> Path:
        markdown = self._markmap_markdown()
        html = self._markmap_preview_html(markdown)
        path = Path("markmap_preview.html")
        path.write_text(html, encoding="utf-8")
        return path

    def _open_markmap_preview(self, path: Path) -> None:
        uri = path.resolve().as_uri()
        if sys.platform == "darwin":
            try:
                subprocess.Popen(["open", "-g", uri])
                return
            except Exception:
                pass
        webbrowser.open(uri, new=2)

    @staticmethod
    def _hostname_from_url(url: str) -> str:
        parsed = urlparse(url)
        return parsed.netloc or url

    def _download_url_text(self, url: str) -> str:
        parsed = urlparse(url)
        if not parsed.scheme:
            url = f"https://{url}"
        req = request.Request(
            url,
            headers={
                "User-Agent": "m1ndm4p/1.0 (+https://github.com/m1ndm4p)",
                "Accept": "text/html, text/plain;q=0.9",
            },
        )
        with request.urlopen(req, timeout=15) as response:
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
            body = response.read().decode(charset, errors="replace")
        if "html" in content_type.lower():
            return self._strip_html(body)
        return body

    async def _import_context_from_url(self, url: str) -> None:
        target = url.strip()
        if not target:
            self.show_status("URL import cancelled.")
            return
        parsed = urlparse(target)
        if not parsed.scheme:
            target = f"https://{target}"
        host = self._hostname_from_url(target)
        self.show_status(f"Fetching context from {host}…")
        try:
            text = await asyncio.to_thread(self._download_url_text, target)
        except Exception as exc:
            self.bell()
            self.show_status(f"Failed to fetch {host}: {exc}")
            return
        cleaned = text.strip()
        if not cleaned:
            self.show_status(f"No usable text returned from {host}.")
            return
        self._set_external_context(cleaned)
        self._context_url = target
        self.show_status(f"Imported web context from {host}.")

    def _resolve_level_anchor(self) -> Tree.Node[MindmapNode]:
        tree = self.require_tree()
        if self._level_anchor_id:
            try:
                node = tree.get_node_by_id(self._level_anchor_id)
            except UnknownNodeID:
                node = None
            else:
                if node is not None:
                    return node
        return tree.root

    def _apply_level_limit(
        self,
        level_limit: int,
        *,
        anchor: Tree.Node[MindmapNode] | None = None,
        announce: bool = True,
    ) -> None:
        tree = self.require_tree()
        target = anchor or tree.root
        current = target
        while current.parent is not None:
            current.parent.expand()
            current = current.parent

        branch_node = target.data if isinstance(target.data, MindmapNode) else None
        branch_max_level = (
            self._calculate_max_level(branch_node) if branch_node is not None else self._calculate_max_level(self.mindmap_root)
        )

        if level_limit <= 0:
            display_level = branch_max_level
            target.expand_all()
        else:
            display_level = max(0, min(level_limit, branch_max_level))

            def clamp(node: Tree.Node[MindmapNode], depth: int) -> None:
                if depth >= display_level:
                    node.collapse()
                    return
                node.expand()
                for child in node.children:
                    clamp(child, depth + 1)

            clamp(target, 0)

        tree.refresh(layout=True)
        if target is tree.root:
            self._current_level_limit = level_limit
        max_for_message = branch_max_level if target is not tree.root else self._calculate_max_level(self.mindmap_root)
        status = self._format_level_status(
            display_level,
            max_for_message,
            target if target is not tree.root else None,
        )
        if announce:
            self.show_status(level_status=status)

    def _calculate_max_level(self, node: MindmapNode | None) -> int:
        if node is None or not node.children:
            return 0
        return 1 + max(self._calculate_max_level(child) for child in node.children)

    def _format_level_status(
        self,
        display_level: int,
        max_level: int,
        target: Tree.Node[MindmapNode] | None = None,
    ) -> str:
        base = f"Level {display_level} of {max_level}"
        if target is not None:
            data = target.data
            if isinstance(data, MindmapNode):
                title = data.title
            else:
                title = "selection"
            base = f'{base} under "{title}"'
        return base

    def _current_level_status(self) -> str:
        max_level = self._calculate_max_level(self.mindmap_root)
        limit = self._current_level_limit
        if limit <= 0:
            display = max_level
        else:
            display = max(0, min(limit, max_level))
        return self._format_level_status(display, max_level)

    def _apply_current_level_limit(self, *, announce: bool = True) -> None:
        tree = self.require_tree()
        self._apply_level_limit(self._current_level_limit, anchor=tree.root, announce=announce)

    def _start_inline_edit(
        self,
        tree_node: Tree.Node,
        initial_text: str,
        *,
        kind: str,
        context: dict[str, object],
    ) -> None:
        original_label = tree_node._label.copy()
        self._edit_state = {
            "tree_node": tree_node,
            "text": initial_text,
            "cursor": len(initial_text),
            "selection_anchor": 0 if initial_text else None,
            "kind": kind,
            "context": context,
            "original_label": original_label,
            "initial": initial_text,
            "tree_node_id": tree_node.id,
        }
        self._update_edit_label()
        self.show_status("Editing… Enter to save, Esc to cancel.")

    def _edit_selection_bounds(self) -> tuple[int, int]:
        if self._edit_state is None:
            return (0, 0)
        cursor = self._edit_state["cursor"]
        anchor = self._edit_state.get("selection_anchor")
        if anchor is None or anchor == cursor:
            return (cursor, cursor)
        start = min(cursor, anchor)
        end = max(cursor, anchor)
        return (start, end)

    def _edit_delete_selection(self) -> bool:
        if self._edit_state is None:
            return False
        start, end = self._edit_selection_bounds()
        if start == end:
            return False
        text_value: str = self._edit_state["text"]
        self._edit_state["text"] = text_value[:start] + text_value[end:]
        self._edit_state["cursor"] = start
        self._edit_state["selection_anchor"] = None
        return True

    def _edit_move_cursor(
        self,
        *,
        delta: Optional[int] = None,
        to: Optional[int] = None,
        extend: bool = False,
    ) -> None:
        if self._edit_state is None:
            return
        state = self._edit_state
        if to is None and delta is None:
            return
        target = to if to is not None else state["cursor"] + (delta or 0)
        length = len(state["text"])
        target = max(0, min(length, target))
        if extend:
            if state.get("selection_anchor") is None:
                state["selection_anchor"] = state["cursor"]
        else:
            state["selection_anchor"] = None
        state["cursor"] = target
        self._update_edit_label()

    def _edit_insert_text(self, value: str) -> None:
        if self._edit_state is None or not value:
            return
        state = self._edit_state
        if self._edit_delete_selection():
            pass
        cursor = state["cursor"]
        text_value: str = state["text"]
        state["text"] = text_value[:cursor] + value + text_value[cursor:]
        state["cursor"] = cursor + len(value)
        state["selection_anchor"] = None
        self._update_edit_label()

    def _edit_backspace(self) -> None:
        if self._edit_state is None:
            return
        if self._edit_delete_selection():
            self._update_edit_label()
            return
        cursor = self._edit_state["cursor"]
        if cursor == 0:
            return
        text_value: str = self._edit_state["text"]
        self._edit_state["text"] = text_value[: cursor - 1] + text_value[cursor:]
        self._edit_state["cursor"] = cursor - 1
        self._update_edit_label()

    def _edit_delete_forward(self) -> None:
        if self._edit_state is None:
            return
        if self._edit_delete_selection():
            self._update_edit_label()
            return
        cursor = self._edit_state["cursor"]
        text_value: str = self._edit_state["text"]
        if cursor >= len(text_value):
            return
        self._edit_state["text"] = text_value[:cursor] + text_value[cursor + 1 :]
        self._update_edit_label()

    def _edit_select_all(self) -> None:
        if self._edit_state is None:
            return
        state = self._edit_state
        state["selection_anchor"] = 0
        state["cursor"] = len(state["text"])
        self._update_edit_label()

    def _update_edit_label(self) -> None:
        if self._edit_state is None:
            return
        tree_node = self._edit_state["tree_node"]
        kind = self._edit_state["kind"]
        text_value: str = self._edit_state["text"]
        cursor: int = self._edit_state["cursor"]
        anchor = self._edit_state.get("selection_anchor")
        selection: Optional[tuple[int, int]] = None
        if anchor is not None and anchor != cursor:
            selection = (min(anchor, cursor), max(anchor, cursor))
        caret = "▌"
        prefix = "  " if kind == "body_line" else ""
        base_style = "dim italic" if kind == "body_line" else None
        highlight_style = f"{base_style} reverse" if base_style else "reverse"
        caret_style = base_style

        label = Text()
        if prefix:
            label.append(prefix, style=base_style)

        caret_inserted = False
        if cursor == 0:
            label.append(caret, style=caret_style)
            caret_inserted = True

        for index, character in enumerate(text_value):
            if not caret_inserted and cursor == index:
                label.append(caret, style=caret_style)
                caret_inserted = True
            char_style = base_style
            if selection and selection[0] <= index < selection[1]:
                char_style = highlight_style
            label.append(character, style=char_style)

        if not caret_inserted:
            label.append(caret, style=caret_style)

        tree_node.set_label(label)
        self.require_tree().refresh(layout=True)

    def _handle_edit_key(self, event: events.Key) -> None:
        if self._edit_state is None:
            return
        raw_key = getattr(event, "key", "")
        key_name, modifiers = _key_name_and_modifiers(raw_key or "")
        shift_held = "shift" in modifiers
        control_held = bool({"ctrl", "control", "cmd", "command", "meta"} & modifiers)
        if key_name == "escape":
            self._cancel_inline_edit()
            event.stop()
            return
        if key_name == "enter":
            self._commit_inline_edit()
            event.stop()
            return
        if key_name == "backspace":
            self._edit_backspace()
            event.stop()
            return
        if key_name == "delete":
            self._edit_delete_forward()
            event.stop()
            return
        if key_name in {"left", "right"}:
            delta = -1 if key_name == "left" else 1
            self._edit_move_cursor(delta=delta, extend=shift_held)
            event.stop()
            return
        if key_name == "home":
            self._edit_move_cursor(to=0, extend=shift_held)
            event.stop()
            return
        if key_name == "end":
            self._edit_move_cursor(to=len(self._edit_state["text"]), extend=shift_held)
            event.stop()
            return
        if key_name == "tab":
            event.stop()
            return
        if key_name == "a" and control_held:
            self._edit_select_all()
            event.stop()
            return
        character = event.character
        if event.is_printable and character and len(character) == 1:
            self._edit_insert_text(character)
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
        expand_state = self._capture_expand_state()
        text_value = self._edit_state["text"].strip()
        initial = (self._edit_state.get("initial") or "").strip()
        kind = self._edit_state["kind"]
        tree_node = self._edit_state["tree_node"]
        context = self._edit_state["context"]
        tree = self.require_tree()

        try:
            if kind == "body_line":
                parent_node: MindmapNode = context["parent_node"]  # type: ignore[assignment]
                line_index: int = context["line_index"]  # type: ignore[assignment]
                new_line_value = text_value
                self._update_body_line(parent_node, tree_node, line_index, new_line_value)
                self.show_status("Text updated.")
            else:
                node: MindmapNode = context["node"]  # type: ignore[assignment]
                new_title = text_value or initial or node.title
                node.title = new_title
            tree_node.set_label(self._format_node_label(node))
            tree.refresh(layout=True)
            self._select_tree_node_without_toggle(tree_node)
            self.show_status("Node title updated.")
        finally:
            self._edit_state = None
            self._restore_expand_state(expand_state)

    def _update_body_line(
        self,
        parent_node: MindmapNode,
        line_tree_node: Tree.Node[MindmapNode],
        line_index: int,
        new_value: str,
    ) -> None:
        lines = self._body_lines(parent_node.body or "")
        while len(lines) <= line_index:
            lines.append("")
        lines[line_index] = new_value
        updated_body = "\n".join(lines).strip()
        parent_node.body = updated_body[:200]
        display = f"  {new_value}" if new_value else "  "
        line_tree_node.set_label(Text(display, style="dim italic"))
        tree = self.require_tree()
        tree.refresh(layout=True)
        self._select_tree_node_without_toggle(line_tree_node)
        tree.focus()

    async def _request_emoji(
        self,
        tree_node: Tree.Node[MindmapNode],
        text: str,
        path_label: str,
    ) -> str:
        snippet = text.strip()
        if not snippet:
            return ""
        limited = snippet[:200]
        result = await self._call_ai_with_spinner(
            tree_node,
            "Emoji suggestion",
            ai.suggest_emoji,
            limited,
            path_label,
        )
        if isinstance(result, str):
            emoji = result.strip()
            if emoji.upper() == "NONE":
                return ""
            return emoji
        return ""

    def _tree_node_level(self, tree_node: Tree.Node[MindmapNode]) -> int:
        level = 0
        current = tree_node
        while current.parent is not None:
            level += 1
            current = current.parent
        return level

    def _node_path(self, tree_node: Tree.Node[MindmapNode]) -> list[str]:
        path: list[str] = []
        current: Tree.Node[MindmapNode] | None = tree_node
        while current is not None:
            data = current.data
            if isinstance(data, MindmapNode):
                path.append(data.title)
            current = current.parent
        if not path:
            return [self.mindmap_root.title]
        return list(reversed(path))

    def _resolve_model_tree_node(
        self, tree_node: Tree.Node[MindmapNode]
    ) -> tuple[Tree.Node[MindmapNode], MindmapNode] | tuple[None, None]:
        data = tree_node.data
        if isinstance(data, MindmapNode):
            return tree_node, data
        if isinstance(data, dict) and data.get("kind") == "body_line":
            parent_node = tree_node.parent
            if parent_node is not None and isinstance(parent_node.data, MindmapNode):
                return parent_node, parent_node.data
        return None, None

    def _apply_generated_subtree(
        self,
        tree_node: Tree.Node[MindmapNode],
        model_node: MindmapNode,
        generated_root: MindmapNode,
        *,
        replace_children: bool,
        update_body: bool,
        expand_all: bool,
    ) -> None:
        if update_body:
            body_text = (generated_root.body or "").strip()
            if body_text:
                model_node.body = body_text
        if replace_children:
            model_node.children = generated_root.children
        self.populate_tree(tree_node, model_node)
        tree = self.require_tree()
        if replace_children:
            if expand_all:
                tree_node.expand_all()
            else:
                tree_node.expand()
        self._select_tree_node_without_toggle(tree_node)
        tree.refresh(layout=True)

    async def _request_structured_subtree(
        self,
        tree_node: Tree.Node[MindmapNode],
        *,
        depth: int,
        spinner_label: str,
        mode: Literal["auto", "exact"],
        exact_children: int | None = None,
        include_body: bool = False,
        level_targets: list[int] | None = None,
    ) -> Optional[MindmapNode]:
        node_path = self._node_path(tree_node)
        context = self._contextual_markdown()
        markdown = await self._call_ai_with_spinner(
            tree_node,
            spinner_label,
            ai.generate_structured_subtree,
            node_path,
            depth,
            context_markdown=context,
            include_body=include_body,
            mode=mode,
            exact_children=exact_children,
            level_targets=level_targets,
        )
        if not isinstance(markdown, str):
            return None
        cleaned = markdown.strip()
        if not cleaned:
            return None
        snippet = cleaned if cleaned.endswith("\n") else f"{cleaned}\n"
        try:
            return from_markdown(snippet)
        except ValueError:
            self.show_status("AI returned malformed outline.")
            return None

    @staticmethod
    def _unique_child_title(raw_title: str | None, parent: MindmapNode) -> str:
        candidate = (raw_title or "").strip()
        existing_titles = {child.title for child in parent.children}
        if not candidate or candidate in existing_titles:
            base_index = len(parent.children) + 1
            candidate = f"{parent.title} idea {base_index}"
            while candidate in existing_titles:
                base_index += 1
                candidate = f"{parent.title} idea {base_index}"
        return candidate

    def _select_tree_node_without_toggle(self, node: Tree.Node[MindmapNode]) -> None:
        tree = self.require_tree()
        auto_expand_initial = tree.auto_expand
        if auto_expand_initial:
            tree.auto_expand = False

            def _restore_auto_expand() -> None:
                tree.auto_expand = auto_expand_initial

            self.call_after_refresh(_restore_auto_expand)
        tree.select_node(node)

    async def _generate_body_for_node(
        self,
        tree_node: Tree.Node[MindmapNode],
        model_node: MindmapNode,
    ) -> None:
        context = self._contextual_markdown()
        node_path = self._node_path(tree_node)
        markdown = await self._call_ai_with_spinner(
            tree_node,
            "generate paragraph",
            ai.generate_structured_subtree,
            node_path,
            0,
            context_markdown=context,
            include_body=True,
            mode="body",
        )
        if not isinstance(markdown, str):
            return
        cleaned = markdown.strip()
        if not cleaned:
            self.show_status(f"No text returned for {model_node.title}.")
            return
        snippet = cleaned if cleaned.endswith("\n") else f"{cleaned}\n"
        try:
            generated_root = from_markdown(snippet)
        except ValueError:
            self.show_status("AI returned malformed text.")
            return
        body_text = (generated_root.body or "").strip()
        if not body_text:
            self.show_status(f"No text returned for {model_node.title}.")
            return
        self._apply_generated_subtree(
            tree_node,
            model_node,
            generated_root,
            replace_children=False,
            update_body=True,
            expand_all=False,
        )

    def _gather_leaf_nodes(
        self,
        root_tree_node: Tree.Node[MindmapNode],
    ) -> list[tuple[Tree.Node[MindmapNode], MindmapNode]]:
        leaves: list[tuple[Tree.Node[MindmapNode], MindmapNode]] = []
        stack: list[Tree.Node[MindmapNode]] = [root_tree_node]
        while stack:
            node = stack.pop()
            data = node.data
            if not isinstance(data, MindmapNode):
                continue
            if data.children:
                for child_node in node.children:
                    stack.append(child_node)
            else:
                leaves.append((node, data))
        return leaves

    def _start_full_task(self, coro: Awaitable[None], *, label: str) -> None:
        if self._full_task and not self._full_task.done():
            self.bell()
            self.show_status(f"{label} already running.")
            return
        task: asyncio.Task[None] = asyncio.create_task(coro)
        self._full_task = task

        def _on_done(completed: asyncio.Task[None]) -> None:
            if self._full_task is completed:
                self._full_task = None
            if completed.cancelled():
                return
            exc = completed.exception()
            if exc is not None:
                self.bell()
                self.show_status(f"{label} error: {exc}")

        task.add_done_callback(_on_done)

    def _resolve_pending_node(
        self, pending: Optional[dict[str, object]]
    ) -> Optional[Tree.Node[MindmapNode]]:
        if not isinstance(pending, dict):
            return None
        tree = self.require_tree()
        pending_node = pending.get("node_ref") if isinstance(pending, dict) else None
        target_node = pending_node if isinstance(pending_node, TreeNode) else None
        node_id = pending.get("node_id") if isinstance(pending, dict) else None
        if target_node is None and node_id is not None:
            try:
                target_node = tree.get_node_by_id(node_id)
            except UnknownNodeID:
                target_node = None
        if target_node is None:
            target_node = tree.root
        if target_node is tree.root and not isinstance(target_node.data, MindmapNode):
            target_node.data = self.mindmap_root
        if target_node is None or not isinstance(target_node.data, MindmapNode):
            return None
        return target_node

    @staticmethod
    def _parse_full_counts(text: str) -> tuple[int, list[int] | None]:
        cleaned = text.strip().strip(",")
        if not cleaned:
            raise ValueError("Enter a depth (1-5) or comma-separated counts before running full build.")
        if "," in cleaned:
            parts = [part.strip() for part in cleaned.split(",") if part.strip()]
            if not parts:
                raise ValueError("Provide at least one count when using commas (e.g., 8,5).")
            if len(parts) > 5:
                raise ValueError("At most 5 levels are supported.")
            counts: list[int] = []
            for part in parts:
                value = int(part)
                if value <= 0:
                    raise ValueError("Counts must be positive integers.")
                counts.append(value)
            return len(counts), counts
        depth = int(cleaned)
        if not 1 <= depth <= 5:
            raise ValueError("Depth must be between 1 and 5.")
        return depth, None

    def _full_prompt_message(self, pending: dict[str, object]) -> str:
        include_body = bool(pending.get("include_body"))
        mode = "with text" if include_body else "titles only"
        counts_input = str(pending.get("counts_input") or "")
        if counts_input:
            return (
                f"Full build ({mode}), counts '{counts_input}'. "
                "Press Enter to run, T toggles text, Esc cancels."
            )
        return (
            f"Full build pending ({mode}). Type a depth 1-5 or counts like '8,5'. "
            "Press Enter to run, T toggles text, Esc cancels."
        )

    async def _generate_text_for_leaves(
        self,
        root_tree_node: Tree.Node[MindmapNode],
    ) -> None:
        data = root_tree_node.data
        if not isinstance(data, MindmapNode):
            self.show_status("Selected node no longer available.")
            return
        self.populate_tree(root_tree_node, data)
        leaves = self._gather_leaf_nodes(root_tree_node)
        if not leaves:
            self.show_status("No leaf nodes to annotate.")
            return
        pending_leaves = [
            (node, leaf_data) for node, leaf_data in leaves if not (leaf_data.body or "").strip()
        ]
        if not pending_leaves:
            self.show_status("All leaf nodes already have text.")
            return
        context_markdown = self._contextual_markdown()
        with self._prompt_log_session("Generate leaf text"):
            self._full_in_progress = True
            self._stop_after_current_step = False
            tree = self.require_tree()
            root_tree_node.expand_all()
            self._select_tree_node_without_toggle(root_tree_node)
            tree.refresh(layout=True)
            completed = 0
            cancelled = False
            try:
                batch_size = 6
                for batch_start in range(0, len(pending_leaves), batch_size):
                    if self._stop_after_current_step:
                        cancelled = True
                        break
                    batch = pending_leaves[batch_start : batch_start + batch_size]
                    entries: list[dict[str, object]] = []
                    for node, model in batch:
                        entries.append(
                            {
                                "path": self._node_path(node),
                                "title": model.title,
                                "level": self._tree_node_level(node),
                            }
                        )
                    bodies = await self._call_ai_with_spinner(
                        root_tree_node,
                        "generate leaf text",
                        ai.generate_leaf_bodies,
                        entries,
                        context_markdown=context_markdown,
                    )
                    if bodies is None:
                        continue
                    if not isinstance(bodies, list):
                        continue
                    if self._stop_after_current_step:
                        cancelled = True
                        break
                    for (leaf_node, leaf_model), body_text in zip(batch, bodies):
                        if not isinstance(body_text, str):
                            continue
                        cleaned = body_text.strip()
                        if not cleaned:
                            continue
                        leaf_model.body = cleaned
                        self.populate_tree(leaf_node, leaf_model)
                        leaf_node.expand_all()
                        completed += 1
            finally:
                self._full_in_progress = False
        if cancelled:
            self._stop_after_current_step = False
            root_tree_node.expand_all()
            self._select_tree_node_without_toggle(root_tree_node)
            tree.refresh(layout=True)
            self.show_status(f"Stopped after creating {completed} text section{'s' if completed != 1 else ''}.")
            return
        root_tree_node.expand_all()
        self._select_tree_node_without_toggle(root_tree_node)
        tree.refresh(layout=True)
        self.show_status(
            f"Generated text for {completed} leaf node{'s' if completed != 1 else ''}."
        )

    async def _handle_full_pending_key(self, event: events.Key) -> bool:
        if self._full_pending is None:
            return False
        raw_key = getattr(event, "key", "")
        key_name, modifiers = _key_name_and_modifiers(raw_key or "")
        if key_name == "escape":
            self._full_pending = None
            self.show_status("Full build cancelled.")
            event.stop()
            return True
        pending = self._full_pending
        if key_name == "t":
            if not isinstance(pending, dict):
                return False
            include_body = bool(pending.get("include_body"))
            pending["include_body"] = not include_body
            self.show_status(self._full_prompt_message(pending))
            event.stop()
            return True
        if self._full_in_progress:
            self.bell()
            self.show_status("Full build already running.")
            event.stop()
            return True
        counts_input = str(pending.get("counts_input") or "")
        if key_name in {"enter", "return"}:
            cleaned = counts_input.strip()
            if not cleaned:
                self.bell()
                self.show_status("Enter a depth (1-5) or counts like '8,5' before pressing Enter.")
                event.stop()
                return True
            try:
                depth, level_targets = self._parse_full_counts(cleaned)
            except (ValueError, TypeError) as exc:
                self.bell()
                message = str(exc) or "Invalid depth/count input."
                self.show_status(message)
                event.stop()
                return True
            self._full_pending = None
            target_node = self._resolve_pending_node(pending)
            if target_node is None:
                self.bell()
                self.show_status("Selected node no longer available.")
                event.stop()
                return True
            event.stop()
            self._start_full_task(
                self._execute_full_build(
                    target_node,
                    depth,
                    include_body=bool(pending.get("include_body")),
                    level_targets=level_targets,
                ),
                label="Full build",
            )
            return True
        if key_name == "backspace":
            if counts_input:
                pending["counts_input"] = counts_input[:-1]
            self.show_status(self._full_prompt_message(pending))
            event.stop()
            return True
        if key_name == "," or (key_name == "comma" and not modifiers):
            if not counts_input or counts_input.endswith(","):
                self.bell()
                self.show_status("Add a number before inserting another comma (e.g., 8,5).")
                event.stop()
                return True
            segments = counts_input.split(",")
            if len(segments) >= 5:
                self.bell()
                self.show_status("At most 5 levels are supported.")
                event.stop()
                return True
            pending["counts_input"] = counts_input + ","
            self.show_status(self._full_prompt_message(pending))
            event.stop()
            return True
        if key_name.isdigit():
            pending["counts_input"] = counts_input + key_name
            self.show_status(self._full_prompt_message(pending))
            event.stop()
            return True
        # allow direct digit bindings (Textual gives uppercase numerals as actual digits)
        if raw_key.isdigit():
            pending["counts_input"] = counts_input + raw_key
            self.show_status(self._full_prompt_message(pending))
            event.stop()
            return True
        # consume unexpected keys while pending
        self.bell()
        self.show_status(
            "Full build: type a depth (1-5) or comma-separated counts, T toggles text, Enter runs, Esc cancels."
        )
        event.stop()
        return True

    async def _call_ai_with_spinner(
        self,
        tree_node: Tree.Node[MindmapNode],
        label: str,
        callable_: Callable[..., object | None],
        *args: object,
        **kwargs: object,
    ) -> object | None:
        tree = self.require_tree()
        spinner_frames = ["  .", "  ..", "  ..."]
        spinner_index = 0
        spinner_node = tree_node.add_leaf(Text(spinner_frames[0], style="dim italic"))
        spinner_node.allow_expand = False
        tree_node.expand()
        tree.refresh(layout=True)

        def tick() -> None:
            nonlocal spinner_index
            spinner_index = (spinner_index + 1) % len(spinner_frames)
            spinner_node.set_label(Text(spinner_frames[spinner_index], style="dim italic"))
            tree.refresh(layout=False)

        spinner_timer = self.set_interval(0.25, tick)
        task = asyncio.create_task(asyncio.to_thread(callable_, *args, **kwargs))
        result: object | None = None
        try:
            result = await task
        except asyncio.CancelledError:
            self.show_status("AI request cancelled.")
            result = None
        except Exception as exc:  # pragma: no cover - defensive programming
            self.handle_ai_error(label, exc)
        finally:
            spinner_timer.stop()
            spinner_node.remove()
            tree.refresh(layout=True)
        return result

    def handle_ai_error(self, action_label: str, exc: Exception) -> None:
        """Provide a consistent error experience for AI failures."""
        self.bell()
        self.show_status(f"{action_label.capitalize()} failed: {exc}")

    async def _execute_full_build(
        self,
        tree_node: Tree.Node[MindmapNode],
        depth: int,
        *,
        include_body: bool = False,
        level_targets: list[int] | None = None,
    ) -> None:
        data = tree_node.data
        if not isinstance(data, MindmapNode):
            self.bell()
            self.show_status("Full build requires a node selection.")
            return
        generated_root: MindmapNode | None = None
        with self._prompt_log_session(f"Full build depth {depth}"):
            self._full_in_progress = True
            self._stop_after_current_step = False
            self._select_tree_node_without_toggle(tree_node)
            self.show_status(f"Full build to depth {depth} in progress…")
            try:
                generated_root = await self._request_structured_subtree(
                    tree_node,
                    depth=depth,
                    spinner_label="full build",
                    mode="auto",
                    include_body=include_body,
                    level_targets=level_targets,
                )
            finally:
                self._full_in_progress = False
        if self._stop_after_current_step:
            self._stop_after_current_step = False
            self.show_status("Full build cancelled.")
            return
        if generated_root is None:
            self.show_status("Full build returned no changes.")
            return
        if not generated_root.children:
            self.show_status(f"Full build: no children added for {data.title}.")
            return
        self._apply_generated_subtree(
            tree_node,
            data,
            generated_root,
            replace_children=True,
            update_body=include_body,
            expand_all=True,
        )
        mode = "with text" if include_body else "titles only"
        if level_targets:
            targets = ", ".join(str(count) for count in level_targets)
            self.show_status(f"Full build complete to depth {depth} ({mode}, targets: {targets}).")
        else:
            self.show_status(f"Full build complete to depth {depth} ({mode}).")

    async def action_add_child_suggestion(self) -> None:
        selected_tree_node = self.get_selected_tree_node()
        if selected_tree_node is None:
            self.bell()
            self.show_status("No node selected.")
            return

        target_tree_node, model_node = self._resolve_model_tree_node(selected_tree_node)
        if target_tree_node is None or model_node is None:
            self.bell()
            self.show_status("Cannot add a child to this entry.")
            return

        target_node_id = target_tree_node.id
        with self._prompt_log_session("Add child suggestion"):
            generated_root = await self._request_structured_subtree(
                target_tree_node,
                depth=1,
                spinner_label="add child suggestion",
                mode="exact",
                exact_children=1,
            )
        if generated_root is None:
            return

        if not generated_root.children:
            self.show_status("No suggestion returned.")
            return

        suggestion_node = generated_root.children[0]
        candidate_title = suggestion_node.title.strip()
        existing_titles = {child.title for child in model_node.children}
        if not candidate_title or candidate_title in existing_titles:
            self.show_status("Suggestion duplicates existing child.")
            return

        new_title = self._unique_child_title(candidate_title, model_node)
        suggestion_node.title = new_title
        model_node.children.append(suggestion_node)
        self.populate_tree(target_tree_node, model_node)

        tree = self.require_tree()
        refreshed_parent = tree.get_node_by_id(target_node_id) or tree.root
        refreshed_parent.expand()
        self._select_tree_node_without_toggle(refreshed_parent)
        tree.refresh(layout=True)
        tree.focus()
        self.show_status(f"Added child '{new_title}'.")

    def action_add_manual_child(self) -> None:
        if self._edit_state is not None:
            self.bell()
            self.show_status("Finish editing before adding a node.")
            return
        selected_tree_node = self.get_selected_tree_node()
        if selected_tree_node is None:
            self.bell()
            self.show_status("No node selected.")
            return

        target_tree_node, model_node = self._resolve_model_tree_node(selected_tree_node)
        if target_tree_node is None or model_node is None:
            self.bell()
            self.show_status("Cannot add a child to this entry.")
            return

        new_title = self._unique_child_title("New idea", model_node)
        new_node = MindmapNode(new_title)
        model_node.children.append(new_node)
        self.populate_tree(target_tree_node, model_node)

        tree = self.require_tree()
        target_tree_node.expand()
        tree.refresh(layout=True)

        new_tree_node = next(
            (child for child in target_tree_node.children if child.data is new_node), None
        )
        if new_tree_node is None:
            self._select_tree_node_without_toggle(target_tree_node)
            self.show_status("Added child node.")
            return

        tree.select_node(new_tree_node)
        tree.focus()
        self._start_inline_edit(new_tree_node, new_node.title, kind="title", context={"node": new_node})

    def _stop_generation(self) -> bool:
        handled = False
        if self._full_pending is not None:
            self._full_pending = None
            self.show_status("Full build cancelled.")
            handled = True
        elif self._full_in_progress:
            if not self._stop_after_current_step:
                self._stop_after_current_step = True
                self.show_status("Stopping after the current step completes…")
            else:
                self.show_status("Stop already requested; waiting for current step.")
            handled = True
        return handled

    def action_remove_child_suggestion(self) -> None:
        selected_tree_node = self.get_selected_tree_node()
        if selected_tree_node is None:
            self.bell()
            self.show_status("No node selected.")
            return

        target_tree_node, model_node = self._resolve_model_tree_node(selected_tree_node)
        if target_tree_node is None or model_node is None:
            self.bell()
            self.show_status("Cannot remove a child from this entry.")
            return

        if not model_node.children:
            self.bell()
            self.show_status("No child nodes to remove.")
            return

        tree = self.require_tree()
        parent_node_id = target_tree_node.id
        removed_child = model_node.children.pop()
        self.populate_tree(target_tree_node, model_node)

        refreshed_parent = tree.get_node_by_id(parent_node_id) or tree.root
        refreshed_parent.expand()
        self._select_tree_node_without_toggle(refreshed_parent)
        tree.refresh(layout=True)
        tree.focus()
        self.show_status(f"Removed child '{removed_child.title}'.")

    def action_full_generate(self) -> None:
        if self._edit_state is not None:
            self.bell()
            self.show_status("Finish editing before running a full build.")
            return
        if self._full_in_progress:
            self.bell()
            self.show_status("Full build already running.")
            return
        if self._full_pending is not None:
            self.bell()
            self.show_status("Waiting for depth digit (1-5) or Esc.")
            return

        selected_tree_node = self.get_selected_tree_node()
        if selected_tree_node is None:
            self.bell()
            self.show_status("No node selected.")
            return
        target_tree_node, model_node = self._resolve_model_tree_node(selected_tree_node)
        if target_tree_node is None or model_node is None:
            self.bell()
            self.show_status("Full build requires a node selection.")
            return

        self._full_pending = {
            "node_id": target_tree_node.id,
            "node_ref": target_tree_node,
            "include_body": False,
            "counts_input": "",
            "level_targets": None,
        }
        self._select_tree_node_without_toggle(target_tree_node)
        self.show_status(
            "Full build: type depth 1-5 or counts like '8,5'. T toggles text, Enter to run, Esc to cancel."
        )

    async def action_generate_children(self, count: int) -> None:
        selected_tree_node = self.get_selected_tree_node()
        if selected_tree_node is None or selected_tree_node.data is None:
            self.bell()
            self.show_status("No node selected.")
            return
        if count <= 0:
            self.show_status("Count must be positive.")
            return
        if not isinstance(selected_tree_node.data, MindmapNode):
            self.bell()
            self.show_status("Cannot generate children for this entry.")
            return
        model_node = selected_tree_node.data
        with self._prompt_log_session("Generate children"):
            generated_root = await self._request_structured_subtree(
                selected_tree_node,
                depth=1,
                spinner_label="generate child nodes",
                mode="exact",
                exact_children=count,
            )
        if generated_root is None:
            return
        if not generated_root.children:
            self.show_status("No child titles returned.")
            return
        self._apply_generated_subtree(
            selected_tree_node,
            model_node,
            generated_root,
            replace_children=True,
            update_body=False,
            expand_all=False,
        )
        child_level = self._tree_node_level(selected_tree_node) + 1
        self.show_status(
            f"Generated {len(generated_root.children)} child{'ren' if len(generated_root.children) != 1 else ''} at level {child_level}."
        )

    async def action_auto_generate_children(self) -> None:
        selected_tree_node = self.get_selected_tree_node()
        if selected_tree_node is None or selected_tree_node.data is None:
            self.bell()
            self.show_status("No node selected.")
            return
        if not isinstance(selected_tree_node.data, MindmapNode):
            self.bell()
            self.show_status("Cannot generate children for this entry.")
            return
        model_node = selected_tree_node.data
        with self._prompt_log_session("Auto-generate children"):
            generated_root = await self._request_structured_subtree(
                selected_tree_node,
                depth=1,
                spinner_label="generate child nodes",
                mode="auto",
            )
        if generated_root is None:
            return
        if not generated_root.children:
            self.show_status("No child titles returned.")
            return
        self._apply_generated_subtree(
            selected_tree_node,
            model_node,
            generated_root,
            replace_children=True,
            update_body=False,
            expand_all=False,
        )
        child_level = self._tree_node_level(selected_tree_node) + 1
        self.show_status(
            f"Generated {len(generated_root.children)} AI-selected child{'ren' if len(generated_root.children) != 1 else ''} at level {child_level}."
        )

    def action_expand_all(self) -> None:
        tree = self.require_tree()
        target = self.get_selected_tree_node() or tree.root
        self._apply_level_limit(0, anchor=target)

    def action_preview_markmap(self) -> None:
        try:
            path = self._write_markmap_preview()
        except Exception as exc:  # pragma: no cover - defensive
            self.bell()
            self.show_status(f"Markmap preview failed: {exc}")
            return
        self._markmap_preview_path = path
        self._open_markmap_preview(path)
        self.show_status("Opened Markmap preview in a browser tab.")

    def action_choose_model(self) -> None:
        if not self.model_choices:
            self.bell()
            self.show_status("No models configured.")
            return

        def apply_selection(selection: str | None) -> None:
            if not selection:
                self.show_status(f"Model unchanged ({self.selected_model}).")
                return
            if selection not in self.model_choices:
                self.model_choices.append(selection)
            self.model_choices.sort()
            if selection == self.selected_model:
                self.show_status(f"Model unchanged ({self.selected_model}).")
                return
            self.selected_model = selection
            ai.set_active_model(selection)
            self.show_status(f"Model set to {selection}.")

        self.push_screen(
            ModelSelectorScreen(self.model_choices, self.selected_model),
            apply_selection,
        )

    def action_edit_context(self) -> None:
        def apply_context(result: dict[str, str] | None) -> None:
            if not isinstance(result, dict):
                self.show_status("Context unchanged.")
                return
            action = result.get("action")
            if action == "apply":
                previous = self._external_context
                next_value = result.get("text", "")
                self._set_external_context(next_value)
                if self._has_external_context():
                    self.show_status("External context updated.")
                else:
                    self.show_status(
                        "External context cleared." if previous else "External context unchanged."
                    )
                return
            self.show_status("Context unchanged.")

        self.push_screen(ContextEditorScreen(self._external_context), apply_context)

    def action_level_prompt(self) -> None:
        if self._level_pending:
            self._level_pending = False
            self._level_anchor_id = None
            self.show_status("Level display unchanged.")
            return
        if self._full_pending is not None:
            self.bell()
            self.show_status("Finish pending requests before adjusting levels.")
            return
        anchor_node = self.get_selected_tree_node() or self.require_tree().root
        self._level_anchor_id = anchor_node.id
        self._level_pending = True
        if anchor_node is self.require_tree().root:
            max_depth = self._calculate_max_level(self.mindmap_root)
        else:
            data = anchor_node.data
            if isinstance(data, MindmapNode):
                max_depth = self._calculate_max_level(data)
            else:
                max_depth = self._calculate_max_level(self.mindmap_root)
        if max_depth < 1:
            self._level_pending = False
            self._level_anchor_id = None
            self.show_status("No additional levels.")
            return
        self.show_status(f"Press 1-{max_depth} for depth. Esc to cancel.")

    def action_import_web_context(self) -> None:
        def apply_url(result: dict[str, str] | None) -> None:
            if not isinstance(result, dict) or result.get("action") != "fetch":
                self.show_status("URL import cancelled.")
                return
            url_value = result.get("url", "").strip()
            if not url_value:
                self.show_status("URL import cancelled.")
                return
            asyncio.create_task(self._import_context_from_url(url_value))

        self.push_screen(URLImportScreen(self._context_url), apply_url)

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

    async def action_add_emoji(self) -> None:
        tree = self.require_tree()
        selected_tree_node = self.get_selected_tree_node()
        if selected_tree_node is None or selected_tree_node.data is None:
            self.bell()
            self.show_status("No node selected.")
            return
        data = selected_tree_node.data

        expand_state = self._capture_expand_state()

        async def apply_to_title(node: MindmapNode, tree_node: Tree.Node[MindmapNode]) -> None:
            base_text, _ = self._strip_leading_emoji(node.title)
            candidate = base_text or node.title.strip()
            if not candidate:
                self.show_status("Nothing to annotate.")
                return
            path = " > ".join(self._node_path(tree_node))
            emoji = await self._request_emoji(tree_node, candidate, path)
            if not emoji:
                self.show_status("Emoji unchanged.")
                return
            new_title = self._with_prefixed_emoji(node.title, emoji)
            if new_title == node.title:
                self.show_status("Emoji unchanged.")
                return
            node.title = new_title
            tree_node.set_label(self._format_node_label(node))
            tree.refresh(layout=True)
            self._select_tree_node_without_toggle(tree_node)
            tree.focus()
            self.show_status(f"Emoji set to {emoji}")

        async def apply_to_body_line(
            body_info: dict[str, object], tree_node: Tree.Node[MindmapNode]
        ) -> None:
            parent_node: MindmapNode = body_info["node"]  # type: ignore[assignment]
            line_index: int = body_info["index"]  # type: ignore[assignment]
            lines = self._body_lines(parent_node.body or "")
            if not lines or not (0 <= line_index < len(lines)):
                self.show_status("Nothing to annotate.")
                return
            current_line = lines[line_index]
            base_text, _ = self._strip_leading_emoji(current_line)
            candidate = base_text or current_line.strip()
            if not candidate:
                self.show_status("Nothing to annotate.")
                return
            parent_tree_node = tree_node.parent or tree.root
            path = " > ".join(self._node_path(parent_tree_node))
            spinner_node = parent_tree_node if isinstance(parent_tree_node.data, MindmapNode) else tree.root
            emoji = await self._request_emoji(spinner_node, candidate, f"{path} (text)")
            if not emoji:
                self.show_status("Emoji unchanged.")
                return
            new_line = self._with_prefixed_emoji(current_line, emoji)
            if new_line == current_line.strip():
                self.show_status("Emoji unchanged.")
                return
            self._update_body_line(parent_node, tree_node, line_index, new_line)
            self._restore_expand_state(expand_state)
            self.show_status(f"Emoji set to {emoji}")

        try:
            if isinstance(data, MindmapNode):
                await apply_to_title(data, selected_tree_node)
            elif isinstance(data, dict) and data.get("kind") == "body_line":
                await apply_to_body_line(data, selected_tree_node)
            else:
                self.bell()
                self.show_status("Cannot add emoji to this entry.")
        finally:
            self._restore_expand_state(expand_state)

    async def action_generate_body(self) -> None:
        selected_tree_node = self.get_selected_tree_node()
        if selected_tree_node is None or selected_tree_node.data is None:
            self.bell()
            self.show_status("No node selected.")
            return
        model_node = selected_tree_node.data
        if not isinstance(model_node, MindmapNode):
            self.bell()
            self.show_status("Cannot annotate this entry.")
            return
        if model_node.children:
            await self._generate_text_for_leaves(selected_tree_node)
            return
        with self._prompt_log_session("Generate body text"):
            await self._generate_body_for_node(selected_tree_node, model_node)

    async def on_event(self, event: events.Event) -> None:  # noqa: D401
        if isinstance(event, events.Key):
            if self._edit_state is not None:
                self._handle_edit_key(event)
                return
            key_name, _ = _key_name_and_modifiers(event.key)
            if self._level_pending:
                if key_name == "escape":
                    self._level_pending = False
                    self._level_anchor_id = None
                    self.show_status("Level display unchanged.")
                    event.stop()
                    return
                if key_name.isdigit():
                    anchor_node = self._resolve_level_anchor()
                    anchor_data = anchor_node.data if isinstance(anchor_node.data, MindmapNode) else None
                    max_depth = (
                        self._calculate_max_level(anchor_data)
                        if anchor_data
                        else self._calculate_max_level(self.mindmap_root)
                    )
                    if max_depth < 1:
                        self._level_pending = False
                        self._level_anchor_id = None
                        self.show_status("No additional levels to adjust.")
                        event.stop()
                        return
                    depth = int(key_name)
                    if depth < 1 or depth > max_depth:
                        self.bell()
                        self.show_status(f"Depth must be 1-{max_depth}. Esc to cancel.")
                        event.stop()
                        return
                    self._level_pending = False
                    self._apply_level_limit(depth, anchor=anchor_node)
                    self._level_anchor_id = None
                    event.stop()
                    return
                # consume other keys while waiting for level input
                event.stop()
                return
            if key_name == "escape":
                if self._stop_generation():
                    event.stop()
                    return
            if await self._handle_full_pending_key(event):
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

    def _default_mindmap_path(self) -> Path:
        return self._active_path or Path("mindmap.md")

    def _load_mindmap(self, path: Path) -> bool:
        target = path.expanduser()
        if not target.exists():
            self.bell()
            self.show_status(f"{target} not found.")
            return False
        try:
            markdown = target.read_text(encoding="utf-8")
            self.mindmap_root = from_markdown(markdown)
        except Exception as exc:  # pragma: no cover - defensive programming
            self.bell()
            self.show_status(f"Failed to load {target}: {exc}")
            return False
        self._active_path = target
        self.rebuild_tree()
        self.show_status(f"Loaded {target}")
        return True

    def action_save(self) -> None:
        path = self._default_mindmap_path().expanduser()
        markdown = to_markdown(self.mindmap_root)
        path.write_text(markdown, encoding="utf-8")
        self._active_path = path
        self.show_status(f"Saved to {path}")

    def action_open(self) -> None:
        self._load_mindmap(self._default_mindmap_path())

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

    def show_status(self, message: str | None = None, *, level_status: str | None = None) -> None:
        context_state = self._context_state_label()
        level_text = level_status or self._current_level_status()
        if message:
            composed = f"{level_text} · {message}"
        else:
            composed = level_text
        self.sub_title = f"{composed} | Model: {self.selected_model} | Context: {context_state}"

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
    initial_path = sys.argv[1] if len(sys.argv) > 1 else None
    MindmapApp(initial_path).run()
