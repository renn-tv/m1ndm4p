# m1ndm4p

## Overview
- Terminal mind‑mapping tool built on [Textual](https://textual.textualize.io/); renders Markdown-backed trees and keeps the file (`mindmap.md`) as the single source of truth.
- Node titles are rendered plainly; body text is split into 40-character lines and shown as indented grey children. Markdown export preserves the original paragraphs.
- AI hooks (`ai.generate_children`, `ai.generate_paragraph`) are stubbed to local helpers when API access is unavailable; workflow still functions with deterministic mock text.

## Features
- Single-source Markdown mind map that stays in sync with the TUI.
- Inline editing for node titles and wrapped text lines (no modal popups).
- AI-assisted idea generation via OpenRouter (child nodes + body text). Auto mode (`?`) lets the model pick a sensible number of children.
- Instant `.`, `..`, `...` spinner feedback during remote calls.
- Keyboard-only workflow with fast child generation (`1`–`9`), text creation (`t`), editing (`e`), and clearing (`0`).

## Run
1. Activate the virtualenv (`source .venv/bin/activate`).
2. Launch the TUI: `python app.py`.
3. Keep `mindmap.md` in the project root—`o` reloads it and expands the whole tree automatically.

### Optional: Enable OpenRouter AI
- Create an account at [openrouter.ai](https://openrouter.ai/) and obtain an API key.
- Export the key before launching the app  
  ```bash
  export OPENROUTER_API_KEY="sk-..."
  # optional model override
  export OPENROUTER_MODEL="minimax/minimax-chat"
  ```
- Install `requests` in the virtualenv (`pip install requests`).
- With the key present, `1`–`9` and `t` call OpenRouter; without it, the app falls back to deterministic mock text.

## Key Bindings
- `1`–`9`: generate child nodes (count = key).
- `?`: let the AI pick a sensible number of child nodes.
- `t`: generate placeholder body text for the selected node.
- `e`: inline edit (node titles or text lines). Existing text highlights; typing replaces it.  
  - `Enter` saves, staying on the row.  
  - `Esc` cancels.
- `0`: delete the selected node (except the root).
- `a`: expand the entire tree.
- `o`: reload `mindmap.md`.
- `s`: save current tree back to `mindmap.md`.
- Arrow keys navigate; `q` quits.

## Editing Behaviour
- Text lines are stored per node in `MindmapNode.body`; inline edits rebuild the wrapped lines while keeping focus on the edited entry.
- AI paragraphs target roughly 200 characters, but the full response is kept (wrapped at 40 chars for display) so longer context is preserved in Markdown.

## Markdown Format
```
# Root title
## Child title
Paragraph text…
```
- Headings define structure; following paragraphs become body text.
- When reloaded, paragraphs are split and displayed as the grey child rows under their parent heading.

## Notes for Development / AI Agents
- Core logic lives in `app.py`. Helpers for Markdown serialisation are in `md_io.py`; `models.py` houses the simple `MindmapNode` dataclass.
- The TUI expects synchronous filesystem access; avoid blocking network calls in the event loop unless wrapped in async tasks.
- When extending editing features, reuse `_start_inline_edit` and `_commit_inline_edit` to keep state consistent.
