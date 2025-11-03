# AI powered Mind Map creation in Terminal.

![m1ndm4p demo](m1ndm4p.gif)

## Overview
- Textual-based TUI that keeps `mindmap.md` as the single source of truth while you browse, edit, and enrich the tree in place.
- Nodes render as a classic outline: titles sit on bold rows, body text is wrapped into 40-character grey leaves for quick skim reading.
- AI helpers run through OpenRouter when an API key is present, but deterministic fallbacks keep the tool useful offline.

## Highlights
- **Model palette (`m`)**: swap between curated free OpenRouter models or the built-in “No AI (offline dummy output)” entry; the active choice is always in the status bar.
- **Incremental ideation (`+` / `-`)**: ask the AI for *one more* child or prune the last one to fine-tune a branch without regenerating everything.
- **Full build (`f` → digit)**: recursively populate a subtree `n` levels deep with AI-chosen children, keeping focus on the node that initiated the action.
- **Leaf narration (`f` → `t`)**: after expanding a branch, instantly request text for every leaf so you can watch paragraphs appear live.
- **Inline editing**: titles and wrapped body lines edit in situ with the familiar `Enter`/`Esc` flow; focus never jumps away.
- **Fast feedback**: consistent `.`, `..`, `...` spinners for background work and a status banner that reports every action.

## Run
1. Activate the virtualenv (`source .venv/bin/activate`).
2. Launch the interface: `python app.py`.
3. Keep `mindmap.md` in the project root so `o` can reload it and the tree can rehydrate automatically.

### Optional: Enable OpenRouter AI
- Create an account at [openrouter.ai](https://openrouter.ai/) and obtain an API key.
- Export credentials before starting the app  
  ```bash
  export OPENROUTER_API_KEY="sk-..."
  # optional: choose a different default model
  export OPENROUTER_MODEL="minimax/minimax-chat"
  ```
- No extra Python packages are required—the app ships with a lightweight HTTP client.
- With the key present, AI shortcuts hit OpenRouter; otherwise the tool falls back to predictable mock suggestions.

## Key Bindings
- Navigation: arrow keys move the cursor; `a` expands the entire tree; `q` quits.
- Generation: `1`–`9` add that many children; `?` lets the AI decide; `+` requests one more child; `-` removes the last child.
- Content: `t` writes body text for the selected node; `f` followed by a digit performs a full multi-level build; `f` followed by `t` writes text for every leaf under the current node.
- Editing: `e` toggles inline edit (`Enter` saves, `Esc` cancels); `0` clears the current node (body + children).
- Data management: `o` reloads `mindmap.md`; `s` saves; `m` opens the model selector palette (including the offline “No AI” toggle).

## Editing Behaviour
- Text lines are stored per node in `MindmapNode.body`; inline edits rebuild the wrapped lines while keeping focus on the edited entry.
- AI paragraphs target roughly 200 characters and expand their branch automatically so you can see the newly generated text immediately.
- Full builds always expand the active subtree and keep focus on the node that initiated the action.

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
