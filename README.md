# AI powered Mind Map creation in Terminal.

m1ndm4p is a Textual-based TUI for iterating on Markdown mind maps with optional OpenRouter-powered assistance. You edit nodes inline, call the model for suggestions, and keep full control over the `mindmap.md` file stored in the repo.

## TL;DR

- **Keyboard-first terminal UI** – zero mouse, no visual clutter, fast navigation.
- **Markdown storage** – plain text mind maps you can diff, version, or script.
- **Mind map workflow** – focus on the essentials, branch level by level.
- **One-shot subtree generation** – enter depth (`2`) or counts (`8,5,3`) after `f` and the AI emits the entire outline (titles + optional text) in one prompt.
- **Batch leaf text** – press `t` on any branch to annotate all of its leaves in one go, keeping tone consistent and token usage low.
- **Manual mode always available** – every edit works offline; use AI only when you need a boost.
- **Markmap preview** – hit `p` at any time to export the current tree as an interactive HTML mind map.

![m1ndm4p demo](m1ndm4p.gif)

## Prerequisites

- Python 3.11+
- `pip`
- *(Optional for AI)* `OPENROUTER_API_KEY`; without it the app falls back to deterministic offline output.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install textual rich
```

Clone/copy this repo into a writable directory. The app creates or updates `mindmap.md`, `prompt.log`, and `connection.log` in-place.

## Running

```bash
python app.py
```

The UI opens in your terminal, loads `mindmap.md` if present, or starts with a “Central Idea” root node.

To jump straight into another map, pass the Markdown path as the first argument—the app loads it on startup and keeps saving back to the same file:

```bash
python app.py path/to/brainstorm.md
```

When the UI opens it automatically expands the root plus two levels (root = level 0) so you can scan the outline immediately. Press `l` and enter a digit to focus on a shallower/deeper slice, or hit `a` to view the entire selected branch.

## Key bindings

| Key | Action |
| --- | --- |
| `q` | Quit |
| `s` | Save to `mindmap.md` |
| `o` | Load `mindmap.md` |
| `t` | Generate text for the focused leaf or entire branch (AI) |
| `e` | Inline edit node title/body line |
| `tab` | Add manual child + rename |
| `?` | Auto-generate 0–10 curated child nodes (AI) |
| `1`–`9` | Generate exactly *n* child nodes (AI) |
| `+` / `-` | Add/remove one AI suggestion |
| `w` | Import external context from URL |
| `i` | Edit external context buffer (Enter saves) |
| `l` + digit | Limit levels under the focused node (0 = all under that branch) |
| Arrow keys / `left` `right` | Navigate / expand / collapse |
| `Esc` | Cancel edits, level prompts, or stop AI runs |
| `a` | Expand only the focused branch (root = entire tree) |
| `:` | Ask AI for an emoji that prefixes the focused title/body line |
| `0` | Clear selected entry |
| `f` | Launch multi-level generation (type `2` or counts like `8,5,3`; `t` toggles text before Enter) |
| `m` | Choose OpenRouter model |
| `p` | Preview the tree as an interactive Markmap |

## Markmap integration

Press `p` (or select **Preview Markmap** from the footer hints) to regenerate `markmap_preview.html`. The app writes the current tree into that single HTML file, opens it in your default browser, and reuses the same file for future previews so you can refresh the tab to see updates. The output uses the [markmap](https://markmap.js.org/) library, so you get collapsible branches, zoom, and search without leaving the terminal workflow.
Branches you collapse inside the TUI are still embedded in the preview—they simply start folded, so you can expand them in the browser when you need the hidden details.

## Logs

- `prompt.log`: timestamped prompt/response pairs for every AI call
- `connection.log`: success/failure entries for each OpenRouter request

Both logs reset on startup.

## Best practices

1. **Design the skeleton first.** Sketch the top level by hand (`tab` + `e`), then hand the branch to AI so the outline stays aligned with your intent.
2. **Use structured full builds.** `f` + counts (e.g., `6,4,2`) gives you a predictable footprint. Rerun `f` on any branch to iterate rapidly.
3. **Layer context intentionally.** Paste external notes with `i`/`w` before triggering AI so subtree generations remain grounded in the facts you care about.
4. **Annotate in batches.** Press `t` on any branch to fill every leaf with a single batched request—faster, cheaper, and stylistically consistent.
5. **Audit as you go.** Collapse/expand views with `l` (per branch) and `a` so you can spot redundant headings or shallow spots before regenerating.
6. **Commit frequently.** `s` writes `mindmap.md`; `prompt.log` and `connection.log` are rotated on startup, so use them for learning but rely on Git for lasting history.

## Troubleshooting

- **AI unavailable?** Without `OPENROUTER_API_KEY`, the app falls back to deterministic dummy output; set the key or switch models via `m`.
- **Structure feels flat?** Rerun `f` with explicit counts (e.g., `8,5`) or toggle text mode (`t`) to force deeper groupings and accompanying summaries.
- **Text landed on the wrong level?** Remember sentences only appear on leaves—delete or add children, then hit `t` again to refresh only the necessary nodes.
- **Prompt budget exploding?** Batch leaf generation (`t` on a branch) and leverage full builds instead of per-node requests; logs (`prompt.log`) show exactly what ran.
- **UI quirks?** Ensure you’re inside a Unicode-capable terminal and that `textual`/`rich` are installed in the active virtual environment.

Launch `python app.py`, keep iterating on `mindmap.md`, and let AI assist without taking control.
