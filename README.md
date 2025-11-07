# AI powered Mind Map creation in Terminal.

m1ndm4p is a Textual-based TUI for iterating on Markdown mind maps with optional OpenRouter-powered assistance. You edit nodes inline, call the model for suggestions, and keep full control over the `mindmap.md` file stored in the repo.

## TL;DR

- **Keyboard-first terminal UI** – zero mouse, no visual clutter, fast navigation.
- **Markdown storage** – plain text mind maps you can diff, version, or script.
- **Mind map workflow** – focus on the essentials, branch level by level.
- **AI assistant when you want it** – compare OpenRouter models, auto-generate concise nodes or body text.
- **Manual mode always available** – every edit works offline; use AI only when you need a boost.

![m1ndm4p demo](https://github.com/renn-tv/m1ndm4p/blob/main/m1ndm4p.gif)

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

## Key bindings

| Key | Action |
| --- | --- |
| `q` | Quit |
| `s` | Save to `mindmap.md` |
| `o` | Load `mindmap.md` |
| `t` | Generate short body text (AI) |
| `e` | Inline edit node title/body line |
| `tab` | Add manual child + rename |
| `?` | Auto-generate 0–10 curated child nodes (AI) |
| `1`–`9` | Generate exactly *n* child nodes (AI) |
| `+` / `-` | Add/remove one AI suggestion |
| `w` | Import external context from URL |
| `i` | Edit external context buffer (Enter saves) |
| `l` + digit | Show only that many levels (0 = all) |
| Arrow keys / `left` `right` | Navigate / expand / collapse |
| `Esc` | Cancel edits, level prompts, or stop AI runs |
| `a` | Expand entire tree |
| `0` | Clear selected entry |
| `f` | Launch multi-level generation workflow |
| `m` | Choose OpenRouter model |

## Logs

- `prompt.log`: timestamped prompt/response pairs for every AI call
- `connection.log`: success/failure entries for each OpenRouter request

Both logs reset on startup.

## Best practices

1. **Shape the top level manually** first; use `tab` + `e` before involving AI.
2. **Provide concise context** via `i` or `w` so the model knows what’s already known.
3. **Keep headings short** (single words or ≤25 characters) even after AI generation.
4. **Iterate level by level**—prune duplicates, then re-run `?`. Press `Esc` to cancel long AI runs.
5. **Limit the view** with `l` + digit when presenting or focusing on a subset.
6. **Save frequently** with `s`; `mindmap.md` is the single source of truth.

## Troubleshooting

- **No API key?** You’ll still get dummy output; set `OPENROUTER_API_KEY` for live completions.
- **Too many redundant nodes?** Delete overlaps, refine context, and regenerate.
- **Rendering issues?** Use a Unicode-capable terminal and ensure the virtual environment has `textual` installed.

Launch `python app.py`, keep iterating on `mindmap.md`, and let AI assist without taking control.
