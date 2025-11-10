# m1ndm4p

AI Powered Mind Map Creation in Terminal.

![m1ndm4p demo](./m1ndm4p.gif)

m1ndm4p turns any structured Markdown file into a live, keyboard-driven mind map – and back again.  
You can mix fast manual editing with AI-assisted expansion, plug in simple RAG-style context, and export an interactive Markmap for presentation.

## Key Features

- AI-assisted outlining
  - Generate children, subtrees, body text, emojis, and suggestions via OpenRouter.
  - Deterministic offline mode available for zero-network environments.
- Markdown-native
  - Load existing structured Markdown (headings + lists) into a Mindmap tree.
  - Save back to a strict, canonical Markdown format for stable round-trips and diffs.
- Terminal UI (Python + Textual)
  - Compact, focused interface built for flow and keyboard use.
  - Inline body text rendering; minimal chrome.
- Strong keyboard workflow
  - Tab / n for children & siblings, e to edit, - / 0 to delete, a to expand, l for depth.
  - h shows an in-app keybinding cheat sheet; footer surfaces key shortcuts.
- Interactive Markmap export
  - p generates Markmap HTML from the current map and opens it in your browser.
  - Uses layout and styling tuned to match the TUI.
- Simple RAG-style context
  - i to edit a lightweight context block.
  - w to fetch and strip text from a URL.
  - Context influences AI generations without complex setup.
- OpenRouter integration
  - m opens a model picker.
  - Uses OpenRouter’s /chat/completions API with curated defaults.
  - Just set `OPENROUTER_API_KEY` and choose a model.

## Canonical Markdown Format

m1ndm4p enforces a predictable structure on save:

- The in-app MindmapNode tree is the single source of truth.
- Export rule per parent level:
  - If any child at that level has children:
    - All siblings at that level are written as headings (`#`…`######`).
  - If all children at that level are leaves:
    - All siblings at that level are written as bullets with exactly two spaces:
      - `  * Title`
      - or `  1. Title` etc. for ordered lists.
- This avoids mixed “heading + peer bullet” artifacts and normalizes imported Markdown into a clean, stable format.

Load structured Markdown → explore & extend in the TUI → save back as canonical Markdown.

## Quick Start

### 1. Install

Requirements:

- Python 3.10+
- Recommended dependencies:
  - `textual`
  - `rich`

Install:

```bash
pip install textual rich
```

For AI features (optional):

- Set `OPENROUTER_API_KEY` in your environment.
- Optionally set `OPENROUTER_MODEL` (or choose via the in-app model picker).

### 2. Run

From the repo:

```bash
# Start with an empty map
python app.py

# Or load an existing Markdown mind map
python app.py Mindmaps/your_map.md
```

By default, save writes to `mindmap.md` in the repo root (or back to the opened file).

## Core Keyboard Shortcuts

- Navigation:
  - Arrow keys: move selection
  - Space: expand / collapse
  - a: expand all from current node
  - l: set visible depth (level limit)
- Editing:
  - e: edit title / text
  - tab: add child node
  - n: add sibling node
  - -: delete node + subtree
  - 0: clear children/body under current node (keep node)
- AI:
  - 1–9: generate that many child nodes
  - ?: auto-select how many child nodes to add
  - +: add one more AI suggestion child under current node
  - f: full multi-level AI build (guided counts)
  - t: generate body text
  - : (colon): suggest an emoji prefix
- Context & Web:
  - i: edit simple text context (RAG-style)
  - w: fetch and strip text from a URL into context
- Preview & Models:
  - p: open Markmap HTML preview in browser
  - m: choose OpenRouter model
- Meta:
  - s: save current mind map
  - o: open `mindmap.md`
  - h: help / keybinding overview
  - q: quit

## Architecture Overview

- `node_models.py`
  - Defines `MindmapNode` (title, body, children, kind, list_marker).
  - Serves as the structural backbone between TUI, parser, and exporter.
- `md_io.py`
  - Parses headings + lists into a `MindmapNode` tree (`from_markdown`).
  - Serializes the tree to canonical Markdown (`to_markdown`) using the rules above.
- `ai.py`
  - Wraps OpenRouter calls and offline fallbacks.
  - Structured prompts for subtree generation, leaf bodies, emojis.
  - Logs prompts/responses to `prompt.log` and `connection.log`.
- `app.py`
  - Textual-based `MindmapApp`.
  - Manages tree rendering, keyboard interactions, AI workflows, context, and preview.
  - Ensures MindmapNode stays normalized so save/export is stable.

## Why use m1ndm4p?

- Think and design at the speed of the keyboard.
- Start from scratch or from existing notes; keep everything as Markdown.
- Explore new knowledge - type a subject in the root and let AI lay it out.
- Let AI propose structure and text, but keep full control and visibility.
- Present your maps interactively with Markmap.
- Use simple text/URL context to guide AI to your domain.

Map out your own ideas or digest new content in an easy, inspectable, version-control-friendly way.
