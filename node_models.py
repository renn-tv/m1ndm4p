from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class MindmapNode:
    title: str
    children: List["MindmapNode"] = field(default_factory=list)
    body: Optional[str] = None
    # Optional: original list marker for serialization, e.g. "-", "*", "1.", "2)"
    list_marker: Optional[str] = None
    # Distinguish between heading-style and list-style nodes so we can
    # round-trip Markdown without promoting list items to headings.
    kind: Literal["heading", "list"] = "heading"
