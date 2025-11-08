from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MindmapNode:
    title: str
    children: List["MindmapNode"] = field(default_factory=list)
    body: Optional[str] = None
