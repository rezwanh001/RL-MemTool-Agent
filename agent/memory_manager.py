# agent/memory_manager.py
"""
MemoryManager implements a fixed-token memory with priority eviction.

Args:
    budget_tokens (int): max tokens to keep.
    priority_keys (List[str]): keys that get higher retention priority.
    tokenizer: HF tokenizer for token counting.

Methods:
    add(note: dict) -> None
    snapshot() -> str       # serialize memory to a compact string
    evict() -> None         # evict low-priority notes until under budget
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Note:
    kind: str
    text: str
    priority: int = 0

@dataclass
class MemoryManager:
    budget_tokens: int
    tokenizer: object = None
    priority_keys: List[str] = field(default_factory=list)
    notes: List[Note] = field(default_factory=list)

    def _toklen(self, s: str) -> int:
        return len(self.tokenizer.encode(s))

    def add(self, kind: str, text: str):
        prio = 2 if kind.lower() in [k.lower() for k in self.priority_keys] else 1
        self.notes.append(Note(kind=kind, text=text, priority=prio))
        self.evict()

    def total_tokens(self) -> int:
        return sum(self._toklen(n.text) for n in self.notes)

    def evict(self):
        while self.total_tokens() > self.budget_tokens and self.notes:
            # Evict lowest priority, then FIFO among equals
            min_prio = min(n.priority for n in self.notes)
            for i, n in enumerate(self.notes):
                if n.priority == min_prio:
                    self.notes.pop(i)
                    break

    def snapshot(self) -> str:
        lines = []
        for n in self.notes:
            lines.append(f"{n.kind.upper()}: {n.text}")
        return "\n".join(lines)
