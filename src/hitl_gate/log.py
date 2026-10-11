"""Decision records and the decision log."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

OUTCOMES = ("approve", "edit", "retry", "escalate")


@dataclass
class DecisionRecord:
    """One gate decision and (once known) its outcome.

    ``decision`` is what the *policy* decided. Whether a human was actually
    asked is ``human_outcome is not None``: in ``shadow`` mode the human is
    always asked, in ``audit`` mode an auto decision was sampled for review.
    """

    run_id: str
    gate_id: str
    features: dict
    decision: Literal["ask", "auto"]
    policy_name: str
    reason: str
    mode: Literal["live", "shadow", "audit"] = "live"
    human_outcome: str | None = None  # "approve" | "edit" | "retry" | "escalate" | None
    changed: bool | None = None  # human changed the result (outcome != "approve")
    review_seconds: float | None = None
    downstream_error: bool | None = None  # filled later via DecisionLog.mark_downstream
    ts: float = field(default_factory=time.time)

    @property
    def asked(self) -> bool:
        return self.human_outcome is not None


class DecisionLog:
    """In-memory list of :class:`DecisionRecord`, optionally mirrored to a JSONL file.

    With ``path`` set, every ``append`` writes one line and ``mark_downstream``
    rewrites the file so it stays in sync.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.records: list[DecisionRecord] = []

    def append(self, rec: DecisionRecord) -> None:
        self.records.append(rec)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def mark_downstream(self, run_id: str, gate_id: str, error: bool) -> int:
        """Set ``downstream_error`` on every record of (run_id, gate_id). Returns how many matched."""
        n = 0
        for r in self.records:
            if r.run_id == run_id and r.gate_id == gate_id:
                r.downstream_error = error
                n += 1
        if n and self.path:
            # ponytail: full rewrite, O(log size) per call; batch marks or use a DB for huge logs
            self.to_jsonl(self.path)
        return n

    def filter(self, gate_id: str | None = None) -> list[DecisionRecord]:
        return [r for r in self.records if gate_id is None or r.gate_id == gate_id]

    def to_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "DecisionLog":
        """Load records from a JSONL file. The returned log is not bound to ``path``."""
        log = cls()
        with Path(path).open(encoding="utf-8") as f:
            log.records = [DecisionRecord(**json.loads(line)) for line in f if line.strip()]
        return log

    def __len__(self) -> int:
        return len(self.records)
