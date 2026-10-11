"""Decision records and the decision log."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

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
    ts: float | None = None  # stamped by DecisionLog.append via its clock

    @property
    def asked(self) -> bool:
        return self.human_outcome is not None


class DecisionLog:
    """In-memory list of :class:`DecisionRecord`, optionally mirrored to an append-only JSONL file.

    With ``path`` set, ``append`` writes one record line and ``mark_downstream``
    writes one small patch line (``{"_patch": {"run_id", "gate_id", "downstream_error"}}``);
    the file is never rewritten. :meth:`from_jsonl` replays patches on load, and
    :meth:`compact` rewrites the file once with patches merged in.
    An existing file at ``path`` is appended to, not loaded or truncated.

    ``clock`` stamps ``ts`` on records appended without one (default ``time.time``).
    Pass a fixed clock, e.g. ``clock=lambda: 0.0``, for byte-reproducible logs.
    """

    def __init__(self, path: str | Path | None = None, clock: Callable[[], float] = time.time):
        self.path = Path(path) if path else None
        self.clock = clock
        self.records: list[DecisionRecord] = []
        self._by_key: dict[tuple[str, str], list[DecisionRecord]] = {}

    def _write(self, obj: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _add(self, rec: DecisionRecord) -> None:
        self.records.append(rec)
        self._by_key.setdefault((rec.run_id, rec.gate_id), []).append(rec)

    def append(self, rec: DecisionRecord) -> None:
        """Add a record (stamping ``ts`` if unset). Use this rather than ``records.append``."""
        if rec.ts is None:
            rec.ts = self.clock()
        self._add(rec)
        if self.path:
            self._write(asdict(rec))

    def _patch(self, run_id: str, gate_id: str, error: bool) -> int:
        recs = self._by_key.get((run_id, gate_id), ())
        for r in recs:
            r.downstream_error = error
        return len(recs)

    def mark_downstream(self, run_id: str, gate_id: str, error: bool) -> int:
        """Set ``downstream_error`` on every record of (run_id, gate_id) so far. Returns how many matched."""
        n = self._patch(run_id, gate_id, error)
        if n and self.path:
            self._write({"_patch": {"run_id": run_id, "gate_id": gate_id, "downstream_error": error}})
        return n

    def compact(self) -> None:
        """Rewrite the bound file once as plain records with all patches merged."""
        if not self.path:
            raise ValueError("compact() needs a file-backed log (DecisionLog(path))")
        self.to_jsonl(self.path)

    def filter(self, gate_id: str | None = None) -> list[DecisionRecord]:
        return [r for r in self.records if gate_id is None or r.gate_id == gate_id]

    def to_jsonl(self, path: str | Path) -> None:
        """Write all records (patches merged, no patch lines) to ``path``, overwriting it."""
        with Path(path).open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "DecisionLog":
        """Load records from a JSONL file, replaying patch lines. The returned log is not bound to ``path``."""
        log = cls()
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "_patch" in obj:
                    p = obj["_patch"]
                    log._patch(p["run_id"], p["gate_id"], p["downstream_error"])
                else:
                    log._add(DecisionRecord(**obj))
        return log

    def __len__(self) -> int:
        return len(self.records)
