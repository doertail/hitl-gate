"""Policies decide, per item, whether to ask a human (``True``) or proceed automatically (``False``)."""

from __future__ import annotations

import bisect
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Literal, Protocol, Union, runtime_checkable

from .log import DecisionRecord


@runtime_checkable
class Policy(Protocol):
    name: str

    def decide(self, gate_id: str, features: dict) -> tuple[bool, str]:
        """Return ``(ask, reason)``."""
        ...


class AlwaysAsk:
    """Today's default: interrupt at every review point."""

    name = "always_ask"

    def decide(self, gate_id: str, features: dict) -> tuple[bool, str]:
        return True, "always ask"


class NeverAsk:
    """No human in the loop. Useful as a baseline, rarely as a policy."""

    name = "never_ask"

    def decide(self, gate_id: str, features: dict) -> tuple[bool, str]:
        return False, "never ask"


def _bucket(gate_id: str, features: dict, keys: Iterable[str]) -> tuple:
    # keys a gate doesn't produce are left out (not stored as None), so per-gate rules stay readable
    return (gate_id, tuple((k, features[k]) for k in keys if k in features))


class RulePolicy:
    """Evidence-based allow-list: auto only for buckets humans demonstrably never needed to fix.

    Build with :meth:`fit`. A bucket is ``(gate_id, values of the selected feature keys
    that this item's features contain)``, so gates with different feature sets can share ``keys``;
    feature values must be hashable and discrete (bucketize continuous values in ``featurize``).
    Anything unseen or under-sampled is asked.
    """

    name = "rule"

    def __init__(self, keys: list[str], stats: dict, min_count: int, max_change_rate: float):
        self.keys = keys
        self.stats = stats  # bucket -> [n, misses]
        self.min_count = min_count
        self.max_change_rate = max_change_rate

    @classmethod
    def fit(
        cls,
        records: Iterable[DecisionRecord],
        keys: list[str] | None = None,
        min_count: int = 30,
        max_change_rate: float = 0.02,
        count_downstream_errors: bool = True,
    ) -> "RulePolicy":
        """Fit from logged decisions.

        Samples are records where a human was asked (live asks, shadow, audit).
        A *miss* is a sample where the human changed the result, or — with
        ``count_downstream_errors`` — where the human approved but the item later
        failed downstream (rubber-stamp protection). Auto-passed records with
        ``downstream_error=True`` are also counted as misses, so a bucket that
        starts failing in production rolls back to "ask" on the next fit.
        """
        records = list(records)
        if keys is None:
            keys = sorted({k for r in records for k in r.features})
        stats: dict = defaultdict(lambda: [0, 0])
        for r in records:
            downstream = count_downstream_errors and bool(r.downstream_error)
            if r.asked:
                miss = bool(r.changed) or downstream
            elif downstream:
                miss = True
            else:
                continue  # an un-reviewed auto pass is not evidence of correctness
            s = stats[_bucket(r.gate_id, r.features, keys)]
            s[0] += 1
            s[1] += miss
        return cls(keys, dict(stats), min_count, max_change_rate)

    def _is_auto(self, n: int, misses: int) -> bool:
        return n >= self.min_count and misses / n <= self.max_change_rate

    def decide(self, gate_id: str, features: dict) -> tuple[bool, str]:
        n, misses = self.stats.get(_bucket(gate_id, features, self.keys), (0, 0))
        if n < self.min_count:
            return True, f"insufficient evidence (n={n} < {self.min_count})"
        rate = misses / n
        if rate > self.max_change_rate:
            return True, f"miss rate {rate:.1%} > {self.max_change_rate:.1%} (n={n})"
        return False, f"miss rate {rate:.1%} <= {self.max_change_rate:.1%} (n={n})"

    def describe(self) -> str:
        """Human-readable rule list, one line per bucket."""
        lines = []
        for (gate_id, kv), (n, m) in sorted(self.stats.items(), key=lambda x: repr(x[0])):
            cond = ", ".join(f"{k}={v!r}" for k, v in kv) or "*"
            verdict = "AUTO" if self._is_auto(n, m) else "ask"
            lines.append(f"{verdict:<4}  [{gate_id}] {cond}  (n={n}, misses={m}, rate={m / n:.1%})")
        lines.append(f"{'ask':<4}  anything else (unseen, or n < {self.min_count})")
        return "\n".join(lines)


Alpha = Union[float, Literal["visit"], Callable[[int], float]]


class QTablePolicy:
    """Tabular Q-learning over ``(gate_id, discretized features)`` with actions ``ask`` / ``auto``.

    Rewards are yours to define; a typical choice is ``-review_cost`` when asking
    and ``-error_cost`` when an error slips through. ``decide`` is greedy and
    breaks ties (including never-seen states, where both Q are 0) toward ``ask``.

    ``bins`` maps a feature name to sorted cut points; that feature is replaced by
    its bin index. Other features are used as-is and must be JSON-serializable.

    ``update`` / ``learn_episode`` take ``alpha`` (learning rate) as:

    * a float: fixed rate (the 0.1.0 behaviour);
    * ``"visit"``: ``1/n`` where ``n`` counts updates of that ``(state, action)``,
      so Q is the running mean of its targets. Robust to rare, large costs;
    * a callable ``alpha(n) -> float``, e.g. ``lambda n: max(1 / n, 0.001)`` for
      ``1/n`` with a floor (keeps adapting once ``n`` is large).

    Visit counts (``self.n``) are tracked for every update and saved with the table.
    """

    ACTIONS = ("ask", "auto")

    def __init__(self, keys: list[str] | None = None, bins: dict[str, list[float]] | None = None, name: str = "qtable"):
        self.keys = keys
        self.bins = bins or {}
        self.name = name
        self.q: dict[str, dict[str, float]] = {}
        self.n: dict[str, dict[str, int]] = {}  # visit counts per (key, action)

    def key(self, gate_id: str, features: dict) -> str:
        keys = self.keys if self.keys is not None else sorted(features)
        vals = []
        for k in keys:
            v = features.get(k)
            if k in self.bins and v is not None:
                v = bisect.bisect_right(self.bins[k], v)
            vals.append([k, v])
        return json.dumps([gate_id, vals])

    def _q(self, key: str) -> dict[str, float]:
        return self.q.setdefault(key, {"ask": 0.0, "auto": 0.0})

    def decide(self, gate_id: str, features: dict) -> tuple[bool, str]:
        q = self.q.get(self.key(gate_id, features), {"ask": 0.0, "auto": 0.0})
        ask = q["ask"] >= q["auto"]
        return ask, f"Q(ask)={q['ask']:.3f} Q(auto)={q['auto']:.3f}"

    def act(self, gate_id: str, features: dict, epsilon: float, rng: random.Random) -> bool:
        """Epsilon-greedy action for training. Returns ``ask``."""
        if rng.random() < epsilon:
            return rng.random() < 0.5
        return self.decide(gate_id, features)[0]

    def update(self, key: str, action: str, target: float, alpha: Alpha = 0.1) -> None:
        q = self._q(key)
        counts = self.n.setdefault(key, {"ask": 0, "auto": 0})
        counts[action] += 1
        n = counts[action]
        if alpha == "visit":
            a = 1.0 / n
        elif callable(alpha):
            a = alpha(n)
        elif isinstance(alpha, (int, float)):
            a = alpha
        else:
            raise ValueError(f"alpha must be a float, 'visit' or a callable, got {alpha!r}")
        q[action] += a * (target - q[action])

    def learn_episode(self, transitions: list[tuple], alpha: Alpha = 0.1, gamma: float = 1.0) -> None:
        """One Q-learning pass over an episode.

        Each transition is ``(gate_id, features, action, reward, next_gate_id, next_features)``
        where ``action`` is ``"ask"``/``"auto"`` (or a bool meaning ask) and the
        ``next_*`` pair is ``None`` at the end of the episode.
        """
        for gate_id, feats, action, reward, ngate, nfeats in transitions:
            if isinstance(action, bool):
                action = "ask" if action else "auto"
            target = reward
            if ngate is not None:
                target += gamma * max(self._q(self.key(ngate, nfeats)).values())
            self.update(self.key(gate_id, feats), action, target, alpha)

    def save(self, path: str | Path) -> None:
        data = {"name": self.name, "keys": self.keys, "bins": self.bins, "q": self.q, "n": self.n}
        Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "QTablePolicy":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        p = cls(keys=data["keys"], bins=data["bins"], name=data["name"])
        p.q = data["q"]
        p.n = data.get("n", {})  # files from 0.1.0 have no counts
        return p
