"""Simulated 3-step document pipeline: extract -> classify -> compose -> publish.

Every step can silently introduce an error. How likely that is depends on
signals the agent can observe (how many fields changed, whether sources
conflicted, whether a validator passed). A human reviewer catches 90% of the
errors they are shown (the other 10% are rubber-stamped) and spends time on
every review.

Phases
  1. Collect decision logs with AlwaysAsk (500 runs through the real graph).
  2. Fit a RulePolicy on those logs; train a QTablePolicy on a fast pure-Python
     mirror of the *same* dynamics (same sampling functions, no LangGraph), since
     Q-learning needs tens of thousands of episodes.
  3. Evaluate every policy on the same 1,000 held-out runs through the real graph.

All numbers are from this simulation and depend on its assumptions.
Run:  python examples/document_pipeline/run.py   (writes docs/tradeoff.png)
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from hitl_gate import AlwaysAsk, DecisionLog, LogisticPolicy, NeverAsk, QTablePolicy, RulePolicy, add_gated_review, merge_dicts
from hitl_gate.sim import run_episode

STEPS = ["extract", "classify", "compose"]
REVIEW_COST = 1.0  # cost unit = one human review
MISS_COST = 25.0  # an error reaching publish costs as much as 25 reviews
CATCH_RATE = 0.9
BASE_ERROR = {"extract": 0.004, "classify": 0.003, "compose": 0.005}
FIELDS_MULT = {"0": 1, "1-3": 4, "4+": 12}

# ---------------------------------------------------------------- dynamics (shared by graph and mirror)


def error_prob(step: str, f: dict) -> float:
    p = BASE_ERROR[step] * FIELDS_MULT[f["n_fields_changed"]]
    p *= 5 if f["source_conflict"] else 1
    p *= 1 if f["validator_passed"] else 6
    return min(p, 0.6)


def make_scenario(seed: int) -> dict:
    """Everything random about one run, drawn up front so every policy sees identical runs."""
    rng = random.Random(seed)
    sc = {}
    for step in STEPS:
        f = {
            "n_fields_changed": rng.choices(["0", "1-3", "4+"], [0.5, 0.35, 0.15])[0],
            "source_conflict": rng.random() < 0.15,
            "validator_passed": rng.random() < 0.8,
        }
        sc[step] = {
            "features": f,
            "has_error": rng.random() < error_prob(step, f),
            "caught": rng.random() < CATCH_RATE,
            "seconds": 20 + rng.expovariate(1 / 40),
        }
    return sc


def make_reviewer(scenario: dict):
    def reviewer(interrupt_value: dict) -> dict:
        s = scenario[interrupt_value["gate_id"]]
        if s["has_error"] and s["caught"]:
            return {"outcome": "edit", "changes": {"fixed": True}, "review_seconds": s["seconds"] * 2}
        return {"outcome": "approve", "review_seconds": s["seconds"]}

    return reviewer


# ---------------------------------------------------------------- the real LangGraph pipeline


class DocState(TypedDict, total=False):
    run_id: str
    hitl_decisions: Annotated[dict, merge_dicts]
    scenario: dict  # hidden ground truth; nodes "do work" by reading it
    signals: Annotated[dict, merge_dicts]  # what the agent can observe per step
    errors: Annotated[dict, merge_dicts]  # step -> error currently present
    published: bool


def build_graph(policy, log: DecisionLog, audit_rate: float = 0.0, seed: int = 0):
    b = StateGraph(DocState)

    def work(step):
        def node(state):
            s = state["scenario"][step]
            return {"signals": {step: s["features"]}, "errors": {step: s["has_error"]}}

        return node

    for step in STEPS:
        b.add_node(step, work(step))
    b.add_node("publish", lambda s: {"published": True})
    b.add_edge(START, "extract")
    for step, nxt in zip(STEPS, STEPS[1:] + ["publish"]):
        add_gated_review(
            b, step, source=step, target=nxt, policy=policy, log=log,
            featurize=lambda s, step=step: s["signals"][step],
            make_review_payload=lambda s, step=step: {"step": step, "signals": s["signals"][step]},
            apply_answer=lambda s, a, step=step: {} if a["outcome"] == "approve" else {"errors": {step: False}},
            audit_rate=audit_rate, seed=seed,
        )
    b.add_edge("publish", END)
    return b.compile(checkpointer=MemorySaver())


def evaluate(name: str, policy, seeds: list[int], audit_rate: float = 0.0) -> dict:
    log = DecisionLog()
    graph = build_graph(policy, log, audit_rate=audit_rate)
    missed = 0
    for seed in seeds:
        sc = make_scenario(seed)
        run_id = f"{name}-{seed}"
        final = run_episode(graph, {"run_id": run_id, "scenario": sc}, make_reviewer(sc), thread_id=run_id)
        for step in STEPS:
            log.mark_downstream(run_id, step, final["errors"][step])
            missed += final["errors"][step]
    asked = [r for r in log.records if r.asked]
    n = len(seeds)
    reviews, misses = len(asked) / n, missed / n
    return {
        "name": name,
        "reviews": reviews,
        "missed": misses,
        "minutes": sum(r.review_seconds for r in asked) / 60 / n,
        "cost": reviews * REVIEW_COST + misses * MISS_COST,
        "log": log,
    }


# ---------------------------------------------------------------- fast mirror for Q-learning


def train_qtable(episodes: int, seed: int) -> QTablePolicy:
    q = QTablePolicy()
    rng = random.Random(seed)
    for ep in range(episodes):
        sc = make_scenario(10_000_000 + ep)
        trans = []
        for i, step in enumerate(STEPS):
            s = sc[step]
            ask = q.act(step, s["features"], epsilon=0.2, rng=rng)
            missed = s["has_error"] and not (ask and s["caught"])
            reward = -(REVIEW_COST if ask else 0.0) - (MISS_COST if missed else 0.0)
            nxt = STEPS[i + 1] if i + 1 < len(STEPS) else None
            trans.append((step, s["features"], ask, reward, nxt, sc[nxt]["features"] if nxt else None))
        q.learn_episode(trans, alpha=0.02, gamma=1.0)
    return q


# ---------------------------------------------------------------- plot


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    color = {"AlwaysAsk": ink2, "NeverAsk": ink2, "RulePolicy": "#2a78d6", "RulePolicy + 5% audit": "#2a78d6", "QTablePolicy": "#eb6834", "LogisticPolicy": "#1b9e77"}
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    # label offsets (points) and alignment, hand-placed to avoid collisions
    place = {"AlwaysAsk": (-10, 8, "right"), "NeverAsk": (10, -6, "left"), "RulePolicy": (-6, -12, "right"),
             "RulePolicy + 5% audit": (10, -12, "left"), "QTablePolicy": (8, 12, "left"), "LogisticPolicy": (-10, -14, "right")}
    for r in rows:
        hollow = "audit" in r["name"]
        y = r["missed"] * 100
        ax.scatter(r["reviews"], y, s=90, zorder=3, linewidths=2,
                   facecolors="none" if hollow else color[r["name"]], edgecolors=color[r["name"]])
        dx, dy, ha = place[r["name"]]
        ax.annotate(f"{r['name']}\ncost {r['cost']:.2f}/run", (r["reviews"], y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8.5, color=ink, ha=ha, va="top" if dy < 0 else "bottom")
    ax.set_xlabel("Human reviews per run  (lower = less human time)", color=ink2)
    ax.set_ylabel("Errors reaching publish per 100 runs", color=ink2)
    ax.set_title("How many reviews can you skip, and what does it cost? (simulated)", color=ink, loc="left", fontsize=11)
    ax.set_xlim(-0.3, 3.6)
    ax.set_ylim(-0.5, max(r["missed"] for r in rows) * 100 * 1.12)
    ax.grid(color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=ink2)
    fig.text(0.01, 0.01, f"Total cost = reviews x {REVIEW_COST:g} + missed errors x {MISS_COST:g}. Same 1,000 held-out runs for every policy.",
             fontsize=7.5, color=ink2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=surface)


# ---------------------------------------------------------------- main


def main(n_collect: int = 500, n_eval: int = 1000, q_episodes: int = 60_000) -> list[dict]:
    t0 = time.time()
    print(f"phase 1: collecting logs with AlwaysAsk ({n_collect} runs through LangGraph)...")
    collected = evaluate("collect", AlwaysAsk(), list(range(n_collect)))["log"]

    rule = RulePolicy.fit(collected.records, min_count=30, max_change_rate=0.02)
    print("\nfitted RulePolicy (min_count=30, max_change_rate=2%):")
    print(rule.describe())

    logit = LogisticPolicy.fit(collected.records, threshold=REVIEW_COST / MISS_COST)
    print(f"fitted LogisticPolicy (threshold = review/miss cost = {REVIEW_COST / MISS_COST:.0%})")

    print(f"\nphase 2: training QTablePolicy on {q_episodes} mirror episodes...")
    qpol = train_qtable(q_episodes, seed=1)

    print(f"phase 3: evaluating on {n_eval} held-out runs through LangGraph...")
    held_out = list(range(1_000_000, 1_000_000 + n_eval))
    rows = [
        evaluate("AlwaysAsk", AlwaysAsk(), held_out),
        evaluate("RulePolicy", rule, held_out),
        evaluate("RulePolicy + 5% audit", rule, held_out, audit_rate=0.05),
        evaluate("LogisticPolicy", logit, held_out),
        evaluate("QTablePolicy", qpol, held_out),
        evaluate("NeverAsk", NeverAsk(), held_out),
    ]
    print(f"\n{'policy':<24}{'reviews/run':>12}{'missed/run':>12}{'review min/run':>16}{'cost/run':>10}")
    for r in rows:
        print(f"{r['name']:<24}{r['reviews']:>12.2f}{r['missed']:>12.3f}{r['minutes']:>16.1f}{r['cost']:>10.2f}")

    out = Path(__file__).resolve().parents[2] / "docs" / "tradeoff.png"
    try:
        plot(rows, out)
        print(f"\nsaved {out}")
    except ImportError:
        print("\nmatplotlib not installed; skipping plot (pip install 'hitl-gate[examples]')", file=sys.stderr)
    print(f"done in {time.time() - t0:.1f}s")
    return rows


if __name__ == "__main__":
    main()
