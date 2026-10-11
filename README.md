# hitl-gate

**Stop interrupting humans for things they always approve.** A learnable "should I ask a human?" gate for LangGraph human-in-the-loop steps.

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg) ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

[한국어 README](README.ko.md)

![Tradeoff between human reviews and missed errors](docs/tradeoff.png)

In the bundled simulated document pipeline ([`examples/document_pipeline`](examples/document_pipeline/run.py), 1,000 held-out runs, all numbers from a simulation):

- **AlwaysAsk** sends 3.00 items per run to a human and lets 0.9 errors per 100 runs through.
- A **RulePolicy** fitted from 500 logged runs asks 1.29 times per run (**57% fewer reviews**) at 2.2 errors per 100 runs, cutting total cost from 3.23 to 1.84 per run.
- **NeverAsk** lets 13.9 errors per 100 runs through. Removing review entirely is not the answer.

## Why

Most agent graphs put `interrupt()` at every review point. That is safe on day one, and then:

- **Review fatigue.** Humans see hundreds of items that are fine, and start approving without looking (rubber-stamping). The review step keeps its cost and stops catching errors.
- **Removing the step is unsafe.** Some items really do need a human, and you can't tell which ones by intuition.
- **Decide per item, with evidence.** Most items fall into situations where humans have approved unchanged hundreds of times. Others don't. `hitl-gate` logs every review decision and its outcome, and uses that log to decide per item whether asking is worth it.

## 30-second quickstart

```bash
pip install git+<this-repo-url>   # not on PyPI yet; or clone and `pip install -e .`
```

```python
from typing import Annotated, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from hitl_gate import AlwaysAsk, DecisionLog, RulePolicy, add_gated_review, merge_dicts

class State(TypedDict, total=False):
    run_id: str
    hitl_decisions: Annotated[dict, merge_dicts]  # where gate decisions live
    draft: str
    confidence: str

builder = StateGraph(State)
builder.add_node("draft", lambda s: {"draft": "Dear customer, ...", "confidence": "high"})
builder.add_node("send", lambda s: {})
builder.add_edge(START, "draft")

log = DecisionLog("decisions.jsonl")
gate = add_gated_review(
    builder, "review_draft", source="draft", target="send",
    policy=AlwaysAsk(),                                   # start safe
    featurize=lambda s: {"confidence": s["confidence"]},  # discrete, observable signals
    make_review_payload=lambda s: {"draft": s["draft"]},  # what the human sees
    apply_answer=lambda s, a: a.get("changes", {}),       # human edits -> state update
    log=log,
)
builder.add_edge("send", END)
graph = builder.compile(checkpointer=MemorySaver())       # interrupts need a checkpointer

# later, once the log has evidence:
gate.policy = RulePolicy.fit(log.records, keys=["confidence"])
```

Resume an interrupt the usual LangGraph way, with the answer format the gate expects:

```python
from langgraph.types import Command

graph.invoke(Command(resume={"outcome": "approve", "review_seconds": 12.0}), config)
# outcome: "approve" | "edit" | "retry" | "escalate";  optional "changes": {...}
```

`hitl_gate.sim.run_episode(graph, inputs, reviewer, thread_id)` does the invoke / interrupt / resume loop for you with a scripted reviewer, which is handy for tests and simulations.

## How it works

```mermaid
flowchart LR
    S[source node] --> G{"gate node<br/>policy.decide()"}
    G -- ask --> R["review node<br/>interrupt()"] --> H((human)) --> T[target node]
    G -- auto --> T
    G -. record .-> L[(DecisionLog)]
    R -. record + outcome .-> L
    T -. downstream errors .-> L
    L -- RulePolicy.fit / Q-learning --> P[better policy] --> G
```

`add_gated_review` inserts two nodes between `source` and `target`:

- **`{gate_id}__gate`**: featurizes the state, asks the policy, applies shadow mode and audit sampling, and stores the decision in `state["hitl_decisions"][gate_id]`. It routes with a conditional edge.
- **`{gate_id}__review`**: calls `interrupt({"gate_id", "reason", "payload"})`, then applies the human answer and logs the outcome.

The decision lives in its own node on purpose. **When LangGraph resumes after `interrupt()`, the interrupting node runs again from the top.** If the ask/auto coin flip (audit sampling, exploration) happened inside that node, it would be re-rolled on resume. Here the decision is checkpointed before the interrupt and the review node only reads it. The test suite checks this.

If you wire graphs yourself, use `GateDecider(policy, mode, audit_rate, seed)` directly: it returns a plain dict you store in state, and `make_record(run_id, gate_id, decision, answer)` builds the log entry.

**Seeding.** Audit sampling draws from one RNG per gate, seeded from `f"{seed}:{gate_id}"` (hashed with SHA-512, stable across processes, unlike Python's salted `hash()`). Passing the same `seed` to every gate is fine: each gate gets an independent, reproducible stream, and one `GateDecider` serving several gates behaves like one decider per gate.

## Policies

| Policy | Asks when | Learns from |
|---|---|---|
| `AlwaysAsk` | always (today's default) | — |
| `NeverAsk` | never (baseline only) | — |
| `RulePolicy.fit(records, keys, min_count=30, max_change_rate=0.02)` | the bucket `(gate_id, feature values)` has fewer than `min_count` reviewed samples **or** a miss rate above `max_change_rate`. Unknown buckets → ask. `describe()` prints the rules. | logged reviews |
| `QTablePolicy` | `Q(ask) >= Q(auto)` for `(gate_id, discretized features)`. Ties and unseen states → ask. `act()` for epsilon-greedy training, `learn_episode()`, `save()`/`load()` JSON. | your reward signal (e.g. `-review_cost`, `-error_cost`) |

`RulePolicy` buckets only use the `keys` a gate's features actually contain, so one `keys` list can span gates with different features, and `describe()` shows each gate's own conditions.

**`QTablePolicy` learning rate.** `update()` and `learn_episode()` take `alpha` as a float (fixed rate), `"visit"` (`1/n` per `(state, action)`, i.e. Q is the running mean of its targets), or a callable `alpha(n)`. With rare, large costs (an error that slips through is much more expensive than a review) a fixed rate makes Q jump every time one of those costs lands; `"visit"` averages them out. For `1/n` with a floor, so the table keeps adapting later on, pass `alpha=lambda n: max(1 / n, 0.001)`. Visit counts live in `policy.n` and are saved by `save()`; files from 0.1.0 load with empty counts.

Any object with `name: str` and `decide(gate_id, features) -> (ask: bool, reason: str)` is a policy.

### Rubber-stamp protection

A review that says "approve" is not proof that the item was fine. `RulePolicy` counts a sample as a **miss** if the human changed the result **or** if it was approved and later failed downstream (`DecisionLog.mark_downstream(run_id, gate_id, error=True)`). A bucket where reviewers wave everything through, but errors keep appearing later, stays "ask" instead of being learned as safe.

## Safe rollout

1. **Shadow.** `add_gated_review(..., mode="shadow")`: the human is always asked; the log records what the policy *would* have done (`mode="shadow"`). Compare would-be autos against real outcomes before trusting it.
2. **Capped auto with audits.** Go live with `audit_rate=0.05`: 5% of auto decisions are still sent to a human and logged as `mode="audit"`. That gives you unbiased samples of what auto lets through.
3. **Roll back on downstream errors.** Mark downstream failures in the log and refit. Auto-passed items that failed downstream count as misses, so the affected bucket returns to "ask" on the next fit. Swap the policy in place with `gate.policy = RulePolicy.fit(...)`.

## Logging format

One `DecisionRecord` per gate decision (JSONL when `DecisionLog(path)` is given).

The file is **append-only**: `append` writes one record line and `mark_downstream` writes one small patch line, `{"_patch": {"run_id", "gate_id", "downstream_error"}}`, so marking stays O(1) however large the log is. `DecisionLog.from_jsonl(path)` replays the patches (and reads 0.1.0 files unchanged). Call `log.compact()` to rewrite the file once with patches merged. `DecisionLog(path)` appends to an existing file; it does not load or truncate it.

| field | meaning |
|---|---|
| `run_id`, `gate_id` | which run, which gate (`run_id` read from `state[run_id_key]`) |
| `features` | output of `featurize(state)` |
| `decision` | what the **policy** decided: `"ask"` / `"auto"` |
| `policy_name`, `reason` | which policy, and why |
| `mode` | `"live"`, `"shadow"` (always asked) or `"audit"` (auto, sampled for review) |
| `human_outcome` | `"approve"`, `"edit"`, `"retry"`, `"escalate"`, or `None` if nobody was asked |
| `changed` | `human_outcome != "approve"` |
| `review_seconds` | from the answer |
| `downstream_error` | filled later via `mark_downstream` |
| `ts` | unix time, stamped by the log's `clock` on `append` |

For byte-reproducible logs, pass a fixed clock: `DecisionLog(path, clock=lambda: 0.0)`. Together with a fixed `seed`, two runs then write identical files.

Auto passes are logged by the gate node. Asked items are logged once the human answers, so a review that never comes back leaves no record.

## Limitations

- **Simulated results depend on their assumptions.** The plot comes from a toy pipeline with made-up error rates, catch rate (90%) and costs (1 review = 1, 1 missed error = 25). Your numbers will differ. Run shadow mode on your own traffic first.
- **Never gate legally or contractually required approvals.** If a human must sign off, keep `interrupt()` unconditional.
- **Features must be discrete.** Buckets and Q-tables need a handful of values per feature. Bucket continuous signals in `featurize` (or use `QTablePolicy(bins=...)`). Too many features means too few samples per bucket, so everything stays "ask", which is the safe failure mode.
- **Tabular RL is simple by design.** `QTablePolicy` is a transparent baseline, not a state-of-the-art learner. In the example it trains on a fast pure-Python mirror of the simulation (fixed `alpha=0.02`) and does not beat the rule policy.
- `retry` and `escalate` are logged, but routing stays `review -> target`. For retry loops, wire your own edges using `GateDecider`.

## Related work

- **Learning to Defer** (Madras, Pitassi & Zemel, NeurIPS 2018): train a model to decide when to hand a prediction to a human expert.
- **Ask4Help** (Singh et al., [arXiv 2211.09960](https://arxiv.org/abs/2211.09960)): learn a policy for when an embodied agent should request expert help.

## Development

```bash
python3 -m venv .venv            # tested with Python 3.14 and langgraph 1.2
.venv/bin/pip install -e ".[dev,examples]"
.venv/bin/pytest -q
.venv/bin/python examples/document_pipeline/run.py   # ~25 s, writes docs/tradeoff.png
```

The example is seeded and reproducible. See [CHANGELOG.md](CHANGELOG.md) for changes between versions.

## License

MIT. See [LICENSE](LICENSE).
