# Changelog

## 0.2.0

- **Tool-call approval.** New `gated_tool(tool, *, policy, log, gate_id=None, featurize=lambda name, args: {}, mode="live", audit_rate=0.0, seed=None, allowed_decisions=..., edit_notice=...)` wraps any LangChain tool so each call is gated: auto runs it, ask interrupts with a LangChain `HITLRequest`-shaped payload and accepts `{"decisions": [approve | edit | reject | respond]}`. Works in `ToolNode`, hand-built graphs and `create_agent`, with only `langgraph` installed. Sync and async.
- **`GatedHumanInTheLoopMiddleware`** (`hitl_gate.middleware`, extra `hitl-gate[langchain]`, langchain>=1.4): the official `HumanInTheLoopMiddleware` with the policy as its `when` predicate, logging one record per gated call.
- **Stable decisions on resume.** Tool-call decisions are cached per `(thread_id, gate_id, tool_call_id)` until logged, and audit draws can be keyed: `GateDecider.__call__(gate_id, features, key=None)` draws from `(seed, gate_id, key)` when `key` and `seed` are given. Existing calls without `key` behave as before.
- `OUTCOMES` gains `"reject"` and `"respond"`. For tool calls, `run_id` is the tool call id and `gate_id` defaults to the tool name.
- New example `examples/tool_approval` (offline support agent).

## 0.1.1

- **Per-gate audit RNG.** `GateDecider` / `add_gated_review` now derive one RNG per gate from `f"{seed}:{gate_id}"`. Before, gates sharing a `seed` drew identical audit sequences, so audits hit the same runs at every gate. Audit picks for a given seed differ from 0.1.0. `GateDecider.rng` is now a method, `rng(gate_id)`.
- **Append-only `DecisionLog` file.** `mark_downstream` appends a `{"_patch": ...}` line instead of rewriting the whole file (was O(n) per call). `from_jsonl` replays patches and still reads 0.1.0 files. New `DecisionLog.compact()` rewrites the file once with patches merged. Marks use an index instead of scanning all records.
- **`QTablePolicy` learning-rate schedules.** `alpha` accepts a float (unchanged), `"visit"` (`1/n` per state-action, i.e. a running mean) or a callable `alpha(n)`. Visit counts are kept in `policy.n` and saved; old files load with empty counts.
- **Injectable clock.** `DecisionLog(path, clock=time.time)` stamps `ts` on append. `DecisionRecord.ts` now defaults to `None` until a log stamps it. A fixed clock gives byte-identical JSONL across seeded runs.
- **`RulePolicy` buckets skip absent keys.** Features a gate doesn't produce are no longer stored as `None`, so `describe()` no longer prints `key=None` for them. Decisions are unchanged.

## 0.1.0

- Initial release.
