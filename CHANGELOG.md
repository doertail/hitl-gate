# Changelog

## 0.1.1

- **Per-gate audit RNG.** `GateDecider` / `add_gated_review` now derive one RNG per gate from `f"{seed}:{gate_id}"`. Before, gates sharing a `seed` drew identical audit sequences, so audits hit the same runs at every gate. Audit picks for a given seed differ from 0.1.0. `GateDecider.rng` is now a method, `rng(gate_id)`.
- **Append-only `DecisionLog` file.** `mark_downstream` appends a `{"_patch": ...}` line instead of rewriting the whole file (was O(n) per call). `from_jsonl` replays patches and still reads 0.1.0 files. New `DecisionLog.compact()` rewrites the file once with patches merged. Marks use an index instead of scanning all records.
- **`QTablePolicy` learning-rate schedules.** `alpha` accepts a float (unchanged), `"visit"` (`1/n` per state-action, i.e. a running mean) or a callable `alpha(n)`. Visit counts are kept in `policy.n` and saved; old files load with empty counts.
- **Injectable clock.** `DecisionLog(path, clock=time.time)` stamps `ts` on append. `DecisionRecord.ts` now defaults to `None` until a log stamps it. A fixed clock gives byte-identical JSONL across seeded runs.
- **`RulePolicy` buckets skip absent keys.** Features a gate doesn't produce are no longer stored as `None`, so `describe()` no longer prints `key=None` for them. Decisions are unchanged.

## 0.1.0

- Initial release.
