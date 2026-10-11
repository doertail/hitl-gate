"""LangGraph integration: a gate node that decides ask/auto, and a review node that interrupts."""

from __future__ import annotations

import random
from typing import Annotated, Any, Callable, Literal, TypedDict

from langgraph.graph import StateGraph
from langgraph.types import interrupt

from .log import OUTCOMES, DecisionLog, DecisionRecord
from .policies import Policy


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Reducer for ``hitl_decisions``: shallow-merge, right wins."""
    return {**(left or {}), **(right or {})}


class HITLState(TypedDict, total=False):
    """Mix these keys into your graph state (subclass it, or copy the two lines)."""

    run_id: str
    hitl_decisions: Annotated[dict, merge_dicts]


Mode = Literal["live", "shadow", "audit"]


class GateDecider:
    """Turns a policy verdict into a gate decision, applying shadow mode and audit sampling.

    Call it from a node that does **not** call ``interrupt()`` (the interrupting
    node re-runs from the top on resume, which would re-roll audit sampling).
    The returned dict is plain data, safe to store in graph state::

        {"ask": bool, "decision": "ask"|"auto", "mode": "live"|"shadow"|"audit",
         "reason": str, "policy_name": str, "features": dict}

    ``ask`` is whether a human will review; ``decision`` is what the policy said.
    """

    def __init__(self, policy: Policy, mode: Literal["live", "shadow"] = "live", audit_rate: float = 0.0, seed: int | None = None):
        if mode not in ("live", "shadow"):
            raise ValueError("mode must be 'live' or 'shadow' (audit is set per decision via audit_rate)")
        if not 0.0 <= audit_rate <= 1.0:
            raise ValueError("audit_rate must be in [0, 1]")
        self.policy = policy
        self.mode = mode
        self.audit_rate = audit_rate
        self.rng = random.Random(seed)

    def __call__(self, gate_id: str, features: dict) -> dict:
        ask, reason = self.policy.decide(gate_id, features)
        mode: Mode = self.mode
        will_ask = ask
        if mode == "shadow":
            will_ask = True
            reason = f"shadow (policy would {'ask' if ask else 'auto'}): {reason}"
        elif not ask and self.audit_rate and self.rng.random() < self.audit_rate:
            will_ask, mode = True, "audit"
            reason = f"audit sample: {reason}"
        return {
            "ask": will_ask,
            "decision": "ask" if ask else "auto",
            "mode": mode,
            "reason": reason,
            "policy_name": self.policy.name,
            "features": features,
        }


def make_record(run_id: str, gate_id: str, decision: dict, answer: dict | None = None) -> DecisionRecord:
    """Build a :class:`DecisionRecord` from a :class:`GateDecider` decision and an optional human answer."""
    rec = DecisionRecord(
        run_id=run_id,
        gate_id=gate_id,
        features=decision["features"],
        decision=decision["decision"],
        policy_name=decision["policy_name"],
        reason=decision["reason"],
        mode=decision["mode"],
    )
    if answer is not None:
        outcome = answer.get("outcome")
        if outcome not in OUTCOMES:
            raise ValueError(f"answer['outcome'] must be one of {OUTCOMES}, got {outcome!r}")
        rec.human_outcome = outcome
        rec.changed = outcome != "approve"
        rec.review_seconds = answer.get("review_seconds")
    return rec


def add_gated_review(
    builder: StateGraph,
    gate_id: str,
    *,
    source: str,
    target: str,
    policy: Policy,
    featurize: Callable[[Any], dict],
    make_review_payload: Callable[[Any], dict],
    apply_answer: Callable[[Any, dict], dict],
    log: DecisionLog,
    run_id_key: str = "run_id",
    mode: Literal["live", "shadow"] = "live",
    audit_rate: float = 0.0,
    seed: int | None = None,
) -> GateDecider:
    """Insert ``source -> {gate_id}__gate -> ({gate_id}__review | target)`` into ``builder``.

    * ``{gate_id}__gate`` featurizes the state, asks the policy, stores the decision in
      ``state["hitl_decisions"][gate_id]`` and logs auto passes immediately.
    * ``{gate_id}__review`` calls ``interrupt({"gate_id", "reason", "payload"})``, where
      ``payload = make_review_payload(state)``. Resume with
      ``Command(resume={"outcome": "approve"|"edit"|"retry"|"escalate", "changes": {...}, "review_seconds": float})``.
      It returns ``apply_answer(state, answer)`` as the state update and logs the outcome.
    * Both route to ``target`` (which may be ``END``).

    Your state must include :class:`HITLState`'s keys and the graph must be compiled
    with a checkpointer. Do not add your own edge out of ``source``.
    Returns the :class:`GateDecider` (handy for swapping ``.policy`` later).
    """
    decider = GateDecider(policy, mode=mode, audit_rate=audit_rate, seed=seed)
    gate_node, review_node = f"{gate_id}__gate", f"{gate_id}__review"

    def gate(state) -> dict:
        d = decider(gate_id, featurize(state))
        if not d["ask"]:
            log.append(make_record(state.get(run_id_key, ""), gate_id, d))
        return {"hitl_decisions": {gate_id: d}}

    def review(state) -> dict:
        # Re-executes from the top on resume: only read the decision here, never make it.
        d = state["hitl_decisions"][gate_id]
        answer = interrupt({"gate_id": gate_id, "reason": d["reason"], "payload": make_review_payload(state)})
        rec = make_record(state.get(run_id_key, ""), gate_id, d, answer)
        update = apply_answer(state, answer) or {}
        log.append(rec)
        return update

    def route(state) -> str:
        return "ask" if state["hitl_decisions"][gate_id]["ask"] else "auto"

    builder.add_node(gate_node, gate)
    builder.add_node(review_node, review)
    builder.add_edge(source, gate_node)
    builder.add_conditional_edges(gate_node, route, {"ask": review_node, "auto": target})
    builder.add_edge(review_node, target)
    return decider
