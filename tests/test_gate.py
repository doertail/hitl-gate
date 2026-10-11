import random
from typing import Annotated, TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from hitl_gate import (
    AlwaysAsk,
    DecisionLog,
    DecisionRecord,
    GateDecider,
    NeverAsk,
    QTablePolicy,
    RulePolicy,
    add_gated_review,
    merge_dicts,
)
from hitl_gate.sim import run_episode


class State(TypedDict, total=False):
    run_id: str
    hitl_decisions: Annotated[dict, merge_dicts]
    risky: bool
    value: str
    trace: Annotated[list, lambda a, b: (a or []) + (b or [])]


class ByFeature:
    name = "by_feature"

    def decide(self, gate_id, features):
        return features["risky"], "risky" if features["risky"] else "safe"


class CountingCoin:
    """Random policy that counts calls: re-rolling on resume would show up as extra calls."""

    name = "coin"

    def __init__(self):
        self.calls = 0
        self.rng = random.Random()

    def decide(self, gate_id, features):
        self.calls += 1
        return self.rng.random() < 0.5, "coin"


def build(policy, log, **kw):
    b = StateGraph(State)
    b.add_node("work", lambda s: {"value": "draft", "trace": ["work"]})
    b.add_node("publish", lambda s: {"trace": ["publish"]})
    b.add_edge(START, "work")
    add_gated_review(
        b, "check", source="work", target="publish", policy=policy,
        featurize=lambda s: {"risky": s["risky"]},
        make_review_payload=lambda s: {"value": s["value"]},
        apply_answer=lambda s, a: {"value": a.get("changes", {}).get("value", s["value"]), "trace": ["review"]},
        log=log, **kw,
    )
    b.add_edge("publish", END)
    return b.compile(checkpointer=MemorySaver())


def approve(_):
    return {"outcome": "approve", "review_seconds": 3.0}


def test_routes_ask_and_auto():
    log = DecisionLog()
    g = build(ByFeature(), log)
    safe = run_episode(g, {"run_id": "a", "risky": False}, approve, "t-a")
    risky = run_episode(g, {"run_id": "b", "risky": True}, approve, "t-b")
    assert safe["trace"] == ["work", "publish"]
    assert risky["trace"] == ["work", "review", "publish"]
    a, b = log.records
    assert (a.decision, a.human_outcome, a.mode) == ("auto", None, "live")
    assert (b.decision, b.human_outcome, b.changed, b.review_seconds) == ("ask", "approve", False, 3.0)


def test_interrupt_resume_applies_answer_and_does_not_reroll():
    for i in range(20):
        log, policy = DecisionLog(), CountingCoin()
        g = build(policy, log, audit_rate=0.5, seed=i)
        cfg = {"configurable": {"thread_id": f"t{i}"}}
        out = g.invoke({"run_id": f"r{i}", "risky": True}, cfg)
        decided = g.get_state(cfg).values["hitl_decisions"]["check"]
        if "__interrupt__" in out:
            assert decided["ask"]
            assert out["__interrupt__"][0].value == {"gate_id": "check", "reason": decided["reason"], "payload": {"value": "draft"}}
            out = g.invoke(Command(resume={"outcome": "edit", "changes": {"value": "fixed"}, "review_seconds": 1.5}), cfg)
            assert out["value"] == "fixed"
            assert out["trace"] == ["work", "review", "publish"]  # review node re-ran on resume, but only its update counts
        assert policy.calls == 1  # decided once, before the interrupt
        assert out["hitl_decisions"]["check"] == decided
        assert len(log.records) == 1
        rec = log.records[0]
        assert rec.decision == decided["decision"] and rec.mode == decided["mode"]
        if decided["ask"]:
            assert rec.human_outcome == "edit" and rec.changed is True


def test_invalid_outcome_rejected():
    g = build(AlwaysAsk(), DecisionLog())
    with pytest.raises(ValueError):
        run_episode(g, {"run_id": "x", "risky": True}, lambda _: {"outcome": "lgtm"}, "t-x")


def test_shadow_always_asks_but_logs_would_be_decision():
    log = DecisionLog()
    g = build(NeverAsk(), log, mode="shadow")
    out = run_episode(g, {"run_id": "s", "risky": False}, approve, "t-s")
    assert "review" in out["trace"]
    rec = log.records[0]
    assert (rec.mode, rec.decision, rec.human_outcome) == ("shadow", "auto", "approve")


def test_audit_rate_is_approximately_respected():
    d = GateDecider(NeverAsk(), audit_rate=0.1, seed=0)
    out = [d("g", {}) for _ in range(5000)]
    audited = [x for x in out if x["ask"]]
    assert all(x["mode"] == "audit" and x["decision"] == "auto" for x in audited)
    assert 0.085 < len(audited) / 5000 < 0.115
    # policy "ask" decisions are never turned into audits
    assert all(x["mode"] == "live" for x in (GateDecider(AlwaysAsk(), audit_rate=0.5)("g", {}) for _ in range(100)))


def _recs(n, gate="g", feats=None, outcome="approve", downstream=None, decision="ask"):
    return [
        DecisionRecord("r", gate, feats or {"k": 1}, decision, "always_ask", "", "live",
                       human_outcome=outcome, changed=None if outcome is None else outcome != "approve",
                       downstream_error=downstream)
        for _ in range(n)
    ]


def test_rule_policy_fit():
    recs = _recs(40, feats={"k": "safe"}) + _recs(40, feats={"k": "bad"}) + _recs(5, feats={"k": "bad"}, outcome="edit")
    recs += _recs(10, feats={"k": "rare"})
    p = RulePolicy.fit(recs, min_count=30, max_change_rate=0.02)
    assert p.decide("g", {"k": "safe"})[0] is False
    assert p.decide("g", {"k": "bad"})[0] is True
    assert p.decide("g", {"k": "rare"})[0] is True  # insufficient
    assert p.decide("g", {"k": "never-seen"})[0] is True
    assert p.decide("other_gate", {"k": "safe"})[0] is True
    assert "AUTO" in p.describe() and "k='safe'" in p.describe()


def test_rule_policy_rubber_stamp_protection():
    # every item approved, but 10% later failed downstream: approvals were rubber stamps
    recs = _recs(90, feats={"k": "x"}) + _recs(10, feats={"k": "x"}, downstream=True)
    assert RulePolicy.fit(recs).decide("g", {"k": "x"})[0] is True
    assert RulePolicy.fit(recs, count_downstream_errors=False).decide("g", {"k": "x"})[0] is False


def test_rule_policy_rolls_back_on_auto_downstream_errors():
    recs = _recs(100, feats={"k": "x"})
    assert RulePolicy.fit(recs).decide("g", {"k": "x"})[0] is False
    recs += _recs(5, feats={"k": "x"}, outcome=None, downstream=True, decision="auto")
    assert RulePolicy.fit(recs).decide("g", {"k": "x"})[0] is True


def test_log_jsonl_roundtrip_and_mark_downstream(tmp_path):
    path = tmp_path / "log.jsonl"
    log = DecisionLog(path)
    for r in _recs(3):
        log.append(r)
    log.append(_recs(1, gate="h")[0])
    assert log.mark_downstream("r", "g", True) == 3
    loaded = DecisionLog.from_jsonl(path)
    assert [r.downstream_error for r in loaded.records] == [True, True, True, None]
    assert len(loaded.filter("h")) == 1 and len(loaded.filter()) == 4


def test_qtable_learns_and_roundtrips(tmp_path):
    q = QTablePolicy(bins={"score": [0.5]})
    rng = random.Random(0)
    for _ in range(2000):
        score = rng.random()
        ask = q.act("g", {"score": score}, epsilon=0.3, rng=rng)
        error = rng.random() < (0.5 if score > 0.5 else 0.0)
        reward = -1.0 if ask else (-10.0 if error else 0.0)
        q.learn_episode([("g", {"score": score}, ask, reward, None, None)], alpha=0.1)
    assert q.decide("g", {"score": 0.2})[0] is False
    assert q.decide("g", {"score": 0.9})[0] is True
    assert q.decide("unseen", {"score": 0.2})[0] is True  # ties -> ask
    q.save(tmp_path / "q.json")
    q2 = QTablePolicy.load(tmp_path / "q.json")
    assert q2.q == q.q and q2.decide("g", {"score": 0.2}) == q.decide("g", {"score": 0.2})
