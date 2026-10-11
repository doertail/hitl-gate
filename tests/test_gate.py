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


# ---------------------------------------------------------------- 0.1.1 regressions


def test_audit_rng_is_per_gate_and_reproducible():
    def picks(gate, seed=7):
        d = GateDecider(NeverAsk(), audit_rate=0.5, seed=seed)
        return [d(gate, {})["ask"] for _ in range(2000)]

    a, b = picks("g1"), picks("g2")
    assert a == picks("g1")  # same seed + gate_id reproduces
    assert a != b
    agree = sum(x == y for x, y in zip(a, b)) / len(a)
    assert 0.45 < agree < 0.55  # independent coins at p=0.5 agree ~50% of the time
    # one decider serving several gates gives each gate its own stream, same as separate deciders
    shared = GateDecider(NeverAsk(), audit_rate=0.5, seed=7)
    mixed = [(shared("g1", {})["ask"], shared("g2", {})["ask"]) for _ in range(2000)]
    assert [x for x, _ in mixed] == a and [y for _, y in mixed] == b


def test_audit_picks_independent_across_gates_in_graph():
    log = DecisionLog()
    b = StateGraph(State)
    b.add_node("work", lambda s: {"value": "draft"})
    b.add_node("mid", lambda s: {})
    b.add_edge(START, "work")
    common = dict(policy=NeverAsk(), featurize=lambda s: {}, make_review_payload=lambda s: {},
                  apply_answer=lambda s, a: {}, log=log, audit_rate=0.5, seed=3)
    add_gated_review(b, "g1", source="work", target="mid", **common)
    add_gated_review(b, "g2", source="mid", target=END, **common)
    g = b.compile(checkpointer=MemorySaver())
    for i in range(200):
        run_episode(g, {"run_id": str(i)}, approve, f"t{i}")
    audited = {gid: [r.run_id for r in log.filter(gid) if r.mode == "audit"] for gid in ("g1", "g2")}
    assert audited["g1"] != audited["g2"]


def test_mark_downstream_is_append_only(tmp_path):
    path = tmp_path / "log.jsonl"
    log = DecisionLog(path)
    for i in range(50):
        for gate in ("g", "h"):
            log.append(DecisionRecord(f"r{i}", gate, {}, "auto", "p", ""))
    size = len(path.read_bytes().splitlines())
    for i in range(50):
        before = path.read_bytes()
        assert log.mark_downstream(f"r{i}", "g", i % 3 == 0) == 1
        after = path.read_bytes()
        assert after.startswith(before) and len(after.splitlines()) == len(before.splitlines()) + 1
    assert log.mark_downstream("nope", "g", True) == 0
    assert len(path.read_bytes().splitlines()) == size + 50
    log.mark_downstream("r0", "g", False)  # later patches win
    expected = [r.downstream_error for r in log.records]
    loaded = DecisionLog.from_jsonl(path)
    assert [r.downstream_error for r in loaded.records] == expected
    assert expected[0] is False and expected[6] is True and expected[1] is None
    log.compact()
    lines = path.read_text().splitlines()
    assert len(lines) == 100 and not any("_patch" in ln for ln in lines)
    assert [r.downstream_error for r in DecisionLog.from_jsonl(path).records] == expected


def test_qtable_visit_alpha_is_running_mean(tmp_path):
    rng = random.Random(0)
    q = QTablePolicy()
    targets = [rng.choice([-1.0, -1.0, -1.0, -500.0]) for _ in range(1000)]
    for t in targets:
        q.learn_episode([("g", {"x": 1}, "auto", t, None, None)], alpha="visit")
    key = q.key("g", {"x": 1})
    assert q.q[key]["auto"] == pytest.approx(sum(targets) / len(targets), rel=1e-12)
    assert q.n[key] == {"ask": 0, "auto": 1000}
    q.save(tmp_path / "q.json")
    q2 = QTablePolicy.load(tmp_path / "q.json")
    assert q2.n == q.n and q2.q == q.q
    # callable schedule: 1/n with a floor; a float keeps the old fixed-rate behaviour
    qc, qf = QTablePolicy(), QTablePolicy()
    for t in targets:
        qc.update("k", "ask", t, alpha=lambda n: max(1 / n, 0.01))
        qf.update("k", "ask", t, alpha=0.01)
    assert qc.q["k"]["ask"] != qf.q["k"]["ask"] and qc.n["k"]["ask"] == 1000
    with pytest.raises(ValueError):
        q.update("k", "ask", 0.0, alpha="fast")


def test_qtable_loads_files_without_counts(tmp_path):
    import json

    path = tmp_path / "old.json"
    path.write_text(json.dumps({"name": "qtable", "keys": None, "bins": {}, "q": {"k": {"ask": -1.0, "auto": 0.0}}}))
    q = QTablePolicy.load(path)
    assert q.n == {} and q.decide("x", {})[0] is True
    q.update("k", "ask", -3.0, alpha="visit")  # first visit after load: alpha = 1
    assert q.q["k"]["ask"] == -3.0


def test_fixed_clock_gives_byte_identical_jsonl(tmp_path):
    def run(path):
        log = DecisionLog(path, clock=lambda: 1700000000.0)
        g = build(NeverAsk(), log, audit_rate=0.3, seed=11)
        for i in range(30):
            run_episode(g, {"run_id": f"r{i}", "risky": False}, approve, f"t{i}")
            log.mark_downstream(f"r{i}", "check", i % 4 == 0)
        return path.read_bytes()

    a, b = run(tmp_path / "a.jsonl"), run(tmp_path / "b.jsonl")
    assert a == b and b'"ts": 1700000000.0' in a
    rec = DecisionRecord("r", "g", {}, "ask", "p", "")
    assert rec.ts is None
    DecisionLog().append(rec)
    assert isinstance(rec.ts, float)  # default clock is time.time


def test_rule_describe_omits_features_a_gate_does_not_use():
    recs = _recs(40, gate="a", feats={"x": 1}) + _recs(40, gate="b", feats={"y": "z", "w": None})
    p = RulePolicy.fit(recs, min_count=30)
    text = p.describe()
    assert "y=None" not in text and "x=None" not in text
    assert "[a] x=1  " in text and "[b] w=None, y='z'" in text  # explicit None values are kept
    assert p.decide("a", {"x": 1})[0] is False and p.decide("b", {"y": "z", "w": None})[0] is False
    assert text.splitlines()[-1].startswith("ask   anything else")
