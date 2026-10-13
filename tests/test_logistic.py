import random

from hitl_gate import DecisionRecord, LogisticPolicy, RulePolicy


def _rec(gate, feats, changed, asked=True, downstream=None):
    return DecisionRecord(run_id="r", gate_id=gate, features=feats, decision="ask", policy_name="always_ask",
                          reason="", human_outcome=("edit" if changed else "approve") if asked else None,
                          changed=changed if asked else None, downstream_error=downstream)


def _logs(gate, n, seed):
    """unmapped=0 → never wrong; each unmapped code adds risk. Same bucket for the rule (no unmapped key)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        unmapped = rng.choice([0, 0, 0, 0, 1, 2, 3])
        out.append(_rec(gate, {"product": "A", "unmapped": unmapped}, rng.random() < 0.12 * unmapped))
    return out


def test_scores_items_inside_one_bucket():
    logs = _logs("H3", 2000, 0)
    pol = LogisticPolicy.fit(logs, threshold=0.05)
    ask_clean, _ = pol.decide("H3", {"product": "A", "unmapped": 0})
    ask_risky, reason = pol.decide("H3", {"product": "A", "unmapped": 3})
    assert not ask_clean and ask_risky and "p(miss)" in reason
    # a rule on the shared bucket can't separate them: overall miss rate is too high to auto-pass anything
    rule = RulePolicy.fit(logs, keys=["product"], max_change_rate=0.05)
    assert rule.decide("H3", {"product": "A", "unmapped": 0})[0]


def test_pooled_gate_with_few_records_borrows_shared_weights():
    logs = _logs("big", 2000, 1) + _logs("small", 15, 2)
    pol = LogisticPolicy.fit(logs, threshold=0.05, min_gate_samples=0)  # opt in to pooling a 15-record gate
    assert not pol.decide("small", {"product": "A", "unmapped": 0})[0]
    assert pol.decide("small", {"product": "A", "unmapped": 3})[0]
    strict = LogisticPolicy.fit(logs, threshold=0.05)  # default 30: a 15-record gate keeps asking
    assert strict.decide("small", {"product": "A", "unmapped": 0})[0]


def test_unseen_gate_asks_and_downstream_error_is_a_miss(tmp_path):
    logs = _logs("H3", 1000, 3)
    # rubber stamps: approved, then failed downstream → label is a miss
    logs += [_rec("H3", {"product": "A", "unmapped": 0}, False, downstream=True) for _ in range(300)]
    pol = LogisticPolicy.fit(logs, threshold={"H3": 0.05})
    assert pol.decide("H3", {"product": "A", "unmapped": 0})[0]
    assert pol.decide("never-seen", {"product": "A", "unmapped": 0})[0]
    pol.save(tmp_path / "p.json")
    again = LogisticPolicy.load(tmp_path / "p.json")
    assert abs(again.proba("H3", {"product": "A", "unmapped": 1}) - pol.proba("H3", {"product": "A", "unmapped": 1})) < 1e-12
