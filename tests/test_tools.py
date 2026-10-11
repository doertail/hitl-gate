import asyncio
import itertools
import random
from typing import Any, Callable

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from hitl_gate import AlwaysAsk, DecisionLog, GateDecider, NeverAsk, gated_tool
from hitl_gate.sim import run_episode

_ids = itertools.count()


class Scripted(BaseChatModel):
    """Offline chat model: the first turn emits ``calls`` as tool calls, later turns say "done"."""

    calls: list = []

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if isinstance(messages[-1], HumanMessage):
            msg = AIMessage("", tool_calls=[{"name": n, "args": a, "id": f"call_{next(_ids)}"} for n, a in self.calls])
        else:
            msg = AIMessage("done")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self):
        return "scripted"


RAN: list = []


@tool
def issue_refund(amount: int) -> str:
    """Refund the customer."""
    RAN.append(("issue_refund", amount))
    return f"refunded {amount}"


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    RAN.append(("lookup_order", order_id))
    return f"order {order_id}: shipped"


class BigRefunds:
    name = "big_refunds"

    def decide(self, gate_id, features):
        return features["big"], "big" if features["big"] else "small"


def by_amount(name, args):
    return {"big": args["amount"] > 100}


class Coin:
    """Unseeded random policy that counts calls: re-deciding on resume shows up as extra calls."""

    name = "coin"

    def __init__(self):
        self.calls, self.rng = 0, random.Random()

    def decide(self, gate_id, features):
        self.calls += 1
        return self.rng.random() < 0.5, "coin"


def agent(tools, calls):
    """Hand-built ReAct graph (core langgraph only): model -> ToolNode -> model."""
    model = Scripted(calls=calls)
    b = StateGraph(MessagesState)
    b.add_node("model", lambda s: {"messages": [model.invoke(s["messages"])]})
    b.add_node("tools", ToolNode(tools))
    b.add_edge(START, "model")
    b.add_conditional_edges("model", tools_condition, {"tools": "tools", END: END})
    b.add_edge("tools", "model")
    return b.compile(checkpointer=MemorySaver())


def start(graph, thread):
    cfg = {"configurable": {"thread_id": thread}}
    return graph.invoke({"messages": [HumanMessage("hi")]}, cfg), cfg


def tool_msgs(result):
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


@pytest.fixture(autouse=True)
def _clear():
    RAN.clear()


def test_auto_runs_tool_without_interrupt():
    log = DecisionLog()
    g = agent([gated_tool(issue_refund, policy=BigRefunds(), featurize=by_amount, log=log)], [("issue_refund", {"amount": 5})])
    out, _ = start(g, "t1")
    assert "__interrupt__" not in out and RAN == [("issue_refund", 5)]
    (rec,) = log.records
    tm = tool_msgs(out)[0]
    assert (rec.run_id, rec.gate_id, rec.decision, rec.human_outcome) == (tm.tool_call_id, "issue_refund", "auto", None)
    assert rec.features == {"big": False} and out["messages"][-1].content == "done"


def test_ask_interrupts_with_official_shape_then_approve_runs():
    log = DecisionLog()
    g = agent([gated_tool(issue_refund, policy=BigRefunds(), featurize=by_amount, log=log)], [("issue_refund", {"amount": 500})])
    out, cfg = start(g, "t2")
    (intr,) = out["__interrupt__"]
    req = intr.value
    assert req["action_requests"][0]["name"] == "issue_refund" and req["action_requests"][0]["args"] == {"amount": 500}
    assert req["review_configs"] == [{"action_name": "issue_refund", "allowed_decisions": ["approve", "edit", "reject", "respond"]}]
    assert req["hitl_gate"]["reason"] == "big" and RAN == [] and len(log) == 0
    out = g.invoke(Command(resume={"decisions": [{"type": "approve", "review_seconds": 4.0}]}), cfg)
    assert RAN == [("issue_refund", 500)]
    (rec,) = log.records
    assert (rec.decision, rec.human_outcome, rec.changed, rec.review_seconds) == ("ask", "approve", False, 4.0)


def test_edit_runs_with_new_args_and_tells_model():
    log = DecisionLog()
    g = agent([gated_tool(issue_refund, policy=AlwaysAsk(), log=log)], [("issue_refund", {"amount": 500})])
    _, cfg = start(g, "t3")
    out = g.invoke(Command(resume={"decisions": [{"type": "edit", "edited_action": {"name": "issue_refund", "args": {"amount": 50}}}]}), cfg)
    assert RAN == [("issue_refund", 50)]
    content = tool_msgs(out)[0].content
    assert "edited" in content and '{"amount": 50}' in content and content.endswith("refunded 50")
    assert (log.records[0].human_outcome, log.records[0].changed) == ("edit", True)


def test_reject_does_not_run_tool():
    log = DecisionLog()
    g = agent([gated_tool(issue_refund, policy=AlwaysAsk(), log=log)], [("issue_refund", {"amount": 500})])
    _, cfg = start(g, "t4")
    out = g.invoke(Command(resume={"decisions": [{"type": "reject", "message": "over limit"}]}), cfg)
    assert RAN == []
    tm = tool_msgs(out)[0]
    assert tm.status == "error" and "over limit" in tm.content
    assert log.records[0].human_outcome == "reject" and out["messages"][-1].content == "done"


def test_respond_and_disallowed_decision():
    log = DecisionLog()
    g = agent([gated_tool(issue_refund, policy=AlwaysAsk(), log=log, allowed_decisions=["approve", "reject", "respond"])], [("issue_refund", {"amount": 1})])
    _, cfg = start(g, "t5")
    with pytest.raises(ValueError, match="not allowed"):
        g.invoke(Command(resume={"decisions": [{"type": "edit", "edited_action": {"name": "issue_refund", "args": {}}}]}), cfg)
    _, cfg = start(g, "t5b")
    out = g.invoke(Command(resume={"decisions": [{"type": "respond", "message": "refund already sent"}]}), cfg)
    assert RAN == [] and tool_msgs(out)[0].content == "refund already sent"
    assert [r.human_outcome for r in log.records] == ["respond"]
    with pytest.raises(ValueError):
        gated_tool(issue_refund, policy=AlwaysAsk(), log=log, allowed_decisions=["maybe"])


def test_shadow_always_asks():
    log = DecisionLog()
    g = agent([gated_tool(issue_refund, policy=NeverAsk(), log=log, mode="shadow")], [("issue_refund", {"amount": 1})])
    out, cfg = start(g, "t6")
    assert out["__interrupt__"]
    g.invoke(Command(resume={"decisions": [{"type": "approve"}]}), cfg)
    (rec,) = log.records
    assert (rec.decision, rec.mode, rec.human_outcome) == ("auto", "shadow", "approve")


def approve(_):
    return {"decisions": [{"type": "approve"}]}


def test_audit_sampling_is_a_stable_function_of_the_call():
    def run(seed):
        log = DecisionLog()
        g = agent([gated_tool(issue_refund, policy=NeverAsk(), log=log, audit_rate=0.3, seed=seed)], [("issue_refund", {"amount": 1})])
        for i in range(60):
            run_episode(g, {"messages": [HumanMessage("hi")]}, approve, f"a{i}")
        return [r.mode for r in log.records]

    global _ids
    _ids = itertools.count()
    a = run(seed=3)
    _ids = itertools.count()
    b = run(seed=3)
    assert a == b and 5 < a.count("audit") < 35 and set(a) == {"live", "audit"}
    d = GateDecider(NeverAsk(), audit_rate=0.5, seed=1)
    assert [d("g", {}, key="k")["mode"] for _ in range(5)] == [d("g", {}, key="k")["mode"]] * 5


def test_resume_reuses_decision_and_logs_once_per_call():
    coin, log = Coin(), DecisionLog()
    g = agent([gated_tool(issue_refund, policy=coin, log=log)], [("issue_refund", {"amount": 1})])
    asked = 0
    for i in range(30):
        out, cfg = start(g, f"r{i}")
        if "__interrupt__" in out:
            asked += 1
            g.invoke(Command(resume={"decisions": [{"type": "approve"}]}), cfg)
    assert coin.calls == 30 and len(log) == 30 and 0 < asked < 30
    assert sum(r.asked for r in log.records) == asked and len({r.run_id for r in log.records}) == 30
    assert not g.nodes["tools"].bound.tools_by_name["issue_refund"].gate._pending  # nothing left over


def test_audit_decision_survives_losing_the_in_memory_cache():
    # Simulates a process restart between interrupt and resume: the decision is recomputed from the hash.
    log = DecisionLog()
    t = gated_tool(issue_refund, policy=NeverAsk(), log=log, audit_rate=0.5, seed=11)
    g = agent([t], [("issue_refund", {"amount": 1})])
    for i in range(20):
        out, cfg = start(g, f"c{i}")
        if "__interrupt__" in out:
            t.gate._pending.clear()
            g.invoke(Command(resume={"decisions": [{"type": "approve"}]}), cfg)
            assert log.records[-1].mode == "audit" and log.records[-1].human_outcome == "approve"
            assert RAN[-1] == ("issue_refund", 1)
            return
    pytest.fail("no audit sampled")


def test_mark_downstream_by_tool_call_id():
    log = DecisionLog()
    g = agent([gated_tool(issue_refund, policy=NeverAsk(), log=log)], [("issue_refund", {"amount": 5})])
    out, _ = start(g, "m1")
    assert log.mark_downstream(tool_msgs(out)[0].tool_call_id, "issue_refund", True) == 1
    assert log.records[0].downstream_error is True


def test_requires_tool_call_input():
    t = gated_tool(issue_refund, policy=NeverAsk(), log=DecisionLog())
    with pytest.raises(TypeError, match="ToolCall"):
        t.invoke({"amount": 3})


def test_async_agent():
    log = DecisionLog()
    g = agent([gated_tool(issue_refund, policy=BigRefunds(), featurize=by_amount, log=log)], [("issue_refund", {"amount": 500})])

    async def main():
        cfg = {"configurable": {"thread_id": "async"}}
        out = await g.ainvoke({"messages": [HumanMessage("hi")]}, cfg)
        assert out["__interrupt__"]
        return await g.ainvoke(Command(resume={"decisions": [{"type": "edit", "edited_action": {"name": "issue_refund", "args": {"amount": 99}}}]}), cfg)

    out = asyncio.run(main())
    assert RAN == [("issue_refund", 99)] and log.records[0].human_outcome == "edit"
    assert out["messages"][-1].content == "done"


# --- LangChain v1 create_agent ---------------------------------------------------------------

langchain_agents = pytest.importorskip("langchain.agents")


def lc_agent(calls, tools=None, middleware=()):
    return langchain_agents.create_agent(
        Scripted(calls=calls), tools=tools or [issue_refund, lookup_order],
        middleware=list(middleware), checkpointer=MemorySaver(),
    )


def test_gated_tool_in_create_agent_parallel_calls():
    log = DecisionLog()
    tools = [gated_tool(issue_refund, policy=AlwaysAsk(), log=log), lookup_order]
    g = lc_agent([("issue_refund", {"amount": 1}), ("issue_refund", {"amount": 2}), ("lookup_order", {"order_id": "A"})], tools)
    out, cfg = start(g, "p1")
    assert len(out["__interrupt__"]) == 2  # one task per tool call, resumable by interrupt id
    RAN.clear()
    out = run_episode(g, {"messages": [HumanMessage("hi")]}, approve, "p2")
    assert sorted(RAN) == [("issue_refund", 1), ("issue_refund", 2), ("lookup_order", "A")]
    assert len(log) == 2 and all(r.human_outcome == "approve" for r in log.records)


def test_middleware_interrupts_only_flagged_calls():
    from hitl_gate.middleware import GatedHumanInTheLoopMiddleware

    log = DecisionLog()
    mw = GatedHumanInTheLoopMiddleware({"issue_refund": True}, policy=BigRefunds(), featurize=by_amount, log=log)
    g = lc_agent([("issue_refund", {"amount": 5}), ("issue_refund", {"amount": 500}), ("lookup_order", {"order_id": "A"})], middleware=[mw])
    out, cfg = start(g, "mw1")
    (intr,) = out["__interrupt__"]
    assert [a["args"] for a in intr.value["action_requests"]] == [{"amount": 500}]  # official HITLRequest, small one not asked
    assert RAN == [] and len(log) == 0
    out = g.invoke(Command(resume={"decisions": [{"type": "reject", "message": "too much"}]}), cfg)
    assert sorted(RAN) == [("issue_refund", 5), ("lookup_order", "A")]
    assert [(r.features["big"], r.decision, r.human_outcome) for r in log.records] == [(True, "ask", "reject"), (False, "auto", None)]
    assert any(isinstance(m, ToolMessage) and "too much" in m.content for m in out["messages"])


def test_middleware_edit_shadow_and_resume_determinism():
    from hitl_gate.middleware import GatedHumanInTheLoopMiddleware

    coin, log = Coin(), DecisionLog()
    mw = GatedHumanInTheLoopMiddleware({"issue_refund": True}, policy=coin, log=log)
    g = lc_agent([("issue_refund", {"amount": 500})], middleware=[mw])
    for i in range(20):
        out, cfg = start(g, f"md{i}")
        if "__interrupt__" in out:
            g.invoke(Command(resume={"decisions": [{"type": "edit", "edited_action": {"name": "issue_refund", "args": {"amount": 7}}}]}), cfg)
    assert coin.calls == 20 and len(log) == 20 and not mw.gate._pending
    edits = sum(r.human_outcome == "edit" for r in log.records)
    assert 0 < edits < 20 and RAN.count(("issue_refund", 7)) == edits

    shadow_log = DecisionLog()
    mw = GatedHumanInTheLoopMiddleware({"issue_refund": True}, policy=NeverAsk(), log=shadow_log, mode="shadow")
    g = lc_agent([("issue_refund", {"amount": 1})], middleware=[mw])
    run_episode(g, {"messages": [HumanMessage("hi")]}, approve, "sh")
    assert [(r.decision, r.mode, r.human_outcome) for r in shadow_log.records] == [("auto", "shadow", "approve")]
    with pytest.raises(ValueError, match="when"):
        GatedHumanInTheLoopMiddleware({"x": {"allowed_decisions": ["approve"], "when": bool}}, policy=NeverAsk(), log=log)
