"""Tool-call approval for a support agent, offline (scripted model, simulated reviewer).

Tools: ``lookup_order`` (never gated), ``issue_refund(order_id, amount)`` and
``send_email(to, body)`` (gated). Phase 1 asks a human for every gated call and logs
the answers. A RulePolicy fitted on that log then auto-approves small refunds,
while large refunds and emails still go to a human.

    python examples/tool_approval/run.py      # ~5 s, seeded
"""

import random
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from hitl_gate import AlwaysAsk, DecisionLog, RulePolicy, gated_tool
from hitl_gate.sim import run_episode


class ScriptedSupportModel(BaseChatModel):
    """Stands in for an LLM: lookup -> refund -> email -> final answer, one tool call per turn."""

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        order, amount = re.match(r"refund (\S+) (\d+)", messages[0].content).groups()
        step = sum(isinstance(m, ToolMessage) for m in messages)
        plan = [
            ("lookup_order", {"order_id": order}),
            ("issue_refund", {"order_id": order, "amount": int(amount)}),
            ("send_email", {"to": "customer@example.com", "body": f"Refunded {amount} for {order}."}),
        ]
        msg = AIMessage("Done.") if step >= len(plan) else AIMessage("", tool_calls=[{"name": plan[step][0], "args": plan[step][1], "id": f"{order}-{step}"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self):
        return "scripted"


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return f"{order_id}: delivered"


@tool
def issue_refund(order_id: str, amount: int) -> str:
    """Refund ``amount`` for an order."""
    return f"refunded {amount} on {order_id}"


@tool
def send_email(to: str, body: str) -> str:
    """Email the customer."""
    return "sent"


def featurize(name, args):
    if name != "issue_refund":
        return {}
    a = args["amount"]
    return {"amount": "<=50" if a <= 50 else "50-200" if a <= 200 else ">200"}


rng = random.Random(0)


def reviewer(request):
    """Simulated human: small refunds always fine, large ones and emails sometimes edited."""
    action = request["action_requests"][0]
    if action["name"] == "issue_refund":
        a = action["args"]["amount"]
        p_edit = 0.0 if a <= 50 else 0.1 if a <= 200 else 0.3
        if rng.random() < p_edit:
            return {"decisions": [{"type": "edit", "edited_action": {"name": "issue_refund", "args": {**action["args"], "amount": a // 2}}}]}
    elif rng.random() < 0.15:
        return {"decisions": [{"type": "edit", "edited_action": {"name": "send_email", "args": {**action["args"], "body": "Sorry for the trouble. " + action["args"]["body"]}}}]}
    return {"decisions": [{"type": "approve"}]}


def build(log):
    gated = [gated_tool(t, policy=AlwaysAsk(), featurize=featurize, log=log, audit_rate=0.05, seed=0) for t in (issue_refund, send_email)]
    model = ScriptedSupportModel()
    b = StateGraph(MessagesState)
    b.add_node("model", lambda s: {"messages": [model.invoke(s["messages"])]})
    b.add_node("tools", ToolNode([lookup_order, *gated]))
    b.add_edge(START, "model")
    b.add_conditional_edges("model", tools_condition, {"tools": "tools", END: END})
    b.add_edge("tools", "model")
    return b.compile(checkpointer=MemorySaver()), gated


def run(graph, phase, n):
    data = random.Random(phase)
    for i in range(n):
        amount = data.choice([10, 20, 35, 50]) if data.random() < 0.7 else data.randint(51, 900)
        run_episode(graph, {"messages": [HumanMessage(f"refund {phase}{i} {amount}")]}, reviewer, f"{phase}{i}")


def summary(log, phase):
    recs = [r for r in log.records if r.run_id.startswith(phase)]
    for gate in ("issue_refund", "send_email"):
        rs = [r for r in recs if r.gate_id == gate]
        print(f"  {gate:<13} calls={len(rs):>3}  asked={sum(r.asked for r in rs):>3}  edited={sum(bool(r.changed) for r in rs):>3}")


def main():
    log = DecisionLog(clock=lambda: 0.0)
    graph, gated = build(log)

    print("Phase 1: AlwaysAsk, 200 conversations")
    run(graph, "p1-", 200)
    summary(log, "p1-")

    policy = RulePolicy.fit(log.records, keys=["amount"], min_count=30, max_change_rate=0.02)
    print("\nFitted RulePolicy:\n" + policy.describe())
    for t in gated:
        t.gate.policy = policy

    print("\nPhase 2: RulePolicy + 5% audits, 200 conversations")
    run(graph, "p2-", 200)
    summary(log, "p2-")
    small = [r for r in log.records if r.run_id.startswith("p2-") and r.features.get("amount") == "<=50"]
    print(f"\nSmall refunds in phase 2: {len(small)}, auto={sum(not r.asked for r in small)}, audited={sum(r.mode == 'audit' for r in small)}")


if __name__ == "__main__":
    main()
