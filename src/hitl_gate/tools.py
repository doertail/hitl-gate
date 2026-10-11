"""Gated tool-call approval: ask a human only for the tool calls the policy flags.

Request / response shapes follow LangChain's ``HumanInTheLoopMiddleware``
(``HITLRequest`` / ``HITLResponse``), as plain dicts, so the core package needs
no ``langchain`` install and existing approval UIs keep working::

    interrupt value: {"action_requests": [{"name", "args", "description"}],
                      "review_configs": [{"action_name", "allowed_decisions"}],
                      "hitl_gate": {"gate_id", "reason", "mode", "tool_call_id"}}
    resume value:    {"decisions": [{"type": "approve"}
                                    | {"type": "edit", "edited_action": {"name", "args"}}
                                    | {"type": "reject", "message": str}
                                    | {"type": "respond", "message": str}]}

Each decision may carry ``"review_seconds"``; it is copied into the log.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Literal, Sequence

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.config import get_config
from langgraph.types import interrupt

from .gate import GateDecider, make_record
from .log import DecisionLog
from .policies import Policy

DECISION_TYPES = ("approve", "edit", "reject", "respond")
Featurize = Callable[[str, dict], dict]


def _thread_id() -> str:
    try:
        return str(get_config().get("configurable", {}).get("thread_id", ""))
    except RuntimeError:  # outside a runnable context
        return ""


def reject_message(tool_name: str, tool_call_id: str, reason: str | None) -> str:
    """Same wording as LangChain's ``HumanInTheLoopMiddleware`` uses for a rejection."""
    if reason:
        return f"User rejected the tool call for `{tool_name}` with reason: {reason}"
    return (
        f"User rejected the tool call for `{tool_name}` with id {tool_call_id}. The tool was not "
        "executed. Do not retry this tool call unless the user explicitly requests it."
    )


class ToolGate:
    """Shared ask/auto bookkeeping for :func:`gated_tool` and the LangChain middleware.

    A decision is keyed by ``(thread_id, gate_id, tool_call_id)``. Audit sampling is
    a stable hash of ``(seed, gate_id, thread_id:tool_call_id)``, and the decision is
    also cached in memory until the call is logged, so a node re-executed on resume
    reaches the same decision even if the policy is random or was swapped meanwhile.
    """

    def __init__(self, policy: Policy, featurize: Featurize, log: DecisionLog,
                 mode: Literal["live", "shadow"] = "live", audit_rate: float = 0.0, seed: int | None = None):
        self.decider = GateDecider(policy, mode=mode, audit_rate=audit_rate, seed=seed)
        self.featurize = featurize
        self.log = log
        # ponytail: in-process cache; a call interrupted and never resumed stays here. Bounded by open interrupts.
        self._pending: dict[tuple[str, str, str], dict] = {}

    @property
    def policy(self) -> Policy:
        return self.decider.policy

    @policy.setter
    def policy(self, policy: Policy) -> None:  # swap in a refitted policy
        self.decider.policy = policy

    def decide(self, gate_id: str, name: str, args: dict, tool_call_id: str, thread_id: str) -> dict:
        key = (thread_id, gate_id, tool_call_id)
        if key not in self._pending:
            self._pending[key] = self.decider(gate_id, self.featurize(name, args), key=f"{thread_id}:{tool_call_id}")
        return self._pending[key]

    def finish(self, gate_id: str, tool_call_id: str, thread_id: str, answer: dict | None) -> None:
        """Log the call once (``run_id`` = ``tool_call_id``) and drop it from the cache."""
        d = self._pending.pop((thread_id, gate_id, tool_call_id))
        outcome = None if answer is None else {"outcome": answer["type"], "review_seconds": answer.get("review_seconds")}
        self.log.append(make_record(tool_call_id, gate_id, d, outcome))


def hitl_request(name: str, args: dict, tool_call_id: str, gate_id: str, d: dict, allowed: Sequence[str]) -> dict:
    return {
        "action_requests": [{"name": name, "args": args, "description": f"Tool execution requires approval ({d['reason']})\n\nTool: {name}\nArgs: {args}"}],
        "review_configs": [{"action_name": name, "allowed_decisions": list(allowed)}],
        "hitl_gate": {"gate_id": gate_id, "reason": d["reason"], "mode": d["mode"], "tool_call_id": tool_call_id},
    }


def _model_args(tool: BaseTool, args: dict) -> dict:
    """Args the model produced, i.e. without values ToolNode injected (state, runtime, ...)."""
    schema = tool.tool_call_schema
    fields = schema.get("properties", {}) if isinstance(schema, dict) else schema.model_fields
    return {k: v for k, v in args.items() if k in fields}


class GatedTool(BaseTool):
    """A tool that asks the gate before running. Build it with :func:`gated_tool`."""

    inner: BaseTool
    gate: Any  # ToolGate
    gate_id: str
    allowed_decisions: tuple = DECISION_TYPES
    edit_notice: str | None = None

    def _run(self, *args, **kwargs):  # pragma: no cover - invoke is overridden
        raise NotImplementedError

    def _prepare(self, input: Any) -> tuple[dict, dict | None]:
        """Decide (maybe interrupt), log, and return ``(call to run, human decision or None)``.

        Logged before the tool runs: the record is about the gate decision, and the
        tool's own success or failure is a downstream outcome (``mark_downstream``).
        """
        if not (isinstance(input, dict) and input.get("type") == "tool_call" and input.get("id")):
            raise TypeError(f"gated tool {self.name!r} must be invoked with a ToolCall (as ToolNode does)")
        call_id, args, thread = input["id"], input["args"], _thread_id()
        visible = _model_args(self.inner, args)
        d = self.gate.decide(self.gate_id, self.name, visible, call_id, thread)
        if not d["ask"]:
            self.gate.finish(self.gate_id, call_id, thread, None)
            return input, None
        resume = interrupt(hitl_request(self.name, visible, call_id, self.gate_id, d, self.allowed_decisions))
        decisions = resume.get("decisions") if isinstance(resume, dict) else None
        if not decisions or len(decisions) != 1:
            raise ValueError(f"expected {{'decisions': [one decision]}} for tool call {call_id}, got {resume!r}")
        answer = decisions[0]
        if answer.get("type") not in self.allowed_decisions:
            raise ValueError(f"decision type {answer.get('type')!r} not allowed for {self.name!r}; allowed: {self.allowed_decisions}")
        if answer["type"] == "edit":
            edited = answer["edited_action"]
            if edited.get("name", self.name) != self.name:
                raise ValueError(f"gated_tool cannot redirect {self.name!r} to another tool ({edited['name']!r})")
            input = {**input, "args": {**args, **edited["args"]}}
        self.gate.finish(self.gate_id, call_id, thread, answer)
        return input, answer

    def _finish(self, call: dict, answer: dict | None, result: Any) -> Any:
        if _skips_tool(answer):
            reject = answer["type"] == "reject"
            return ToolMessage(
                content=reject_message(self.name, call["id"], answer.get("message")) if reject else answer["message"],
                name=self.name, tool_call_id=call["id"], status="error" if reject else "success",
            )
        if answer and answer["type"] == "edit" and self.edit_notice and isinstance(result, ToolMessage):
            note = f"{self.edit_notice} Executed with arguments {json.dumps(_model_args(self.inner, call['args']), default=str)}."
            result = result.model_copy(update={"content": f"{note}\n\nTool response:\n{result.content}"})
        return result

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        call, answer = self._prepare(input)
        return self._finish(call, answer, None if _skips_tool(answer) else self.inner.invoke(call, config, **kwargs))

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        call, answer = self._prepare(input)
        return self._finish(call, answer, None if _skips_tool(answer) else await self.inner.ainvoke(call, config, **kwargs))


def _skips_tool(answer: dict | None) -> bool:
    return answer is not None and answer["type"] in ("reject", "respond")


EDIT_NOTICE = "Note: a human reviewer edited this tool call before it ran."


def gated_tool(
    tool: BaseTool | Callable,
    *,
    policy: Policy,
    log: DecisionLog,
    gate_id: str | None = None,
    featurize: Featurize = lambda name, args: {},
    mode: Literal["live", "shadow"] = "live",
    audit_rate: float = 0.0,
    seed: int | None = None,
    allowed_decisions: Sequence[str] = DECISION_TYPES,
    edit_notice: str | None = EDIT_NOTICE,
) -> GatedTool:
    """Wrap ``tool`` so each call is gated: auto runs it, ask interrupts for a human decision.

    Use the result anywhere a tool goes (``ToolNode``, ``create_agent(tools=...)``).
    The graph needs a checkpointer. ``featurize(tool_name, args)`` returns the
    discrete features the policy sees (``args`` excludes injected arguments).
    ``gate_id`` defaults to the tool name. One :class:`DecisionRecord` is logged per
    call with ``run_id = tool_call_id``, so mark a bad outcome later with
    ``log.mark_downstream(tool_message.tool_call_id, gate_id, True)``.

    Resume with ``Command(resume={"decisions": [decision]})`` (see the module doc).
    ``reject`` and ``respond`` return a ``ToolMessage`` to the model without running
    the tool; ``edit`` runs it with the edited args (``edit_notice`` is prepended to
    the result so the model knows; ``None`` disables it).

    Several gated calls in one model turn: each must run as its own task (as
    ``create_agent`` and ``create_react_agent`` do via ``Send``), because
    interrupts inside one node are matched by order, which is racy for parallel calls.
    """
    from langchain_core.tools import tool as as_tool

    if not isinstance(tool, BaseTool):
        tool = as_tool(tool)
    bad = set(allowed_decisions) - set(DECISION_TYPES)
    if bad or not allowed_decisions:
        raise ValueError(f"allowed_decisions must be a non-empty subset of {DECISION_TYPES}, got {allowed_decisions!r}")
    return GatedTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        return_direct=tool.return_direct,
        inner=tool,
        gate=ToolGate(policy, featurize, log, mode=mode, audit_rate=audit_rate, seed=seed),
        gate_id=gate_id or tool.name,
        allowed_decisions=tuple(allowed_decisions),
        edit_notice=edit_notice,
    )
