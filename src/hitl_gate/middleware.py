"""LangChain v1 ``create_agent`` integration (``pip install "hitl-gate[langchain]"``, langchain>=1.4)."""

from __future__ import annotations

from typing import Any, Literal

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
from langchain_core.messages import AIMessage

from .log import DecisionLog
from .policies import Policy
from .tools import Featurize, ToolGate, _thread_id

if "when" not in InterruptOnConfig.__annotations__ or not hasattr(HumanInTheLoopMiddleware, "_process_decision"):
    raise ImportError("hitl_gate.middleware needs langchain>=1.4 (tested with 1.4.4)")


class GatedHumanInTheLoopMiddleware(HumanInTheLoopMiddleware):
    """``HumanInTheLoopMiddleware`` that interrupts only for the tool calls the policy flags.

    ``interrupt_on`` lists the gated tools exactly as for the official middleware
    (``True`` or an ``InterruptOnConfig`` without ``when``; the policy is the ``when``).
    Interrupt payloads, resume format and edit/reject/respond handling are the
    official ones, so existing approval UIs keep working.

    One :class:`DecisionRecord` per gated tool call: ``gate_id`` = tool name,
    ``run_id`` = tool call id. Calls to tools not in ``interrupt_on`` are not gated
    or logged. Swap policies with ``middleware.gate.policy = ...``.
    """

    def __init__(
        self,
        interrupt_on: dict[str, bool | InterruptOnConfig],
        *,
        policy: Policy,
        log: DecisionLog,
        featurize: Featurize = lambda name, args: {},
        mode: Literal["live", "shadow"] = "live",
        audit_rate: float = 0.0,
        seed: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(interrupt_on, **kwargs)
        self.gate = ToolGate(policy, featurize, log, mode=mode, audit_rate=audit_rate, seed=seed)
        for name, cfg in self.interrupt_on.items():
            if "when" in cfg:
                raise ValueError(f"{name!r}: 'when' is replaced by the policy; remove it")
            self.interrupt_on[name] = {**cfg, "when": self._when}

    def _decide(self, tool_call: dict) -> dict:
        return self.gate.decide(tool_call["name"], tool_call["name"], tool_call["args"], tool_call["id"], _thread_id())

    def _when(self, request: Any) -> bool:
        return self._decide(request.tool_call)["ask"]

    def _process_decision(self, decision, tool_call, config):  # type: ignore[override]
        out = super()._process_decision(decision, tool_call, config)  # validates first
        self.gate.finish(tool_call["name"], tool_call["id"], _thread_id(), decision)
        return out

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        last = next((m for m in reversed(state.get("messages") or []) if isinstance(m, AIMessage)), None)
        calls = [tc for tc in (last.tool_calls if last else []) if tc["name"] in self.interrupt_on]
        autos = [tc for tc in calls if not self._decide(tc)["ask"]]  # decide before super pops asked ones
        result = super().after_model(state, runtime)  # may interrupt; asked calls are logged in _process_decision
        for tc in autos:
            self.gate.finish(tc["name"], tc["id"], _thread_id(), None)
        return result
