"""Drive a compiled graph through runs with a simulated (or scripted) reviewer.

The graph must be compiled with a checkpointer, e.g.
``builder.compile(checkpointer=MemorySaver())`` — interrupts cannot resume without one.
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.types import Command


def run_episode(graph, inputs: dict, reviewer: Callable[[Any], dict], thread_id: str) -> dict:
    """Invoke ``graph`` and answer every interrupt with ``reviewer(interrupt_value)`` until it finishes.

    Returns the final state values. With several simultaneous interrupts (parallel
    branches) each one is answered and resumed by interrupt id.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(inputs, config)
    while True:
        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if not interrupts:
            interrupts = [i for t in graph.get_state(config).tasks for i in t.interrupts]
        if not interrupts:
            return result
        if len(interrupts) == 1:
            resume = reviewer(interrupts[0].value)
        else:
            resume = {i.id: reviewer(i.value) for i in interrupts}
        result = graph.invoke(Command(resume=resume), config)
