"""
Regression tests for two bugs that both failed SILENTLY.

Neither raised anything a customer or a log would show. Both were found by
walking the app by hand, which is exactly why they are pinned here — the next
person to touch these files will not know to look.
"""

import inspect

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import app as web
from agent.graph import build_graph
from agent.nodes import submit_lead as submit_node


def test_the_config_parameter_is_named_config():
    """
    LangGraph injects RunnableConfig by PARAMETER NAME.

    Renaming it to anything else (it was `runnable_config`) means it is never
    supplied, the node raises TypeError, LangGraph rolls the whole superstep
    back, and the customer is shown the empty form again with no error anywhere.
    """
    params = list(inspect.signature(submit_node.submit_lead).parameters)
    assert params == ["state", "config"], params


def test_a_second_interrupt_in_the_same_node_still_reads_as_paused():
    """
    `snapshot.next` is EMPTY when the form has been re-shown with validation
    errors, even though the graph is stopped. Reading paused-ness off `next`
    made the server treat the corrected form as a new chat message, which
    restarts the graph and throws the form away without erroring.

    This walks a real graph to the second interrupt and asserts the helper the
    server actually uses still reports it as paused.
    """
    graph = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "wiring-test"}}

    # Enter the form lane directly, skipping the model-backed guard and router.
    graph.invoke({"intent": "estimate", "messages": []}, cfg, interrupt_before=["guard"])
    graph.update_state(cfg, {"intent": "estimate"}, as_node="router")
    graph.invoke(None, cfg)

    assert web._pending_interrupt(graph, cfg) is not None, "first interrupt not seen"

    # A submission that fails validation re-interrupts inside the same node.
    graph.invoke(Command(resume={"values": {"name": "J"}, "photo_ids": []}), cfg)

    snapshot = graph.get_state(cfg)
    pending = web._pending_interrupt(graph, cfg)

    assert pending is not None, "second interrupt reported as not paused"
    assert pending["errors"], "the re-shown form carries no validation errors"
    # The whole point: `next` is unreliable here, `tasks` is not.
    assert not snapshot.next, (
        "snapshot.next is populated after all — if LangGraph has changed this, "
        "the comment in app._pending_interrupt needs revisiting"
    )
