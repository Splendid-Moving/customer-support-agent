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

    # Enter the interview directly, skipping the model-backed guard, router and
    # prefill — this test is about the pause, not about routing.
    graph.invoke({"intent": "estimate", "messages": []}, cfg, interrupt_before=["guard"])
    graph.update_state(cfg, {"lead_type": "estimate", "lead": {}}, as_node="prefill")
    graph.invoke(None, cfg)

    first = web._pending_interrupt(graph, cfg)
    assert first is not None, "first question not seen"
    assert first["field"]["name"] == "name"

    # A blank answer to a required question re-asks inside the same node — a
    # second interrupt, same invocation. This is the case that used to look
    # "not paused" and silently restart the whole conversation.
    graph.invoke(Command(resume={"answer": "  "}), cfg)

    snapshot = graph.get_state(cfg)
    pending = web._pending_interrupt(graph, cfg)

    assert pending is not None, "second interrupt reported as not paused"
    assert pending["field"]["name"] == "name", "moved on despite a blank answer"
    # The whole point: `next` is unreliable here, `tasks` is not.
    assert not snapshot.next, (
        "snapshot.next is populated after all — if LangGraph has changed this, "
        "the comment in app._pending_interrupt needs revisiting"
    )
