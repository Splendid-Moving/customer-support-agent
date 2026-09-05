"""
Graph wiring. The whole topology lives here, readable top to bottom.

    START
      |
      v
    guard --- blocked ---> refuse ------------------------> END
      | clean
      v
    router --+--> knowledge --> answer_check --> END
             |        ^              |
             |        +-- rewrite ---+
             |
             +--> prefill --> collect_lead  [PAUSES per question]
             |                     |
             |                     +--> submit_lead --> END
             |                     +-- cancelled -----> END
             |
             +--> handoff ----------------------------------> END

An estimate, an out-of-state move and a question for the office all take the
prefill lane; they are three field lists behind one interview and one email.

Two things about this shape are deliberate and worth keeping.

**The guard is the only entry.** Every turn goes through it before anything else
runs. Not a check inside each lane — one gate, in front of all of them, so a new
lane cannot be added that quietly bypasses it.

**The only side effect is the last node of the longest path.** `submit_lead` is
reachable only by passing the guard, being routed to the form, and filling that
form in by hand. There is no sequence of words a customer can type that reaches
it directly, which is what makes a public-facing agent safe to leave running.
"""

import logging

from langgraph.graph import END, START, StateGraph

from agent.nodes import answer_check as answer_check_node
from agent.nodes import collect_lead as collect_lead_node
from agent.nodes import guard as guard_node
from agent.nodes import handoff as handoff_node
from agent.nodes import knowledge as knowledge_node
from agent.nodes import prefill as prefill_node
from agent.nodes import refuse as refuse_node
from agent.nodes import router as router_node
from agent.nodes import submit_lead as submit_lead_node
from agent.state import SupportState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def build_graph(checkpointer=None):
    """
    Compile the graph.

    A checkpointer is REQUIRED for the form to work — without one the graph
    cannot pause and resume, and `collect_lead` raises. It also provides the
    conversation memory, keyed by thread_id.

    Passing None lets LangGraph Server inject its own managed checkpointer, which
    is what `langgraph dev` does.
    """
    builder = StateGraph(SupportState)

    builder.add_node("guard", guard_node.guard)
    builder.add_node("refuse", refuse_node.refuse)
    builder.add_node("router", router_node.route)

    builder.add_node("knowledge", knowledge_node.answer)
    builder.add_node("answer_check", answer_check_node.check)

    builder.add_node("prefill", prefill_node.prefill)
    builder.add_node("collect_lead", collect_lead_node.collect_lead)
    builder.add_node("submit_lead", submit_lead_node.submit_lead)

    builder.add_node("handoff", handoff_node.handoff)

    builder.add_edge(START, "guard")
    builder.add_conditional_edges(
        "guard", guard_node.next_step, {"router": "router", "refuse": "refuse"}
    )
    builder.add_edge("refuse", END)

    builder.add_conditional_edges(
        "router",
        router_node.pick_lane,
        {"knowledge": "knowledge", "prefill": "prefill", "handoff": "handoff"},
    )
    builder.add_edge("prefill", "collect_lead")

    # The rewrite loop. answer_check either commits the draft as a message and
    # ends, or sends it back with one specific complaint. It gives up on its own
    # after MAX_ATTEMPTS, so this cannot spin.
    builder.add_edge("knowledge", "answer_check")
    builder.add_conditional_edges(
        "answer_check",
        answer_check_node.next_step,
        {"knowledge": "knowledge", "__end__": END},
    )

    # collect_lead returns a Command and routes itself to submit_lead or END.
    builder.add_edge("submit_lead", END)
    builder.add_edge("handoff", END)

    return builder.compile(checkpointer=checkpointer)


#: Entry point referenced by langgraph.json.
graph = build_graph()
