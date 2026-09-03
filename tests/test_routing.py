"""
The routing functions and the graph's shape.

No models here — these are the pure functions that decide where a turn goes, plus
a set of structural assertions about the graph itself. Those structural tests
matter more than they look: they are what stops someone adding a lane next year
that quietly bypasses the guard, or wiring the email somewhere it can be reached
without a human filling in a form.
"""

from agent import graph as graph_module
from agent.nodes import refuse, router


def test_both_lead_types_share_the_one_lane():
    """prefill first, so the interview can skip what they already told us."""
    assert router.pick_lane({"intent": "estimate"}) == "prefill"
    assert router.pick_lane({"intent": "long_distance"}) == "prefill"


def test_the_other_lanes_map_to_themselves():
    assert router.pick_lane({"intent": "knowledge"}) == "knowledge"
    assert router.pick_lane({"intent": "handoff"}) == "handoff"


def test_an_unset_intent_falls_back_to_answering_the_question():
    """Never to the form — opening one on someone who did not ask loses the job."""
    assert router.pick_lane({}) == "knowledge"
    assert router.pick_lane({"intent": "nonsense"}) == "knowledge"


# ── Structure ──────────────────────────────────────────────────────────────────

def _edges():
    compiled = graph_module.build_graph()
    return [(e.source, e.target) for e in compiled.get_graph().edges]


def test_the_guard_is_the_only_way_in():
    sources_from_start = {t for s, t in _edges() if s == "__start__"}
    assert sources_from_start == {"guard"}


def test_nothing_reaches_a_lane_without_passing_the_guard():
    """Every lane's only inbound edge comes from the router or its own loop."""
    edges = _edges()
    for lane in ("knowledge", "prefill", "handoff"):
        inbound = {s for s, t in edges if t == lane}
        assert inbound <= {"router", "answer_check"}, f"{lane} reachable from {inbound}"
    # collect_lead is one step further in, behind prefill.
    assert {s for s, t in edges if t == "collect_lead"} == {"prefill"}


def test_the_email_is_not_reachable_from_the_router():
    """
    submit_lead must sit behind the form, which a human fills in by hand. If a
    router edge ever points at it directly, no message can be trusted again.
    """
    inbound = {s for s, t in _edges() if t == "submit_lead"}
    assert inbound == {"collect_lead"}


def test_the_rewrite_loop_can_exit():
    edges = _edges()
    assert ("knowledge", "answer_check") in edges
    assert ("answer_check", "__end__") in edges


# ── Refusals ───────────────────────────────────────────────────────────────────

def test_every_guard_verdict_has_a_refusal_written_for_it():
    for verdict in ("injection", "off_topic", "too_long", "turn_limit"):
        assert verdict in refuse.REPLIES


def test_a_refusal_never_explains_the_rules_it_just_enforced():
    """Someone probing should learn nothing; a confused customer should see a person."""
    tell_tales = ("prompt", "instruction", "injection", "system", "policy", "violat", "detect")
    for verdict, text in refuse.REPLIES.items():
        lowered = text.lower()
        for word in tell_tales:
            assert word not in lowered, f"{verdict} reply mentions {word!r}"


def test_refusing_costs_no_model_call():
    result = refuse.refuse({"guard": "injection"})
    assert result["messages"][0].content == refuse.REPLIES["injection"]
