"""
Voice mode changes HOW the agent writes, and nothing else.

When the customer is talking, the reply is read aloud word for word, so it is
written for the ear: no lists, no price sheet, a figure and a question. Every
rule the graph enforces — the guard, the router, grounding, injection
resistance, the price check — is untouched; the spoken style is one more block
added to the same system prompt, and the checks run on its output as before.
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app as web
from agent.nodes import answer_check, handoff, knowledge
from schemas import persona


# ── The prompt: one extra block, everything else identical ───────────────────

def test_spoken_mode_adds_the_spoken_block_and_keeps_every_rule():
    typed = persona.system_prompt(reference="REF", extra="TASK")
    spoken = persona.system_prompt(reference="REF", extra="TASK", spoken=True)
    assert persona.SPOKEN not in typed
    assert persona.SPOKEN in spoken
    for rule in (persona.VOICE, persona.GROUNDING, persona.INJECTION_RESISTANCE, persona.HANDLING_NOTES):
        assert rule in spoken, "voice mode must never drop a rule"
    assert spoken.replace(persona.SPOKEN + "\n\n", "") == typed


def test_the_reference_material_stays_last_and_fenced_in_spoken_mode():
    spoken = persona.system_prompt(reference="REF", extra="TASK", spoken=True)
    assert spoken.rstrip().endswith("REFERENCE>>>")


class CapturingModel:
    def __init__(self, reply="Sure."):
        self.prompts, self.reply = [], reply

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        return AIMessage(content=self.reply)


@pytest.fixture
def model(monkeypatch):
    fake = CapturingModel()
    monkeypatch.setattr(knowledge, "get_model", lambda *_: fake)
    monkeypatch.setattr(handoff, "get_model", lambda *_: fake)
    return fake


STATE = {"messages": [HumanMessage(content="what are your rates")]}


@pytest.mark.parametrize("node", [knowledge.answer, handoff.handoff])
def test_a_spoken_turn_is_written_for_the_ear(model, node):
    node({**STATE, "spoken": True})
    assert persona.SPOKEN in model.prompts[-1]


@pytest.mark.parametrize("node", [knowledge.answer, handoff.handoff])
def test_a_typed_turn_is_written_exactly_as_before(model, node):
    node({**STATE, "spoken": False})
    node(STATE)
    assert all(persona.SPOKEN not in p for p in model.prompts)


# ── The flag comes from the request, per turn ────────────────────────────────

class _NotPaused:
    def get_state(self, cfg):
        return SimpleNamespace(tasks=[])


def test_a_voice_turn_marks_the_graph_input_as_spoken():
    graph_input, _ = web._graph_input(_NotPaused(), {}, web.Turn(message="hi", voice=True))
    assert graph_input["spoken"] is True


def test_a_typed_turn_marks_it_as_not_spoken():
    """Explicitly False, so switching voice off mid-conversation switches the style back."""
    graph_input, _ = web._graph_input(_NotPaused(), {}, web.Turn(message="hi"))
    assert graph_input["spoken"] is False


# ── The price check still sees every amount, however it is written ───────────

@pytest.mark.parametrize("said", [
    "Two movers is 115 an hour.",
    "It's 115 dollars an hour for two movers.",
    "That's 400 bucks.",
    "We charge 999 per hour.",
    "It's a hundred and fifteen dollars an hour.",
])
def test_an_amount_without_a_dollar_sign_is_sent_back(said):
    assert answer_check.find_amount_without_sign(said)


@pytest.mark.parametrize("said", [
    "Two movers is $115 an hour.",
    "There's a 3-hour minimum.",
    "We'd send 2 movers and a 26ft truck.",
    "Call us on (323) 645-2636, any day from 6am.",
    "That usually takes about 4 hours.",
])
def test_ordinary_numbers_are_left_alone(said):
    assert answer_check.find_amount_without_sign(said) is None


def test_the_check_node_rejects_a_spoken_amount_without_a_sign(monkeypatch):
    monkeypatch.setattr(answer_check, "find_unpublished_price", lambda text: None)
    out = answer_check.check({"draft": "Two movers is 115 an hour.", "answer_attempts": 1})
    assert out.get("answer_complaint"), "a figure the price check cannot see must not reach a customer"
