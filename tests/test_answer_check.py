"""
The voice and grounding rules, which are the two things a customer notices.

These run on strings, not on model output, so they are fast and exact. The
phrasings in THIRD_PERSON are the ones a model actually produces when it slips —
each was chosen because it reads perfectly fine until you notice the agent has
stepped outside the company to describe it.
"""

import pytest
from langchain_core.messages import AIMessage

from agent.nodes import answer_check

KB = """\
# Rates

2 movers + truck: $115 - $125 per hour.
3 movers + truck: $145 - $155 per hour.
There is a 3 hour minimum. New bookings take a $50 deposit.
"""


@pytest.fixture(autouse=True)
def kb(knowledge_dir):
    knowledge_dir("rates.md", KB)


# ── First person ───────────────────────────────────────────────────────────────

THIRD_PERSON = [
    "Splendid Moving offers free estimates on every job.",
    "Splendid Moving's rates start at $115 per hour.",
    "Moving companies usually charge a 3 hour minimum.",
    "Most movers include shrink wrap these days.",
    "In the moving industry that's standard practice.",
    "The company requires a deposit to hold the date.",
    "They provide blankets and wardrobe boxes at no charge.",
    "Their rates depend on the crew size.",
]


@pytest.mark.parametrize("draft", THIRD_PERSON)
def test_third_person_is_rejected(draft):
    assert answer_check.find_third_person(draft) is not None, f"allowed: {draft!r}"


FIRST_PERSON = [
    "We bring a 26ft truck and it's included in the hourly rate.",
    "Our rate for two movers is $115 to $125 an hour, with a 3 hour minimum.",
    "I can get a manager to put together a real estimate for you.",
    "We're open every day from 6am, so a weekend move is no problem.",
    "Here at Splendid Moving we include shrink wrap and tape.",
    "That's us — we cover Pasadena and most of the Valley.",
    "They can be tricky, so send me a photo and I'll check.",
]


@pytest.mark.parametrize("draft", FIRST_PERSON)
def test_first_person_answers_pass(draft):
    caught = answer_check.find_third_person(draft)
    assert caught is None, f"false positive on {draft!r} ({caught})"


# ── Prices ─────────────────────────────────────────────────────────────────────

def test_published_rates_pass():
    assert answer_check.find_unpublished_price(
        "Two movers is $115 to $125 an hour and the deposit is $50."
    ) is None


def test_price_formatting_differences_are_not_a_failure():
    """The model writes '$ 125.00' and the file says '$125'. Same number."""
    assert answer_check.find_unpublished_price("It's $ 125.00 an hour.") is None


def test_an_invented_total_is_rejected():
    complaint = answer_check.find_unpublished_price(
        "For a 2 bedroom that usually works out around $1,200 all in."
    )
    assert complaint is not None
    assert "$1200" in complaint


def test_arithmetic_on_our_own_rates_is_still_rejected():
    """$125 x 3 hours is the single most expensive thing it could volunteer."""
    assert answer_check.find_unpublished_price("Three hours at $125 comes to $375.") is not None


# ── The node ───────────────────────────────────────────────────────────────────

def test_a_clean_draft_becomes_the_reply():
    result = answer_check.check({"draft": "We bring a 26ft truck.", "answer_attempts": 1})
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "We bring a 26ft truck."
    assert answer_check.next_step(result) == "__end__"


def test_a_bad_draft_is_sent_back_and_never_becomes_a_message():
    result = answer_check.check(
        {"draft": "Splendid Moving offers free estimates.", "answer_attempts": 1}
    )
    assert "messages" not in result
    assert result["answer_complaint"]
    assert answer_check.next_step(result) == "knowledge"


def test_it_gives_up_rather_than_looping():
    result = answer_check.check(
        {"draft": "Splendid Moving offers free estimates.", "answer_attempts": answer_check.MAX_ATTEMPTS}
    )
    reply = result["messages"][0].content
    assert "645-2636" in reply
    assert "Splendid Moving offers" not in reply
    assert answer_check.next_step(result) == "__end__"


def test_an_empty_draft_is_treated_as_a_failure():
    result = answer_check.check({"draft": "", "answer_attempts": 1})
    assert result.get("answer_complaint")
