"""
Reading what the customer already told us — and refusing to invent the rest.

The check tested here exists because of one live conversation. A customer typed
"do you do haul away" and nothing else. The agent offered to ask the office, they
said yes, and the read-back showed the office would be asked "How much extra to
haul away a fridge." — a fridge nobody had mentioned, in a sentence that was
word for word the first example in the extraction prompt.

An example in a prompt is an answer the model is allowed to give, and where the
real answer is thin it will give that one. The examples are gone; this is the
part that catches the next one.
"""

from langchain_core.messages import AIMessage, HumanMessage

from agent.nodes import prefill

#: The conversation exactly as it happened, minus a fridge.
HAUL_AWAY = [
    HumanMessage(content="Do you do haul away"),
    AIMessage(
        content="We can help move unwanted furniture to your curbside, or take it to a "
                "dumpster for an extra fee. If you want to know the exact cost for "
                "haul-away, I can send your question over to the office so someone can "
                "get back to you. Would you like me to do that?"
    ),
    HumanMessage(content="Yes"),
]


def _check(text, messages=HAUL_AWAY):
    return prefill.ungrounded_terms(text, prefill.conversation_vocabulary(messages))


# ── The bug ────────────────────────────────────────────────────────────────────

def test_the_fridge_nobody_mentioned():
    assert _check("How much extra to haul away a fridge.") == ["fridge"]


def test_the_other_two_examples_that_used_to_be_in_the_prompt():
    """All three were copyable, so all three are checked."""
    assert _check("Whether we can move a 700lb gun safe down two flights of stairs.")
    assert _check("What storage costs for about a month.")


def test_the_prompt_no_longer_hands_the_model_an_answer():
    """
    A literal check, because the fix is a deletion and deletions come back. If
    you are adding examples to this prompt: the model will use them verbatim on
    a conversation too thin to summarise.
    """
    prompt = prefill._question_prompt()
    assert "fridge" not in prompt.lower()
    assert "gun safe" not in prompt.lower()
    assert "words that are already in the conversation" in prompt.lower()


# ── What it still lets through ─────────────────────────────────────────────────

def test_a_summary_built_from_what_was_actually_said_is_kept():
    assert _check("How much the exact cost is for haul-away.") == []
    assert _check("The exact haul-away fee.") == []


def test_our_own_words_count_as_grounding():
    """
    The agent is the side that said "the exact cost for haul-away" — the customer
    only said yes. A summary drawn from our half of the conversation is grounded
    in what was genuinely discussed.
    """
    assert _check("Whether there is a dumpster fee.") == []


def test_a_question_the_customer_really_asked_survives():
    """The check must not cost us the feature. This is the original conversation,
    where they did say fridge."""
    said_it = [
        HumanMessage(content="Do you do haul away"),
        AIMessage(content="We can help you leave unwanted furniture by your curbside."),
        HumanMessage(content="Just a fridge, how much extra"),
    ]
    assert _check("How much extra to haul away a fridge.", said_it) == []


def test_ordinary_words_are_not_inventions():
    """A summary may say "how much" about a conversation that never said it."""
    assert _check("How much would that be, and what should they know about it?") == []


def test_a_plural_or_a_tense_still_counts_as_the_same_word():
    """A check that tripped over an -s would reject every honest summary."""
    convo = [HumanMessage(content="what does it cost to move a piano")]
    assert _check("What the costs are for moving a piano.", convo) == []


def test_an_empty_conversation_grounds_nothing():
    assert _check("How much for haul-away.", []) == ["haul", "away"]


# ── Details already given ──────────────────────────────────────────────────────

def test_contact_details_from_an_earlier_lead_are_re_validated_not_trusted():
    state = {"known_contact": {"name": "Nick", "phone": "(818) 505-4576",
                               "contact_method": "Phone", "email": "not an address"}}
    carried = prefill._already_given("question", state)
    assert carried["phone"] == "(818) 505-4576"
    assert "email" not in carried, "an invalid address was carried anyway"
