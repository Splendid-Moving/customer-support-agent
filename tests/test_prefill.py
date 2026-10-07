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


# ── Contact details said in passing ────────────────────────────────────────────
# "hi my name is nik", then three turns later "First off — what's your name?"
# Name, phone and email may now be picked up from the chat, but ONLY when the
# exact value is something the customer typed. The model can copy; it cannot
# invent, and it cannot take a value out of our own replies.

from schemas import lead_form  # noqa: E402

#: The conversation from the screenshot that started this.
NIK = [
    HumanMessage(content="hi my name is nik"),
    AIMessage(content="Hi Nik! How can I help you with your move today?"),
    HumanMessage(content="what are your rates"),
    AIMessage(content="For local moves, 2 movers are $115/hr cash. How many movers "
                      "do you think you'll need?"),
    HumanMessage(content="can i get an estimate"),
]


def _typed(field, value, messages=NIK):
    return prefill.typed_by_customer(field, value, messages)


def test_a_name_the_customer_typed_is_kept():
    assert _typed("name", "Nik")


def test_a_name_only_we_said_is_not():
    """Our side greeting them by a name proves nothing about what they said."""
    convo = [HumanMessage(content="hi"), AIMessage(content="Hi Jordan Lee!")]
    assert not _typed("name", "Jordan Lee", convo)


def test_a_full_name_needs_every_part_typed():
    assert not _typed("name", "Nik Petrov")


def test_a_confused_reply_is_never_a_name():
    convo = [HumanMessage(content="what")]
    assert not _typed("name", "what", convo)


def test_a_phone_number_matches_however_it_was_formatted():
    convo = [HumanMessage(content="call me at 818.505.4576 after 5")]
    assert _typed("phone", "(818) 505-4576", convo)
    assert _typed("phone", "+1 818 505 4576", convo)


def test_a_phone_number_one_digit_off_is_dropped():
    convo = [HumanMessage(content="call me at 818 505 4576")]
    assert not _typed("phone", "(818) 505-4577", convo)


def test_digits_from_unrelated_numbers_do_not_make_a_phone_number():
    """A zip and a date run together are ten digits, and nobody's number."""
    convo = [HumanMessage(content="from 90026 to 90291 on 10/20")]
    assert not _typed("phone", "9002690291", convo)


def test_an_email_must_appear_as_typed():
    convo = [HumanMessage(content="it's Nik.T@Example.com")]
    assert _typed("email", "nik.t@example.com", convo)
    assert not _typed("email", "nik@example.com", convo)


def test_an_email_at_the_end_of_a_sentence_still_counts():
    """Found live: "...maria.g@gmail.com." was dropped over the full stop."""
    convo = [HumanMessage(content="I'm Maria, maria.g@gmail.com. do you move pianos?")]
    assert _typed("email", "maria.g@gmail.com", convo)
    assert not _typed("email", "maria.g@gmail.co", convo)


class _FakeModel:
    """Stands in for the extraction model: returns whatever it is told to."""

    def __init__(self, **values):
        self.values = values

    def with_structured_output(self, shape):
        values = self.values

        class _Runner:
            def invoke(self, _messages):
                return shape(**values)

        return _Runner()


def _prefill(monkeypatch, messages, intent="estimate", **model_says):
    monkeypatch.setattr(prefill, "get_model", lambda _node: _FakeModel(**model_says))
    return prefill.prefill({"messages": messages, "intent": intent})["lead"]


def test_the_screenshot_conversation_is_not_asked_for_a_name(monkeypatch):
    assert _prefill(monkeypatch, NIK, name="Nik")["name"] == "Nik"


def test_an_invented_phone_number_never_reaches_the_form(monkeypatch):
    lead = _prefill(monkeypatch, NIK, name="Nik", phone="(323) 555-0142",
                    email="nik@example.com")
    assert lead == {"name": "Nik"}


def test_a_name_from_early_in_a_long_chat_is_still_found(monkeypatch):
    """It used to read only the last six messages."""
    padding = [HumanMessage(content="and how about stairs?"),
               AIMessage(content="Stairs are fine.")] * 6
    lead = _prefill(monkeypatch, [*NIK[:2], *padding, NIK[-1]], name="Nik")
    assert lead.get("name") == "Nik"


def test_the_question_form_picks_up_contact_details_too(monkeypatch):
    convo = [HumanMessage(content="I'm Nik, 818 505 4576. do you do haul away"),
             AIMessage(content="Want me to ask the office?"),
             HumanMessage(content="yes")]
    lead = _prefill(monkeypatch, convo, intent="question",
                    question="Whether we do haul away.", name="Nik",
                    phone="818-505-4576")
    assert lead["name"] == "Nik" and lead["phone"] == "818-505-4576"


def test_a_name_from_the_chat_does_not_claim_their_details_are_on_file():
    """
    `opening_again` says "I've still got your details". That is true after an
    earlier confirmed lead, and false when all we have is a name from the chat:
    they would then be asked how to reach them right after being told not to
    worry about it.
    """
    from tests.test_wiring import _start_question

    _graph, _cfg, ask = _start_question(
        "prefilled-name", {"question": "Whether we do haul away.", "name": "Nik"}
    )
    again = lead_form.spec("question")["opening_again"]
    assert ask["field"]["name"] == "contact_method"
    assert again not in ask["opening"]
    assert "Nik" in ask["message"], "the next question should use their name"


def test_contact_fields_are_now_extractable():
    for lead_type in ("estimate", "long_distance", "question"):
        names = {f.name for f in lead_form.fields_for(lead_type) if f.extractable}
        assert {"name", "phone", "email"} <= names, lead_type


def test_a_name_from_the_chat_is_capitalised(monkeypatch):
    assert _prefill(monkeypatch, NIK, name="nik")["name"] == "Nik"


def test_a_name_typed_into_the_form_is_capitalised():
    from tests.test_wiring import _resume, _start_question

    graph, cfg, ask = _start_question("typed-lower", {"question": "Whether we do haul away."})
    ask = _resume(graph, cfg, {"answer": "nik"})
    assert "Nik" in ask["message"]
