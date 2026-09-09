"""
The interview in the customer's language.

Python still owns which questions get asked, in what order, and what counts as a
valid answer — only the WORDING is translated. So the thing worth asserting is
the seam: that everything a customer reads goes through the phrasebook, and that
nothing the code compares against does.

No model runs here. The translation itself is one prompt; what breaks in
production is the plumbing around it.
"""

from langchain_core.messages import AIMessage, HumanMessage

from agent.nodes import collect_lead, translate
from agent.nodes.translate import Phrase
from schemas import lead_form


def _fake_book(strings) -> dict[str, str]:
    """A phrasebook that marks every string, so an untranslated one is visible."""
    return {s: f"<{s}>" for s in strings}


# ── What gets translated, and what must not ───────────────────────────────────

def test_every_line_the_interview_says_is_offered_for_translation():
    """
    The failure this catches is a new question added with its copy inline: it
    works, it reads fine in English, and it is the one sentence that stays
    English in every other language.
    """
    for lead_type in ("estimate", "long_distance", "question"):
        offered = set(lead_form.customer_strings(lead_type)) | set(collect_lead.INTERVIEW_LINES)
        for field in lead_form.fields_for(lead_type):
            for text in (field.ask, field.clarify, field.placeholder, field.help):
                if text:
                    assert text in offered, f"{lead_type}.{field.name}: {text!r} never translated"
            assert field.recap_label() in offered
            for option in field.options:
                assert option in offered


def test_the_email_labels_are_never_translated():
    """A manager reads the lead in English however the chat went."""
    for lead_type in ("estimate", "long_distance", "question"):
        offered = set(lead_form.customer_strings(lead_type))
        for field in lead_form.fields_for(lead_type):
            # `recap` is what the customer sees and is translated; `label` heads
            # a column in the office's email and must not be.
            if field.label != field.recap_label():
                assert field.label not in offered, f"{field.name}: office label was translated"


def test_a_choice_shows_one_language_and_sends_another():
    """
    The buttons are in the customer's language; the value posted back is the
    English one, because that is what HOME_SIZES is matched against. Translating
    the value would fail every check the answer has to pass.
    """
    field = next(f for f in lead_form.fields_for("estimate") if f.name == "home_size")
    book = _fake_book(lead_form.customer_strings("estimate"))
    payload = field.localized(lambda text: book.get(text, text))

    assert payload["options"] == list(lead_form.HOME_SIZES)
    assert payload["option_labels"] == [f"<{o}>" for o in lead_form.HOME_SIZES]
    for option in payload["options"]:
        assert lead_form.validate_one(field, option) is None


def test_the_customers_own_words_are_never_translated():
    """Their name is their name. Only OUR side of the read-back is ours to change."""
    answers = {"name": "Джордан", "home_size": "2 bedrooms", "phone": "(323) 555-0142"}
    book = _fake_book(lead_form.customer_strings("estimate"))
    rows = lead_form.summary_for("estimate", answers, lambda t: book.get(t, t))
    by_value = {row["value"] for row in rows}

    assert "Джордан" in by_value
    assert "(323) 555-0142" in by_value
    # The one exception: a multiple-choice answer is our string, not theirs.
    assert "<2 bedrooms>" in by_value


# ── Placeholders ──────────────────────────────────────────────────────────────

def test_a_name_is_filled_in_after_translation_not_before():
    """
    Translating a filled-in question would hand the model a real customer's name
    to translate. The template goes through the phrasebook; the name never does.
    """
    field = next(f for f in lead_form.fields_for("estimate") if f.name == "phone")
    assert "{name}" in field.ask

    book = {field.ask: "Спасибо, {name}. Какой номер телефона?"}
    asked = field.question({"name": "Jordan Lee"}, lambda t: book.get(t, t))

    assert asked == "Спасибо, Jordan. Какой номер телефона?"
    assert "{name}" not in asked


def test_a_translation_that_loses_a_placeholder_is_thrown_away():
    """
    These strings get .format()ed. A missing placeholder is not a clumsy
    sentence, it is a KeyError halfway through an interview — so English wins.
    """
    strings = ["Thanks {name}, what's your number?", "Question {step} of {total}"]
    book = translate.reconcile(strings, [
        Phrase(source=strings[0], translated="Спасибо, как ваш номер?"),          # lost {name}
        Phrase(source=strings[1], translated="Вопрос {step} из {total}"),         # kept both
    ])

    assert strings[0] not in book, "a translation missing its placeholder was kept"
    assert book[strings[1]] == "Вопрос {step} из {total}"
    assert book[strings[1]].format(step=3, total=9) == "Вопрос 3 из 9"


def test_a_paraphrased_source_is_discarded_rather_than_guessed_at():
    """
    The source is the key. A model that "tidied" it has produced an entry that
    matches nothing, and keeping it would look like a translation that silently
    failed to apply.
    """
    book = translate.reconcile(
        ["First off — what's your name?"],
        [Phrase(source="First off - whats your name", translated="Как вас зовут?")],
    )
    assert book == {}


def test_surrounding_whitespace_survives_the_round_trip():
    """
    Some of these strings are concatenated onto the end of another sentence, so
    a leading space is load-bearing — and it is the first thing a model trims.
    Matching on the trimmed text and restoring the spacing is what stops one
    phrase silently staying English.
    """
    source = " I've got a few bits already, so this'll be quick."
    book = translate.reconcile(
        [source],
        # Note the model hands the source back trimmed, as it really does.
        [Phrase(source=source.strip(), translated="У меня уже есть кое-что.")],
    )
    assert book[source] == " У меня уже есть кое-что."


def test_an_empty_translation_is_not_an_answer():
    book = translate.reconcile(["Skip"], [Phrase(source="Skip", translated="   ")])
    assert book == {}


# ── The node's own decisions ──────────────────────────────────────────────────

def test_an_empty_conversation_stays_in_english():
    """
    Asking a model to name the language of no text gets a confident answer to a
    question nobody asked — the first version of this returned Spanish.
    """
    assert translate.translate({"lead_type": "question", "messages": []}) == {
        "lang": "en", "phrasebook": {},
    }


def test_the_companys_own_replies_do_not_decide_the_language():
    """
    Sam answers in whatever it is written to. Reading its replies to work out
    the language is circular — the customer sets it, and only they do.
    """
    messages = [
        AIMessage(content="Happy to help — what are you moving?"),
        HumanMessage(content="Здравствуйте, сколько стоит переезд?"),
        AIMessage(content="Наша ставка — $115 в час."),
    ]
    assert translate._customer_text(messages) == [messages[1]]


def test_an_english_conversation_costs_nothing_at_the_interview():
    """The phrasebook is empty, and `_translator` collapses to the identity."""
    tr = collect_lead._translator({"lang": "en", "phrasebook": {}})
    for text in collect_lead.INTERVIEW_LINES:
        assert tr(text) == text


def test_a_missing_translation_falls_back_to_english_not_to_nothing():
    """A phrase the translator skipped comes out in English, never blank."""
    tr = collect_lead._translator({"lang": "ru", "phrasebook": {"Skip": "Пропустить"}})
    assert tr("Skip") == "Пропустить"
    assert tr(collect_lead.CANCEL_REPLY) == collect_lead.CANCEL_REPLY
