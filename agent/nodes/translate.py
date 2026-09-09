"""
NODE: Translate
PURPOSE: Put the interview into the customer's language before it starts.
INPUT:   state.messages, state.lead_type
OUTPUT:  {"lang": "ru", "phrasebook": {english: translated}}

Sam already answers in whatever language it is written to — that is one line in
the persona. The interview did not, because its questions are not written by a
model: they are fixed strings in schemas/lead_form.py, chosen deliberately and
validated in Python. So a customer could hold an entire conversation in Russian,
ask for an estimate, and be met with "First off — what's your name?".

WHY NOT JUST LET A MODEL RUN THE INTERVIEW
------------------------------------------
Because the interview's fixed shape is the reason it works. It always ends, it
always asks for exactly what the office needs, and every answer is validated
before it reaches an email a manager dials from. Handing all of that to a model
to fix the language would trade a guarantee for a phrasing.

So the structure stays in Python and ONLY THE WORDING is translated. Same
questions, same order, same validation — in Russian.

WHY IT IS ITS OWN NODE, BEFORE collect_lead
-------------------------------------------
Same reason as prefill: collect_lead re-runs from the top on every single answer,
so a model call inside it would fire once per question and could come back
differently worded each time. Out here it runs once per lead.

WHAT IS NOT TRANSLATED
----------------------
The email to the office. A manager reads that in English whatever language the
chat was in, so `Field_.label` never comes through here — only `recap`, which is
what the customer sees read back to them.

Nor the stored value of a multiple-choice answer. The buttons show translated
text and send back the English value, because "2 bedrooms" is matched against
HOME_SIZES in Python and a Russian string would fail every check we have.
"""

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.models import get_model
from agent.nodes import collect_lead
from agent.state import SupportState
from schemas import lead_form

logger = logging.getLogger(__name__)


#: Translations are deterministic (temperature 0) and the string list only
#: changes when the code does, so the same language asks the same questions every
#: time. Cached per process: the second Russian customer of the day costs
#: nothing. Cleared on deploy, which is exactly when the strings may have moved.
_CACHE: dict[tuple[str, str], dict[str, str]] = {}

#: `{name}`, `{step}`, `{total}`, `{n}` — the holes a value gets formatted into.
_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")

#: Cap on how much of a phrasebook we will accept back. A defensive bound, not a
#: business rule — the model is being handed ~50 strings and should return ~50.
_MAX_PHRASES = 200


class Language(BaseModel):
    """What language the CUSTOMER is writing in."""

    code: str = Field(
        description="Lowercase ISO 639-1 code of the language the customer is "
                    "writing in — 'en', 'es', 'ru', 'ko'. Use 'en' if unsure, or "
                    "if they have only sent things like 'ok' or a phone number "
                    "that could be any language."
    )
    name: str = Field(description="The language's English name, e.g. 'Russian'.")


class Phrase(BaseModel):
    source: str = Field(description="The English string, copied EXACTLY as given.")
    translated: str = Field(description="That string in the target language.")


class Phrasebook(BaseModel):
    phrases: list[Phrase]


DETECT = """\
You are given the recent messages of a chat between a customer and a Los Angeles \
moving company. Say what language THE CUSTOMER is writing in.

Judge by the customer's own messages, never by the company's replies. If the \
customer has written in more than one language, take the one they used most \
recently for a real sentence."""


def _translate_prompt(language: str) -> str:
    return f"""\
You translate the questions a moving company's assistant asks its customers, \
from English into {language}.

This is a chat on a moving company's website. The assistant is a member of the \
team called Sam, not a machine — translate so that it still sounds like a person \
who works there. Natural, warm, and short. Match the register a moving company \
would actually use with a customer in {language}: not formal officialese, not \
slang.

Rules, all of them absolute:

- Translate EVERY string you are given. Return one entry per string.
- Copy `source` back EXACTLY as it was given to you, character for character. It \
is how the translation is matched up, not something to tidy.
- Keep every `{{placeholder}}` EXACTLY as it appears, braces and English word \
included — `{{name}}`, `{{step}}`, `{{total}}`, `{{n}}`. A value is substituted \
into each one, so do not translate the word inside the braces, do not remove the \
braces, and do not drop one. Put it where that language would naturally want the \
value; just make sure it is still there.
- Leave phone numbers, prices, and "Splendid Moving" alone.
- Some strings are short labels for a form row ("Your name", "Move date"), some \
are example text shown greyed out in a box ("Jordan Lee", "90026"). Translate \
them as labels and examples, not as sentences. Where an example is a name or a \
zip code, give one that looks native to {language} speakers in Los Angeles if \
that makes sense, or keep it as is.
- Never add a sentence that is not in the original. Never answer the question, \
explain it, or make it more polite than it was."""


def _customer_text(messages) -> list:
    """The customer's own turns. Language is theirs to set, not ours."""
    return [m for m in messages if isinstance(m, HumanMessage)][-4:]


def _detect(messages) -> Language:
    model = get_model("router").with_structured_output(Language)
    return model.invoke([SystemMessage(content=DETECT), *_customer_text(messages)])


def reconcile(strings: list[str], phrases) -> dict[str, str]:
    """
    What came back from the model, reduced to what is safe to actually use.

    Separate from the call so it can be tested without one: this is where a bad
    translation is caught, and it is the half that breaks.
    """
    # Matched on the trimmed text, keyed by the original. A model hands back what
    # looks to it like the same string, and a leading space is exactly the kind
    # of thing it tidies away — which silently cost one phrase its translation
    # the first time this ran.
    wanted = {text.strip(): text for text in reversed(strings)}
    book: dict[str, str] = {}

    for phrase in list(phrases or [])[:_MAX_PHRASES]:
        source = str(getattr(phrase, "source", "")).strip()
        translated = str(getattr(phrase, "translated", "")).strip()

        # Anything we did not ask for is discarded rather than trusted. A model
        # that paraphrased the source has produced a key that never matches, and
        # keeping it would look, from the outside, like a translation that
        # silently "didn't apply".
        if source not in wanted or not translated:
            continue
        source = wanted[source]

        # A dropped or renamed placeholder is not a worse translation, it is a
        # crash: these strings get `.format()`ed, and a missing key raises
        # mid-interview. English is by far the better outcome.
        if set(_PLACEHOLDER.findall(source)) != set(_PLACEHOLDER.findall(translated)):
            logger.warning("Translate: dropped %r — placeholders mangled", source[:40])
            continue

        # Whatever spacing the original carried, the translation carries too —
        # some of these are concatenated onto the end of another sentence.
        lead = source[: len(source) - len(source.lstrip())]
        trail = source[len(source.rstrip()):]
        book[source] = f"{lead}{translated}{trail}"

    return book


def _phrasebook(strings: list[str], language: str) -> dict[str, str]:
    model = get_model("translate").with_structured_output(Phrasebook)
    numbered = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(strings))
    result = model.invoke([
        SystemMessage(content=_translate_prompt(language)),
        HumanMessage(content=f"Translate these {len(strings)} strings:\n\n{numbered}"),
    ])
    return reconcile(strings, result.phrases)


def translate(state: SupportState) -> dict:
    lead_type = state.get("lead_type") or "estimate"
    messages = state.get("messages", [])

    # Nothing from the customer yet — an interview entered straight from state,
    # or a thread whose history has not landed. There is nothing to detect FROM,
    # and asking a model to name the language of no text gets a confident answer
    # to a question nobody asked: the first version of this returned Spanish for
    # an empty conversation.
    if not _customer_text(messages):
        logger.info("Translate: nothing from the customer yet; staying in English")
        return {"lang": "en", "phrasebook": {}}

    try:
        language = _detect(messages)
        code = (language.code or "en").strip().lower()[:5]
    except Exception:
        # The interview in English is a small problem. No interview at all is a
        # lost job, so nothing in this node is allowed to be fatal.
        logger.exception("Translate: language detection failed; staying in English")
        return {"lang": "en", "phrasebook": {}}

    if code.startswith("en"):
        return {"lang": "en", "phrasebook": {}}

    key = (lead_type, code)
    if key in _CACHE:
        logger.info("Translate: %s phrasebook from cache", code)
        return {"lang": code, "phrasebook": _CACHE[key]}

    # Two sources, one list: the copy that belongs to the fields, and the copy
    # that belongs to the loop around them.
    strings = lead_form.customer_strings(lead_type) + list(collect_lead.INTERVIEW_LINES)
    try:
        book = _phrasebook(strings, language.name or code)
    except Exception:
        logger.exception("Translate: failed for %s; staying in English", code)
        return {"lang": "en", "phrasebook": {}}

    missed = len(strings) - len(book)
    if missed:
        # Each miss is one question that comes out in English. Worth a log line
        # and not worth failing over — a form that is 90% Russian still works.
        logger.warning("Translate: %s missing %d of %d strings", code, missed, len(strings))

    if book:
        _CACHE[key] = book
    logger.info("Translate: %s, %d strings", code, len(book))
    return {"lang": code, "phrasebook": book}
