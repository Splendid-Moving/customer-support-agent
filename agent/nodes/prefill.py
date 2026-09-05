"""
NODE: Prefill
PURPOSE: Notice what the customer has ALREADY told us, so we don't ask again.
INPUT:   state.messages, state.intent
OUTPUT:  {"lead_type": ..., "lead": {...partial...}}

Someone who opens with "I'm moving from a 2 bedroom in Silver Lake to Culver
City on the 20th" and is then asked "where are we moving you from?" has just
learned they are talking to a machine. This node reads the conversation once and
fills in whatever was already said.

WHY IT IS ITS OWN NODE
----------------------
It could have been the first few lines of collect_lead. It must not be. That node
interrupts once per question, and a node containing an interrupt RE-RUNS FROM THE
TOP every time the graph resumes — so a model call up there would fire again on
every single answer, roughly ten times per form, for an identical result. Out
here it runs once.

WHAT IT WILL NOT TOUCH
----------------------
Name, phone and email are never extracted, however clearly they appear. Those
three are the entire value of the lead: a manager with a mis-heard phone number
has nothing. They are always asked and always read back by the customer's own
typing. Everything else — addresses, size, date — is both easy for the customer
to correct and harmless for the office to have slightly wrong, because a human
calls them anyway.

The one exception is `known_contact`: details this same customer already typed
into an earlier form in this conversation and confirmed on the read-back. Those
are not a guess, they are their own answer, and asking for a phone number ninety
seconds after they gave it is exactly the kind of thing that makes a chat agent
feel like a machine.
"""

import logging
import re
from datetime import date

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from agent.models import get_model
from agent.state import SupportState
from schemas import lead_form

logger = logging.getLogger(__name__)


class KnownMove(BaseModel):
    """Only what the customer actually said. Everything is optional."""

    from_address: str = Field("", description="Where they are moving FROM, if stated.")
    to_address: str = Field("", description="Where they are moving TO, if stated.")
    move_date: str = Field("", description="Move date as YYYY-MM-DD, only if unambiguous.")
    home_size: str = Field("", description="Must be EXACTLY one of the listed options, or empty.")


class KnownQuestion(BaseModel):
    """What the customer wants the office to answer, in one line."""

    question: str = Field(
        "",
        description="The customer's outstanding question, in one sentence, in "
                    "their own terms. Empty if it is not clear what they asked.",
    )


def _move_prompt() -> str:
    sizes = "\n".join(f"  - {size}" for size in lead_form.HOME_SIZES)
    return f"""\
You read one customer conversation and pull out the move details they have \
already given, so a moving company doesn't ask them twice.

Today's date is {date.today().isoformat()}.

Return a field ONLY if the customer stated it plainly. Leave it empty otherwise.

- Do not infer. "I'm moving soon" is not a date. "A small place" is not a size.
- Do not guess a year. If they said "the 20th" and that is ambiguous, take the \
next one still in the future.
- home_size must be EXACTLY one of these strings, or empty:
{sizes}
- Addresses can be partial — "Silver Lake" is a useful answer, take it as given.

An empty field costs one extra question. A wrong field ends up in an email a \
manager acts on. When in doubt, leave it empty."""


def _question_prompt() -> str:
    """
    The prompt for reading a customer's outstanding question out of the chat.

    It carries no example answers, and that is deliberate. It used to open with
    three sample sentences, the first of which was "How much extra to haul away a
    fridge." A customer typed "do you do haul away" — no object, nothing to build
    a sentence around — and the model reached for the nearest one available,
    which was mine. The read-back offered to ask the office about a fridge that
    had never been mentioned.

    An example in a prompt is an answer the model is allowed to give. Where the
    real answer is thin, it will give that one. So the shape is described here
    instead of demonstrated, and `ungrounded_terms` catches it in Python if it
    invents something anyway.
    """
    return """\
You read one conversation between a customer and a Los Angeles moving company, \
and write down the question they are still waiting on an answer to. It goes into \
an email a manager will answer by phone.

Write it as ONE short sentence — a note to a colleague, not a reply to the \
customer. Begin it however fits: "How much...", "Whether we...", "What it costs \
to...".

Use ONLY words that are already in the conversation in front of you.

- NEVER name an item, room, place, date, price or quantity that nobody in this \
conversation has named. Not one they probably meant, not a typical example, not \
something that would make the sentence read better. If they did not say what \
they want hauled away, then the question is about hauling away, full stop.
- Do not answer it, soften it, or add context.
- If several things are unresolved, take the LAST one they asked about.
- Vague is fine. "How much haul-away costs" is a good answer to a vague \
question — a manager phones them and asks. An invented detail is not: they \
phone up talking about the wrong thing, and the customer knows we were guessing.
- If there is no clear question, return empty. They are simply asked, which \
costs one message and is always better than a wrong one."""


# ── Nothing invented reaches the office ────────────────────────────────────────
# `answer_check` refuses to let the knowledge lane state a price that is not in
# the knowledge base. Same principle, same reason: a prompt rule alone leaks, and
# what leaks here lands in an email somebody acts on.

_WORD = re.compile(r"[a-z]+")

#: Words too ordinary to mean anything by their presence. A summary is allowed
#: to say "how much" about a conversation that never used the phrase; it is not
#: allowed to say "fridge".
COMMON_WORDS = frozenset("""
about after again against all also and another any anything are around asked
asking back because been before being both but can come could did does doing
done down each else even ever every for from get give going gone got had has
have here how into its just know like long make many may maybe might more most
much need needs new not now off one only other our out over said same say see
should since some something still such sure take than that the their them then
there these they thing things think this those time too under until upon use
used very want wanted wants was way well were what when where whether which
while who why will with would yes you your yours
""".split())


def _words(text: str) -> list[str]:
    return _WORD.findall(str(text).lower())


def _stem(word: str) -> str:
    """
    A word cut down to what two forms of it have in common.

    Four characters, after the one suffix that moves the letters underneath: a
    dropped silent e. "move" and "moving" share only "mov", so comparing four
    characters flat calls them different words and rejects an honest summary.
    """
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)][:4]
    return word[:4]


def conversation_vocabulary(messages) -> set[str]:
    """
    Every word said in this conversation, by either side.

    The agent's own turns count. Ours is the side that says "if you want to know
    the exact cost for haul-away" — the customer answers "yes", and a summary
    built out of our words is grounded in what was actually discussed.
    """
    vocabulary: set[str] = set()
    for message in messages:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            vocabulary.update(_words(content))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    vocabulary.update(_words(part.get("text", "")))
    return vocabulary


def ungrounded_terms(text: str, vocabulary: set[str]) -> list[str]:
    """
    Words in `text` that nobody in the conversation said, ordinary ones aside.

    Matching is by stem, so "cost" grounds "costs", "move" grounds "moving" and
    "haul-away" grounds both "haul" and "away". Loose on purpose: the failure
    being caught is a whole invented noun, and a check that tripped over a plural
    would reject every honest summary.
    """
    known = {_stem(word) for word in vocabulary}
    found = []
    for token in _words(text):
        if len(token) < 3 or token in COMMON_WORDS or token in vocabulary:
            continue
        stem = _stem(token)
        if any(other.startswith(stem) or stem.startswith(other) for other in known):
            continue
        found.append(token)
    return found


#: Which form each routed intent opens.
_LEAD_TYPES = {"long_distance": "long_distance", "question": "question"}


def _already_given(lead_type: str, state: SupportState) -> dict[str, str]:
    """
    Contact details this customer typed and confirmed earlier in this conversation.

    Re-validated rather than trusted: the form they were collected on is not
    necessarily the form being filled in now, and a value that does not belong on
    this one is simply left out.
    """
    by_name = {f.name: f for f in lead_form.fields_for(lead_type)}
    carried = {}
    for name, value in (state.get("known_contact") or {}).items():
        field = by_name.get(name)
        if field and value and lead_form.validate_one(field, value) is None:
            carried[name] = value
    return carried


def prefill(state: SupportState) -> dict:
    lead_type = _LEAD_TYPES.get(state.get("intent", ""), "estimate")

    known: dict[str, str] = _already_given(lead_type, state)
    if known:
        logger.info("Prefill: carried over %s", ", ".join(sorted(known)))

    shape = KnownQuestion if lead_type == "question" else KnownMove
    prompt = _question_prompt() if lead_type == "question" else _move_prompt()
    recent = state.get("messages", [])[-6:]

    try:
        result = get_model("extract").with_structured_output(shape).invoke(
            [SystemMessage(content=prompt), *recent]
        )
    except Exception:
        # Not worth failing a lead over. Without this the customer answers one or
        # two extra questions, which is a far better outcome than an error.
        logger.exception("Prefill failed; asking everything")
        return {"lead_type": lead_type, "lead": known, "lead_submitted": False}

    vocabulary = conversation_vocabulary(recent)

    by_name = {f.name: f for f in lead_form.fields_for(lead_type)}
    for name, value in result.model_dump().items():
        field = by_name.get(name)
        if not field or not field.extractable or not value:
            continue

        # The one free-text field, and so the only one where a model can invent
        # a whole noun rather than get a listed value wrong. If it names
        # something nobody in the conversation named, it is thrown away and the
        # customer is asked — one message, against a manager phoning someone
        # about a fridge they never mentioned.
        #
        # Deliberately not applied to the addresses: those are a customer's own
        # place, obvious to them in the read-back, and a manager calls anyway.
        if name == "question" and (invented := ungrounded_terms(value, vocabulary)):
            logger.warning(
                "Prefill: dropped an invented question — nobody said %s (%r)",
                ", ".join(invented), value,
            )
            continue

        # Held to exactly the same validation as a typed answer. A model-invented
        # date in the past or a size that is not on the list is dropped, and the
        # customer simply gets asked.
        if lead_form.validate_one(field, value) is None:
            known[name] = str(value).strip()[:1000]

    if known:
        logger.info("Prefill: already knew %s", ", ".join(sorted(known)))

    # Clearing lead_submitted matters more than it looks. submit_lead refuses to
    # send twice, which is right for a retried request and WRONG for a customer
    # who genuinely wants a second quote — a move for their office, say — in the
    # same conversation. Without this reset that second lead is silently dropped
    # and nobody ever finds out. Entering this lane means a new lead is starting.
    return {"lead_type": lead_type, "lead": known, "lead_submitted": False}
