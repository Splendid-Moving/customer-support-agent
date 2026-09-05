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
    return """\
You read one conversation between a customer and a Los Angeles moving company, \
and write down the question they are still waiting on an answer to. It is going \
into an email a manager will answer by phone.

Write ONE sentence, in the customer's own terms, as a note to a colleague:
  "How much extra to haul away a fridge."
  "Whether we can move a 700lb gun safe down two flights of stairs."
  "What storage costs for about a month."

Rules:
- Only what they actually asked. Do not add detail they did not give, do not \
answer it, and do not soften it.
- If several things are unresolved, take the LAST one they asked about.
- If the conversation does not contain a clear question, return empty. They will \
simply be asked, which is far better than a manager calling back about the \
wrong thing."""


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

    try:
        result = get_model("extract").with_structured_output(shape).invoke(
            [SystemMessage(content=prompt), *state.get("messages", [])[-6:]]
        )
    except Exception:
        # Not worth failing a lead over. Without this the customer answers one or
        # two extra questions, which is a far better outcome than an error.
        logger.exception("Prefill failed; asking everything")
        return {"lead_type": lead_type, "lead": known, "lead_submitted": False}

    by_name = {f.name: f for f in lead_form.fields_for(lead_type)}
    for name, value in result.model_dump().items():
        field = by_name.get(name)
        if not field or not field.extractable or not value:
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
