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
"""

import logging
from datetime import date

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from agent.models import get_model
from agent.state import SupportState
from schemas import lead_form

logger = logging.getLogger(__name__)


class Known(BaseModel):
    """Only what the customer actually said. Everything is optional."""

    from_address: str = Field("", description="Where they are moving FROM, if stated.")
    to_address: str = Field("", description="Where they are moving TO, if stated.")
    move_date: str = Field("", description="Move date as YYYY-MM-DD, only if unambiguous.")
    home_size: str = Field("", description="Must be EXACTLY one of the listed options, or empty.")


def _prompt(lead_type: str) -> str:
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


def prefill(state: SupportState) -> dict:
    lead_type = "long_distance" if state.get("intent") == "long_distance" else "estimate"
    known: dict[str, str] = {}

    try:
        result = get_model("extract").with_structured_output(Known).invoke(
            [SystemMessage(content=_prompt(lead_type)), *state.get("messages", [])[-6:]]
        )
    except Exception:
        # Not worth failing a lead over. Without this the customer answers one or
        # two extra questions, which is a far better outcome than an error.
        logger.exception("Prefill failed; asking everything")
        return {"lead_type": lead_type, "lead": {}}

    by_name = {f.name: f for f in lead_form.fields_for(lead_type)}
    for name, value in result.model_dump().items():
        field = by_name.get(name)
        if not field or not field.extractable or not value:
            continue
        # Held to exactly the same validation as a typed answer. A model-invented
        # date in the past or a size that is not on the list is dropped, and the
        # customer simply gets asked.
        if lead_form.validate_one(field, value) is None:
            known[name] = value

    if known:
        logger.info("Prefill: already knew %s", ", ".join(sorted(known)))
    return {"lead_type": lead_type, "lead": known}
