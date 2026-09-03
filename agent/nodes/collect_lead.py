"""
NODE: Collect lead
PURPOSE: Show the estimate form and wait — however long it takes — for it back.
INPUT:   state.intent
OUTPUT:  {"lead": {...}, "lead_type": ..., "photo_ids": [...]} then on to submit

THIS IS THE NODE THE WHOLE THING IS BUILT AROUND
------------------------------------------------
`interrupt()` genuinely stops the graph. The run ends, the state is written to
the checkpoint database, and the process is free. When the customer finally
submits — a minute later, or after they have gone to find the new landlord's
address and come back — a `Command(resume=...)` picks up on this exact line with
the whole conversation intact.

That is why this is a LangGraph app and not a script with a while loop. A script
would have to hold the conversation in memory and hope the server never restarts.

THE LOOP
--------
Validation failures re-interrupt with per-field errors rather than failing the
run. On resume the node re-runs from the top — which is why there is nothing
above the loop that touches the outside world. The only side effect in this lane
is the email, and it lives in the node after this one, deliberately.
"""

import logging

from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.types import Command, interrupt

from agent.state import SupportState
from schemas import lead_form
from services import config

logger = logging.getLogger(__name__)


CANCEL_REPLY = (
    "No problem. I'm here if you want to pick it back up — or you can always call "
    f"the office on {config.COMPANY_PHONE}."
)


def _intro_message(lead_type: str) -> str:
    """
    The line in the chat that sits above the form.

    Fixed text rather than a model call: the customer is already waiting on a
    form to render, and a second round-trip to write "here you go" is latency
    with nothing to show for it.
    """
    if lead_type == "long_distance":
        return (
            "Out-of-state moves aren't billed hourly like our local jobs, so a "
            "manager prices each one individually. Give me these details and "
            "they'll come back to you directly."
        )
    return (
        "Happy to get that moving. Fill this in and a manager will put together a "
        "real estimate for you — photos of your stuff help a lot."
    )


def collect_lead(state: SupportState) -> Command[Literal["submit_lead", "__end__"]]:
    lead_type = "long_distance" if state.get("intent") == "long_distance" else "estimate"
    spec = lead_form.spec(lead_type)

    errors: dict[str, str] = {}
    values: dict = {}

    while True:
        submitted = interrupt(
            {
                "type": "form",
                **spec,
                "intro_message": _intro_message(lead_type),
                # Echoed back so a rejected form comes up filled in. Making
                # someone retype nine fields because one date was wrong is how a
                # lead gets abandoned on the second attempt.
                "values": values,
                "errors": errors,
            }
        )

        # The customer closed the form. Not an error — they changed their mind.
        if submitted is None or (isinstance(submitted, dict) and submitted.get("cancelled")):
            logger.info("Lead form: cancelled by customer")
            return Command(
                update={"messages": [AIMessage(content=CANCEL_REPLY)], "lead_type": lead_type},
                goto=END,
            )

        if not isinstance(submitted, dict):
            # Someone typed into the chat box instead of the form. Show it again.
            errors = {}
            continue

        values = submitted.get("values", submitted)
        errors = lead_form.validate(lead_type, values)
        if not errors:
            break

        logger.info("Lead form: %d field(s) rejected", len(errors))

    photo_ids = [str(p)[:32] for p in (submitted.get("photo_ids") or [])][
        : config.MAX_UPLOADS_PER_THREAD
    ]

    logger.info("Lead form: accepted (%s, %d photos)", lead_type, len(photo_ids))
    return Command(
        update={
            "lead_type": lead_type,
            "lead": lead_form.clean(lead_type, values),
            "photo_ids": photo_ids,
        },
        goto="submit_lead",
    )
