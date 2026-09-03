"""
NODE: Collect lead
PURPOSE: Ask for the estimate details one at a time, in the conversation.
INPUT:   state.lead_type, state.lead (whatever prefill already worked out)
OUTPUT:  {"lead": {...}, "photo_ids": [...]} then on to submit_lead

WHY ONE QUESTION AT A TIME
--------------------------
This used to open a nine-field modal. It worked, and it felt like being handed a
form — the conversation stopped and paperwork started. People abandon paperwork.

So each field is now its own turn: the agent asks, the customer answers, the
agent asks the next thing. Same fields, same validation, same email at the end.

WHY THE ORDER IS FIXED AND PYTHON DECIDES IT
--------------------------------------------
The obvious alternative is to let a model run the interview — decide what to ask
next, parse whatever comes back. It interviews better and it is the wrong choice
here, for two reasons. It can loop or drift, so there is no guarantee the form
ever finishes. And whatever it decides a field contains ends up in an email a
manager acts on, with nothing between the model and the customer.

This loop always terminates, always asks for exactly what the office needs, and
costs nothing per question. The one model call in this flow is prefill, which
runs before this node and only fills fields the customer can see are wrong.

RE-ENTRANCY
-----------
Every interrupt() below re-runs from the top of this node on each resume, so
there is deliberately nothing here that touches the outside world — no model
call, no write, no upload. Just replaying answers already given.
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
    "No problem, I'll leave it there. If you change your mind just say so — or "
    f"the office is on {config.COMPANY_PHONE} any day from 6am."
)

#: An answer that is plainly a question rather than an answer. Someone mid-form
#: who suddenly asks "wait, how much is 3 movers?" should not end up with that
#: sentence recorded as their street address.
def _looks_like_a_question(text: str, field: lead_form.Field_) -> bool:
    if field.kind in ("select", "date"):
        return False
    stripped = str(text).strip()
    return stripped.endswith("?") and len(stripped.split()) > 2


def _ask(payload: dict) -> dict:
    """
    One turn of the interview. Returns the resume payload from the browser.

    Shape coming back is one of:
      {"answer": "..."}   they typed something
      {"skip": true}      they skipped an optional question
      {"photo_ids": [..]} the photo step
      {"cancelled": true} they backed out
    """
    reply = interrupt(payload)
    return reply if isinstance(reply, dict) else {"answer": str(reply or "")}


def collect_lead(state: SupportState) -> Command[Literal["submit_lead", "__end__"]]:
    lead_type = state.get("lead_type") or "estimate"
    spec = lead_form.spec(lead_type)

    # Whatever prefill already pulled out of the conversation.
    answers: dict[str, str] = dict(state.get("lead") or {})
    opening = spec["opening"]
    if answers:
        opening += " I've got a few bits already, so this'll be quick."

    # Worked out once, up front, so the count cannot drift as answers come in —
    # and so it survives this node re-running on every resume. A customer who can
    # see "3 of 6" finishes; one being asked an open-ended series of questions
    # with no end in sight starts wondering how long this goes on for.
    pending = [f for f in lead_form.fields_for(lead_type) if not answers.get(f.name)]
    total = len(pending) + 1  # + the photo step
    step = 0

    first = True

    for field in pending:
        step += 1
        question = field.question(answers)
        while True:
            reply = _ask(
                {
                    "type": "question",
                    "lead_type": lead_type,
                    "field": field.to_json(),
                    "message": question,
                    # Only ever on the very first question, so the opening line
                    # is not repeated every time something fails validation.
                    "opening": opening if first else "",
                    "skippable": not field.required,
                    "step": step,
                    "total": total,
                }
            )
            first = False

            if reply.get("cancelled"):
                logger.info("Lead form: cancelled at %s", field.name)
                return Command(
                    update={"messages": [AIMessage(content=CANCEL_REPLY)]}, goto=END
                )

            if reply.get("skip"):
                if field.required:
                    question = "I do need this one, sorry — what should I put down?"
                    continue
                break

            value = str(reply.get("answer") or "").strip()

            # "not sure yet" is a real answer to the date question, not a failure.
            if field.kind == "date" and lead_form.is_undecided(value):
                value = ""

            if _looks_like_a_question(value, field):
                question = (
                    "Good question — let me get these details over first and a "
                    f"manager will cover that properly. {field.question(answers)}"
                )
                continue

            if message := lead_form.validate_one(field, value):
                question = message
                continue

            if value:
                answers[field.name] = value
            break

    # Photos last. Asked after everything else because it is the one step that
    # needs them to go and do something, and by then they have already invested
    # eight answers — asking first loses people who don't have photos to hand.
    photo_ids: list[str] = []
    while True:
        reply = _ask(
            {
                "type": "photos",
                "lead_type": lead_type,
                "message": lead_form.PHOTO_STEP["ask"],
                "opening": opening if first else "",
                "skippable": True,
                "step": total,
                "total": total,
            }
        )
        first = False
        if reply.get("cancelled"):
            logger.info("Lead form: cancelled at photos")
            return Command(update={"messages": [AIMessage(content=CANCEL_REPLY)]}, goto=END)
        photo_ids = [str(p)[:32] for p in (reply.get("photo_ids") or [])][
            : config.MAX_UPLOADS_PER_THREAD
        ]
        break

    logger.info("Lead form: complete (%s, %d photos)", lead_type, len(photo_ids))
    return Command(
        update={
            "lead_type": lead_type,
            "lead": lead_form.clean(lead_type, answers),
            "photo_ids": photo_ids,
        },
        goto="submit_lead",
    )
