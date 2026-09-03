"""
NODE: Submit lead
PURPOSE: Email the completed form to the office, once.

NOTE ON THE SIGNATURE: the second parameter MUST be named `config`. LangGraph
decides what to inject into a node by inspecting parameter NAMES, so calling it
anything else means it is never supplied — the node raises TypeError, LangGraph
rolls the whole superstep back, and the customer sees the empty form again with
no indication anything went wrong. `services.config` is therefore imported as
`settings` in this one module to keep the name free.
INPUT:   state.lead, state.lead_type, state.photo_ids
OUTPUT:  {"messages": [AIMessage], "lead_submitted": True}

The only side effect in this application. It is deliberately the last node in the
lane, after the interrupt, so that:

  - nothing an attacker types can reach it — the form has to be filled in by
    hand first, and the guard runs before any of that;
  - it never re-runs on resume. Code ABOVE an interrupt re-executes every time
    the graph is resumed; code below it does not.

`lead_submitted` is belt and braces on top of that ordering, for the case the
node itself is retried — a double-clicked Submit, a browser retrying a request
that actually succeeded. A manager getting the same lead twice is a small
annoyance; it is also entirely avoidable with one boolean.

IF THE SEND FAILS
-----------------
The customer is told, plainly, with the phone number. The alternative — "thanks,
someone will be in touch" over a lead that evaporated — is the worst outcome
available here, because nobody finds out until the customer has already hired
somebody else.
"""

import logging

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agent.state import SupportState
from services import config as settings
from services import email, uploads

logger = logging.getLogger(__name__)


def _thread_id(config: RunnableConfig) -> str:
    return (config or {}).get("configurable", {}).get("thread_id", "")


def _confirmation(lead: dict, photo_count: int) -> str:
    name = (lead.get("name") or "").split()[0] if lead.get("name") else ""
    opener = f"Got it, {name}." if name else "Got it."
    photos = (
        f" Your {photo_count} photos came through too."
        if photo_count > 1
        else " Your photo came through too."
        if photo_count
        else ""
    )
    return (
        f"{opener} That's with the office now.{photos} A manager will go through "
        "it and get back to you — usually the same day. If you'd rather not wait, "
        f"we're on {settings.COMPANY_PHONE}, every day from 6am."
    )


def _failure(lead: dict) -> str:
    return (
        "I've got everything down, but something went wrong sending it over to "
        f"the office — I don't want to tell you it's handled when it isn't. Give "
        f"us a call on {settings.COMPANY_PHONE} and mention you filled in the form; "
        "they'll take it from there."
    )


def submit_lead(state: SupportState, config: RunnableConfig) -> dict:
    if state.get("lead_submitted"):
        logger.info("Submit lead: already sent for this thread, skipping")
        return {}

    lead = state.get("lead") or {}
    lead_type = state.get("lead_type", "estimate")
    thread_id = _thread_id(config)

    attachments = uploads.collect(thread_id, state.get("photo_ids") or [])

    try:
        email.send_lead(lead_type, lead, attachments)
    except Exception:
        # Logged with a full stack trace, because this is the failure that costs
        # an actual job and someone will want to know exactly why.
        logger.exception("Submit lead: send failed for thread %s", thread_id)
        return {"messages": [AIMessage(content=_failure(lead))]}

    # Only once the office has them. Deleting before the send would lose the
    # photos on a failure, and they are the part a customer will not redo.
    uploads.discard(thread_id)

    return {
        "messages": [AIMessage(content=_confirmation(lead, len(attachments)))],
        "lead_submitted": True,
    }
