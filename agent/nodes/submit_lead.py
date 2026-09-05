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
import re

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agent.state import SupportState
from services import config as settings
from services import email, uploads

logger = logging.getLogger(__name__)


def _thread_id(config: RunnableConfig) -> str:
    return (config or {}).get("configurable", {}).get("thread_id", "")


#: Letters, apostrophes and hyphens only. Whatever someone typed into "what's
#: your name?" is echoed back in the confirmation, and that reply becomes an
#: AIMessage in the history that later model calls read. It is the main path by
#: which text from the interview — which the guard never screened, because the
#: guard does not run while the graph is paused — reaches a prompt. One short,
#: alphabetic word cannot carry an instruction.
#:
#: A question confirmation also names the phone number or email address the
#: answer is coming back on, which is the same path and held to the same bar: a
#: phone number has been reduced to ten digits by `normalize_phone`, and an
#: address has been through a regex that permits no whitespace anywhere and is
#: only repeated at all if it is a sane length. Neither can carry a sentence.
_NAMEISH = re.compile(r"[^\w'’\-]", re.UNICODE)


def _first_name(raw: str) -> str:
    if not raw:
        return ""
    return _NAMEISH.sub("", raw.split()[0])[:24]


#: The contact details a later lead in the same conversation can reuse instead of
#: asking for them again. Only ever these — nothing about a particular move.
CARRIED_OVER = ("name", "phone", "email", "contact_method")

#: Longest contact detail the confirmation will repeat back. Real ones are well
#: under this; the cap is about what goes into the conversation history, not
#: about what the office receives.
MAX_ECHOED = 100


def _confirmation(lead_type: str, lead: dict, photo_count: int) -> str:
    if lead_type == "question":
        return _question_confirmation(lead)

    name = _first_name(lead.get("name") or "")
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


def _question_confirmation(lead: dict) -> str:
    """
    What the customer sees once their question is actually with the office.

    It names the number or address they will be answered on, because the whole
    reason this lane exists is a conversation where the agent took someone's
    details and then told them to phone us — saying where the answer is going is
    the difference between a promise and a form swallowing an answer.

    Their own contact detail is safe to repeat back: a phone number has been
    reduced to ten digits by `normalize_phone`, and an address has been through a
    regex that permits no whitespace, so neither can carry a sentence. Anything
    longer than a real one is simply not repeated — the sentence reads fine
    without it, and the office still has the address itself.
    """
    name = _first_name(lead.get("name") or "")
    opener = f"Sent, {name}." if name else "Sent."

    where = ""
    email = lead.get("email") or ""
    phone = lead.get("phone") or ""
    if lead.get("contact_method") == "Email" and 0 < len(email) <= MAX_ECHOED:
        where = f" by email at {email}"
    elif 0 < len(phone) <= MAX_ECHOED:
        where = f" on {phone}"

    return (
        f"{opener} Your question is with the office and someone will come back to "
        f"you{where} — usually the same day. Anything else I can help with in the "
        "meantime, just ask."
    )


def _failure(lead_type: str) -> str:
    what = (
        "you asked me a question in the chat"
        if lead_type == "question"
        else "you filled in the form"
    )
    return (
        "I've got everything down, but something went wrong sending it over to "
        "the office — I don't want to tell you it's handled when it isn't. Give "
        f"us a call on {settings.COMPANY_PHONE} and mention {what}; they'll take "
        "it from there."
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
        return {"messages": [AIMessage(content=_failure(lead_type))]}

    # Only once the office has them. Deleting before the send would lose the
    # photos on a failure, and they are the part a customer will not redo.
    uploads.discard(thread_id)

    return {
        "messages": [AIMessage(content=_confirmation(lead_type, lead, len(attachments)))],
        "lead_submitted": True,
        # Kept for the rest of the conversation. If they come back with a second
        # question — which is exactly what someone does once they find out this
        # works — we already have their name and how to reach them, and asking
        # again reads as not having been listening the first time.
        "known_contact": {k: lead[k] for k in CARRIED_OVER if lead.get(k)},
    }
