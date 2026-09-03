"""
NODE: Handoff
PURPOSE: Hand the customer to a person, without pretending we did anything.
INPUT:   state.messages
OUTPUT:  {"messages": [AIMessage]}

Reached when the customer needs something this agent has no access to: an
existing booking, a bill, a change of date, damage to their furniture, or simply
a human.

The one rule this node exists to enforce is that it must not imply action. "I've
passed this on to a manager" is a lie — nothing was passed anywhere — and it is
the kind of lie a customer only discovers when nobody calls them back. It
acknowledges what they said, says plainly that it cannot do it, and gives the
phone number.
"""

import logging

from langchain_core.messages import SystemMessage

from agent.models import get_model
from agent.state import SupportState
from schemas import persona
from services import config

logger = logging.getLogger(__name__)


TASK = f"""\
# This turn

The customer needs something you cannot do — it involves an existing booking, a \
bill, a complaint, damage, or they have asked for a person.

Write two or three sentences, no more:

1. Acknowledge specifically what they said. If something went wrong on a job, say \
sorry like a person would, not like a policy.
2. Say plainly that you can't handle that side of things yourself.
3. Give them {config.COMPANY_PHONE} and say the office is open every day, 6am to \
11pm.

Never say you have passed it on, flagged it, escalated it, notified anyone, or \
created a ticket. You have not. Nothing you do here reaches anybody, and a \
customer who believes otherwise sits waiting for a call that is not coming.

Never guess at what happened, what we will do about it, or what anything will \
cost."""


def handoff(state: SupportState) -> dict:
    prompt = persona.system_prompt(extra=TASK)
    response = get_model("reply").invoke(
        [SystemMessage(content=prompt), *state["messages"][-4:]]
    )
    logger.info("Handoff: pointed customer at the office")
    return {"messages": [response]}
