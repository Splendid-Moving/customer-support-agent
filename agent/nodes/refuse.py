"""
NODE: Refuse
PURPOSE: Say no, in persona, without a model call.
INPUT:   state.guard
OUTPUT:  {"messages": [AIMessage]}

WHY THIS TEXT IS FIXED
----------------------
The obvious implementation asks the model to write a polite refusal. That hands
the attacker exactly what they wanted: their text, in the model's context,
shaping the model's output. Prompt-extraction attempts in particular work by
getting the agent to *talk about* its instructions — and a generated refusal is
the agent talking about its instructions.

Fixed strings cannot be argued with. They also cannot leak, cannot be
manipulated into a different shape, and cost nothing.

The wording matters too. A blocked message gets a normal, unbothered reply that
steers back to moving. It does not accuse anyone, does not say "I detected a
prompt injection", and does not explain what the rules are. Someone probing
learns nothing; the far more common case — a customer whose ordinary question
tripped a pattern — sees a helpful human, not a security system.
"""

import logging

from langchain_core.messages import AIMessage

from agent.state import SupportState
from services import config

logger = logging.getLogger(__name__)


REPLIES: dict[str, str] = {
    # Someone trying to reprogram the agent. Answer as if they'd said something
    # unremarkable, and get straight back to work.
    "injection": (
        "I'm just here to help with moving — rates, what's included, getting you "
        "an estimate, that sort of thing. What can I help you with?"
    ),
    # Genuinely unrelated. Friendly, brief, and points at what we do.
    "off_topic": (
        "That one's outside what I can help with, sorry. I'm the moving side of "
        "things — happy to talk rates, scheduling, or getting an estimate put "
        "together for you."
    ),
    # A wall of text. Almost always a paste, occasionally a very thorough
    # customer, so the reply assumes the second.
    "too_long": (
        "That's a lot to take in at once — could you give me the short version? "
        "Or if it's all detail about your move, it's better on an estimate form "
        "so a manager sees the whole thing properly."
    ),
    # A long conversation. Close it out with a human rather than looping forever.
    "turn_limit": (
        "We've covered a lot here. At this point you'll get further with a real "
        f"person — give the office a call on {config.COMPANY_PHONE} and they'll "
        "pick it up from here."
    ),
}

FALLBACK = (
    "I'm not sure I followed that. I can help with rates, what's included in a "
    "move, or getting you an estimate — what were you after?"
)


def refuse(state: SupportState) -> dict:
    verdict = state.get("guard", "off_topic")
    logger.info("Refuse: %s (%s)", verdict, state.get("guard_reason", ""))
    return {"messages": [AIMessage(content=REPLIES.get(verdict, FALLBACK))]}
