"""
NODE: Router
PURPOSE: Pick which lane answers this turn.
INPUT:   state.messages
OUTPUT:  {"intent": ...}

Only runs on turns the guard passed, and only on genuinely new turns — when the
graph is paused at the form, a Command(resume=...) re-enters the paused node
directly and the router is never called.

THE JUDGEMENT CALL THIS NODE MAKES
----------------------------------
The expensive mistake is not sending a question to the wrong lane. It is opening
the form on someone who only asked a question. A customer who asks "what do you
charge for two guys?" and gets a nine-field form instead of the rate has been
handed a chore, and they close the tab.

So the bar for the form is intent, not topic: the customer has to be asking for
an estimate, or describing their own move in a way that only a real quote can
answer. Merely mentioning price, or asking what things generally cost, is the
knowledge lane's job — and the knowledge lane will offer the form itself when it
genuinely can't answer.

Out-of-state is the one exception. We do not price those hourly, so there is no
answer the knowledge lane can give, and it goes to the form on topic alone.
"""

import logging

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from agent.models import get_model
from agent.state import SupportState

logger = logging.getLogger(__name__)


class Route(BaseModel):
    """Structured routing decision."""

    intent: str = Field(
        description="One of: 'knowledge', 'estimate', 'long_distance', 'handoff'."
    )
    reasoning: str = Field(description="One short sentence explaining the choice.")


VALID = ("knowledge", "estimate", "long_distance", "handoff")


SYSTEM_PROMPT = """\
You route messages arriving in the chat on a Los Angeles moving company's \
website. You do not answer them. Pick exactly one lane.

**knowledge** — the default. Any question we can answer from our own material: \
rates, the 3-hour minimum, what's included, packing, furniture disassembly, \
service area, hours, licensing, deposits, how estimates work. Also greetings, \
thanks, small talk, and anything vague.
  "how much for 2 movers?"
  "do you take apart beds?"
  "are you insured?"
  "hey"
  "do you cover Pasadena?"
  "¿cuánto cuesta mudar un apartamento de una recámara?"

**estimate** — the customer is asking us to price or size up THEIR move, here in \
the LA area, or is offering the details of it.
  "can I get a quote?"
  "how much would it cost to move my 2 bedroom in Silver Lake on the 14th?"
  "I'm moving from Echo Park to Culver City next month, what would that run?"
  "can I send you photos of my stuff?"

**long_distance** — the move leaves California, or is far enough that it is not \
an hourly local job. Route here on the destination alone, even for a bare \
question, because we do not price these hourly and nothing in our material \
answers them.
  "do you do moves to Texas?"
  "I'm relocating to Seattle in June"
  "how much to move from LA to Phoenix?"

**handoff** — they want something DONE to a specific existing booking, or they \
have a complaint about a job we already did. The test is whether a person needs \
to look their record up.
  "I need to move my booking to Saturday"
  "your guys scratched my table"
  "I paid the deposit but never got a confirmation"
  "can I talk to a human?"

Rules:
- If they are only asking what something costs IN GENERAL, that is **knowledge** \
— we publish our hourly rates by crew size and they should just get the number. \
This holds in every language: "how much for a 1 bedroom" and "cuánto cuesta \
mudar un apartamento de una recámara" are the same question, and both are \
**knowledge**. Mentioning the size of a home is describing the question, not \
requesting a quote.
- It is **estimate** only when the move being priced is clearly their own.
- Anything leaving California is **long_distance**, whichever way it is phrased.
- Asking what a POLICY IS — cancellation, deposits, damage, payment, heavy items \
— is **knowledge**, even when phrased personally. "Can I cancel if my closing \
falls through?" wants the cancellation policy, and we publish it. It only \
becomes **handoff** when they want us to actually cancel or change a booking \
that already exists.
- When torn between knowledge and estimate, choose **knowledge**. Answering a \
question costs nothing; opening a form on someone who wanted a one-line answer \
loses the job."""


def route(state: SupportState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "knowledge"}

    model = get_model("router").with_structured_output(Route)

    # Only the recent turns. Routing is about the current message, and a long
    # history drags the classifier toward whatever dominated earlier in the
    # conversation.
    recent = messages[-4:]
    decision = model.invoke([SystemMessage(content=SYSTEM_PROMPT), *recent])

    intent = decision.intent if decision.intent in VALID else "knowledge"
    logger.info("Router: %s — %s", intent, decision.reasoning)
    return {"intent": intent}


def pick_lane(state: SupportState) -> str:
    """
    Conditional-edge function. Maps state.intent to a node name.

    Anything unrecognised goes to the knowledge lane rather than through. A name
    that is not a node raises inside LangGraph and takes the whole turn down —
    and the fallback is the safe one anyway: answering a question costs nothing,
    where opening the form on someone who did not ask for it loses the job.
    """
    intent = state.get("intent", "knowledge")
    # Both lead types share one node; only the form it shows differs.
    if intent in ("estimate", "long_distance"):
        return "collect_lead"
    return intent if intent in ("knowledge", "handoff") else "knowledge"
