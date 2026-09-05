"""
NODE: Knowledge
PURPOSE: Answer a customer's question using only the knowledge base.
INPUT:   state.messages, and on a retry, state.draft + the check's complaint
OUTPUT:  {"draft": str, "answer_attempts": int}

Note that this node writes to `draft`, NOT to `messages`. The answer only joins
the conversation once agent/nodes/answer_check.py has passed it. An answer that
breaks the voice rules or quotes a price we never published should never have
existed as far as the customer is concerned, and keeping it out of `messages`
also keeps it out of the history the next turn is built from — otherwise the
model reads its own bad phrasing as an example to follow.
"""

import logging

from langchain_core.messages import SystemMessage

from agent.models import get_model
from agent.state import SupportState
from schemas import persona
from services import knowledge

logger = logging.getLogger(__name__)


TASK = """\
# This turn

Answer the customer's latest question using the reference material below.

If the reference material answers it, answer it — briefly, in your own voice, \
and without repeating the question back to them.

If it does not, say you are not sure off the top of your head and OFFER TO SEND \
THE QUESTION OVER TO THE OFFICE so someone can come back to them with the \
answer. Ask whether they would like you to. Do not apologise twice, do not \
explain why you cannot answer, and do not substitute something close. One honest \
sentence beats a paragraph of hedging.

Make that offer as a plain yes-or-no question, and do NOT ask for their name, \
number or email in the same breath. If they say yes, you will take those details \
properly, one at a time, on the next turn — asking for them here means asking \
for them twice.

Never tell someone to call us INSTEAD of offering to send their question over. \
The number is worth giving as the faster option if they are in a hurry, never as \
the only one.

If the question needs a real estimate — how long a specific move will take, what \
a specific move will cost, whether we can handle a particular heavy item — say \
that is worth getting a proper estimate for and that you can take some details. \
Do not attempt the calculation."""


#: Appended when answer_check sends a draft back. Kept short and specific: a
#: general "try again" produces a differently-worded version of the same mistake.
RETRY_TEMPLATE = """\
# Correction

Your previous attempt was rejected: {complaint}

Here is what you wrote:

{draft}

Write it again, fixing exactly that. Keep everything that was fine."""


def answer(state: SupportState) -> dict:
    attempts = state.get("answer_attempts", 0)
    task = TASK

    if attempts and (complaint := state.get("answer_complaint")):
        task += "\n\n" + RETRY_TEMPLATE.format(
            complaint=complaint, draft=state.get("draft", "")
        )

    prompt = persona.system_prompt(reference=knowledge.all_context(), extra=task)

    # Only the recent turns. A long history pulls the model toward whatever it
    # said earlier — including, on a retry, the phrasing that just got rejected.
    recent = state["messages"][-6:]

    response = get_model("knowledge").invoke([SystemMessage(content=prompt), *recent])
    text = response.content if isinstance(response.content, str) else str(response.content)

    logger.info("Knowledge: drafted %d chars (attempt %d)", len(text), attempts + 1)
    return {"draft": text.strip(), "answer_attempts": attempts + 1}
