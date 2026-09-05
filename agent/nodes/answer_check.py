"""
NODE: Answer check
PURPOSE: Refuse to let a bad draft reach the customer.
INPUT:   state.draft
OUTPUT:  an AIMessage on `messages`, or a complaint and a trip back to knowledge

WHY THIS IS PYTHON AND NOT A PROMPT
-----------------------------------
Both rules enforced here are already stated plainly in the system prompt, and the
model follows them most of the time. Most of the time is the problem. These are
the two mistakes that are worth catching every single time:

1. THIRD PERSON. "Splendid Moving offers free estimates" instead of "we do free
   estimates". A customer reading the first one knows immediately they are
   talking to a bot reciting a page. It is the difference between the agent being
   a member of the team and being a brochure with a text box.

2. A PRICE WE NEVER PUBLISHED. Every dollar figure in the answer has to appear
   verbatim in the knowledge base. This is a cheap, complete grounding check on
   the one class of fact that costs real money when it is wrong: a customer told
   "$400" in chat will hold us to $400.

A rejected draft goes back once with a specific complaint. If the second attempt
fails too, the customer gets a phone number rather than a third try at the same
mistake — a model that has broken the same rule twice is not about to get it
right unprompted.
"""

import logging
import re

from langchain_core.messages import AIMessage

from agent.state import SupportState
from services import config, knowledge

logger = logging.getLogger(__name__)

#: One rewrite, then hand off. Attempt 1 drafts, attempt 2 is the retry.
MAX_ATTEMPTS = 2


# ── Rule 1: first person ───────────────────────────────────────────────────────
# Each entry is (pattern, what to tell the model it did wrong). The complaints
# are written to be actionable on their own, because that string is the entire
# instruction the retry gets.

THIRD_PERSON_RULES: list[tuple[str, str]] = [
    (
        r"\bmoving compan(?:y|ies)\s+(?:usually|typically|generally|often|tend|will|can|charge|do)\b",
        "you described what moving companies in general do. You do not speak for "
        "the industry — answer for us, in the first person.",
    ),
    (
        r"\bmost (?:movers|moving companies)\b",
        "you generalised about other movers. Answer only for us, as one of us.",
    ),
    (
        r"\bin the moving industry\b",
        "you referred to the moving industry from the outside. You work here — "
        "just answer for us.",
    ),
    (
        r"\bSplendid Moving(?:'s|s')\b",
        "you wrote about Splendid Moving in the possessive, from outside the "
        "company. It is 'our', not 'Splendid Moving's'.",
    ),
    (
        r"\bSplendid Moving\s+(?:offers?|provides?|charges?|includes?|does|do|has|have|will|can|is|are|brings?|uses?)\b",
        "you wrote about Splendid Moving in the third person. You work here — "
        "say 'we offer', 'we charge', 'we bring'.",
    ),
    (
        r"\bthe company\s+(?:offers?|provides?|charges?|does|has|will|can|is|requires?)\b",
        "you called us 'the company'. Say 'we'.",
    ),
    (
        r"\bthey\s+(?:offer|provide|charge|include|require|bring|use)\b",
        "you referred to us as 'they'. Say 'we'.",
    ),
    (
        r"\btheir\s+(?:rates?|prices?|crew|movers|team|trucks?|polic(?:y|ies)|minimum)\b",
        "you called our own rates and crew 'theirs'. Say 'our'.",
    ),
]

_THIRD_PERSON = [(re.compile(p, re.IGNORECASE), why) for p, why in THIRD_PERSON_RULES]


def find_third_person(text: str) -> str | None:
    """The complaint for the first voice rule the text breaks, or None."""
    for pattern, why in _THIRD_PERSON:
        if pattern.search(text):
            return why
    return None


# ── Rule 2: no price we did not publish ────────────────────────────────────────

_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)")


def _amounts(text: str) -> set[str]:
    """
    Dollar figures in a form that can be compared.

    Normalised so that "$1,200", "$ 1200" and "$1200.00" are one amount — the
    knowledge base and the model will not agree on formatting, and a check that
    trips over a comma would reject every correct answer.
    """
    found = set()
    for raw in _MONEY.findall(text):
        value = raw.replace(",", "")
        value = value.rstrip("0").rstrip(".") if "." in value else value
        found.add(value)
    return found


def find_unpublished_price(text: str) -> str | None:
    """
    The complaint if the draft quotes a dollar figure the knowledge base doesn't.

    Note what this does NOT try to do: it does not check that the number is used
    in the right context. That needs a model. What it does catch completely is
    the failure that actually happens — a number that exists nowhere in our
    material, which is always either invented or arithmetic the agent was told
    not to do.
    """
    published = _amounts(knowledge.all_context())
    invented = sorted(_amounts(text) - published)
    if not invented:
        return None
    listed = ", ".join(f"${a}" for a in invented)
    return (
        f"you quoted {listed}, which does not appear anywhere in the reference "
        "material. Never invent a figure and never add ours up into a total. "
        "Quote our rates exactly as written, or say the job needs a real estimate."
    )


# ── The node ───────────────────────────────────────────────────────────────────

def _handoff_line() -> str:
    """
    What the customer gets when the agent could not write a clean answer twice.

    Fixed text, not model-generated: the model has just demonstrated twice that
    it is not reliable on this turn, so asking it for one more sentence is not a
    recovery, it is a third roll of the dice.
    """
    return (
        "I want to make sure I get this exactly right rather than guess. Want me "
        "to send it over to the office so someone can come back to you with it? "
        f"Or the quickest way is to call us on {config.COMPANY_PHONE}, any day "
        "from 6am."
    )


def check(state: SupportState) -> dict:
    draft = (state.get("draft") or "").strip()
    attempts = state.get("answer_attempts", 0)

    if not draft:
        logger.warning("Answer check: empty draft on attempt %d", attempts)
        complaint = "you returned nothing at all."
    else:
        complaint = find_third_person(draft) or find_unpublished_price(draft)

    if complaint is None:
        return {"messages": [AIMessage(content=draft)], "answer_complaint": ""}

    if attempts >= MAX_ATTEMPTS:
        logger.warning("Answer check: giving up after %d attempts — %s", attempts, complaint)
        return {"messages": [AIMessage(content=_handoff_line())], "answer_complaint": ""}

    logger.info("Answer check: rejected attempt %d — %s", attempts, complaint)
    return {"answer_complaint": complaint}


def next_step(state: SupportState) -> str:
    """
    Conditional edge. A complaint still standing means the draft was sent back.

    `check` clears `answer_complaint` on every path that produces a message, so
    a non-empty complaint means exactly one thing: rewrite.
    """
    return "knowledge" if state.get("answer_complaint") else "__end__"
