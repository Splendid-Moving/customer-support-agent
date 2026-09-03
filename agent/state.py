"""
Graph state — the shared memory every node reads and writes.

Two decisions here are worth explaining.

`photo_ids` holds IDENTIFIERS, never image data. Photos are uploaded to the
server separately and land on disk; only their short ids travel through the
graph. LangGraph writes the whole state into the checkpoint database after every
step, so a few base64 phone photos in state would be re-serialised on every turn
of the conversation — the database balloons and each message gets slower. Ids are
a dozen bytes.

`lead_submitted` exists so the email cannot be sent twice. The node that sends it
sits after an interrupt, and anything after an interrupt can be re-entered if the
browser retries or the customer double-clicks Submit.
"""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

#: Where the router can send a turn. Each is a lane with its own node.
Intent = Literal["knowledge", "estimate", "long_distance", "handoff"]

#: Why the guard blocked a turn. "clean" means it did not.
GuardVerdict = Literal[
    "clean",
    "off_topic",       # not about moving, or about us, at all
    "injection",       # trying to change the agent's instructions or extract them
    "too_long",        # over MAX_MESSAGE_CHARS
    "turn_limit",      # this conversation has gone on long enough
]

#: Which form to show. Same node, different fields.
LeadType = Literal["estimate", "long_distance"]


class SupportState(TypedDict, total=False):
    """
    `total=False` because most fields only exist once the relevant branch has
    run — a plain FAQ turn never populates any of the lead fields.
    """

    #: Conversation history. add_messages appends and de-duplicates by id.
    messages: Annotated[list[AnyMessage], add_messages]

    #: The guard's decision about the CURRENT turn.
    guard: GuardVerdict

    #: One short sentence explaining the guard's call. Logged, never shown.
    guard_reason: str

    #: Which lane the router picked for the current turn.
    intent: Intent

    #: Which form is being collected, if any.
    lead_type: LeadType

    #: The submitted form, once the customer has filled it in.
    lead: dict[str, Any]

    #: Ids of photos uploaded for this thread. NOT the photos themselves.
    photo_ids: list[str]

    #: True once the lead email has actually gone out. Guards against a second send.
    lead_submitted: bool

    #: The knowledge lane's answer BEFORE answer_check has passed it. It is not
    #: appended to `messages` until it clears the check, so a draft that breaks
    #: the voice rules is never part of the conversation the customer sees.
    draft: str

    #: How many times the knowledge lane has tried to write this answer. Two is
    #: the limit; after that the customer gets a phone number instead of a third
    #: attempt at the same mistake.
    answer_attempts: int

    #: What answer_check objected to, in one sentence, fed back into the retry.
    #: A general "try again" just produces a reworded version of the same slip.
    answer_complaint: str
