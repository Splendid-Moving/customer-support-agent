"""
The estimate form: what it asks, and what counts as a valid answer.

ONE definition, used three times — the browser renders the form from it, Python
validates the submission against it, and the email to the office is laid out from
it. Three hand-maintained copies of the same field list is how a form ends up
collecting something nobody reads, or emailing a field that was quietly renamed.

WHY THE FORM AND NOT A CONVERSATION
-----------------------------------
An agent could ask these nine questions one at a time in chat. It would be worse.
People abandon a chat interrogation halfway; they fill in a form. And the answers
arrive as clean fields instead of prose someone has to re-read, which matters
because a human — not this agent — is the one acting on them.

WHAT WE ASK FOR AND WHY
-----------------------
Only what a manager cannot work out for themselves and cannot quote without. Each
extra field costs completed forms, so anything a manager can ask on the phone in
ten seconds is not here.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Literal

LeadType = Literal["estimate", "long_distance"]


@dataclass(frozen=True)
class Field_:
    """
    One thing we need from the customer.

    `label` is how it is titled in the email a manager reads. `ask` is how the
    agent asks for it in the chat — a different job, and the reason both exist.
    "Moving from" is a good column heading and a terrible thing to say out loud.

    `ask` may contain `{name}`, filled in with what they have already told us, so
    the second question can use their first name.
    """

    name: str
    label: str
    ask: str = ""
    kind: Literal["text", "tel", "email", "date", "select", "textarea"] = "text"
    required: bool = False
    placeholder: str = ""
    help: str = ""
    options: tuple[str, ...] = ()

    #: Can be prefilled from what the customer already said in the conversation.
    #: Only ever set on fields where a wrong guess is obvious to the customer and
    #: harmless to the office. Never on name, phone or email — a mis-heard phone
    #: number is a lead nobody can call back.
    extractable: bool = False

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["options"] = list(self.options)
        return out

    def question(self, answers: dict[str, str]) -> str:
        first = (answers.get("name") or "").split()
        return (self.ask or f"What's your {self.label.lower()}?").format(
            name=first[0] if first else "there"
        )


HOME_SIZES = (
    "Studio",
    "1 bedroom",
    "2 bedrooms",
    "3 bedrooms",
    "4+ bedrooms",
    "Office / commercial",
    "Just a few items",
)

FLEXIBILITY = (
    "That exact date",
    "Within a few days",
    "Within a couple of weeks",
    "Not decided yet",
)


# ── Shared fields ──────────────────────────────────────────────────────────────

_CONTACT = [
    Field_("name", "Your name",
           ask="Happy to help with that. First off — what's your name?",
           required=True, placeholder="Jordan Lee"),
    Field_("phone", "Phone", kind="tel",
           ask="Thanks {name}. What's the best number for a manager to reach you on?",
           required=True, placeholder="(323) 555-0142"),
    Field_("email", "Email", kind="email",
           ask="And an email address?",
           required=True, placeholder="you@example.com"),
]

_FROM = Field_("from_address", "Moving from",
               ask="Where are we moving you from? Street and city is plenty.",
               required=True, placeholder="Street, city", extractable=True)

_MOVE = [
    Field_("move_date", "Move date", kind="date",
           ask="What date are you looking at? If it's not settled yet, just say so.",
           help="Or tell me you're not sure.", extractable=True),
    Field_("home_size", "Size of the place", kind="select",
           ask="How big is the place we're moving?",
           required=True, options=HOME_SIZES, extractable=True),
]

_EXTRAS = [
    Field_("access", "Stairs or elevator?", kind="text",
           ask="Any stairs or elevators involved, at either end? That's usually the "
               "biggest factor in how long a move takes.",
           placeholder="3rd floor walk-up, elevator at the new place"),
    Field_("notes", "Anything else we should know?", kind="textarea",
           ask="Last one — anything else I should pass on? Piano, a safe, packing "
               "help, tight parking, that sort of thing.",
           placeholder="Anything at all"),
]


FORMS: dict[str, dict[str, Any]] = {
    "estimate": {
        "title": "Get an estimate",
        "opening": (
            "Happy to get that sorted. Let me grab a few details and a manager "
            "will put together a real estimate for you."
        ),
        "fields": [
            *_CONTACT,
            _FROM,
            Field_("to_address", "Moving to",
                   ask="And where are we taking it?",
                   required=True, placeholder="Street, city", extractable=True),
            *_MOVE,
            *_EXTRAS,
        ],
    },
    "long_distance": {
        "title": "Long-distance move",
        "opening": (
            "Out-of-state moves aren't billed hourly like our local jobs — a "
            "manager prices each one individually. Let me take a few details and "
            "they'll come straight back to you."
        ),
        "fields": [
            *_CONTACT,
            _FROM,
            Field_("to_address", "Moving to",
                   ask="And where are you heading? City and state is enough.",
                   required=True, placeholder="City and state", extractable=True),
            *_MOVE,
            Field_("flexibility", "How firm is that date?", kind="select",
                   ask="How firm is that date?", options=FLEXIBILITY),
            *_EXTRAS,
        ],
    },
}

#: The photo step. Not a Field_ because it is not typed and not validated — it is
#: its own kind of turn, and it is the single most valuable thing on the form:
#: a manager quoting from photos is quoting, a manager quoting from "2 bedrooms"
#: is guessing.
PHOTO_STEP = {
    "name": "photos",
    "kind": "photos",
    "ask": (
        "Last thing, and it's the one that makes the biggest difference — can you "
        "send a few photos of what's moving? A shot of each room is plenty. "
        "Skip it if now's not a good time."
    ),
}


def spec(lead_type: str) -> dict[str, Any]:
    """The form definition, in the shape the browser renders."""
    form = FORMS.get(lead_type) or FORMS["estimate"]
    return {
        "lead_type": lead_type if lead_type in FORMS else "estimate",
        "title": form["title"],
        "opening": form["opening"],
        "fields": [f.to_json() for f in form["fields"]],
        "photo_step": dict(PHOTO_STEP),
    }


def fields_for(lead_type: str) -> list[Field_]:
    return list((FORMS.get(lead_type) or FORMS["estimate"])["fields"])


# ── Validation ─────────────────────────────────────────────────────────────────
# Everything here runs in Python on the server. The browser validates too, for a
# decent experience, but that check is a convenience and can be skipped by
# anyone posting to the endpoint directly — so it is never the one that counts.

_EMAIL = re.compile(r"^[^@\s]+@[^@\s.]+(?:\.[^@\s.]+)+$")


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def normalize_phone(value: str) -> str:
    """
    US phone as (323) 555-0142, or the input unchanged if it isn't one.

    A manager reads this off a screen and dials it, so consistent formatting is
    worth the six lines. Leading US country code is dropped.
    """
    digits = _digits(value)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return value.strip()
    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"


#: Ways a customer says "I don't know yet" to the date question. Treated as a
#: blank answer rather than an error — a manager would rather hear "not sure"
#: than watch someone invent a date to get past the question.
UNDECIDED = {
    "not sure", "no", "nope", "unsure", "dont know", "don't know", "no idea",
    "flexible", "not yet", "tbd", "havent decided", "haven't decided", "skip",
    "not settled", "undecided", "idk", "n/a", "na",
}


def is_undecided(value: str) -> bool:
    cleaned = re.sub(r"[^a-z' ]", "", str(value).lower()).strip()
    return cleaned in UNDECIDED


def validate_one(field: Field_, value: Any) -> str | None:
    """
    Check one answer. Returns a message to say back, or None if it's fine.

    Phrased the way a person would say it, because the agent repeats it verbatim
    into the conversation — "That doesn't look like a full phone number, can you
    give me all ten digits?" reads like a colleague; "Invalid input" does not.
    """
    value = value.strip() if isinstance(value, str) else value
    text = str(value or "")

    if not text:
        if field.required:
            return "I do need that one — what should I put down?"
        return None

    if field.kind == "email" and not _EMAIL.match(text):
        return "That doesn't look quite like an email address — mind checking it?"

    if field.kind == "tel" and len(_digits(text)) not in (10, 11):
        return "That doesn't look like a full number — can I get all ten digits?"

    if field.kind == "date":
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d").date()
        except ValueError:
            return "I didn't catch that as a date — use the picker, or tell me you're not sure yet."
        if parsed < date.today():
            return "That date's already gone by — did you mean a later one?"

    if field.kind == "select" and field.options and text not in field.options:
        return "Pick whichever of those is closest."

    return None


def validate(lead_type: str, submitted: dict[str, Any]) -> dict[str, str]:
    """
    Check a whole submission at once. Returns {field_name: message}; empty is good.

    Still the authority even though the agent now collects one answer at a time:
    `validate_one` guards the conversation, this guards what is about to be
    emailed, and it runs over everything regardless of how it got there.
    """
    errors: dict[str, str] = {}
    values = {k: (v.strip() if isinstance(v, str) else v) for k, v in (submitted or {}).items()}

    for f in fields_for(lead_type):
        if message := validate_one(f, values.get(f.name, "")):
            errors[f.name] = message
    return errors


def clean(lead_type: str, submitted: dict[str, Any]) -> dict[str, str]:
    """
    A validated submission, tidied and limited to fields the form actually asks
    for.

    Unknown keys are dropped rather than passed through. The submission arrives
    from the browser, so it is whatever someone chose to post — and anything that
    survives this function ends up in an email a manager reads.
    """
    values = {k: (v.strip() if isinstance(v, str) else v) for k, v in (submitted or {}).items()}
    out: dict[str, str] = {}
    for f in fields_for(lead_type):
        value = values.get(f.name) or ""
        if not value:
            continue
        if f.kind == "tel":
            value = normalize_phone(str(value))
        # Belt and braces against a pathological paste in a free-text box.
        out[f.name] = str(value)[:1000]
    return out


def label_for(lead_type: str, name: str) -> str:
    """Human label for a field name, for laying out the email."""
    for f in fields_for(lead_type):
        if f.name == name:
            return f.label
    return name.replace("_", " ").capitalize()
