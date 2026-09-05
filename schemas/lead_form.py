"""
What the agent collects before it emails the office, and what counts as a valid
answer.

Three of them now. Two ask a manager to price a move — `estimate` and
`long_distance`. The third, `question`, exists because of a real conversation:
someone asked what we charge to haul away a fridge, the agent correctly said it
did not know and offered to have the office get back to them, took their name and
number — and then told them to call us, because taking a message was not
something it could actually do. The offer was honest when it was made and a dead
end one turn later.

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

LeadType = Literal["estimate", "long_distance", "question"]


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

    #: `(field_name, value)` — only ask this if that earlier answer came back
    #: exactly that. It is data rather than a callable so the whole field can be
    #: JSON-encoded into the interrupt payload and written to the checkpoint.
    #:
    #: One branch uses it: someone who says they would rather be phoned is asked
    #: for a number and never for an email address. Asking for both is how a
    #: two-question detour becomes a four-question form.
    only_if: tuple[str, str] | None = None

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["options"] = list(self.options)
        out["only_if"] = list(self.only_if) if self.only_if else None
        return out

    def applies(self, answers: dict[str, Any]) -> bool:
        """Whether this field is part of the form given what has been answered."""
        if not self.only_if:
            return True
        depends_on, value = self.only_if
        return str(answers.get(depends_on) or "").strip() == value

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

#: How the office should come back to them. Two options, tapped not typed, and
#: they decide which contact field is asked for next.
CONTACT_METHODS = ("Phone", "Email")


# ── Shared fields ──────────────────────────────────────────────────────────────

_CONTACT = [
    # No greeting here. The lane's `opening` has already said hello one bubble
    # earlier, and two welcomes back to back is the most obviously robotic thing
    # a chat agent can do.
    Field_("name", "Your name",
           ask="First off — what's your name?",
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


# ── The question form ──────────────────────────────────────────────────────────
# Shorter than the estimate on purpose. Nobody who asked one question expects to
# be interviewed for it, and every field here past the third is a customer
# deciding it was not that important after all.

_QUESTION = [
    Field_("question", "Their question", kind="textarea",
           ask="So they answer the right thing — what would you like to know?",
           required=True, placeholder="What you'd like to know",
           extractable=True),
    Field_("name", "Your name",
           ask="And who am I sending this over for?",
           required=True, placeholder="Jordan Lee"),
    Field_("contact_method", "Get back to them by", kind="select",
           ask="What's the best way for them to get back to you, {name} — phone or "
               "email?",
           required=True, options=CONTACT_METHODS),
    Field_("phone", "Phone", kind="tel",
           ask="What's the best number to reach you on?",
           required=True, placeholder="(323) 555-0142",
           only_if=("contact_method", "Phone")),
    Field_("email", "Email", kind="email",
           ask="What's the best email address for you?",
           required=True, placeholder="you@example.com",
           only_if=("contact_method", "Email")),
    # The "anything else?" the office would ask on the phone. It is the last
    # question rather than a throwaway line because a second question answered in
    # the same callback is one fewer reason for anybody to call twice.
    Field_("anything_else", "Also asked", kind="textarea",
           ask="Anything else you wanted to ask while I've got you? If not, just "
               "say no and I'll send this over.",
           placeholder="Anything else at all"),
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
    "question": {
        "title": "Question for the office",
        "opening": (
            "I'd rather get you a proper answer than guess at it. Let me take a "
            "couple of details and someone from the office will come straight "
            "back to you."
        ),
        # Used instead when this customer has already given us their details
        # earlier in the conversation. Offering to "take a couple of details"
        # off someone who handed them over two minutes ago is the same failure
        # as asking for them twice.
        "opening_again": (
            "I'll send that one over as well. I've still got your details, so "
            "there's nothing to fill in again."
        ),
        "fields": _QUESTION,
        # No photos. They earn their place on a move a manager has to price;
        # on a two-line question they are one more thing to say no to.
        "photos": False,
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
        "opening_again": form.get("opening_again", ""),
        "fields": [f.to_json() for f in form["fields"]],
        "photo_step": dict(PHOTO_STEP),
        "photos": wants_photos(lead_type),
    }


def fields_for(lead_type: str) -> list[Field_]:
    return list((FORMS.get(lead_type) or FORMS["estimate"])["fields"])


def wants_photos(lead_type: str) -> bool:
    """Whether this form asks for photos at all. Every move does; a question doesn't."""
    return bool((FORMS.get(lead_type) or FORMS["estimate"]).get("photos", True))


def applicable_fields(lead_type: str, answers: dict[str, Any]) -> list[Field_]:
    """The fields that are part of this form given what has been answered so far."""
    return [f for f in fields_for(lead_type) if f.applies(answers)]


def steps_remaining(lead_type: str, answers: dict[str, Any]) -> int:
    """
    How many questions are still to come, for the progress rail.

    An either/or branch counts ONCE while it is still open. Someone who has not
    yet said whether they want a call or an email will be asked for exactly one
    of the two, so counting both would show a form one step longer than it is —
    and a progress bar that jumps backwards is worse than no progress bar.
    """
    count = 0
    open_branches: set[str] = set()

    for f in fields_for(lead_type):
        if answers.get(f.name):
            continue
        if not f.only_if:
            count += 1
            continue
        depends_on, _ = f.only_if
        if answers.get(depends_on):
            # The branch is decided: this field is either in or out for real.
            count += int(f.applies(answers))
        elif depends_on not in open_branches:
            open_branches.add(depends_on)
            count += 1
    return count


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


#: Ways of saying "nothing to add" to an open question. The last thing an
#: optional box asks is "anything else?", and the honest answer is usually no —
#: which is not a piece of information about the move. Emailing a manager a row
#: reading "Anything else we should know? — no" is worse than emailing no row at
#: all, because they read it looking for the point.
NOTHING = {
    "no", "nope", "nah", "no thanks", "no thank you", "nothing", "nothing else",
    "none", "that's it", "thats it", "that's all", "thats all", "all good",
    "im good", "i'm good", "no im good", "no i'm good", "nada", "n/a", "na",
    "skip", "not really", "no not really", "dont think so", "don't think so",
}


def is_nothing(value: str) -> bool:
    cleaned = re.sub(r"[^a-z' ]", "", str(value).lower()).strip()
    return cleaned in NOTHING


def coerce_option(field: Field_, value: str) -> str:
    """
    A typed answer to a multiple-choice question, matched to one of the options.

    The buttons are there to be tapped, but the message box never goes away — so
    someone will type "email" to a question offering "Email", and being told to
    "pick whichever of those is closest" when they just did is the kind of small
    stupidity that ends a conversation.

    Only ever returns an option or the input unchanged, and only when the match
    is unambiguous. Two candidates means we genuinely do not know which they
    meant, and asking again is the right answer.
    """
    if field.kind != "select" or not field.options:
        return value

    text = str(value or "").strip().lower()
    if not text:
        return value

    for option in field.options:
        if text == option.lower():
            return option

    near = [
        option for option in field.options
        if option.lower().startswith(text) or option.lower() in text
    ]
    return near[0] if len(near) == 1 else value


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

    # `applicable_fields`, not `fields_for`: a customer who asked to be phoned is
    # not missing an email address, and marking one required would fail every
    # question lead that ever gets sent.
    for f in applicable_fields(lead_type, values):
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
    for f in applicable_fields(lead_type, values):
        value = values.get(f.name) or ""
        if not value:
            continue
        if f.kind == "tel":
            value = normalize_phone(str(value))
        # Belt and braces against a pathological paste in a free-text box.
        out[f.name] = str(value)[:1000]
    return out


def summary_for(lead_type: str, answers: dict[str, Any]) -> list[dict[str, str]]:
    """
    The answers as label/value pairs, in the order they were asked.

    Read back to the customer before anything is sent. Skipped fields are left
    out rather than shown empty — a list with four blanks in it looks like
    something went wrong, when in fact they just had nothing to add.
    """
    rows = []
    for f in applicable_fields(lead_type, answers):
        value = str(answers.get(f.name) or "").strip()
        if value:
            rows.append({"label": f.label, "value": value})
    return rows


def label_for(lead_type: str, name: str) -> str:
    """Human label for a field name, for laying out the email."""
    for f in fields_for(lead_type):
        if f.name == name:
            return f.label
    return name.replace("_", " ").capitalize()
