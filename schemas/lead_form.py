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
    """One question on the form."""

    name: str
    label: str
    kind: Literal["text", "tel", "email", "date", "select", "textarea"] = "text"
    required: bool = False
    placeholder: str = ""
    help: str = ""
    options: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["options"] = list(self.options)
        return out


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
    Field_("name", "Your name", required=True, placeholder="Jordan Lee"),
    Field_("phone", "Phone", kind="tel", required=True, placeholder="(323) 555-0142",
           help="How a manager will reach you."),
    Field_("email", "Email", kind="email", required=True, placeholder="you@example.com"),
]

_MOVE = [
    Field_("move_date", "Move date", kind="date",
           help="Leave blank if you haven't settled on one."),
    Field_("home_size", "Size of the place", kind="select", required=True,
           options=HOME_SIZES),
]

_EXTRAS = [
    Field_("access", "Stairs or elevator?", kind="text",
           placeholder="3rd floor walk-up, elevator at the new place",
           help="Access is usually the biggest factor in how long a move takes."),
    Field_("notes", "Anything else we should know?", kind="textarea",
           placeholder="Piano, a safe, packing help needed, parking is tight..."),
]


FORMS: dict[str, dict[str, Any]] = {
    "estimate": {
        "title": "Get an estimate",
        "intro": (
            "Fill this in and a manager will put together a real estimate and get "
            "back to you. Photos help a lot — the more we can see, the closer the "
            "number lands."
        ),
        "fields": [
            *_CONTACT,
            Field_("from_address", "Moving from", required=True,
                   placeholder="Street, city"),
            Field_("to_address", "Moving to", required=True,
                   placeholder="Street, city"),
            *_MOVE,
            *_EXTRAS,
        ],
    },
    "long_distance": {
        "title": "Long-distance move",
        "intro": (
            "Long-distance moves aren't billed hourly like our local jobs, so a "
            "manager prices each one individually. Give us the details and they'll "
            "come back to you directly."
        ),
        "fields": [
            *_CONTACT,
            Field_("from_address", "Moving from", required=True,
                   placeholder="Street, city"),
            Field_("to_address", "Moving to", required=True,
                   placeholder="City and state",
                   help="Where the move is going — city and state is enough."),
            *_MOVE,
            Field_("flexibility", "How firm is that date?", kind="select",
                   options=FLEXIBILITY),
            *_EXTRAS,
        ],
    },
}


def spec(lead_type: str) -> dict[str, Any]:
    """The form definition, in the shape the browser renders."""
    form = FORMS.get(lead_type) or FORMS["estimate"]
    return {
        "lead_type": lead_type if lead_type in FORMS else "estimate",
        "title": form["title"],
        "intro": form["intro"],
        "fields": [f.to_json() for f in form["fields"]],
        "photos": {
            "label": "Photos of what's moving",
            "help": (
                "Optional, but it's the difference between a rough guess and a "
                "real estimate. A shot of each room is plenty."
            ),
        },
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


def validate(lead_type: str, submitted: dict[str, Any]) -> dict[str, str]:
    """
    Check a submission. Returns {field_name: message} — empty means it is good.

    Errors are phrased as a person would say them, because they are shown to the
    customer verbatim under the field they belong to.
    """
    errors: dict[str, str] = {}
    values = {k: (v.strip() if isinstance(v, str) else v) for k, v in (submitted or {}).items()}

    for f in fields_for(lead_type):
        value = values.get(f.name) or ""
        if f.required and not value:
            errors[f.name] = "We need this one."
            continue
        if not value:
            continue

        if f.kind == "email" and not _EMAIL.match(value):
            errors[f.name] = "That doesn't look like an email address."

        elif f.kind == "tel" and len(_digits(value)) not in (10, 11):
            errors[f.name] = "That doesn't look like a full phone number."

        elif f.kind == "date":
            try:
                parsed = datetime.strptime(value, "%Y-%m-%d").date()
            except ValueError:
                errors[f.name] = "Use the date picker, or leave it blank."
            else:
                if parsed < date.today():
                    errors[f.name] = "That date has already passed."

        elif f.kind == "select" and f.options and value not in f.options:
            errors[f.name] = "Pick one of the options."

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
