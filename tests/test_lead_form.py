"""
Form validation. This is the last line before a lead reaches a manager's inbox.

The browser validates too, but that check is a convenience — anyone can post
straight to the endpoint — so everything that matters is asserted here.
"""

from datetime import date, timedelta

import pytest

from schemas import lead_form

GOOD = {
    "name": "Jordan Lee",
    "phone": "(323) 555-0142",
    "email": "jordan@example.com",
    "from_address": "1200 Sunset Blvd, Los Angeles",
    "to_address": "88 Ocean Ave, Santa Monica",
    "home_size": "2 bedrooms",
}


def test_a_complete_form_passes():
    assert lead_form.validate("estimate", GOOD) == {}


def test_every_required_field_is_actually_required():
    for name in ("name", "phone", "email", "from_address", "to_address", "home_size"):
        missing = {k: v for k, v in GOOD.items() if k != name}
        assert name in lead_form.validate("estimate", missing), f"{name} was not required"


def test_whitespace_only_does_not_count_as_filled_in():
    assert "name" in lead_form.validate("estimate", {**GOOD, "name": "   "})


@pytest.mark.parametrize("bad", ["nope", "a@b", "@example.com", "jordan@", "jordan example.com"])
def test_bad_emails_are_rejected(bad):
    assert "email" in lead_form.validate("estimate", {**GOOD, "email": bad})


@pytest.mark.parametrize("good", ["a@b.co", "jordan.lee+moving@example.co.uk"])
def test_real_emails_are_accepted(good):
    assert "email" not in lead_form.validate("estimate", {**GOOD, "email": good})


@pytest.mark.parametrize("bad", ["555", "12345", "not a phone"])
def test_short_phone_numbers_are_rejected(bad):
    assert "phone" in lead_form.validate("estimate", {**GOOD, "phone": bad})


@pytest.mark.parametrize(
    "given,expected",
    [
        ("3236452636", "(323) 645-2636"),
        ("+1 323-645-2636", "(323) 645-2636"),
        ("(323) 645 2636", "(323) 645-2636"),
        ("1 (323) 645.2636", "(323) 645-2636"),
    ],
)
def test_phone_numbers_are_normalised_for_whoever_dials_them(given, expected):
    assert lead_form.normalize_phone(given) == expected


def test_an_unrecognised_phone_format_is_left_alone_rather_than_mangled():
    """A UK or made-up number should reach a manager as typed, not half-formatted."""
    assert lead_form.normalize_phone("+44 20 7946 0958") == "+44 20 7946 0958"


def test_a_date_in_the_past_is_rejected():
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    assert "move_date" in lead_form.validate("estimate", {**GOOD, "move_date": yesterday})


def test_a_future_date_is_fine_and_a_blank_one_is_too():
    soon = (date.today() + timedelta(days=14)).isoformat()
    assert lead_form.validate("estimate", {**GOOD, "move_date": soon}) == {}
    assert lead_form.validate("estimate", {**GOOD, "move_date": ""}) == {}


def test_a_made_up_dropdown_value_is_rejected():
    assert "home_size" in lead_form.validate("estimate", {**GOOD, "home_size": "mansion"})


# ── clean() ────────────────────────────────────────────────────────────────────

def test_unknown_fields_are_dropped_rather_than_forwarded():
    """The submission is whatever someone chose to POST, and it ends up in an email."""
    cleaned = lead_form.clean("estimate", {**GOOD, "admin": "true", "internal_note": "x"})
    assert "admin" not in cleaned and "internal_note" not in cleaned


def test_clean_normalises_the_phone_and_trims():
    cleaned = lead_form.clean("estimate", {**GOOD, "phone": " 3236452636 ", "name": "  Jordan  "})
    assert cleaned["phone"] == "(323) 645-2636"
    assert cleaned["name"] == "Jordan"


def test_a_pathological_paste_is_truncated():
    cleaned = lead_form.clean("estimate", {**GOOD, "notes": "x" * 50_000})
    assert len(cleaned["notes"]) == 1000


def test_empty_optional_fields_do_not_reach_the_email():
    cleaned = lead_form.clean("estimate", {**GOOD, "notes": "", "access": ""})
    assert "notes" not in cleaned and "access" not in cleaned


# ── The two variants ───────────────────────────────────────────────────────────

def test_long_distance_asks_how_firm_the_date_is_and_estimate_does_not():
    ld = [f["name"] for f in lead_form.spec("long_distance")["fields"]]
    local = [f["name"] for f in lead_form.spec("estimate")["fields"]]
    assert "flexibility" in ld
    assert "flexibility" not in local


def test_an_unknown_lead_type_falls_back_to_the_estimate_form():
    assert lead_form.spec("nonsense")["lead_type"] == "estimate"


def test_the_spec_is_json_serialisable():
    """It is sent to the browser as JSON; a dataclass leaking through breaks the form."""
    import json
    json.dumps(lead_form.spec("long_distance"))


# ── Wording ────────────────────────────────────────────────────────────────────

def test_the_first_question_does_not_greet_again():
    """
    The lane's opening line has already said hello one message earlier. Two
    welcomes back to back — "Happy to get that sorted." / "Happy to help with
    that." — is the most obviously robotic thing a chat agent can do, and it
    shipped.
    """
    first = lead_form.fields_for("estimate")[0]
    assert first.name == "name"
    greetings = ("happy to", "hello", "hi ", "thanks for", "great!")
    lowered = first.ask.lower()
    for greeting in greetings:
        assert greeting not in lowered, f"first question opens with {greeting!r}"


def test_every_field_has_something_to_say():
    """A missing `ask` silently falls back to 'What's your <label>?'."""
    for lead_type in ("estimate", "long_distance"):
        for f in lead_form.fields_for(lead_type):
            assert f.ask.strip(), f"{lead_type}/{f.name} has no question written"
