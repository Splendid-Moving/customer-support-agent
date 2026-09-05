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


# ── The question form, and its one branch ──────────────────────────────────────
# It exists because of a conversation that dead-ended: the agent said it would
# have the office get back to someone, took their name and number, and then told
# them to call us. These are the checks that the details it takes are usable.

ASKED_PHONE = {
    "question": "How much extra to haul away a fridge.",
    "name": "Nick",
    "contact_method": "Phone",
    "phone": "(818) 505-4576",
}

ASKED_EMAIL = {
    "question": "Whether you can take a piano down two flights.",
    "name": "Ana Ruiz",
    "contact_method": "Email",
    "email": "ana@example.com",
}


@pytest.mark.parametrize("asked", [ASKED_PHONE, ASKED_EMAIL])
def test_either_way_of_being_contacted_is_a_complete_question(asked):
    assert lead_form.validate("question", asked) == {}


def test_the_branch_they_did_not_pick_is_not_missing():
    """
    Both contact fields are required, and only one of them is ever asked. If the
    unasked one still counted as required, every question lead would fail
    validation on the way out — and the customer would be told the send failed
    for a field they were never shown.
    """
    assert "email" not in lead_form.validate("question", ASKED_PHONE)
    assert "phone" not in lead_form.validate("question", ASKED_EMAIL)


def test_the_branch_they_did_pick_is_still_required():
    without = {k: v for k, v in ASKED_PHONE.items() if k != "phone"}
    assert "phone" in lead_form.validate("question", without)


def test_a_question_needs_a_question_and_a_name():
    for name in ("question", "name", "contact_method"):
        missing = {k: v for k, v in ASKED_PHONE.items() if k != name}
        assert name in lead_form.validate("question", missing), f"{name} was not required"


def test_an_answer_from_the_road_not_taken_never_reaches_the_office():
    """
    Posted directly, or left over from a form that was restarted. A manager
    seeing both a number and an address does not know which one the customer
    actually asked to be reached on.
    """
    cleaned = lead_form.clean("question", {**ASKED_PHONE, "email": "stale@example.com"})
    assert "email" not in cleaned
    assert cleaned["phone"] == "(818) 505-4576"

    rows = [r["label"] for r in lead_form.summary_for("question", {**ASKED_PHONE,
                                                                  "email": "x@y.co"})]
    assert "Email" not in rows


def test_the_progress_bar_counts_an_open_branch_once():
    """
    Someone who has not yet said phone-or-email will be asked for exactly ONE of
    them. Counting both shows a form a step longer than it is, and a progress
    bar that jumps backwards is worse than none.
    """
    assert lead_form.steps_remaining("question", {}) == 5
    assert lead_form.steps_remaining("question", {"contact_method": "Phone"}) == 4
    assert lead_form.steps_remaining("question", ASKED_PHONE) == 1  # anything_else


def test_a_question_never_asks_for_photos():
    """There is nothing to photograph, and a step whose only answer is 'skip'."""
    assert lead_form.wants_photos("question") is False
    assert lead_form.wants_photos("estimate") is True
    assert lead_form.wants_photos("long_distance") is True


def test_the_question_spec_is_json_serialisable():
    """`only_if` is a tuple, and a tuple that reaches the browser un-encoded is a
    form that never renders."""
    import json
    json.dumps(lead_form.spec("question"))


# ── Typed answers to a multiple-choice question ────────────────────────────────

@pytest.mark.parametrize(
    "typed,expected",
    [
        ("Phone", "Phone"),
        ("phone", "Phone"),
        ("email", "Email"),
        ("  EMAIL ", "Email"),
        ("by email please", "Email"),
        ("call me on the phone", "Phone"),
    ],
)
def test_typing_the_option_counts_as_tapping_it(typed, expected):
    """The buttons are there to be tapped, but the message box never goes away."""
    field = {f.name: f for f in lead_form.fields_for("question")}["contact_method"]
    assert lead_form.coerce_option(field, typed) == expected


@pytest.mark.parametrize("typed", ["whichever", "phone or email", "yes"])
def test_an_ambiguous_answer_is_left_to_be_asked_again(typed):
    field = {f.name: f for f in lead_form.fields_for("question")}["contact_method"]
    assert lead_form.coerce_option(field, typed) == typed


def test_a_size_can_be_typed_too():
    field = {f.name: f for f in lead_form.fields_for("estimate")}["home_size"]
    assert lead_form.coerce_option(field, "2 bed") == "2 bedrooms"
    assert lead_form.coerce_option(field, "studio") == "Studio"


# ── "no" is not a piece of information ─────────────────────────────────────────

@pytest.mark.parametrize(
    "said", ["no", "No.", "nope", "nothing else", "that's it", "all good", "not really"]
)
def test_declining_the_last_question_is_not_recorded_as_an_answer(said):
    """A manager reading 'Also asked — no' is reading a row looking for a point."""
    assert lead_form.is_nothing(said)


@pytest.mark.parametrize("said", ["no elevator at the new place", "nothing fragile, but a piano"])
def test_an_answer_that_merely_starts_with_no_is_kept(said):
    assert not lead_form.is_nothing(said)


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
    for lead_type in ("estimate", "long_distance", "question"):
        for f in lead_form.fields_for(lead_type):
            assert f.ask.strip(), f"{lead_type}/{f.name} has no question written"


# ── The read-back before anything is sent ──────────────────────────────────────

def test_the_summary_reads_back_in_the_order_it_was_asked():
    rows = lead_form.summary_for("estimate", {**GOOD, "notes": "one piano"})
    labels = [r["label"] for r in rows]
    assert labels[0] == lead_form.fields_for("estimate")[0].label
    assert {"label": "Anything else we should know?", "value": "one piano"} in rows


def test_skipped_answers_are_left_out_of_the_read_back():
    """Four blank rows look like something broke, not like nothing was added."""
    rows = lead_form.summary_for("estimate", GOOD)
    assert all(r["value"] for r in rows)
    assert "Anything else we should know?" not in [r["label"] for r in rows]


def test_the_read_back_shows_what_the_office_will_actually_see():
    """
    It reads back the CLEANED answers, not the raw ones — so the phone number
    the customer checks is the reformatted one a manager will dial.
    """
    cleaned = lead_form.clean("estimate", {**GOOD, "phone": " 3236452636 "})
    rows = {r["label"]: r["value"] for r in lead_form.summary_for("estimate", cleaned)}
    assert rows["Phone"] == "(323) 645-2636"


def test_the_read_back_is_json_serialisable():
    import json
    json.dumps(lead_form.summary_for("long_distance", GOOD))
