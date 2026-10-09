"""
Answers the way people SAY them.

Voice mode sends the interview whatever the transcriber heard: "Nik.", "Two
bedroom.", "next Friday", "nik at gmail dot com". Every one of those is an answer
a person at the office would accept, and every one used to be rejected — or
worse, accepted with the full stop still on the name. `lead_form.understand`
turns them into what the form stores, and typed answers benefit just the same.

It only ever tidies or recognises. Anything it cannot read confidently it hands
back unchanged, so validation still decides and still asks again.
"""

from datetime import date

import pytest

from schemas import lead_form

# A Wednesday, so "next Friday" and "Friday" have one right answer each.
TODAY = date(2026, 10, 7)


def field(name: str, lead_type: str = "estimate") -> lead_form.Field_:
    return next(f for f in lead_form.fields_for(lead_type) if f.name == name)


def understood(name: str, value: str, lead_type: str = "estimate") -> str:
    f = field(name, lead_type)
    return lead_form.coerce_option(f, lead_form.understand(f, value, today=TODAY))


# ── The transcriber's full stop ──────────────────────────────────────────────

@pytest.mark.parametrize("said, stored", [
    ("Nik.", "Nik"),
    ("Jordan Lee.", "Jordan Lee"),
    ("  Maria!  ", "Maria"),
])
def test_a_spoken_name_loses_its_full_stop(said, stored):
    assert understood("name", said) == stored


def test_a_question_mark_is_kept_because_it_means_they_asked_something():
    assert understood("name", "wait how much is it?") == "wait how much is it?"


def test_free_text_answers_are_left_exactly_as_said():
    assert understood("notes", "There's a piano. And a safe.") == "There's a piano. And a safe."


# ── Phone numbers and zips, said out loud ────────────────────────────────────

@pytest.mark.parametrize("said", [
    "323 645 2636.",
    "three two three six four five two six three six",
    "(323) 645-2636",
])
def test_a_spoken_phone_number_validates(said):
    value = understood("phone", said)
    assert lead_form.validate_one(field("phone"), value) is None, value


@pytest.mark.parametrize("said", ["90026.", "9 0 0 2 6", "nine oh oh two six", "90026"])
def test_a_spoken_zip_is_five_digits(said):
    assert understood("from_zip", said) == "90026"


def test_a_zip_with_words_in_it_is_not_guessed_at():
    assert understood("from_zip", "I'm in Echo Park") == "I'm in Echo Park"


# ── Email addresses, said out loud ───────────────────────────────────────────

@pytest.mark.parametrize("said, stored", [
    ("nik at gmail dot com", "nik@gmail.com"),
    ("Nik@Gmail.com.", "nik@gmail.com"),
    ("jordan dot lee at example dot co dot uk", "jordan.lee@example.co.uk"),
    ("jordan@example.com", "jordan@example.com"),
])
def test_a_spoken_email_becomes_an_address(said, stored):
    assert understood("email", said) == stored
    assert lead_form.validate_one(field("email"), stored) is None


def test_an_email_that_cannot_be_read_still_fails_validation():
    value = understood("email", "my usual one")
    assert lead_form.validate_one(field("email"), value) is not None


# ── Dates, said the way people say them ──────────────────────────────────────

@pytest.mark.parametrize("said, expected", [
    # "Next Friday" is next WEEK's Friday, the way a calendar app reads it; "this
    # Friday" and a bare "Friday" are the coming one. Either way the read-back
    # shows the date before anything is sent.
    ("next Friday", date(2026, 10, 16)),
    ("Next, Friday.", date(2026, 10, 16)),   # as the transcriber actually wrote it
    ("this Friday", date(2026, 10, 9)),
    ("Friday.", date(2026, 10, 9)),
    ("tomorrow", date(2026, 10, 8)),
    ("November 14th", date(2026, 11, 14)),
    ("on the 20th", date(2026, 10, 20)),
    ("the 20th of October", date(2026, 10, 20)),
    ("10/20", date(2026, 10, 20)),
    ("2026-12-01", date(2026, 12, 1)),
    ("el 20 de octubre", date(2026, 10, 20)),
    ("в пятницу", date(2026, 10, 9)),
])
def test_a_spoken_date_is_understood(said, expected):
    assert understood("move_date", said) == expected.isoformat()


def test_a_month_already_gone_this_year_means_next_year():
    assert understood("move_date", "March 3rd") == "2027-03-03"


@pytest.mark.parametrize("said", [
    "whenever the lease ends",
    "next week",        # a week is not a day; storing one would be our guess
    "end of the month",
    "3pm",              # a time, not a date
    "20.10",            # could be either; asking beats guessing
])
def test_an_unreadable_date_is_left_for_validation_to_reject(said):
    value = understood("move_date", said)
    assert lead_form.validate_one(field("move_date"), value) is not None, value


def test_not_sure_is_still_not_sure():
    assert lead_form.is_undecided("Not sure yet.")


# ── Multiple choice, said rather than tapped ─────────────────────────────────

@pytest.mark.parametrize("said, option", [
    ("Two bedroom.", "2 bedrooms"),
    ("it's a two bedroom apartment", "2 bedrooms"),
    ("one bedroom", "1 bedroom"),
    ("A studio.", "Studio"),
    ("four bedrooms", "4+ bedrooms"),
    ("five bedroom house", "4+ bedrooms"),
    ("an office", "Office / commercial"),
    ("just a few items", "Just a few items"),
    ("2 bedrooms", "2 bedrooms"),
])
def test_a_spoken_home_size_matches_its_option(said, option):
    assert understood("home_size", said) == option


@pytest.mark.parametrize("said, option", [
    ("Call me.", "Phone"),
    ("phone please", "Phone"),
    ("email is better", "Email"),
    ("Email.", "Email"),
])
def test_a_spoken_contact_preference_matches_its_option(said, option):
    assert understood("contact_method", said, "question") == option


@pytest.mark.parametrize("said, option", [
    ("a few days either way", "Within a few days"),
    ("that exact date", "That exact date"),
    ("not decided", "Not decided yet"),
])
def test_a_spoken_flexibility_matches_its_option(said, option):
    assert understood("flexibility", said, "long_distance") == option


def test_two_possible_options_is_not_a_choice():
    """Asking again beats guessing which of two places a manager should price."""
    said = "two or three bedrooms"
    assert understood("home_size", said) == said
    assert lead_form.validate_one(field("home_size"), said) is not None
