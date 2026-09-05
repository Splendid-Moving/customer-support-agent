"""
The lead email.

Rendering is tested rather than sending: what a manager sees at 7am decides
whether the lead gets worked, and it is the part that changes.
"""

import pytest

from schemas import lead_form
from services import email

LEAD = {
    "name": "Jordan Lee",
    "phone": "(323) 555-0142",
    "email": "jordan@example.com",
    "from_address": "1200 Sunset Blvd",
    "to_address": "88 Ocean Ave",
    "home_size": "2 bedrooms",
    "move_date": "2026-10-14",
}


def test_the_subject_line_is_readable_on_a_phone():
    subject = email.subject_for("estimate", LEAD)
    assert "Jordan Lee" in subject
    assert "2 bedrooms" in subject
    assert "2026-10-14" in subject


def test_long_distance_is_obvious_from_the_subject_alone():
    assert "LONG DISTANCE" in email.subject_for("long_distance", LEAD)


def test_a_lead_with_only_the_required_fields_still_has_a_subject():
    assert email.subject_for("estimate", {"name": "Sam"}).strip()


def test_every_answer_appears_in_the_email():
    html = email.render("estimate", LEAD)
    for value in LEAD.values():
        assert value in html


def test_fields_are_labelled_the_way_the_form_asked_them():
    html = email.render("estimate", LEAD)
    assert lead_form.label_for("estimate", "from_address") in html


def test_long_distance_carries_a_banner_so_nobody_quotes_it_hourly():
    html = email.render("long_distance", LEAD)
    assert "not an hourly local job" in html


def test_the_photo_count_is_stated():
    assert "3 photos attached" in email.render("estimate", LEAD, photo_count=3)
    assert "1 photo attached" in email.render("estimate", LEAD, photo_count=1)
    assert "attached" not in email.render("estimate", LEAD, photo_count=0)


@pytest.mark.parametrize(
    "hostile",
    ['<script>alert(1)</script>', 'Bob & "Sons" <bob>', "a > b < c"],
)
def test_customer_typed_html_cannot_break_the_layout(hostile):
    """Every value here was typed into a public web form."""
    html = email.render("estimate", {**LEAD, "notes": hostile})
    assert hostile not in html
    assert "&lt;" in html or "&amp;" in html


# ── Questions ──────────────────────────────────────────────────────────────────
# A question is answered by a person picking up the phone, so what the email has
# to carry is different: the question itself, and how to reach whoever asked it.

ASKED = {
    "question": "How much extra to haul away a fridge.",
    "name": "Nick",
    "contact_method": "Phone",
    "phone": "(818) 505-4576",
}


def test_a_question_says_so_and_carries_the_number_in_the_subject():
    subject = email.subject_for("question", ASKED)
    assert subject.startswith("Question")
    assert "Nick" in subject
    assert "(818) 505-4576" in subject, "the number is what they need to act on it"


def test_an_emailed_question_carries_the_address_instead():
    asked = {"question": "What does storage cost?", "name": "Ana",
             "contact_method": "Email", "email": "ana@example.com"}
    assert "ana@example.com" in email.subject_for("question", asked)


def test_the_question_and_anything_else_they_asked_both_appear():
    html = email.render("question", {**ASKED, "anything_else": "Do you sell boxes?"})
    assert "How much extra to haul away a fridge." in html
    assert "Do you sell boxes?" in html


def test_a_question_is_marked_as_something_we_could_not_answer():
    assert "could not answer" in email.render("question", ASKED)


def test_nobody_is_told_to_hit_reply_on_a_lead_with_no_address():
    """
    `reply_to` is only set when there is an email address. On a question from
    someone who asked to be phoned there isn't one — so "reply to this email"
    is an instruction to write to nobody.
    """
    phoned = email.render("question", ASKED)
    assert "Reply to this email" not in phoned
    assert "asked to be phoned" in phoned

    emailed = email.render("question", {"question": "q", "name": "Ana",
                                        "contact_method": "Email",
                                        "email": "ana@example.com"})
    assert "Reply to this email" in emailed


def test_a_question_typed_as_html_still_cannot_break_the_layout():
    html = email.render("question", {**ASKED, "question": "<b>how much?</b>"})
    assert "<b>how much?</b>" not in html
    assert "&lt;b&gt;" in html


def test_dry_run_sends_nothing(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "true")
    result = email.send_lead("estimate", LEAD)
    assert result["dry_run"] is True


def test_going_live_without_a_key_fails_loudly_rather_than_silently(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setenv("RESEND_API_KEY", "")
    with pytest.raises(email.EmailError, match="RESEND_API_KEY"):
        email.send_lead("estimate", LEAD)
