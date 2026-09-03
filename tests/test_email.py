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


def test_dry_run_sends_nothing(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "true")
    result = email.send_lead("estimate", LEAD)
    assert result["dry_run"] is True


def test_going_live_without_a_key_fails_loudly_rather_than_silently(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setenv("RESEND_API_KEY", "")
    with pytest.raises(email.EmailError, match="RESEND_API_KEY"):
        email.send_lead("estimate", LEAD)
