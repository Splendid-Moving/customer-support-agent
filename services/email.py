"""
The one thing this agent can cause to happen in the world: an email to the office.

Either an estimate a manager prices, or a question the chat could not answer and
a person now has to — same send, same inbox, different subject line.

Sent through Resend, which ghl_calendar_sync already uses, so it is one account
and one API key across the workspace. Resend takes attachments natively, which is
the whole reason it is here rather than a webhook — the customer's photos are the
most useful part of the lead and they need to arrive with it, not as a link
somebody has to click.

TWO DETAILS THAT MATTER MORE THAN THEY LOOK
-------------------------------------------
`reply_to` is the CUSTOMER'S address. A manager opens the lead, hits reply, and
is talking to the customer — no copying an address out of the body, no risk of
mistyping it. It is the difference between a lead worked in ten seconds and one
that waits until someone has time.

The subject line carries the name, the size and the date, because it is read on a
phone in a list of forty other emails. Everything else can be inside.
"""

import base64
import logging

import requests

from schemas import lead_form
from services import config

logger = logging.getLogger(__name__)

RESEND_URL = "https://api.resend.com/emails"
_TIMEOUT = 20


class EmailError(RuntimeError):
    """The send failed. Carries the response body, which Resend uses for detail."""

    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(f"{message} (status {status}): {body}" if status else message)
        self.status = status
        self.body = body


# ── The email itself ───────────────────────────────────────────────────────────

LEAD_LABEL = {
    "estimate": "Estimate request",
    "long_distance": "LONG DISTANCE",
    "question": "Question",
}


def subject_for(lead_type: str, lead: dict[str, str]) -> str:
    """What a manager sees in their inbox list, on a phone."""
    bits = [LEAD_LABEL.get(lead_type, "Lead"), lead.get("name", "no name")]

    if lead_type == "question":
        # How to reach them, in the subject line. A question is answered by
        # somebody picking up the phone, and the number they need is the first
        # thing they would otherwise have to open the email to find.
        if lead.get("contact_method") == "Email" and lead.get("email"):
            bits.append(lead["email"])
        elif lead.get("phone"):
            bits.append(lead["phone"])
        return " — ".join(bits)

    if size := lead.get("home_size"):
        bits.append(size)
    if when := lead.get("move_date"):
        bits.append(when)
    return " — ".join(bits)


def _row(label: str, value: str) -> str:
    return (
        '<tr>'
        f'<td style="padding:6px 14px 6px 0;color:#5b6b7f;font-size:13px;'
        f'white-space:nowrap;vertical-align:top">{label}</td>'
        f'<td style="padding:6px 0;color:#032449;font-size:15px;font-weight:600">'
        f'{_escape(value)}</td>'
        '</tr>'
    )


def _escape(text: str) -> str:
    """
    Every value here was typed by a member of the public into a web form.

    Un-escaped, a name of `<script>` or a stray `<` silently mangles the layout of
    the email a manager is trying to read — and the whole point of this message is
    that it can be read at a glance.
    """
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _reply_hint(lead_type: str, lead: dict[str, str]) -> str:
    """
    The line under the heading telling a manager how to answer this one.

    On a question it is not always "hit reply": someone who asked to be phoned
    left no email address, and `reply_to` is unset on that send — so a manager
    who hits reply is writing to nobody.
    """
    if lead_type == "question" and lead.get("contact_method") == "Phone":
        phone = _escape(lead.get("phone", ""))
        return f"They asked to be phoned{f' on {phone}' if phone else ''}."
    return "Reply to this email to go straight back to the customer."


def render(lead_type: str, lead: dict[str, str], *, photo_count: int = 0) -> str:
    """The lead as HTML, laid out to be read in five seconds."""
    rows = "".join(
        _row(lead_form.label_for(lead_type, name), value)
        for name, value in lead.items()
        if value
    )

    banner = ""
    if lead_type == "long_distance":
        banner = (
            '<div style="background:#032449;color:#fff;padding:10px 16px;'
            'border-radius:8px;font-size:14px;font-weight:600;margin-bottom:18px">'
            "Long-distance move — not an hourly local job, needs individual pricing."
            "</div>"
        )
    elif lead_type == "question":
        how = lead.get("contact_method") or ""
        by = f" Answer by {how.lower()}." if how else ""
        banner = (
            '<div style="background:#032449;color:#fff;padding:10px 16px;'
            'border-radius:8px;font-size:14px;font-weight:600;margin-bottom:18px">'
            "The chat could not answer this — the customer is waiting on us."
            f"{_escape(by)}</div>"
        )

    photos = ""
    if photo_count:
        photos = (
            f'<p style="margin:18px 0 0;color:#5b6b7f;font-size:14px">'
            f'{photo_count} photo{"s" if photo_count != 1 else ""} attached.</p>'
        )

    return f"""\
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
            max-width:560px;margin:0 auto;padding:24px;background:#f6f9ff">
  <div style="background:#fff;border-radius:12px;padding:24px">
    {banner}
    <h2 style="margin:0 0 4px;color:#032449;font-size:19px">
      {_escape(LEAD_LABEL.get(lead_type, "Lead"))} from the website chat
    </h2>
    <p style="margin:0 0 18px;color:#5b6b7f;font-size:14px">
      {_reply_hint(lead_type, lead)}
    </p>
    <table style="border-collapse:collapse;width:100%">{rows}</table>
    {photos}
  </div>
</div>"""


# ── Sending ────────────────────────────────────────────────────────────────────

def send_lead(
    lead_type: str,
    lead: dict[str, str],
    attachments: list[dict] | None = None,
) -> dict:
    """
    Email one lead to the office.

    In dry-run the whole thing is logged and nothing is sent — that is the
    default, so a half-configured deploy is silent rather than noisy.
    """
    attachments = attachments or []
    subject = subject_for(lead_type, lead)
    html = render(lead_type, lead, photo_count=len(attachments))

    if config.dry_run():
        logger.info(
            "[DRY RUN] lead email\n  to: %s\n  reply-to: %s\n  subject: %s\n"
            "  attachments: %d\n%s",
            config.lead_email_to(), lead.get("email", ""), subject, len(attachments), html,
        )
        return {"sent": True, "dry_run": True, "subject": subject}

    if not config.resend_api_key():
        raise EmailError("RESEND_API_KEY is not set — the lead was not sent")
    if not config.lead_email_from():
        raise EmailError("RESEND_FROM is not set — the lead was not sent")

    payload: dict = {
        "from": config.lead_email_from(),
        "to": config.lead_email_to(),
        "subject": subject,
        "html": html,
    }
    # Only if it looks like an address; a malformed reply_to makes Resend reject
    # the entire send, which would lose a lead over a typo in one field.
    if (reply_to := lead.get("email", "")) and "@" in reply_to:
        payload["reply_to"] = reply_to

    if attachments:
        payload["attachments"] = [
            {
                "filename": a["filename"],
                "content": base64.b64encode(a["content"]).decode(),
            }
            for a in attachments
        ]

    response = requests.post(
        RESEND_URL,
        headers={
            "Authorization": f"Bearer {config.resend_api_key()}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=_TIMEOUT,
    )
    if not response.ok:
        raise EmailError("Lead email failed", response.status_code, response.text)

    logger.info("Lead emailed to %s: %s", config.lead_email_to(), subject)
    return {"sent": True, "dry_run": False, "subject": subject, "response": response.json()}
