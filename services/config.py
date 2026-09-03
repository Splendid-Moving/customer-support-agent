"""
Central config. Every env var this app reads is declared here and nowhere else.

Values are read lazily through functions rather than captured at import time, so
that `langgraph dev` hot-reload and the tests can change the environment without
restarting the process.

Copied in shape from ops-agent/services/config.py deliberately — same habits,
same safety defaults, so anyone who has read one can read the other. The content
is different because this agent talks to CUSTOMERS, not staff: it has no CRM
credentials, no calendar access and no ability to book anything. The only thing
it can cause to happen in the world is one email to the office.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root regardless of where python was invoked from.
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

PROJECT_ROOT = _ROOT


# ── The business ───────────────────────────────────────────────────────────────

COMPANY_NAME = "Splendid Moving"
COMPANY_PHONE = "(323) 645-2636"
COMPANY_EMAIL = "info@splendidmoving.com"
TIMEZONE = "America/Los_Angeles"


# ── Models ─────────────────────────────────────────────────────────────────────

def model_backend() -> str:
    """'openai' or 'openrouter'. See agent/models.py."""
    return os.getenv("MODEL_BACKEND", "openai").strip().lower()


# ── Knowledge base ─────────────────────────────────────────────────────────────

def knowledge_dir() -> Path:
    """
    Directory of markdown files the agent is allowed to answer from.

    Overridable so tests can point at a fixture directory instead of the real
    knowledge base.
    """
    return Path(os.getenv("KNOWLEDGE_DIR", str(_ROOT / "knowledge")))


# ── Lead email ─────────────────────────────────────────────────────────────────
# Estimate requests and long-distance enquiries are emailed to the office, and a
# human contacts the customer. This app never replies to the customer by email.

def resend_api_key() -> str:
    return os.getenv("RESEND_API_KEY", "").strip()


def lead_email_from() -> str:
    """
    Sender address. MUST be on a domain verified in Resend, or every send 403s.
    Same variable name as ghl_calendar_sync so one Resend account serves both.
    """
    return os.getenv("RESEND_FROM", "").strip()


def lead_email_to() -> list[str]:
    """Who gets the lead. Comma-separated for more than one manager."""
    raw = os.getenv("LEAD_NOTIFY_EMAIL", COMPANY_EMAIL)
    return [e.strip() for e in raw.split(",") if e.strip()]


# ── Uploads ────────────────────────────────────────────────────────────────────
# Photos of the customer's items. Held on disk between the form being opened and
# the lead being emailed, then deleted.

def checkpoint_db() -> Path:
    """
    Where a paused estimate form is kept.

    On Railway this MUST point at a mounted volume. The default filesystem is
    wiped on every deploy, so without one, anyone midway through the form at the
    moment you ship loses it — and they find out by clicking Submit.
    """
    return Path(os.getenv("CHECKPOINT_DB", str(_ROOT / ".support_threads.sqlite")))


def upload_dir() -> Path:
    return Path(os.getenv("UPLOAD_DIR", str(_ROOT / ".uploads")))


#: Per photo. Phone cameras produce 3-8 MB files, so this is generous but finite.
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

#: Per lead. Enough to show every room; few enough to fit in one email.
MAX_UPLOADS_PER_THREAD = 12

#: Only these actually reach the office inbox. Checked by sniffing the file's
#: leading bytes, never by trusting the filename or the browser's content-type.
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic"}


# ── Abuse limits ───────────────────────────────────────────────────────────────
# This agent sits on the public internet behind a link anyone can share. Without
# these, one bored person with a script is an unbounded API bill.

#: Longest single message accepted. Real questions are a sentence or two; a
#: 20,000-character message is someone pasting an attack, not a customer.
MAX_MESSAGE_CHARS = 2000

#: Turns allowed in one conversation before it is closed out with a phone number.
MAX_TURNS_PER_THREAD = 60

#: Requests per IP per minute — chat turns and photo uploads share the budget.
#:
#: Was 12, which was fine when the estimate was one modal costing two requests.
#: The interview costs one request PER QUESTION, and each photo is another, so a
#: customer answering briskly and attaching six pictures now spends well over
#: twenty inside a minute. At 12 they hit the limit halfway through giving us
#: their details and got told to slow down, which is the worst possible moment
#: to interrupt someone who is in the middle of handing you a job.
#:
#: 40 still stops a script dead — an actual abuser makes hundreds — and the real
#: ceiling on cost is MAX_TURNS_PER_THREAD, which bounds one conversation
#: regardless of how fast it is driven.
RATE_LIMIT_PER_MINUTE = 40


# ── Safety ─────────────────────────────────────────────────────────────────────

def dry_run() -> bool:
    """
    True  -> the lead email is logged in full and never sent.
    False -> LIVE. Real email lands in the office inbox.

    Defaults to True. A missing or malformed value must never mean "go live".
    """
    return os.getenv("DRY_RUN", "true").strip().lower() not in ("false", "0", "no")


# ── Observability ──────────────────────────────────────────────────────────────

def langsmith_tracing() -> bool:
    """
    Whether runs are uploaded to LangSmith.

    Off by default. Traces carry the full contents of every run — for this agent
    that is customer names, phone numbers, addresses and the photos of their
    belongings — so uploading them is an opt-in decision, not a default.
    """
    return os.getenv("LANGSMITH_TRACING", "false").strip().lower() in ("true", "1", "yes")


def langsmith_project() -> str:
    return os.getenv("LANGSMITH_PROJECT", "splendid-support-agent").strip()


def langsmith_api_key() -> str:
    return os.getenv("LANGSMITH_API_KEY", "").strip()
