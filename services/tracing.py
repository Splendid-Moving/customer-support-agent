"""
LangSmith tracing.

A trace is a recording of one turn: which lane the message took, what the guard
decided, what the model was asked and answered, and how long each step took.

For this agent the single most useful thing it buys you is seeing MISROUTES. If
a customer asked a simple FAQ question and got pushed into the estimate form,
the logs show a normal-looking turn; the trace shows the router's reasoning in
one line.

⚠️  PRIVACY
-----------
Tracing uploads run contents to LangSmith's servers. Here that means customer
names, phone numbers, addresses and everything they type into the form. That is
what makes traces useful and exactly why LANGSMITH_TRACING defaults to false.

Adapted from ops-agent/services/tracing.py — same labels, same reasoning.
"""

import logging
import os

from services import config

logger = logging.getLogger(__name__)


# ── Startup ────────────────────────────────────────────────────────────────────

def configure() -> str:
    """
    Check the tracing setup once at boot and say plainly what it will do.

    Every LangSmith misconfiguration fails the same silent way: the app runs
    perfectly and the project stays empty, because the SDK uploads from a
    background thread and swallows its own errors. Checking at boot turns half an
    hour of confusion into one log line.
    """
    if not config.langsmith_tracing():
        summary = "LangSmith tracing: OFF (set LANGSMITH_TRACING=true to enable)"
        logger.info(summary)
        return summary

    if not config.langsmith_api_key():
        summary = (
            "LangSmith tracing: ON but LANGSMITH_API_KEY is empty — no runs will "
            "be uploaded. The SDK fails silently, so this is your only warning."
        )
        logger.warning(summary)
        return summary

    # The SDK reads LANGSMITH_PROJECT itself, but only if it is actually set;
    # config.langsmith_project() has a default the SDK cannot see.
    os.environ.setdefault("LANGSMITH_PROJECT", config.langsmith_project())

    summary = f"LangSmith tracing: ON -> project {config.langsmith_project()!r}"
    logger.info(summary)

    if not config.dry_run():
        logger.warning(
            "LangSmith tracing is ON in LIVE mode — real customer names, phone "
            "numbers and addresses will be uploaded to LangSmith."
        )
    return summary


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("true", "1", "yes")


def status() -> dict:
    """Tracing state for /api/status. Never reports the key itself."""
    return {
        "enabled": config.langsmith_tracing(),
        "project": config.langsmith_project(),
        "api_key_present": bool(config.langsmith_api_key()),
        "hide_inputs": _env_true("LANGSMITH_HIDE_INPUTS"),
        "hide_outputs": _env_true("LANGSMITH_HIDE_OUTPUTS"),
    }


# ── Per-run labelling ──────────────────────────────────────────────────────────

#: Where a run came in from. Kept short and stable — renaming one orphans every
#: historical trace filtered on the old name.
CHANNEL_WEB = "web"
CHANNEL_CLI = "cli"

#: What kind of input started the run. A "resume" is the second half of a form
#: that paused at an interrupt, so its trace is short and starts mid-graph.
#: Without this label those look like broken runs.
TURN_MESSAGE = "message"
TURN_RESUME = "resume"


def run_config(thread_id: str, *, channel: str, turn: str = TURN_MESSAGE, **metadata) -> dict:
    """
    The config dict passed to `graph.invoke()` / `graph.stream()`.

    Carries the thread_id — load-bearing, it is how a paused form is resumed —
    plus the labels that make a list of hundreds of runs searchable. `thread_id`
    is duplicated into metadata on purpose: that is the key LangSmith's Threads
    view groups on, and without it one conversation reads as six unrelated rows.

    Costs nothing when tracing is off.
    """
    backend = config.model_backend()
    mode = "dry_run" if config.dry_run() else "live"

    return {
        "configurable": {"thread_id": thread_id},
        "run_name": f"support-agent:{channel}:{turn}",
        "tags": [f"channel:{channel}", f"turn:{turn}", f"backend:{backend}", mode],
        "metadata": {
            "thread_id": thread_id,
            "channel": channel,
            "turn": turn,
            "dry_run": config.dry_run(),
            "model_backend": backend,
            **metadata,
        },
    }


def as_resume(cfg: dict) -> dict:
    """Relabel a run config as the resume half of a paused form."""
    out = {**cfg, "tags": list(cfg.get("tags", [])), "metadata": {**cfg.get("metadata", {})}}
    out["tags"] = [t for t in out["tags"] if not t.startswith("turn:")] + [f"turn:{TURN_RESUME}"]
    out["metadata"]["turn"] = TURN_RESUME
    out["run_name"] = out.get("run_name", "").replace(f":{TURN_MESSAGE}", f":{TURN_RESUME}")
    return out
