#!/usr/bin/env python
"""
The web app. Serves the chat page, the conversation, and the photo uploads.

    local:      python server.py          -> http://localhost:8080
    production: uvicorn app:app --host 0.0.0.0 --port $PORT

THE ONE THING TO GET RIGHT HERE
-------------------------------
The resume protocol. When the graph is paused at the estimate form, input must
arrive as `Command(resume=...)`, not as a new message. Send the wrong one and
LangGraph silently starts the graph over: the form the customer just filled in is
discarded, the conversation restarts, and nothing errors. It is the single
easiest way to break this app, so the decision is made in exactly one place —
`_graph_input` — and the browser never has to know which mode it is in.
"""

import json
import logging
import sqlite3
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from agent.graph import build_graph
from services import config, knowledge, tracing, uploads

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Splendid Moving — chat")

# Before the graph is built, so a bad LangSmith key is the first thing in the log
# rather than a silently empty project.
tracing.configure()


# ── The graph ──────────────────────────────────────────────────────────────────

#: Module-level so the SQLite connection is never garbage collected. If the only
#: reference lives in a local variable it is closed the moment the function
#: returns — underneath a perfectly live checkpointer — and the failure surfaces
#: much later as "Cannot operate on a closed database".
_conn = None
_graph = None


def get_graph():
    """
    The compiled graph, built on first use.

    A half-filled form must survive a restart, and Railway restarts on every
    deploy, so the checkpointer is SQLite on disk rather than in memory. Point
    CHECKPOINT_DB at a mounted volume in production or each deploy throws away
    every form someone is midway through.

    `check_same_thread=False` is required because requests are served from a
    thread pool; `timeout` makes concurrent writes wait rather than immediately
    raising "database is locked".
    """
    global _conn, _graph
    if _graph is None:
        path = config.checkpoint_db()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            # A CHECKPOINT_DB pointing at a volume that was never mounted. Better
            # a local file that loses forms on deploy than an app that will not
            # boot at all.
            path = config.PROJECT_ROOT / ".support_threads.sqlite"
            logger.warning("CHECKPOINT_DB unusable; falling back to %s", path)

        logger.info("Checkpoints: %s", path)
        _conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        saver = SqliteSaver(_conn)
        saver.setup()
        _graph = build_graph(checkpointer=saver)
    return _graph


# ── Abuse limiting ─────────────────────────────────────────────────────────────

_hits: dict[str, deque] = defaultdict(deque)


def _rate_limited(request: Request) -> bool:
    """
    A crude per-IP limit, held in memory.

    In memory means it resets on deploy and does not span replicas. That is fine
    for what it defends against — one person with a script, not a distributed
    attack — and a Redis dependency for this would be more moving parts than the
    problem deserves. Revisit it if this ever runs on more than one instance.
    """
    ip = (request.client.host if request.client else "") or "unknown"
    now = time.monotonic()
    window = _hits[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= config.RATE_LIMIT_PER_MINUTE:
        return True
    window.append(now)
    return False


# ── Request bodies ─────────────────────────────────────────────────────────────

class Turn(BaseModel):
    """
    One input from the browser.

    Exactly one of these is set. `message` is a normal chat turn; `form` is the
    estimate form coming back; `cancelled` is the customer closing it.
    """

    thread_id: str = ""
    message: str = ""
    form: dict | None = None
    cancelled: bool = False


def _pending_interrupt(graph, cfg: dict):
    """
    The interrupt payload if the graph is paused, else None.

    Paused-ness is read off the TASKS, not off `snapshot.next`. That distinction
    cost an afternoon: when the form is re-shown with validation errors, the node
    has interrupted a second time within the same invocation, and `next` comes
    back EMPTY even though the graph is very much stopped. Trusting `next` there
    makes the server treat the customer's corrected form as a brand-new chat
    message — which restarts the graph, throws the form away, and reports no
    error at all. `tasks` is correct in both cases.
    """
    state = graph.get_state(cfg)
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def _graph_input(graph, cfg: dict, turn: Turn):
    """
    Translate a browser request into the right kind of graph input.

    THIS is the resume protocol, in one place. If the graph is paused, whatever
    arrives is an answer to the interrupt and must be wrapped in a Command. If it
    is not paused, it is a new message.
    """
    if _pending_interrupt(graph, cfg) is None:
        return {"messages": [HumanMessage(content=turn.message)]}, cfg

    resume_cfg = tracing.as_resume(cfg)
    if turn.cancelled:
        return Command(resume={"cancelled": True}), resume_cfg
    if turn.form is not None:
        return Command(resume=turn.form), resume_cfg
    # They typed in the chat box while the form was open. collect_lead treats a
    # non-dict as "show it again" rather than losing what they had entered.
    return Command(resume=turn.message), resume_cfg


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    """Railway's health check. Deliberately reveals no configuration."""
    return {"status": "ok"}


@app.get("/api/status")
def status():
    return {
        "dry_run": config.dry_run(),
        "backend": config.model_backend(),
        "knowledge": knowledge.describe(),
        "tracing": tracing.status(),
    }


@app.post("/api/reset")
def reset():
    """Start a fresh conversation. The old thread stays in the database."""
    return {"thread_id": str(uuid.uuid4())}


@app.post("/api/upload")
async def upload(
    request: Request,
    thread_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    One photo. Called before the form is submitted; returns an id, not a URL.

    Nothing here is served back out. The photos exist to be attached to one
    email, so there is no endpoint that reads them — which means no way for
    anyone to fish for other customers' uploads.
    """
    if _rate_limited(request):
        return JSONResponse({"error": "Slow down a moment and try again."}, status_code=429)

    data = await file.read()
    try:
        saved = uploads.save(thread_id, data)
    except uploads.UploadRejected as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    return {"id": saved["id"], "bytes": int(saved["bytes"])}


@app.post("/api/chat")
async def chat(request: Request, turn: Turn):
    """
    One turn, streamed back as server-sent events.

    Streaming is not decoration. The knowledge lane can take a few seconds, and a
    customer watching a dead box assumes it is broken and leaves.
    """
    if _rate_limited(request):
        return JSONResponse(
            {"error": "Slow down a moment and try again."}, status_code=429
        )

    thread_id = turn.thread_id or str(uuid.uuid4())
    graph = get_graph()
    cfg = tracing.run_config(thread_id, channel=tracing.CHANNEL_WEB)

    def event(kind: str, data: dict) -> str:
        return f"data: {json.dumps({'event': kind, **data})}\n\n"

    def run():
        yield event("start", {"thread_id": thread_id})
        try:
            graph_input, run_cfg = _graph_input(graph, cfg, turn)

            final: dict = {}
            for chunk in graph.stream(graph_input, run_cfg, stream_mode="values"):
                final = chunk

            # A pause is read back off the state rather than out of the stream:
            # `values` does not carry __interrupt__, and the state is the
            # authority on whether this run actually stopped at the form.
            if (pending := _pending_interrupt(graph, cfg)) is not None:
                yield event("form", {"paused": True, "form": pending})
                return

            messages = final.get("messages") or []
            reply = messages[-1].content if messages else ""
            yield event(
                "done",
                {
                    "paused": False,
                    "reply": reply or "Sorry, I lost that one — say it again?",
                    "intent": final.get("intent", ""),
                },
            )

        except Exception as exc:
            # The customer gets a human sentence with a phone number; the stack
            # trace goes to the log, where someone can actually act on it.
            logger.exception("Chat turn failed (thread %s)", thread_id)
            yield event(
                "done",
                {
                    "paused": False,
                    "error": f"{type(exc).__name__}",
                    "reply": (
                        "Something went wrong on my end — sorry. Give the office a "
                        f"call on {config.COMPANY_PHONE} and they'll sort you out."
                    ),
                },
            )

    return StreamingResponse(
        run(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
