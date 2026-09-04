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
import warnings
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

# Streaming with stream_mode="messages" makes LangChain serialise every chunk of
# every model call, including the structured-output ones in the guard, router and
# prefill — whose `parsed` field pydantic then warns about, several times per
# turn. The warning is about a field we never read, on chunks we filter out by
# node name anyway. Left unfiltered it is most of what ends up in the Railway
# log, which is the same as having no log.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Splendid Moving — chat")


#: Generous ceiling on any request body. The guard enforces MAX_MESSAGE_CHARS on
#: chat messages, but the guard does not run while the estimate interview is
#: paused — a Command(resume=...) re-enters the paused node directly. So an
#: answer to "what's your name?" had no size limit at all, and a huge one would
#: be written into the checkpoint and re-serialised on every step after it.
#: Checked from Content-Length so an oversized body is refused before it is read.
MAX_BODY_BYTES = 256 * 1024


@app.middleware("http")
async def cap_body_size(request: Request, call_next):
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        # Photos have their own, much larger allowance on /api/upload.
        if not request.url.path.startswith("/api/upload"):
            return JSONResponse({"error": "That's too long for me — shorten it a bit?"},
                                status_code=413)
    return await call_next(request)

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

    `message` is an ordinary chat turn. `reply` is an answer to whatever the
    agent is currently asking during the estimate interview — one of
    {"answer": "..."} / {"skip": true} / {"photo_ids": [...]}. `cancelled` backs
    out of the interview.
    """

    thread_id: str = ""
    message: str = ""
    reply: dict | None = None
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


def oversized(turn: Turn) -> bool:
    """
    True if anything in this turn is longer than a person would ever type.

    Belt and braces behind the middleware: Content-Length is absent on chunked
    requests, and this is the check that decides whether the text reaches graph
    state — which is the part that actually costs something, because state is
    rewritten to disk on every step for the rest of the conversation.
    """
    if len(turn.message) > config.MAX_MESSAGE_CHARS:
        return True
    for value in (turn.reply or {}).values():
        if isinstance(value, str) and len(value) > config.MAX_MESSAGE_CHARS:
            return True
        if isinstance(value, list) and len(value) > config.MAX_UPLOADS_PER_THREAD:
            return True
    return False


def _graph_input(graph, cfg: dict, turn: Turn):
    """
    Translate a browser request into the right kind of graph input.

    THIS is the resume protocol, in one place. If the graph is paused mid-
    interview, whatever arrives is an answer and must be wrapped in a Command. If
    it is not paused, it is a new message.
    """
    if _pending_interrupt(graph, cfg) is None:
        return {"messages": [HumanMessage(content=turn.message)]}, cfg

    resume_cfg = tracing.as_resume(cfg)
    if turn.cancelled:
        return Command(resume={"cancelled": True}), resume_cfg
    if turn.reply is not None:
        return Command(resume=turn.reply), resume_cfg
    # They typed in the message box while a question was on screen. That IS the
    # answer — the whole point of the conversational form is that there is no
    # difference between talking and answering.
    return Command(resume={"answer": turn.message}), resume_cfg


#: Nodes whose model output is the customer's reply, and may therefore be
#: streamed to the browser as it is generated. Everything else — the guard, the
#: router, prefill — is machinery producing structured data, and forwarding its
#: tokens would render classification JSON into the chat.
STREAMING_NODES = {"knowledge", "handoff"}


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

    if oversized(turn):
        return JSONResponse(
            {"error": "That's a lot in one go — could you shorten it?"},
            status_code=413,
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
            streamed = ""

            for mode, chunk in graph.stream(
                graph_input, run_cfg, stream_mode=["messages", "values"]
            ):
                if mode == "messages":
                    token, meta = chunk
                    if meta.get("langgraph_node") not in STREAMING_NODES:
                        continue
                    text = token.content if isinstance(token.content, str) else ""
                    if text:
                        streamed += text
                        yield event("token", {"text": text})
                elif mode == "values":
                    final = chunk

            # Paused mid-interview: the next question, not a finished answer.
            if (pending := _pending_interrupt(graph, cfg)) is not None:
                yield event("ask", {"paused": True, "ask": pending})
                return

            messages = final.get("messages") or []
            reply = messages[-1].content if messages else ""

            # What was streamed is the knowledge node's DRAFT. answer_check may
            # have sent it back and had it rewritten, or replaced it outright
            # with the handoff line — in which case the browser is showing text
            # that never got approved, and has to be corrected. In the common
            # case these are identical and nothing is sent.
            corrected = reply if reply.strip() != streamed.strip() else ""
            yield event(
                "done",
                {
                    "paused": False,
                    "reply": corrected,
                    "streamed": bool(streamed) and not corrected,
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
                    "streamed": False,
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
