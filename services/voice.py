"""
Voice mode's server half: what Sam may say out loud, and the two OpenAI calls.

The browser never decides what is spoken. The server signs every line it wants
voiced (an approved reply, an interview question, a fixed hint), and the speak
endpoint only synthesizes text carrying a valid signature — so neither a draft
that answer_check rejected nor text typed into the browser console can come out
of the speaker. The transcription side hands the browser a short-lived secret
that can only open a transcription session; the API key never leaves here.
"""

import hashlib
import hmac
import logging
import re
import secrets
import threading
from collections import OrderedDict

import requests

from schemas import voice as lines
from services import config

logger = logging.getLogger(__name__)

OPENAI = "https://api.openai.com/v1"


class VoiceUnavailable(Exception):
    """OpenAI could not be reached or refused. Safe to show: carries no upstream detail."""


def enabled() -> bool:
    return config.voice_enabled()


# ── Signing ──────────────────────────────────────────────────────────────────
# Random per process. A deploy invalidates signatures still in a browser, which
# costs at most one unspoken sentence that is already on screen as text — far
# cheaper than one more secret to configure and rotate.
_KEY = secrets.token_bytes(32)


def sign(text: str) -> str:
    return hmac.new(_KEY, text.encode("utf-8"), hashlib.sha256).hexdigest()


def verify(text: str, sig: str) -> bool:
    return bool(sig) and hmac.compare_digest(sign(text), str(sig))


def signed(text: str) -> dict:
    return {"text": text, "sig": sign(text)}


def _squash(text: str) -> str:
    return " ".join(str(text or "").split())


def part_of(part: str, text: str) -> bool:
    """
    Whether `part` is a stretch of `text`, ignoring whitespace.

    The browser speaks a reply a sentence at a time and splits the sentences
    itself, so it sends the whole signed text with each piece. The piece only
    has to be IN that text — which is what stops a valid signature being
    reused to voice something else.
    """
    part = _squash(part)
    return bool(part) and part in _squash(text)


# ── What gets said ───────────────────────────────────────────────────────────

_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+", re.MULTILINE)
_MARKS = re.compile(r"[*_`#>]+")


def plain(text: str) -> str:
    """
    Text as it should sound: no markdown read out as symbols, no URLs spelled out.

    The replies are written for a chat window and sometimes carry a bullet or
    some bold. A voice reading "asterisk asterisk two movers" is the fastest way
    to sound like a bot.
    """
    text = _LINK.sub(r"\1", str(text or ""))
    text = _URL.sub("", text)
    text = _BULLET.sub("", text)
    text = _MARKS.sub("", text)
    # A list becomes a run of sentences, not one breathless line.
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(line if line[-1] in ".!?:;…" else line + ".")
    return " ".join(out).strip()


def speech_for_reply(reply: str) -> list[dict]:
    """A finished reply, signed for speaking. Only ever the approved text."""
    said = plain(reply)
    return [signed(said)] if said else []


def _interview_in_english(ask: dict) -> bool:
    ui = ask.get("ui") or {}
    shared = [k for k in lines.ENGLISH_UI if k in ui]
    return all(ui[k] == lines.ENGLISH_UI[k] for k in shared)


def speech_for_ask(ask: dict) -> list[dict]:
    """
    An interview step, signed for speaking: its opening line, the question,
    and — at the two steps that need a tap or a clear yes — how to answer it.

    The answers on the read-back are not read aloud. They are on screen, laid
    out to be checked, and nine fields recited one after another is the opposite
    of the "quick look" the line asks for.
    """
    said = [plain(ask.get("opening") or ""), plain(ask.get("message") or "")]
    hint = lines.HINT_AFTER.get(ask.get("type") or "")
    if hint and _interview_in_english(ask):
        said.append(hint)
    return [signed(s) for s in said if s]


def fixed_lines() -> dict[str, dict]:
    """The handful of lines the page may say on its own, each signed."""
    return {key: signed(text) for key, text in lines.LINES.items()}


# ── Transcription: a short-lived secret for the browser ──────────────────────

def mint_transcription_token() -> dict:
    """
    A client secret that opens ONE kind of session: transcription, with our
    model and settings fixed here. It expires a minute after minting if
    unused, so one that leaks from a page is worth almost nothing.

    turn_detection is off on purpose: gpt-live-transcribe rejects server VAD,
    and the browser's own voice detector decides when a turn ends.
    """
    body = {
        "expires_after": {"anchor": "created_at", "seconds": 60},
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {
                        "model": config.voice_stt_model(),
                        "prompt": lines.STT_PROMPT,
                        "keywords": lines.STT_KEYWORDS,
                        "delay": "low",
                    },
                    "turn_detection": None,
                    "noise_reduction": {"type": "near_field"},
                }
            },
        },
    }
    try:
        res = requests.post(
            f"{OPENAI}/realtime/client_secrets",
            headers={"Authorization": f"Bearer {config.openai_api_key()}"},
            json=body,
            timeout=10,
        )
    except requests.RequestException as exc:
        logger.warning("Voice token: OpenAI unreachable: %s", type(exc).__name__)
        raise VoiceUnavailable("transcription is unreachable") from exc

    if not res.ok:
        # The upstream body can echo part of the key; log the status, not it.
        logger.warning("Voice token: OpenAI refused (HTTP %s)", res.status_code)
        raise VoiceUnavailable("transcription refused")

    data = res.json()
    token = str(data.get("value") or "")
    if not token.startswith("ek_"):
        logger.warning("Voice token: unexpected response shape")
        raise VoiceUnavailable("transcription refused")
    return {"token": token, "expires_at": data.get("expires_at"), "model": config.voice_stt_model()}


# ── Speech: synthesis with a cache ───────────────────────────────────────────
# Interview questions are the same sentence for every customer, so after the
# first person hears "What's the zip code we're moving you out of?" it costs
# nothing and starts instantly. Bounded by bytes, oldest out first.

_CACHE_BYTES = 48 * 1024 * 1024
_speech_cache: "OrderedDict[str, bytes]" = OrderedDict()
_cache_lock = threading.Lock()

AUDIO_TYPE = "audio/mpeg"


def _cache_key(text: str) -> str:
    raw = "\x1f".join([config.voice_tts_model(), config.voice_tts_voice(), lines.TTS_INSTRUCTIONS, text])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _openai_speech(text: str) -> bytes:
    try:
        res = requests.post(
            f"{OPENAI}/audio/speech",
            headers={"Authorization": f"Bearer {config.openai_api_key()}"},
            json={
                "model": config.voice_tts_model(),
                "voice": config.voice_tts_voice(),
                "input": text,
                "instructions": lines.TTS_INSTRUCTIONS,
                "response_format": "mp3",
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        logger.warning("Voice speak: OpenAI unreachable: %s", type(exc).__name__)
        raise VoiceUnavailable("speech is unreachable") from exc
    if not res.ok or not res.content:
        logger.warning("Voice speak: OpenAI refused (HTTP %s)", res.status_code)
        raise VoiceUnavailable("speech refused")
    return res.content


def synthesize(text: str) -> bytes:
    """Audio for one sentence, from the cache when we have said it before."""
    text = _squash(text)
    key = _cache_key(text)
    with _cache_lock:
        if key in _speech_cache:
            _speech_cache.move_to_end(key)
            return _speech_cache[key]

    audio = _openai_speech(text)

    with _cache_lock:
        _speech_cache[key] = audio
        total = sum(len(v) for v in _speech_cache.values())
        while total > _CACHE_BYTES and len(_speech_cache) > 1:
            _, dropped = _speech_cache.popitem(last=False)
            total -= len(dropped)
    return audio
