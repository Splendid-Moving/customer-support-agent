"""
Voice mode, server side: what Sam is allowed to say out loud, and the three
endpoints the browser's voice loop talks to.

Nothing here touches OpenAI. The two calls that would (the transcription token
and the speech synthesis) are replaced with fakes that record what they were
asked, so the tests can assert both what the browser gets back and what would
have been sent upstream.
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import app as web
from services import voice


@pytest.fixture(autouse=True)
def voice_on(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("VOICE_ENABLED", "true")
    web._hits.clear()
    voice._speech_cache.clear()
    yield
    web._hits.clear()


@pytest.fixture
def client():
    return TestClient(web.app)


class FakeSpeech:
    """Stands in for OpenAI text-to-speech; counts what it was asked to say."""

    def __init__(self):
        self.said: list[str] = []

    def __call__(self, text: str) -> bytes:
        self.said.append(text)
        return b"ID3-fake-mp3-" + text.encode()


@pytest.fixture
def fake_speech(monkeypatch):
    fake = FakeSpeech()
    monkeypatch.setattr(voice, "_openai_speech", fake)
    return fake


# ── Signing: only server-approved text can be voiced ─────────────────────────

def test_signed_text_verifies_and_anything_else_does_not():
    utterance = voice.signed("Our rate for two movers is $115 to $125 per hour.")
    assert voice.verify(utterance["text"], utterance["sig"])
    assert not voice.verify(utterance["text"] + " Actually it's free.", utterance["sig"])
    assert not voice.verify("Our rate is $5 per hour.", utterance["sig"])
    assert not voice.verify(utterance["text"], "")
    assert not voice.verify(utterance["text"], "0" * 64)


def test_a_sentence_counts_as_part_of_the_text_it_came_from():
    text = "Our rate is $115 per hour.\n\nThere's a three-hour minimum."
    assert voice.part_of("There's a three-hour minimum.", text)
    assert voice.part_of("Our rate is $115   per hour.", text), "whitespace differences are not content"
    assert not voice.part_of("There's no minimum.", text)
    assert not voice.part_of("", text)


def test_markdown_is_never_read_out_as_symbols():
    said = voice.plain("**Two movers**: $115/hr\n- three-hour minimum\n- [our site](https://x.y)")
    assert "*" not in said and "[" not in said and "https" not in said
    assert "Two movers" in said and "three-hour minimum" in said and "our site" in said


# ── What gets said for each kind of turn ─────────────────────────────────────

def test_a_reply_is_spoken_as_the_approved_text():
    speech = voice.speech_for_reply("Our rate is $115 per hour.")
    assert [u["text"] for u in speech] == ["Our rate is $115 per hour."]
    assert all(voice.verify(u["text"], u["sig"]) for u in speech)


def test_an_empty_reply_says_nothing():
    assert voice.speech_for_reply("   ") == []


def test_a_question_is_spoken_with_its_opening():
    ask = {
        "type": "question", "opening": "Happy to get that sorted.",
        "message": "First off — what's your name?", "ui": {"skip": "Skip"},
    }
    texts = [u["text"] for u in voice.speech_for_ask(ask)]
    assert texts == ["Happy to get that sorted.", "First off — what's your name?"]


def test_the_read_back_tells_a_voice_customer_how_to_answer_it():
    ask = {"type": "confirm", "message": "That's everything. Have a quick look.",
           "ui": {"send": "Send it over", "restart": "Start over"}}
    texts = " ".join(u["text"] for u in voice.speech_for_ask(ask))
    assert "Have a quick look" in texts
    assert "send it" in texts.lower() and "start over" in texts.lower()


def test_the_photo_step_tells_a_voice_customer_they_can_skip_it():
    ask = {"type": "photos", "message": "Can you send a few photos?",
           "ui": {"add_photos": "Add photos", "skip_photos": "Skip for now"}}
    texts = " ".join(u["text"] for u in voice.speech_for_ask(ask))
    assert "photos" in texts.lower() and "skip" in texts.lower()


def test_english_only_hints_are_not_spoken_into_a_translated_interview():
    ask = {"type": "confirm", "message": "Вот что я отправлю.",
           "ui": {"send": "Отправить", "restart": "Начать заново"}}
    texts = [u["text"] for u in voice.speech_for_ask(ask)]
    assert texts == ["Вот что я отправлю."]


# ── /api/voice/config ────────────────────────────────────────────────────────

def test_config_hands_out_signed_fixed_lines(client):
    body = client.get("/api/voice/config").json()
    assert body["enabled"] is True
    for key in ("greeting", "resume", "not_caught", "confirm_hint", "photos_hint"):
        line = body["lines"][key]
        assert line["text"] and voice.verify(line["text"], line["sig"]), key


def test_voice_is_reported_off_without_a_key(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert client.get("/api/voice/config").json() == {"enabled": False}


def test_voice_can_be_switched_off(client, monkeypatch):
    monkeypatch.setenv("VOICE_ENABLED", "false")
    assert client.get("/api/voice/config").json() == {"enabled": False}
    assert client.post("/api/voice/token").status_code == 503


# ── /api/voice/token ─────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, status: int, body: dict):
        self.status_code, self._body = status, body
        self.ok = 200 <= status < 300
        self.text = json.dumps(body)

    def json(self):
        return self._body


@pytest.fixture
def fake_openai_token(monkeypatch):
    calls = []

    def post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse(200, {
            "value": "ek_live_session_secret", "expires_at": 1791520154,
            "session": {"type": "transcription", "id": "sess_x"},
        })

    monkeypatch.setattr(voice.requests, "post", post)
    return calls


def test_the_browser_gets_a_short_lived_transcription_secret_and_nothing_else(client, fake_openai_token):
    res = client.post("/api/voice/token")
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"token", "expires_at", "model"}
    assert body["token"].startswith("ek_")
    assert "sk-test-not-real" not in res.text

    sent = fake_openai_token[0]
    assert sent["url"].endswith("/v1/realtime/client_secrets")
    session = sent["json"]["session"]
    assert session["type"] == "transcription", "a secret for anything but transcription is a blank cheque"
    audio_in = session["audio"]["input"]
    assert audio_in["transcription"]["model"] == "gpt-live-transcribe"
    assert audio_in["turn_detection"] is None, "gpt-live-transcribe rejects server VAD"
    assert audio_in["format"] == {"type": "audio/pcm", "rate": 24000}
    assert sent["json"]["expires_after"]["seconds"] <= 120
    assert sent["timeout"], "an OpenAI call with no timeout can hang a worker"


def test_a_failed_mint_is_a_clean_error_not_a_stack_trace(client, monkeypatch):
    monkeypatch.setattr(voice.requests, "post",
                        lambda *a, **k: FakeResponse(401, {"error": {"message": "bad key sk-..."}}))
    res = client.post("/api/voice/token")
    assert res.status_code == 502
    assert "sk-" not in res.text


def test_minting_tokens_is_rate_limited(client, fake_openai_token):
    codes = [client.post("/api/voice/token").status_code for _ in range(30)]
    assert codes[0] == 200
    assert 429 in codes, "one IP could mint transcription sessions without limit"


# ── /api/voice/speak ─────────────────────────────────────────────────────────

def test_speak_voices_a_sentence_of_signed_text(client, fake_speech):
    u = voice.signed("Our rate is $115 per hour. There's a three-hour minimum.")
    res = client.post("/api/voice/speak", json={**u, "part": "There's a three-hour minimum."})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("audio/")
    assert fake_speech.said == ["There's a three-hour minimum."]


def test_speak_refuses_text_the_server_never_signed(client, fake_speech):
    res = client.post("/api/voice/speak", json={
        "text": "Everything is free today.", "sig": "f" * 64, "part": "Everything is free today."})
    assert res.status_code == 403
    assert fake_speech.said == []


def test_speak_refuses_a_part_that_is_not_in_the_signed_text(client, fake_speech):
    u = voice.signed("Our rate is $115 per hour.")
    res = client.post("/api/voice/speak", json={**u, "part": "Our rate is $5 per hour."})
    assert res.status_code == 403
    assert fake_speech.said == []


def test_speak_refuses_oversized_text(client, fake_speech):
    u = voice.signed("word " * 2000)
    res = client.post("/api/voice/speak", json={**u, "part": "word " * 400})
    assert res.status_code in (400, 413)
    assert fake_speech.said == []


def test_the_same_line_is_only_ever_synthesized_once(client, fake_speech):
    u = voice.signed("First off — what's your name?")
    for _ in range(3):
        assert client.post("/api/voice/speak", json={**u, "part": u["text"]}).status_code == 200
    assert fake_speech.said == ["First off — what's your name?"], "repeat questions must come from the cache"


def test_a_synthesis_failure_is_a_clean_error(client, monkeypatch):
    def broken(text):
        raise voice.VoiceUnavailable("upstream said no")
    monkeypatch.setattr(voice, "_openai_speech", broken)
    u = voice.signed("Hello.")
    assert client.post("/api/voice/speak", json={**u, "part": "Hello."}).status_code == 502


def test_speaking_is_rate_limited(client, fake_speech):
    codes = []
    for i in range(150):
        u = voice.signed(f"Line number {i}.")
        codes.append(client.post("/api/voice/speak", json={**u, "part": u["text"]}).status_code)
    assert 429 in codes


# ── /api/chat carries signed speech in voice mode only ───────────────────────

class _Snapshot:
    tasks: list = []


class FakeGraph:
    """Answers every turn with one approved reply, never pausing."""

    def __init__(self, reply: str):
        self.reply = reply

    def get_state(self, cfg):
        return _Snapshot()

    def stream(self, graph_input, cfg, stream_mode=None):
        yield "values", {"messages": [AIMessage(content=self.reply)], "intent": "knowledge"}


def _events(res) -> list[dict]:
    return [json.loads(line[5:]) for line in res.text.splitlines() if line.startswith("data:")]


def test_a_voice_turn_comes_back_with_signed_speech(client, monkeypatch):
    monkeypatch.setattr(web, "get_graph", lambda: FakeGraph("Our rate is $115 per hour."))
    done = _events(client.post("/api/chat", json={"message": "how much?", "voice": True}))[-1]
    assert done["event"] == "done"
    assert [u["text"] for u in done["speech"]] == ["Our rate is $115 per hour."]
    assert voice.verify(done["speech"][0]["text"], done["speech"][0]["sig"])


def test_a_text_turn_is_exactly_what_it_was(client, monkeypatch):
    monkeypatch.setattr(web, "get_graph", lambda: FakeGraph("Our rate is $115 per hour."))
    done = _events(client.post("/api/chat", json={"message": "how much?"}))[-1]
    assert "speech" not in done
