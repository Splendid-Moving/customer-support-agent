# Voice mode for the support agent

## What we're building
A hands-free "talk to Sam" mode on the existing chat page. The customer taps 🎤 once.
After that they talk and Sam talks back, with no buttons, and they can interrupt Sam
mid-sentence. The conversation, the estimate interview and the email are the same
LangGraph agent as before. Voice is a layer on top: speech becomes text going in, and
approved text becomes speech coming out.

The safe point to return to is git tag `pre-voice-mode` (text-only, known good).
This work lives on branch `voice-mode`.

## Contract

GOAL:
- One tap on 🎤 → mic permission → Sam greets out loud → the customer asks a question
  by voice and hears the approved answer. Hands-free from then on.
- A whole estimate can be given by voice, with photos as the only tap: name, phone,
  email, zips, "next Friday", "two bedroom", "yes, send it".
- Speaking while Sam talks cuts Sam off within about a second.
- Latency (p50, measured locally): first audio within 4s of the customer stopping on a
  knowledge answer, and within 1.5s on an interview question that is already cached.

CONSTRAINTS:
- Speech-to-text: OpenAI `gpt-live-transcribe` (streaming, released 2026-07-27, the
  newest). Text-to-speech: OpenAI `gpt-4o-mini-tts-2025-12-15`, voice `cedar` (the
  newest TTS snapshot; `gpt-live-1` and `gpt-realtime-*` are speech-to-speech and
  would bypass the graph). Both can be overridden by env var.
- The browser loop is voiceloop 0.1.25 (MIT), vendored, not loaded from a CDN.
- No new paid services. The existing `OPENAI_API_KEY` covers both.
- Text mode is unchanged. All existing tests still pass.
- No build step. The page stays plain HTML and ES modules.

FORMAT:
- `static/voice/`: the browser side (see Layout in README).
- `services/voice.py`: OpenAI token, speech synthesis with a cache, signing.
- `schemas/voice.py`: how Sam sounds and listens (voice, tone, fixed spoken lines).
- `app.py`: `/api/voice/config`, `/api/voice/token` and `/api/voice/speak`. `/api/chat`
  adds signed `speech` to its events when `voice: true`.
- `schemas/lead_form.py`: accepts spoken-style answers (dates in words, number words,
  "at"/"dot" emails, spaced zips). Typed answers benefit too.

FAILURE (any of these = not done):
1. Sam says anything the server didn't sign. That includes an unapproved draft or text
   made up in the browser. → pytest: `/api/voice/speak` rejects unsigned, tampered or
   out-of-text requests.
2. The OpenAI key, or a token usable for anything but transcription, reaches the
   browser. → pytest: the token response holds only a short-lived `ek_` secret.
3. An interruption aborts an in-flight `/api/chat`, or two turns hit the server at
   once. Either can corrupt the interview's pause/resume. → node test.
4. Voiceloop answers early on a half-finished sentence (speculative prefetch).
   → node test.
5. Any existing text-mode behaviour changes. → `pytest` (277+) green, plus the browser
   suite.
6. A spoken answer a human would accept gets rejected ("next friday", "Two bedroom.",
   "nik at gmail dot com", "9 0 0 2 6", "Nik."). → pytest + node tests.
7. Voice fails silently. A denied mic, a missing key, a token failure or a dropped
   socket must each show a plain message, and typing must keep working.
   → node tests + manual check.
8. Costs are unbounded. → per-IP limits on token and speak, a size cap on spoken text,
   and a cache for repeated lines. pytest.
9. The mic stays on after the customer stops voice or goes idle for 2 minutes.
   → manual check that the browser's mic indicator turns off.
10. There's no AI-voice disclosure. OpenAI's usage policy requires one.
    → visible "AI voice" label.

## Built on top of
- **voiceloop 0.1.25** (github.com/todoforai/voiceloop, MIT, active, 184+ tests): mic
  capture, Silero VAD, echo-filtered word-level interruption, sentence-pipelined TTS
  playback, and turn ordering. We add one STT provider (OpenAI) and one TTS class
  (our server). Its own audit lists known rough edges, which is why the copy is
  pinned and vendored.
- **@ricky0123/vad-web + onnxruntime-web** (loaded by voiceloop from jsDelivr, pinned
  versions): the in-browser speech detector.
- **dateparser 1.4.3** (BSD, maintained): turns "next Friday" or "el próximo viernes"
  into a date, in any language the agent speaks.
- **OpenAI Realtime transcription** over a browser WebSocket, authenticated with a
  60-second client secret minted by our server.

## Gotchas we're handling
- **Unapproved drafts.** `/api/chat` streams a draft that `answer_check` may reject.
  We speak only the final, approved text from the `done`/`ask` event, never the
  streamed tokens.
- **Speak endpoint as free TTS.** We handle it with HMAC-signed speech. The server
  signs what it wants said, and `/speak` only voices part of a signed text.
- **Resume protocol.** Interruption stops audio only. A voice turn waits for any
  in-flight request to finish, so there is always exactly one request at a time,
  which is the same rule as `busy` in text mode.
- **Prefetch.** voiceloop guesses at half-finished sentences. We switch that off with
  its documented `executesTools` flag, because a guess here would be a real answer
  to a real question.
- **Audio rates.** OpenAI requires 24kHz PCM and voiceloop captures 16kHz. We
  resample in the provider.
- **Autoplay rules.** Browsers block sound until the customer taps, so the mic is
  requested on the 🎤 tap rather than on page load. A page asking for the mic
  unprompted also gets blocked more often.
- **Long sessions.** OpenAI sessions end after 60 minutes, and sockets drop. The
  provider reconnects with a fresh token on the next utterance, and gives up with a
  message after 3 failures.
- **Cost.** Audio only goes to OpenAI while someone is speaking (VAD-gated).
  Interview questions are identical for everyone, so their audio is cached. The
  estimated cost of a 5-minute voice chat is about $0.05–0.10.
- **Multilingual.** STT auto-detects the language. TTS speaks whatever language the
  approved text is in. The few spoken hints ("say send it…") are English-only.
- **Restarts.** The signing key is random per process. A deploy mid-sentence costs
  one unspoken sentence, and the text is still on screen.

## Build sequence
1. Server: `schemas/voice.py`, `services/voice.py` (sign/verify, token, TTS + cache),
   endpoints, signed `speech` on chat events. Tests first.
2. Interview: spoken-answer normalisers in `lead_form` + collect_lead. Tests first.
3. Browser: vendor voiceloop, then add the OpenAI STT provider, the server TTS class,
   spoken-answer mapping and voice-mode wiring. Node tests first.
4. Page: 🎤 button, voice bar (state, live transcript, AI-voice label), and errors.
5. Verify: pytest, node tests, the Playwright suite, and a voice end-to-end with a fake
   mic fed by recorded speech. Manual check on a phone.

## How to run and test
- Run: `.venv/bin/python server.py` → http://localhost:8080 (needs `OPENAI_API_KEY`)
- Python tests: `.venv/bin/python -m pytest` (offline, no key)
- Voice JS tests: `node --test tests/voice_js/*.test.mjs`
- Browser suite: see `tests/browser/README.md`. The voice run is
  `tests/browser/voice.js`, which costs a few cents of OpenAI.
- Env (optional): `VOICE_ENABLED`, `VOICE_STT_MODEL`, `VOICE_TTS_MODEL`, `VOICE_TTS_VOICE`

## Status
_Updated: 2026-10-08_
- Built: all phases. Server (signed speech, token, TTS + cache), spoken-answer
  understanding, browser voice loop (OpenAI live STT provider, server TTS, page
  wiring, 🎤 button + voice bar). 355 pytest + 40 node tests green.
- Verified live: a spoken question was answered hands-free and the interview
  took spoken answers. Cached questions start speaking about 0.1s after the
  customer stops; a fresh answer takes about 2s.
- Changed from plan: (1) whole-sentence TTS instead of voiceloop's first-word
  clips, which were choppy over the network. (2) A VAD "misfire" on one-word
  answers is treated as end of speech: voiceloop 0.1.25 ignores misfires and
  merged "Skip." into the next turn. (3) The date reader ignores commas
  ("Next, Friday.").
- Spoken style: voice turns set `spoken` on the graph input, which adds
  persona.SPOKEN (no lists, a figure and a question) to the knowledge and
  handoff prompts. Guard, router and checks are unchanged. answer_check now also
  rejects amounts written without a $ sign, which the price check can't see.
  The voice is `cedar` (male).
- Next: Nikita tests by voice on desktop and phone. The full fake-mic run
  (tests/browser/voice.js) was cut short after the fixes above, so rerun it
  before merging.
- Watch out for: `gpt-live-transcribe` rejects `server_vad`/`semantic_vad` and
  rates below 24000. The browser WS is `wss://api.openai.com/v1/realtime?intent=transcription`
  with subprotocols `realtime` and `openai-insecure-api-key.<ek>`.
