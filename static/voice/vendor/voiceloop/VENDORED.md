# voiceloop — vendored copy

`@todoforai/voiceloop` **0.1.25** (npm, published 2026-09-30), MIT — see `LICENSE`.
Source: https://github.com/todoforai/voiceloop

Copied unmodified from the npm tarball's `src/` (tests and type definitions left
out). It is vendored rather than loaded from a CDN so the version can only change
on purpose, in a commit someone reviewed.

Our code never edits these files. Extensions live next door in `static/voice/`:
the OpenAI transcription provider is registered into `STT_PROVIDERS`, and our TTS
subclasses `StreamingTTS` — both documented extension points.

voiceloop itself still loads the speech detector (`@ricky0123/vad-web@0.0.29`)
and `onnxruntime-web@1.22.0` from jsDelivr at pinned versions, on first use.

To upgrade: `npm pack @todoforai/voiceloop@<version>`, copy `src/*.js` (minus
`*.test.js`) over these, read the changelog, run `node --test tests/voice_js/`.
