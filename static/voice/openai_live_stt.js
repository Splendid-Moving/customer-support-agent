// OpenAI gpt-live-transcribe as a voiceloop speech-to-text provider.
//
// voiceloop owns the mic and the speech detector; this streams the detected
// speech to OpenAI over a browser WebSocket and reports the words back — live
// partials while the customer talks (which is what lets them interrupt Sam), and
// one final transcript per turn. The socket authenticates with a one-minute
// secret from our server (/api/voice/token); the API key never reaches the page.
// Tested in tests/voice_js/openai_live_stt.test.mjs.

const URL = 'wss://api.openai.com/v1/realtime?intent=transcription';
const RATE_IN = 16000;          // what voiceloop captures
const RATE_OUT = 24000;         // what OpenAI accepts (it rejects anything lower)
const MIN_COMMIT = RATE_OUT / 10;   // OpenAI refuses to commit under 100ms of audio
const MAX_FAILURES = 3;         // consecutive sockets that died before opening
const FINAL_TIMEOUT_MS = 8000;  // a turn whose transcript never arrives still has to close

/** Linear resample 16kHz → 24kHz (exactly 3 output samples per 2 input). */
export function resample16kTo24k(input) {
  const out = new Int16Array(Math.floor((input.length * 3) / 2));
  for (let i = 0; i < out.length; i++) {
    const pos = (i * 2) / 3;
    const k = Math.floor(pos);
    const a = input[k], b = input[Math.min(k + 1, input.length - 1)];
    out[i] = Math.round(a + (b - a) * (pos - k));
  }
  return out;
}

function base64(int16) {
  const bytes = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

async function fetchToken(tokenUrl) {
  const res = await fetch(tokenUrl, { method: 'POST' });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || !body.token) throw new Error(body.error || `token ${res.status}`);
  return body;
}

export function makeOpenAILiveSTT(opts) {
  const { onPartial, onFinal, onError, onFatal, onClose, isClosed } = opts;
  const WS = opts.WebSocketImpl || globalThis.WebSocket;
  const getToken = opts.getToken || (() => fetchToken(opts.sttTokenUrl || '/api/voice/token'));

  let ws = null, opening = false, dead = false, failures = 0;
  let outbox = [];              // messages queued while the socket connects
  let samplesSinceCommit = 0;
  // One entry per commit we sent, in order. OpenAI names the turn (item_id) when it
  // acknowledges the commit; transcripts can complete out of order, so finals are
  // released strictly in the order the customer spoke.
  let pending = [];
  const text = new Map();       // item_id → transcript so far (deltas)
  const done = new Map();       // item_id → final transcript
  let liveItem = null;          // the turn currently being spoken

  const stopped = () => dead || isClosed();

  const send = (msg) => {
    if (ws && ws.readyState === 1) ws.send(JSON.stringify(msg));
    else outbox.push(msg);
  };

  const release = () => {
    while (pending.length) {
      const head = pending[0];
      let final;
      if (head.empty) final = '';
      else if (head.item && done.has(head.item)) final = done.get(head.item);
      else if (head.expired) final = (head.item && text.get(head.item)) || '';
      else return;
      pending.shift();
      clearTimeout(head.timer);
      if (head.item) { text.delete(head.item); done.delete(head.item); }
      if (!stopped()) onFinal(final.trim(), 0);
    }
  };

  const onMessage = (ev) => {
    if (stopped()) return;
    let m;
    try { m = JSON.parse(ev.data); } catch { return; }
    switch (m.type) {
      case 'conversation.item.input_audio_transcription.delta': {
        const so = (text.get(m.item_id) || '') + (m.delta || '');
        text.set(m.item_id, so);
        if (!done.has(m.item_id)) { liveItem = m.item_id; onPartial(so.trim(), 0); }
        break;
      }
      case 'input_audio_buffer.committed': {
        const waiting = pending.find((p) => !p.item && !p.empty);
        if (waiting) waiting.item = m.item_id;
        if (liveItem === m.item_id) liveItem = null;
        release();
        break;
      }
      case 'conversation.item.input_audio_transcription.completed':
        done.set(m.item_id, m.transcript || '');
        release();
        break;
      case 'error': {
        const code = m.error?.code || '';
        if (code === 'input_audio_buffer_commit_empty') {
          const waiting = pending.find((p) => !p.item && !p.empty);
          if (waiting) { waiting.empty = true; release(); }
        } else {
          onError(`transcription: ${m.error?.message || code || 'error'}`);
        }
        break;
      }
      default:
        break;
    }
  };

  const open = async () => {
    if (opening || stopped() || (ws && ws.readyState < 2)) return;
    opening = true;
    let token;
    try {
      token = (await getToken()).token;
    } catch (e) {
      opening = false;
      dead = true;
      outbox = [];
      if (!isClosed()) onFatal(`Voice isn't available right now (${e?.message || e}).`);
      return;
    }
    opening = false;
    if (stopped()) return;
    let socket;
    try {
      socket = new WS(URL, ['realtime', `openai-insecure-api-key.${token}`]);
    } catch (e) {
      dead = true;
      onFatal(`Voice couldn't connect (${e?.message || e}).`);
      return;
    }
    ws = socket;
    let opened = false;
    socket.onopen = () => {
      opened = true; failures = 0;
      const queued = outbox; outbox = [];
      for (const msg of queued) socket.send(JSON.stringify(msg));
    };
    socket.onmessage = onMessage;
    socket.onerror = () => {};   // onclose follows with the code; reported there
    socket.onclose = (ev) => {
      if (ws === socket) ws = null;
      onClose?.();
      if (stopped() || ev.code === 1000) return;
      // Turns in flight on a dead socket will never complete: close them with
      // whatever was heard so the agent isn't left waiting.
      for (const p of pending) p.expired = true;
      release();
      if (!opened && ++failures >= MAX_FAILURES) {
        dead = true;
        onFatal("Voice keeps losing its connection — typing still works.");
        return;
      }
      // Reconnect lazily: the next utterance opens a fresh socket with a fresh
      // secret. Audio queued for the dead one belongs to a turn already closed
      // above — replaying it would glue half a sentence onto the next one.
      outbox = []; samplesSinceCommit = 0;
    };
  };

  return {
    // Called by voiceloop on start and on every resume after a pause. A fatal
    // failure ends this session; a fresh start is a fresh chance.
    open() { dead = false; failures = 0; open(); },
    feed(pcm16k) {
      if (stopped()) return;
      if (!ws || ws.readyState >= 2) open();
      const pcm = resample16kTo24k(pcm16k);
      samplesSinceCommit += pcm.length;
      send({ type: 'input_audio_buffer.append', audio: base64(pcm) });
    },
    commit() {
      if (stopped()) return;
      if (samplesSinceCommit < MIN_COMMIT) {
        // Too short to transcribe (a click, a breath). Throw it away and close the
        // turn empty, which voiceloop treats as "nobody said anything".
        if (samplesSinceCommit > 0) send({ type: 'input_audio_buffer.clear' });
        samplesSinceCommit = 0;
        onFinal('', 0);
        return;
      }
      samplesSinceCommit = 0;
      const entry = { item: null, empty: false, expired: false };
      entry.timer = setTimeout(() => { entry.expired = true; release(); }, FINAL_TIMEOUT_MS);
      pending.push(entry);
      send({ type: 'input_audio_buffer.commit' });
    },
    // voiceloop closes on pause as well as on stop, and opens again on resume —
    // so this is "hang up", not "never again". Late messages from the old socket
    // are cut off here rather than being mistaken for the next turn's.
    close() {
      for (const p of pending) clearTimeout(p.timer);
      pending = []; outbox = []; text.clear(); done.clear();
      samplesSinceCommit = 0; liveItem = null;
      const socket = ws; ws = null;
      if (socket) { socket.onmessage = null; socket.onclose = null; }
      try { socket?.close(1000); } catch {}
    },
  };
}
