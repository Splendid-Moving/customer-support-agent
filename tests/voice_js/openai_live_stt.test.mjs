// The OpenAI gpt-live-transcribe provider for voiceloop, against a fake socket.
// The wire format asserted here was recorded from the real API (see SPEC.md).
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { makeOpenAILiveSTT, resample16kTo24k } from '../../static/voice/openai_live_stt.js';

class FakeSocket {
  static all = [];
  constructor(url, protocols) {
    this.url = url; this.protocols = protocols; this.sent = []; this.readyState = 0;
    FakeSocket.all.push(this);
  }
  send(data) { this.sent.push(JSON.parse(data)); }
  close() { this.readyState = 3; this.onclose?.({ code: 1000, reason: '' }); }
  // test helpers
  open() { this.readyState = 1; this.onopen?.(); }
  emit(event) { this.onmessage?.({ data: JSON.stringify(event) }); }
  drop(code = 1006) { this.readyState = 3; this.onclose?.({ code, reason: 'gone' }); }
  ofType(type) { return this.sent.filter((m) => m.type === type); }
}

const tick = () => new Promise((r) => setTimeout(r, 0));

function harness({ tokens = ['ek_one', 'ek_two', 'ek_three', 'ek_four'], failTokens = false } = {}) {
  FakeSocket.all = [];
  const log = { partials: [], finals: [], errors: [], fatals: [], minted: 0 };
  let closed = false;
  const stt = makeOpenAILiveSTT({
    WebSocketImpl: FakeSocket,
    getToken: async () => {
      log.minted++;
      if (failTokens) throw new Error('503');
      return { token: tokens[log.minted - 1] };
    },
    onPartial: (t) => log.partials.push(t),
    onFinal: (t) => log.finals.push(t),
    onError: (e) => log.errors.push(String(e)),
    onFatal: (e) => log.fatals.push(String(e)),
    onClose: () => {},
    isClosed: () => closed,
  });
  return { stt, log, close: () => { closed = true; stt.close(); } };
}

const speech = (ms) => new Int16Array(16 * ms).fill(1000);   // 16kHz mono

test('connects with a short-lived secret, never an API key, on the transcription intent', async () => {
  const { stt } = harness();
  stt.open();
  await tick();
  const ws = FakeSocket.all[0];
  assert.equal(ws.url, 'wss://api.openai.com/v1/realtime?intent=transcription');
  assert.deepEqual(ws.protocols, ['realtime', 'openai-insecure-api-key.ek_one']);
});

test('16kHz capture is resampled to the 24kHz OpenAI requires', () => {
  const out = resample16kTo24k(new Int16Array([0, 300, 600, 900]));
  assert.equal(out.length, 6);
  assert.equal(out[0], 0);
  assert.ok(out.every((v, i) => i === 0 || v >= out[i - 1]), 'a ramp stays a ramp');
});

test('audio fed before the socket opens is sent once it does, in order, as base64 PCM16', async () => {
  const { stt } = harness();
  stt.feed(speech(100)); stt.feed(speech(100));
  await tick();
  const ws = FakeSocket.all[0];
  assert.equal(ws.ofType('input_audio_buffer.append').length, 0);
  ws.open();
  const appends = ws.ofType('input_audio_buffer.append');
  assert.equal(appends.length, 2);
  const bytes = Buffer.from(appends[0].audio, 'base64');
  assert.equal(bytes.length, 2400 * 2, '100ms at 24kHz, 16-bit');
});

test('live words become partials, and the commit becomes exactly one final', async () => {
  const { stt, log } = harness();
  stt.open(); await tick();
  const ws = FakeSocket.all[0]; ws.open();
  stt.feed(speech(500));
  ws.emit({ type: 'conversation.item.input_audio_transcription.delta', item_id: 'i1', delta: ' How' });
  ws.emit({ type: 'conversation.item.input_audio_transcription.delta', item_id: 'i1', delta: ' much' });
  assert.deepEqual(log.partials, ['How', 'How much']);

  stt.commit();
  assert.equal(ws.ofType('input_audio_buffer.commit').length, 1);
  ws.emit({ type: 'input_audio_buffer.committed', item_id: 'i1' });
  ws.emit({ type: 'conversation.item.input_audio_transcription.delta', item_id: 'i1', delta: '?' });
  ws.emit({ type: 'conversation.item.input_audio_transcription.completed', item_id: 'i1', transcript: 'How much?' });
  assert.deepEqual(log.finals, ['How much?']);
});

test('a commit with too little audio is never sent — OpenAI rejects under 100ms — but still closes the turn', async () => {
  const { stt, log } = harness();
  stt.open(); await tick();
  const ws = FakeSocket.all[0]; ws.open();
  stt.feed(speech(40));
  stt.commit();
  assert.equal(ws.ofType('input_audio_buffer.commit').length, 0);
  assert.equal(ws.ofType('input_audio_buffer.clear').length, 1);
  assert.deepEqual(log.finals, ['']);
});

test('a commit with no audio at all closes the turn empty', async () => {
  const { stt, log } = harness();
  stt.commit();
  assert.deepEqual(log.finals, ['']);
});

test('finals are delivered in the order the turns were spoken, even if they complete out of order', async () => {
  const { stt, log } = harness();
  stt.open(); await tick();
  const ws = FakeSocket.all[0]; ws.open();
  stt.feed(speech(300)); stt.commit(); ws.emit({ type: 'input_audio_buffer.committed', item_id: 'a' });
  stt.feed(speech(300)); stt.commit(); ws.emit({ type: 'input_audio_buffer.committed', item_id: 'b' });
  ws.emit({ type: 'conversation.item.input_audio_transcription.completed', item_id: 'b', transcript: 'second' });
  assert.deepEqual(log.finals, []);
  ws.emit({ type: 'conversation.item.input_audio_transcription.completed', item_id: 'a', transcript: 'first' });
  assert.deepEqual(log.finals, ['first', 'second']);
});

test('an empty-commit error from OpenAI closes the pending turn rather than wedging it', async () => {
  const { stt, log } = harness();
  stt.open(); await tick();
  const ws = FakeSocket.all[0]; ws.open();
  stt.feed(speech(200)); stt.commit();
  ws.emit({ type: 'error', error: { code: 'input_audio_buffer_commit_empty', message: 'buffer too small' } });
  assert.deepEqual(log.finals, ['']);
  assert.deepEqual(log.fatals, []);
});

test('a dropped socket reconnects with a fresh secret on the next utterance', async () => {
  const { stt, log } = harness();
  stt.open(); await tick();
  FakeSocket.all[0].open();
  FakeSocket.all[0].drop();
  assert.deepEqual(log.fatals, []);
  stt.feed(speech(100)); await tick();
  assert.equal(FakeSocket.all.length, 2);
  assert.deepEqual(FakeSocket.all[1].protocols, ['realtime', 'openai-insecure-api-key.ek_two']);
});

test('a socket that keeps failing gives up with a fatal error instead of looping forever', async () => {
  const { stt, log } = harness();
  for (let i = 0; i < 4; i++) {
    stt.feed(speech(100)); await tick();
    FakeSocket.all[FakeSocket.all.length - 1].drop();   // dies before ever opening
  }
  assert.equal(log.fatals.length, 1);
});

test('a token that cannot be minted is fatal, with a message a person can read', async () => {
  const { stt, log } = harness({ failTokens: true });
  stt.open(); await tick(); await tick();
  assert.equal(log.fatals.length, 1);
  assert.equal(FakeSocket.all.length, 0);
});

test('close() is final: no reconnect, no late callbacks', async () => {
  const h = harness();
  h.stt.open(); await tick();
  const ws = FakeSocket.all[0]; ws.open();
  h.stt.feed(speech(300)); h.stt.commit();
  h.close();
  ws.emit({ type: 'conversation.item.input_audio_transcription.completed', item_id: 'x', transcript: 'late' });
  h.stt.feed(speech(100)); await tick();
  assert.equal(FakeSocket.all.length, 1);
  assert.deepEqual(h.log.finals, []);
});

test('a pause closes the socket and a resume opens a new one', async () => {
  const { stt } = harness();
  stt.open(); await tick();
  FakeSocket.all[0].open();
  stt.close();                       // voiceloop's stop() on pause
  assert.equal(FakeSocket.all[0].readyState, 3);
  stt.open(); await tick();          // voiceloop's start() on resume
  assert.equal(FakeSocket.all.length, 2);
});

test('after a fatal failure, starting voice again gets a fresh chance', async () => {
  const { stt, log } = harness();
  for (let i = 0; i < 3; i++) { stt.feed(speech(100)); await tick(); FakeSocket.all.at(-1).drop(); }
  assert.equal(log.fatals.length, 1);
  stt.feed(speech(100)); await tick();
  assert.equal(FakeSocket.all.length, 3, 'no reconnect loop after giving up');
  stt.open(); await tick();
  assert.equal(FakeSocket.all.length, 4);
});
