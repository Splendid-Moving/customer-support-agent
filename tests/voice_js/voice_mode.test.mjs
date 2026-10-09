// How voiceloop is wired to the existing conversation, and the TTS that only
// voices server-signed text. Run: node --test tests/voice_js/
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { makeConversationLLM, vadOptionsFor } from '../../static/voice/voice_mode.js';
import { ServerTTS, splitSentences } from '../../static/voice/server_tts.js';
import { VoiceAgent, STT_PROVIDERS } from '../../static/voice/vendor/voiceloop/index.js';

const LINES = {
  confirm_hint: { text: 'Say send it, or start over.', sig: 's-confirm' },
  photos_hint: { text: 'Tap Add photos, or say skip.', sig: 's-photos' },
  not_caught: { text: 'Sorry, I missed that.', sig: 's-missed' },
};

function fakePage({ asking = null, photoIds = [], reply = 'Our rate is $115 per hour.', delay = 0 } = {}) {
  const page = {
    sent: [], active: 0, maxActive: 0, signals: [],
    asking: () => asking,
    photoIds: () => photoIds,
    async send(payload, echo, opts) {
      page.sent.push({ payload, echo });
      page.signals.push(opts?.signal);
      page.active++; page.maxActive = Math.max(page.maxActive, page.active);
      await new Promise((r) => setTimeout(r, delay));
      page.active--;
      return { event: 'done', speech: [{ text: reply, sig: 'sig-' + page.sent.length }] };
    },
  };
  return page;
}

function fakeTts() {
  const tts = { remembered: [], remember: (u) => tts.remembered.push(...u) };
  return tts;
}

async function drain(gen) {
  const out = [];
  for await (const chunk of gen) out.push(typeof chunk === 'string' ? chunk : chunk.text);
  return out.join('');
}

const turn = (llm, text, signal = new AbortController().signal) =>
  llm([{ role: 'user', content: text }], '', signal);

test('voiceloop never speculates on a half-finished sentence: the adapter opts out of prefetch', () => {
  const llm = makeConversationLLM({ page: fakePage(), tts: fakeTts(), lines: LINES });
  assert.equal(llm.executesTools, true);
  assert.ok(!llm.acceptsToolGate);
});

test('the real VoiceAgent refuses to prefetch with this adapter', () => {
  let calls = 0;
  STT_PROVIDERS.__fake = () => ({ open() {}, feed() {}, commit() {}, close() {} });
  const llm = makeConversationLLM({ page: fakePage(), tts: fakeTts(), lines: LINES });
  const counted = Object.assign((...a) => { calls++; return llm(...a); }, { executesTools: llm.executesTools });
  const agent = new VoiceAgent({ sttProvider: '__fake', llm: counted, tts: { speak: async () => '' } });
  agent.state = 'listening'; agent._turnText = 'how much do';
  agent._startPrefetch('how much do');
  assert.equal(calls, 0, 'a speculative request here would be a real answer to a half-asked question');
  delete STT_PROVIDERS.__fake;
});

test('a spoken message goes to the conversation and the signed reply is what gets spoken', async () => {
  const page = fakePage(), tts = fakeTts();
  const llm = makeConversationLLM({ page, tts, lines: LINES });
  const said = await drain(turn(llm, 'How much for two movers?'));
  assert.deepEqual(page.sent[0], { payload: { message: 'How much for two movers?' }, echo: 'How much for two movers?' });
  assert.equal(said.trim(), 'Our rate is $115 per hour.');
  assert.deepEqual(tts.remembered, [{ text: 'Our rate is $115 per hour.', sig: 'sig-1' }]);
});

test('an interruption stops the voice but never aborts the server request', async () => {
  const page = fakePage({ delay: 20 });
  const llm = makeConversationLLM({ page, tts: fakeTts(), lines: LINES });
  const ctl = new AbortController();
  const pending = drain(turn(llm, 'hello', ctl.signal));
  await new Promise((r) => setTimeout(r, 5));
  ctl.abort();
  const said = await pending;
  assert.equal(said, '', 'nothing is spoken for a turn the customer talked over');
  assert.equal(page.sent.length, 1);
  assert.equal(page.signals[0], undefined, 'the request must not be tied to the barge-in signal');
});

test('turns reach the server one at a time, in order', async () => {
  const page = fakePage({ delay: 15 });
  const llm = makeConversationLLM({ page, tts: fakeTts(), lines: LINES });
  await Promise.all([drain(turn(llm, 'first')), drain(turn(llm, 'second'))]);
  assert.equal(page.maxActive, 1, 'two requests in flight at once can corrupt a paused interview');
  assert.deepEqual(page.sent.map((s) => s.payload.message), ['first', 'second']);
});

test('an unclear answer at the read-back is answered locally and sends nothing', async () => {
  const page = fakePage({ asking: { type: 'confirm' } }), tts = fakeTts();
  const llm = makeConversationLLM({ page, tts, lines: LINES });
  const said = await drain(turn(llm, 'Hmm, the phone is wrong.'));
  assert.equal(page.sent.length, 0);
  assert.equal(said.trim(), LINES.confirm_hint.text);
  assert.deepEqual(tts.remembered, [LINES.confirm_hint]);
});

test('a spoken answer during the interview goes out as an answer, not a new message', async () => {
  const page = fakePage({ asking: { type: 'question', field: { name: 'name', kind: 'text' } } });
  const llm = makeConversationLLM({ page, tts: fakeTts(), lines: LINES });
  await drain(turn(llm, 'Nik.'));
  assert.deepEqual(page.sent[0].payload, { reply: { answer: 'Nik.' } });
});

test('a failed turn is spoken as a short apology, never as silence', async () => {
  const page = fakePage();
  page.send = async () => { throw new Error('network'); };
  const llm = makeConversationLLM({ page, tts: fakeTts(), lines: LINES });
  assert.equal((await drain(turn(llm, 'hello'))).trim(), LINES.not_caught.text);
});

// ── ServerTTS ────────────────────────────────────────────────────────────────

function fakeFetch(status = 200) {
  const calls = [];
  const impl = async (url, init) => {
    calls.push({ url, body: JSON.parse(init.body), signal: init.signal });
    return { ok: status === 200, status, blob: async () => new Blob(['mp3'], { type: 'audio/mpeg' }) };
  };
  return { impl, calls };
}

test('the TTS sends each sentence with the signed text it came from', async () => {
  const f = fakeFetch();
  const tts = new ServerTTS({ speakUrl: '/api/voice/speak', fetchImpl: f.impl });
  tts.remember([{ text: 'Our rate is $115 per hour. There is a three-hour minimum.', sig: 'abc' }]);
  const blob = await tts._synth('There is a three-hour minimum.');
  assert.ok(blob instanceof Blob);
  assert.deepEqual(f.calls[0].body, {
    text: 'Our rate is $115 per hour. There is a three-hour minimum.', sig: 'abc',
    part: 'There is a three-hour minimum.',
  });
});

test('the TTS refuses to voice text the server never signed', async () => {
  const f = fakeFetch();
  const tts = new ServerTTS({ speakUrl: '/api/voice/speak', fetchImpl: f.impl });
  tts.remember([{ text: 'Hello there.', sig: 'abc' }]);
  assert.equal(await tts._synth('Something else entirely.'), null);
  assert.equal(f.calls.length, 0);
});

test('a failed synthesis skips the sentence and reports it, rather than hanging the reply', async () => {
  const f = fakeFetch(502);
  const errors = [];
  const tts = new ServerTTS({ speakUrl: '/api/voice/speak', fetchImpl: f.impl, onError: (e) => errors.push(e) });
  tts.remember([{ text: 'Hello there.', sig: 'abc' }]);
  assert.equal(await tts._synth('Hello there.'), null);
  assert.equal(errors.length, 1);
});

test('a one-word answer the speech detector calls a "misfire" still ends the turn', () => {
  STT_PROVIDERS.__fake = () => ({ open() {}, feed() {}, commit() {}, close() {} });
  let agent;
  agent = new VoiceAgent({ sttProvider: '__fake', llm: makeConversationLLM({ page: fakePage(), tts: fakeTts(), lines: LINES }),
    tts: { speak: async () => '' }, vadOptions: vadOptionsFor(() => agent) });
  // If an upgrade renames these, this fails loudly instead of answers silently merging again.
  assert.equal(typeof agent._onSpeechEnd, 'function');
  let ended = 0;
  agent._onSpeechEnd = () => { ended++; };
  agent._speaking = true;
  agent.vadOptions.onVADMisfire();
  assert.equal(ended, 1);
  agent._speaking = false;
  agent.vadOptions.onVADMisfire();
  assert.equal(ended, 1, 'no end without a start');
  delete STT_PROVIDERS.__fake;
});

test('replies are voiced as whole sentences, so prosody holds and repeats hit the cache', async () => {
  assert.deepEqual(splitSentences('And an email address?'), ['And an email address?']);
  assert.deepEqual(splitSentences("First off — what's your name?"), ["First off — what's your name?"]);
  assert.deepEqual(splitSentences('Our rate is $115.50 per hour. There is a 3.5 hour minimum! Fine?'),
    ['Our rate is $115.50 per hour.', 'There is a 3.5 hour minimum!', 'Fine?']);
  const tts = new ServerTTS({ fetchImpl: async () => ({}) });
  const out = [];
  for await (const s of tts._sentences((async function* () { yield 'Hi there. '; yield 'How are you?'; })())) out.push(s);
  assert.deepEqual(out, ['Hi there.', 'How are you?']);
});
