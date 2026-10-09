// Sam's voice: voiceloop's streaming TTS, synthesized by our server.
//
// voiceloop splits a reply into sentences and asks for one clip at a time,
// playing sentence one while sentence two is made. Each request carries the
// whole server-signed text the sentence came from, because /api/voice/speak only
// voices part of something the server signed. A sentence we hold no signature
// for is simply not spoken. Tested in tests/voice_js/voice_mode.test.mjs.

import { StreamingTTS } from './vendor/voiceloop/index.js';

const squash = (s) => String(s || '').replace(/\s+/g, ' ').trim();

/** Whole sentences, split after . ! ? or … — never inside "$1.50" or "3.5". */
export function splitSentences(text) {
  return squash(text).split(/(?<=[.!?…])\s+(?=\S)/).filter(Boolean);
}

export class ServerTTS extends StreamingTTS {
  constructor({ speakUrl = '/api/voice/speak', fetchImpl, onError } = {}) {
    super();
    this.speakUrl = speakUrl;
    this.fetchImpl = fetchImpl || ((...a) => fetch(...a));
    this.onError = onError || (() => {});
    this.signed = [];   // recent {text, sig} the server handed us, newest last
  }

  /** Keep the signatures for lines about to be spoken. Older ones age out. */
  remember(utterances) {
    for (const u of utterances || []) if (u?.text && u?.sig) this.signed.push(u);
    if (this.signed.length > 40) this.signed.splice(0, this.signed.length - 40);
  }

  // voiceloop's own splitter cuts the first word or two off as a separate clip —
  // right for a voice synthesized in the browser, where a tiny clip is instant.
  // Ours is a network round trip per clip, so a two-word opener only adds a gap
  // and breaks the intonation. Whole sentences sound like a person, and they are
  // the same sentences every time, which is what lets the server cache them.
  async *_sentences(src, signal) {
    let text = '';
    for await (const piece of src) {
      if (signal?.aborted) return;
      text += piece;
    }
    yield* splitSentences(text);
  }

  async _synth(text, signal) {
    const part = squash(text);
    if (!part) return null;
    const source = [...this.signed].reverse().find((u) => squash(u.text).includes(part));
    if (!source) return null;
    try {
      const res = await this.fetchImpl(this.speakUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: source.text, sig: source.sig, part }),
        signal,
      });
      if (!res.ok) throw new Error(`speak ${res.status}`);
      return await res.blob();
    } catch (e) {
      if (e?.name === 'AbortError') return null;   // interrupted: not an error
      this.onError(e);
      return null;   // skip this sentence; the text is on screen regardless
    }
  }
}
