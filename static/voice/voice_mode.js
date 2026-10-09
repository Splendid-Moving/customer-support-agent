// Voice mode: voiceloop wired to the existing chat page.
//
// voiceloop runs the audio loop (mic, speech detection, interruptions, playback).
// Where it would normally call an LLM, it calls the page's own conversation —
// the same /api/chat turn a typed message makes — and speaks the signed reply
// that comes back. The page draws everything exactly as it does for typing;
// this file only decides what to send and what to say. DOM-free on purpose:
// the page passes in callbacks. Tested in tests/voice_js/voice_mode.test.mjs.

import { VoiceAgent, STT_PROVIDERS, unlockAudio, prewarmVoice } from './vendor/voiceloop/index.js';
import { makeOpenAILiveSTT } from './openai_live_stt.js';
import { ServerTTS } from './server_tts.js';
import { replyFor } from './spoken_reply.js';

STT_PROVIDERS.openai = makeOpenAILiveSTT;

const IDLE_MS = 2 * 60 * 1000;   // no speech for this long → hang up and free the mic

/**
 * The "LLM" voiceloop calls for each spoken turn.
 *
 * Three rules that keep the interview safe, each pinned by a test:
 *  - Turns reach the server one at a time, in order. The interview's pause/resume
 *    breaks silently if two answers overlap.
 *  - An interruption stops the VOICE, never the request. The server finishes the
 *    turn and the page shows it; Sam just doesn't read it out.
 *  - `executesTools` tells voiceloop not to guess at half-finished sentences
 *    (its "prefetch"). A guess here would be a real answer to a real question.
 */
export function makeConversationLLM({ page, tts, lines, onTurnError = () => {} }) {
  let queue = Promise.resolve();

  const llm = async function* (history, _system, signal) {
    const last = history[history.length - 1];
    const said = typeof last?.content === 'string' ? last.content : '';

    // Decided only once the previous turn is finished: what Sam is "asking" is
    // whatever that turn left on screen.
    const run = queue.then(async () => {
      const decided = replyFor(page.asking(), said, { photoIds: page.photoIds() });
      if (!decided) return [];
      if (decided.hint) return [lines[decided.hint]].filter(Boolean);
      try {
        const evt = await page.send(decided.payload, decided.echo);
        return evt?.speech || [];
      } catch (e) {
        onTurnError(e);
        return [lines.not_caught].filter(Boolean);
      }
    });
    queue = run.catch(() => {});

    const speech = await run;
    if (signal?.aborted) return;
    tts.remember(speech);
    for (const u of speech) yield { text: `${u.text} ` };
  };
  llm.executesTools = true;
  return llm;
}

/**
 * Speech-detector settings voiceloop passes straight through to Silero.
 *
 * Silero calls a very short utterance — "Yes.", "Skip.", "No." — a "misfire"
 * instead of speech that ended. voiceloop 0.1.25 only listens for the ending,
 * so after a misfire it believes the customer is still talking and glues their
 * one-word answer onto whatever they say next ("Skip. No, that's it."). In an
 * interview made of short answers that is most turns, so a misfire is treated
 * as the end of speech it is. A real blip (a cough) transcribes to nothing and
 * closes empty, which voiceloop already ignores.
 */
export function vadOptionsFor(getAgent) {
  return {
    onVADMisfire: () => {
      const agent = getAgent();
      if (agent && agent._speaking) agent._onSpeechEnd();
    },
  };
}

/** Turn a browser/voiceloop error into a sentence for the customer. */
export function explain(error) {
  const msg = String(error || '');
  if (/NotAllowed|denied|Permission/i.test(msg)) {
    return 'Your browser blocked the microphone. Allow it from the address bar to talk — or just keep typing.';
  }
  if (/NotFound|Requested device not found|no audio input/i.test(msg)) {
    return "I can't find a microphone on this device — typing works just the same.";
  }
  if (/^Voice /.test(msg)) return msg;   // our own wording, from the transcription provider
  return "Voice stopped working for a moment — tap the mic to try again, or keep typing.";
}

/**
 * The 🎤 button's brain.
 *
 * page: { asking(), photoIds(), send(payload, echo), hasConversation() }
 * ui:   { onState(state), onTranscript(text, final), onLevel(0..1), onStart(), onStop(reason, message) }
 */
export function createVoiceMode({ page, ui, config, Agent = VoiceAgent }) {
  let agent = null, tts = null, idleTimer = null;
  let active = false;

  const bumpIdle = () => {
    clearTimeout(idleTimer);
    if (active) idleTimer = setTimeout(() => stop('idle'), IDLE_MS);
  };

  const onEvent = (e) => {
    switch (e.type) {
      case 'state':
        ui.onState(e.state);
        if (e.state === 'speaking') bumpIdle();
        break;
      case 'stt':
        ui.onTranscript([e.committed, e.text].filter(Boolean).join(' ').trim(), e.turnComplete);
        bumpIdle();
        break;
      case 'vad':
        if (e.active) bumpIdle();
        break;
      case 'level':
        ui.onLevel?.(e.level);
        break;
      case 'error':
        // voiceloop stops itself BEFORE reporting anything it can't recover from
        // (mic refused, transcription gone), so `closed` tells the two apart.
        if (agent?.closed) stop('error', explain(e.error));
        else console.warn('[voice]', e.error);
        break;
      default:
        break;
    }
  };

  function say(speech) {
    if (!active || !speech?.length) return;
    tts.remember(speech);
    agent.replay(speech.map((u) => u.text).join(' '));
  }

  async function start() {
    // Must run inside the tap: iOS only lets a page play sound it was tapped into.
    unlockAudio();
    if (active) return true;
    if (!config?.enabled) {
      ui.onStop('error', "Voice isn't available right now — typing still works.");
      return false;
    }
    tts = new ServerTTS({ onError: (err) => console.warn('[voice] speech', err) });
    const llm = makeConversationLLM({
      page, tts, lines: config.lines,
      onTurnError: (err) => console.warn('[voice] turn', err),
    });
    agent = new Agent({ sttProvider: 'openai', llm, tts, onEvent, vadOptions: vadOptionsFor(() => agent) });
    active = true;
    ui.onStart();
    await agent.start();
    if (!active) return false;   // the mic was refused while starting
    say([page.hasConversation() ? config.lines.resume : config.lines.greeting]);
    bumpIdle();
    return true;
  }

  function stop(reason = 'user', message = '') {
    clearTimeout(idleTimer);
    const was = agent;
    agent = null;
    active = false;
    // destroy, not stop: releases the mic so the browser's recording light goes off.
    try { was?.destroy(); } catch (err) { console.warn('[voice] stop', err); }
    ui.onStop(reason, message);
  }

  return {
    start,
    stop,
    /** Speak the reply to something the customer typed or tapped while voice is on. */
    say,
    get active() { return active; },
    /** Fetch the speech detector while the customer is still reaching for the button. */
    prewarm: () => prewarmVoice(),
  };
}
