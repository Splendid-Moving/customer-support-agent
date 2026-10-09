// What a spoken sentence means, given what Sam is asking right now.
//
// Turns one transcript into either the request the page would have sent if the
// customer had typed or tapped it, or — at the two steps a voice can't simply
// answer (the read-back, the photos) — the name of a fixed hint to say instead.
// Pure functions, no DOM; tested in tests/voice_js/spoken_reply.test.mjs.

const words = (text) => ` ${String(text).toLowerCase().replace(/[’']/g, "'").replace(/[^a-z' ]+/g, ' ')} `.replace(/\s+/g, ' ');
const has = (text, phrases) => phrases.some((p) => words(text).includes(` ${p} `));

const SKIP = ['skip', 'skip it', 'skip that', 'skip that one', 'skip this one', 'pass', 'next question'];

// The read-back sends a lead a manager will act on, so a yes has to be a yes.
// Any hedge or correction — "yes but the date is wrong" — is not.
const YES = ['yes', 'yeah', 'yep', 'yup', 'sure', 'correct', "that's right", 'thats right',
  'looks good', 'look good', 'all good', 'go ahead', 'send it', 'send it over', 'perfect',
  'that is right', "that's correct", 'right', 'ok', 'okay', 'confirm', 'sounds good'];
const HEDGE = ['but', 'except', 'wrong', 'change', 'not', 'no', 'wait', 'actually', 'fix', 'mistake'];
const RESTART = ['start over', 'start again', 'redo', 'do it again', 'from the top', 'begin again'];

const NO_PHOTOS = ['no', 'nope', 'skip', 'no photos', 'no pictures', 'not now', 'not right now',
  'skip for now', 'no thanks', "don't have any", 'dont have any', 'later', 'maybe later'];
const SEND_PHOTOS = ['send', 'send them', 'send those', 'send these', 'done', "that's all", 'thats all',
  'yes', 'yeah', 'ok', 'okay', 'go ahead'];

const isYes = (text) => has(text, YES) && !has(text, HEDGE);

/**
 * @param ask       the question on screen (the `ask` from the server), or null in plain chat
 * @param transcript what the customer said
 * @param photoIds  ids of photos already uploaded at the photo step
 * @returns {{payload, echo?} | {hint: string} | null}
 */
export function replyFor(ask, transcript, { photoIds = [] } = {}) {
  const text = String(transcript || '').trim();
  if (!text) return null;

  if (!ask) return { payload: { message: text }, echo: text };

  if (ask.type === 'confirm') {
    if (has(text, RESTART)) return { payload: { reply: { restart: true } } };
    if (isYes(text)) return { payload: { reply: { confirmed: true } } };
    return { hint: 'confirm_hint' };
  }

  if (ask.type === 'photos') {
    if (photoIds.length && has(text, SEND_PHOTOS) && !has(text, ['no', 'nope'])) {
      return { payload: { reply: { photo_ids: [...photoIds] } } };
    }
    if (!photoIds.length && has(text, NO_PHOTOS)) return { payload: { reply: { photo_ids: [] } } };
    return { hint: 'photos_hint' };
  }

  // A question. The date and "anything else?" already understand "not sure" and
  // "no" on the server, and a required field must hear "skip" so it can say it
  // needs the answer — so only an optional, plain field turns "skip" into a skip.
  const kind = ask.field?.kind;
  if (ask.skippable && kind !== 'date' && kind !== 'textarea' && has(text, SKIP)) {
    return { payload: { reply: { skip: true } } };
  }
  return { payload: { reply: { answer: text } } };
}
