// What a spoken sentence means, given what Sam is currently asking.
// Run: node --test tests/voice_js/
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { replyFor } from '../../static/voice/spoken_reply.js';

const question = (field, extra = {}) => ({ type: 'question', field, skippable: false, ...extra });
const NAME = { name: 'name', kind: 'text' };
const ACCESS = { name: 'access', kind: 'text' };
const DATE = { name: 'move_date', kind: 'date' };
const NOTES = { name: 'notes', kind: 'textarea' };

test('outside the interview, speech is an ordinary chat message shown in the transcript', () => {
  assert.deepEqual(replyFor(null, 'How much for two movers?'),
    { payload: { message: 'How much for two movers?' }, echo: 'How much for two movers?' });
});

test('during a question, speech is the answer, sent the same way typing is', () => {
  assert.deepEqual(replyFor(question(NAME), 'Nik.'), { payload: { reply: { answer: 'Nik.' } } });
});

test('"skip" skips an optional question', () => {
  for (const said of ['Skip.', 'skip it', 'Skip that one.', 'pass']) {
    assert.deepEqual(replyFor(question(ACCESS, { skippable: true }), said),
      { payload: { reply: { skip: true } } }, said);
  }
});

test('"skip" on a required question is still an answer, so the server can say it needs it', () => {
  assert.deepEqual(replyFor(question(NAME), 'skip'), { payload: { reply: { answer: 'skip' } } });
});

test('"no" to the date or to "anything else?" is left for the server, which already understands it', () => {
  assert.deepEqual(replyFor(question(DATE, { skippable: true }), 'Not sure yet.'),
    { payload: { reply: { answer: 'Not sure yet.' } } });
  assert.deepEqual(replyFor(question(NOTES, { skippable: true }), 'No, that’s it.'),
    { payload: { reply: { answer: 'No, that’s it.' } } });
});

// ── The read-back ────────────────────────────────────────────────────────────

const CONFIRM = { type: 'confirm' };

test('yes at the read-back sends it', () => {
  for (const said of ['Yes.', 'Yeah, send it.', 'Looks good!', 'That’s right.', 'Correct.', 'Go ahead.', 'yep', 'Sure, send it over.']) {
    assert.deepEqual(replyFor(CONFIRM, said), { payload: { reply: { confirmed: true } } }, said);
  }
});

test('start over at the read-back starts over', () => {
  for (const said of ['Start over.', 'Can we start again?', 'Let me redo it.']) {
    assert.deepEqual(replyFor(CONFIRM, said), { payload: { reply: { restart: true } } }, said);
  }
});

test('anything unclear at the read-back sends NOTHING and explains instead', () => {
  // A lead a manager will act on is not sent on a maybe.
  for (const said of ['Hmm.', 'The phone number is wrong.', 'no', 'Yes but the date is wrong']) {
    assert.deepEqual(replyFor(CONFIRM, said), { hint: 'confirm_hint' }, said);
  }
});

// ── Photos ───────────────────────────────────────────────────────────────────

const PHOTOS = { type: 'photos' };

test('saying no to photos skips them', () => {
  for (const said of ['No.', 'Skip.', 'No photos.', 'Not right now.', 'Skip for now.', "I don't have any."]) {
    assert.deepEqual(replyFor(PHOTOS, said), { payload: { reply: { photo_ids: [] } } }, said);
  }
});

test('saying send with photos attached sends them', () => {
  assert.deepEqual(replyFor(PHOTOS, 'Okay, send those.', { photoIds: ['a1', 'b2'] }),
    { payload: { reply: { photo_ids: ['a1', 'b2'] } } });
});

test('yes to photos with none attached explains how to add them rather than skipping', () => {
  assert.deepEqual(replyFor(PHOTOS, 'Yes sure.'), { hint: 'photos_hint' });
  assert.deepEqual(replyFor(PHOTOS, 'Send them.'), { hint: 'photos_hint' });
});

test('anything else at the photo step explains rather than silently skipping', () => {
  assert.deepEqual(replyFor(PHOTOS, 'What kind of photos?'), { hint: 'photos_hint' });
});

test('silence is never an answer', () => {
  assert.equal(replyFor(question(NAME), '   '), null);
  assert.equal(replyFor(null, ''), null);
});
