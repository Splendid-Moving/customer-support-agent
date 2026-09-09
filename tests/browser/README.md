# Browser tests

The Python suite next door runs offline and asserts the rules. These drive the
actual page in a real browser and assert what a customer sees — which is where
the last few bugs lived, and none of them were visible from Python:

- the date step answering itself the moment the picker opened, because iOS fires
  `change` as the wheel settles
- the read-back numbered "Question 10 of 10" when it is not a question
- the read-back clipped halfway down the one screen a customer is asked to check

They cost model calls, so they are not part of `pytest`.

## Running them

    # once
    npm i playwright && npx playwright install chromium

    # a server on a throwaway database, with no email sending
    DRY_RUN=true CHECKPOINT_DB=/tmp/t.sqlite \
      .venv/bin/python -m uvicorn app:app --port 8812

    BASE=http://localhost:8812 node tests/browser/interview.js
    BASE=http://localhost:8812 node tests/browser/edges.js
    BASE=http://localhost:8812 node tests/browser/conversation.js

`DRY_RUN=true` matters: these fill in whole leads, and without it every run
emails the office.

| File | What it covers |
|---|---|
| `interview.js` | A whole estimate on a phone-sized screen: one card at a time, the counter, the date step's two taps, choices, skips, photos asked twice, the read-back, sending. |
| `edges.js` | A small phone (375×667), "Something else", a rejected phone number, and the same interview in Russian. |
| `conversation.js` | That an ordinary question still reads as a chat, that an estimate then starts an interview, and that "Start over" leaves nothing behind. |

## Writing more

The one thing to get right is waiting. Two races cost an hour between them:

- **The card renders a beat before the turn ends.** The page ignores input while
  it is mid-turn — that is what stops a double submit — so an `Enter` sent in
  that gap is dropped, and it looks exactly like a hang. Wait for the thinking
  dots to clear and the outgoing card to be gone, not just for the text.
- **The dots disappear on the first streamed token**, so they mark the start of
  an answer, not the end of one. `conversation.js` has an `idle()` that waits
  for the reply to stop changing.
