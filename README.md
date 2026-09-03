# Splendid Moving — customer support agent

The chat a customer lands on from splendidmoving.com. It answers their questions
about us, and when they want a real price it collects the details and emails them
to the office so a manager can call them back.

It is the customer-facing counterpart to [`../ops-agent`](../ops-agent), which is
the internal one the team talks to in Google Chat. They share no code and no
credentials on purpose: this one is on the public internet and can do exactly one
thing to the outside world — send one email.

---

## What it does

**Answers questions, in our voice, from our material only.**

> *how much for 2 movers?*
> Our rate for two movers and a truck is $115 to $125 per hour, with a three-hour
> minimum.

It answers as a member of the team — "we", "our trucks" — never "Splendid Moving
offers…". And it answers **only** from the files in `knowledge/`. Ask it something
that isn't in there and it says so and offers to have someone follow up, rather
than inventing a plausible answer.

**Collects estimates, by asking.** Ask for a quote, or mention a move out of
state, and the agent just keeps talking — one question at a time, contact
details through to photos of what's moving. Anything the customer already
mentioned is skipped rather than asked again. At the end it emails the office
with the photos attached, and a manager takes it from there.

**Ignores anyone trying to reprogram it.** "Ignore all previous instructions" and
its many cousins get a polite line about moving and nothing else.

---

## The one thing to understand

**Every question is a genuine pause.**

When the agent asks for a phone number, the conversation *stops*. Not "waits" —
the run ends, the state is written to disk, and the server is free. When the
answer arrives, minutes later, it picks up on the exact line it stopped at with
the whole conversation intact. Nine questions is nine of those.

That is what LangGraph is for, and it is the reason this is a graph rather than a
script. It is also the part that is easiest to break: input arriving while the
graph is paused has to be sent back as a *resume*, not as a new message. Send the
wrong one and it silently starts over and throws the form away. That decision is
made in exactly one place — `_graph_input` in `app.py` — and nothing else in the
codebase needs to know about it.

---

## How it works

```
START
  │
  ▼
guard ──── blocked ───► refuse ────────────────────────► END
  │ clean
  ▼
router ─┬─► knowledge ──► answer_check ────────────────► END
        │        ▲              │
        │        └── rewrite ───┘
        │
        ├─► prefill ──► collect_lead ──► submit_lead ──► END
        │                    ⏸
        │              pauses once per question
        │
        └─► handoff ───────────────────────────────────► END
```

**The guard is the only way in.** Every message passes through it before anything
else runs — not a check inside each lane, one gate in front of all of them, so a
lane added later cannot quietly skip it.

**The email is the last node of the longest path.** `submit_lead` is reachable
only by passing the guard, being routed to the form, and filling that form in by
hand. There is no sequence of words a customer can type that reaches it, which is
what makes this safe to leave running on a public URL.

---

## Layout

| Path | What lives there |
|---|---|
| `knowledge/` | **The facts the agent may state.** Generated from `knowledge/source/splendid_moving_kb.xlsx` — edit the spreadsheet, run `python scripts/import_kb.py`. Start with `knowledge/README.md`. |
| `schemas/persona.py` | Who it is and how it talks. Change this to change the voice. |
| `schemas/lead_form.py` | What the agent asks for, how it phrases each question, and what counts as a valid answer. One definition; the interview, the validator and the email all read it. |
| `agent/` | The graph. `state.py` is the shared memory, `graph.py` wires it together, `nodes/` is one file per step. |
| `services/` | The outside world — the knowledge loader, Resend, photo storage. Nothing here knows the agent exists. |
| `static/index.html` | The chat page. One file, no build step. |
| `scripts/import_kb.py` | Turns the KB spreadsheet into the markdown the agent reads. |
| `tests/` | 145 checks that run in under a second, with no API key and no network. |

Entry points:

| File | Purpose |
|---|---|
| `server.py` | `python server.py` — local, on localhost:8080 |
| `app.py` | What Railway runs. Also where the resume protocol lives. |
| `langgraph.json` | For `langgraph dev`, if you want the LangGraph Studio view |
| `.python-version` | Pins Python 3.12 for Railway's builder. The pinned dependency versions are not all built for 3.13. |

---

## Running it locally

```bash
uv venv --python 3.12
uv pip install -r requirements.txt pytest httpx

cp .env.example .env      # then fill in OPENAI_API_KEY
python server.py          # http://localhost:8080
```

`DRY_RUN` defaults to **true**, which means the lead email is printed to the
terminal in full and never sent. Read one before you turn it off.

**Keep it true in your local `.env`.** `DRY_RUN=false` belongs on Railway and
nowhere else — with it off locally, walking through the estimate to check a
wording change puts a real lead in the office inbox, and it looks exactly like a
customer's.

```bash
pytest                    # 145 checks, no API key needed
```

---

## Changing what it knows

The knowledge base is one spreadsheet: `knowledge/source/splendid_moving_kb.xlsx`.

```bash
# 1. edit the spreadsheet
# 2. regenerate the markdown the agent reads
python scripts/import_kb.py
# 3. commit both
```

The importer refuses duplicate ids, and prints any row whose `notes` say
"confirm" so an unresolved answer does not go live by accident.

The spreadsheet is the editing surface because the office can add a row without
touching git. The markdown is what ships because a spreadsheet is a binary file —
a commit that quietly changes the hourly rate would show up in git as "the file
changed" and nothing more, with nobody able to review what the agent is about to
start telling customers.

`company.md` is the one hand-written file, holding background the spreadsheet
doesn't cover (hours, service area, what comes with the crew). The importer
leaves it alone.

The agent can only say what is in `knowledge/`. If a customer asks something it
doesn't cover, it says so and offers a follow-up — which is correct behaviour and
also your signal that a row is missing. `knowledge/README.md` explains how to
write a good one.

**The `notes` column is operating knowledge, not a comment.** "Ask how many
movers before quoting" and "escalate the exact amount to the office" are carried
into the prompt as `HANDLING:` lines. The agent follows them and never repeats
them to the customer.

---

## Deploying

One Railway service, built by Railpack, which picks up `Procfile` and
`.python-version` on its own.

**There is deliberately no `railway.json`.** Railway deprecated Config as Code,
and services created after 2026-08-28 cannot opt in at all — a config file
committed here would look authoritative and be silently ignored. These settings
have to be set in the Railway dashboard by hand:

| Setting | Value | Why |
|---|---|---|
| Volume mount path | `/data` | Added from the project canvas (⌘K → "volume"), **not** from service Settings |
| Healthcheck Path | `/health` | Stops traffic being routed to a deploy that failed to boot |
| Replicas | `1` | The checkpointer is SQLite on one volume and the rate limiter is in memory. Neither works across two instances. |
| Start Command | (blank) | Railpack reads `Procfile`. Set it explicitly to `uvicorn app:app --host 0.0.0.0 --port $PORT` if you would rather not rely on that. |

Then:

1. **Add the volume first** and point `CHECKPOINT_DB` and `UPLOAD_DIR` inside it
   (`/data/support_threads.sqlite` and `/data/uploads`). Without a volume, every
   deploy throws away whatever forms people are midway through filling in — and
   they find out by clicking Submit.
2. Set the variables from `.env.example`. `RESEND_FROM` must be on a domain
   verified in Resend or every send fails with a 403.
3. Leave `DRY_RUN=true` for the first deploy. Send yourself a test lead, read it,
   *then* set it to false.
4. Add the custom domain, and link to it from the website.

---

## Things worth knowing before you change something

**Voice rules are enforced in Python, not just in the prompt.**
`agent/nodes/answer_check.py` rejects a draft that slips into the third person,
or that quotes a dollar figure not present in `knowledge/`. The draft goes back
once with a specific complaint; a second failure sends the phone number instead.
Prompts alone leaked about one message in twenty.

**Photos never enter the conversation.** They upload separately to `/api/upload`
and only their ids travel through the graph. LangGraph writes the whole state to
disk after every step, so images in state would be re-saved on every turn and the
database would balloon.

**Answers stream before they are approved.** The knowledge node's draft is sent
to the browser token by token, and `answer_check` only finishes afterwards. If a
draft is rejected, the `done` event carries corrected text and the browser swaps
it in. In the common case they are identical and nothing is swapped.

**The interview is Python, not a model.** `collect_lead` walks a fixed list of
fields. A model-run interview asks better questions and can loop forever, and
whatever it decides a field contains lands in an email a manager acts on. The one
model call in that lane is `prefill`, which only fills fields the customer can
see are wrong — never name, phone or email.

**Refusals are fixed strings.** Asking the model to write a polite refusal hands
the attacker their text in the model's context, which is the thing being defended
against. `agent/nodes/refuse.py` has no model call.

**The agent never claims it did something.** It cannot book, reschedule, look
anything up, or pass a message to anybody. `handoff` gives the phone number
rather than saying "I've escalated this", because a customer who believes that
waits for a call that is not coming.

---

## What it deliberately cannot do

- Book, change or cancel a job
- Look up an existing booking, invoice or crew
- Quote a total price for a move — only the published hourly rates
- Reply to anyone by email
- Reach GoHighLevel or the calendar at all

All of that lives in `ops-agent`, behind the team.
