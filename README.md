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

**Collects estimates.** Ask for a quote, or mention a move out of state, and a
form slides up in the chat: contact details, addresses, size of the place, and
photos of what's moving. That goes straight to the office inbox with the photos
attached, and a manager takes it from there.

**Ignores anyone trying to reprogram it.** "Ignore all previous instructions" and
its many cousins get a polite line about moving and nothing else.

---

## The one thing to understand

**The form is a genuine pause.**

When the agent shows the estimate form, the conversation *stops*. Not "waits" —
the run ends, the state is written to disk, and the server is free. When the
customer submits, minutes later, it picks up on the exact line it stopped at with
the whole conversation intact.

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
        ├─► collect_lead  ⏸ SHOWS THE FORM ──► submit_lead ► END
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
| `knowledge/` | **The facts the agent may state.** Plain markdown. Edit these to change what it knows — no rebuild, no restart. Start with `knowledge/README.md`. |
| `schemas/persona.py` | Who it is and how it talks. Change this to change the voice. |
| `schemas/lead_form.py` | What the estimate form asks and what counts as a valid answer. One definition; the browser, the validator and the email all read it. |
| `agent/` | The graph. `state.py` is the shared memory, `graph.py` wires it together, `nodes/` is one file per step. |
| `services/` | The outside world — the knowledge loader, Resend, photo storage. Nothing here knows the agent exists. |
| `static/index.html` | The chat page. One file, no build step. |
| `tests/` | 140 checks that run in under a second, with no API key and no network. |

Entry points:

| File | Purpose |
|---|---|
| `server.py` | `python server.py` — local, on localhost:8080 |
| `app.py` | What Railway runs. Also where the resume protocol lives. |
| `langgraph.json` | For `langgraph dev`, if you want the LangGraph Studio view |

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

```bash
pytest                    # 140 checks, no API key needed
```

---

## Changing what it knows

Edit or add a markdown file in `knowledge/`. That's the whole process — the next
message picks it up.

The agent can only say what is in there. If a customer asks something the files
don't cover, it will tell them it doesn't know and offer a follow-up, which is
the correct behaviour and also your signal that something is missing from the
knowledge base. `knowledge/README.md` explains how to write it well.

---

## Deploying

One Railway service.

1. **Add a volume** and point `CHECKPOINT_DB` and `UPLOAD_DIR` at it
   (`/data/support_threads.sqlite` and `/data/uploads`). Without a volume, every
   deploy throws away whatever forms people are midway through filling in — and
   they find out by clicking Submit.
2. Set the variables from `.env.example`. `RESEND_FROM` must be on a domain
   verified in Resend or every send fails with a 403.
3. Leave `DRY_RUN=true` for the first deploy. Send yourself a test lead, read it,
   *then* set it to false.
4. Link to it from the website.

`numReplicas` stays at 1. The checkpointer is SQLite on one volume and the rate
limiter is in memory; neither works across two instances.

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
