# Knowledge base

Everything the agent is allowed to say about us lives in this folder. If a fact
isn't in here, the agent will tell the customer it doesn't know and offer to have
someone from the office follow up — it will **not** guess.

## How to edit it

Drop in plain markdown files. Any `.md` file in this folder is loaded; this
`README.md` is skipped. Save the file and the next message picks up the change —
nothing to rebuild, no restart, no re-indexing.

## How to write it

Write it the way you'd brief a new employee on their first day.

- **Facts, not marketing.** "Two movers and a truck is $115–$125/hr, 3 hour
  minimum" is useful. "We pride ourselves on excellence" is not — the agent will
  repeat it, and it sounds like a brochure.
- **Say the awkward things too.** "We can take a TV off the wall but we don't
  mount it at the new place." Half the questions customers ask are about the
  limits, and if the answer isn't here the agent has to punt to a human.
- **Numbers exactly as you want them repeated.** The agent quotes rates verbatim
  from this file. It is never allowed to add them up into a total.
- **One topic per heading.** Short `##` sections beat long paragraphs.
- **Don't put instructions to the agent in here.** This folder is facts. How the
  agent behaves lives in `schemas/persona.py`.

## Files

| File | What belongs in it |
|---|---|
| `company.md` | Who we are, hours, service area, rates, what's included |
| `faq.md` | The questions customers actually ask, with our real answers |

Add more files freely — split by topic once one gets long.

> **Note:** the files here now were seeded from `main_website/BRAND_INFO.md` so
> the agent had something to work with on day one. Replace them with the real
> knowledge base when it's ready.
