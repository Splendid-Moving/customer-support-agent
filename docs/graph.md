# Support agent graph

Generated from `agent/graph.py` by `scripts/draw_graph.py`. Do not edit by hand.

- **Solid arrow**: always goes to the next step.
- **Dotted arrow**: a decision. The step picks one of the routes.
- `guard` is the only way in, so every message is checked before anything else runs.
- `collect_lead` pauses after each question and waits for the customer's answer.
- `answer_check` can send a draft back to `knowledge` for a rewrite (a capped loop).

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	guard(guard)
	refuse(refuse)
	router(router)
	knowledge(knowledge)
	answer_check(answer_check)
	prefill(prefill)
	translate(translate)
	collect_lead(collect_lead)
	submit_lead(submit_lead)
	handoff(handoff)
	__end__([<p>__end__</p>]):::last
	__start__ --> guard;
	answer_check -.-> __end__;
	answer_check -.-> knowledge;
	collect_lead -.-> __end__;
	collect_lead -.-> submit_lead;
	guard -.-> refuse;
	guard -.-> router;
	knowledge --> answer_check;
	prefill --> translate;
	router -.-> handoff;
	router -.-> knowledge;
	router -.-> prefill;
	translate --> collect_lead;
	handoff --> __end__;
	refuse --> __end__;
	submit_lead --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
