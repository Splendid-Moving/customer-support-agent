#!/usr/bin/env python
"""
Draws the agent's graph and saves it to docs/graph.md.

    python scripts/draw_graph.py

It asks the compiled LangGraph graph to describe itself as a Mermaid diagram,
so the picture always matches the real wiring in agent/graph.py. GitHub renders
the diagram automatically. Re-run after changing the graph, then commit both.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
logging.disable(logging.INFO)

from agent.graph import graph  # noqa: E402

OUTPUT_FILE = PROJECT_ROOT / "docs" / "graph.md"

HEADER = """# Support agent graph

Generated from `agent/graph.py` by `scripts/draw_graph.py`. Do not edit by hand.

- **Solid arrow**: always goes to the next step.
- **Dotted arrow**: a decision. The step picks one of the routes.
- `guard` is the only way in, so every message is checked before anything else runs.
- `collect_lead` pauses after each question and waits for the customer's answer.
- `answer_check` can send a draft back to `knowledge` for a rewrite (a capped loop).

"""


def main() -> None:
    mermaid_diagram = graph.get_graph().draw_mermaid()
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text(HEADER + "```mermaid\n" + mermaid_diagram + "```\n")
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
