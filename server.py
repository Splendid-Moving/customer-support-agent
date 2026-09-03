#!/usr/bin/env python
"""
Local dev runner.

    python server.py    ->  http://localhost:8080

Binds to localhost, which is the only protection it needs — the deployed app is
`app.py` behind Railway. The banner exists so you cannot start it in live mode by
accident and find out afterwards.
"""

import uvicorn

from app import app
from services import config, knowledge, tracing

if __name__ == "__main__":
    mode = (
        "DRY RUN — the lead email is logged, never sent"
        if config.dry_run()
        else "*** LIVE — lead emails go to the office inbox ***"
    )
    print("\n  Splendid Moving — customer chat")
    print(f"  {mode}")
    print(f"  {knowledge.describe()}")
    print(f"  {tracing.configure()}")
    print("  http://localhost:8080\n")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")
