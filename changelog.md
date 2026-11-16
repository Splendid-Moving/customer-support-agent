# Changelog

## 2026-11-16 — Chat page redesign
- Changed: rebuilt the look of `static/index.html`: frosted-glass header, bubbles, question card and message bar over a slow-moving blue backdrop; Geist font in place of the italic serif; every icon from Phosphor; a strong "I'd like an estimate" button; a segmented progress bar on the question card; smoother card and message animations. Page logic unchanged apart from inserting icons.
- Changed: two page checks in `tests/test_wiring.py` updated for the new markup (greeting without `<em>`, chips that contain icons). They still check the same rules.
- Fixed during the work: the read-back card squashed itself instead of scrolling in the new layout; caught by `interview.js`, fixed with `.thread > * { flex-shrink: 0 }`.
- Verified: `pytest` 277 passed; browser tests run one at a time on a local DRY_RUN server: interview 46/46, edges 14/14, conversation 9/9 (same as the run before the change). Screenshots checked at 390×844 and 1440×900.

## 2026-11-16
- Added `scripts/draw_graph.py`, which saves the real LangGraph wiring as a Mermaid diagram in `docs/graph.md` (GitHub renders it). README links to it.
