# Changelog

## 2026-11-18 — Smoother streaming replies
- Changed: incoming words now fade in gently through a short adaptive buffer, with one persistent breathing cursor instead of rebuilding the reply on every chunk. Added a quiet three-bar thinking wave and subtle send-button press feedback.
- Preserved: approved final text replaces drafts immediately; interview transitions cancel pending text; reduced-motion mode skips pacing. Dropped streams stop their indicator and show a retry message.
- Verified: `pytest -q` 277 passed, 1 deselected; controlled-stream browser checks passed for bursts, cursor continuity, corrections, interruptions, scrolling and reduced motion; real local DRY_RUN conversation checks 9/9 passed. Inspected thinking, streaming and completed replies at 390×844 and 1440×900.

## 2026-11-18 — Official Splendid header logo
- Changed: replaced the typed S avatar with the existing official blue-and-white S image from the operations app, served locally at `/logo.png`.
- Verified: `pytest -q` 277 passed, 1 deselected; local logo response matches the source image byte for byte; inspected mobile (390×844) and desktop (1440×900) screenshots.

## 2026-11-18 — Fewer welcome shortcuts
- Changed: removed “How much do you charge?” and “Is my stuff insured?” from the welcome list, leaving four shortcuts to reduce clutter.
- Verified: `pytest -q` 277 passed, 1 deselected; inspected mobile (390×844) and desktop (1440×900) screenshots. Updated the existing browser reset check to expect four shortcuts.

## 2026-11-18 — Simpler chat welcome
- Changed: replaced the long introduction with “Ask me anything” and one short sentence. Merged estimate, out-of-state and question shortcuts into one “Start here” list; removed the separate card, icon badges and instant-reply status.
- Verified: `pytest -q` 277 passed, 1 deselected; local DRY_RUN browser conversation checks 9/9 passed. Inspected screenshots at 390×844, 320×568 and 1440×900; shortcut rows retain 44px touch targets, the smaller screen scrolls, and the composer stays visible.

## 2026-11-16 — Chat page: flat light theme
- Changed: replaced the glass-and-blue look with a flat, neutral light theme (off-white page, white cards, hairline borders, near-black primary button, Hanken Grotesk, thin Phosphor icons). Replies read as plain text under a name label.
- Changed: the welcome screen now shows one white card with two start rows (estimate, out of state; the estimate marked by a dark icon badge) and a "Common questions" list, instead of a cloud of pills. Same six buttons and messages, so behaviour is unchanged.
- Verified: `pytest` 277 passed; browser tests on a local DRY_RUN server: interview 46/46, edges 14/14, conversation all passed. Screenshots checked at 390×844 and 1440×900.

## 2026-11-16 — Chat page redesign
- Changed: rebuilt the look of `static/index.html`: frosted-glass header, bubbles, question card and message bar over a slow-moving blue backdrop; Geist font in place of the italic serif; every icon from Phosphor; a strong "I'd like an estimate" button; a segmented progress bar on the question card; smoother card and message animations. Page logic unchanged apart from inserting icons.
- Changed: two page checks in `tests/test_wiring.py` updated for the new markup (greeting without `<em>`, chips that contain icons). They still check the same rules.
- Fixed during the work: the read-back card squashed itself instead of scrolling in the new layout; caught by `interview.js`, fixed with `.thread > * { flex-shrink: 0 }`.
- Verified: `pytest` 277 passed; browser tests run one at a time on a local DRY_RUN server: interview 46/46, edges 14/14, conversation 9/9 (same as the run before the change). Screenshots checked at 390×844 and 1440×900.

## 2026-11-16
- Added `scripts/draw_graph.py`, which saves the real LangGraph wiring as a Mermaid diagram in `docs/graph.md` (GitHub renders it). README links to it.
