Patch 010 — BTC-Only Live Readiness Alignment

Purpose:
- Fastest safe path toward live in controlled BTC-only mode
- Preserve future Path B multi-symbol architecture
- Add scanner-side alignment controls and main-side truth for BTC-only readiness

Key scanner envs (safe defaults are OFF):
- BTC_ONLY_ALIGNMENT_ENABLED=1
- SCANNER_FORCE_EMIT_SYMBOLS=BTC/USD
- SCANNER_EMIT_ONLY_SYMBOLS=1

Optional aliases also supported:
- BTC_ONLY_ALIGNMENT_SYMBOLS=BTC/USD
- BTC_ONLY_ALIGNMENT_EMIT_ONLY=1

Main diagnostics added/updated:
- /compatibility -> btc_only_live_alignment block
- /diagnostics/btc_only_live_alignment

Notes:
- This patch does not remove or weaken Path B controls.
- This patch does not change strategy or execution logic.
- Path B future architecture remains intact.
