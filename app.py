from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# ✅ root-level imports (repo is flat)
from kraken_spot import list_spot_pairs, ticker_24h, ohlc_close_series, best_bid_ask
from scoring import compute_atr, score_spot, score_futures_bonus

FUTURES_ENABLED = os.getenv("FUTURES_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
if FUTURES_ENABLED:
    from kraken_futures import futures_snapshot

REFRESH_SEC = int(os.getenv("SCAN_REFRESH_SEC", "300") or 300)  # 5m
TOP_N = int(os.getenv("TOP_N", "5") or 5)
QUOTE_ALLOW = [q.strip().upper() for q in os.getenv("QUOTE_ALLOW", "USD,USDT,USDC").split(",") if q.strip()]
MIN_24H_USD_VOL = float(os.getenv("MIN_24H_USD_VOL", "2500000") or 2500000)  # $2.5m
MIN_24H_RANGE_PCT = float(os.getenv("MIN_24H_RANGE_PCT", "0.05") or 0.05)     # 5%
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.004") or 0.004)         # 0.40%
MAX_PAIRS = int(os.getenv("MAX_PAIRS", "250") or 250)

# NEW: de-dupe by base (avoid ETH/USDT + ETH/USDC consuming two slots)
DEDUP_BY_BASE = os.getenv("DEDUP_BY_BASE", "1").strip().lower() in ("1", "true", "yes", "on")

# NEW: ensure we always return TOP_N (fill from best remaining candidates)
FILL_TO_TOP_N = os.getenv("FILL_TO_TOP_N", "1").strip().lower() in ("1", "true", "yes", "on")

app = FastAPI(title="Crypto Scanner", version="1.0.2")

CACHE: Dict[str, Any] = {
    "ts": None,              # epoch seconds
    "active_symbols": [],
    "reasons": {},
    "scores": {},
    "last_error": None,
    "raw": None,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def should_refresh() -> bool:
    ts = CACHE["ts"]
    if ts is None:
        return True
    return (time.time() - ts) >= REFRESH_SEC


def _base(sym: str) -> str:
    return sym.split("/", 1)[0].upper()


def refresh() -> None:
    pairs = list_spot_pairs(quotes=QUOTE_ALLOW, limit=MAX_PAIRS)
    tick = ticker_24h(pairs)

    fut = None
    if FUTURES_ENABLED:
        try:
            fut = futures_snapshot()
        except Exception:
            fut = None

    scored: List[Tuple[str, float, List[str], float, float]] = []
    # tuple: (sym, total_score, reasons, vol_usd, range_pct)

    for sym in pairs:
        t = tick.get(sym)
        if not t:
            continue

        usd_vol = float(t["vol_usd"])
        rng = float(t["range_pct"])

        if usd_vol < MIN_24H_USD_VOL:
            continue
        if rng < MIN_24H_RANGE_PCT:
            continue

        # Spread sanity
        spread_pct = None
        try:
            bid, ask = best_bid_ask(sym)
            mid = (bid + ask) / 2.0
            spread_pct = (ask - bid) / mid if mid > 0 else 1.0
            if spread_pct > MAX_SPREAD_PCT:
                continue
        except Exception:
            spread_pct = None

        # ATR activity proxy
        closes = ohlc_close_series(sym, interval_min=15, bars=80)
        atr = compute_atr(closes, period=14)

        spot_score, spot_reasons = score_spot(t, atr=atr, spread_pct=spread_pct)

        bonus = 0.0
        bonus_reasons: List[str] = []
        if fut is not None:
            bonus, bonus_reasons = score_futures_bonus(sym, fut)

        total = float(spot_score + bonus)
        if total <= 0:
            continue

        reasons = spot_reasons + bonus_reasons
        scored.append((sym, total, reasons, usd_vol, rng))

    pre_dedup_count = len(scored)

    # ✅ NEW: de-dupe by base (keep best quote per base)
    if DEDUP_BY_BASE and scored:
        best_by_base: Dict[str, Tuple[str, float, List[str], float, float]] = {}
        for sym, total, reasons, usd_vol, rng in scored:
            b = _base(sym)
            prev = best_by_base.get(b)
            if prev is None:
                best_by_base[b] = (sym, total, reasons, usd_vol, rng)
                continue

            prev_sym, prev_total, prev_reasons, prev_vol, prev_rng = prev

            # prefer higher score
            if total > prev_total:
                best_by_base[b] = (sym, total, reasons, usd_vol, rng)
            # tie-breaker: higher volume
            elif total == prev_total and usd_vol > prev_vol:
                best_by_base[b] = (sym, total, reasons, usd_vol, rng)
            # next tie-breaker: higher range
            elif total == prev_total and usd_vol == prev_vol and rng > prev_rng:
                best_by_base[b] = (sym, total, reasons, usd_vol, rng)

        scored = list(best_by_base.values())

    post_dedup_count = len(scored)

    # Sort by score, then by volume, then by range
    scored.sort(key=lambda x: (x[1], x[3], x[4]), reverse=True)

    top = scored[:TOP_N]

    # ✅ NEW: fill to TOP_N if we have fewer than TOP_N
    if FILL_TO_TOP_N and len(top) < TOP_N and scored:
        selected = {s for (s, _, _, _, _) in top}
        remainder = [x for x in scored if x[0] not in selected]
        remainder.sort(key=lambda x: (x[1], x[3], x[4]), reverse=True)
        top = top + remainder[: (TOP_N - len(top))]

    CACHE["ts"] = time.time()
    CACHE["active_symbols"] = [s for (s, _, _, _, _) in top]
    CACHE["scores"] = {s: float(sc) for (s, sc, _, _, _) in top}
    CACHE["reasons"] = {s: rs for (s, _, rs, _, _) in top}
    CACHE["last_error"] = None
    CACHE["raw"] = {
        "universe": len(pairs),
        "scored_prefilter": pre_dedup_count,
        "scored_postdedup": post_dedup_count,
        "returned": len(top),
        "dedup_by_base": DEDUP_BY_BASE,
        "fill_to_top_n": FILL_TO_TOP_N,
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "utc": utc_now_iso(),
        "futures_enabled": FUTURES_ENABLED,
        "refresh_sec": REFRESH_SEC,
        "quote_allow": QUOTE_ALLOW,
        "dedup_by_base": DEDUP_BY_BASE,
        "fill_to_top_n": FILL_TO_TOP_N,
        "last_error": CACHE["last_error"],
    }


@app.get("/active_coins")
def active_coins():
    if should_refresh():
        try:
            refresh()
        except Exception as e:
            CACHE["ts"] = time.time()
            CACHE["last_error"] = str(e)
            return JSONResponse(
                status_code=200,
                content={
                    "ok": False,
                    "utc": utc_now_iso(),
                    "error": str(e),
                    "active_symbols": CACHE.get("active_symbols", []),
                },
            )

    return {
        "ok": True,
        "utc": utc_now_iso(),
        "active_symbols": CACHE["active_symbols"],
        "scores": CACHE["scores"],
        "reasons": CACHE["reasons"],
        "meta": CACHE["raw"],
        "refresh_sec": REFRESH_SEC,
        "quote_allow": QUOTE_ALLOW,
        "futures_enabled": FUTURES_ENABLED,
        "last_error": CACHE["last_error"],
    }
