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

# Refresh + selection
REFRESH_SEC = int(os.getenv("SCAN_REFRESH_SEC", "300") or 300)  # 5m
TOP_N = int(os.getenv("TOP_N", "5") or 5)
QUOTE_ALLOW = [q.strip().upper() for q in os.getenv("QUOTE_ALLOW", "USD,USDT,USDC").split(",") if q.strip()]
MAX_PAIRS = int(os.getenv("MAX_PAIRS", "250") or 250)

# Strict "IN PLAY" thresholds
MIN_24H_USD_VOL = float(os.getenv("MIN_24H_USD_VOL", "2500000") or 2500000)  # $2.5m
MIN_24H_RANGE_PCT = float(os.getenv("MIN_24H_RANGE_PCT", "0.05") or 0.05)     # 5%

# Spread sanity
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.004") or 0.004)         # 0.40%

# De-dupe and fill behavior
DEDUP_BY_BASE = os.getenv("DEDUP_BY_BASE", "1").strip().lower() in ("1", "true", "yes", "on")
FILL_TO_TOP_N = os.getenv("FILL_TO_TOP_N", "1").strip().lower() in ("1", "true", "yes", "on")

# Fallback thresholds (used ONLY to fill to TOP_N)
# These are intentionally looser.
FALLBACK_MIN_24H_USD_VOL = float(os.getenv("FALLBACK_MIN_24H_USD_VOL", "750000") or 750000)  # $0.75m
FALLBACK_MIN_24H_RANGE_PCT = float(os.getenv("FALLBACK_MIN_24H_RANGE_PCT", "0.03") or 0.03)  # 3%

app = FastAPI(title="Crypto Scanner", version="1.0.3")

CACHE: Dict[str, Any] = {
    "ts": None,
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


def _spread_ok(sym: str) -> Tuple[bool, float | None]:
    """
    Returns (ok, spread_pct_or_none).
    If the book call fails, we treat spread as unknown (None) and allow it.
    """
    try:
        bid, ask = best_bid_ask(sym)
        mid = (bid + ask) / 2.0
        spread_pct = (ask - bid) / mid if mid > 0 else 1.0
        return (spread_pct <= MAX_SPREAD_PCT), spread_pct
    except Exception:
        return True, None


def _atr_for(sym: str) -> float:
    closes = ohlc_close_series(sym, interval_min=15, bars=80)
    return compute_atr(closes, period=14)


def refresh() -> None:
    pairs = list_spot_pairs(quotes=QUOTE_ALLOW, limit=MAX_PAIRS)
    tick = ticker_24h(pairs)

    fut = None
    if FUTURES_ENABLED:
        try:
            fut = futures_snapshot()
        except Exception:
            fut = None

    # Pools:
    # - in_play: strict thresholds
    # - fallback: looser thresholds, used only to fill up to TOP_N
    in_play: List[Tuple[str, float, List[str], float, float]] = []
    fallback: List[Tuple[str, float, List[str], float, float]] = []
    # tuple = (sym, total_score, reasons, vol_usd, range_pct)

    # For meta/debug
    seen_pairs = 0
    spread_filtered = 0
    in_play_prefilter = 0
    fallback_prefilter = 0

    for sym in pairs:
        seen_pairs += 1
        t = tick.get(sym)
        if not t:
            continue

        usd_vol = float(t["vol_usd"])
        rng = float(t["range_pct"])

        ok_spread, spread_pct = _spread_ok(sym)
        if not ok_spread:
            spread_filtered += 1
            continue

        # Compute ATR + score (once)
        atr = _atr_for(sym)
        spot_score, spot_reasons = score_spot(t, atr=atr, spread_pct=spread_pct)

        bonus = 0.0
        bonus_reasons: List[str] = []
        if fut is not None:
            bonus, bonus_reasons = score_futures_bonus(sym, fut)

        total = float(spot_score + bonus)
        reasons = spot_reasons + bonus_reasons

        # Strict in-play eligibility
        if usd_vol >= MIN_24H_USD_VOL and rng >= MIN_24H_RANGE_PCT:
            in_play_prefilter += 1
            if total > 0:
                in_play.append((sym, total, reasons, usd_vol, rng))

        # Fallback eligibility (looser)
        if usd_vol >= FALLBACK_MIN_24H_USD_VOL and rng >= FALLBACK_MIN_24H_RANGE_PCT:
            fallback_prefilter += 1
            if total > 0:
                # Mark that this came from fallback if it wasn't already in_play
                fb_reasons = reasons[:]  # copy
                if not (usd_vol >= MIN_24H_USD_VOL and rng >= MIN_24H_RANGE_PCT):
                    fb_reasons = fb_reasons + ["fallback_pool"]
                fallback.append((sym, total, fb_reasons, usd_vol, rng))

    pre_in_play = len(in_play)
    pre_fallback = len(fallback)

    # De-dupe by base in each pool (keep best quote per base)
    def dedup(pool: List[Tuple[str, float, List[str], float, float]]) -> List[Tuple[str, float, List[str], float, float]]:
        if not DEDUP_BY_BASE or not pool:
            return pool
        best: Dict[str, Tuple[str, float, List[str], float, float]] = {}
        for sym, total, reasons, vol, rng in pool:
            b = _base(sym)
            prev = best.get(b)
            if prev is None:
                best[b] = (sym, total, reasons, vol, rng)
                continue
            ps, pt, pr, pv, prng = prev
            # score > volume > range tie-break
            if total > pt or (total == pt and vol > pv) or (total == pt and vol == pv and rng > prng):
                best[b] = (sym, total, reasons, vol, rng)
        return list(best.values())

    in_play = dedup(in_play)
    fallback = dedup(fallback)

    post_in_play = len(in_play)
    post_fallback = len(fallback)

    # Sort helper: score then volume then range
    def sort_key(x):
        return (x[1], x[3], x[4])

    in_play.sort(key=sort_key, reverse=True)
    fallback.sort(key=sort_key, reverse=True)

    # Select from in_play first
    top: List[Tuple[str, float, List[str], float, float]] = in_play[:TOP_N]

    # Fill from fallback if needed
    if FILL_TO_TOP_N and len(top) < TOP_N:
        chosen_bases = {_base(s) for (s, _, _, _, _) in top}
        for item in fallback:
            if len(top) >= TOP_N:
                break
            b = _base(item[0])
            if b in chosen_bases:
                continue
            top.append(item)
            chosen_bases.add(b)

    # Final cache
    CACHE["ts"] = time.time()
    CACHE["active_symbols"] = [s for (s, _, _, _, _) in top]
    CACHE["scores"] = {s: float(sc) for (s, sc, _, _, _) in top}
    CACHE["reasons"] = {s: rs for (s, _, rs, _, _) in top}
    CACHE["last_error"] = None
    CACHE["raw"] = {
        "universe": len(pairs),
        "seen_pairs": seen_pairs,
        "spread_filtered": spread_filtered,
        "in_play_prefilter_count": in_play_prefilter,
        "fallback_prefilter_count": fallback_prefilter,
        "in_play_scored_prefilter": pre_in_play,
        "fallback_scored_prefilter": pre_fallback,
        "in_play_scored_postdedup": post_in_play,
        "fallback_scored_postdedup": post_fallback,
        "returned": len(top),
        "dedup_by_base": DEDUP_BY_BASE,
        "fill_to_top_n": FILL_TO_TOP_N,
        "strict_thresholds": {
            "min_24h_usd_vol": MIN_24H_USD_VOL,
            "min_24h_range_pct": MIN_24H_RANGE_PCT,
        },
        "fallback_thresholds": {
            "min_24h_usd_vol": FALLBACK_MIN_24H_USD_VOL,
            "min_24h_range_pct": FALLBACK_MIN_24H_RANGE_PCT,
        },
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
