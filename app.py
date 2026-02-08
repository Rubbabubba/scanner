from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# ✅ repo is flat (root-level imports)
from kraken_spot import list_spot_pairs, ticker_24h, ohlc_close_series, best_bid_ask
from scoring import compute_atr, score_spot, score_futures_bonus

FUTURES_ENABLED = os.getenv("FUTURES_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
if FUTURES_ENABLED:
    from kraken_futures import futures_snapshot


# -------------------------
# Config
# -------------------------
REFRESH_SEC = int(os.getenv("SCAN_REFRESH_SEC", "300") or 300)  # 5m
TOP_N = int(os.getenv("TOP_N", "5") or 5)
QUOTE_ALLOW = [q.strip().upper() for q in os.getenv("QUOTE_ALLOW", "USD,USDT,USDC").split(",") if q.strip()]
MAX_PAIRS = int(os.getenv("MAX_PAIRS", "250") or 250)

# Strict "in play" thresholds
MIN_24H_USD_VOL = float(os.getenv("MIN_24H_USD_VOL", "2500000") or 2500000)  # $2.5m
MIN_24H_RANGE_PCT = float(os.getenv("MIN_24H_RANGE_PCT", "0.05") or 0.05)     # 5%

# Spread filter (applies to both pools; keep tight by default)
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.004") or 0.004)         # 0.40%

# De-dupe and fill behavior
DEDUP_BY_BASE = os.getenv("DEDUP_BY_BASE", "1").strip().lower() in ("1", "true", "yes", "on")
FILL_TO_TOP_N = os.getenv("FILL_TO_TOP_N", "1").strip().lower() in ("1", "true", "yes", "on")

# Fallback thresholds (used only to fill)
FALLBACK_MIN_24H_USD_VOL = float(os.getenv("FALLBACK_MIN_24H_USD_VOL", "1000000") or 1000000)  # $1.0m
FALLBACK_MIN_24H_RANGE_PCT = float(os.getenv("FALLBACK_MIN_24H_RANGE_PCT", "0.025") or 0.025)  # 2.5%

# ✅ Blacklist (base assets only, comma-separated)
# Example: "CC,0G,HYPE"
BASE_BLACKLIST = {b.strip().upper() for b in os.getenv("BASE_BLACKLIST", "").split(",") if b.strip()}

# ✅ Majors floor (bases only, comma-separated)
# Used only if we still haven't filled TOP_N after in_play + fallback.
# Example: "BTC,ETH,SOL,LINK,ADA,DOT,AVAX,MATIC,LTC,XRP"
MAJORS_FLOOR = [b.strip().upper() for b in os.getenv("MAJORS_FLOOR", "").split(",") if b.strip()]

# Background loop behavior
STARTUP_REFRESH = os.getenv("STARTUP_REFRESH", "1").strip().lower() in ("1", "true", "yes", "on")
REFRESH_JITTER_SEC = int(os.getenv("REFRESH_JITTER_SEC", "5") or 5)  # small jitter so we don't align with other services


app = FastAPI(title="Crypto Scanner", version="1.2.0")

_CACHE_LOCK = threading.Lock()
CACHE: Dict[str, Any] = {
    "ts": None,              # epoch seconds
    "utc": None,             # ISO timestamp of last refresh
    "active_symbols": [],
    "reasons": {},
    "scores": {},
    "last_error": None,
    "raw": None,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base(sym: str) -> str:
    return sym.split("/", 1)[0].upper()


def _spread_ok(sym: str) -> Tuple[bool, float | None]:
    """
    Returns (ok, spread_pct_or_none).
    If book call fails, allow the symbol (spread unknown).
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


def _dedup(pool: List[Tuple[str, float, List[str], float, float]]) -> List[Tuple[str, float, List[str], float, float]]:
    """
    Keep best quote per base. Tie-breakers: score > volume > range.
    """
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
        if total > pt or (total == pt and vol > pv) or (total == pt and vol == pv and rng > prng):
            best[b] = (sym, total, reasons, vol, rng)
    return list(best.values())


def _sort_key(x: Tuple[str, float, List[str], float, float]):
    # score, volume, range
    return (x[1], x[3], x[4])


def _compute_scan() -> Dict[str, Any]:
    """
    Heavy work: build in_play + fallback pools and return finalized cache payload.
    """
    pairs = list_spot_pairs(quotes=QUOTE_ALLOW, limit=MAX_PAIRS)

    # ✅ Apply BASE_BLACKLIST BEFORE any scoring / tick fetch usage
    pre_blacklist_count = len(pairs)
    removed_blacklist = 0
    if BASE_BLACKLIST:
        filtered_pairs: List[str] = []
        for sym in pairs:
            try:
                b = _base(sym)
            except Exception:
                continue
            if b in BASE_BLACKLIST:
                removed_blacklist += 1
                continue
            filtered_pairs.append(sym)
        pairs = filtered_pairs

    tick = ticker_24h(pairs)

    fut = None
    if FUTURES_ENABLED:
        try:
            fut = futures_snapshot()
        except Exception:
            fut = None

    in_play: List[Tuple[str, float, List[str], float, float]] = []
    fallback: List[Tuple[str, float, List[str], float, float]] = []

    # Track the best scored pair per base across *all* scored pairs,
    # so we can fill with MAJORS_FLOOR without relaxing thresholds.
    best_any: Dict[str, Tuple[str, float, List[str], float, float]] = {}

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

        atr = _atr_for(sym)
        spot_score, spot_reasons = score_spot(t, atr=atr, spread_pct=spread_pct)

        bonus = 0.0
        bonus_reasons: List[str] = []
        if fut is not None:
            bonus, bonus_reasons = score_futures_bonus(sym, fut)

        total = float(spot_score + bonus)
        if total <= 0:
            continue

        reasons = spot_reasons + bonus_reasons

        # Save best pair for this base regardless of thresholds (for majors floor)
        b = _base(sym)
        prev = best_any.get(b)
        if prev is None:
            best_any[b] = (sym, total, reasons, usd_vol, rng)
        else:
            ps, pt, pr, pv, prng = prev
            if total > pt or (total == pt and usd_vol > pv) or (total == pt and usd_vol == pv and rng > prng):
                best_any[b] = (sym, total, reasons, usd_vol, rng)

        # Strict in-play
        if usd_vol >= MIN_24H_USD_VOL and rng >= MIN_24H_RANGE_PCT:
            in_play_prefilter += 1
            in_play.append((sym, total, reasons, usd_vol, rng))

        # Fallback pool (looser)
        if usd_vol >= FALLBACK_MIN_24H_USD_VOL and rng >= FALLBACK_MIN_24H_RANGE_PCT:
            fallback_prefilter += 1
            fb_reasons = reasons[:]  # copy
            if not (usd_vol >= MIN_24H_USD_VOL and rng >= MIN_24H_RANGE_PCT):
                fb_reasons = fb_reasons + ["fallback_pool"]
            fallback.append((sym, total, fb_reasons, usd_vol, rng))

    pre_in_play = len(in_play)
    pre_fallback = len(fallback)

    in_play = _dedup(in_play)
    fallback = _dedup(fallback)

    post_in_play = len(in_play)
    post_fallback = len(fallback)

    in_play.sort(key=_sort_key, reverse=True)
    fallback.sort(key=_sort_key, reverse=True)

    top: List[Tuple[str, float, List[str], float, float]] = in_play[:TOP_N]

    # Fill from fallback first (dynamic movers)
    majors_added = 0
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

        # If still short, fill from MAJORS_FLOOR in order
        if len(top) < TOP_N and MAJORS_FLOOR:
            for b in MAJORS_FLOOR:
                if len(top) >= TOP_N:
                    break
                if b in chosen_bases:
                    continue
                item = best_any.get(b)
                if not item:
                    continue
                sym, total, reasons, vol, rng = item
                # Add explicit reason marker for transparency
                floor_reasons = list(reasons) + ["majors_floor"]
                top.append((sym, float(total), floor_reasons, float(vol), float(rng)))
                chosen_bases.add(b)
                majors_added += 1

    active_symbols = [s for (s, _, _, _, _) in top]
    scores = {s: float(sc) for (s, sc, _, _, _) in top}
    reasons = {s: rs for (s, _, rs, _, _) in top}

    return {
        "ts": time.time(),
        "utc": utc_now_iso(),
        "active_symbols": active_symbols,
        "scores": scores,
        "reasons": reasons,
        "last_error": None,
        "raw": {
            "universe": len(pairs),
            "seen_pairs": seen_pairs,
            "pre_blacklist_pairs": pre_blacklist_count,
            "removed_by_blacklist": removed_blacklist,
            "base_blacklist": sorted(list(BASE_BLACKLIST))[:50],  # safety cap
            "spread_filtered": spread_filtered,
            "in_play_prefilter_count": in_play_prefilter,
            "fallback_prefilter_count": fallback_prefilter,
            "in_play_scored_prefilter": pre_in_play,
            "fallback_scored_prefilter": pre_fallback,
            "in_play_scored_postdedup": post_in_play,
            "fallback_scored_postdedup": post_fallback,
            "returned": len(top),
            "majors_floor_configured": len(MAJORS_FLOOR),
            "majors_floor_added": majors_added,
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
            "spread_max_pct": MAX_SPREAD_PCT,
        },
    }


def _refresh_forever() -> None:
    """
    Background refresh loop.
    """
    # Optional immediate refresh on startup
    if STARTUP_REFRESH:
        try:
            data = _compute_scan()
            with _CACHE_LOCK:
                CACHE.update(data)
        except Exception as e:
            with _CACHE_LOCK:
                CACHE["ts"] = time.time()
                CACHE["utc"] = utc_now_iso()
                CACHE["last_error"] = str(e)

    while True:
        # Sleep until next cycle
        sleep_for = max(5, int(REFRESH_SEC))
        # small jitter so multiple services don't align
        sleep_for = sleep_for + (REFRESH_JITTER_SEC if REFRESH_JITTER_SEC > 0 else 0)
        time.sleep(sleep_for)

        try:
            data = _compute_scan()
            with _CACHE_LOCK:
                CACHE.update(data)
        except Exception as e:
            with _CACHE_LOCK:
                CACHE["ts"] = time.time()
                CACHE["utc"] = utc_now_iso()
                CACHE["last_error"] = str(e)


@app.on_event("startup")
def _startup():
    t = threading.Thread(target=_refresh_forever, daemon=True)
    t.start()


# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    # Avoid noisy Render health checks showing 404
    with _CACHE_LOCK:
        return {
            "ok": True,
            "service": "crypto-scanner",
            "utc": utc_now_iso(),
            "last_refresh_utc": CACHE.get("utc"),
            "active_count": len(CACHE.get("active_symbols") or []),
        }


@app.get("/health")
def health():
    with _CACHE_LOCK:
        return {
            "ok": True,
            "utc": utc_now_iso(),
            "futures_enabled": FUTURES_ENABLED,
            "refresh_sec": REFRESH_SEC,
            "quote_allow": QUOTE_ALLOW,
            "dedup_by_base": DEDUP_BY_BASE,
            "fill_to_top_n": FILL_TO_TOP_N,
            "max_pairs": MAX_PAIRS,
            "max_spread_pct": MAX_SPREAD_PCT,
            "base_blacklist_count": len(BASE_BLACKLIST),
            "base_blacklist_sample": sorted(list(BASE_BLACKLIST))[:20],
            "majors_floor_count": len(MAJORS_FLOOR),
            "majors_floor_sample": MAJORS_FLOOR[:20],
            "last_refresh_utc": CACHE.get("utc"),
            "last_error": CACHE.get("last_error"),
        }


@app.get("/active_coins")
def active_coins():
    # Return cache immediately; do NOT compute here.
    with _CACHE_LOCK:
        # If we haven't refreshed yet, try once (fast fail-safe)
        if CACHE.get("ts") is None:
            try:
                data = _compute_scan()
                CACHE.update(data)
            except Exception as e:
                CACHE["ts"] = time.time()
                CACHE["utc"] = utc_now_iso()
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
            "active_symbols": CACHE.get("active_symbols", []),
            "scores": CACHE.get("scores", {}),
            "reasons": CACHE.get("reasons", {}),
            "meta": CACHE.get("raw"),
            "refresh_sec": REFRESH_SEC,
            "quote_allow": QUOTE_ALLOW,
            "futures_enabled": FUTURES_ENABLED,
            "last_error": CACHE.get("last_error"),
            "last_refresh_utc": CACHE.get("utc"),
        }
