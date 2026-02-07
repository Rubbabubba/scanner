from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from scanner.kraken_spot import (
    list_spot_pairs,
    ticker_24h,
    ohlc_close_series,
    best_bid_ask,
)
from scanner.scoring import (
    compute_atr,
    score_spot,
    score_futures_bonus,
)

# Optional futures module (bonus only)
FUTURES_ENABLED = os.getenv("FUTURES_ENABLED", "0").strip() in ("1", "true", "yes", "on")
if FUTURES_ENABLED:
    from scanner.kraken_futures import futures_snapshot

REFRESH_SEC = int(os.getenv("SCAN_REFRESH_SEC", "300"))  # 5m
TOP_N = int(os.getenv("TOP_N", "5"))
QUOTE_ALLOW = [q.strip().upper() for q in os.getenv("QUOTE_ALLOW", "USD,USDT,USDC").split(",") if q.strip()]
MIN_24H_USD_VOL = float(os.getenv("MIN_24H_USD_VOL", "2500000"))  # $2.5m
MIN_24H_RANGE_PCT = float(os.getenv("MIN_24H_RANGE_PCT", "0.05")) # 5%
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.004"))      # 0.40%
MAX_PAIRS = int(os.getenv("MAX_PAIRS", "250"))                    # cap universe for speed

app = FastAPI(title="Crypto Scanner", version="1.0.0")

CACHE: Dict[str, Any] = {
    "ts": None,
    "active_symbols": [],
    "reasons": {},
    "scores": {},
}

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def should_refresh() -> bool:
    ts = CACHE["ts"]
    if ts is None:
        return True
    return (time.time() - ts) >= REFRESH_SEC

def refresh() -> None:
    pairs = list_spot_pairs(quotes=QUOTE_ALLOW, limit=MAX_PAIRS)
    tick = ticker_24h(pairs)

    fut = None
    if FUTURES_ENABLED:
        try:
            fut = futures_snapshot()
        except Exception:
            fut = None

    scored: List[Tuple[str, float, List[str]]] = []
    scores_map: Dict[str, float] = {}
    reasons_map: Dict[str, List[str]] = {}

    for sym in pairs:
        t = tick.get(sym)
        if not t:
            continue

        # Liquidity / activity prefilter
        usd_vol = float(t["vol_usd"])
        rng = float(t["range_pct"])

        if usd_vol < MIN_24H_USD_VOL:
            continue
        if rng < MIN_24H_RANGE_PCT:
            continue

        # Spread sanity check
        try:
            bid, ask = best_bid_ask(sym)
            mid = (bid + ask) / 2.0
            spread_pct = (ask - bid) / mid if mid > 0 else 1.0
            if spread_pct > MAX_SPREAD_PCT:
                continue
        except Exception:
            # If book call fails, don't auto-exclude; just proceed.
            spread_pct = None

        # ATR expansion proxy using close series (15m by default)
        closes = ohlc_close_series(sym, interval_min=15, bars=80)
        atr = compute_atr(closes, period=14)

        spot_score, spot_reasons = score_spot(t, atr=atr, spread_pct=spread_pct)

        bonus = 0.0
        bonus_reasons: List[str] = []
        if fut is not None:
            bonus, bonus_reasons = score_futures_bonus(sym, fut)

        total = spot_score + bonus
        if total <= 0:
            continue

        reasons = spot_reasons + bonus_reasons
        scored.append((sym, total, reasons))
        scores_map[sym] = total
        reasons_map[sym] = reasons

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:TOP_N]

    CACHE["ts"] = time.time()
    CACHE["active_symbols"] = [s for (s, _, _) in top]
    CACHE["scores"] = {s: float(sc) for (s, sc, _) in top}
    CACHE["reasons"] = {s: rs for (s, _, rs) in top}

@app.get("/health")
def health():
    return {"ok": True, "utc": utc_now_iso(), "futures_enabled": FUTURES_ENABLED}

@app.get("/active_coins")
def active_coins():
    if should_refresh():
        try:
            refresh()
        except Exception as e:
            return JSONResponse(
                status_code=200,
                content={"ok": False, "utc": utc_now_iso(), "error": str(e), "active_symbols": CACHE["active_symbols"]},
            )
    return {
        "ok": True,
        "utc": utc_now_iso(),
        "active_symbols": CACHE["active_symbols"],
        "scores": CACHE["scores"],
        "reasons": CACHE["reasons"],
        "refresh_sec": REFRESH_SEC,
        "quote_allow": QUOTE_ALLOW,
        "futures_enabled": FUTURES_ENABLED,
    }
