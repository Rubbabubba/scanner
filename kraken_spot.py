from __future__ import annotations

import requests
from typing import Dict, List, Tuple

KRAKEN_REST = "https://api.kraken.com/0/public"

def _req(path: str, params: dict | None = None) -> dict:
    r = requests.get(f"{KRAKEN_REST}/{path}", params=params or {}, timeout=20)
    r.raise_for_status()
    j = r.json()
    if j.get("error"):
        raise RuntimeError(str(j["error"]))
    return j["result"]

def list_spot_pairs(quotes: List[str], limit: int = 250) -> List[str]:
    """Return UI-like symbols BASE/QUOTE from AssetPairs."""
    res = _req("AssetPairs")
    out: List[str] = []
    for _, v in res.items():
        # v["wsname"] often like "XBT/USD" or "ADA/USDT"
        ws = v.get("wsname")
        if not ws or "/" not in ws:
            continue
        base, quote = ws.split("/", 1)
        quote = quote.upper()
        if quote not in quotes:
            continue

        # Normalize XBT -> BTC for UI
        base = "BTC" if base.upper() == "XBT" else base.upper()
        out.append(f"{base}/{quote}")

        if len(out) >= limit:
            break
    return sorted(list(set(out)))

def _to_rest_pair(sym: str) -> str:
    # Kraken REST often accepts altnames like XBTUSD, ETHUSDT etc.
    base, quote = sym.split("/", 1)
    base = "XBT" if base.upper() == "BTC" else base.upper()
    return f"{base}{quote.upper()}"

def ticker_24h(symbols: List[str]) -> Dict[str, dict]:
    """Return dict UI_symbol -> {vol_usd, range_pct, last, vwap}."""
    if not symbols:
        return {}
    pairs = ",".join(_to_rest_pair(s) for s in symbols[:400])
    res = _req("Ticker", params={"pair": pairs})

    out: Dict[str, dict] = {}
    # response keys are Kraken pair codes; we map back by searching
    # We'll just compute using returned values and store into out by best guess:
    # Use p (vwap), v (volume), h/l (high/low), c (last).
    for k, v in res.items():
        # k could be like "XXBTZUSD" or "XBTUSD"; we infer quote by suffix
        last = float(v["c"][0])
        vwap = float(v["p"][1]) if len(v["p"]) > 1 else float(v["p"][0])
        high = float(v["h"][1]) if len(v["h"]) > 1 else float(v["h"][0])
        low = float(v["l"][1]) if len(v["l"]) > 1 else float(v["l"][0])
        vol = float(v["v"][1]) if len(v["v"]) > 1 else float(v["v"][0])
        bid = float(v["b"][0])
        ask = float(v["a"][0])
        mid = (bid + ask) / 2.0
        spread_pct = (ask - bid) / mid if mid > 0 else None

        # quote detection
        quote = "USD"
        for q in ("USDT", "USDC", "USD", "EUR"):
            if q in k:
                quote = q
                break

        # base detection (best-effort)
        base = "BTC" if ("XBT" in k) else None
        if base is None:
            # crude: remove quote and prefixes
            tmp = k.replace("X", "").replace("Z", "")
            for q in ("USDT", "USDC", "USD", "EUR"):
                tmp = tmp.replace(q, "")
            base = tmp[:6].strip()
            base = "BTC" if base == "XBT" else base

        ui = f"{base}/{quote}".upper()

        rng_pct = (high - low) / low if low > 0 else 0.0
        vol_usd = vol * vwap

        out[ui] = {
            "last": last,
            "vwap": vwap,
            "range_pct": rng_pct,
            "vol_usd": vol_usd,
            "bid": bid,
            "ask": ask,
            "spread_pct": spread_pct,
        }
    return out

def ohlc_close_series(symbol: str, interval_min: int = 15, bars: int = 80) -> List[float]:
    pair = _to_rest_pair(symbol)
    res = _req("OHLC", params={"pair": pair, "interval": int(interval_min)})
    # result is {PAIRKEY: [[t,o,h,l,c,vwap,vol,count], ...], "last": "..."}
    key = [k for k in res.keys() if k != "last"][0]
    rows = res[key][-bars:]
    closes = [float(r[4]) for r in rows]
    return closes

def best_bid_ask(symbol: str) -> Tuple[float, float]:
    pair = _to_rest_pair(symbol)
    res = _req("Ticker", params={"pair": pair})
    k = list(res.keys())[0]
    v = res[k]
    bid = float(v["b"][0])
    ask = float(v["a"][0])
    return bid, ask
