from __future__ import annotations

import os
import requests
from typing import Dict, Any

# NOTE: Kraken derivatives API base may differ by region/account.
# Keep this configurable.
FUTURES_BASE = os.getenv("KRAKEN_FUTURES_BASE", "https://api.futures.kraken.com/derivatives/api/v3")

def _get(path: str, params: dict | None = None) -> dict:
    r = requests.get(f"{FUTURES_BASE}{path}", params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()

def futures_snapshot() -> Dict[str, Any]:
    """
    Returns a dict keyed by base symbol (BTC, ETH, SOL, LINK, etc):
      {
        "BTC": {"funding_rate_per_hour": ..., "open_interest_usd": ..., "open_interest_change_pct_24h": ...},
        ...
      }

    If endpoint formats differ, we still fail gracefully at caller.
    """
    # Public instruments/tickers often include mark price, OI, funding (varies).
    # Try a common public endpoint path:
    data = _get("/tickers")

    out: Dict[str, Any] = {}

    # Expecting a list of tickers. We pick perpetual USD tickers where possible.
    tickers = data.get("tickers") or data.get("result") or data
    if not isinstance(tickers, list):
        return out

    for t in tickers:
        sym = (t.get("symbol") or t.get("instrument") or "").upper()
        if not sym:
            continue
        # Prefer perps
        if "PERP" not in sym and "PI_" not in sym:
            continue

        base = sym.replace("PI_", "").split("_")[0]
        # Pull common fields (best-effort)
        fr = t.get("funding_rate") or t.get("fundingRate") or t.get("funding_rate_per_hour")
        oi = t.get("open_interest") or t.get("openInterest") or t.get("open_interest_usd")
        oi_chg = t.get("open_interest_change_24h") or t.get("openInterestChange24h") or t.get("open_interest_change_pct_24h")

        out[base] = {
            "funding_rate_per_hour": fr,
            "open_interest_usd": oi,
            "open_interest_change_pct_24h": oi_chg,
        }

    return out
