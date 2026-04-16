
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Iterable


def compute_atr(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 2:
        return 0.0
    trs = []
    for i in range(1, len(closes)):
        trs.append(abs(closes[i] - closes[i - 1]))
    window = trs[-period:]
    return sum(window) / max(1, len(window))


def score_spot(t: Dict[str, Any], atr: float, spread_pct: float | None) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 0.0

    rng = float(t["range_pct"])
    vol_usd = float(t["vol_usd"])

    # Range
    if rng >= 0.10:
        score += 3; reasons.append("range_24h_10p")
    elif rng >= 0.07:
        score += 2; reasons.append("range_24h_7p")
    elif rng >= 0.05:
        score += 1; reasons.append("range_24h_5p")

    # Volume tiers
    if vol_usd >= 50_000_000:
        score += 3; reasons.append("vol_24h_50m")
    elif vol_usd >= 10_000_000:
        score += 2; reasons.append("vol_24h_10m")
    elif vol_usd >= 2_500_000:
        score += 1; reasons.append("vol_24h_2_5m")

    # ATR (proxy for "alive now")
    if atr > 0:
        score += 1; reasons.append("atr_active")

    # Spread penalty/reward
    if spread_pct is not None:
        if spread_pct <= 0.0015:
            score += 1; reasons.append("tight_spread")
        elif spread_pct >= 0.004:
            score -= 2; reasons.append("wide_spread_penalty")

    return score, reasons


def score_futures_bonus(symbol_ui: str, fut: Dict[str, Any]) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    base = symbol_ui.split("/", 1)[0].upper()

    x = fut.get(base)
    if not x:
        return 0.0, reasons

    score = 0.0

    fr = x.get("funding_rate_per_hour")
    oi = x.get("open_interest_usd")
    oi_chg = x.get("open_interest_change_pct_24h")

    if fr is not None:
        afr = abs(float(fr))
        if afr >= 0.0002:
            score += 2; reasons.append("funding_extreme")
        elif afr >= 0.0001:
            score += 1; reasons.append("funding_elevated")

    if oi_chg is not None:
        if float(oi_chg) >= 0.20:
            score += 2; reasons.append("oi_spike_20p")
        elif float(oi_chg) >= 0.10:
            score += 1; reasons.append("oi_spike_10p")

    if oi is not None and float(oi) >= 50_000_000:
        score += 1; reasons.append("oi_large")

    return score, reasons


def score_ranking_bias(
    symbol_ui: str,
    t: Dict[str, Any],
    spread_pct: float | None,
    preferred_bases: Iterable[str] | None = None,
) -> Tuple[float, List[str]]:
    """Bias rankings toward liquid, tighter-spread preferred majors without requiring env changes."""
    reasons: List[str] = []
    score = 0.0
    base = symbol_ui.split("/", 1)[0].upper()
    preferred = {str(x).strip().upper() for x in (preferred_bases or []) if str(x).strip()}
    vol_usd = float(t.get("vol_usd") or 0.0)
    rng = float(t.get("range_pct") or 0.0)

    if base in preferred:
        score += 2.5; reasons.append("preferred_major_bias")

    if vol_usd >= 100_000_000:
        score += 2.0; reasons.append("liq_100m_bias")
    elif vol_usd >= 25_000_000:
        score += 1.25; reasons.append("liq_25m_bias")
    elif vol_usd >= 10_000_000:
        score += 0.75; reasons.append("liq_10m_bias")

    if spread_pct is not None:
        if spread_pct <= 0.0010:
            score += 1.0; reasons.append("spread_elite_bias")
        elif spread_pct <= 0.0018:
            score += 0.5; reasons.append("spread_good_bias")
        elif spread_pct >= 0.0025:
            score -= 1.0; reasons.append("spread_drag_penalty")

    if rng >= 0.07 and vol_usd >= 10_000_000:
        score += 1.0; reasons.append("trend_liquidity_combo")
    elif rng >= 0.05 and vol_usd >= 5_000_000:
        score += 0.5; reasons.append("trend_support_combo")

    if base not in preferred and vol_usd < 5_000_000:
        score -= 1.25; reasons.append("tail_liquidity_penalty")

    return score, reasons
