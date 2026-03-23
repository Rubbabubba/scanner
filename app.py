from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import requests

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from build_info import build_payload

# ✅ repo is flat (root-level imports)
from kraken_spot import list_spot_pairs, ticker_24h
from scoring import score_spot, score_futures_bonus

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

# ATR-active proxy (we avoid per-symbol OHLC calls for reliability)
ATR_ACTIVE_MIN_RANGE_PCT = float(os.getenv("ATR_ACTIVE_MIN_RANGE_PCT", "0.02") or 0.02)  # 2%

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
SCANNER_COORDINATION_URL = os.getenv("SCANNER_COORDINATION_URL", "").strip()
SCANNER_COORDINATION_TIMEOUT_SEC = float(os.getenv("SCANNER_COORDINATION_TIMEOUT_SEC", "3.0") or 3.0)
SCANNER_COORDINATION_LOOKBACK_SEC = int(os.getenv("SCANNER_COORDINATION_LOOKBACK_SEC", "900") or 900)
SCANNER_SYMBOL_HOLDOFF_SEC = int(os.getenv("SCANNER_SYMBOL_HOLDOFF_SEC", "0") or 0)
SCANNER_FINGERPRINT_TTL_SEC = int(os.getenv("SCANNER_FINGERPRINT_TTL_SEC", str(max(900, SCANNER_COORDINATION_LOOKBACK_SEC or 900))) or max(900, SCANNER_COORDINATION_LOOKBACK_SEC or 900))
SCANNER_BAR_LOCK_SEC = int(os.getenv("SCANNER_BAR_LOCK_SEC", str(max(60, REFRESH_SEC))) or max(60, REFRESH_SEC))
SCANNER_INFLIGHT_HOLDOFF_SEC = int(os.getenv("SCANNER_INFLIGHT_HOLDOFF_SEC", str(max(SCANNER_SYMBOL_HOLDOFF_SEC, SCANNER_BAR_LOCK_SEC))) or max(SCANNER_SYMBOL_HOLDOFF_SEC, SCANNER_BAR_LOCK_SEC))


app = FastAPI(title="Crypto Scanner", version="1.4.0")

PATCH_BUILD = build_payload()

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

_EMIT_HISTORY: Dict[str, float] = {}
_EMIT_FINGERPRINTS: Dict[str, float] = {}
_EMIT_BAR_LOCKS: Dict[str, float] = {}
_EMIT_LOCK = threading.Lock()
_SUPPRESSION_STATS: Dict[str, Any] = {
    "last_refresh_utc": None,
    "last_refresh_ts": None,
    "last_refresh_counts": {},
    "cumulative_counts": defaultdict(int),
    "recent_suppressed_symbols": {},
    "recent_emitted_symbols": [],
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base(sym: str) -> str:
    return sym.split("/", 1)[0].upper()


def _now_ts() -> float:
    return time.time()


def _bar_bucket(ts: float) -> int:
    width = max(60, int(SCANNER_BAR_LOCK_SEC or REFRESH_SEC or 300))
    return int(ts // width) * width


def _scanner_symbol_fingerprint(symbol: str, ts: float | None = None) -> str:
    now = float(ts or _now_ts())
    return f"{str(symbol or '').upper()}|bar:{_bar_bucket(now)}"


def _record_refresh_stats(counts: Dict[str, int], suppressed_symbols: Dict[str, List[str]], emitted_symbols: List[str]) -> None:
    now_iso = utc_now_iso()
    with _EMIT_LOCK:
        _SUPPRESSION_STATS["last_refresh_utc"] = now_iso
        _SUPPRESSION_STATS["last_refresh_ts"] = _now_ts()
        _SUPPRESSION_STATS["last_refresh_counts"] = dict(counts)
        cc = _SUPPRESSION_STATS.setdefault("cumulative_counts", defaultdict(int))
        for k, v in counts.items():
            cc[k] += int(v or 0)
        recent = _SUPPRESSION_STATS.setdefault("recent_suppressed_symbols", {})
        for reason, syms in suppressed_symbols.items():
            if syms:
                recent[reason] = list(syms)[-50:]
        _SUPPRESSION_STATS["recent_emitted_symbols"] = list(emitted_symbols)[-50:]


def _suppression_snapshot() -> Dict[str, Any]:
    with _EMIT_LOCK:
        return {
            "last_refresh_utc": _SUPPRESSION_STATS.get("last_refresh_utc"),
            "last_refresh_ts": _SUPPRESSION_STATS.get("last_refresh_ts"),
            "last_refresh_counts": dict(_SUPPRESSION_STATS.get("last_refresh_counts") or {}),
            "cumulative_counts": dict(_SUPPRESSION_STATS.get("cumulative_counts") or {}),
            "recent_suppressed_symbols": dict(_SUPPRESSION_STATS.get("recent_suppressed_symbols") or {}),
            "recent_emitted_symbols": list(_SUPPRESSION_STATS.get("recent_emitted_symbols") or []),
            "active_emit_history": len(_EMIT_HISTORY),
            "active_emit_fingerprints": len(_EMIT_FINGERPRINTS),
            "active_bar_locks": len(_EMIT_BAR_LOCKS),
        }


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




def _extract_symbols_from_objects(items: List[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items or []:
        sym = None
        if isinstance(item, dict):
            sym = item.get("symbol") or item.get("symbol_id") or item.get("pair")
        elif isinstance(item, str):
            sym = item
        s = str(sym or "").strip().upper()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _fetch_coordination_state() -> Dict[str, Any]:
    if not SCANNER_COORDINATION_URL:
        return {"ok": False, "reason": "missing_coordination_url", "suppressed_symbols": [], "hard_suppressed_symbols": []}
    try:
        r = requests.get(
            SCANNER_COORDINATION_URL,
            params={"lookback_sec": SCANNER_COORDINATION_LOOKBACK_SEC, "limit": max(25, TOP_N * 5)},
            timeout=SCANNER_COORDINATION_TIMEOUT_SEC,
        )
        r.raise_for_status()
        data = r.json() if r.content else {}
        coord = data.get("coordination") if isinstance(data, dict) else {}
        suppressed_symbols = coord.get("suppressed_symbols") or []
        clean = []
        seen = set()
        for sym in suppressed_symbols:
            s = str(sym or "").strip().upper()
            if not s or s in seen:
                continue
            seen.add(s)
            clean.append(s)
        active_workflow_locks = coord.get("active_workflow_locks") or []
        recent_admission_passed = coord.get("recent_admission_passed") or []
        active_signal_fingerprints = coord.get("active_signal_fingerprints") or []
        hard = set(clean)
        hard.update(_extract_symbols_from_objects(active_workflow_locks))
        hard.update(_extract_symbols_from_objects(recent_admission_passed))
        hard.update(_extract_symbols_from_objects(active_signal_fingerprints))
        return {
            "ok": bool(data.get("ok", True)),
            "reason": None,
            "suppressed_symbols": clean,
            "hard_suppressed_symbols": sorted(hard),
            "active_workflow_locks": active_workflow_locks,
            "recent_admission_passed": recent_admission_passed,
            "active_signal_fingerprints": active_signal_fingerprints,
        }
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}", "suppressed_symbols": [], "hard_suppressed_symbols": []}


def _apply_scanner_symbol_holdoff(symbols: List[str]) -> tuple[List[str], Dict[str, Any]]:
    holdoff_sec = max(0, int(SCANNER_INFLIGHT_HOLDOFF_SEC or SCANNER_SYMBOL_HOLDOFF_SEC or 0))
    if holdoff_sec <= 0:
        return symbols, {"holdoff_sec": 0, "suppressed_symbols": [], "active_symbol_count": len(symbols)}
    now = _now_ts()
    out: List[str] = []
    suppressed: List[str] = []
    with _EMIT_LOCK:
        expired = [sym for sym, ts in _EMIT_HISTORY.items() if (now - float(ts)) >= float(holdoff_sec)]
        for sym in expired:
            _EMIT_HISTORY.pop(sym, None)
        for sym in symbols:
            last_ts = float(_EMIT_HISTORY.get(sym) or 0.0)
            if last_ts and (now - last_ts) < float(holdoff_sec):
                suppressed.append(sym)
                continue
            out.append(sym)
    return out, {
        "holdoff_sec": holdoff_sec,
        "suppressed_symbols": suppressed,
        "active_symbol_count": len(out),
    }


def _apply_coordination_suppression(pool: List[Tuple[str, float, List[str], float, float]], suppressed_symbols: set[str]) -> tuple[List[Tuple[str, float, List[str], float, float]], Dict[str, Any]]:
    if not suppressed_symbols:
        return pool, {"suppressed_symbols": [], "remaining": len(pool)}
    kept: List[Tuple[str, float, List[str], float, float]] = []
    suppressed: List[str] = []
    for item in pool:
        sym = str(item[0] or '').upper()
        if sym in suppressed_symbols:
            suppressed.append(sym)
            continue
        kept.append(item)
    return kept, {"suppressed_symbols": suppressed, "remaining": len(kept)}

def _apply_scanner_emission_controls(symbols: List[str]) -> tuple[List[str], Dict[str, Any]]:
    now = _now_ts()
    ttl_sec = max(60, int(SCANNER_FINGERPRINT_TTL_SEC or 900))
    bar_lock_sec = max(60, int(SCANNER_BAR_LOCK_SEC or REFRESH_SEC or 300))
    emitted: List[str] = []
    duplicate_suppressed: List[str] = []
    bar_lock_suppressed: List[str] = []
    with _EMIT_LOCK:
        expired_fp = [fp for fp, ts in _EMIT_FINGERPRINTS.items() if (now - float(ts)) >= float(ttl_sec)]
        for fp in expired_fp:
            _EMIT_FINGERPRINTS.pop(fp, None)
        expired_bar = [k for k, ts in _EMIT_BAR_LOCKS.items() if (now - float(ts)) >= float(bar_lock_sec)]
        for k in expired_bar:
            _EMIT_BAR_LOCKS.pop(k, None)
        for sym in symbols:
            fp = _scanner_symbol_fingerprint(sym, now)
            bar_key = f"{str(sym or '').upper()}|bar:{_bar_bucket(now)}"
            if fp in _EMIT_FINGERPRINTS:
                duplicate_suppressed.append(sym)
                continue
            if bar_key in _EMIT_BAR_LOCKS:
                bar_lock_suppressed.append(sym)
                continue
            emitted.append(sym)
            _EMIT_FINGERPRINTS[fp] = now
            _EMIT_BAR_LOCKS[bar_key] = now
            _EMIT_HISTORY[str(sym or '').upper()] = now
    return emitted, {
        "fingerprint_ttl_sec": ttl_sec,
        "bar_lock_sec": bar_lock_sec,
        "suppressed_duplicate": duplicate_suppressed,
        "suppressed_bar_lock": bar_lock_suppressed,
        "emitted_symbols": emitted,
    }

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

        spread_pct = t.get("spread_pct")
        # Spread filter (from ticker; no extra REST calls)
        if spread_pct is not None and float(spread_pct) > MAX_SPREAD_PCT:
            spread_filtered += 1
            continue

        # ATR-active proxy: treat as "active" if 24h range is at least a small floor
        atr_proxy = 1.0 if float(rng) >= ATR_ACTIVE_MIN_RANGE_PCT else 0.0
        spot_score, spot_reasons = score_spot(t, atr=atr_proxy, spread_pct=spread_pct)

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

    coordination = _fetch_coordination_state()
    coordination_suppressed = set(str(s or '').strip().upper() for s in (coordination.get('hard_suppressed_symbols') or coordination.get('suppressed_symbols') or []))
    filtered_top, coordination_meta = _apply_coordination_suppression(top, coordination_suppressed)

    if FILL_TO_TOP_N and len(filtered_top) < TOP_N:
        chosen = {_base(s) for (s, _, _, _, _) in filtered_top}
        existing = {str(s or '').upper() for (s, _, _, _, _) in filtered_top}
        for item in (in_play + fallback):
            if len(filtered_top) >= TOP_N:
                break
            sym = str(item[0] or '').upper()
            if sym in coordination_suppressed or sym in existing:
                continue
            b = _base(sym)
            if b in chosen:
                continue
            filtered_top.append(item)
            chosen.add(b)
            existing.add(sym)

    candidate_symbols = [str(s or '').upper() for (s, _, _, _, _) in filtered_top]
    candidate_symbols, holdoff_meta = _apply_scanner_symbol_holdoff(candidate_symbols)
    active_symbols, emission_meta = _apply_scanner_emission_controls(candidate_symbols)
    top_by_symbol = {str(s).upper(): (sc, rs) for (s, sc, rs, _, _) in filtered_top}
    scores = {s: float(top_by_symbol[s][0]) for s in active_symbols if s in top_by_symbol}
    reasons = {s: top_by_symbol[s][1] for s in active_symbols if s in top_by_symbol}

    refresh_counts = {
        "scanner_candidates": len(top),
        "scanner_coordination_suppressed": len(coordination_meta.get("suppressed_symbols") or []),
        "scanner_holdoff_suppressed": len(holdoff_meta.get("suppressed_symbols") or []),
        "scanner_suppressed_duplicate": len(emission_meta.get("suppressed_duplicate") or []),
        "scanner_suppressed_bar_lock": len(emission_meta.get("suppressed_bar_lock") or []),
        "scanner_emitted": len(active_symbols),
    }
    _record_refresh_stats(refresh_counts, {
        "coordination": coordination_meta.get("suppressed_symbols") or [],
        "holdoff": holdoff_meta.get("suppressed_symbols") or [],
        "duplicate": emission_meta.get("suppressed_duplicate") or [],
        "bar_lock": emission_meta.get("suppressed_bar_lock") or [],
    }, active_symbols)

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
            "coordination": {
                "enabled": bool(SCANNER_COORDINATION_URL),
                "ok": bool(coordination.get("ok")),
                "reason": coordination.get("reason"),
                "suppressed_symbols": coordination_meta.get("suppressed_symbols") or [],
                "hard_suppressed_symbols": coordination.get("hard_suppressed_symbols") or [],
                "active_workflow_locks": len(coordination.get("active_workflow_locks") or []),
                "recent_admission_passed": len(coordination.get("recent_admission_passed") or []),
                "active_signal_fingerprints": len(coordination.get("active_signal_fingerprints") or []),
            },
            "scanner_symbol_holdoff": holdoff_meta,
            "scanner_emission_controls": emission_meta,
            "scanner_telemetry": _suppression_snapshot(),
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




def _scanner_compatibility_snapshot() -> Dict[str, Any]:
    with _CACHE_LOCK:
        active_symbols = list(CACHE.get("active_symbols") or [])
        return {
            "mode": "ranked_multi_symbol_scanner",
            "supports_multi_symbol": True,
            "quote_allow": list(QUOTE_ALLOW),
            "futures_enabled": bool(FUTURES_ENABLED),
            "refresh_sec": int(REFRESH_SEC),
            "top_n": int(TOP_N),
            "max_pairs": int(MAX_PAIRS),
            "max_spread_pct": float(MAX_SPREAD_PCT),
            "scanner_symbol_holdoff_sec": int(SCANNER_SYMBOL_HOLDOFF_SEC),
            "scanner_inflight_holdoff_sec": int(SCANNER_INFLIGHT_HOLDOFF_SEC),
            "scanner_bar_lock_sec": int(SCANNER_BAR_LOCK_SEC),
            "scanner_fingerprint_ttl_sec": int(SCANNER_FINGERPRINT_TTL_SEC),
            "coordination_url_configured": bool(SCANNER_COORDINATION_URL),
            "cache_warm": CACHE.get("ts") is not None,
            "active_count": len(active_symbols),
            "active_symbols_sample": active_symbols[:12],
            "active_symbols_truncated": len(active_symbols) > 12,
            "last_refresh_utc": CACHE.get("utc"),
            "last_error": CACHE.get("last_error"),
            "fee_churn_guardrails": {
                "emission_controls_active": True,
                "symbol_holdoff_active": int(SCANNER_INFLIGHT_HOLDOFF_SEC) > 0 or int(SCANNER_SYMBOL_HOLDOFF_SEC) > 0,
                "bar_lock_active": int(SCANNER_BAR_LOCK_SEC) > 0,
                "fingerprint_ttl_active": int(SCANNER_FINGERPRINT_TTL_SEC) > 0,
            },
        }


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
            "build": PATCH_BUILD,
            "utc": utc_now_iso(),
            "last_refresh_utc": CACHE.get("utc"),
            "active_count": len(CACHE.get("active_symbols") or []),
            "compatibility": _scanner_compatibility_snapshot(),
        }


@app.get("/health")
def health():
    with _CACHE_LOCK:
        return {
            "ok": True,
            "utc": utc_now_iso(),
            "build": PATCH_BUILD,
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
            "scanner_coordination_url": SCANNER_COORDINATION_URL or None,
            "scanner_symbol_holdoff_sec": SCANNER_SYMBOL_HOLDOFF_SEC,
            "scanner_fingerprint_ttl_sec": SCANNER_FINGERPRINT_TTL_SEC,
            "scanner_bar_lock_sec": SCANNER_BAR_LOCK_SEC,
            "scanner_inflight_holdoff_sec": SCANNER_INFLIGHT_HOLDOFF_SEC,
            "scanner_telemetry": _suppression_snapshot(),
        }




@app.get("/diagnostics/scanner_suppression")
def diagnostics_scanner_suppression():
    with _CACHE_LOCK:
        return {
            "ok": True,
            "utc": utc_now_iso(),
            "scanner_fingerprint_ttl_sec": SCANNER_FINGERPRINT_TTL_SEC,
            "scanner_bar_lock_sec": SCANNER_BAR_LOCK_SEC,
            "scanner_inflight_holdoff_sec": SCANNER_INFLIGHT_HOLDOFF_SEC,
            "scanner_symbol_holdoff_sec": SCANNER_SYMBOL_HOLDOFF_SEC,
            "telemetry": _suppression_snapshot(),
            "last_refresh_utc": CACHE.get("utc"),
            "last_meta": (CACHE.get("raw") or {}),
        }


@app.get("/build")
def build_info_endpoint():
    return {**PATCH_BUILD}


@app.get("/compatibility")
def compatibility_endpoint():
    return {
        "ok": True,
        "utc": utc_now_iso(),
        "build": PATCH_BUILD,
        "service": {
            "name": PATCH_BUILD.get("system_name"),
            "role": PATCH_BUILD.get("service_role"),
            "env_name": PATCH_BUILD.get("env_name"),
            "release_stage": PATCH_BUILD.get("release_stage_configured"),
        },
        "compatibility": _scanner_compatibility_snapshot(),
    }


@app.get("/runtime")
def runtime_endpoint():
    with _CACHE_LOCK:
        return {
            "ok": True,
            "utc": utc_now_iso(),
            "build": PATCH_BUILD,
            "service": {
                "name": PATCH_BUILD.get("system_name"),
                "role": PATCH_BUILD.get("service_role"),
                "env_name": PATCH_BUILD.get("env_name"),
                "release_stage": PATCH_BUILD.get("release_stage_configured"),
            },
            "runtime": {
                "futures_enabled": FUTURES_ENABLED,
                "refresh_sec": REFRESH_SEC,
                "top_n": TOP_N,
                "quote_allow": QUOTE_ALLOW,
                "max_pairs": MAX_PAIRS,
                "max_spread_pct": MAX_SPREAD_PCT,
                "base_blacklist_count": len(BASE_BLACKLIST),
                "majors_floor_count": len(MAJORS_FLOOR),
                "scanner_coordination_url": SCANNER_COORDINATION_URL or None,
                "last_refresh_utc": CACHE.get("utc"),
                "last_error": CACHE.get("last_error"),
                "active_count": len(CACHE.get("active_symbols") or []),
                "active_symbols": CACHE.get("active_symbols") or [],
                "telemetry": _suppression_snapshot(),
                "compatibility": _scanner_compatibility_snapshot(),
            },
        }


@app.get("/ready")
def ready_endpoint():
    with _CACHE_LOCK:
        issues = []
        cache_warm = CACHE.get("ts") is not None
        if not cache_warm:
            issues.append("scanner_cache_cold")
        if CACHE.get("last_error"):
            issues.append("scanner_last_error_present")
        if not QUOTE_ALLOW:
            issues.append("quote_allow_empty")
        ready = len(issues) == 0
        return {
            "ok": True,
            "ready": ready,
            "utc": utc_now_iso(),
            "build": PATCH_BUILD,
            "service": {
                "name": PATCH_BUILD.get("system_name"),
                "role": PATCH_BUILD.get("service_role"),
                "env_name": PATCH_BUILD.get("env_name"),
                "release_stage": PATCH_BUILD.get("release_stage_configured"),
            },
            "issues": issues,
            "cache_warm": cache_warm,
            "last_refresh_utc": CACHE.get("utc"),
            "last_error": CACHE.get("last_error"),
            "active_count": len(CACHE.get("active_symbols") or []),
        }


@app.get("/active_coins")
def active_coins():
    """
    Return cache immediately. Never run a full scan in the request path.
    On cold start, return ok=False + warming_up until background refresh populates CACHE.
    """
    with _CACHE_LOCK:
        if CACHE.get("ts") is None:
            return JSONResponse(
                status_code=200,
                content={
                    "ok": False,
                    "utc": utc_now_iso(),
                    "status": "warming_up",
                    "refresh_sec": REFRESH_SEC,
                    "quote_allow": QUOTE_ALLOW,
                    "futures_enabled": FUTURES_ENABLED,
                    "last_error": CACHE.get("last_error"),
                    "last_refresh_utc": CACHE.get("utc"),
                    "active_symbols": [],
                    "compatibility": _scanner_compatibility_snapshot(),
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
            "compatibility": _scanner_compatibility_snapshot(),
        }
