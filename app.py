from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

import requests

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from build_info import build_payload

# ✅ repo is flat (root-level imports)
from kraken_spot import list_spot_pairs, ticker_24h
from scoring import score_spot, score_futures_bonus, score_ranking_bias

FUTURES_ENABLED = os.getenv("FUTURES_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
if FUTURES_ENABLED:
    from kraken_futures import futures_snapshot


# -------------------------
# Config
# -------------------------
REFRESH_SEC = int(os.getenv("SCAN_REFRESH_SEC", "300") or 300)  # 5m
TOP_N = int(os.getenv("TOP_N", "7") or 7)
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

# Stricter thresholds for vetted tier-2 names
STRICT_MIN_24H_USD_VOL = float(os.getenv("STRICT_MIN_24H_USD_VOL", "1500000") or 1500000)
STRICT_MIN_24H_RANGE_PCT = float(os.getenv("STRICT_MIN_24H_RANGE_PCT", "0.018") or 0.018)

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
SCANNER_BAR_LOCK_SEC = max(60, int(os.getenv("SCANNER_BAR_LOCK_SEC", str(max(60, REFRESH_SEC))) or max(60, REFRESH_SEC)))
SCANNER_INFLIGHT_HOLDOFF_SEC = max(60, int(os.getenv("SCANNER_INFLIGHT_HOLDOFF_SEC", str(max(SCANNER_SYMBOL_HOLDOFF_SEC, SCANNER_BAR_LOCK_SEC))) or max(SCANNER_SYMBOL_HOLDOFF_SEC, SCANNER_BAR_LOCK_SEC)))


app = FastAPI(title="Crypto Scanner", version="1.4.3")

PATCH_BUILD = build_payload()

_CACHE_LOCK = threading.RLock()
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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _csv(name: str, default: str = "") -> List[str]:
    raw = os.getenv(name, default)
    return [part.strip() for part in str(raw or "").split(",") if part and part.strip()]


SCANNER_MIN_SCORE = float(os.getenv("SCANNER_MIN_SCORE", "0") or 0)
SCANNER_MIN_REASON_COUNT = int(os.getenv("SCANNER_MIN_REASON_COUNT", "0") or 0)
SCANNER_REQUIRE_RANGE_OR_VOLUME = _env_bool("SCANNER_REQUIRE_RANGE_OR_VOLUME", False)
SCANNER_DROP_TIGHT_SPREAD_ONLY = _env_bool("SCANNER_DROP_TIGHT_SPREAD_ONLY", False)
SCANNER_DROP_ATR_AND_SPREAD_ONLY = _env_bool("SCANNER_DROP_ATR_AND_SPREAD_ONLY", False)
SMART_RANKING_BIAS_ENABLED = _env_bool("SMART_RANKING_BIAS_ENABLED", True)
SMART_RANKING_PREFERRED_BASES = [
    s.strip().upper() for s in os.getenv(
        "SMART_RANKING_PREFERRED_BASES",
        "BTC,ETH,SOL,ADA,LINK,AVAX,DOT"
    ).split(",") if s.strip()
]
SMART_RANKING_FINAL_EMIT_HARD_FILTER_ENABLED = _env_bool("SMART_RANKING_FINAL_EMIT_HARD_FILTER_ENABLED", True)
OPPORTUNITY_MODE_ENABLED = _env_bool("OPPORTUNITY_MODE_ENABLED", True)
OPPORTUNITY_MODE_TARGET_ACTIVE = int(os.getenv("OPPORTUNITY_MODE_TARGET_ACTIVE", "4") or 4)
OPPORTUNITY_MODE_PREFERRED_BASES = [
    s.strip().upper() for s in os.getenv(
        "OPPORTUNITY_MODE_PREFERRED_BASES",
        "BTC,ETH,SOL,ADA,LINK,AVAX,DOT"
    ).split(",") if s.strip()
]
OPPORTUNITY_MODE_MIN_BACKFILL_SCORE = float(os.getenv("OPPORTUNITY_MODE_MIN_BACKFILL_SCORE", "1.25") or 1.25)
OPPORTUNITY_MODE_SOFT_BACKFILL_ENABLED = _env_bool("OPPORTUNITY_MODE_SOFT_BACKFILL_ENABLED", True)
OPPORTUNITY_MODE_SOFT_MIN_USD_VOL = float(os.getenv("OPPORTUNITY_MODE_SOFT_MIN_USD_VOL", str(FALLBACK_MIN_24H_USD_VOL)) or FALLBACK_MIN_24H_USD_VOL)
OPPORTUNITY_MODE_SOFT_MIN_RANGE_PCT = float(os.getenv("OPPORTUNITY_MODE_SOFT_MIN_RANGE_PCT", str(ATR_ACTIVE_MIN_RANGE_PCT)) or ATR_ACTIVE_MIN_RANGE_PCT)
OPPORTUNITY_MODE_SOFT_MIN_SCORE = float(os.getenv("OPPORTUNITY_MODE_SOFT_MIN_SCORE", "0.5") or 0.5)
TIERED_UNIVERSE_ENABLED = _env_bool("TIERED_UNIVERSE_ENABLED", True)
TIER1_PREFERRED_BASES = [str(s or '').upper() for s in SMART_RANKING_PREFERRED_BASES if str(s or '').strip()]
TIER2_CANDIDATE_BASES = [str(s or '').upper() for s in _csv("TIER2_CANDIDATE_BASES", "AAVE,NEAR,UNI,SUI,ONDO,TON,BCH,LTC,XRP,DOGE,CRV,COMP,MORPHO,NIGHT,MON,PENGU,FET,ENA,GWEI") if str(s or '').strip()]
TIER2_TARGET_ACTIVE = int(os.getenv("TIER2_TARGET_ACTIVE", str(OPPORTUNITY_MODE_TARGET_ACTIVE or 4)) or (OPPORTUNITY_MODE_TARGET_ACTIVE or 4))
TIER2_MIN_SCORE = float(os.getenv("TIER2_MIN_SCORE", "4.0") or 4.0)
TIER2_MIN_USD_VOL = float(os.getenv("TIER2_MIN_USD_VOL", str(max(STRICT_MIN_24H_USD_VOL, FALLBACK_MIN_24H_USD_VOL))) or max(STRICT_MIN_24H_USD_VOL, FALLBACK_MIN_24H_USD_VOL))
TIER2_MIN_RANGE_PCT = float(os.getenv("TIER2_MIN_RANGE_PCT", str(max(STRICT_MIN_24H_RANGE_PCT, ATR_ACTIVE_MIN_RANGE_PCT))) or max(STRICT_MIN_24H_RANGE_PCT, ATR_ACTIVE_MIN_RANGE_PCT))

def _apply_final_emit_hard_filter(candidate_symbols: List[str]) -> tuple[List[str], Dict[str, Any]]:
    symbols = [str(s or '').upper() for s in (candidate_symbols or []) if str(s or '').strip()]
    preferred = {str(s or '').upper() for s in SMART_RANKING_PREFERRED_BASES if str(s or '').strip()}
    if not SMART_RANKING_FINAL_EMIT_HARD_FILTER_ENABLED or not preferred:
        return symbols, {
            "enabled": bool(SMART_RANKING_FINAL_EMIT_HARD_FILTER_ENABLED),
            "preferred_bases": list(SMART_RANKING_PREFERRED_BASES),
            "before_count": len(symbols),
            "after_count": len(symbols),
            "removed_symbols": [],
        }
    filtered = []
    removed = []
    for sym in symbols:
        base = _base(sym)
        if base in preferred:
            filtered.append(sym)
        else:
            removed.append(sym)
    return filtered, {
        "enabled": True,
        "preferred_bases": list(SMART_RANKING_PREFERRED_BASES),
        "before_count": len(symbols),
        "after_count": len(filtered),
        "removed_symbols": removed[:24],
    }


def _apply_opportunity_mode(candidate_symbols: List[str], best_any: Dict[str, Tuple[str, float, List[str], float, float]], scored_lookup: Dict[str, Tuple[float, List[str]]], backfill_any: Optional[Dict[str, Tuple[str, float, List[str], float, float]]] = None, soft_backfill_any: Optional[Dict[str, Tuple[str, float, List[str], float, float]]] = None) -> tuple[List[str], Dict[str, Any]]:
    symbols = [str(s or '').upper() for s in (candidate_symbols or []) if str(s or '').strip()]
    target = max(0, int(TIER2_TARGET_ACTIVE if TIERED_UNIVERSE_ENABLED else OPPORTUNITY_MODE_TARGET_ACTIVE))
    preferred = [str(s or '').upper() for s in (OPPORTUNITY_MODE_PREFERRED_BASES or []) if str(s or '').strip()]
    tier2_bases = [str(s or '').upper() for s in (TIER2_CANDIDATE_BASES or []) if str(s or '').strip()]
    meta = {
        "enabled": bool(OPPORTUNITY_MODE_ENABLED),
        "tiered_universe_enabled": bool(TIERED_UNIVERSE_ENABLED),
        "target_active": target,
        "preferred_bases": preferred,
        "tier2_bases": tier2_bases,
        "before_count": len(symbols),
        "after_count": len(symbols),
        "added_symbols": [],
        "tier2_added_symbols": [],
        "vetted_symbols": list(symbols),
    }
    if not OPPORTUNITY_MODE_ENABLED or target <= 0 or len(symbols) >= target:
        return symbols, meta
    chosen = set(symbols)
    fallback_any = backfill_any or {}
    soft_any = soft_backfill_any or {}

    def _choose_item(base: str):
        return best_any.get(base) or fallback_any.get(base) or soft_any.get(base)

    def _append(sym: str, total: float, reasons: List[str], extra_reason: str, *, tier2: bool = False):
        symbols.append(sym)
        chosen.add(sym)
        meta["added_symbols"].append(sym)
        if tier2:
            meta["tier2_added_symbols"].append(sym)
        if sym not in scored_lookup:
            scored_lookup[sym] = (float(total), list(reasons) + [extra_reason])
        else:
            sc, rs = scored_lookup[sym]
            rs2 = list(rs)
            if extra_reason not in rs2:
                rs2.append(extra_reason)
            scored_lookup[sym] = (float(sc), rs2)

    for base in preferred:
        if len(symbols) >= target:
            break
        item = _choose_item(base)
        if not item:
            continue
        sym, total, reasons, usd_vol, rng = item
        sym = str(sym or '').upper()
        if not sym or sym in chosen or float(total) < float(OPPORTUNITY_MODE_MIN_BACKFILL_SCORE):
            continue
        extra_reason = "opportunity_mode"
        if base not in best_any and base in fallback_any:
            extra_reason = "opportunity_backfill"
        elif base not in best_any and base in soft_any:
            extra_reason = "opportunity_soft_backfill"
        _append(sym, total, list(reasons), extra_reason, tier2=False)

    if TIERED_UNIVERSE_ENABLED and len(symbols) < target:
        tier2_pool = []
        for base in tier2_bases:
            item = _choose_item(base)
            if not item:
                continue
            sym, total, reasons, usd_vol, rng = item
            sym = str(sym or '').upper()
            if not sym or sym in chosen:
                continue
            if float(total) < float(TIER2_MIN_SCORE):
                continue
            if float(usd_vol) < float(TIER2_MIN_USD_VOL):
                continue
            if float(rng) < float(TIER2_MIN_RANGE_PCT):
                continue
            if 'tight_spread' not in set(str(r or '').strip() for r in (reasons or [])):
                continue
            tier2_pool.append((sym, float(total), list(reasons), float(usd_vol), float(rng)))
        tier2_pool.sort(key=lambda x: (x[1], x[3], x[4]), reverse=True)
        for sym, total, reasons, usd_vol, rng in tier2_pool:
            if len(symbols) >= target:
                break
            _append(sym, total, reasons, 'tier2_opportunity', tier2=True)

    meta["after_count"] = len(symbols)
    meta["vetted_symbols"] = list(symbols)
    return symbols, meta

def _quality_gate(total: float, reasons: List[str]) -> tuple[bool, Dict[str, Any]]:
    reason_list = [str(r or '').strip() for r in (reasons or []) if str(r or '').strip()]
    structural = [r for r in reason_list if r not in ("fallback_pool", "majors_floor")]
    structural_set = set(structural)
    has_range = any(r.startswith("range_24h_") for r in structural_set)
    has_volume = any(r.startswith("vol_24h_") for r in structural_set)
    reason_count = len(structural_set)
    tight_spread_only = structural_set == {"tight_spread"}
    atr_and_spread_only = structural_set == {"atr_active", "tight_spread"}
    failures: List[str] = []
    if float(total) < float(SCANNER_MIN_SCORE):
        failures.append("min_score")
    if int(reason_count) < int(SCANNER_MIN_REASON_COUNT):
        failures.append("min_reason_count")
    if bool(SCANNER_REQUIRE_RANGE_OR_VOLUME) and not (has_range or has_volume):
        failures.append("range_or_volume_required")
    if bool(SCANNER_DROP_TIGHT_SPREAD_ONLY) and tight_spread_only:
        failures.append("tight_spread_only")
    if bool(SCANNER_DROP_ATR_AND_SPREAD_ONLY) and atr_and_spread_only:
        failures.append("atr_and_spread_only")
    return (len(failures) == 0), {
        "pass": len(failures) == 0,
        "failures": failures,
        "score": float(total),
        "reason_count": int(reason_count),
        "has_range": bool(has_range),
        "has_volume": bool(has_volume),
        "tight_spread_only": bool(tight_spread_only),
        "atr_and_spread_only": bool(atr_and_spread_only),
        "config": {
            "min_score": float(SCANNER_MIN_SCORE),
            "min_reason_count": int(SCANNER_MIN_REASON_COUNT),
            "require_range_or_volume": bool(SCANNER_REQUIRE_RANGE_OR_VOLUME),
            "drop_tight_spread_only": bool(SCANNER_DROP_TIGHT_SPREAD_ONLY),
            "drop_atr_and_spread_only": bool(SCANNER_DROP_ATR_AND_SPREAD_ONLY),
        },
    }

def _env_symbol_list(name: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in str(os.getenv(name, "") or "").split(','):
        sym = str(item or '').strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _normalize_emit_symbol(sym: str) -> str:
    s = str(sym or '').strip().upper().replace('-', '/').replace(':', '/')
    if not s:
        return ''
    if '/' not in s:
        for q in sorted(QUOTE_ALLOW, key=len, reverse=True):
            if s.endswith(q):
                s = f"{s[:-len(q)]}/{q}"
                break
    if '/' not in s:
        return s
    base, quote = s.split('/', 1)
    alias_map = {
        'XBT': 'BTC',
        'XXBT': 'BTC',
        'XXBTZ': 'BTC',
        'XETH': 'ETH',
        'XXETH': 'ETH',
        'XETHZ': 'ETH',
        'XSOL': 'SOL',
        'XXSOL': 'SOL',
        'XSOLZ': 'SOL',
    }
    base = alias_map.get(base, base)
    quote = quote.upper()
    return f"{base}/{quote}"


def _alignment_symbol_candidates(sym: str) -> List[str]:
    n = _normalize_emit_symbol(sym)
    if not n:
        return []
    out: List[str] = []
    seen = set()
    def add(v: str):
        v = str(v or '').strip().upper()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    add(n)
    if '/' in n:
        base, quote = n.split('/', 1)
        alias_bases = {
            'BTC': ['XBT', 'XXBT', 'XXBTZ'],
            'ETH': ['XETH', 'XXETH', 'XETHZ'],
            'SOL': ['XSOL', 'XXSOL', 'XSOLZ'],
            'XBT': ['BTC', 'XXBT', 'XXBTZ'],
        }
        for alias_base in alias_bases.get(base, []):
            add(f"{alias_base}/{quote}")
            add(f"{alias_base}{quote}")
            if quote == 'USD':
                add(f"{alias_base}ZUSD")
        add(f"{base}{quote}")
        if quote == 'USD':
            add(f"{base}ZUSD")
    return out


def _resolve_force_emit_symbol(sym: str, tradable_symbols: set[str], scored_lookup: Dict[str, Tuple[float, List[str]]]) -> tuple[str | None, Dict[str, Any]]:
    tradable = {str(s or '').upper() for s in (tradable_symbols or set())}
    scored = {str(s or '').upper() for s in (scored_lookup or {}).keys()}
    candidates = _alignment_symbol_candidates(sym)
    for cand in candidates:
        if cand in tradable or cand in scored:
            return cand, {"requested": str(sym or '').strip().upper(), "normalized": _normalize_emit_symbol(sym), "resolved": cand, "resolution": "matched"}
    normalized = _normalize_emit_symbol(sym)
    if '/' in normalized:
        base, quote = normalized.split('/', 1)
        if quote in QUOTE_ALLOW and base in ('BTC', 'ETH', 'SOL'):
            return normalized, {
                "requested": str(sym or '').strip().upper(),
                "normalized": normalized,
                "resolved": normalized,
                "resolution": f"{base.lower()}_alias_fallback",
            }
    return None, {"requested": str(sym or '').strip().upper(), "normalized": normalized, "resolved": None, "resolution": "unresolved"}


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


def _pilot_force_emit_symbols() -> List[str]:
    requested = _env_symbol_list("SCANNER_FORCE_EMIT_SYMBOLS") or _env_symbol_list("BTC_ONLY_ALIGNMENT_SYMBOLS")
    contract_safe_7 = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "LINK/USD", "AVAX/USD", "DOT/USD"]
    contract_unsafe_10 = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "XRP/USD", "DOGE/USD", "LINK/USD", "AVAX/USD", "LTC/USD", "DOT/USD"]
    if requested == ["BTC/USD"]:
        return ["BTC/USD", "ETH/USD", "SOL/USD"]
    if requested == ["BTC/USD", "ETH/USD", "SOL/USD"]:
        return contract_safe_7
    if requested == contract_unsafe_10:
        return contract_safe_7
    return requested

def _scanner_alignment_config() -> Dict[str, Any]:
    alignment_enabled = _env_bool("BTC_ONLY_ALIGNMENT_ENABLED", False) or _env_bool("SCANNER_ALIGNMENT_ENABLED", False)
    emit_only = _env_bool("SCANNER_EMIT_ONLY_SYMBOLS", False) or _env_bool("BTC_ONLY_ALIGNMENT_EMIT_ONLY", False)
    force_symbols = _pilot_force_emit_symbols()
    if emit_only and force_symbols:
        alignment_enabled = True
    return {
        "enabled": bool(alignment_enabled),
        "emit_only": bool(emit_only),
        "force_emit_symbols": force_symbols,
        "mode": "emit_only" if emit_only else ("prepend_force_symbols" if alignment_enabled and force_symbols else ("enabled_no_force_symbols" if alignment_enabled else "off")),
    }


def _apply_alignment(candidate_symbols: List[str], scored_lookup: Dict[str, Tuple[float, List[str]]], tradable_symbols: set[str]) -> tuple[List[str], Dict[str, Any]]:
    cfg = _scanner_alignment_config()
    pre = []
    seen = set()
    for sym in candidate_symbols or []:
        s = str(sym or '').strip().upper()
        if not s or s in seen:
            continue
        seen.add(s)
        pre.append(s)
    force_requested = list(cfg.get("force_emit_symbols") or [])
    force_valid: List[str] = []
    force_invalid: List[str] = []
    force_resolution: List[Dict[str, Any]] = []
    for sym in force_requested:
        resolved, info = _resolve_force_emit_symbol(sym, tradable_symbols, scored_lookup)
        force_resolution.append(info)
        if resolved:
            if resolved not in force_valid:
                force_valid.append(resolved)
        else:
            requested = str(sym or '').strip().upper()
            if requested and requested not in force_invalid:
                force_invalid.append(requested)
    out: List[str] = []
    seen_out = set()
    if cfg.get("emit_only"):
        for sym in force_valid:
            if sym in seen_out:
                continue
            seen_out.add(sym)
            out.append(sym)
    else:
        for sym in force_valid + pre:
            if sym in seen_out:
                continue
            seen_out.add(sym)
            out.append(sym)
    meta = {
        "enabled": bool(cfg.get("enabled")),
        "emit_only": bool(cfg.get("emit_only")),
        "mode": cfg.get("mode"),
        "force_emit_symbols": force_requested,
        "force_emit_symbols_valid": force_valid,
        "force_emit_symbols_invalid": force_invalid,
        "force_emit_resolution": force_resolution,
        "pre_alignment_candidate_count": len(pre),
        "pre_alignment_candidate_sample": pre[:12],
        "post_alignment_candidate_count": len(out),
        "post_alignment_candidate_sample": out[:12],
        "alignment_active": bool(cfg.get("enabled") or cfg.get("emit_only") or force_requested),
    }
    return out, meta


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
    scored_lookup: Dict[str, Tuple[float, List[str]]] = {}

    # Track the best scored pair per base across *all* scored pairs,
    # so we can fill with MAJORS_FLOOR without relaxing thresholds.
    best_any: Dict[str, Tuple[str, float, List[str], float, float]] = {}

    # Best preferred-major candidate after spread filter and positive score, even if strict quality gate fails.
    backfill_any: Dict[str, Tuple[str, float, List[str], float, float]] = {}
    # Soft preferred-major fallback for opportunity expansion when the strict pools are too thin.
    soft_backfill_any: Dict[str, Tuple[str, float, List[str], float, float]] = {}

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

        ranking_bias = 0.0
        ranking_bias_reasons: List[str] = []
        if SMART_RANKING_BIAS_ENABLED:
            ranking_bias, ranking_bias_reasons = score_ranking_bias(
                sym,
                t,
                spread_pct=spread_pct,
                preferred_bases=SMART_RANKING_PREFERRED_BASES,
            )

        total = float(spot_score + bonus + ranking_bias)
        if total <= 0:
            continue

        reasons = spot_reasons + bonus_reasons + ranking_bias_reasons
        b = _base(sym)
        if b in set(OPPORTUNITY_MODE_PREFERRED_BASES):
            prev_backfill = backfill_any.get(b)
            if prev_backfill is None or total > prev_backfill[1] or (total == prev_backfill[1] and usd_vol > prev_backfill[3]) or (total == prev_backfill[1] and usd_vol == prev_backfill[3] and rng > prev_backfill[4]):
                backfill_any[b] = (sym, total, list(reasons), usd_vol, rng)
            if OPPORTUNITY_MODE_SOFT_BACKFILL_ENABLED and float(total) >= float(OPPORTUNITY_MODE_SOFT_MIN_SCORE) and float(usd_vol) >= float(OPPORTUNITY_MODE_SOFT_MIN_USD_VOL) and float(rng) >= float(OPPORTUNITY_MODE_SOFT_MIN_RANGE_PCT):
                prev_soft = soft_backfill_any.get(b)
                soft_reasons = list(reasons) + ["opportunity_soft_pool"]
                if prev_soft is None or total > prev_soft[1] or (total == prev_soft[1] and usd_vol > prev_soft[3]) or (total == prev_soft[1] and usd_vol == prev_soft[3] and rng > prev_soft[4]):
                    soft_backfill_any[b] = (sym, total, soft_reasons, usd_vol, rng)
        quality_ok, quality_meta = _quality_gate(float(total), list(reasons))
        if not quality_ok:
            continue
        scored_lookup[str(sym).upper()] = (float(total), list(reasons))

        # Save best pair for this base regardless of thresholds (for majors floor)
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
    candidate_symbols, alignment_meta = _apply_alignment(candidate_symbols, scored_lookup, set(str(s or '').upper() for s in pairs))
    candidate_symbols, holdoff_meta = _apply_scanner_symbol_holdoff(candidate_symbols)
    candidate_symbols, final_emit_meta = _apply_final_emit_hard_filter(candidate_symbols)
    candidate_symbols, opportunity_meta = _apply_opportunity_mode(candidate_symbols, best_any, scored_lookup, backfill_any, soft_backfill_any)
    active_symbols, emission_meta = _apply_scanner_emission_controls(candidate_symbols)
    top_by_symbol = {str(s).upper(): (sc, rs) for (s, sc, rs, _, _) in filtered_top}
    top_by_symbol.update(scored_lookup)
    scores = {s: float(top_by_symbol[s][0]) for s in active_symbols if s in top_by_symbol}
    reasons = {s: top_by_symbol[s][1] for s in active_symbols if s in top_by_symbol}
    alignment_force_valid = list(alignment_meta.get("force_emit_symbols_valid") or [])
    for s in active_symbols:
        if s in scores:
            continue
        if s in alignment_force_valid:
            scores[s] = 0.0
            reasons[s] = ["alignment_forced_emit"]

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
            "quality_gate": {
                "min_score": float(SCANNER_MIN_SCORE),
                "min_reason_count": int(SCANNER_MIN_REASON_COUNT),
                "require_range_or_volume": bool(SCANNER_REQUIRE_RANGE_OR_VOLUME),
                "drop_tight_spread_only": bool(SCANNER_DROP_TIGHT_SPREAD_ONLY),
                "drop_atr_and_spread_only": bool(SCANNER_DROP_ATR_AND_SPREAD_ONLY),
            },
            "smart_ranking_bias": {
                "enabled": bool(SMART_RANKING_BIAS_ENABLED),
                "preferred_bases": list(SMART_RANKING_PREFERRED_BASES),
            },
            "alignment": alignment_meta,
            "opportunity_mode": opportunity_meta,
            "tiered_universe": {"enabled": bool(TIERED_UNIVERSE_ENABLED), "tier1_bases": list(TIER1_PREFERRED_BASES), "tier2_bases": list(TIER2_CANDIDATE_BASES), "vetted_symbols": list(opportunity_meta.get("vetted_symbols") or []), "tier2_added_symbols": list(opportunity_meta.get("tier2_added_symbols") or [])},
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
            "compatibility": _compatibility_payload_unlocked(),
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



def _scanner_mode() -> str:
    return "ranked_multi_symbol_scanner"


def _guardrails_snapshot() -> Dict[str, Any]:
    return {
        "coordination_enabled": bool(SCANNER_COORDINATION_URL),
        "symbol_holdoff_sec": int(SCANNER_SYMBOL_HOLDOFF_SEC),
        "fingerprint_ttl_sec": int(SCANNER_FINGERPRINT_TTL_SEC),
        "bar_lock_sec": int(SCANNER_BAR_LOCK_SEC),
        "inflight_holdoff_sec": int(SCANNER_INFLIGHT_HOLDOFF_SEC),
        "max_spread_pct": float(MAX_SPREAD_PCT),
        "top_n": int(TOP_N),
        "refresh_sec": int(REFRESH_SEC),
    }


def _safe_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        try:
            return max(0, int(value))
        except Exception:
            return 0
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, (list, tuple, set)):
        return len(value)
    try:
        return len(value)  # type: ignore[arg-type]
    except Exception:
        return 0


def _spread_bps_each_side_from_guardrails() -> float:
    try:
        return max(0.0, float(MAX_SPREAD_PCT) * 10000.0 / 2.0)
    except Exception:
        return 0.0



def _compatibility_payload_unlocked() -> Dict[str, Any]:
    active_symbols = list(CACHE.get("active_symbols") or [])
    raw = dict(CACHE.get("raw") or {})
    telemetry = _suppression_snapshot()
    active_scores = dict(CACHE.get("scores") or {})
    ranked = sorted(active_symbols, key=lambda s: float(active_scores.get(s, 0.0)), reverse=True)
    coord_raw = dict(raw.get("coordination") or {})
    coord_summary = {
        "ok": bool(coord_raw.get("ok")),
        "reason": coord_raw.get("reason"),
        "suppressed_symbols_count": int(coord_raw.get("suppressed_symbols_count") or _safe_count(coord_raw.get("suppressed_symbols"))),
        "suppressed_symbols_sample": list(coord_raw.get("suppressed_symbols") or [])[:12],
        "hard_suppressed_symbols_count": int(coord_raw.get("hard_suppressed_symbols_count") or _safe_count(coord_raw.get("hard_suppressed_symbols"))),
        "hard_suppressed_symbols_sample": list(coord_raw.get("hard_suppressed_symbols") or [])[:12],
        "active_workflow_locks_count": int(coord_raw.get("active_workflow_locks_count") or _safe_count(coord_raw.get("active_workflow_locks"))),
        "recent_admission_passed_count": int(coord_raw.get("recent_admission_passed_count") or _safe_count(coord_raw.get("recent_admission_passed"))),
        "active_signal_fingerprints_count": int(coord_raw.get("active_signal_fingerprints_count") or _safe_count(coord_raw.get("active_signal_fingerprints"))),
    }
    guardrails = _guardrails_snapshot()
    fee_churn_truth = {
        "spread_model": {
            "max_spread_pct": float(MAX_SPREAD_PCT),
            "expected_spread_bps_each_side": round(_spread_bps_each_side_from_guardrails(), 3),
        },
        "churn_model_inputs": {
            "top_n": int(TOP_N),
            "refresh_sec": int(REFRESH_SEC),
            "symbol_holdoff_sec": int(SCANNER_SYMBOL_HOLDOFF_SEC),
            "fingerprint_ttl_sec": int(SCANNER_FINGERPRINT_TTL_SEC),
            "bar_lock_sec": int(SCANNER_BAR_LOCK_SEC),
            "inflight_holdoff_sec": int(SCANNER_INFLIGHT_HOLDOFF_SEC),
        },
    }
    alignment = dict(raw.get("alignment") or {})
    return {
        "scanner_ok": bool(CACHE.get("ts") is not None) and not bool(CACHE.get("last_error")),
        "mode": _scanner_mode(),
        "multi_symbol_capable": True,
        "supports_multi_symbol": True,
        "active_count": len(active_symbols),
        "active_symbols": active_symbols,
        "active_symbols_sample": active_symbols[:12],
        "ranked_active_symbols": ranked[:12],
        "ranked_count": len(ranked),
        "last_refresh_utc": CACHE.get("utc"),
        "last_error": CACHE.get("last_error"),
        "guardrails": guardrails,
        "telemetry": telemetry,
        "suppression_counts": telemetry.get("last_refresh_counts") or {},
        "coordination": coord_summary,
        "coordination_raw": coord_raw,
        "active_symbols_source": "scanner_cache",
        "alignment": alignment,
        "btc_only_live_alignment": {
            "enabled": bool(alignment.get("alignment_active")),
            "emit_only": bool(alignment.get("emit_only")),
            "force_emit_symbols": list(alignment.get("force_emit_symbols") or []),
            "force_emit_resolution": list(alignment.get("force_emit_resolution") or []),
            "active_symbols_all_admissible": bool(active_symbols) and all(str(s or '').upper() in set(list(alignment.get("force_emit_symbols_valid") or []) or list(alignment.get("force_emit_symbols") or [])) for s in active_symbols) if bool(alignment.get("emit_only")) else False,
        },
        "fee_churn_truth": fee_churn_truth,
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
            },
        }




@app.get("/compatibility")
def compatibility_endpoint():
    with _CACHE_LOCK:
        payload = _compatibility_payload_unlocked()
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
            "compatibility": payload,
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
            "compatibility": _compatibility_payload_unlocked(),
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
                    "compatibility": _compatibility_payload_unlocked(),
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
            "compatibility": _compatibility_payload_unlocked(),
        }
