"""ICT investigation utilities producing JSON-ready multi-timeframe snapshots."""

from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf


ET = ZoneInfo("US/Eastern")
TIMEFRAMES = {
    "weekly": {"interval": "1wk", "period": "5y"},
    "daily": {"interval": "1d", "period": "2y"},
    "1h": {"interval": "1h", "period": "90d"},
    "30m": {"interval": "30m", "period": "60d"},
    "15m": {"interval": "15m", "period": "30d"},
}

INTENT_PATTERN = re.compile(r"\bict\s+investigation\b", flags=re.IGNORECASE)
TICKER_PATTERN = re.compile(r"\b(?:for|on)\s+([A-Za-z.\-]{1,15})\b", flags=re.IGNORECASE)
DATE_PATTERN = re.compile(r"\b(?:as\s+of|on|date)\s+(\d{4}-\d{2}-\d{2})\b", flags=re.IGNORECASE)


def extract_ict_entity(prompt: str) -> str:
    """Extract ticker from prompts like 'ICT investigation for INFY on 2025-08-01'."""
    if not prompt or not INTENT_PATTERN.search(prompt):
        return ""

    match = TICKER_PATTERN.search(prompt)
    if not match:
        return ""
    return match.group(1).upper()


def extract_ict_date(prompt: str) -> str | None:
    """Extract optional as-of date in YYYY-MM-DD format."""
    if not prompt or not INTENT_PATTERN.search(prompt):
        return None

    match = DATE_PATTERN.search(prompt)
    if not match:
        return None

    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def _safe_float(v, n=4):
    if v is None or pd.isna(v):
        return None
    return round(float(v), n)


def _compute_ipda(ohlc: pd.DataFrame, n: int) -> dict:
    if len(ohlc) < n:
        return {
            f"ipda{n}_high": None,
            f"ipda{n}_low": None,
            f"ipda{n}_pct": None,
            f"ipda{n}_state": None,
        }

    hi = ohlc["high"].rolling(window=n, min_periods=n).max().iloc[-1]
    lo = ohlc["low"].rolling(window=n, min_periods=n).min().iloc[-1]
    close = float(ohlc["close"].iloc[-1])
    if pd.isna(hi) or pd.isna(lo) or hi == lo:
        pct = None
        state = None
    else:
        pct = float(np.clip((close - lo) / (hi - lo), 0.0, 1.0) * 100.0)
        state = "premium" if close >= ((hi + lo) / 2.0) else "discount"

    return {
        f"ipda{n}_high": _safe_float(hi),
        f"ipda{n}_low": _safe_float(lo),
        f"ipda{n}_pct": _safe_float(pct, 2),
        f"ipda{n}_state": state,
    }


def _fetch_ohlc(ticker: str, interval: str, period: str, as_of: str | None) -> pd.DataFrame:
    df = yf.download(
        ticker,
        interval=interval,
        period=period,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns.to_flat_index()]

    ohlc = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    ohlc.columns = ["open", "high", "low", "close", "volume"]
    idx = pd.to_datetime(ohlc.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(ET).tz_localize(None)
    ohlc.index = idx

    if as_of:
        as_of_ts = pd.to_datetime(as_of).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
        ohlc = ohlc.loc[ohlc.index <= as_of_ts]

    return ohlc


def _build_tf_snapshot(ohlc: pd.DataFrame) -> dict:
    if ohlc.empty:
        return {"error": "no data"}

    last = ohlc.iloc[-1]
    ipda = {}
    for n in (20, 40, 60):
        ipda.update(_compute_ipda(ohlc, n))

    return {
        "as_of": pd.Timestamp(ohlc.index[-1]).strftime("%Y-%m-%d %H:%M:%S"),
        "last_bar": {
            "open": _safe_float(last.get("open")),
            "high": _safe_float(last.get("high")),
            "low": _safe_float(last.get("low")),
            "close": _safe_float(last.get("close")),
            "volume": _safe_float(last.get("volume"), 0),
        },
        "ipda": ipda,
    }


def run_mtf_ict_snapshot(ticker: str, as_of: str | None = None, route_name: str = "generic_ict_route") -> dict:
    """Run MTF snapshot (JSON-ready) for the requested ticker/date."""
    result = {
        "ok": True,
        "route": route_name,
        "ticker": (ticker or "").upper(),
        "generated_at_utc": datetime.utcnow().isoformat(),
        "as_of_input": as_of,
        "timeframes": {},
    }

    if not result["ticker"]:
        result["ok"] = False
        result["error"] = "No stock/ticker provided."
        return result

    try:
        for tf_name, cfg in TIMEFRAMES.items():
            ohlc = _fetch_ohlc(result["ticker"], cfg["interval"], cfg["period"], as_of)
            result["timeframes"][tf_name] = _build_tf_snapshot(ohlc)

        if all("error" in tf for tf in result["timeframes"].values()):
            result["ok"] = False
            result["error"] = "No timeframe data available from yfinance."
    except Exception as exc:
        result["ok"] = False
        result["error"] = str(exc)

    return result


def run_infy_route(as_of: str | None = None) -> dict:
    """Dedicated INFY route implementation."""
    return run_mtf_ict_snapshot("INFY", as_of=as_of, route_name="infy_ict_route")


def run_ict_investigation(ticker: str, as_of: str | None = None) -> dict:
    """Compatibility wrapper expected by app.py; returns JSON payload."""
    normalized = (ticker or "").upper().strip()
    if normalized == "INFY":
        return run_infy_route(as_of=as_of)
    return run_mtf_ict_snapshot(normalized, as_of=as_of, route_name="ict_investigation_route")
