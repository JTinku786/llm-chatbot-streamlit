"""ICT investigation routing based on SMC feature pipeline (JSON output)."""

from __future__ import annotations

import importlib
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

ET = ZoneInfo("US/Eastern")

SWING_LEN = 4
SWING_LEN_OB = 10
SWING_LEN_BC = 3
SWING_LEN_LQ = 3

ATR_WIN = 14
BIG_BODY_ATR_MULT = 1.25
HUGE_WICK_TO_BODY = 2.0
CONSOLID_WIN = 10
CONSOLIDATR_MULT = 0.75
FOLLOW_THROUGH_BARS = 3
FAIL_FAST_BARS = 3
NEAR_PCT = 0.004

TIMEFRAMES = {
    "weekly": {"interval": "1wk", "period": "5y"},
    "daily": {"interval": "1d", "period": "2y"},
    "1h": {"interval": "1h", "period": "90d"},
    "30m": {"interval": "30m", "period": "60d"},
    "15m": {"interval": "15m", "period": "30d"},
}


def _smc():
    return importlib.import_module("smartmoneyconcepts").smc


def extract_ict_entity(prompt: str) -> str:
    """Recommended format: ICT investigation for {STOCK}."""
    if not prompt:
        return ""
    strict = re.search(r"\bict\s+investigation\s+for\s+([A-Za-z][A-Za-z0-9.\-]{0,14})\b", prompt, flags=re.IGNORECASE)
    if strict:
        return strict.group(1).upper()
    fallback = re.search(r"\bict\s+investigation(?:\s+(?:for|on|about|into|:))?\s+([A-Za-z][A-Za-z0-9.\-]{0,14})\b", prompt, flags=re.IGNORECASE)
    return fallback.group(1).upper() if fallback else ""


def extract_ict_date(prompt: str) -> str | None:
    if not prompt:
        return None
    match = re.search(r"\b(?:as\s+of|on|date)\s+(\d{4}-\d{2}-\d{2})\b", prompt, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def _to_et(ts):
    if ts is None:
        return None
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(ET)


def _to_et_date_or_none(ts):
    if ts is None or pd.isna(ts):
        return None
    return _to_et(ts).strftime("%Y-%m-%d")


def _end_of_day_et(date_str: str):
    return pd.to_datetime(date_str).tz_localize(ET).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


def fetch_ohlc_yf(ticker: str, interval: str, period: str, as_of_str: str | None):
    df = yf.download(ticker, interval=interval, period=period, auto_adjust=False, group_by="column", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns.to_flat_index()]
    ohlc = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    ohlc.columns = ["open", "high", "low", "close", "volume"]
    ohlc = ohlc.astype({"open": "float64", "high": "float64", "low": "float64", "close": "float64", "volume": "float64"})
    idx = pd.to_datetime(ohlc.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    ohlc.index = idx.tz_convert(ET)
    if as_of_str:
        ohlc = ohlc.loc[ohlc.index <= _end_of_day_et(as_of_str)]
    return ohlc


def _resolve_row_timestamp(row, hist_index):
    idx_val = row.name
    if isinstance(idx_val, pd.Timestamp):
        return idx_val
    if isinstance(idx_val, (np.integer, int)):
        return hist_index[max(0, min(int(idx_val), len(hist_index) - 1))]
    for key in ("Index", "CreatedIndex", "StartIndex"):
        if key in row and pd.notna(row[key]):
            try:
                return hist_index[max(0, min(int(row[key]), len(hist_index) - 1))]
            except Exception:
                pass
    return idx_val if idx_val in hist_index else None


def _resolve_index_pos(val, hist_index):
    if isinstance(val, (np.integer, int)):
        pos = int(val)
        return pos if 0 <= pos < len(hist_index) else None
    try:
        ts = pd.Timestamp(val)
        loc = hist_index.get_loc(ts)
        if isinstance(loc, slice) or not isinstance(loc, (int, np.integer)):
            loc = int(np.where(hist_index == ts)[0][0])
        return loc
    except Exception:
        return None


def _alive_by_full_fill(hist: pd.DataFrame, fvg_df: pd.DataFrame) -> pd.DataFrame:
    if fvg_df is None or fvg_df.empty:
        return fvg_df
    df = fvg_df[fvg_df["FVG"].notna()].copy()
    if df.empty:
        return df
    keep_idx = []
    for idx_row, row in df.iterrows():
        if "Top" not in row or "Bottom" not in row:
            continue
        top = float(row["Top"])
        bot = float(row["Bottom"])
        crt_ts = _resolve_row_timestamp(row, hist.index)
        if crt_ts is None:
            keep_idx.append(idx_row)
            continue
        try:
            pos = hist.index.get_loc(crt_ts)
            if isinstance(pos, slice) or not isinstance(pos, (int, np.integer)):
                pos = int(np.where(hist.index == crt_ts)[0][0])
        except KeyError:
            pos = max(0, hist.index.searchsorted(crt_ts))
        sub = hist.iloc[pos + 1 :]
        if sub.empty:
            keep_idx.append(idx_row)
            continue
        if not ((sub["high"].max() >= top) and (sub["low"].min() <= bot)):
            keep_idx.append(idx_row)
    return df.loc[keep_idx]


def _gap_fill_pct(hist: pd.DataFrame, gap_row, top: float, bot: float, mode: str):
    crt_ts = _resolve_row_timestamp(gap_row, hist.index)
    if crt_ts is None:
        return None
    try:
        pos = hist.index.get_loc(crt_ts)
        if isinstance(pos, slice) or not isinstance(pos, (int, np.integer)):
            pos = int(np.where(hist.index == crt_ts)[0][0])
    except KeyError:
        pos = max(0, hist.index.searchsorted(crt_ts))
    sub = hist.iloc[pos + 1 :]
    if sub.empty:
        return 0.0
    width = max(float(top) - float(bot), 1e-12)
    if mode == "above":
        progress = max(0.0, min(float(sub["high"].max()), float(top)) - float(bot))
    else:
        progress = max(0.0, float(top) - max(float(bot), float(sub["low"].min())))
    return round(max(0.0, min(1.0, progress / width)) * 100.0, 2)


def _pick_nearest_above(fvg_df, ref):
    cands = fvg_df[(fvg_df["Bottom"] > ref) & fvg_df["Bottom"].notna() & fvg_df["Top"].notna()].copy()
    if cands.empty:
        return None
    cands["dist"] = (cands["Bottom"] - ref).abs()
    cands["_ts"] = pd.to_datetime(cands.apply(lambda r: _resolve_row_timestamp(r, fvg_df.index), axis=1))
    row = cands.sort_values(["dist", "_ts"], ascending=[True, False]).iloc[0]
    return float(row["Top"]), float(row["Bottom"]), row


def _pick_nearest_below(fvg_df, ref):
    cands = fvg_df[(fvg_df["Top"] < ref) & fvg_df["Bottom"].notna() & fvg_df["Top"].notna()].copy()
    if cands.empty:
        return None
    cands["dist"] = (cands["Top"] - ref).abs()
    cands["_ts"] = pd.to_datetime(cands.apply(lambda r: _resolve_row_timestamp(r, fvg_df.index), axis=1))
    row = cands.sort_values(["dist", "_ts"], ascending=[True, False]).iloc[0]
    return float(row["Top"]), float(row["Bottom"]), row


def _alive_swings_by_mitigation(hist: pd.DataFrame, sw_df: pd.DataFrame) -> pd.DataFrame:
    if sw_df is None or sw_df.empty:
        return sw_df
    if "HighLow" not in sw_df.columns or "Level" not in sw_df.columns:
        return pd.DataFrame(columns=sw_df.columns)
    df = sw_df[sw_df["HighLow"].notna()].copy()
    df["Level"] = pd.to_numeric(df["Level"], errors="coerce")
    df = df[df["Level"].notna()]
    if df.empty:
        return df
    keep_idx = []
    for idx_row, row in df.iterrows():
        lvl = float(row["Level"])
        hl = int(row["HighLow"])
        crt_ts = _resolve_row_timestamp(row, hist.index)
        if crt_ts is None:
            keep_idx.append(idx_row)
            continue
        try:
            pos = hist.index.get_loc(crt_ts)
            if isinstance(pos, slice) or not isinstance(pos, (int, np.integer)):
                pos = int(np.where(hist.index == crt_ts)[0][0])
        except KeyError:
            pos = max(0, hist.index.searchsorted(crt_ts))
        sub = hist.iloc[pos + 1 :]
        if sub.empty:
            keep_idx.append(idx_row)
            continue
        mitigated = (sub["high"] >= lvl).any() if hl == 1 else (sub["low"] <= lvl).any()
        if not mitigated:
            keep_idx.append(idx_row)
    return df.loc[keep_idx]


def _pick_nearest_swing_above(sw_df: pd.DataFrame, ref: float):
    c = sw_df[(sw_df["HighLow"] == 1) & (pd.to_numeric(sw_df["Level"], errors="coerce") > ref)].copy()
    if c.empty:
        return None
    c["Level"] = pd.to_numeric(c["Level"], errors="coerce")
    c["dist"] = (c["Level"] - ref).abs()
    c["_ts"] = pd.to_datetime(c.apply(lambda r: _resolve_row_timestamp(r, c.index), axis=1))
    row = c.sort_values(["dist", "_ts"], ascending=[True, False]).iloc[0]
    return float(row["Level"]), row


def _pick_nearest_swing_below(sw_df: pd.DataFrame, ref: float):
    c = sw_df[(sw_df["HighLow"] == -1) & (pd.to_numeric(sw_df["Level"], errors="coerce") < ref)].copy()
    if c.empty:
        return None
    c["Level"] = pd.to_numeric(c["Level"], errors="coerce")
    c["dist"] = (c["Level"] - ref).abs()
    c["_ts"] = pd.to_datetime(c.apply(lambda r: _resolve_row_timestamp(r, c.index), axis=1))
    row = c.sort_values(["dist", "_ts"], ascending=[True, False]).iloc[0]
    return float(row["Level"]), row


def _prepare_ob_frames(hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    smc = _smc()
    sw_ob = smc.swing_highs_lows(hist, swing_length=SWING_LEN_OB)
    ob_df = smc.ob(hist, sw_ob, close_mitigation=False)
    if ob_df is None or ob_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    needed = {"OB", "Top", "Bottom", "MitigatedIndex"}
    if not needed.issubset(set(ob_df.columns)):
        return pd.DataFrame(), pd.DataFrame()
    df = ob_df.dropna(subset=["OB"]).copy()
    df["Top"] = pd.to_numeric(df["Top"], errors="coerce")
    df["Bottom"] = pd.to_numeric(df["Bottom"], errors="coerce")
    df = df[df["Top"].notna() & df["Bottom"].notna()]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df["MitigatedIndex"] = pd.to_numeric(df["MitigatedIndex"], errors="coerce").fillna(0).astype(int)
    return df[df["MitigatedIndex"] == 0].copy(), df[df["MitigatedIndex"] != 0].copy()


def _pick_nearest_ob_above(ob_df: pd.DataFrame, ref: float):
    cands = ob_df[ob_df["Bottom"] > ref].copy()
    if cands.empty:
        return None
    cands["dist"] = (cands["Bottom"] - ref).abs()
    cands["_ts"] = pd.to_datetime(cands.apply(lambda r: _resolve_row_timestamp(r, ob_df.index), axis=1))
    row = cands.sort_values(["dist", "_ts"], ascending=[True, False]).iloc[0]
    return float(row["Top"]), float(row["Bottom"]), row


def _pick_nearest_ob_below(ob_df: pd.DataFrame, ref: float):
    cands = ob_df[ob_df["Top"] < ref].copy()
    if cands.empty:
        return None
    cands["dist"] = (cands["Top"] - ref).abs()
    cands["_ts"] = pd.to_datetime(cands.apply(lambda r: _resolve_row_timestamp(r, ob_df.index), axis=1))
    row = cands.sort_values(["dist", "_ts"], ascending=[True, False]).iloc[0]
    return float(row["Top"]), float(row["Bottom"]), row


def _prepare_bos_frame(hist: pd.DataFrame) -> pd.DataFrame:
    smc = _smc()
    sw_bc = smc.swing_highs_lows(hist, swing_length=SWING_LEN_BC)
    bc = smc.bos_choch(hist, sw_bc)
    if bc is None or bc.empty:
        return pd.DataFrame()
    df = bc.copy()
    df["BOS"] = pd.to_numeric(df.get("BOS", np.nan), errors="coerce")
    df["CHOCH"] = pd.to_numeric(df.get("CHOCH", np.nan), errors="coerce")
    df["Level"] = pd.to_numeric(df.get("Level", np.nan), errors="coerce")
    df["BrokenIndex"] = pd.to_numeric(df.get("BrokenIndex", np.nan), errors="coerce")
    df = df[df["Level"].notna() & (df["BOS"].isin([-1, 1]) | df["CHOCH"].isin([-1, 1])) & df["BrokenIndex"].notna()]
    if df.empty:
        return df
    df["BrokenIndex"] = df["BrokenIndex"].astype(float).astype(int)
    valid_rows = [idx_row for idx_row, row in df.iterrows() if 0 <= int(row["BrokenIndex"]) < len(hist)]
    return df.loc[valid_rows]


def _resolve_broken_timestamp(row, hist_index):
    pos = _resolve_index_pos(row.get("BrokenIndex", np.nan), hist_index)
    return None if pos is None else hist_index[pos]


def _atr(df: pd.DataFrame, win: int = ATR_WIN) -> pd.Series:
    prev_c = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]), (df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(win, min_periods=1).mean()


def _body_wicks(bar) -> tuple[float, float, float]:
    o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
    return abs(c - o), h - max(o, c), min(o, c) - l


def _pct_dist(p1: float, p2: float) -> float:
    return 0.0 if p2 == 0 else abs(p1 - p2) / p2


def _recent_swing_level(hist: pd.DataFrame, pos: int, swing_len: int = SWING_LEN_BC, want: int = 1):
    smc = _smc()
    sw = smc.swing_highs_lows(hist.iloc[:pos], swing_length=swing_len)
    if sw is None or sw.empty or "HighLow" not in sw.columns or "Level" not in sw.columns:
        return None, None
    sw = sw[sw["HighLow"] == want]
    if sw.empty:
        return None, None
    if isinstance(sw.index, pd.DatetimeIndex):
        idx_pos = hist.index.get_loc(sw.index[-1])
    else:
        idx_pos = int(sw.index[-1])
    return float(sw["Level"].iloc[-1]), int(idx_pos)


def _swept_level_by_wick(hist: pd.DataFrame, pos: int, direction: int) -> bool:
    want = 1 if direction == -1 else -1
    lvl, _ = _recent_swing_level(hist, pos, want=want)
    if lvl is None:
        return False
    for k in range(max(0, pos - 1), pos + 1):
        bar = hist.iloc[k]
        body, up, lo = _body_wicks(bar)
        if direction == -1:
            if float(bar["high"]) > lvl and float(bar["close"]) < max(float(bar["open"]), lvl) and up >= HUGE_WICK_TO_BODY * max(body, 1e-9):
                return True
        else:
            if float(bar["low"]) < lvl and float(bar["close"]) > min(float(bar["open"]), lvl) and lo >= HUGE_WICK_TO_BODY * max(body, 1e-9):
                return True
    return False


def _pre_consolidation(hist: pd.DataFrame, end_pos: int) -> bool:
    if end_pos < CONSOLID_WIN + 5:
        return False
    atr = _atr(hist)
    pre = atr.iloc[end_pos - CONSOLID_WIN : end_pos].mean()
    base = atr.rolling(50, min_periods=10).median().iloc[end_pos]
    if pd.isna(pre) or pd.isna(base) or base == 0:
        return False
    return pre <= CONSOLIDATR_MULT * base


def _follow_through(hist: pd.DataFrame, end_pos: int, direction: int, bars: int = FOLLOW_THROUGH_BARS) -> bool:
    nxt = hist.iloc[end_pos + 1 : end_pos + 1 + bars]
    if nxt.empty:
        return False
    return nxt["close"].max() > hist["close"].iloc[end_pos] if direction == 1 else nxt["close"].min() < hist["close"].iloc[end_pos]


def _fail_fast(hist: pd.DataFrame, level: float, end_pos: int, direction: int, bars: int = FAIL_FAST_BARS) -> bool:
    nxt = hist.iloc[end_pos + 1 : end_pos + 1 + bars]
    if nxt.empty:
        return False
    return (nxt["close"] < level).any() if direction == 1 else (nxt["close"] > level).any()


def _near_breaker_zone(hist: pd.DataFrame, end_pos: int, price: float) -> bool:
    _, ob_breaker = _prepare_ob_frames(hist.iloc[: end_pos + 1])
    if ob_breaker is None or ob_breaker.empty:
        return False
    for _, r in ob_breaker.iterrows():
        if pd.isna(r["Top"]) or pd.isna(r["Bottom"]):
            continue
        mid = 0.5 * (float(r["Top"]) + float(r["Bottom"]))
        if _pct_dist(mid, price) <= NEAR_PCT:
            return True
    return False


def _classify_bos_reason(hist: pd.DataFrame, bos_row: pd.Series, hist_index: pd.DatetimeIndex) -> tuple[int, dict]:
    val = bos_row.get("BOS", np.nan)
    if pd.isna(val):
        val = bos_row.get("CHOCH", np.nan)
    direction = 1 if float(val) > 0 else -1
    level = float(bos_row.get("Level", np.nan)) if pd.notna(bos_row.get("Level", np.nan)) else None
    formed_ts = _resolve_row_timestamp(bos_row, hist_index)
    end_pos = _resolve_index_pos(bos_row.get("BrokenIndex", np.nan), hist_index)
    if end_pos is None or end_pos <= 0 or end_pos >= len(hist):
        return 7, {"note": "invalid end index"}
    bar = hist.iloc[end_pos]
    body, up, lo = _body_wicks(bar)
    atr_now = _atr(hist).iloc[end_pos]
    big_body = body >= BIG_BODY_ATR_MULT * max(atr_now, 1e-9)
    swept = _swept_level_by_wick(hist, end_pos, direction)
    huge_wick = (up >= HUGE_WICK_TO_BODY * max(body, 1e-9)) if direction == -1 else (lo >= HUGE_WICK_TO_BODY * max(body, 1e-9))
    if swept and huge_wick:
        return 6, {"direction": direction, "formed_ts": formed_ts, "end_pos": end_pos}
    if swept:
        return 2, {"direction": direction, "formed_ts": formed_ts, "end_pos": end_pos}
    if _near_breaker_zone(hist, end_pos, float(bar["close"])):
        return 5, {"direction": direction, "formed_ts": formed_ts, "end_pos": end_pos}
    if _pre_consolidation(hist, end_pos):
        return 4, {"direction": direction, "formed_ts": formed_ts, "end_pos": end_pos}
    if big_body and _follow_through(hist, end_pos, direction):
        return 1, {"direction": direction, "formed_ts": formed_ts, "end_pos": end_pos}
    if level is not None and _fail_fast(hist, level, end_pos, direction) and not _follow_through(hist, end_pos, direction):
        return 3, {"direction": direction, "formed_ts": formed_ts, "end_pos": end_pos, "level": level}
    return 7, {"direction": direction, "formed_ts": formed_ts, "end_pos": end_pos}


def _bos_reason_text(reason_id: int, direction: int, start_date: str | None, end_date: str | None, details: dict) -> str:
    side = "bullish" if direction == 1 else "bearish"
    if reason_id == 1:
        return f"Genuine expansion {side} break. Start={start_date}, End={end_date}."
    if reason_id == 2:
        return f"Liquidity sweep & reversal {side} break. Start={start_date}, End={end_date}."
    if reason_id == 3:
        return f"Market protraction in {side} break. Start={start_date}, End={end_date}."
    if reason_id == 4:
        return f"Consolidation breakout into {side} break. Start={start_date}, End={end_date}."
    if reason_id == 5:
        return f"Breaker block activation for {side} break. Start={start_date}, End={end_date}."
    if reason_id == 6:
        return f"Rejection block into {side} break. Start={start_date}, End={end_date}."
    return f"Other contextual {side} break. Start={start_date}, End={end_date}."


def _extract_recent_bos_events(bos_df: pd.DataFrame, hist_index: pd.DatetimeIndex, n: int = 3):
    if bos_df is None or bos_df.empty:
        return []
    df = bos_df.copy()
    df["_formed_ts"] = pd.to_datetime(df.apply(lambda r: _resolve_row_timestamp(r, hist_index), axis=1), errors="coerce")
    df["_end_pos"] = df.apply(lambda r: _resolve_index_pos(r.get("BrokenIndex", np.nan), hist_index), axis=1)
    df = df[df["_formed_ts"].notna() & df["_end_pos"].notna()]
    if df.empty:
        return []
    df = df.sort_values("_formed_ts", ascending=False).head(n)
    out = []
    for _, r in df.iterrows():
        val = r.get("BOS", np.nan)
        if pd.isna(val) and "CHOCH" in r:
            val = r.get("CHOCH", np.nan)
        val_clean = int(val) if pd.notna(val) else None
        end_pos = int(r["_end_pos"])
        end_ts = hist_index[end_pos] if 0 <= end_pos < len(hist_index) else None
        out.append({"value": val_clean, "start": _to_et_date_or_none(r["_formed_ts"]), "end": _to_et_date_or_none(end_ts), "row": r})
    return out


def _prepare_liquidity_frame(hist: pd.DataFrame) -> pd.DataFrame:
    smc = _smc()
    sw_lq = smc.swing_highs_lows(hist, swing_length=SWING_LEN_LQ)
    liq = smc.liquidity(hist, sw_lq)
    if liq is None or liq.empty:
        return pd.DataFrame()
    df = liq.copy()
    df["Level"] = pd.to_numeric(df.get("Level", np.nan), errors="coerce")
    if "Liquidity" in df.columns:
        df = df[df["Liquidity"].notna()]
    df = df[df["Level"].notna()]
    if df.empty:
        return df
    if "Swept" in df.columns:
        df["Swept"] = pd.to_numeric(df["Swept"], errors="coerce").fillna(0).astype(int)
        df = df[df["Swept"] == 0]
    elif "SweptIndex" in df.columns:
        df = df[df["SweptIndex"].isna()]
    return df


def _pick_nearest_liq_above(liq_df: pd.DataFrame, ref: float, hist_index: pd.DatetimeIndex):
    if liq_df is None or liq_df.empty:
        return None
    c = liq_df[pd.to_numeric(liq_df.get("Level"), errors="coerce") > ref].copy()
    if c.empty:
        return None
    c["Level"] = pd.to_numeric(c["Level"], errors="coerce")
    c["dist"] = (c["Level"] - ref).abs()
    c["_ts"] = pd.to_datetime(c.apply(lambda r: _resolve_row_timestamp(r, hist_index), axis=1))
    row = c.sort_values(["dist", "_ts"], ascending=[True, False]).iloc[0]
    return float(row["Level"]), row


def _pick_nearest_liq_below(liq_df: pd.DataFrame, ref: float, hist_index: pd.DatetimeIndex):
    if liq_df is None or liq_df.empty:
        return None
    c = liq_df[pd.to_numeric(liq_df.get("Level"), errors="coerce") < ref].copy()
    if c.empty:
        return None
    c["Level"] = pd.to_numeric(c["Level"], errors="coerce")
    c["dist"] = (c["Level"] - ref).abs()
    c["_ts"] = pd.to_datetime(c.apply(lambda r: _resolve_row_timestamp(r, hist_index), axis=1))
    row = c.sort_values(["dist", "_ts"], ascending=[True, False]).iloc[0]
    return float(row["Level"]), row


FEAT_COLS = [
    "fvg_above_top", "fvg_above_bottom", "fvg_below_top", "fvg_below_bottom", "fvg_above_date", "fvg_below_date", "fvg_above_fill_pct", "fvg_below_fill_pct",
    "Swing_High", "Swing_High_Date", "Swing_Low", "Swing_Low_Date",
    "OB_Above_Top", "OB_Above_Bottom", "OB_Above_Date", "OB_Below_Top", "OB_Below_Bottom", "OB_Below_Date",
    "Breaker_OB_Above_Top", "Breaker_OB_Above_Bottom", "Breaker_OB_Above_Date", "Breaker_OB_Below_Top", "Breaker_OB_Below_Bottom", "Breaker_OB_Below_Date",
    "Recent_1_BOS_Value", "Recent_1_BOS_Start_Date", "Recent_1_BOS_End_Date", "Recent_1_BOS_Reason", "Recent_1_BOS_Text",
    "Recent_2_BOS_Value", "Recent_2_BOS_Start_Date", "Recent_2_BOS_End_Date", "Recent_2_BOS_Reason", "Recent_2_BOS_Text",
    "Recent_3_BOS_Value", "Recent_3_BOS_Start_Date", "Recent_3_BOS_End_Date", "Recent_3_BOS_Reason", "Recent_3_BOS_Text",
    "liquidity_untouched_above", "liquidity_untouched_below", "liquidity_untouched_above_date", "liquidity_untouched_below_date",
]


def compute_last_row_features(ohlc_et: pd.DataFrame):
    if ohlc_et is None or ohlc_et.empty or len(ohlc_et) < 5:
        return None, None
    ohlc = ohlc_et[["open", "high", "low", "close", "volume"]].copy()
    for c in FEAT_COLS:
        ohlc[c] = None
    smc = _smc()
    idx = ohlc.index
    for i in range(1, len(idx)):
        ts_i = idx[i]
        prev_close = float(ohlc["close"].iloc[i - 1])
        hist = ohlc.iloc[:i][["open", "high", "low", "close", "volume"]].copy()
        if len(hist) < 5:
            continue
        fvg_i = smc.fvg(hist, join_consecutive=False)
        if fvg_i is not None and not fvg_i.empty:
            alive = _alive_by_full_fill(hist, fvg_i)
            if alive is not None and not alive.empty:
                abv = _pick_nearest_above(alive, prev_close)
                bel = _pick_nearest_below(alive, prev_close)
                if abv is not None:
                    top, bottom, row = abv
                    gap_ts = _resolve_row_timestamp(row, hist.index)
                    ohlc.at[ts_i, "fvg_above_top"] = round(top, 2)
                    ohlc.at[ts_i, "fvg_above_bottom"] = round(bottom, 2)
                    ohlc.at[ts_i, "fvg_above_date"] = _to_et_date_or_none(gap_ts)
                    ohlc.at[ts_i, "fvg_above_fill_pct"] = _gap_fill_pct(hist, row, top, bottom, mode="above")
                if bel is not None:
                    top, bottom, row = bel
                    gap_ts = _resolve_row_timestamp(row, hist.index)
                    ohlc.at[ts_i, "fvg_below_top"] = round(top, 2)
                    ohlc.at[ts_i, "fvg_below_bottom"] = round(bottom, 2)
                    ohlc.at[ts_i, "fvg_below_date"] = _to_et_date_or_none(gap_ts)
                    ohlc.at[ts_i, "fvg_below_fill_pct"] = _gap_fill_pct(hist, row, top, bottom, mode="below")

        sw = smc.swing_highs_lows(hist, swing_length=SWING_LEN)
        if sw is not None and not sw.empty:
            if "HighLow" in sw.columns:
                sw = sw[sw["HighLow"].notna()].copy()
            if isinstance(sw.index, pd.DatetimeIndex):
                sw = sw.drop(index=hist.index[-1], errors="ignore")
            elif np.issubdtype(sw.index.dtype, np.integer):
                sw = sw.drop(index=len(hist) - 1, errors="ignore")
            sw_alive = _alive_swings_by_mitigation(hist, sw)
            if sw_alive is not None and not sw_alive.empty:
                sh = _pick_nearest_swing_above(sw_alive, prev_close)
                sl = _pick_nearest_swing_below(sw_alive, prev_close)
                if sh is not None:
                    level, row = sh
                    ohlc.at[ts_i, "Swing_High"] = round(level, 2)
                    ohlc.at[ts_i, "Swing_High_Date"] = _to_et_date_or_none(_resolve_row_timestamp(row, hist.index))
                if sl is not None:
                    level, row = sl
                    ohlc.at[ts_i, "Swing_Low"] = round(level, 2)
                    ohlc.at[ts_i, "Swing_Low_Date"] = _to_et_date_or_none(_resolve_row_timestamp(row, hist.index))

        ob_alive, ob_breaker = _prepare_ob_frames(hist)
        if ob_alive is not None and not ob_alive.empty:
            ob_abv = _pick_nearest_ob_above(ob_alive, prev_close)
            ob_bel = _pick_nearest_ob_below(ob_alive, prev_close)
            if ob_abv is not None:
                top, bottom, row = ob_abv
                ohlc.at[ts_i, "OB_Above_Top"] = round(top, 2)
                ohlc.at[ts_i, "OB_Above_Bottom"] = round(bottom, 2)
                ohlc.at[ts_i, "OB_Above_Date"] = _to_et_date_or_none(_resolve_row_timestamp(row, hist.index))
            if ob_bel is not None:
                top, bottom, row = ob_bel
                ohlc.at[ts_i, "OB_Below_Top"] = round(top, 2)
                ohlc.at[ts_i, "OB_Below_Bottom"] = round(bottom, 2)
                ohlc.at[ts_i, "OB_Below_Date"] = _to_et_date_or_none(_resolve_row_timestamp(row, hist.index))

        if ob_breaker is not None and not ob_breaker.empty:
            br_abv = _pick_nearest_ob_above(ob_breaker, prev_close)
            br_bel = _pick_nearest_ob_below(ob_breaker, prev_close)
            if br_abv is not None:
                top, bottom, row = br_abv
                ohlc.at[ts_i, "Breaker_OB_Above_Top"] = round(top, 2)
                ohlc.at[ts_i, "Breaker_OB_Above_Bottom"] = round(bottom, 2)
                ohlc.at[ts_i, "Breaker_OB_Above_Date"] = _to_et_date_or_none(_resolve_row_timestamp(row, hist.index))
            if br_bel is not None:
                top, bottom, row = br_bel
                ohlc.at[ts_i, "Breaker_OB_Below_Top"] = round(top, 2)
                ohlc.at[ts_i, "Breaker_OB_Below_Bottom"] = round(bottom, 2)
                ohlc.at[ts_i, "Breaker_OB_Below_Date"] = _to_et_date_or_none(_resolve_row_timestamp(row, hist.index))

        bos = _prepare_bos_frame(hist)
        if bos is not None and not bos.empty:
            events = _extract_recent_bos_events(bos, hist.index, n=3)
            for k in range(3):
                col_v = f"Recent_{k+1}_BOS_Value"
                col_s = f"Recent_{k+1}_BOS_Start_Date"
                col_e = f"Recent_{k+1}_BOS_End_Date"
                col_r = f"Recent_{k+1}_BOS_Reason"
                col_t = f"Recent_{k+1}_BOS_Text"
                if k < len(events):
                    ev = events[k]
                    ohlc.at[ts_i, col_v] = ev["value"]
                    ohlc.at[ts_i, col_s] = ev["start"]
                    ohlc.at[ts_i, col_e] = ev["end"]
                    rid, det = _classify_bos_reason(hist, ev["row"], hist.index)
                    ohlc.at[ts_i, col_r] = rid
                    ohlc.at[ts_i, col_t] = _bos_reason_text(rid, 1 if (ev["value"] and ev["value"] > 0) else -1, ev["start"], ev["end"], det)
                else:
                    ohlc.at[ts_i, col_v] = None
                    ohlc.at[ts_i, col_s] = None
                    ohlc.at[ts_i, col_e] = None
                    ohlc.at[ts_i, col_r] = None
                    ohlc.at[ts_i, col_t] = None

        liq = _prepare_liquidity_frame(hist)
        if liq is not None and not liq.empty:
            liq_abv = _pick_nearest_liq_above(liq, prev_close, hist.index)
            liq_bel = _pick_nearest_liq_below(liq, prev_close, hist.index)
            if liq_abv is not None:
                lvl, row = liq_abv
                ohlc.at[ts_i, "liquidity_untouched_above"] = round(float(lvl), 2)
                ohlc.at[ts_i, "liquidity_untouched_above_date"] = _to_et_date_or_none(_resolve_row_timestamp(row, hist.index))
            if liq_bel is not None:
                lvl, row = liq_bel
                ohlc.at[ts_i, "liquidity_untouched_below"] = round(float(lvl), 2)
                ohlc.at[ts_i, "liquidity_untouched_below_date"] = _to_et_date_or_none(_resolve_row_timestamp(row, hist.index))

    last_ts = ohlc.index[-1]
    last = ohlc.iloc[-1].to_dict()
    last = {k: (None if pd.isna(v) else v) for k, v in last.items()}
    return last_ts, last


def run_for_timeframe(name: str, cfg: dict, ticker: str, as_of_str: str | None):
    ohlc = fetch_ohlc_yf(ticker, cfg["interval"], cfg["period"], as_of_str)
    if ohlc.empty or len(ohlc) < 5:
        return {"available": False, "reason": "data not available"}
    last_ts, last_row = compute_last_row_features(ohlc)
    if last_row is None:
        return {"available": False, "reason": "data not available"}
    return {"available": True, "timestamp_et": _to_et(last_ts).isoformat(), "date_et": _to_et(last_ts).strftime("%Y-%m-%d"), "data": last_row}


def run_mtf_ict_snapshot(ticker: str, as_of: str | None = None, route_name: str = "generic_ict_route") -> dict:
    out = {
        "ok": True,
        "route": route_name,
        "ticker": (ticker or "").upper(),
        "as_of": as_of if as_of else None,
        "generated_at_utc": datetime.utcnow().isoformat(),
        "timeframes": {},
    }
    for tf_name, cfg in TIMEFRAMES.items():
        try:
            out["timeframes"][tf_name] = run_for_timeframe(tf_name, cfg, out["ticker"], as_of)
        except Exception as exc:
            out["timeframes"][tf_name] = {"available": False, "reason": f"data not available ({type(exc).__name__}: {exc})"}
    if all(not tf.get("available", False) for tf in out["timeframes"].values()):
        out["ok"] = False
        out["error"] = "No timeframe data available. Ensure market data and smartmoneyconcepts are available."
    return out


def run_infy_route(as_of: str | None = None) -> dict:
    return run_mtf_ict_snapshot("INFY", as_of=as_of, route_name="infy_ict_route")


def run_ict_investigation(ticker: str, as_of: str | None = None) -> dict:
    normalized = (ticker or "").upper().strip()
    return run_infy_route(as_of=as_of) if normalized == "INFY" else run_mtf_ict_snapshot(normalized, as_of=as_of, route_name="ict_investigation_route")
