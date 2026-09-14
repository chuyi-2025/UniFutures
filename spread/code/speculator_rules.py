"""China futures speculator (natural person) holding eligibility by exchange."""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

# Exchange map for spread universe
EXCHANGE: dict[str, str] = {
    "RB": "SHFE", "HC": "SHFE", "WR": "SHFE", "SS": "SHFE", "BU": "SHFE",
    "FU": "SHFE", "RU": "SHFE", "SP": "SHFE", "AO": "SHFE", "BR": "SHFE",
    "AU": "SHFE", "AG": "SHFE", "CU": "SHFE", "AL": "SHFE", "ZN": "SHFE",
    "NI": "SHFE", "SN": "SHFE", "PB": "SHFE",
    "M": "DCE", "Y": "DCE", "P": "DCE", "C": "DCE", "CS": "DCE",
    "A": "DCE", "B": "DCE", "I": "DCE", "J": "DCE", "JM": "DCE",
    "L": "DCE", "V": "DCE", "PP": "DCE", "EG": "DCE", "EB": "DCE",
    "PG": "DCE", "LH": "DCE", "JD": "DCE",
    "CF": "CZCE", "SR": "CZCE", "TA": "CZCE", "MA": "CZCE", "OI": "CZCE",
    "RM": "CZCE", "FG": "CZCE", "SA": "CZCE", "AP": "CZCE", "CJ": "CZCE",
    "PK": "CZCE", "PF": "CZCE", "UR": "CZCE", "SF": "CZCE", "SM": "CZCE",
    "SC": "INE", "LU": "INE", "NR": "INE", "BC": "INE", "EC": "INE",
}

# SHFE/INE: flat by close of T-N before last trading day (broker conservative T-5)
SHFE_FLAT_BEFORE_LTD = 5
INE_SC_FLAT_BEFORE_LTD = 8  # crude oil special case


def contract_ord(code: str) -> int | None:
    m = re.match(r"^[A-Z]+(\d{4})$", str(code).upper())
    if not m:
        return None
    yy = 2000 + int(m.group(1)[:2])
    mm = int(m.group(1)[2:])
    if mm < 1 or mm > 12:
        return None
    return yy * 12 + mm


def contract_ym(code: str) -> tuple[int, int] | None:
    m = re.match(r"^[A-Z]+(\d{4})$", str(code).upper())
    if not m:
        return None
    yy = 2000 + int(m.group(1)[:2])
    mm = int(m.group(1)[2:])
    if mm < 1 or mm > 12:
        return None
    return yy, mm


def last_trading_date(code: str) -> pd.Timestamp | None:
    """SHFE-style: 15th of delivery month, roll forward to weekday."""
    ym = contract_ym(code)
    if ym is None:
        return None
    y, m = ym
    d = pd.Timestamp(y, m, 15)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def _trading_days_before(all_days: np.ndarray, target: pd.Timestamp, n: int) -> pd.Timestamp | None:
    """Nth trading day before target (exclusive of target day if in calendar)."""
    idx = np.searchsorted(all_days, np.datetime64(target), side="left")
    pos = idx - n
    if pos < 0:
        return None
    return pd.Timestamp(all_days[pos])


def speculator_can_hold(
    date: pd.Timestamp,
    code: str,
    symbol: str,
    trading_days: np.ndarray,
) -> bool:
    """
    Natural-person holding eligibility on `date` for `code`.

    DCE/CZCE/GFEX: cannot enter delivery month (trade month < delivery month).
    SHFE/INE: may enter delivery month but must flat by T-N before last trading day.
    """
    deliv = contract_ord(code)
    if deliv is None:
        return False
    trade_ord = int(date.year) * 12 + int(date.month)
    exch = EXCHANGE.get(symbol.upper(), "DCE")  # default conservative (no delivery month)

    if exch in ("DCE", "CZCE", "GFEX"):
        return trade_ord < deliv

    if exch in ("SHFE", "INE"):
        ltd = last_trading_date(code)
        if ltd is None:
            return trade_ord < deliv
        n = INE_SC_FLAT_BEFORE_LTD if (exch == "INE" and symbol.upper() == "SC") else SHFE_FLAT_BEFORE_LTD
        deadline = _trading_days_before(trading_days, ltd, n)
        if deadline is None:
            return trade_ord < deliv
        return pd.Timestamp(date).normalize() <= deadline.normalize()

    return trade_ord < deliv


def pick_spec_near_far(
    day: pd.DataFrame,
    symbol: str,
    trading_days: np.ndarray,
) -> tuple[str | None, float | None, str | None, float | None]:
    """
    Speculator near/far: eligible contracts sorted by expiry.
    Near prefers month+3 main among eligible; far = next eligible month.
    """
    from linear_ridge_walkforward import pick_main_fast  # lazy import

    rows = []
    for _, r in day.iterrows():
        code = str(r["code"])
        if not speculator_can_hold(r["date"], code, symbol, trading_days):
            continue
        rows.append(r)
    if len(rows) < 2:
        return None, None, None, None
    elig = pd.DataFrame(rows).sort_values("_ord")
    sub = elig[["date", "code", "close", "_ord"]].copy()
    main = pick_main_fast(sub.rename(columns={"_ord": "_ord"}))
    if main.empty:
        near_row = elig.iloc[0]
    else:
        near_code = str(main.iloc[0]["code"])
        near_row = elig[elig["code"] == near_code].iloc[0] if (elig["code"] == near_code).any() else elig.iloc[0]
    far_cands = elig[elig["_ord"] > near_row["_ord"]]
    if far_cands.empty:
        return None, None, None, None
    far_row = far_cands.iloc[0]
    return (
        str(near_row["code"]), float(near_row["close"]),
        str(far_row["code"]), float(far_row["close"]),
    )
