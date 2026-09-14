#!/usr/bin/env python3
"""China futures contract size + broker margin snapshot (2026-09-09).

notional = close * MULTIPLIER
margin ≈ notional * BROKER_MARGIN (Eastmoney futures, includes FC add-on)
Exchange-only rates from 9qihuo are lower; see EXCHANGE_MARGIN.

Fees from shouxufei (exchange standard): data/reference/shouxufei_fees.json
  https://github.com/qhcg66/shouxufei
"""

from __future__ import annotations

import json
from pathlib import Path

# Price × this = 1-lot notional (CNY). JD quoted 元/500kg with 5t/lot → 10.
MULTIPLIER: dict[str, float] = {
    "AU": 1000, "AG": 15, "CU": 5, "AL": 5, "ZN": 5, "PB": 5, "NI": 1, "SN": 1,
    "RB": 10, "HC": 10, "WR": 10, "SS": 5, "BU": 10, "FU": 10, "RU": 10, "SP": 10,
    "AO": 20, "BR": 5, "AD": 10, "OP": 40,
    "SC": 1000, "LU": 10, "NR": 10, "BC": 5, "EC": 50,
    "A": 10, "B": 10, "M": 10, "Y": 10, "P": 10, "C": 10, "CS": 10,
    "L": 5, "V": 5, "PP": 5, "J": 100, "JM": 60, "I": 100, "JD": 10, "LH": 16,
    "EG": 10, "EB": 5, "PG": 20, "LG": 90, "BZ": 30, "RR": 10, "FB": 10, "BB": 500,
    "CF": 5, "SR": 10, "TA": 5, "MA": 10, "OI": 10, "RM": 10, "FG": 20, "SA": 20,
    "UR": 20, "SF": 5, "SM": 5, "AP": 10, "CJ": 5, "CY": 5, "PK": 5, "PF": 5,
    "PX": 5, "SH": 30, "PL": 20, "PR": 15, "RS": 10, "WH": 20, "RI": 20, "JR": 20,
    "PM": 50, "ZC": 100, "SI": 5, "PS": 3, "LC": 1, "PT": 1000, "PD": 1000,
    "IF": 300, "IH": 300, "IC": 200, "IM": 200,
    "T": 10000, "TF": 10000, "TS": 20000, "TL": 10000,
}

# Eastmoney 期货公司参考保证金率, 2026-09-09 15:58. Broker > exchange.
BROKER_MARGIN: dict[str, float] = {
    "RB": 0.12, "HC": 0.12, "SP": 0.13, "AO": 0.16, "SS": 0.15, "FU": 0.30,
    "PB": 0.15, "BU": 0.25, "BR": 0.18, "OP": 0.12, "ZN": 0.15, "AL": 0.17,
    "NI": 0.18, "WR": 0.80, "RU": 0.16, "AD": 0.15, "AG": 0.32, "SN": 0.20,
    "CU": 0.17, "AU": 0.22,
    "C": 0.11, "CS": 0.11, "RR": 0.10, "V": 0.16, "M": 0.13, "B": 0.11, "A": 0.11,
    "JD": 0.16, "L": 0.16, "PP": 0.16, "LG": 0.10, "EB": 0.20, "FB": 0.80,
    "EG": 0.20, "Y": 0.14, "I": 0.18, "P": 0.14, "JM": 0.25, "LH": 0.15,
    "PG": 0.25, "BZ": 0.18, "J": 0.25, "BB": 0.80,
    "RM": 0.15, "FG": 0.22, "SA": 0.20, "SF": 0.15, "SM": 0.16, "PK": 0.12,
    "TA": 0.17, "SR": 0.10, "UR": 0.18, "PF": 0.16, "CJ": 0.17, "PX": 0.16,
    "MA": 0.24, "SH": 0.15, "CF": 0.12, "AP": 0.15, "CY": 0.10, "OI": 0.13,
    "PR": 0.16, "PL": 0.16, "RI": 0.80, "RS": 0.80, "JR": 0.80, "WH": 0.80,
    "ZC": 0.80, "PM": 0.80,
    "SI": 0.19, "PS": 0.23, "LC": 0.25, "PD": 0.25, "PT": 0.25,
    "LU": 0.30, "NR": 0.16, "EC": 0.35, "BC": 0.17, "SC": 0.30,
    "TS": 0.01, "TF": 0.02, "T": 0.03, "TL": 0.045,
    "IH": 0.15, "IF": 0.15, "IM": 0.15, "IC": 0.15,
}

# 9qihuo exchange-standard snapshot 2026-09-09 (lower bound).
EXCHANGE_MARGIN: dict[str, float] = {
    "AU": 0.16, "AG": 0.22, "CU": 0.11, "AL": 0.11, "ZN": 0.11, "PB": 0.11,
    "NI": 0.12, "SN": 0.14, "RB": 0.07, "HC": 0.07, "SS": 0.10, "RU": 0.09,
    "BU": 0.12, "FU": 0.16, "BR": 0.12, "AO": 0.11, "AD": 0.10, "SP": 0.07,
    "JD": 0.07, "LH": 0.08, "I": 0.11, "J": 0.12, "JM": 0.12, "M": 0.07,
    "Y": 0.07, "P": 0.08, "C": 0.07, "CF": 0.07, "TA": 0.07, "MA": 0.11,
    "IC": 0.12, "IF": 0.12, "IH": 0.12, "IM": 0.12, "SC": 0.16, "LU": 0.16,
    "LC": 0.15, "PS": 0.13, "SI": 0.10, "T": 0.02, "TF": 0.012, "TL": 0.035, "TS": 0.005,
}


def lot_value(symbol: str, price: float | None) -> float | None:
    if price is None or not (price == price) or price <= 0:
        return None
    m = MULTIPLIER.get(str(symbol).upper())
    if not m:
        return None
    return float(price) * m


def lot_margin(symbol: str, price: float | None, broker: bool = True) -> tuple[float | None, float | None]:
    """Return (margin_cny, rate)."""
    val = lot_value(symbol, price)
    rate = (BROKER_MARGIN if broker else EXCHANGE_MARGIN).get(str(symbol).upper())
    if val is None or rate is None:
        return None, rate
    return val * rate, rate


_REF = Path(__file__).resolve().parents[2] / "data/reference/shouxufei_fees.json"
FEE: dict[str, dict] = json.loads(_REF.read_text()) if _REF.is_file() else {}


def lot_fee(symbol: str, price: float | None, kind: str, lots: int = 1) -> float:
    """Exchange-standard fee (CNY). kind: open | close_prev | close_today."""
    if lots <= 0 or price is None or not (price == price) or price <= 0:
        return 0.0
    sym = str(symbol).upper()
    spec = FEE.get(sym)
    if not spec:
        return 0.0
    key = {"open": "open", "close_prev": "close_prev", "close_today": "close_today"}.get(kind)
    if not key:
        return 0.0
    rate = float(spec.get(key) or 0)
    if spec.get("fee_type") == "fixed":
        return rate * lots
    mult = MULTIPLIER.get(sym) or spec.get("mult")
    if not mult:
        return 0.0
    return float(price) * float(mult) * rate / 10_000.0 * lots


def exchange_margin(symbol: str, price: float | None) -> tuple[float | None, float | None]:
    """Shouxufei exchange-standard margin (CNY, rate)."""
    spec = FEE.get(str(symbol).upper())
    val = lot_value(symbol, price)
    rate = spec.get("margin_ex") if spec else EXCHANGE_MARGIN.get(str(symbol).upper())
    if val is None or rate is None:
        return None, rate
    return val * float(rate), float(rate)
