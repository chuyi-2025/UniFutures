#!/usr/bin/env python3
"""Large account-aware macro strategy search (wall ≥ --minutes, default 65).

Constraints aligned with final_scheme spirit:
  capital=3e6, broker margin, shouxufei fees, integer lots, max names/day,
  optional per-leg margin cap.
"""

from __future__ import annotations

import argparse
import gc
import json
import signal
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

KEEP_TOP = 12_000
_STOP = False


def _request_stop(signum, frame):
    global _STOP
    _STOP = True
    print(f"\n[signal {signum}] soft-stop requested — will finalize soon", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/home/workspace/lab/UniFutures/code/experiment")))

from common import FACTOR_DIR, RESULT_DIR, ensure_dirs  # noqa: E402
from futures_lot_specs import lot_fee, lot_margin, lot_value  # noqa: E402
from linear_ridge_walkforward import TREE, pick_main_fast, _read_contract_csv  # noqa: E402

OUT = RESULT_DIR / "search_account"
CAPITAL = 3_000_000.0
START = pd.Timestamp("2018-01-01")

UNIVERSE = (
    "AU", "AG", "SC", "CU", "AL", "ZN", "NI", "PB", "SN",
    "FU", "LU", "NR", "BU", "RB", "HC", "I", "J", "JM",
    "TA", "MA", "PP", "L", "V", "EG", "EB", "PG",
    "M", "Y", "P", "OI", "RM", "CF", "SR",
)

FACTORS = [
    "dxy_z60", "dxy_chg5", "dxy_chg20", "usd_broad_z60",
    "hike_proxy_z60", "hike_proxy_chg", "us_2y_z60", "us_2y_chg5",
    "us_10y_z60", "us_curve_10y2y_z60", "curve_10_2_z60",
    "vix_z60", "vix_chg", "unemp_chg", "us_unemp_z60", "cpi_yoy_z60",
    "usd_hawkish", "risk_off", "labor_soft",
]

UNIVERSE_SETS = {
    "precious_energy": ("AU", "AG", "SC", "FU", "LU"),
    "metals": ("AU", "AG", "CU", "AL", "ZN", "NI", "PB", "SN"),
    "energy": ("SC", "FU", "LU", "BU", "PG", "TA", "MA"),
    "industrial": ("CU", "AL", "ZN", "RB", "HC", "I", "J", "JM", "NI"),
    "broad_macro": ("AU", "AG", "SC", "CU", "AL", "ZN", "NI", "FU", "RB", "I", "TA", "MA", "M", "Y"),
    "au_only": ("AU",),
    "sc_only": ("SC",),
    "au_ag_sc": ("AU", "AG", "SC"),
    "cu_al_zn": ("CU", "AL", "ZN"),
    "black": ("RB", "HC", "I", "J", "JM"),
}


def spearman_ic(x, y, min_n=30) -> float:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    m = a.notna() & b.notna()
    if int(m.sum()) < min_n:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = a[m].corr(b[m], method="spearman")
    return float(v) if v == v else float("nan")


def sharpe(r: np.ndarray) -> float | None:
    r = np.asarray(r, dtype=float)
    r = np.nan_to_num(r, nan=0.0)
    if len(r) < 40 or float(np.std(r)) == 0:
        return None
    return float(np.mean(r) / np.std(r) * np.sqrt(252))


def max_dd_from_ret(r: np.ndarray) -> float | None:
    r = np.nan_to_num(np.asarray(r, dtype=float), nan=0.0)
    if len(r) < 5:
        return None
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    return float(np.min(nav / peak - 1.0))


def load_symbol_px(symbol: str) -> pd.DataFrame:
    sym_dir = TREE / symbol
    if not sym_dir.is_dir():
        return pd.DataFrame()
    parts = []
    for csv in sorted(sym_dir.glob("*.csv")):
        try:
            raw = _read_contract_csv(csv)
        except Exception:
            continue
        parts.append(raw[["date", "code", "close"]])
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True).sort_values(["code", "date"])
    nxt = df.groupby("code")["close"].shift(-1)
    same = df.groupby("code")["date"].shift(-1)
    gap = (same - df["date"]).dt.days
    df["fwd_ret"] = np.log(nxt / df["close"])
    df.loc[gap.isna() | (gap > 10), "fwd_ret"] = np.nan
    df["symbol"] = symbol
    main = pick_main_fast(df)
    out = main[["date", "symbol", "code", "close", "fwd_ret"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date")


def build_expanded_panel(symbols: tuple[str, ...], lag: int = 1) -> pd.DataFrame:
    fac = pd.read_parquet(FACTOR_DIR / "macro_factors_daily.parquet")
    fac["date"] = pd.to_datetime(fac["date"]) + pd.Timedelta(days=lag)
    fac = fac.sort_values("date")
    frames = []
    for sym in symbols:
        print(f"[px] {sym}", flush=True)
        px = load_symbol_px(sym)
        if px.empty:
            continue
        m = pd.merge_asof(px.sort_values("date"), fac, on="date", direction="backward")
        mars, fee_o, fee_c, notionals = [], [], [], []
        for pxv in m["close"].to_numpy():
            ok = np.isfinite(pxv)
            mar, _ = lot_margin(sym, float(pxv) if ok else None, broker=True)
            mars.append(mar if mar and np.isfinite(mar) else np.nan)
            val = lot_value(sym, float(pxv) if ok else None)
            notionals.append(val if val and np.isfinite(val) else np.nan)
            try:
                fee_o.append(float(lot_fee(sym, float(pxv), "open", 1)) if ok else np.nan)
                fee_c.append(float(lot_fee(sym, float(pxv), "close_prev", 1)) if ok else np.nan)
            except Exception:
                fee_o.append(np.nan)
                fee_c.append(np.nan)
        m["m1"] = mars
        m["notional1"] = notionals
        m["fee1_open"] = fee_o
        m["fee1_close"] = fee_c
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


class BookEngine:
    """Precompute date-aligned tensors for fast account backtests."""

    def __init__(self, panel: pd.DataFrame, factors: list[str]):
        self.symbols = sorted(panel["symbol"].unique())
        self.sym_index = {s: i for i, s in enumerate(self.symbols)}
        self.dates = np.array(sorted(panel["date"].unique()))
        self.date_index = {pd.Timestamp(d): i for i, d in enumerate(self.dates)}
        n_t, n_s = len(self.dates), len(self.symbols)
        self.fwd = np.full((n_t, n_s), np.nan)
        self.m1 = np.full((n_t, n_s), np.nan)
        self.notional = np.full((n_t, n_s), np.nan)
        self.fee_o = np.full((n_t, n_s), np.nan)
        self.fee_c = np.full((n_t, n_s), np.nan)
        self.fac = {f: np.full((n_t, n_s), np.nan) for f in factors if f in panel.columns}
        for sym, g in panel.groupby("symbol"):
            j = self.sym_index[sym]
            for _, r in g.iterrows():
                i = self.date_index.get(pd.Timestamp(r["date"]))
                if i is None:
                    continue
                self.fwd[i, j] = r["fwd_ret"]
                self.m1[i, j] = r["m1"]
                self.notional[i, j] = r["notional1"]
                self.fee_o[i, j] = r["fee1_open"]
                self.fee_c[i, j] = r["fee1_close"]
                for f, arr in self.fac.items():
                    arr[i, j] = r[f]
        self._sig_cache: dict[tuple, np.ndarray] = {}
        print(f"engine T={n_t} S={n_s}", flush=True)

    def _signal(self, fac_name: str, mode: str, thr: float, direction: float, hold: int) -> np.ndarray:
        key = (fac_name, mode, float(thr), float(direction), int(hold))
        cached = self._sig_cache.get(key)
        if cached is not None:
            return cached
        x = self.fac[fac_name]  # T x S
        if mode == "cont":
            raw = np.clip(np.nan_to_num(x, nan=0.0), -2, 2) / 2.0 * direction
        elif mode == "sign":
            xx = np.nan_to_num(x, nan=0.0)
            raw = np.where(np.abs(xx) >= thr, np.sign(xx) * direction, 0.0)
        elif mode == "long_only":
            xx = np.nan_to_num(x, nan=0.0)
            raw = np.where(xx * direction >= thr, direction, 0.0)
        else:
            raise ValueError(mode)
        if hold <= 1:
            out = raw
        else:
            out = np.zeros_like(raw)
            for j in range(raw.shape[1]):
                left = 0
                last = 0.0
                col = raw[:, j]
                for i, v in enumerate(col):
                    if v != 0:
                        left = hold
                        last = float(np.sign(v))
                    if left > 0:
                        out[i, j] = last
                        left -= 1
        if len(self._sig_cache) > 400:
            self._sig_cache.clear()
        self._sig_cache[key] = out
        return out

    def run(
        self,
        *,
        factor: str,
        mode: str,
        thr: float,
        direction: float,
        hold: int,
        sym_list: tuple[str, ...],
        max_names: int,
        max_lots: int,
        margin_cap: float | None,
        util: float,
        capital: float = CAPITAL,
    ) -> dict | None:
        if factor not in self.fac:
            return None
        js = [self.sym_index[s] for s in sym_list if s in self.sym_index]
        if not js:
            return None
        sig = self._signal(factor, mode, thr, direction, hold)[:, js]
        fwd = self.fwd[:, js]
        m1 = self.m1[:, js]
        notion = self.notional[:, js]
        fee_o = self.fee_o[:, js]
        fee_c = self.fee_c[:, js]
        n_t, n_s = sig.shape
        prev = np.zeros(n_s, dtype=int)
        pnl = np.zeros(n_t)
        ret = np.zeros(n_t)
        margin = np.zeros(n_t)
        n_names = np.zeros(n_t)
        fees = np.zeros(n_t)

        for t in range(n_t):
            # candidates
            scores = []
            for j in range(n_s):
                s = sig[t, j]
                if abs(s) < 1e-12:
                    continue
                mj = m1[t, j]
                if not np.isfinite(mj) or mj <= 0:
                    continue
                if margin_cap is not None and mj > margin_cap:
                    continue
                scores.append((abs(s), s, j, mj))
            scores.sort(reverse=True)
            scores = scores[:max_names]
            target = np.zeros(n_s, dtype=int)
            budget = util * capital
            used = 0.0
            if scores:
                share = budget / len(scores)
                for _, s, j, mj in scores:
                    lots = int(min(max_lots, max(0, np.floor(share / mj))))
                    if lots <= 0 and mj <= budget - used:
                        lots = 1
                    while lots > 0 and used + lots * mj > budget:
                        lots -= 1
                    lots = min(lots, max_lots)
                    if lots > 0:
                        target[j] = int(np.sign(s) * lots)
                        used += lots * mj

            # fees
            fee = 0.0
            for j in range(n_s):
                old, new = int(prev[j]), int(target[j])
                fo = fee_o[t, j] if np.isfinite(fee_o[t, j]) else 0.0
                fc = fee_c[t, j] if np.isfinite(fee_c[t, j]) else fo
                if old != 0 and (new == 0 or np.sign(new) != np.sign(old)):
                    fee += fc * abs(old)
                    old = 0
                if new != 0 and old == 0:
                    fee += fo * abs(new)
                elif new != 0 and old != 0 and np.sign(new) == np.sign(old):
                    d = abs(new) - abs(old)
                    if d > 0:
                        fee += fo * d
                    elif d < 0:
                        fee += fc * (-d)

            gross = 0.0
            mar = 0.0
            nn = 0
            for j in range(n_s):
                lots = int(target[j])
                if lots == 0:
                    continue
                nn += 1
                fr = fwd[t, j] if np.isfinite(fwd[t, j]) else 0.0
                val = notion[t, j]
                if not np.isfinite(val):
                    val = (m1[t, j] / 0.15) if np.isfinite(m1[t, j]) else 0.0
                gross += lots * fr * float(val)
                mar += abs(lots) * (float(m1[t, j]) if np.isfinite(m1[t, j]) else 0.0)

            pnl[t] = gross - fee
            ret[t] = pnl[t] / capital
            margin[t] = mar
            n_names[t] = nn
            fees[t] = fee
            prev[:] = target

        sh = sharpe(ret)
        if sh is None:
            return None
        on = margin > 0
        mid = n_t // 2
        s1, s2 = sharpe(ret[: mid + 1]), sharpe(ret[mid + 1 :])
        out = {
            "sharpe": round(sh, 3),
            "ret": round(float(np.nansum(ret)), 4),
            "max_dd": None if max_dd_from_ret(ret) is None else round(max_dd_from_ret(ret), 4),
            "worst_day_pnl": round(float(np.nanmin(pnl)), 1),
            "avg_margin_on": round(float(np.nanmean(margin[on])), 1) if on.any() else 0.0,
            "fee_sum": round(float(np.nansum(fees)), 1),
            "active_days": int(on.sum()),
            "avg_names": round(float(np.nanmean(n_names[on])), 2) if on.any() else 0.0,
            "sh_first": None if s1 is None else round(s1, 3),
            "sh_second": None if s2 is None else round(s2, 3),
            "oos_both_pos": bool(s1 is not None and s2 is not None and s1 > 0 and s2 > 0),
        }
        return out

    def run_full(self, **kwargs) -> dict | None:
        """Like run(), but also returns daily arrays for export."""
        bt = self.run(**kwargs)
        if bt is None:
            return None
        # recompute once with arrays
        # fall through by calling internal — simplest: duplicate last path
        factor = kwargs["factor"]
        mode = kwargs["mode"]
        thr = kwargs["thr"]
        direction = kwargs["direction"]
        hold = kwargs["hold"]
        sym_list = kwargs["sym_list"]
        max_names = kwargs["max_names"]
        max_lots = kwargs["max_lots"]
        margin_cap = kwargs["margin_cap"]
        util = kwargs["util"]
        capital = kwargs.get("capital", CAPITAL)
        js = [self.sym_index[s] for s in sym_list if s in self.sym_index]
        sig = self._signal(factor, mode, thr, direction, hold)[:, js]
        fwd = self.fwd[:, js]
        m1 = self.m1[:, js]
        notion = self.notional[:, js]
        fee_o = self.fee_o[:, js]
        fee_c = self.fee_c[:, js]
        n_t, n_s = sig.shape
        prev = np.zeros(n_s, dtype=int)
        pnl = np.zeros(n_t)
        ret = np.zeros(n_t)
        margin = np.zeros(n_t)
        for t in range(n_t):
            scores = []
            for j in range(n_s):
                s = sig[t, j]
                if abs(s) < 1e-12:
                    continue
                mj = m1[t, j]
                if not np.isfinite(mj) or mj <= 0:
                    continue
                if margin_cap is not None and mj > margin_cap:
                    continue
                scores.append((abs(s), s, j, mj))
            scores.sort(reverse=True)
            scores = scores[:max_names]
            target = np.zeros(n_s, dtype=int)
            budget = util * capital
            used = 0.0
            if scores:
                share = budget / len(scores)
                for _, s, j, mj in scores:
                    lots = int(min(max_lots, max(0, np.floor(share / mj))))
                    if lots <= 0 and mj <= budget - used:
                        lots = 1
                    while lots > 0 and used + lots * mj > budget:
                        lots -= 1
                    lots = min(lots, max_lots)
                    if lots > 0:
                        target[j] = int(np.sign(s) * lots)
                        used += lots * mj
            fee = 0.0
            for j in range(n_s):
                old, new = int(prev[j]), int(target[j])
                fo = fee_o[t, j] if np.isfinite(fee_o[t, j]) else 0.0
                fc = fee_c[t, j] if np.isfinite(fee_c[t, j]) else fo
                if old != 0 and (new == 0 or np.sign(new) != np.sign(old)):
                    fee += fc * abs(old)
                    old = 0
                if new != 0 and old == 0:
                    fee += fo * abs(new)
                elif new != 0 and old != 0 and np.sign(new) == np.sign(old):
                    d = abs(new) - abs(old)
                    if d > 0:
                        fee += fo * d
                    elif d < 0:
                        fee += fc * (-d)
            gross = 0.0
            mar = 0.0
            for j in range(n_s):
                lots = int(target[j])
                if lots == 0:
                    continue
                fr = fwd[t, j] if np.isfinite(fwd[t, j]) else 0.0
                val = notion[t, j]
                if not np.isfinite(val):
                    val = (m1[t, j] / 0.15) if np.isfinite(m1[t, j]) else 0.0
                gross += lots * fr * float(val)
                mar += abs(lots) * (float(m1[t, j]) if np.isfinite(m1[t, j]) else 0.0)
            pnl[t] = gross - fee
            ret[t] = pnl[t] / capital
            margin[t] = mar
            prev[:] = target
        bt["pnl"] = pnl
        bt["ret_arr"] = ret
        bt["margin_arr"] = margin
        bt["dates"] = self.dates
        return bt


def _trim_results(results: list[dict]) -> list[dict]:
    """Return a NEW list of top schemes (never alias the input list)."""
    if not results:
        return []
    ranked = sorted(
        results,
        key=lambda r: (r.get("quality") or -9e9, r.get("sharpe") or -9e9),
        reverse=True,
    )
    return ranked[:KEEP_TOP]


def main() -> None:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=65.0)
    ap.add_argument("--rebuild-panel", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Load checkpoint_schemes.csv and continue")
    ap.add_argument("--skip-phase-a", action="store_true", help="Skip systematic phase A")
    args = ap.parse_args()
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    t_end = t0 + args.minutes * 60.0
    print(f"search start minutes={args.minutes} until={time.strftime('%H:%M:%S', time.localtime(t_end))} resume={args.resume}", flush=True)

    panel_path = OUT / "expanded_aligned_panel.parquet"
    if args.rebuild_panel or not panel_path.exists():
        syms = tuple(s for s in UNIVERSE if (TREE / s).is_dir())
        panel = build_expanded_panel(syms)
        panel = panel[panel["date"] >= START].copy()
        panel.to_parquet(panel_path, index=False)
    else:
        panel = pd.read_parquet(panel_path)
        panel["date"] = pd.to_datetime(panel["date"])
        panel = panel[panel["date"] >= START].copy()

    avail_fac = [f for f in FACTORS if f in panel.columns]
    print(f"panel rows={len(panel)} building engine...", flush=True)
    engine = BookEngine(panel, avail_fac)

    # IC board (reuse if present unless rebuilding)
    ic_path = OUT / "ic_ir_expanded.csv"
    if ic_path.exists() and not args.rebuild_panel:
        ic_df = pd.read_csv(ic_path)
        print(f"loaded IC board rows={len(ic_df)}", flush=True)
    else:
        ic_rows = []
        for sym in engine.symbols:
            g = panel[panel["symbol"] == sym]
            for fac in avail_fac:
                for direction in (1.0, -1.0):
                    ic = spearman_ic(g[fac] * direction, g["fwd_ret"], min_n=40)
                    months = []
                    sub = pd.DataFrame({"f": g[fac] * direction, "r": g["fwd_ret"], "d": g["date"]}).dropna()
                    for _, gg in sub.groupby(pd.to_datetime(sub["d"]).dt.to_period("M")):
                        if len(gg) < 8:
                            continue
                        months.append(spearman_ic(gg["f"], gg["r"], min_n=8))
                    months = [m for m in months if m == m]
                    ir = float(np.mean(months) / np.std(months)) if len(months) >= 8 and np.std(months) > 1e-12 else None
                    ic_rows.append({"symbol": sym, "factor": fac, "direction": direction,
                                    "ic": None if ic != ic else round(ic, 4),
                                    "ir": None if ir is None else round(ir, 3)})
        ic_df = pd.DataFrame(ic_rows)
        ic_df.to_csv(ic_path, index=False)

    modes = ["cont", "sign", "long_only"]
    thrs = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    holds = [1, 2, 3, 5, 8, 10, 15]
    max_names_g = [1, 2, 3, 4, 6]
    max_lots_g = [1, 2, 3]
    margin_caps = [50_000.0, 80_000.0, 120_000.0, 200_000.0, None]
    utils = [0.08, 0.12, 0.18, 0.25, 0.35, 0.45]

    results: list[dict] = []
    n_try = 0
    n_ok = 0
    if args.resume and (OUT / "checkpoint_schemes.csv").exists():
        prev = pd.read_csv(OUT / "checkpoint_schemes.csv")
        results = prev.replace({np.nan: None}).to_dict(orient="records")
        for r in results:
            if r.get("margin_cap") is not None:
                try:
                    r["margin_cap"] = float(r["margin_cap"])
                except Exception:
                    r["margin_cap"] = None
        n_ok = len(results)
        print(f"resumed {n_ok} schemes from checkpoint, best_sh={max((r.get('sharpe') or -9) for r in results)}", flush=True)

    rng = np.random.default_rng(20260913 + int(n_ok))

    def consider(cfg, bt):
        nonlocal n_ok
        if bt is None:
            return
        ic_sub = ic_df[(ic_df.factor == cfg["factor"]) & (ic_df.direction == cfg["direction"]) & (ic_df.symbol.isin(cfg["symbols"]))]
        ic_m = float(ic_sub["ic"].mean()) if len(ic_sub) and ic_sub["ic"].notna().any() else None
        ir_m = float(ic_sub["ir"].mean()) if len(ic_sub) and ic_sub["ir"].notna().any() else None
        row = {
            "factor": cfg["factor"], "mode": cfg["mode"], "thr": cfg["thr"], "direction": cfg["direction"],
            "hold": cfg["hold"], "universe": cfg.get("universe", ""), "symbols": ",".join(cfg["symbols"]),
            "n_sym": len(cfg["symbols"]), "max_names": cfg["max_names"], "max_lots": cfg["max_lots"],
            "margin_cap": cfg["margin_cap"], "util": cfg["util"],
            "sharpe": bt["sharpe"], "ret": bt["ret"], "max_dd": bt["max_dd"],
            "worst_day_pnl": bt["worst_day_pnl"], "avg_margin_on": bt["avg_margin_on"],
            "fee_sum": bt["fee_sum"], "active_days": bt["active_days"], "avg_names": bt["avg_names"],
            "sh_first": bt["sh_first"], "sh_second": bt["sh_second"], "oos_both_pos": bt["oos_both_pos"],
            "ic_mean": None if ic_m is None or ic_m != ic_m else round(ic_m, 4),
            "ir_mean": None if ir_m is None or ir_m != ir_m else round(ir_m, 3),
        }
        q = (row["sharpe"] or 0) + (0.4 if row["oos_both_pos"] else 0)
        q += 0.35 * (row["ir_mean"] or 0) + 1.2 * abs(row["ic_mean"] or 0)
        if row["max_dd"] is not None:
            q += 0.5 * max(-0.5, row["max_dd"])
        if row["worst_day_pnl"] is not None and row["worst_day_pnl"] < -80_000:
            q -= 0.25
        row["quality"] = round(q, 4)
        results.append(row)
        n_ok += 1
        if n_ok % 1500 == 0:
            results[:] = _trim_results(results)
            try:
                pd.DataFrame(results).to_csv(OUT / "checkpoint_schemes.csv", index=False)
                best_sh = max((r["sharpe"] for r in results), default=None)
                (OUT / "checkpoint_meta.json").write_text(
                    json.dumps({"n_ok": n_ok, "n_kept": len(results), "best_sh": best_sh}, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"  checkpoint fail: {e}", flush=True)
            gc.collect()

    # Phase A: lighter systematic so most wall time goes to random exploration
    top_fac = (
        ic_df.dropna(subset=["ir"]).assign(a=lambda d: d.ir.abs())
        .sort_values("a", ascending=False)["factor"].drop_duplicates().head(6).tolist()
    )
    if not top_fac:
        top_fac = avail_fac[:6]
    if args.skip_phase_a or args.resume:
        print(f"skip phase A (skip={args.skip_phase_a} resume={args.resume}) top_fac={top_fac}", flush=True)
    else:
        print(f"phase A top_fac={top_fac}", flush=True)
        phase_a_deadline = min(t_end, time.time() + min(18 * 60.0, args.minutes * 60.0 * 0.30))
        stop_a = False
        for fac in top_fac:
            if stop_a:
                break
            for mode in modes:
                if stop_a:
                    break
                for direction in (1.0, -1.0):
                    if stop_a:
                        break
                    for uni_name, syms in list(UNIVERSE_SETS.items())[:8]:
                        if stop_a:
                            break
                        for thr in ([0.0] if mode == "cont" else [0.5, 1.0, 1.5]):
                            if stop_a:
                                break
                            for hold in ([1] if mode == "cont" else [1, 5, 10]):
                                if stop_a:
                                    break
                                for max_names in [2, 4]:
                                    if stop_a:
                                        break
                                    for max_lots in [1, 2]:
                                        if stop_a:
                                            break
                                        for mcap in [50_000.0, 120_000.0, None]:
                                            if stop_a:
                                                break
                                            for util in [0.12, 0.25, 0.35]:
                                                if _STOP or time.time() >= phase_a_deadline or time.time() >= t_end:
                                                    stop_a = True
                                                    break
                                                n_try += 1
                                                try:
                                                    bt = engine.run(
                                                        factor=fac, mode=mode, thr=thr, direction=direction, hold=hold,
                                                        sym_list=syms, max_names=max_names, max_lots=max_lots,
                                                        margin_cap=mcap, util=util,
                                                    )
                                                except Exception as e:
                                                    if "allocate" in str(e).lower() or "Memory" in type(e).__name__:
                                                        gc.collect()
                                                        results[:] = _trim_results(results)
                                                        print("  A MemoryError — trim+continue", flush=True)
                                                        bt = None
                                                    else:
                                                        bt = None
                                                consider({
                                                    "factor": fac, "mode": mode, "thr": thr, "direction": direction,
                                                    "hold": hold, "symbols": syms, "universe": uni_name,
                                                    "max_names": max_names, "max_lots": max_lots,
                                                    "margin_cap": mcap, "util": util,
                                                }, bt)
                                                if n_try % 500 == 0:
                                                    best = max((r["sharpe"] for r in results), default=None)
                                                    print(f"  A try={n_try} ok={n_ok} best_sh={best} left={t_end-time.time():.0f}s", flush=True)
                                                    (OUT / "heartbeat.json").write_text(
                                                        json.dumps({"phase": "A", "n_try": n_try, "n_ok": n_ok, "best_sh": best, "t": time.time()}),
                                                        encoding="utf-8",
                                                    )

    print("phase B random until budget...", flush=True)
    uni_items = list(UNIVERSE_SETS.items())
    mem_fails = 0
    while (not _STOP) and time.time() < t_end:
        n_try += 1
        fac = str(rng.choice(avail_fac))
        mode = str(rng.choice(modes))
        direction = float(rng.choice([1.0, -1.0]))
        thr = 0.0 if mode == "cont" else float(rng.choice(thrs))
        hold = 1 if mode == "cont" else int(rng.choice(holds))
        if rng.random() < 0.7:
            uni_name, syms = uni_items[int(rng.integers(0, len(uni_items)))]
        else:
            uni_name = "random"
            k = int(rng.integers(2, 9))
            syms = tuple(rng.choice(np.array(engine.symbols), size=min(k, len(engine.symbols)), replace=False).tolist())
        cfg = {
            "factor": fac, "mode": mode, "thr": thr, "direction": direction, "hold": hold,
            "symbols": syms, "universe": uni_name,
            "max_names": int(rng.choice(max_names_g)),
            "max_lots": int(rng.choice(max_lots_g)),
            "margin_cap": margin_caps[int(rng.integers(0, len(margin_caps)))],
            "util": float(rng.choice(utils)),
        }
        try:
            bt = engine.run(
                factor=cfg["factor"], mode=cfg["mode"], thr=cfg["thr"], direction=cfg["direction"], hold=cfg["hold"],
                sym_list=cfg["symbols"], max_names=cfg["max_names"], max_lots=cfg["max_lots"],
                margin_cap=cfg["margin_cap"], util=cfg["util"],
            )
        except Exception as e:
            if "Memory" in type(e).__name__ or "allocate" in str(e).lower():
                mem_fails += 1
                trimmed = _trim_results(results)
                results.clear()
                results.extend(trimmed)
                gc.collect()
                if mem_fails >= 3:
                    print(f"  too many MemoryErrors ({mem_fails}), early finalize", flush=True)
                    break
                continue
            bt = None
        consider(cfg, bt)
        if n_try % 800 == 0:
            best = max(results, key=lambda r: r["sharpe"]) if results else None
            print(f"  B try={n_try} ok={n_ok} kept={len(results)} best_sh={None if not best else best['sharpe']} left={t_end-time.time():.0f}s", flush=True)
            (OUT / "heartbeat.json").write_text(
                json.dumps({
                    "phase": "B", "n_try": n_try, "n_ok": n_ok, "kept": len(results),
                    "best_sh": None if not best else best["sharpe"], "t": time.time(),
                }),
                encoding="utf-8",
            )

    trimmed = _trim_results(results)
    results.clear()
    results.extend(trimmed)
    res = pd.DataFrame(results)
    if res.empty:
        print("empty", flush=True)
        return
    res = res.sort_values(["quality", "sharpe"], ascending=False)
    # dedupe identical configs
    res = res.drop_duplicates(
        subset=["factor", "mode", "thr", "direction", "hold", "symbols", "max_names", "max_lots", "margin_cap", "util"],
        keep="first",
    )
    res.to_csv(OUT / "all_account_schemes.csv", index=False)
    robust = res[(res.sharpe >= 0.8) & (res.oos_both_pos) & (res.max_dd.fillna(-1) >= -0.20) & (res.ir_mean.fillna(-9) >= 0.08)]
    soft = res[(res.sharpe >= 0.6) & (res.oos_both_pos) & (res.max_dd.fillna(-1) >= -0.25)]
    robust.to_csv(OUT / "robust_account.csv", index=False)
    soft.to_csv(OUT / "soft_account.csv", index=False)
    res.head(50).to_csv(OUT / "top50_account.csv", index=False)

    best = res.iloc[0].to_dict()
    bt = engine.run_full(
        factor=best["factor"], mode=best["mode"], thr=float(best["thr"]), direction=float(best["direction"]),
        hold=int(best["hold"]), sym_list=tuple(best["symbols"].split(",")),
        max_names=int(best["max_names"]), max_lots=int(best["max_lots"]),
        margin_cap=None if pd.isna(best["margin_cap"]) else float(best["margin_cap"]),
        util=float(best["util"]),
    )
    if bt is None:
        print("best export failed", flush=True)
        return
    daily = pd.DataFrame({
        "日期": pd.to_datetime(bt["dates"]).strftime("%Y-%m-%d"),
        "账户日盈亏": bt["pnl"],
        "账户日收益": bt["ret_arr"],
        "合计保证金": bt["margin_arr"],
    })
    daily.to_csv(OUT / "best_account_daily.csv", index=False)
    lin = Path("/home/workspace/lab/UniFutures/data/infer/results/daily/linear")
    lin.mkdir(parents=True, exist_ok=True)
    daily.to_csv(lin / "macro_account_best_daily.csv", index=False)

    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, str):
            return o
        try:
            if pd.isna(o):
                return None
        except Exception:
            pass
        return o

    report = {
        "capital": CAPITAL,
        "minutes": args.minutes,
        "n_try": n_try,
        "n_ok": n_ok,
        "n_unique": int(len(res)),
        "n_robust": int(len(robust)),
        "n_soft": int(len(soft)),
        "best": best,
        "top10": res.head(10).to_dict(orient="records"),
        "constraints": {
            "capital": CAPITAL,
            "broker_margin": True,
            "shouxufei_fees": True,
            "integer_lots": True,
            "max_names_grid": max_names_g,
            "margin_cap_grid": margin_caps,
            "aligned_with": "final_scheme spirit (1手整数/保证金/手续费/品种上限); capital 300万 vs final 100万",
        },
    }
    (OUT / "account_search_report.json").write_text(json.dumps(conv(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print("BEST", {k: best.get(k) for k in ("factor", "mode", "thr", "direction", "hold", "universe", "symbols", "max_names", "max_lots", "margin_cap", "util", "sharpe", "max_dd", "ir_mean", "quality")}, flush=True)
    print(f"done elapsed={time.time()-t0:.0f}s try={n_try} ok={n_ok} unique={len(res)} robust={len(robust)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
