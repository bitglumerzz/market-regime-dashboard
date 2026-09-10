"""Portfolio risk analysis — correlations, regime overlay, stress tests.

Input: a list of positions ({ticker, qty, cost_basis?}). Output:
  - Current market value + P&L (if cost basis provided)
  - Allocation by weight (and by dollar)
  - Pairwise correlation matrix of log returns
  - Per-position current HMM regime
  - Stress-test summary over historical crisis windows

This module is broker-agnostic — it operates on a positions list, so
plugging a real broker (Alpaca, IBKR) just means writing a function that
returns that list.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_loader import load_data
from feature_engineering import LOG_RETURN_COL, compute_features, feature_matrix
from hmm_model import forward_filter, train_best_hmm
from regime_labeler import apply_stability_filter, label_regimes
from stress_tests import CRISIS_WINDOWS, slice_window


@dataclass
class Position:
    """One holding in the portfolio."""
    ticker: str
    qty: float
    cost_basis: float | None = None   # average cost per share (optional)


@dataclass
class PositionAnalysis:
    """All derived stats for one position."""
    ticker: str
    qty: float
    last_price: float
    market_value: float
    cost_basis: float | None
    cost_value: float | None
    pnl_pct: float | None
    regime: str
    regime_confidence: float
    log_returns: pd.Series
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def parse_positions_csv(text: str) -> list[Position]:
    """Parse a CSV of positions. Required cols: ticker, qty. Optional: cost_basis."""
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df.columns or "qty" not in df.columns:
        raise ValueError(
            "CSV must have columns 'ticker' and 'qty' (case-insensitive). "
            "Optional column: 'cost_basis'."
        )
    out: list[Position] = []
    for _, row in df.iterrows():
        cb = row.get("cost_basis")
        try:
            cb_val: float | None = float(cb) if pd.notna(cb) else None
        except (TypeError, ValueError):
            cb_val = None
        out.append(Position(
            ticker=str(row["ticker"]).strip().upper(),
            qty=float(row["qty"]),
            cost_basis=cb_val,
        ))
    return out


def _analyze_one(
    pos: Position, start: str, end: str
) -> PositionAnalysis:
    """Fetch prices, run quick regime detection, build a stats record."""
    try:
        df = load_data(pos.ticker, start, end)
    except Exception as exc:
        return PositionAnalysis(
            ticker=pos.ticker, qty=pos.qty, last_price=0.0, market_value=0.0,
            cost_basis=pos.cost_basis,
            cost_value=(pos.cost_basis * pos.qty) if pos.cost_basis else None,
            pnl_pct=None, regime="—", regime_confidence=0.0,
            log_returns=pd.Series(dtype=float),
            error=f"{type(exc).__name__}: {exc}",
        )

    feats = compute_features(df)
    log_ret = feats[LOG_RETURN_COL].astype(float)
    last_price = float(df["Close"].iloc[-1])
    market_value = last_price * pos.qty
    cost_value = pos.cost_basis * pos.qty if pos.cost_basis else None
    pnl_pct = ((last_price / pos.cost_basis) - 1.0) if pos.cost_basis else None

    # Regime detection — quick: n=3 fixed, in-sample, last value only.
    regime = "—"
    confidence = 0.0
    try:
        if len(feats) >= 60:
            X = feature_matrix(feats)
            Xs = StandardScaler().fit_transform(X)
            model, n = train_best_hmm(Xs, n_range=(3, 3))
            post = forward_filter(model, Xs)
            state_to_label = label_regimes(model, n)
            raw = pd.Series(
                [state_to_label[int(s)] for s in np.argmax(post, axis=1)],
                index=feats.index,
            )
            smooth = apply_stability_filter(raw)
            regime = str(smooth.iloc[-1])
            confidence = float(post[-1].max())
    except Exception:
        regime = "—"
        confidence = 0.0

    return PositionAnalysis(
        ticker=pos.ticker, qty=pos.qty, last_price=last_price,
        market_value=market_value,
        cost_basis=pos.cost_basis, cost_value=cost_value, pnl_pct=pnl_pct,
        regime=regime, regime_confidence=confidence,
        log_returns=log_ret,
    )


def analyze_portfolio(
    positions: Iterable[Position], start: str, end: str,
    progress_cb=None,
) -> list[PositionAnalysis]:
    """Analyze every position; failures are surfaced per-ticker, not fatal."""
    positions = list(positions)
    out: list[PositionAnalysis] = []
    for i, p in enumerate(positions):
        if progress_cb:
            progress_cb(i / max(1, len(positions)), f"Analyzing {p.ticker}…")
        out.append(_analyze_one(p, start, end))
    if progress_cb:
        progress_cb(1.0, "Done.")
    return out


def correlation_matrix(
    analyses: list[PositionAnalysis], min_overlap: int = 30
) -> pd.DataFrame:
    """Pairwise log-return correlation across positions. NaN where insufficient overlap."""
    ok = [a for a in analyses if a.ok and len(a.log_returns.dropna()) >= min_overlap]
    if len(ok) < 2:
        return pd.DataFrame()
    df = pd.concat(
        {a.ticker: a.log_returns for a in ok},
        axis=1, join="outer",
    )
    return df.corr().round(3)


def portfolio_total_value(analyses: list[PositionAnalysis]) -> dict[str, float]:
    """Sum market and cost values across positions."""
    mv = sum(a.market_value for a in analyses if a.ok)
    cv = sum(a.cost_value for a in analyses if a.ok and a.cost_value is not None)
    return {
        "market_value": float(mv),
        "cost_value": float(cv),
        "pnl_abs": float(mv - cv) if cv else 0.0,
        "pnl_pct": float((mv / cv - 1.0) * 100) if cv else 0.0,
    }


def stress_test_portfolio(
    analyses: list[PositionAnalysis]
) -> pd.DataFrame:
    """For each crisis window, compute the portfolio's max drawdown if you
    held current allocation through it.

    Weight by market value (current snapshot). Returns a tidy DataFrame
    with one row per crisis × per ticker, plus an aggregate row.
    """
    ok = [a for a in analyses if a.ok and len(a.log_returns.dropna()) > 0]
    if not ok:
        return pd.DataFrame()

    total_mv = sum(a.market_value for a in ok)
    weights = {a.ticker: (a.market_value / total_mv) for a in ok} if total_mv else {}

    rows: list[dict] = []
    for window in CRISIS_WINDOWS:
        weighted_ret: pd.Series | None = None
        per_ticker_dd: dict[str, float] = {}
        for a in ok:
            sub = slice_window(
                pd.DataFrame({LOG_RETURN_COL: a.log_returns}), window
            )[LOG_RETURN_COL].dropna()
            if len(sub) < 5:
                continue
            equity = np.exp(sub.cumsum())
            peak = equity.cummax()
            dd = float((equity / peak - 1.0).min())
            per_ticker_dd[a.ticker] = dd
            w = weights.get(a.ticker, 0.0)
            contrib = sub * w
            weighted_ret = contrib if weighted_ret is None else weighted_ret.add(
                contrib, fill_value=0.0
            )

        if weighted_ret is None or weighted_ret.empty:
            continue

        port_equity = np.exp(weighted_ret.cumsum())
        port_peak = port_equity.cummax()
        port_dd = float((port_equity / port_peak - 1.0).min())
        port_return = float(port_equity.iloc[-1] - 1.0)

        rows.append({
            "Window": window.label,
            "Period": f"{window.start} → {window.end}",
            "Portfolio Return": port_return,
            "Portfolio Max DD": port_dd,
            "Worst Single DD": (min(per_ticker_dd.values()) if per_ticker_dd else 0.0),
            "Worst Ticker": (min(per_ticker_dd, key=per_ticker_dd.get)
                             if per_ticker_dd else "—"),
        })

    return pd.DataFrame(rows)
