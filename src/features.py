"""Feature engineering for value, quality, trend and risk."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rank_inverse(series: pd.Series) -> pd.Series:
    return 1 - series.rank(pct=True, na_option="bottom")


def _rank_direct(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, na_option="bottom")


def compute_value_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ratios = ["trailing_pe", "forward_pe", "price_to_book", "price_to_sales", "ev_to_ebitda"]
    scored = [_rank_inverse(out[c]) for c in ratios if c in out.columns]
    out["value_score"] = pd.concat(scored, axis=1).mean(axis=1) if scored else 0.0
    return out


def compute_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parts = []
    if "roe" in out.columns:
        parts.append(_rank_direct(out["roe"]))
    if "current_ratio" in out.columns:
        parts.append(_rank_direct(out["current_ratio"]))
    if "debt_to_equity" in out.columns:
        parts.append(_rank_inverse(out["debt_to_equity"]))
    out["quality_score"] = pd.concat(parts, axis=1).mean(axis=1) if parts else 0.0
    return out


def compute_trend_features(price_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, g in price_df.groupby("ticker"):
        g = g.sort_values("date")
        close = g["adj_close"] if "adj_close" in g.columns else g["close"]
        if len(close) < 252:
            continue
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        ret_6m = close.iloc[-1] / close.iloc[-126] - 1
        ret_12m = close.iloc[-1] / close.iloc[-252] - 1
        rows.append(
            {
                "ticker": ticker,
                "ret_6m": ret_6m,
                "ret_12m": ret_12m,
                "ma_signal": float(ma50 > ma200),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["trend_score"] = out[["ret_6m", "ret_12m", "ma_signal"]].rank(pct=True).mean(axis=1)
    return out


def compute_risk_features(price_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, g in price_df.groupby("ticker"):
        g = g.sort_values("date")
        close = g["adj_close"] if "adj_close" in g.columns else g["close"]
        rets = close.pct_change().dropna()
        if rets.empty:
            continue
        vol = rets.std() * np.sqrt(252)
        downside = rets[rets < 0].std() * np.sqrt(252)
        cum = (1 + rets).cumprod()
        peak = cum.cummax()
        mdd = ((cum / peak) - 1).min()
        rows.append({"ticker": ticker, "volatility": vol, "downside_dev": downside, "max_drawdown": abs(mdd)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["risk_penalty"] = out[["volatility", "downside_dev", "max_drawdown"]].rank(pct=True).mean(axis=1)
    return out


def compute_sector_relative_zscores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["value_score", "quality_score", "trend_score", "risk_penalty"]:
        if col in out.columns:
            out[f"{col}_z"] = out.groupby("sector")[col].transform(
                lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1)
            )
    return out
