"""Data cleaning and validation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


PRICE_COL_MAP = {
    "adj close": "adj_close",
    "adjclose": "adj_close",
}


def normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    out = out.rename(columns=PRICE_COL_MAP)
    return out


def align_trading_dates(price_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    aligned = []
    common_dates = None
    for ticker, df in price_dict.items():
        if df.empty:
            continue
        tmp = normalize_price_columns(df)
        tmp["date"] = pd.to_datetime(tmp["date"])
        dates = set(tmp["date"].dt.normalize())
        common_dates = dates if common_dates is None else common_dates & dates
    if not common_dates:
        return pd.DataFrame()

    idx = pd.DatetimeIndex(sorted(common_dates))
    for ticker, df in price_dict.items():
        if df.empty:
            continue
        tmp = normalize_price_columns(df)
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.normalize()
        tmp = tmp[tmp["date"].isin(idx)].sort_values("date")
        tmp["ticker"] = ticker
        aligned.append(tmp)
    return pd.concat(aligned, ignore_index=True) if aligned else pd.DataFrame()


def handle_missing_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
    return out


def winsorize_outliers(df: pd.DataFrame, cols: list[str], lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            lo, hi = out[col].quantile(lower), out[col].quantile(upper)
            out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def drop_low_quality_tickers(df: pd.DataFrame, min_days: int, max_missing_ratio: float) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = df.groupby("ticker")
    keep = []
    for ticker, g in grouped:
        if len(g) < min_days:
            continue
        miss_ratio = g.isna().mean().mean()
        if miss_ratio <= max_missing_ratio:
            keep.append(ticker)
    return df[df["ticker"].isin(keep)].copy()
