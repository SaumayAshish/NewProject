"""Hard-rule screener for candidate selection."""
from __future__ import annotations

import pandas as pd


def apply_basic_filters(
    df: pd.DataFrame,
    min_market_cap: float,
    max_debt_to_equity: float = 2.0,
    min_roe: float = 0.0,
) -> pd.DataFrame:
    out = df.copy()
    mask = (
        (out["market_cap"] >= min_market_cap)
        & out["sector"].notna()
        & (out["trailing_pe"] > 0)
        & (out["price_to_book"] > 0)
        & (out["debt_to_equity"] <= max_debt_to_equity)
        & (out["roe"] >= min_roe)
    )
    return out[mask].copy()


def apply_sector_relative_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    med = out.groupby("sector")[["trailing_pe", "price_to_book", "roe"]].median().rename(
        columns={"trailing_pe": "med_pe", "price_to_book": "med_pb", "roe": "med_roe"}
    )
    out = out.merge(med, left_on="sector", right_index=True, how="left")
    keep = (out["trailing_pe"] <= out["med_pe"]) & (out["price_to_book"] <= out["med_pb"]) & (out["roe"] >= out["med_roe"])
    return out[keep].copy()


def select_candidate_pool(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    if "composite_score" not in df.columns:
        raise ValueError("composite_score column required")
    return df.sort_values("composite_score", ascending=False).head(top_n).copy()
