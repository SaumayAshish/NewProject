"""Ranking engine using composite scores."""
from __future__ import annotations

import pandas as pd


def normalize_scores(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        lo, hi = out[col].min(), out[col].max()
        out[col] = 0.5 if hi == lo else (out[col] - lo) / (hi - lo)
    return out


def build_composite_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["composite_score"] = 0.0
    for col, wt in weights.items():
        out["composite_score"] += out[col] * wt
    return out


def rank_candidates(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    cols = list(weights.keys())
    out = normalize_scores(df, cols)
    out = build_composite_score(out, weights)
    out = out.sort_values("composite_score", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out
