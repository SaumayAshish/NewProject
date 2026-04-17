import numpy as np
import pandas as pd

from src.features import compute_quality_features, compute_risk_features, compute_trend_features, compute_value_features


def test_value_quality_scores_added():
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "trailing_pe": [10, 20, 15],
            "forward_pe": [8, 18, 11],
            "price_to_book": [1, 3, 2],
            "price_to_sales": [1, 4, 2],
            "ev_to_ebitda": [7, 14, 9],
            "roe": [0.2, 0.1, 0.15],
            "current_ratio": [2, 1.1, 1.5],
            "debt_to_equity": [0.4, 1.2, 0.6],
        }
    )
    scored = compute_quality_features(compute_value_features(df))
    assert "value_score" in scored.columns
    assert "quality_score" in scored.columns
    assert scored["value_score"].between(0, 1).all()


def test_trend_and_risk_features_shape():
    n = 260
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    rows = []
    for t, drift in [("A", 0.001), ("B", 0.0005)]:
        price = 100 * np.cumprod(1 + np.random.default_rng(0).normal(drift, 0.01, n))
        rows.extend({"ticker": t, "date": d, "close": p, "adj_close": p} for d, p in zip(dates, price))
    price_df = pd.DataFrame(rows)
    trend = compute_trend_features(price_df)
    risk = compute_risk_features(price_df)
    assert set(trend["ticker"]) == {"A", "B"}
    assert set(risk["ticker"]) == {"A", "B"}
