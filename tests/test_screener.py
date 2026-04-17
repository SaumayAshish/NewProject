import pandas as pd

from src.screener import apply_basic_filters, apply_sector_relative_filters


def test_basic_filters_remove_invalid_rows():
    df = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "sector": ["Tech", None],
            "market_cap": [2e9, 2e9],
            "trailing_pe": [12, 10],
            "price_to_book": [2, 2],
            "debt_to_equity": [1.0, 1.0],
            "roe": [0.1, 0.1],
        }
    )
    out = apply_basic_filters(df, min_market_cap=1e9)
    assert out["ticker"].tolist() == ["A"]


def test_sector_relative_filters():
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "sector": ["Tech", "Tech", "Health"],
            "trailing_pe": [10, 30, 8],
            "price_to_book": [2, 5, 1],
            "roe": [0.2, 0.1, 0.15],
        }
    )
    out = apply_sector_relative_filters(df)
    assert "A" in out["ticker"].tolist()
    assert "B" not in out["ticker"].tolist()
