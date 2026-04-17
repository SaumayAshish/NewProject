import pandas as pd

from src.cleaner import align_trading_dates


def test_align_trading_dates_intersection():
    a = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "close": [1, 2, 3],
        }
    )
    b = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-04"]),
            "close": [4, 5, 6],
        }
    )
    out = align_trading_dates({"A": a, "B": b})
    dates = out["date"].drop_duplicates().dt.strftime("%Y-%m-%d").tolist()
    assert dates == ["2025-01-02", "2025-01-03"]
