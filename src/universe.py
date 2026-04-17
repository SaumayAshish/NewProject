"""Universe construction utilities."""
from __future__ import annotations

import pandas as pd
import yfinance as yf


DEFAULT_COLUMNS = ["ticker", "sector", "industry", "exchange", "country"]


def load_universe_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError("CSV must include a 'ticker' column")
    return df


def get_sp500_tickers() -> list[str]:
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    table = tables[0]
    return table["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()


def get_nifty50_tickers() -> list[str]:
    table = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_50")[2]
    symbols = table["Symbol"].astype(str).str.strip()
    return [f"{s}.NS" for s in symbols]


def attach_sector_metadata(tickers: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for ticker in tickers:
        info = yf.Ticker(ticker).info
        rows.append(
            {
                "ticker": ticker,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "exchange": info.get("exchange"),
                "country": info.get("country"),
            }
        )
    return pd.DataFrame(rows, columns=DEFAULT_COLUMNS)
