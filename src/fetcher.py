"""Data access layer for yfinance."""
from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_price_history(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(start=start, end=end, interval=interval, auto_adjust=False)
    if not df.empty:
        df = df.reset_index().rename(columns=str.lower)
        df["ticker"] = ticker
    return df


def fetch_prices_batch(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    return {ticker: fetch_price_history(ticker, start=start, end=end) for ticker in tickers}


def fetch_ticker_info(ticker: str) -> dict:
    return yf.Ticker(ticker).info


def fetch_balance_sheet(ticker: str) -> pd.DataFrame:
    return yf.Ticker(ticker).balance_sheet


def fetch_income_statement(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    return t.income_stmt if hasattr(t, "income_stmt") else t.quarterly_income_stmt


def fetch_cashflow(ticker: str) -> pd.DataFrame:
    return yf.Ticker(ticker).cashflow


def fetch_fundamentals_batch(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        info = fetch_ticker_info(ticker)
        rows.append(
            {
                "ticker": ticker,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "roe": info.get("returnOnEquity"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "beta": info.get("beta"),
                "dividend_yield": info.get("dividendYield"),
                "shares_outstanding": info.get("sharesOutstanding"),
            }
        )
    return pd.DataFrame(rows)
