"""Time-series diagnostics, lightweight forecasting, and trust scoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class ForecastMetrics:
    rmse: float
    mae: float


def prepare_series(price_df: pd.DataFrame, ticker: str) -> pd.Series:
    g = price_df[price_df["ticker"] == ticker].sort_values("date")
    if g.empty:
        raise ValueError(f"No price rows found for ticker={ticker}")
    col = "adj_close" if "adj_close" in g.columns else "close"
    return pd.Series(g[col].values, index=pd.to_datetime(g["date"]), name=ticker)


def compute_returns(series: pd.Series) -> pd.Series:
    return series.astype(float).pct_change().dropna()


def moving_averages(series: pd.Series, windows: list[int] | None = None) -> pd.DataFrame:
    windows = windows or [20, 50, 200]
    return pd.DataFrame({f"ma_{w}": series.rolling(w).mean() for w in windows})


def rolling_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)


def rolling_sharpe(returns: pd.Series, window: int = 60, rf: float = 0.0) -> pd.Series:
    roll_mean = returns.rolling(window).mean() * 252
    roll_std = returns.rolling(window).std() * np.sqrt(252)
    return (roll_mean - rf) / roll_std.replace(0, np.nan)


def max_drawdown(series: pd.Series) -> float:
    r = series.pct_change().fillna(0.0)
    cum = (1 + r).cumprod()
    return float(((cum / cum.cummax()) - 1).min())


def trend_regression(series: pd.Series) -> dict[str, float]:
    y = np.log(series.astype(float).clip(lower=1e-9).values)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def adf_stationarity_test(series: pd.Series) -> dict[str, float]:
    stat, pvalue, *_ = adfuller(series.dropna())
    return {"adf_stat": float(stat), "pvalue": float(pvalue)}


def fit_arima(series: pd.Series, order: tuple[int, int, int] = (1, 1, 1)) -> Any:
    return ARIMA(series.astype(float), order=order).fit()


def fit_prophet(df: pd.DataFrame) -> Any:
    """Optional Prophet adapter.

    Returns None if prophet is unavailable so callers can degrade gracefully.
    """
    try:
        from prophet import Prophet  # type: ignore
    except Exception:
        return None

    if "ds" not in df.columns or "y" not in df.columns:
        raise ValueError("Prophet input requires columns: ds, y")
    model = Prophet()
    model.fit(df[["ds", "y"]])
    return model


def forecast_and_backtest(series: pd.Series, horizon: int = 30) -> dict[str, float]:
    if len(series) <= horizon + 30:
        return {"rmse": np.nan, "mae": np.nan}
    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]
    model = fit_arima(train)
    fc = model.forecast(steps=horizon)
    err = fc - test
    return {"rmse": float(np.sqrt(np.mean(np.square(err)))), "mae": float(np.mean(np.abs(err)))}


def summarize_stock_behavior(series: pd.Series) -> dict[str, float]:
    rets = compute_returns(series)
    trend = trend_regression(series)["slope"]
    vol = float(rets.std() * np.sqrt(252)) if not rets.empty else np.nan
    mdd = abs(max_drawdown(series))
    fc = forecast_and_backtest(series)
    return {"trend_slope": trend, "volatility": vol, "max_drawdown": mdd, **fc}


def trust_score(trend_stability: float, volatility_control: float, drawdown_resilience: float, forecast_consistency: float) -> float:
    score = 0.30 * trend_stability + 0.25 * volatility_control + 0.25 * drawdown_resilience + 0.20 * forecast_consistency
    return round(max(0.0, min(100.0, score * 100)), 2)


def trust_label(score: float) -> str:
    if score >= 75:
        return "High"
    if score >= 50:
        return "Moderate"
    return "Low"
