"""Expected return and covariance estimators."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns_matrix(price_panel: pd.DataFrame) -> pd.DataFrame:
    return price_panel.sort_index().pct_change().dropna(how="all")


def mean_historical_return(prices: pd.DataFrame, periods: int = 252) -> pd.Series:
    rets = compute_returns_matrix(prices)
    return rets.mean() * periods


def ema_historical_return(prices: pd.DataFrame, span: int = 60, periods: int = 252) -> pd.Series:
    rets = compute_returns_matrix(prices)
    return rets.ewm(span=span).mean().iloc[-1] * periods


def capm_proxy_return(prices: pd.DataFrame, benchmark: pd.Series, rf: float = 0.03, periods: int = 252) -> pd.Series:
    rets = compute_returns_matrix(prices)
    b = benchmark.pct_change().reindex(rets.index).dropna()
    aligned = rets.loc[b.index]
    mkt_premium = b.mean() * periods - rf
    out = {}
    for col in aligned.columns:
        beta = aligned[col].cov(b) / (b.var() if b.var() else 1)
        out[col] = rf + beta * mkt_premium
    return pd.Series(out)


def blended_expected_return(prices: pd.DataFrame, fundamentals_scores: pd.Series) -> pd.Series:
    ema = ema_historical_return(prices)
    fs = (fundamentals_scores - fundamentals_scores.min()) / (
        (fundamentals_scores.max() - fundamentals_scores.min()) or 1
    )
    fs = fs.reindex(ema.index).fillna(fs.mean())
    return 0.6 * ema + 0.4 * fs


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov()


def shrink_covariance(returns: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    sample = returns.cov()
    diag = np.diag(np.diag(sample.values))
    shrunk = (1 - alpha) * sample.values + alpha * diag
    return pd.DataFrame(shrunk, index=sample.index, columns=sample.columns)


def annualize_covariance(cov: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    return cov * periods
