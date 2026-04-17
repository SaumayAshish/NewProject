"""Portfolio optimization wrappers around PyPortfolioOpt."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, objective_functions


def _ef(mu: pd.Series, cov: pd.DataFrame, bounds: tuple[float, float]) -> EfficientFrontier:
    return EfficientFrontier(mu, cov, weight_bounds=bounds)


def optimize_min_vol(mu: pd.Series, cov: pd.DataFrame, bounds: tuple[float, float]) -> dict[str, float]:
    ef = _ef(mu, cov, bounds)
    ef.min_volatility()
    return ef.clean_weights()


def optimize_max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float, bounds: tuple[float, float]) -> dict[str, float]:
    ef = _ef(mu, cov, bounds)
    ef.max_sharpe(risk_free_rate=rf)
    return ef.clean_weights()


def optimize_target_return(mu: pd.Series, cov: pd.DataFrame, target_return: float, bounds: tuple[float, float]) -> dict[str, float]:
    ef = _ef(mu, cov, bounds)
    ef.efficient_return(target_return)
    return ef.clean_weights()


def optimize_target_risk(mu: pd.Series, cov: pd.DataFrame, target_vol: float, bounds: tuple[float, float]) -> dict[str, float]:
    ef = _ef(mu, cov, bounds)
    ef.efficient_risk(target_vol)
    return ef.clean_weights()


def add_sector_constraints(ef: EfficientFrontier, sector_mapper: dict[str, str], lower: float = 0.0, upper: float = 0.35) -> None:
    sectors = defaultdict(list)
    for ticker, sector in sector_mapper.items():
        sectors[sector].append(ticker)
    ef.add_sector_constraints(sector_mapper, {k: lower for k in sectors}, {k: upper for k in sectors})


def add_l2_regularization(ef: EfficientFrontier, gamma: float = 0.01) -> None:
    ef.add_objective(objective_functions.L2_reg, gamma=gamma)


def evaluate_portfolio(weights: dict[str, float], mu: pd.Series, cov: pd.DataFrame, rf: float = 0.03) -> dict[str, float]:
    w = pd.Series(weights).reindex(mu.index).fillna(0.0)
    ret = float(np.dot(w, mu))
    vol = float(np.sqrt(np.dot(w.values.T, np.dot(cov.values, w.values))))
    sharpe = (ret - rf) / vol if vol else np.nan
    return {"expected_return": ret, "volatility": vol, "sharpe": float(sharpe)}
