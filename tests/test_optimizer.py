import pandas as pd

from src.optimizer import evaluate_portfolio, optimize_min_vol


def test_optimize_min_vol_weights_sum_to_one():
    mu = pd.Series([0.12, 0.10, 0.08], index=["A", "B", "C"])
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.0], [0.01, 0.03, 0.0], [0.0, 0.0, 0.02]],
        index=mu.index,
        columns=mu.index,
    )
    w = optimize_min_vol(mu, cov, (0, 1))
    assert abs(sum(w.values()) - 1) < 1e-4


def test_evaluate_portfolio_outputs_metrics():
    mu = pd.Series([0.1, 0.2], index=["A", "B"])
    cov = pd.DataFrame([[0.02, 0.0], [0.0, 0.03]], index=mu.index, columns=mu.index)
    perf = evaluate_portfolio({"A": 0.5, "B": 0.5}, mu, cov, 0.03)
    assert set(perf.keys()) == {"expected_return", "volatility", "sharpe"}
