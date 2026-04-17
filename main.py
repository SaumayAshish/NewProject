"""End-to-end runner for the portfolio optimization pipeline."""
from __future__ import annotations

import argparse
import csv
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

REQUIRED_PACKAGES = ["numpy", "pandas", "yfinance", "pypfopt", "statsmodels", "matplotlib"]


def missing_packages() -> list[str]:
    return [pkg for pkg in REQUIRED_PACKAGES if importlib.util.find_spec(pkg) is None]


def _write_demo_outputs() -> tuple[str, str, str]:
    Path("reports/tables").mkdir(parents=True, exist_ok=True)

    ranked_path = "reports/tables/ranked_candidates.csv"
    positions_path = "reports/tables/portfolio_positions.csv"
    metrics_path = "reports/tables/portfolio_metrics.csv"

    with open(ranked_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "sector", "composite_score", "trust_score", "final_score", "rank"])
        w.writerow(["DEMO1", "Technology", 0.82, 73.0, 0.7975, 1])
        w.writerow(["DEMO2", "Healthcare", 0.79, 68.0, 0.7625, 2])
        w.writerow(["DEMO3", "Financials", 0.75, 61.0, 0.7150, 3])

    with open(positions_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "weight"])
        w.writerow(["DEMO1", 0.40])
        w.writerow(["DEMO2", 0.35])
        w.writerow(["DEMO3", 0.25])

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["expected_return", "volatility", "sharpe", "mode", "generated_at_utc"])
        w.writerow([0.14, 0.18, 0.61, "demo", datetime.utcnow().isoformat()])

    return ranked_path, positions_path, metrics_path


def run_demo_mode() -> dict[str, Any]:
    outputs = _write_demo_outputs()
    return {
        "mode": "demo",
        "outputs": outputs,
        "message": (
            "Demo mode used because required third-party dependencies are unavailable. "
            "Install requirements.txt and rerun for full live pipeline."
        ),
    }


def run_full_pipeline(tickers: list[str] | None = None) -> tuple[Any, dict[str, float], dict[str, float]]:
    import numpy as np
    import pandas as pd

    from src import config
    from src.cleaner import align_trading_dates, handle_missing_fundamentals
    from src.expected_returns import annualize_covariance, blended_expected_return, compute_returns_matrix, shrink_covariance
    from src.features import compute_quality_features, compute_risk_features, compute_trend_features, compute_value_features
    from src.fetcher import fetch_fundamentals_batch, fetch_prices_batch
    from src.optimizer import evaluate_portfolio, optimize_max_sharpe
    from src.ranker import rank_candidates
    from src.reporting import generate_portfolio_summary, generate_ranked_candidates_table
    from src.screener import apply_basic_filters, apply_sector_relative_filters, select_candidate_pool
    from src.timeseries import prepare_series, summarize_stock_behavior, trust_label, trust_score
    from src.universe import get_sp500_tickers
    from src.utils import setup_logger

    def detect_market_regime(price_data: pd.DataFrame, benchmark_ticker: str = "SPY") -> str:
        bench = price_data[price_data["ticker"] == benchmark_ticker].sort_values("date")
        if bench.empty:
            return "bull"
        col = "adj_close" if "adj_close" in bench.columns else "close"
        rets = bench[col].pct_change().dropna()
        if rets.empty:
            return "bull"
        vol = rets.std() * np.sqrt(252)
        ret_6m = bench[col].iloc[-1] / bench[col].iloc[max(0, len(bench) - 126)] - 1
        if vol > 0.30:
            return "volatile"
        if ret_6m < -0.05:
            return "bear"
        return "bull"

    def apply_trust_scores(candidates: pd.DataFrame, aligned_prices: pd.DataFrame) -> pd.DataFrame:
        out = candidates.copy()
        scores = []
        for ticker in out["ticker"]:
            try:
                series = prepare_series(aligned_prices, ticker)
                summary = summarize_stock_behavior(series)
                trend_stability = float(np.clip((summary["trend_slope"] + 0.002) / 0.004, 0, 1))
                volatility_control = float(np.clip(1 - summary["volatility"] / 0.6, 0, 1))
                drawdown_resilience = float(np.clip(1 - summary["max_drawdown"] / 0.7, 0, 1))
                fc_rmse = summary.get("rmse", np.nan)
                forecast_consistency = 0.5 if np.isnan(fc_rmse) else float(np.clip(1 - fc_rmse / float(series.mean()), 0, 1))
                tscore = trust_score(trend_stability, volatility_control, drawdown_resilience, forecast_consistency)
            except Exception:
                tscore = 50.0
            scores.append(tscore)
        out["trust_score"] = scores
        out["trust_label"] = out["trust_score"].apply(trust_label)
        out["final_score"] = 0.75 * out["composite_score"] + 0.25 * (out["trust_score"] / 100.0)
        out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
        out["rank"] = out.index + 1
        return out

    def _build_price_panel(aligned: pd.DataFrame, tickers_list: list[str]) -> pd.DataFrame:
        panel = []
        for t in tickers_list:
            g = aligned[aligned["ticker"] == t].sort_values("date")
            col = "adj_close" if "adj_close" in g.columns else "close"
            panel.append(g.set_index("date")[[col]].rename(columns={col: t}))
        return pd.concat(panel, axis=1).dropna() if panel else pd.DataFrame()

    logger = setup_logger("pipeline")
    tickers = tickers or get_sp500_tickers()[:100]
    logger.info("Universe size: %s", len(tickers))

    prices = fetch_prices_batch(tickers, config.START_DATE, config.END_DATE)
    fundamentals = handle_missing_fundamentals(fetch_fundamentals_batch(tickers))

    aligned = align_trading_dates(prices)
    if aligned.empty:
        raise RuntimeError("No aligned price data available")

    regime = detect_market_regime(aligned)
    logger.info("Detected regime: %s", regime)

    trend_df = compute_trend_features(aligned)
    risk_df = compute_risk_features(aligned)

    scored = compute_value_features(fundamentals)
    scored = compute_quality_features(scored)
    scored = scored.merge(trend_df[["ticker", "trend_score"]], on="ticker", how="left")
    scored = scored.merge(risk_df[["ticker", "risk_penalty"]], on="ticker", how="left")
    scored = scored.dropna(subset=["trend_score", "risk_penalty"])

    screened = apply_basic_filters(scored, min_market_cap=config.MIN_MARKET_CAP)
    screened = apply_sector_relative_filters(screened)
    weights = config.COMPOSITE_WEIGHTS
    if regime in config.REGIME_SCORE_WEIGHTS:
        w = config.REGIME_SCORE_WEIGHTS[regime]
        weights = {"value_score": w.value, "quality_score": w.quality, "trend_score": w.trend, "risk_penalty": w.risk_penalty}

    ranked = rank_candidates(screened, weights)
    candidates = select_candidate_pool(ranked, top_n=config.TOP_N)
    candidates = apply_trust_scores(candidates, aligned)

    price_panel = _build_price_panel(aligned, candidates["ticker"].tolist())
    if price_panel.empty:
        raise RuntimeError("No candidate price panel available for optimization")

    mu = blended_expected_return(price_panel, candidates.set_index("ticker")["final_score"])
    cov = annualize_covariance(shrink_covariance(compute_returns_matrix(price_panel)))
    weights_map = optimize_max_sharpe(mu, cov, config.RISK_FREE_RATE, config.WEIGHT_BOUNDS)
    perf = evaluate_portfolio(weights_map, mu, cov, config.RISK_FREE_RATE)

    ranked_table = generate_ranked_candidates_table(candidates)
    positions, metrics = generate_portfolio_summary(weights_map, perf)

    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    ranked_table.to_csv("reports/tables/ranked_candidates.csv", index=False)
    positions.to_csv("reports/tables/portfolio_positions.csv", index=False)
    metrics.to_csv("reports/tables/portfolio_metrics.csv", index=False)

    logger.info("Final holdings: %s", positions.to_dict(orient="records"))
    return ranked_table, weights_map, perf


def run_pipeline(tickers: list[str] | None = None, allow_demo_fallback: bool = True) -> dict[str, Any]:
    missing = missing_packages()
    if missing and allow_demo_fallback:
        result = run_demo_mode()
        result["missing_packages"] = missing
        return result

    ranked_table, weights, perf = run_full_pipeline(tickers=tickers)
    return {
        "mode": "full",
        "ranked_rows": len(ranked_table),
        "weights": weights,
        "performance": perf,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stock-selection and portfolio optimization pipeline")
    parser.add_argument("--no-demo-fallback", action="store_true", help="Fail fast if dependencies are missing")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional explicit ticker list")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_pipeline(tickers=args.tickers, allow_demo_fallback=not args.no_demo_fallback)
    print(result)
