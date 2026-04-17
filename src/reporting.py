"""Reporting helpers for tables, text explanations, and charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate_screening_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["ticker", "sector", "market_cap", "trailing_pe", "price_to_book", "roe"] if c in df.columns]
    return df[cols].sort_values("market_cap", ascending=False)


def generate_ranked_candidates_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["ticker", "sector", "value_score", "quality_score", "trend_score", "risk_penalty", "composite_score", "trust_score", "final_score", "rank"]
    keep = [c for c in cols if c in df.columns]
    return df[keep].sort_values("rank")


def plot_stock_diagnostics(ticker_data: pd.DataFrame, ticker: str, out_dir: str = "reports/charts") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    g = ticker_data[ticker_data["ticker"] == ticker].sort_values("date")
    if g.empty:
        raise ValueError(f"No rows for ticker={ticker}")
    col = "adj_close" if "adj_close" in g.columns else "close"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(g["date"]), g[col], label="close")
    ax.set_title(f"{ticker} price history")
    ax.legend()
    path = Path(out_dir) / f"{ticker}_price.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_correlation_heatmap(returns: pd.DataFrame, out_path: str = "reports/charts/correlation_heatmap.png") -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_efficient_frontier(mu: pd.Series, cov: pd.DataFrame, out_path: str = "reports/charts/efficient_frontier.png") -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    vols = cov.apply(lambda c: c.mean() ** 0.5)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(vols.values, mu.values)
    for ticker in mu.index:
        ax.annotate(ticker, (vols[ticker], mu[ticker]))
    ax.set_xlabel("Approx volatility")
    ax.set_ylabel("Expected return")
    ax.set_title("Asset return vs risk")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_portfolio_summary(weights: dict[str, float], perf: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [{"ticker": k, "weight": v} for k, v in weights.items() if v > 0]
    pos = pd.DataFrame(rows).sort_values("weight", ascending=False)
    metrics = pd.DataFrame([perf])
    return pos, metrics


def build_reason_text(row: pd.Series) -> str:
    return (
        f"{row['ticker']} selected for attractive value ({row.get('value_score', 0):.2f}), "
        f"quality ({row.get('quality_score', 0):.2f}), trend ({row.get('trend_score', 0):.2f}) "
        f"and manageable risk penalty ({row.get('risk_penalty', 0):.2f})."
    )
