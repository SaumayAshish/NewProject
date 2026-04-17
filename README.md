# Modular Stock Selection + Portfolio Optimization

This project provides an end-to-end, explainable pipeline for:

1. Building a stock universe
2. Fetching and cleaning yfinance data
3. Computing value/quality/trend/risk features
4. Screening and ranking potentially undervalued stocks
5. Applying time-series diagnostics + trust scoring
6. Estimating expected returns and covariance
7. Running constrained portfolio optimization (PyPortfolioOpt)
8. Exporting ranked candidates and portfolio tables

## Pipeline stages

`Universe -> Fetch -> Clean -> Features -> Screener -> Ranker -> Time-series -> Return/Risk -> Optimizer -> Reporting`

## Quick start

```bash
pip install -r requirements.txt
python main.py
pytest -q
```

## Dependency-safe execution

If third-party dependencies are missing, `python main.py` now automatically runs a **demo fallback mode** and still writes output CSVs. Use `--no-demo-fallback` to force strict full-mode execution.

```bash
python main.py --no-demo-fallback
```

## Outputs

Running `python main.py` writes:

- `reports/tables/ranked_candidates.csv`
- `reports/tables/portfolio_positions.csv`
- `reports/tables/portfolio_metrics.csv`
- `reports/logs/pipeline.log` (full mode)

## Modules

- `src/config.py`: project defaults and regime-aware score weights
- `src/universe.py`: universe loading and metadata
- `src/fetcher.py`: yfinance data access
- `src/cleaner.py`: normalization and quality filtering
- `src/features.py`: value, quality, trend, and risk features
- `src/screener.py`: basic and sector-relative filtering
- `src/ranker.py`: composite ranking engine
- `src/timeseries.py`: diagnostics, forecasting, trust scoring
- `src/expected_returns.py`: expected-return and covariance estimators
- `src/optimizer.py`: optimization wrappers and portfolio evaluation
- `src/reporting.py`: table and chart generation helpers
