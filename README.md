# STAT5291 Capstone: Gold vs. Equity Decoupling and Regime States

This repository supports a capstone on whether gold behaves as a **safe haven** or **hedge** relative to the S&P 500, and whether that relationship is **state-dependent** (e.g., stress vs. calm). The main deliverable is the Jupyter notebook; supporting code and processed data live alongside it.

## Executive summary

We study gold’s protective properties using daily market data: precious-metals futures, the S&P 500, the VIX, and macro series from FRED. The analysis combines exploratory work (returns, descriptive statistics, visuals) with a **two-state Gaussian hidden Markov model** (`hmmlearn`) on engineered features, plus walk-forward checks and a simple illustrative backtest. The broader research agenda still includes formal tail-risk metrics (e.g., CVaR), linear and quantile regression, and an explicit **GARCH → HMM** feature pipeline as extensions beyond the current notebook.

## Repository layout

| Path | Role |
|------|------|
| `STAT5291 Capstone Code.ipynb` | Presentation layer: load processed CSVs, call `utility_func` for plots, stats, and HMM workflow |
| `utility_func.py` | FRED (`fetch_fred_series`), plotting, descriptive stats, stress-event figures, and HMM helpers; path constants (`ROOT`, `DATA_RAW_DIR`, `DATA_PROCESSED_DIR`) |
| `data_pipeline.py` | CLI to refresh data: `python data_pipeline.py` (or `--step commodities|indices|panel|logret|hmm`) writes raw files under `data/data_raw/` and processed tables under `data/data_processed/` |
| `requirements.txt` | Python dependencies |
| `data/data_raw/` | Inputs downloaded or supplied externally (e.g., Kaggle commodities CSV, `sp500.csv`, `vix.csv`, FRED exports, `global_market_stress_events.csv`) |
| `data/data_processed/` | All merged / derived tables from the project pipeline (e.g., `all_commodities_data1.csv`, `data.csv`, `LogRet_data.csv`, `HMM_features.csv`) |
| `figure/` | Saved plots (directory present for outputs) |
| `data/panic_days.csv` | Optional panic-day list used when building the panel (read via `utility_func.ROOT / "data" / "panic_days.csv"` in the notebook) |

Regenerated analysis CSVs are **not** written to the repository root; the notebook and pipeline use `utility_func.DATA_RAW_DIR` / `utility_func.DATA_PROCESSED_DIR` (pathlib `Path` objects).

## Setup

1. **Python:** 3.10+ recommended (see `requirements.txt`).

2. **Virtual environment (optional but recommended):**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **FRED API key:** The HMM data-prep cells call `utility_func.fetch_fred_series`, which reads `FRED_API_KEY` from a `.env` file at the project root. `.env` is gitignored; create it with:

   ```env
   FRED_API_KEY=your_key_here
   ```

   Register for a key at [FRED API documentation](https://fred.stlouisfed.org/docs/api/api_key.html).

4. **Refresh data (before or without the notebook):** From the repo root, run `python data_pipeline.py` to download via `yfinance`, merge the panel, build `LogRet_data.csv` and `HMM_features.csv` (FRED step needs `FRED_API_KEY`). Use `--step` to run a subset.

5. **Run the notebook:** Open `STAT5291 Capstone Code.ipynb` in Jupyter / VS Code, select the environment where you installed `requirements.txt`, and run cells in order. The notebook reads processed tables from `data/data_processed/` and raw inputs from `data/data_raw/` (it does not redefine analysis functions).

## Data sources (as used in code)

- **Precious metals futures (Yahoo Finance via `yfinance`):** e.g. `GC=F` (gold), `SI=F`, `PL=F`, `HG=F` (copper), `PA=F`. 
- **Equity index & volatility:** `^GSPC`, `^VIX`.
- **Macro (FRED):** e.g. **DFII10** (10-year TIPS real yield), **T10YIE** (10-year breakeven inflation). For ratios, the notebook also pulls **CL=F** (WTI) and **HG=F** alongside gold.
- **Processed returns:** `data/data_processed/LogRet_data.csv` is merged into the HMM feature set (e.g., gold and S&P 500 log returns).

## What the notebook implements

- **Data ingestion:** Handled by `data_pipeline.py` (notebook documents the CLI; optional commented cell can call `data_pipeline.run_all()`).
- **EDA:** Time series plots, log returns, descriptive statistics, and visualizations across assets (including stress-event overlays where those cells are run).
- **HMM pipeline:**
  - Build a panel aligned to trading dates (calendar merge / forward-fill of macro as in the notebook).
  - **Features used in the fitted example:** `gold_log_ret`, `sp500_log_ret`, first differences of `DFII10`, and changes in **gold/copper ratio** (`d_gold_copper`).
  - **Model:** `hmmlearn.hmm.GaussianHMM` with two components, full covariance, EM iterations as set in the notebook (note: `GaussianHMM` does **not** take sklearn-style `n_init`; use multiple `random_state` runs and compare `score` if you need many random starts).
  - **Follow-on:** In-sample state labeling, discriminative feature summaries, walk-forward refitting, out-of-sample grouping, and a simple cumulative-return comparison plot.

## Research roadmap (paper / extensions)

These items match the original project design and README outline; implement or tighten them in the notebook as the write-up requires.

- **Tail risk:** Static CVaR / ES comparisons across gold, equities, and copper.
- **Static linear model:** OLS-style sensitivity of gold returns to S&P and VIX (or expanded controls).
- **Quantile regression:** How $\beta(\tau)$ moves from central to tail quantiles.
- **GARCH–HMM:** Use GARCH(1,1) or EGARCH conditional variances (or residuals) as inputs to the HMM instead of or in addition to raw returns and macro deltas.

## Expected results (hypotheses)

- **Regime-specific behavior:** Gold’s safe-haven role may concentrate in high-stress HMM states rather than being stable over time.
- **Downside risk:** Quantitative comparison of tail loss across metals and equities once CVaR (or similar) is fully wired in.
- **Early warning:** Whether state probabilities or transitions add information beyond the VIX for stress timing (to be assessed with the walk-forward and any formal metrics you add).

## Dependencies

See `requirements.txt` for pinned minimum versions, including **numpy**, **pandas**, **scipy**, **scikit-learn**, **yfinance**, **hmmlearn**, **matplotlib**, **seaborn**, **python-dotenv**, and **jupyter** / **ipykernel**.
