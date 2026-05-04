# Regime-Switching Dynamics and Tail Dependence
## A Multi-Stage Statistical Framework for Gold's Safe-Haven Properties

**STAT GR5291 — Capstone Project, Columbia University**  
Jingming Peng · Yucheng Gao · Jiesen Chen · Chengyun Zhao · Qiahao Zou  
*May 2026*

---

## Overview

This project investigates whether gold's safe-haven properties are **conditionally activated by market regimes** rather than being unconditional. Using daily data from 2006 to 2026, we implement a three-stage statistical pipeline to examine the state-dependent relationship between gold and U.S. equities across multiple historical stress episodes.

**Core research questions:**
1. Are gold's safe-haven properties activated conditionally on market regimes?
2. Can a data-driven regime classifier identify market stress onset earlier and with fewer false positives than conventional VIX threshold rules (VIX > 30)?

**Key finding:** The GARCH-HMM switching strategy achieves a **6.3× terminal return** over the out-of-sample period (2011–2026), versus 4.5× for S&P 500 buy-and-hold and 3.0× for the VIX < 30 rule.

---

## Methodology

The framework consists of three sequential stages:

### Stage 1 — GARCH-X Baseline
An extended GARCH(1,1) model with an exogenous equity-stress variable tests whether S&P 500 turbulence spills over into gold's conditional variance. Both symmetric and asymmetric specifications are estimated. Key result: the spillover coefficient γ̂ = 0.380 (p ≈ 0) is symmetric across up- and down-moves, indicating gold is **not** volatility-insulated from systemic shocks.

### Stage 2 — Quantile Regression
Linear quantile regression across τ ∈ {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95} characterizes gold's distributional heterogeneity. Key result: gold's S&P 500 sensitivity compresses from 0.027 at the median to 0.010 at the lower tail (τ = 0.05), while the VIX coefficient reverses sign from −0.043 to +0.056 — safe-haven activation is a **tail phenomenon**. QR-VaR achieves a 5.03% exceedance rate versus 5.40% for GARCH-X VaR.

### Stage 3 — GARCH-HMM Regime Detection
A two-state Gaussian Hidden Markov Model is fit to a six-dimensional feature vector derived from EGARCH conditional volatilities, the gold/copper ratio, TIPS yields, and log returns. The Baum–Welch EM algorithm estimates parameters; the Viterbi algorithm decodes the most probable state sequence. A walk-forward out-of-sample scheme (5-year training window, 22-day forecast horizon) eliminates look-ahead bias.

**HMM feature discriminative scores:**

| Feature | Description | Score |
|---|---|---|
| `sp500_egarch_vol` | S&P 500 EGARCH conditional volatility | 0.7742 |
| `gold_egarch_vol` | Gold EGARCH conditional volatility | 0.7458 |
| `diff_DFII10` | 10-yr TIPS yield (first difference) | 0.1128 |
| `d_gold_copper` | Gold/copper ratio (first difference) | 0.0872 |
| `sp500_log_ret` | S&P 500 log return | 0.0656 |
| `gold_log_ret` | Gold log return | 0.0019 |

---

## Data

| Source | Series | Identifier |
|---|---|---|
| yfinance | Gold futures (front-month) | `GC=F` |
| yfinance | Silver futures | `SI=F` |
| yfinance | Platinum futures | `PL=F` |
| yfinance | Copper futures | `HG=F` |
| yfinance | Palladium futures | `PA=F` |
| yfinance | S&P 500 | `^GSPC` |
| yfinance | VIX | `^VIX` |
| FRED | 10-year TIPS yield | `DFII10` |
| FRED | 10-year breakeven inflation | `T10YIE` |

**Sample period:** February 6, 2006 – February 6, 2026 (≈ 5,032 trading days)

**Stress episodes covered:** 2008 Global Financial Crisis · 2011 Eurozone Crisis · 2020 COVID-19 shock · 2022 Fed tightening / geopolitical cycle

A FRED API key is required to download macroeconomic series. Store it in a `.env` file (see [Setup](#setup)).

---

## Project Structure

```
Capstone/
├── STAT5291 Capstone Code.ipynb   # Main analysis notebook
├── data_pipeline.py               # Data download and preprocessing
├── utility_func.py                # GARCH-X, QR, HMM model helpers
├── settings.py                    # Global date range configuration
├── requirements.txt               # Python dependencies
├── .env                           # FRED API key (gitignored)
│
├── data/
│   ├── data_raw/                  # Raw downloaded price series
│   │   ├── sp500.csv
│   │   └── vix.csv
│   ├── data_processed/            # Cleaned and engineered features
│   │   ├── data.csv               # Full merged panel
│   │   ├── LogRet_data.csv        # Log return series
│   │   ├── HMM_features.csv       # EGARCH + macro features for HMM
│   │   └── all_commodities_data1.csv
│   ├── global_market_stress_events.csv  # Historical stress episode labels
│   └── panic_days.csv
│
└── figure/                        # Output plots
    ├── sp500_sample.png
    ├── vix_hline_30.png
    ├── precious_metals_sp500_regime_scatter.png
    ├── qq_plot_by_regime_metal.png
    ├── rolling_corr_stress_events.png
    └── drawdown_stress_events.png
```

---

## Setup

**Requirements:** Python 3.10+

1. **Clone / download** the repository.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the FRED API key.** Create a `.env` file in the project root:
   ```
   FRED_API_KEY=your_key_here
   ```
   A free API key is available at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).

4. **(Optional) Adjust the sample period** in `settings.py`:
   ```python
   DATA_RANGE_START = date(2006, 2, 6)
   DATA_RANGE_END   = date(2026, 2, 6)  # Set to None to use today's date
   ```

5. **Run the data pipeline** to download and preprocess all series:
   ```bash
   python data_pipeline.py
   ```

6. **Open the main notebook:**
   ```bash
   jupyter notebook "STAT5291 Capstone Code.ipynb"
   ```
   Run all cells sequentially to reproduce the full analysis.

---

## Results Summary

| Model | Key Metric | Value |
|---|---|---|
| GARCH-X (symmetric) | Spillover coefficient γ̂ | 0.380 (p ≈ 0) |
| GARCH-X (asymmetric) | Downside γ̂₂ | 0.364 (p ≈ 0) |
| QR (τ = 0.05) | S&P 500 sensitivity β̂ | 0.010 |
| QR (τ = 0.50) | S&P 500 sensitivity β̂ | 0.027 |
| QR-VaR | Exceedance rate (target 5%) | 5.03% |
| GARCH-X VaR | Exceedance rate (target 5%) | 5.40% |
| HMM (State 0, calm) | Expected duration | ~61 trading days |
| HMM (State 1, stress) | Expected duration | ~30 trading days |
| HMM OOS strategy | Terminal value (2011–2026) | 6.3× |
| S&P 500 buy-and-hold | Terminal value (2011–2026) | 4.5× |
| VIX < 30 strategy | Terminal value (2011–2026) | 3.0× |

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy`, `pandas`, `scipy` | Data manipulation and statistics |
| `arch` | GARCH / EGARCH volatility models |
| `hmmlearn` | Hidden Markov Model (Baum–Welch, Viterbi) |
| `scikit-learn` | Preprocessing (Z-score standardization) |
| `statsmodels` | Quantile regression |
| `yfinance` | Market data download |
| `python-dotenv` | `.env` file loading (FRED API key) |
| `matplotlib`, `seaborn` | Visualization |
| `jupyter`, `ipykernel` | Notebook environment |

---

## References

- Baur, D. G. & McDermott, T. K. (2010). Is gold a safe haven? *Journal of Banking & Finance*, 34(8).
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series. *Econometrica*, 57(2).
- Engle, R. F. (1982). Autoregressive conditional heteroskedasticity. *Econometrica*, 50(4).
- Koenker, R. & Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1).
- Baum, L. E. et al. (1970). A maximization technique for Markov chains. *Annals of Mathematical Statistics*, 41(1).
- Viterbi, A. J. (1967). Error bounds for convolutional codes. *IEEE Transactions on Information Theory*, 13(2).
