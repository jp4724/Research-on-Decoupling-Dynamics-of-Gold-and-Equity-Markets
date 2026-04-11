"""
Refresh raw downloads and processed tables. Processed CSVs are written only under
data/data_processed/; raw downloads under data/data_raw/. See README.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from settings import DATA_RANGE_END, DATA_RANGE_START, get_download_start, get_yfinance_end_exclusive
from utility_func import DATA_PROCESSED_DIR, DATA_RAW_DIR, ROOT, fetch_fred_series

__all__ = [
    "ROOT",
    "DATA_RAW_DIR",
    "DATA_PROCESSED_DIR",
    "download_commodities",
    "download_sp500_vix",
    "build_data_csv",
    "build_logret_csv",
    "build_hmm_feature_csv",
    "run_all",
]

COMMODITIES_PROCESSED_CSV = "all_commodities_data1.csv"
SP500_RAW_CSV = "sp500.csv"
VIX_RAW_CSV = "vix.csv"
DATA_CSV = "data.csv"
LOGRET_CSV = "LogRet_data.csv"
HMM_FEATURES_CSV = "HMM_features.csv"


def _ensure_dirs() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance frames that use a MultiIndex column level."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    out = df.copy()
    out.columns = out.columns.get_level_values(0)
    return out


def _panic_days_csv_path() -> Path:
    p_data = ROOT / "data" / "panic_days.csv"
    p_raw = DATA_RAW_DIR / "panic_days.csv"
    if p_data.is_file():
        return p_data
    if p_raw.is_file():
        return p_raw
    raise FileNotFoundError(
        "panic_days.csv not found. Expected data/panic_days.csv or "
        f"{DATA_RAW_DIR / 'panic_days.csv'}"
    )


def download_commodities() -> Path:
    """
    Download extended commodity daily history via yfinance and write
    data/data_processed/all_commodities_data1.csv.

    Date bounds come from `settings.py` (DATA_RANGE_START / DATA_RANGE_END).
    """
    _ensure_dirs()
    start_date = get_download_start()
    end_exclusive = get_yfinance_end_exclusive()

    ticker_list = ["GC=F", "SI=F", "PL=F", "HG=F", "PA=F"]
    comm_list = ["Gold", "Silver", "Platinum", "Copper", "Palladium"]

    def fetch(ticker_name: str, comm_name: str) -> pd.DataFrame:
        ticker = yf.Ticker(ticker_name)
        df1 = ticker.history(start=start_date, end=end_exclusive)
        if df1.empty:
            raise RuntimeError(f"No yfinance history for {ticker_name}")
        df1 = df1.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
        df1 = df1.reset_index()
        df1["Date"] = df1["Date"].map(lambda x: x.strftime("%Y/%m/%d"))
        df1 = df1.drop_duplicates(subset=["Date"], keep="first")
        df1["Ticker"] = ticker_name
        df1["Commodity"] = comm_name
        return df1

    parts: list[pd.DataFrame] = []
    for ticker, commodity in zip(ticker_list, comm_list):
        parts.append(fetch(ticker, commodity))
    bdf = pd.concat(parts, ignore_index=True)

    out = DATA_PROCESSED_DIR / COMMODITIES_PROCESSED_CSV
    bdf.to_csv(out, index=False)
    return out


def download_sp500_vix() -> tuple[Path, Path]:
    """Download ^GSPC and ^VIX closes to data/data_raw/ (bounds from `settings.py`)."""
    _ensure_dirs()
    start_date = get_download_start()
    end_exclusive = get_yfinance_end_exclusive()

    sp500_raw = yf.download("^GSPC", start=start_date, end=end_exclusive, progress=False)
    sp500_raw = _flatten_yf_columns(sp500_raw)
    sp500_raw = sp500_raw.reset_index()
    if sp500_raw.empty:
        raise RuntimeError("No S&P 500 data from yfinance")
    sp500_data = sp500_raw[["Date", "Close"]].copy()

    vix_raw = yf.download("^VIX", start=start_date, end=end_exclusive, progress=False)
    vix_raw = _flatten_yf_columns(vix_raw)
    vix_raw = vix_raw.reset_index()
    if vix_raw.empty:
        raise RuntimeError("No VIX data from yfinance")
    vix_data = vix_raw[["Date", "Close"]].copy()

    p_sp = DATA_RAW_DIR / SP500_RAW_CSV
    p_vx = DATA_RAW_DIR / VIX_RAW_CSV
    sp500_data.to_csv(p_sp, index=False)
    vix_data.to_csv(p_vx, index=False)
    return p_sp, p_vx


def build_data_csv() -> Path:
    """
    Merge processed commodities with raw sp500/vix and panic-day flags;
    writes data/data_processed/data.csv.
    """
    _ensure_dirs()
    all_commodity = pd.read_csv(DATA_PROCESSED_DIR / COMMODITIES_PROCESSED_CSV)
    pt = pd.pivot_table(
        all_commodity, "Close", index="Date", columns="Commodity", aggfunc="sum"
    )
    # Commodity CSV dates from yfinance are typically YYYY/MM/DD; dayfirst=True
    # mis-parses that pattern and raises on some rows.
    pt.index = pd.to_datetime(pt.index, format="mixed")
    pt = pt.sort_index()

    sp500 = pd.read_csv(DATA_RAW_DIR / SP500_RAW_CSV)
    vix = pd.read_csv(DATA_RAW_DIR / VIX_RAW_CSV)
    sp500 = sp500.set_index(sp500["Date"])
    vix = vix.set_index(vix["Date"])
    sp500.index = pd.to_datetime(sp500.index)
    vix.index = pd.to_datetime(vix.index)

    pt["sp500"] = sp500["Close"]
    pt["vix"] = vix["Close"]

    panic_days = pd.read_csv(_panic_days_csv_path())
    panic_day_list = pd.to_datetime(panic_days["Date"], dayfirst=True).tolist()
    pt["is_panic_day"] = pt.index.isin(panic_day_list).astype(int)

    out = DATA_PROCESSED_DIR / DATA_CSV
    pt.to_csv(out)
    return out


def build_logret_csv() -> Path:
    """Read data.csv, compute log returns + vix/panic columns; write LogRet_data.csv."""
    _ensure_dirs()
    data = pd.read_csv(DATA_PROCESSED_DIR / DATA_CSV)
    data = data.set_index("Date")
    col = ["Copper", "Gold", "Palladium", "Platinum", "Silver", "sp500"]
    log_returns = np.log(data[col]).diff()
    ret = pd.concat([log_returns, data[["vix", "is_panic_day"]]], axis=1)
    ret = ret.iloc[1:]
    out = DATA_PROCESSED_DIR / LOGRET_CSV
    ret.reset_index().to_csv(out, index=False)
    return out


def build_hmm_feature_csv() -> Path:
    """
    FRED + yfinance ratios aligned to LogRet trading dates; write HMM_features.csv
    (only under data_processed).
    """
    _ensure_dirs()
    logret_path = DATA_PROCESSED_DIR / LOGRET_CSV
    if not logret_path.is_file():
        raise FileNotFoundError(f"Missing {logret_path}; run logret step first.")

    _logret_dates = pd.read_csv(logret_path, parse_dates=["Date"])
    if _logret_dates.empty:
        raise ValueError(
            f"{logret_path} is empty (no rows). Log returns need at least two "
            f"price dates in {DATA_PROCESSED_DIR / DATA_CSV}; rebuild the panel "
            "CSV then run the logret step again."
        )
    lr_min_ts = _logret_dates["Date"].min()
    lr_max_ts = _logret_dates["Date"].max()
    if pd.isna(lr_min_ts) or pd.isna(lr_max_ts):
        raise ValueError(
            f"{logret_path} has no valid dates in the Date column; fix the file "
            "or regenerate it with build_logret_csv()."
        )
    lr_min = lr_min_ts.date()
    lr_max = lr_max_ts.date()
    end_cap = DATA_RANGE_END if DATA_RANGE_END is not None else date.today()
    start_cap = max(DATA_RANGE_START, lr_min)
    end_cap = min(end_cap, lr_max)
    obs_start = start_cap.strftime("%Y-%m-%d")
    obs_end = end_cap.strftime("%Y-%m-%d")
    yf_end = (end_cap + timedelta(days=1)).strftime("%Y-%m-%d")

    dfii10 = fetch_fred_series("DFII10", obs_start, obs_end)
    t10yie = fetch_fred_series("T10YIE", obs_start, obs_end)

    comm_raw = yf.download(
        ["GC=F", "CL=F", "HG=F"], start=obs_start, end=yf_end, progress=False
    )
    comm_close = comm_raw["Close"] if "Close" in comm_raw.columns else comm_raw
    if isinstance(comm_close, pd.Series):
        comm_close = comm_close.to_frame()
    comm_dat = comm_close.copy()
    comm_dat["gold_oil_ratio"] = comm_dat["GC=F"] / comm_dat["CL=F"]
    comm_dat["gold_copper_ratio"] = comm_dat["GC=F"] / comm_dat["HG=F"]
    comm_dat = comm_dat.reset_index().rename(columns={"Date": "date"})

    features_df = dfii10.merge(t10yie, on="date", how="outer").sort_values("date")
    features_df = features_df.merge(
        comm_dat[["date", "gold_oil_ratio", "gold_copper_ratio"]],
        on="date",
        how="outer",
    ).sort_values("date")

    features_df = (
        _logret_dates[["Date"]]
        .merge(features_df.rename(columns={"date": "Date"}), on="Date", how="left")
        .sort_values("Date")
    )

    gold_log_ret = pd.read_csv(logret_path)
    gold_log_ret = gold_log_ret[["Date", "Gold", "sp500"]]
    gold_log_ret["Date"] = pd.to_datetime(gold_log_ret["Date"])
    features_df = features_df.merge(gold_log_ret, on="Date", how="outer").sort_values(
        "Date"
    )
    features_df.rename(
        columns={"Gold": "gold_log_ret", "sp500": "sp500_log_ret"}, inplace=True
    )

    out = DATA_PROCESSED_DIR / HMM_FEATURES_CSV
    features_df.to_csv(out, index=False)
    return out


def run_all() -> None:
    """Run full pipeline in dependency order (download range from `settings.py`)."""
    download_commodities()
    download_sp500_vix()
    build_data_csv()
    build_logret_csv()
    build_hmm_feature_csv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and build capstone datasets (processed outputs in data/data_processed/)."
    )
    parser.add_argument(
        "--step",
        choices=("all", "commodities", "indices", "panel", "logret", "hmm"),
        default="all",
        help="Pipeline step to run (default: all). Download range: edit settings.py.",
    )
    args = parser.parse_args()
    _ensure_dirs()

    if args.step == "all":
        run_all()
        print("Completed: all steps.")
        return

    step_map = {
        "commodities": download_commodities,
        "indices": download_sp500_vix,
        "panel": build_data_csv,
        "logret": build_logret_csv,
        "hmm": build_hmm_feature_csv,
    }
    step_map[args.step]()
    print(f"Completed: --step {args.step}")


if __name__ == "__main__":
    main()
