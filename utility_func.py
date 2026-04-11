"""
Shared helpers for the capstone notebook: FRED, plots, descriptive stats, HMM workflow.
"""
from __future__ import annotations

import json
import operator
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import urllib.parse
import urllib.request
from dotenv import load_dotenv
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip("'").strip('"')

ROOT: Path = Path(__file__).resolve().parent
DATA_RAW_DIR: Path = ROOT / "data" / "data_raw"
DATA_PROCESSED_DIR: Path = ROOT / "data" / "data_processed"

# --- FRED -----------------------------------------------------------------

def fetch_fred_series(series_id: str, observation_start: str, observation_end: str) -> pd.DataFrame:
    """FRED series/observations API: https://fred.stlouisfed.org/docs/api/fred/series_observations.html"""
    params = urllib.parse.urlencode(
        {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": observation_start,
            "observation_end": observation_end,
        }
    )
    url = f"https://api.stlouisfed.org/fred/series/observations?{params}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = json.loads(resp.read().decode())
    obs = payload.get("observations", [])
    out = pd.DataFrame(obs)
    if out.empty:
        return pd.DataFrame(columns=["date", series_id])
    out["date"] = pd.to_datetime(out["date"])
    out[series_id] = pd.to_numeric(out["value"], errors="coerce")
    return out[["date", series_id]].sort_values("date")


# --- Time series plot -----------------------------------------------------

def plot_time_series(
    df,
    date_col="Date",
    value_col="Close",
    title="Price Time Series",
    ylabel="Close Price (USD)",
    figsize=(12, 6),
    hline=None,
    hline_color="red",
    hline_style="--",
    hline_label=None,
    save_path=None,
):
    """
    Line plot of a value over dates. If date_col is a pandas Index (e.g. caller passed df.index),
    the frame is reset so the index becomes a column for seaborn.
    """
    data = df.copy()
    if isinstance(date_col, pd.Index) or (
        hasattr(date_col, "equals") and hasattr(df, "index") and date_col.equals(df.index)
    ):
        data = data.reset_index()
        date_col = [c for c in data.columns if c != value_col][0]

    plt.figure(figsize=figsize)
    sns.lineplot(data=data, x=date_col, y=value_col)

    if hline is not None:
        label = hline_label if hline_label else f"Reference: {hline}"
        plt.axhline(
            y=hline,
            color=hline_color,
            linestyle=hline_style,
            linewidth=2,
            label=label,
            alpha=0.7,
        )
        plt.legend()

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# --- Descriptive statistics -----------------------------------------------

def get_stats(data_subset: pd.DataFrame, asset_name: str, regime_name: str) -> dict:
    series = data_subset[asset_name]
    return {
        "Asset": asset_name,
        "Regime": regime_name,
        "Days": len(series),
        "Return": series.mean() * 252,
        "Volatility": series.std() * np.sqrt(252),
        "Skewness": series.skew(),
        "Kurtosis": series.kurt(),
    }


def descriptive_stats_by_regime(
    ret_data: pd.DataFrame,
    asset_list: list[str],
    *,
    panic_col: str = "is_panic_day",
) -> pd.DataFrame:
    """Build multi-index table: Normal / Extreme / Overall per asset (panic flag splits regimes)."""
    results = []
    for asset in asset_list:
        results.append(
            get_stats(ret_data[ret_data[panic_col] == 0], asset, "Normal(VIX < 30)")
        )
        results.append(
            get_stats(ret_data[ret_data[panic_col] == 1], asset, "Extreme(VIX > 30)")
        )
        results.append(get_stats(ret_data, asset, "Overall"))
    stat_df = pd.DataFrame(results)
    stat_df = stat_df[
        ["Asset", "Regime", "Days", "Return", "Volatility", "Skewness", "Kurtosis"]
    ].round(4)
    return stat_df.set_index(["Asset", "Regime"])


# --- EDA figure grids ------------------------------------------------------

def plot_regime_scatter_grid(
    ret_data: pd.DataFrame,
    asset_list: list[str],
    figure_dir: str | Path,
    *,
    panic_col: str = "is_panic_day",
    filename: str = "precious_metals_sp500_regime_scatter.png",
) -> None:
    figure_dir = Path(figure_dir)
    n = len(asset_list)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 8), sharex=True, sharey=True)
    fig.suptitle(
        "Precious Metals vs. S&P 500: Log Daily Return by VIX Regime",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    regimes = [0, 1, "Overall"]
    regime_titles = ["Normal (VIX < 30)", "Extreme (VIX > 30)", "Overall Market"]

    for i, regime in enumerate(regimes):
        for j, metal in enumerate(asset_list):
            ax = axes[i, j]
            if regime == "Overall":
                plot_data = ret_data
            else:
                plot_data = ret_data[ret_data[panic_col] == regime]
            sns.regplot(
                data=plot_data,
                x="sp500",
                y=metal,
                ax=ax,
                scatter_kws={"alpha": 0.5, "s": 10},
                line_kws={"color": "red", "lw": 1, "linestyle": "--"},
            )
            corr = plot_data["sp500"].corr(plot_data[metal])
            ax.text(
                0.05,
                0.9,
                f"Corr: {corr:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.8),
            )
            if i == 0:
                ax.set_title(f"{metal}", fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel(
                    f"{regime_titles[i]}\n Asset Returns", fontsize=10, fontweight="bold"
                )
            else:
                ax.set_ylabel("")
            if i == 2:
                ax.set_xlabel("S&P 500 Returns", fontsize=10)
            else:
                ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(figure_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_qq_by_regime(
    ret_data: pd.DataFrame,
    asset_list: list[str],
    figure_dir: str | Path,
    *,
    layout: str = "regime_rows",
    panic_col: str = "is_panic_day",
) -> None:
    """
    layout='metal_rows': n x 3 grid (one row per asset).
    layout='regime_rows': 3 x n grid (one row per regime), matches original notebook fig 19.
    """
    figure_dir = Path(figure_dir)
    n = len(asset_list)
    regimes = [0, 1, "Overall"]
    regime_titles = ["Normal (VIX < 30)", "Extreme (VIX > 30)", "Overall Market"]

    if layout == "metal_rows":
        fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
        fname = "qq_plot_by_metal_regime.png"
        for i, metal in enumerate(asset_list):
            for j, regime in enumerate(regimes):
                ax = axes[i, j]
                if regime == "Overall":
                    s = ret_data[metal].dropna()
                else:
                    s = ret_data[ret_data[panic_col] == regime][metal].dropna()
                stats.probplot(s, dist="norm", plot=ax)
                ax.get_lines()[0].set_markerfacecolor("C0")
                ax.get_lines()[0].set_alpha(0.4)
                ax.get_lines()[0].set_markersize(4)
                ax.get_lines()[1].set_color("red")
                ax.get_lines()[1].set_linewidth(1)
                ax.get_lines()[1].set_linestyle("--")
                ax.set_title("")
                skew, kurt = s.skew(), s.kurt()
                ax.text(
                    0.05,
                    0.85,
                    f"Skew: {skew:.2f}\nKurt: {kurt:.2f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.7),
                )
                if i == 0:
                    ax.set_title(regime_titles[j], fontsize=10, fontweight="bold")
                if j == 0:
                    ax.set_ylabel(f"{metal} Quantiles", fontsize=10, fontweight="bold")
                else:
                    ax.set_ylabel("")
                if i == n - 1:
                    ax.set_xlabel("Theoretical Quantiles", fontsize=10)
                else:
                    ax.set_xlabel("")
    elif layout == "regime_rows":
        fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 8))
        fname = "qq_plot_by_regime_metal.png"
        for i, regime in enumerate(regimes):
            for j, metal in enumerate(asset_list):
                ax = axes[i, j]
                if regime == "Overall":
                    s = ret_data[metal].dropna()
                else:
                    s = ret_data[ret_data[panic_col] == regime][metal].dropna()
                stats.probplot(s, dist="norm", plot=ax)
                ax.get_lines()[0].set_markerfacecolor("C0")
                ax.get_lines()[0].set_alpha(0.4)
                ax.get_lines()[0].set_markersize(4)
                ax.get_lines()[1].set_color("red")
                ax.get_lines()[1].set_linewidth(1)
                ax.get_lines()[1].set_linestyle("--")
                ax.set_title("")
                skew, kurt = s.skew(), s.kurt()
                ax.text(
                    0.05,
                    0.85,
                    f"Skew: {skew:.2f}\nKurt: {kurt:.2f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.7),
                )
                if i == 0:
                    ax.set_title(f"{metal}", fontsize=10, fontweight="bold")
                if j == 0:
                    ax.set_ylabel(
                        f"{regime_titles[i]}\nQuantiles", fontsize=10, fontweight="bold"
                    )
                else:
                    ax.set_ylabel("")
                if i == 2:
                    ax.set_xlabel("Theoretical Quantiles", fontsize=10)
                else:
                    ax.set_xlabel("")
    else:
        raise ValueError("layout must be 'metal_rows' or 'regime_rows'")

    fig.suptitle(
        "QQ Plots of Precious Metals and S&P 500 Log Daily Returns by VIX Regime",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(figure_dir / fname, dpi=300, bbox_inches="tight")
    plt.show()


def plot_rolling_corr_stress_events(
    ret_data: pd.DataFrame,
    events_df: pd.DataFrame,
    figure_dir: str | Path,
    *,
    window: int = 30,
    metals: tuple[str, ...] = ("Gold", "Silver", "Copper"),
    max_events: int = 9,
    filename: str = "rolling_corr_stress_events.png",
) -> None:
    figure_dir = Path(figure_dir)
    ev = events_df.copy()
    ev["Start Date"] = pd.to_datetime(ev["Start Date"], dayfirst=True)
    ev["End Date"] = pd.to_datetime(ev["End Date"], dayfirst=True)
    colors = {"Gold": "gold", "Silver": "silver", "Copper": "chocolate"}

    fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharey=True)
    axes = axes.flatten()
    last_i = 0
    for i, (_, row) in enumerate(ev.iterrows()):
        if i >= max_events:
            break
        last_i = i
        ax = axes[i]
        start, end = row["Start Date"], row["End Date"]
        plot_start = start - pd.Timedelta(days=30)
        plot_end = end + pd.Timedelta(days=30)
        mask = (ret_data.index >= plot_start) & (ret_data.index <= plot_end)
        subset = ret_data.loc[mask]
        for metal in metals:
            roll_corr = subset[metal].rolling(window=window).corr(subset["sp500"])
            ax.plot(roll_corr, label=f"{metal}", color=colors[metal], lw=2)
        ax.axvspan(start, end, color="red", alpha=0.1, label="Crisis Period")
        ax.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_title(f"{row['Abbreviation']}: {row['Event Name']}", fontsize=11, fontweight="bold")
        ax.set_ylim(-1, 1)
        if i % 3 == 0:
            ax.set_ylabel("Correlation with S&P 500")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    for j in range(last_i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(
        f"Rolling Correlation ({window}-Day) during Global Market Stress Events",
        fontsize=20,
        fontweight="bold",
        y=1.02,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(figure_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_drawdown_stress_events(
    ret_data: pd.DataFrame,
    events_df: pd.DataFrame,
    figure_dir: str | Path,
    *,
    assets_to_compare: tuple[str, ...] = ("Gold", "Silver", "Copper", "sp500"),
    max_events: int = 9,
    filename: str = "drawdown_stress_events.png",
) -> None:
    figure_dir = Path(figure_dir)
    ev = events_df.copy()
    ev["Start Date"] = pd.to_datetime(ev["Start Date"], dayfirst=True)
    ev["End Date"] = pd.to_datetime(ev["End Date"], dayfirst=True)
    colors = {
        "Gold": "#FFD700",
        "Silver": "#A9A9A9",
        "Copper": "#B87333",
        "sp500": "#1f77b4",
    }

    fig, axes = plt.subplots(3, 3, figsize=(20, 16), sharey=True)
    axes = axes.flatten()
    last_i = 0
    for i, (_, row) in enumerate(ev.iterrows()):
        if i >= max_events:
            break
        last_i = i
        ax = axes[i]
        start, end = row["Start Date"], row["End Date"]
        mask = (ret_data.index >= start) & (ret_data.index <= end)
        subset = ret_data.loc[mask].copy()
        if subset.empty:
            continue
        for asset in assets_to_compare:
            cum_ret = (1 + subset[asset]).cumprod()
            running_max = cum_ret.cummax()
            drawdown = (cum_ret / running_max) - 1
            ax.plot(drawdown, label=asset, color=colors[asset], lw=2, zorder=3)
            if asset in ("Gold", "sp500"):
                ax.fill_between(drawdown.index, drawdown, 0, color=colors[asset], alpha=0.05)
        ax.grid(True, which="major", linestyle="--", color="gray", alpha=0.5, zorder=1)
        ax.set_title(
            f"{row['Abbreviation']}\n{row['Event Name']}", fontsize=12, fontweight="bold"
        )
        ax.set_ylim(-0.65, 0.05)
        ax.axhline(0, color="black", lw=1.5, alpha=0.7, zorder=2)
        if i % 3 == 0:
            ax.set_ylabel("Drawdown (%)", fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    for j in range(last_i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(
        "Maximum Drawdown Analysis: The Resilience of Precious Metals vs Risk Assets",
        fontsize=22,
        fontweight="bold",
        y=1.02,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=14
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(figure_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()


# --- HMM ------------------------------------------------------------------

DEFAULT_HMM_FEATURES: tuple[str, ...] = (
    "gold_log_ret",
    "sp500_log_ret",
    "diff_DFII10",
    "d_gold_copper",
)


def prepare_hmm_design_matrix(
    hmm_features: str | Path | pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, StandardScaler, list[str]]:
    """Load HMM_features (path or frame), engineer diffs, dropna, return df_model, X, scaler, feature names."""
    if isinstance(hmm_features, (str, Path)):
        df = pd.read_csv(Path(hmm_features))
    else:
        df = hmm_features.copy()
    df["diff_DFII10"] = df["DFII10"].diff()
    df["diff_T10YIE"] = df["T10YIE"].diff()
    df["d_gold_oil"] = df["gold_oil_ratio"].pct_change(fill_method=None)
    df["d_gold_copper"] = df["gold_copper_ratio"].pct_change(fill_method=None)
    features = list(DEFAULT_HMM_FEATURES)
    df_model = df.dropna(subset=features)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_model[features])
    return df_model, X, scaler, features


def fit_gaussian_hmm_assign_states(
    df_model: pd.DataFrame,
    X: np.ndarray,
    features: list[str],
    scaler: StandardScaler,
    *,
    n_components: int = 2,
    covariance_type: str = "full",
    n_iter: int = 1000,
    random_state: int = 42,
) -> tuple[GaussianHMM, pd.DataFrame]:
    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X)
    hidden_states = model.predict(X)
    out = df_model.copy()
    out["state"] = hidden_states
    return model, out


def print_hmm_training_summaries(
    model: GaussianHMM,
    scaler: StandardScaler,
    features: list[str],
) -> None:
    state_means = pd.DataFrame(
        scaler.inverse_transform(model.means_), columns=features
    )
    print("State conditional means:\n", state_means)
    for i in range(model.n_components):
        print(f"State {i} covariance:\n", model.covars_[i])


def plot_hmm_gold_states_scatter(
    df_model: pd.DataFrame,
    model: GaussianHMM,
    *,
    title: str = "Gold Market Regime Identification via HMM (2006-2026)",
) -> None:
    dm = df_model.copy()
    dm["Date"] = pd.to_datetime(dm["Date"])
    plt.figure(figsize=(15, 5))
    for i in range(model.n_components):
        state_data = dm[dm["state"] == i]
        plt.scatter(
            state_data["Date"],
            state_data["gold_log_ret"],
            label=f"State {i}",
            s=10,
        )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel(r"Daily Log-Returns ($r_{gold,t}$)", fontsize=12)
    plt.xlabel("Date / Market Timeline", fontsize=12)
    plt.legend(loc="upper right")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def discriminative_feature_scores(model: GaussianHMM, features: list[str]) -> list[tuple[str, float]]:
    means = model.means_
    covars = model.covars_
    scores: dict[str, float] = {}
    for i, feature in enumerate(features):
        m0, m1 = means[0, i], means[1, i]
        s0, s1 = np.sqrt(covars[0, i, i]), np.sqrt(covars[1, i, i])
        scores[feature] = abs(m1 - m0) / (s0 + s1)
    return sorted(scores.items(), key=operator.itemgetter(1), reverse=True)


def print_discriminative_feature_scores(model: GaussianHMM, features: list[str]) -> None:
    for f, s in discriminative_feature_scores(model, features):
        print(f"Feature {f} discriminative score: {s:.4f}")


def walk_forward_hmm_oos(
    X: np.ndarray,
    df_model: pd.DataFrame,
    *,
    n_train: int = 252 * 5,
    step: int = 22,
    n_components: int = 2,
    covariance_type: str = "full",
    n_iter: int = 1000,
    random_state: int | None = 0,
) -> pd.DataFrame:
    n_total = len(X)
    oos_states: list[int] = []
    oos_dates: list = []
    for i in range(n_train, n_total, step):
        X_train = X[:i]
        test_end = min(i + step, n_total)
        X_test = X[i:test_end]
        kw: dict = dict(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
        )
        if random_state is not None:
            kw["random_state"] = random_state
        wf_model = GaussianHMM(**kw)
        wf_model.fit(X_train)
        current_oos_states = wf_model.predict(X_test)
        oos_states.extend(current_oos_states)
        oos_dates.extend(df_model.iloc[i:test_end]["Date"].tolist())
    return pd.DataFrame({"Date": oos_dates, "OOS_State": oos_states})


def merge_oos_with_returns(df_oos: pd.DataFrame, df_model: pd.DataFrame) -> pd.DataFrame:
    return df_oos.merge(
        df_model[["Date", "gold_log_ret", "sp500_log_ret"]],
        on="Date",
    )


def print_oos_summary(df_oos: pd.DataFrame) -> None:
    oos_stats = df_oos.groupby("OOS_State").agg(
        {"sp500_log_ret": ["mean", "std"], "gold_log_ret": ["mean", "std"]}
    )
    print(oos_stats)
    for state in [0, 1]:
        corr = (
            df_oos[df_oos["OOS_State"] == state][["gold_log_ret", "sp500_log_ret"]]
            .corr()
            .iloc[0, 1]
        )
        print(f"OOS state {state} correlation: {corr:.4f}")


def add_regime_strategy_returns(df_oos: pd.DataFrame) -> pd.DataFrame:
    out = df_oos.copy()
    out["strategy_ret"] = np.where(
        out["OOS_State"] == 0,
        out["sp500_log_ret"],
        out["gold_log_ret"],
    )
    out["cum_strategy"] = (1 + out["strategy_ret"]).cumprod()
    out["cum_sp500"] = (1 + out["sp500_log_ret"]).cumprod()
    return out


def plot_strategy_vs_buyhold(
    df_oos: pd.DataFrame,
    *,
    title: str = "OOS Strategy Performance vs S&P 500 Buy-and-Hold",
    figsize: tuple[float, float] = (12, 6),
) -> None:
    d = df_oos.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.set_index("Date")
    d[["cum_strategy", "cum_sp500"]].plot(figsize=figsize)
    plt.title(title)
    plt.tight_layout()
    plt.show()
