import json
import pandas as pd
import urllib.parse
import urllib.request
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip("'").strip('"')

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
    out[series_id] = pd.to_numeric(out["value"], errors="coerce")  # FRED 用 "." 表示缺失
    return out[["date", series_id]].sort_values("date")

def plot_time_series(df, date_col='Date', value_col='Close', 
                     title='Price Time Series', 
                     ylabel='Close Price (USD)',
                     figsize=(12, 6),
                     hline=None,
                     hline_color='red',
                     hline_style='--',
                     hline_label=None,
                     save_path=None):
    """
    绘制时间序列数据图表
    
    参数:
        df: DataFrame, 包含日期和数值两列
        date_col: str, 日期列名，默认 'Date'
        value_col: str, 数值列名，默认 'Close'
        title: str, 图表标题
        ylabel: str, y轴标签
        figsize: tuple, 图表尺寸，默认 (12, 6)
        hline: float, 水平参考线的y值，默认 None（不绘制）
        hline_color: str, 参考线颜色，默认 'red'
        hline_style: str, 参考线样式，默认 '--'（虚线）
        hline_label: str, 参考线标签，默认 None
        save_path: str | Path | None, 保存路径，默认 None（不保存）
    """
    # 复制数据避免修改原始df
    data = df.copy()
    
    # 确保日期格式正确
    # data[date_col] = pd.to_datetime(data[date_col])
    # data = data.sort_values(date_col)
    
    # 创建图表
    plt.figure(figsize=figsize)
    sns.lineplot(data=data, x=date_col, y=value_col)
    
    # 添加水平参考线
    if hline is not None:
        label = hline_label if hline_label else f'Reference: {hline}'
        plt.axhline(y=hline, color=hline_color, linestyle=hline_style, 
                   linewidth=2, label=label, alpha=0.7)
        plt.legend()
    
    # 设置 x 轴格式为年份
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    
    # 设置标签和标题
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 使用示例
# plot_time_series(df)  # 不显示参考线
# plot_time_series(df, hline=50)  # 在y=50处显示红色虚线
# plot_time_series(df, hline=30, hline_color='blue', hline_label='Threshold')