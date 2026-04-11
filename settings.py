"""
Project-wide date range for `data_pipeline` downloads (yfinance / merged panel).

Edit DATA_RANGE_START and DATA_RANGE_END here; the pipeline reads only these values.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

# First calendar day of the sample (inclusive for daily bars).
DATA_RANGE_START: date = date(2006, 4, 7)

# Last calendar day of the sample (inclusive). Use None to use the run time as the end.
DATA_RANGE_END: date = date(2026, 4, 7)


def get_download_start() -> datetime:
    """Inclusive start as timezone-naive datetime (midnight)."""
    return datetime.combine(DATA_RANGE_START, datetime.min.time())


def get_yfinance_end_exclusive() -> datetime:
    """
    yfinance daily `end` is exclusive — use the day after DATA_RANGE_END so that
    the last trading day on DATA_RANGE_END is included. When DATA_RANGE_END is
    None, use the current moment (same behavior as before).
    """
    if DATA_RANGE_END is None:
        return datetime.now()
    return datetime.combine(DATA_RANGE_END + timedelta(days=1), datetime.min.time())
