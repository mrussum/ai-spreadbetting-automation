"""Historical data download and caching via yfinance."""

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from sqlalchemy import text

from config.settings import EPIC_TO_TICKER
from database.models import engine

logger = logging.getLogger(__name__)


def _table_name(ticker: str) -> str:
    """Sanitized table name for a ticker."""
    return "ohlcv_" + ticker.replace("^", "").replace("=", "_").replace(".", "_").lower()


def fetch_historical(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    force: bool = False,
) -> pd.DataFrame:
    """Download OHLCV data via yfinance and cache to SQLite.

    Args:
        ticker: yfinance ticker symbol (e.g. "^FTSE")
        period: yfinance period string (e.g. "2y", "5y")
        interval: candle interval (e.g. "1d", "1h")
        force: if True, re-download even if cached data exists

    Returns:
        Clean DataFrame with columns: open, high, low, close, volume
    """
    table = _table_name(ticker)

    if not force:
        try:
            cached = pd.read_sql(f"SELECT * FROM {table}", engine, parse_dates=["date"])
            if not cached.empty:
                last_date = cached["date"].max()
                if (datetime.utcnow() - last_date) < timedelta(days=2):
                    logger.info("Using cached data for %s (%d rows)", ticker, len(cached))
                    cached.set_index("date", inplace=True)
                    return cached
        except Exception:
            pass  # Table doesn't exist yet

    logger.info("Downloading %s data (period=%s, interval=%s)", ticker, period, interval)
    data = yf.download(ticker, period=period, interval=interval, progress=False)

    if data.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Clean column names
    data.columns = [c.lower().strip() for c in data.columns]
    data.index.name = "date"

    # Keep only OHLCV
    expected = ["open", "high", "low", "close", "volume"]
    data = data[[c for c in expected if c in data.columns]]

    # Drop NaN rows
    data.dropna(inplace=True)

    # Cache to SQLite
    data.to_sql(table, engine, if_exists="replace", index=True)
    logger.info("Cached %d rows for %s", len(data), ticker)

    return data


def fetch_latest_candles(ticker: str, n: int = 50) -> pd.DataFrame:
    """Get the most recent N candles for live inference.

    First tries cache, falls back to fresh download.
    """
    try:
        table = _table_name(ticker)
        query = f"SELECT * FROM {table} ORDER BY date DESC LIMIT {n}"
        df = pd.read_sql(query, engine, parse_dates=["date"])
        if not df.empty:
            df.set_index("date", inplace=True)
            return df.sort_index()
    except Exception:
        pass

    # Fallback: download a short period
    data = fetch_historical(ticker, period="3mo", interval="1d")
    return data.tail(n)


def fetch_all_instruments() -> dict[str, pd.DataFrame]:
    """Download historical data for all configured instruments.

    Returns:
        Dict mapping ticker to DataFrame.
    """
    results = {}
    for epic, ticker in EPIC_TO_TICKER.items():
        try:
            df = fetch_historical(ticker)
            results[ticker] = df
            logger.info("Fetched %s → %d rows", ticker, len(df))
        except Exception as e:
            logger.error("Failed to fetch %s: %s", ticker, e)
    return results
