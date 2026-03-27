"""Feature engineering pipeline using the `ta` library."""

import logging

import pandas as pd
import ta

from config.settings import EPIC_TO_TICKER
from data.collector import fetch_latest_candles

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame, include_target: bool = False) -> pd.DataFrame:
    """Add technical indicator features to an OHLCV DataFrame.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
        include_target: if True, add target_3 column (for training)

    Returns:
        DataFrame with all original columns plus feature columns.
    """
    df = df.copy()

    # Ensure we have enough data
    if df.empty:
        return df
    if len(df) < 200:
        logger.warning("Only %d rows — some long-window indicators may be NaN", len(df))

    # ─── Trend ───────────────────────────────────────────────
    df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["ema_200"] = ta.trend.ema_indicator(df["close"], window=200)

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # ─── Momentum ────────────────────────────────────────────
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)

    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["roc_10"] = ta.momentum.roc(df["close"], window=10)
    df["williams_r"] = ta.momentum.williams_r(df["high"], df["low"], df["close"])

    # ─── Volatility ──────────────────────────────────────────
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()

    df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

    # ─── Volume ──────────────────────────────────────────────
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

    # ─── Price returns ───────────────────────────────────────
    df["returns_1"] = df["close"].pct_change(1)
    df["returns_5"] = df["close"].pct_change(5)
    df["returns_20"] = df["close"].pct_change(20)

    # ─── Target (for training only) ─────────────────────────
    if include_target:
        df["target_3"] = (df["close"].shift(-3) > df["close"]).astype(int)

    # Drop rows with NaN from indicator warm-up
    df.dropna(inplace=True)

    return df


# Feature columns used for ML (excludes OHLCV and target)
FEATURE_COLUMNS: list[str] = [
    "ema_20", "ema_50", "ema_200",
    "macd", "macd_signal", "macd_hist",
    "rsi_14", "stoch_k", "stoch_d", "roc_10", "williams_r",
    "bb_upper", "bb_lower", "bb_width", "atr_14",
    "obv", "volume_sma_20", "volume_ratio",
    "returns_1", "returns_5", "returns_20",
]


def get_live_features(ticker: str) -> pd.Series:
    """Fetch latest candles, compute features, return the final row.

    This is the entry point for live prediction.
    """
    df = fetch_latest_candles(ticker, n=250)
    df = engineer_features(df, include_target=False)

    if df.empty:
        raise ValueError(f"No feature data available for {ticker}")

    return df[FEATURE_COLUMNS].iloc[-1]
