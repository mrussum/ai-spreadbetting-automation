"""Tests for the feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from data.features import FEATURE_COLUMNS, engineer_features


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 7000 + np.cumsum(np.random.randn(n) * 20)
    high = close + np.abs(np.random.randn(n) * 10)
    low = close - np.abs(np.random.randn(n) * 10)
    open_ = close + np.random.randn(n) * 5
    volume = np.random.randint(1_000_000, 10_000_000, size=n).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestEngineerFeatures:
    def test_returns_dataframe(self):
        df = engineer_features(_make_ohlcv())
        assert isinstance(df, pd.DataFrame)

    def test_all_feature_columns_present(self):
        df = engineer_features(_make_ohlcv())
        for col in FEATURE_COLUMNS:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_no_nan_values_in_features(self):
        df = engineer_features(_make_ohlcv())
        for col in FEATURE_COLUMNS:
            assert df[col].isna().sum() == 0, f"NaN found in {col}"

    def test_target_column_when_requested(self):
        df = engineer_features(_make_ohlcv(), include_target=True)
        assert "target_3" in df.columns
        assert set(df["target_3"].unique()).issubset({0, 1})

    def test_no_target_by_default(self):
        df = engineer_features(_make_ohlcv())
        assert "target_3" not in df.columns

    def test_rsi_in_valid_range(self):
        df = engineer_features(_make_ohlcv())
        assert df["rsi_14"].between(0, 100).all()

    def test_volume_ratio_positive(self):
        df = engineer_features(_make_ohlcv())
        assert (df["volume_ratio"] > 0).all()

    def test_fewer_rows_than_input(self):
        raw = _make_ohlcv()
        df = engineer_features(raw)
        assert len(df) < len(raw), "Should drop warm-up NaN rows"
        assert len(df) > 0
