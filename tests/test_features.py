"""Tests for the feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from data.features import FEATURE_COLUMNS, engineer_features, get_live_features


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

    def test_does_not_mutate_input(self):
        """engineer_features must not modify the caller's DataFrame."""
        raw = _make_ohlcv()
        original_cols = set(raw.columns)
        engineer_features(raw)
        assert set(raw.columns) == original_cols

    def test_ema_200_requires_sufficient_rows(self):
        """With only 50 rows the EMA-200 warm-up strips all rows — result is empty."""
        raw = _make_ohlcv(n=50)
        df = engineer_features(raw)
        # After dropping NaNs from a 200-period indicator there should be nothing left
        assert len(df) == 0

    def test_target_binary_values(self):
        """target_3 must be strictly 0 or 1."""
        df = engineer_features(_make_ohlcv(400), include_target=True)
        assert df["target_3"].isin([0, 1]).all()

    def test_feature_columns_count(self):
        """FEATURE_COLUMNS list length must remain stable."""
        assert len(FEATURE_COLUMNS) == 21

    def test_feature_columns_are_unique(self):
        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))

    def test_bollinger_width_non_negative(self):
        df = engineer_features(_make_ohlcv())
        assert (df["bb_width"] >= 0).all()

    def test_atr_non_negative(self):
        df = engineer_features(_make_ohlcv())
        assert (df["atr_14"] >= 0).all()


class TestGetLiveFeatures:
    def test_returns_series_with_all_feature_columns(self):
        raw = _make_ohlcv(300)
        with patch("data.features.fetch_latest_candles", return_value=raw):
            series = get_live_features("^FTSE")
        assert isinstance(series, pd.Series)
        for col in FEATURE_COLUMNS:
            assert col in series.index, f"Missing feature: {col}"

    def test_raises_when_data_empty(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        with (
            patch("data.features.fetch_latest_candles", return_value=empty),
            pytest.raises(ValueError, match="No feature data"),
        ):
            get_live_features("^FTSE")
