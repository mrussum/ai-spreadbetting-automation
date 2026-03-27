"""Tests for the XGBoost training pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from data.features import FEATURE_COLUMNS, engineer_features
from models.trainer import _load_training_data, train


def _make_ohlcv(n: int = 500) -> pd.DataFrame:
    """Synthetic OHLCV with enough rows for all indicators to warm up."""
    np.random.seed(0)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 7000 + np.cumsum(np.random.randn(n) * 20)
    high = close + np.abs(np.random.randn(n) * 10)
    low = close - np.abs(np.random.randn(n) * 10)
    open_ = close + np.random.randn(n) * 5
    volume = np.random.randint(1_000_000, 10_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestTrain:
    def test_train_returns_expected_keys(self, tmp_path):
        """train() should return a dict with all required metric keys."""
        raw_df = _make_ohlcv(500)
        featured_df = engineer_features(raw_df, include_target=True).dropna(subset=["target_3"])

        epic = "IX.D.FTSE.DAILY.IP"
        ticker = "^FTSE"

        with (
            patch("models.trainer.fetch_historical", return_value=raw_df),
            patch("models.trainer.MODELS_DIR", tmp_path),
        ):
            result = train(epic, period="5y", n_splits=3, early_stopping_rounds=10)

        assert "epic" in result
        assert "ticker" in result
        assert "accuracy" in result
        assert "auc" in result
        assert "n_train_rows" in result
        assert "model_path" in result

    def test_train_saves_model_file(self, tmp_path):
        """train() should write a .pkl file to MODELS_DIR."""
        raw_df = _make_ohlcv(500)

        epic = "IX.D.FTSE.DAILY.IP"

        with (
            patch("models.trainer.fetch_historical", return_value=raw_df),
            patch("models.trainer.MODELS_DIR", tmp_path),
        ):
            result = train(epic, period="5y", n_splits=3, early_stopping_rounds=10)

        assert Path(result["model_path"]).exists()

    def test_train_saves_feature_metadata(self, tmp_path):
        """train() should write a _features.json file alongside the model."""
        raw_df = _make_ohlcv(500)
        epic = "IX.D.FTSE.DAILY.IP"

        with (
            patch("models.trainer.fetch_historical", return_value=raw_df),
            patch("models.trainer.MODELS_DIR", tmp_path),
        ):
            train(epic, period="5y", n_splits=3, early_stopping_rounds=10)

        safe_epic = epic.replace(".", "_")
        meta_path = tmp_path / f"{safe_epic}_features.json"
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["feature_columns"] == FEATURE_COLUMNS

    def test_train_auc_between_0_and_1(self, tmp_path):
        """AUC must be a valid probability value."""
        raw_df = _make_ohlcv(500)
        epic = "IX.D.FTSE.DAILY.IP"

        with (
            patch("models.trainer.fetch_historical", return_value=raw_df),
            patch("models.trainer.MODELS_DIR", tmp_path),
        ):
            result = train(epic, period="5y", n_splits=3, early_stopping_rounds=10)

        assert 0.0 <= result["auc"] <= 1.0

    def test_train_unknown_epic_raises(self, tmp_path):
        """train() must raise ValueError for unmapped epics."""
        with pytest.raises(ValueError, match="No ticker mapping"):
            train("UNKNOWN.EPIC", period="5y")
