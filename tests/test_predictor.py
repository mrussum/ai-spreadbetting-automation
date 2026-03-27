"""Tests for SignalPredictor."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from data.features import FEATURE_COLUMNS
from models.predictor import SignalPredictor, SignalResult


def _fake_model(n_features: int = len(FEATURE_COLUMNS)) -> XGBClassifier:
    """Train a minimal XGBClassifier on random data for test mocking."""
    np.random.seed(1)
    X = np.random.randn(200, n_features)
    y = (np.random.randn(200) > 0).astype(int)
    m = XGBClassifier(n_estimators=5, use_label_encoder=False, eval_metric="logloss", random_state=1)
    m.fit(X, y)
    return m


def _make_feature_series() -> pd.Series:
    """Return a Series with all FEATURE_COLUMNS populated."""
    np.random.seed(2)
    return pd.Series(np.random.randn(len(FEATURE_COLUMNS)), index=FEATURE_COLUMNS)


class TestSignalPredictor:
    def _build_predictor(self, tmp_path, buy_prob: float = 0.75) -> SignalPredictor:
        """Construct a SignalPredictor with a fake model saved in tmp_path."""
        epic = "IX.D.FTSE.DAILY.IP"
        safe_epic = epic.replace(".", "_")

        model = _fake_model()
        joblib.dump(model, tmp_path / f"{safe_epic}.pkl")
        (tmp_path / f"{safe_epic}_features.json").write_text(
            json.dumps({"feature_columns": FEATURE_COLUMNS, "epic": epic, "ticker": "^FTSE"})
        )

        with patch("models.predictor.MODELS_DIR", tmp_path):
            predictor = SignalPredictor(epic)

        # Override the model's predict_proba to return a fixed probability
        predictor._model = MagicMock()
        predictor._model.predict_proba = MagicMock(
            return_value=np.array([[1 - buy_prob, buy_prob]])
        )
        return predictor

    def test_returns_signal_result(self, tmp_path):
        predictor = self._build_predictor(tmp_path)
        features = _make_feature_series()

        with (
            patch("models.predictor.MODELS_DIR", tmp_path),
            patch("models.predictor.get_live_features", return_value=features),
        ):
            result = predictor.predict()

        assert isinstance(result, SignalResult)

    def test_buy_signal_above_threshold(self, tmp_path):
        """buy_prob ≥ SIGNAL_THRESHOLD should produce a BUY signal."""
        predictor = self._build_predictor(tmp_path, buy_prob=0.80)
        features = _make_feature_series()

        with patch("models.predictor.get_live_features", return_value=features):
            result = predictor.predict()

        assert result.signal == "BUY"
        assert result.confidence == pytest.approx(0.80, abs=1e-6)

    def test_sell_signal_above_threshold(self, tmp_path):
        """sell_prob ≥ SIGNAL_THRESHOLD should produce a SELL signal."""
        predictor = self._build_predictor(tmp_path, buy_prob=0.20)
        features = _make_feature_series()

        with patch("models.predictor.get_live_features", return_value=features):
            result = predictor.predict()

        assert result.signal == "SELL"
        assert result.confidence == pytest.approx(0.80, abs=1e-6)

    def test_hold_when_probabilities_below_threshold(self, tmp_path):
        """Neither probability meeting threshold should produce HOLD."""
        predictor = self._build_predictor(tmp_path, buy_prob=0.55)
        features = _make_feature_series()

        with patch("models.predictor.get_live_features", return_value=features):
            result = predictor.predict()

        assert result.signal == "HOLD"

    def test_features_snapshot_has_all_columns(self, tmp_path):
        """features_snapshot must contain every FEATURE_COLUMNS key."""
        predictor = self._build_predictor(tmp_path, buy_prob=0.75)
        features = _make_feature_series()

        with patch("models.predictor.get_live_features", return_value=features):
            result = predictor.predict()

        for col in FEATURE_COLUMNS:
            assert col in result.features_snapshot

    def test_missing_model_raises_file_not_found(self, tmp_path):
        """SignalPredictor should raise FileNotFoundError if model is absent."""
        with (
            patch("models.predictor.MODELS_DIR", tmp_path),
            pytest.raises(FileNotFoundError, match="No trained model found"),
        ):
            SignalPredictor("IX.D.FTSE.DAILY.IP")

    def test_unknown_epic_raises_value_error(self, tmp_path):
        """Unknown epic should raise ValueError before even touching the filesystem."""
        with (
            patch("models.predictor.MODELS_DIR", tmp_path),
            pytest.raises(ValueError, match="No ticker mapping"),
        ):
            SignalPredictor("UNKNOWN.EPIC")
