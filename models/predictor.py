"""Live signal prediction using a trained XGBoost model."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config.settings import EPIC_TO_TICKER, MODELS_DIR, SIGNAL_THRESHOLD
from data.features import FEATURE_COLUMNS, get_live_features

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """Output of a single prediction cycle."""

    epic: str
    signal: str              # "BUY", "SELL", or "HOLD"
    confidence: float        # probability of predicted direction (0–1)
    buy_prob: float          # raw P(up) from the model
    sell_prob: float         # 1 - buy_prob
    features_snapshot: dict  # feature values used for this prediction


class SignalPredictor:
    """Load a saved XGBoost model and generate trading signals.

    Usage::

        predictor = SignalPredictor("IX.D.FTSE.DAILY.IP")
        result = predictor.predict()
    """

    def __init__(self, epic: str) -> None:
        self.epic = epic
        self.ticker = EPIC_TO_TICKER.get(epic)
        if self.ticker is None:
            raise ValueError(f"No ticker mapping for epic {epic!r}")

        safe_epic = epic.replace(".", "_")
        model_path = MODELS_DIR / f"{safe_epic}.pkl"
        meta_path = MODELS_DIR / f"{safe_epic}_features.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                f"Run `python -m models.trainer` first."
            )

        self._model: XGBClassifier = joblib.load(model_path)

        # Validate feature list matches what the model was trained on
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            saved_features = meta.get("feature_columns", FEATURE_COLUMNS)
            if saved_features != FEATURE_COLUMNS:
                logger.warning(
                    "Feature list mismatch for %s — model was trained with different features. "
                    "Retrain the model.",
                    epic,
                )
        else:
            logger.warning("No feature metadata found for %s; assuming FEATURE_COLUMNS", epic)

        logger.info("Loaded model for %s from %s", epic, model_path)

    def predict(self) -> SignalResult:
        """Fetch latest market data, compute features, and return a signal.

        Returns:
            :class:`SignalResult` with signal direction and probabilities.

        Raises:
            ValueError: if live feature data cannot be fetched.
        """
        features: pd.Series = get_live_features(self.ticker)

        X = features[FEATURE_COLUMNS].values.reshape(1, -1)

        proba = self._model.predict_proba(X)[0]
        buy_prob = float(proba[1])
        sell_prob = float(proba[0])

        if buy_prob >= SIGNAL_THRESHOLD:
            signal = "BUY"
            confidence = buy_prob
        elif sell_prob >= SIGNAL_THRESHOLD:
            signal = "SELL"
            confidence = sell_prob
        else:
            signal = "HOLD"
            confidence = max(buy_prob, sell_prob)

        features_snapshot = {col: float(features[col]) for col in FEATURE_COLUMNS}

        result = SignalResult(
            epic=self.epic,
            signal=signal,
            confidence=confidence,
            buy_prob=buy_prob,
            sell_prob=sell_prob,
            features_snapshot=features_snapshot,
        )

        logger.info(
            "Signal for %s: %s (confidence=%.3f, buy_prob=%.3f, sell_prob=%.3f)",
            self.epic, signal, confidence, buy_prob, sell_prob,
        )
        return result
