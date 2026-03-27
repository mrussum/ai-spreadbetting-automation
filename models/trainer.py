"""XGBoost model training pipeline with time-series cross-validation."""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from config.settings import EPIC_TO_TICKER, MODELS_DIR
from data.collector import fetch_historical
from data.features import FEATURE_COLUMNS, engineer_features

logger = logging.getLogger(__name__)


def _load_training_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download and feature-engineer data for a ticker."""
    df = fetch_historical(ticker, period=period, interval="1d", force=False)
    df = engineer_features(df, include_target=True)
    # Drop the last 3 rows — target is NaN because shift(-3) looks ahead
    df = df.dropna(subset=["target_3"])
    return df


def train(
    epic: str,
    period: str = "5y",
    n_splits: int = 5,
    early_stopping_rounds: int = 50,
) -> dict:
    """Train an XGBoost classifier for a single instrument.

    Uses TimeSeriesSplit to preserve temporal order. Saves the trained model
    and the feature list to MODELS_DIR.

    Args:
        epic: IG instrument epic (e.g. "IX.D.FTSE.DAILY.IP")
        period: yfinance period string for historical download
        n_splits: number of folds for TimeSeriesSplit CV
        early_stopping_rounds: XGBoost early stopping patience

    Returns:
        Dict with keys: epic, ticker, accuracy, auc, n_train_rows, model_path
    """
    ticker = EPIC_TO_TICKER.get(epic)
    if ticker is None:
        raise ValueError(f"No ticker mapping for epic {epic!r}")

    logger.info("Loading training data for %s (%s)", epic, ticker)
    df = _load_training_data(ticker, period=period)

    X = df[FEATURE_COLUMNS].values
    y = df["target_3"].values

    logger.info(
        "Training data: %d rows, %d features, %.1f%% positive",
        len(X), X.shape[1], 100 * y.mean(),
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Collect OOF metrics across folds
    oof_accuracy: list[float] = []
    oof_auc: list[float] = []
    best_model: XGBClassifier | None = None
    best_auc = -1.0

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=1.0,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=float((y_train == 0).sum()) / max((y_train == 1).sum(), 1),
            use_label_encoder=False,
            eval_metric="logloss",
            early_stopping_rounds=early_stopping_rounds,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        proba = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, preds)
        auc = roc_auc_score(y_val, proba)
        oof_accuracy.append(acc)
        oof_auc.append(auc)

        logger.info("Fold %d: accuracy=%.4f, AUC=%.4f", fold, acc, auc)

        if auc > best_auc:
            best_auc = auc
            best_model = model

    mean_acc = float(np.mean(oof_accuracy))
    mean_auc = float(np.mean(oof_auc))
    logger.info(
        "CV complete — mean accuracy=%.4f, mean AUC=%.4f", mean_acc, mean_auc
    )

    # Retrain best hyperparams on ALL data (no early stopping on full set)
    final_model = XGBClassifier(
        n_estimators=best_model.best_iteration + 1 if best_model.best_iteration else 300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=float((y == 0).sum()) / max((y == 1).sum(), 1),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y, verbose=False)

    # Persist model and feature metadata
    safe_epic = epic.replace(".", "_")
    model_path = MODELS_DIR / f"{safe_epic}.pkl"
    meta_path = MODELS_DIR / f"{safe_epic}_features.json"

    joblib.dump(final_model, model_path)
    meta_path.write_text(
        json.dumps({"feature_columns": FEATURE_COLUMNS, "epic": epic, "ticker": ticker}, indent=2)
    )

    logger.info("Model saved to %s", model_path)

    return {
        "epic": epic,
        "ticker": ticker,
        "accuracy": mean_acc,
        "auc": mean_auc,
        "n_train_rows": len(X),
        "model_path": str(model_path),
    }


def train_all(period: str = "5y") -> list[dict]:
    """Train a model for every configured instrument.

    Returns:
        List of result dicts from :func:`train`.
    """
    results = []
    for epic in EPIC_TO_TICKER:
        try:
            result = train(epic, period=period)
            results.append(result)
        except Exception as e:
            logger.error("Training failed for %s: %s", epic, e)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = train_all()
    for r in results:
        print(f"{r['epic']}: AUC={r['auc']:.4f}, acc={r['accuracy']:.4f}, rows={r['n_train_rows']}")
