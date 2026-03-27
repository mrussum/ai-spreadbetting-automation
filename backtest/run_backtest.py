"""Vectorbt backtest with ATR-based stops and performance reporting."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import EPIC_TO_TICKER, SIGNAL_THRESHOLD
from data.collector import fetch_historical
from data.features import FEATURE_COLUMNS, engineer_features
from models.trainer import _load_training_data

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Summary statistics from a single backtest run."""

    epic: str
    ticker: str
    start_date: str
    end_date: str
    n_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    annualised_return: float


def _generate_signals(df: pd.DataFrame, buy_probs: np.ndarray) -> tuple[pd.Series, pd.Series]:
    """Convert model probabilities to vectorbt-compatible entry/exit series.

    Returns:
        Tuple of (entries, exits) boolean Series indexed like df.
    """
    sell_probs = 1.0 - buy_probs

    entries = pd.Series(buy_probs >= SIGNAL_THRESHOLD, index=df.index)
    exits = pd.Series(sell_probs >= SIGNAL_THRESHOLD, index=df.index)

    # Prevent simultaneous entry and exit on the same bar
    conflict = entries & exits
    entries[conflict] = False
    exits[conflict] = False

    return entries, exits


def run_backtest(
    epic: str,
    period: str = "3y",
    atr_stop_multiplier: float = 2.0,
    initial_cash: float = 10_000.0,
) -> BacktestResult:
    """Run a vectorbt backtest for a single instrument.

    The model is trained on the first two-thirds of the data and tested
    on the final third, preserving temporal integrity.

    Args:
        epic: IG instrument epic (e.g. "IX.D.FTSE.DAILY.IP")
        period: yfinance period for historical download
        atr_stop_multiplier: stop loss = atr_stop_multiplier × ATR(14)
        initial_cash: starting portfolio value for the simulation

    Returns:
        :class:`BacktestResult` with key performance metrics.
    """
    try:
        import vectorbt as vbt
    except ImportError:
        raise ImportError("vectorbt is required for backtesting: pip install vectorbt")

    ticker = EPIC_TO_TICKER.get(epic)
    if ticker is None:
        raise ValueError(f"No ticker mapping for epic {epic!r}")

    logger.info("Loading data for backtest: %s (%s, period=%s)", epic, ticker, period)
    df = fetch_historical(ticker, period=period, interval="1d", force=False)
    df = engineer_features(df, include_target=False)

    if len(df) < 100:
        raise ValueError(f"Insufficient data for backtest ({len(df)} rows)")

    # Walk-forward split: train on first 2/3, test on last 1/3
    split_idx = int(len(df) * 2 / 3)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Train a model on training slice
    from xgboost import XGBClassifier
    from sklearn.model_selection import TimeSeriesSplit

    train_df_with_target = train_df.copy()
    # Build target on training data
    train_df_with_target["target_3"] = (
        train_df_with_target["close"].shift(-3) > train_df_with_target["close"]
    ).astype(int)
    train_df_with_target = train_df_with_target.dropna(subset=["target_3"])

    X_train = train_df_with_target[FEATURE_COLUMNS].values
    y_train = train_df_with_target["target_3"].values

    # Quick single-fold train (backtest is for evaluation, not hypertuning)
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=float((y_train == 0).sum()) / max((y_train == 1).sum(), 1),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)

    # Generate signals on the test slice
    X_test = test_df[FEATURE_COLUMNS].values
    buy_probs = model.predict_proba(X_test)[:, 1]

    close = test_df["close"]
    atr = test_df["atr_14"]
    entries, exits = _generate_signals(test_df, buy_probs)

    # Compute ATR-based stop prices: entry_price - multiplier * ATR
    # vectorbt SL is a percentage distance from entry price
    # Convert ATR stop to a fractional distance per bar
    sl_stop = (atr * atr_stop_multiplier / close).clip(upper=0.20)  # cap at 20%

    # Run the portfolio simulation
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        sl_stop=sl_stop,
        init_cash=initial_cash,
        fees=0.001,        # 0.1% round-trip fee approximation
        freq="1D",
    )

    stats = pf.stats()
    n_trades = int(stats.get("Total Trades", 0))
    win_rate = float(stats.get("Win Rate [%]", 0.0)) / 100.0
    sharpe = float(stats.get("Sharpe Ratio", 0.0))
    max_dd = float(stats.get("Max Drawdown [%]", 0.0)) / 100.0
    total_ret = float(stats.get("Total Return [%]", 0.0)) / 100.0
    ann_ret = float(stats.get("Annualized Return [%]", 0.0)) / 100.0

    result = BacktestResult(
        epic=epic,
        ticker=ticker,
        start_date=str(test_df.index[0].date()),
        end_date=str(test_df.index[-1].date()),
        n_trades=n_trades,
        win_rate=win_rate,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        total_return=total_ret,
        annualised_return=ann_ret,
    )

    logger.info(
        "Backtest %s: trades=%d, win_rate=%.1f%%, sharpe=%.2f, "
        "max_dd=%.1f%%, total_return=%.1f%%",
        epic, n_trades, win_rate * 100, sharpe, max_dd * 100, total_ret * 100,
    )
    return result


def run_all_backtests(period: str = "3y") -> list[BacktestResult]:
    """Run backtests for all configured instruments."""
    results = []
    for epic in EPIC_TO_TICKER:
        try:
            result = run_backtest(epic, period=period)
            results.append(result)
        except Exception as e:
            logger.error("Backtest failed for %s: %s", epic, e)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = run_all_backtests()
    print("\n=== Backtest Results ===")
    for r in results:
        print(
            f"\n{r.epic} ({r.ticker}) | {r.start_date} → {r.end_date}\n"
            f"  Trades: {r.n_trades}  Win rate: {r.win_rate:.1%}\n"
            f"  Sharpe: {r.sharpe_ratio:.2f}  Max DD: {r.max_drawdown:.1%}\n"
            f"  Total return: {r.total_return:.1%}  Ann. return: {r.annualised_return:.1%}"
        )
