"""Tests for Trader.run_cycle() — all external calls mocked."""

from unittest.mock import MagicMock, patch

import pytest

from broker.ig_models import AccountInfo, DealConfirmation, Position, Price
from execution.trader import CycleOutcome, Trader
from llm.analyst import LLMDecision
from models.predictor import SignalResult


# ─── Fixtures / factories ─────────────────────────────────────────────────────

EPIC = "IX.D.FTSE.DAILY.IP"

FEATURES = {
    "rsi_14": 55.0, "macd_hist": 0.01, "ema_20": 7500.0, "ema_50": 7400.0,
    "bb_width": 0.02, "atr_14": 25.0, "stoch_k": 60.0, "volume_ratio": 1.1,
    "returns_1": 0.005, "returns_5": 0.012, "ema_200": 7300.0,
    "macd": 10.0, "macd_signal": 9.5, "stoch_d": 58.0, "roc_10": 1.2,
    "williams_r": -30.0, "bb_upper": 7600.0, "bb_lower": 7400.0,
    "obv": 1e8, "volume_sma_20": 5e6,
}


def _signal(signal="BUY", confidence=0.75) -> SignalResult:
    return SignalResult(
        epic=EPIC, signal=signal, confidence=confidence,
        buy_prob=confidence if signal == "BUY" else 1 - confidence,
        sell_prob=confidence if signal == "SELL" else 1 - confidence,
        features_snapshot=FEATURES,
    )


def _llm_decision(action="BUY", confidence=0.80) -> LLMDecision:
    return LLMDecision(
        action=action, confidence=confidence,
        reasoning="Conditions look favourable.", tokens_used=100, latency_ms=200,
    )


def _account_info(balance=10_000.0) -> AccountInfo:
    return AccountInfo("ACC1", "Spread", "SPREADBET", "GBP", balance, balance, 0.0, balance)


def _price(bid=7500.0, offer=7502.0) -> Price:
    return Price(epic=EPIC, bid=bid, offer=offer, mid=(bid + offer) / 2,
                 status="TRADEABLE", update_time="10:00:00")


def _confirmed_deal() -> DealConfirmation:
    return DealConfirmation(
        deal_reference="REF1", deal_id="DEAL1", epic=EPIC,
        direction="BUY", size=1.0, level=7501.0,
        status="ACCEPTED", reason="SUCCESS",
    )


def _make_trader(predictor_signal: SignalResult | None = None) -> Trader:
    ig = MagicMock()
    ig.get_account_info.return_value = _account_info()
    ig.get_positions.return_value = []
    ig.get_price.return_value = _price()
    ig.open_position.return_value = _confirmed_deal()

    analyst = MagicMock()
    analyst.analyse.return_value = _llm_decision("BUY")

    predictor = MagicMock()
    predictor.predict.return_value = predictor_signal or _signal("BUY")

    return Trader(ig_client=ig, analyst=analyst, predictors={EPIC: predictor})


# ─── Kill switch short-circuit ────────────────────────────────────────────────

class TestKillSwitchGuard:
    def test_returns_blocked_when_kill_switch_active(self):
        trader = _make_trader()
        with patch("execution.trader.get_kill_switch_status", return_value=True):
            outcome = trader.run_cycle(EPIC)
        assert outcome.action_taken == "BLOCKED"

    def test_does_not_call_predictor_when_kill_switch_active(self):
        trader = _make_trader()
        with patch("execution.trader.get_kill_switch_status", return_value=True):
            trader.run_cycle(EPIC)
        trader._predictors[EPIC].predict.assert_not_called()


# ─── HOLD signal ──────────────────────────────────────────────────────────────

class TestHoldSignal:
    def test_skip_when_ml_signal_is_hold(self):
        trader = _make_trader(predictor_signal=_signal("HOLD", 0.55))
        with (
            patch("execution.trader.get_kill_switch_status", return_value=False),
            patch("execution.trader.save_trade_decision", return_value=1),
        ):
            outcome = trader.run_cycle(EPIC)
        assert outcome.action_taken == "SKIP"
        assert outcome.ml_signal == "HOLD"


# ─── Risk gate ────────────────────────────────────────────────────────────────

class TestRiskGate:
    def test_skip_when_risk_fails(self):
        trader = _make_trader()
        failing_result = MagicMock()
        failing_result.passed = False
        failing_result.summary = "max positions reached"

        with (
            patch("execution.trader.get_kill_switch_status", return_value=False),
            patch("execution.trader.RiskManager") as MockRM,
            patch("execution.trader.save_trade_decision", return_value=1),
        ):
            MockRM.return_value.check_all.return_value = failing_result
            outcome = trader.run_cycle(EPIC)

        assert outcome.action_taken == "SKIP"
        assert outcome.risk_passed is False


# ─── LLM gate ────────────────────────────────────────────────────────────────

class TestLLMGate:
    def _run_with_llm(self, llm_action: str, llm_conf: float = 0.80) -> CycleOutcome:
        trader = _make_trader()
        trader._analyst.analyse.return_value = _llm_decision(llm_action, llm_conf)

        passing_risk = MagicMock()
        passing_risk.passed = True

        with (
            patch("execution.trader.get_kill_switch_status", return_value=False),
            patch("execution.trader.RiskManager") as MockRM,
            patch("execution.trader.get_daily_stats", return_value=None),
            patch("execution.trader.save_trade_decision", return_value=1),
            patch("execution.trader.save_trade_outcome"),
        ):
            MockRM.return_value.check_all.return_value = passing_risk
            MockRM.return_value.size_position.return_value = MagicMock(
                size=1.0, stop_distance=50.0, risk_amount=100.0
            )
            return trader.run_cycle(EPIC)

    def test_skip_when_llm_says_skip(self):
        outcome = self._run_with_llm("SKIP")
        assert outcome.action_taken == "SKIP"

    def test_skip_when_llm_disagrees_with_ml(self):
        # ML says BUY, LLM says SELL — should not trade
        outcome = self._run_with_llm("SELL")
        assert outcome.action_taken == "SKIP"


# ─── Paper mode ───────────────────────────────────────────────────────────────

class TestPaperMode:
    def test_paper_buy_when_all_checks_pass(self):
        trader = _make_trader()
        passing_risk = MagicMock()
        passing_risk.passed = True

        with (
            patch("execution.trader.TRADING_MODE", "PAPER"),
            patch("execution.trader.get_kill_switch_status", return_value=False),
            patch("execution.trader.RiskManager") as MockRM,
            patch("execution.trader.get_daily_stats", return_value=None),
            patch("execution.trader.save_trade_decision", return_value=1),
            patch("execution.trader.save_trade_outcome"),
        ):
            MockRM.return_value.check_all.return_value = passing_risk
            MockRM.return_value.size_position.return_value = MagicMock(
                size=1.0, stop_distance=50.0, risk_amount=100.0
            )
            outcome = trader.run_cycle(EPIC)

        assert outcome.action_taken == "PAPER_BUY"
        assert outcome.risk_passed is True
        assert outcome.llm_action == "BUY"

    def test_ig_open_position_not_called_in_paper_mode(self):
        trader = _make_trader()
        passing_risk = MagicMock()
        passing_risk.passed = True

        with (
            patch("execution.trader.TRADING_MODE", "PAPER"),
            patch("execution.trader.get_kill_switch_status", return_value=False),
            patch("execution.trader.RiskManager") as MockRM,
            patch("execution.trader.get_daily_stats", return_value=None),
            patch("execution.trader.save_trade_decision", return_value=1),
            patch("execution.trader.save_trade_outcome"),
        ):
            MockRM.return_value.check_all.return_value = passing_risk
            MockRM.return_value.size_position.return_value = MagicMock(
                size=1.0, stop_distance=50.0, risk_amount=100.0
            )
            trader.run_cycle(EPIC)

        trader._ig.open_position.assert_not_called()


# ─── Live mode ────────────────────────────────────────────────────────────────

class TestLiveMode:
    def test_live_buy_returns_deal_id(self):
        trader = _make_trader()
        passing_risk = MagicMock()
        passing_risk.passed = True

        with (
            patch("execution.trader.TRADING_MODE", "LIVE"),
            patch("execution.trader.get_kill_switch_status", return_value=False),
            patch("execution.trader.RiskManager") as MockRM,
            patch("execution.trader.get_daily_stats", return_value=None),
            patch("execution.trader.save_trade_decision", return_value=1),
            patch("execution.trader.save_trade_outcome"),
        ):
            MockRM.return_value.check_all.return_value = passing_risk
            MockRM.return_value.size_position.return_value = MagicMock(
                size=1.0, stop_distance=50.0, risk_amount=100.0
            )
            outcome = trader.run_cycle(EPIC)

        assert outcome.action_taken == "BUY"
        assert outcome.deal_id == "DEAL1"
        assert outcome.entry_price == pytest.approx(7501.0)

    def test_blocked_when_deal_rejected(self):
        trader = _make_trader()
        rejected = DealConfirmation(
            "REF1", "", EPIC, "BUY", 1.0, 0.0, "REJECTED", "MARKET_CLOSED"
        )
        trader._ig.open_position.return_value = rejected
        passing_risk = MagicMock()
        passing_risk.passed = True

        with (
            patch("execution.trader.TRADING_MODE", "LIVE"),
            patch("execution.trader.get_kill_switch_status", return_value=False),
            patch("execution.trader.RiskManager") as MockRM,
            patch("execution.trader.get_daily_stats", return_value=None),
            patch("execution.trader.save_trade_decision", return_value=1),
            patch("execution.trader.save_trade_outcome"),
        ):
            MockRM.return_value.check_all.return_value = passing_risk
            MockRM.return_value.size_position.return_value = MagicMock(
                size=1.0, stop_distance=50.0, risk_amount=100.0
            )
            outcome = trader.run_cycle(EPIC)

        assert outcome.action_taken == "BLOCKED"

    def test_decision_saved_before_execution(self):
        """save_trade_decision must be called before open_position."""
        call_order = []
        trader = _make_trader()
        passing_risk = MagicMock()
        passing_risk.passed = True

        def record_decision(*args, **kwargs):
            call_order.append("save_decision")
            return 1

        def record_open(*args, **kwargs):
            call_order.append("open_position")
            return _confirmed_deal()

        trader._ig.open_position.side_effect = record_open

        with (
            patch("execution.trader.TRADING_MODE", "LIVE"),
            patch("execution.trader.get_kill_switch_status", return_value=False),
            patch("execution.trader.RiskManager") as MockRM,
            patch("execution.trader.get_daily_stats", return_value=None),
            patch("execution.trader.save_trade_decision", side_effect=record_decision),
            patch("execution.trader.save_trade_outcome"),
        ):
            MockRM.return_value.check_all.return_value = passing_risk
            MockRM.return_value.size_position.return_value = MagicMock(
                size=1.0, stop_distance=50.0, risk_amount=100.0
            )
            trader.run_cycle(EPIC)

        assert call_order == ["save_decision", "open_position"], (
            "Trade decision must be persisted BEFORE the order is sent"
        )
