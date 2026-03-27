"""Trade execution orchestrator — signal → risk → LLM → execute/paper."""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from config.settings import TRADING_MODE
from broker.ig_client import IGClient, IGAPIError
from broker.ig_models import DealConfirmation
from database.crud import (
    get_daily_stats,
    get_kill_switch_status,
    save_daily_stats,
    save_trade_decision,
    save_trade_outcome,
)
from llm.analyst import LLMAnalyst, LLMDecision
from models.predictor import SignalPredictor, SignalResult
from risk.manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class CycleOutcome:
    """Result of a single run_cycle() call for one instrument."""

    epic: str
    action_taken: str        # "BUY", "SELL", "SKIP", "PAPER_BUY", "PAPER_SELL", "BLOCKED"
    ml_signal: str
    ml_confidence: float
    llm_action: Optional[str]
    llm_confidence: Optional[float]
    llm_reasoning: Optional[str]
    risk_passed: bool
    deal_id: Optional[str]
    entry_price: Optional[float]
    size: Optional[float]
    stop_distance: Optional[float]


class Trader:
    """Orchestrates one full decision cycle for a single instrument.

    The pipeline is:
      1. Generate ML signal via :class:`~models.predictor.SignalPredictor`
      2. Run risk gate via :class:`~risk.manager.RiskManager`
      3. Validate with LLM via :class:`~llm.analyst.LLMAnalyst`
      4. Execute (live or paper) or skip
      5. Persist decision to DB (before execution in LIVE mode)

    Args:
        ig_client: authenticated :class:`~broker.ig_client.IGClient`.
        analyst: shared :class:`~llm.analyst.LLMAnalyst` instance.
        predictors: dict mapping epic → :class:`~models.predictor.SignalPredictor`.
    """

    def __init__(
        self,
        ig_client: IGClient,
        analyst: LLMAnalyst,
        predictors: dict[str, SignalPredictor],
    ) -> None:
        self._ig = ig_client
        self._analyst = analyst
        self._predictors = predictors

    # ─── Public entry point ──────────────────────────────────────────────────

    def run_cycle(self, epic: str) -> CycleOutcome:
        """Execute a full decision cycle for *epic*.

        Returns a :class:`CycleOutcome` regardless of what happened so the
        scheduler can log and continue.
        """
        logger.info("=== Cycle start: %s [mode=%s] ===", epic, TRADING_MODE)

        # ── 0. Kill switch guard (fast path) ─────────────────────────────────
        if get_kill_switch_status():
            logger.warning("Kill switch active — skipping cycle for %s", epic)
            return self._outcome(epic, "BLOCKED", signal="HOLD", ml_conf=0.0)

        # ── 1. ML signal ─────────────────────────────────────────────────────
        try:
            predictor = self._get_predictor(epic)
            signal_result: SignalResult = predictor.predict()
        except Exception as e:
            logger.error("Signal generation failed for %s: %s", epic, e)
            return self._outcome(epic, "BLOCKED", signal="HOLD", ml_conf=0.0)

        if signal_result.signal == "HOLD":
            logger.info("%s: ML signal is HOLD — skipping", epic)
            self._save_decision(epic, signal_result, None, False, "SKIP", None, None)
            return self._outcome(
                epic, "SKIP",
                signal="HOLD", ml_conf=signal_result.confidence,
            )

        # ── 2. Account state & risk gate ─────────────────────────────────────
        try:
            account_info = self._ig.get_account_info()
            open_positions = self._ig.get_positions()
        except Exception as e:
            logger.error("Failed to fetch account state for %s: %s", epic, e)
            return self._outcome(epic, "BLOCKED", signal=signal_result.signal,
                                 ml_conf=signal_result.confidence)

        risk = RiskManager(
            account_balance=account_info.available,
            open_positions=open_positions,
        )
        risk_result = risk.check_all(epic)

        if not risk_result.passed:
            self._save_decision(epic, signal_result, None, False, "SKIP", None, None)
            return self._outcome(
                epic, "SKIP",
                signal=signal_result.signal, ml_conf=signal_result.confidence,
                risk_passed=False,
            )

        # ── 3. LLM validation ─────────────────────────────────────────────────
        daily_stats = get_daily_stats(date.today())
        daily_pnl = float(daily_stats.total_pnl) if daily_stats else 0.0

        try:
            llm_decision: LLMDecision = self._analyst.analyse(
                epic=epic,
                signal=signal_result.signal,
                ml_confidence=signal_result.confidence,
                features=signal_result.features_snapshot,
                open_positions=len(open_positions),
                daily_pnl=daily_pnl,
            )
        except Exception as e:
            logger.error("LLM analyst raised unexpectedly for %s: %s — SKIP", epic, e)
            self._save_decision(epic, signal_result, None, True, "SKIP", None, None)
            return self._outcome(
                epic, "SKIP",
                signal=signal_result.signal, ml_conf=signal_result.confidence,
                risk_passed=True,
            )

        # LLM must agree with the ML signal direction
        llm_agrees = llm_decision.action == signal_result.signal
        if llm_decision.action == "SKIP" or not llm_agrees:
            self._save_decision(epic, signal_result, llm_decision, True, "SKIP", None, None)
            return self._outcome(
                epic, "SKIP",
                signal=signal_result.signal, ml_conf=signal_result.confidence,
                risk_passed=True,
                llm_decision=llm_decision,
            )

        # ── 4. Position sizing ────────────────────────────────────────────────
        atr = signal_result.features_snapshot.get("atr_14", 10.0)
        try:
            price = self._ig.get_price(epic)
            mid_price = price.mid
        except Exception as e:
            logger.warning("Could not fetch live price for %s: %s — using ATR estimate", epic, e)
            mid_price = 0.0

        spec = risk.size_position(atr=atr, price=mid_price)
        direction = signal_result.signal  # "BUY" or "SELL"

        # ── 5. Save decision BEFORE execution (safety rule) ───────────────────
        decision_id = self._save_decision(
            epic, signal_result, llm_decision, True,
            f"PAPER_{direction}" if TRADING_MODE == "PAPER" else direction,
            spec.size, spec.stop_distance,
        )

        # ── 6. Execute or paper trade ─────────────────────────────────────────
        if TRADING_MODE == "PAPER":
            logger.info(
                "PAPER TRADE: %s %s size=%.1f stop=%.1fpts entry≈%.2f",
                direction, epic, spec.size, spec.stop_distance, mid_price,
            )
            self._record_paper_outcome(epic, direction, spec.size, mid_price, spec.stop_distance)
            return self._outcome(
                epic, f"PAPER_{direction}",
                signal=signal_result.signal, ml_conf=signal_result.confidence,
                risk_passed=True, llm_decision=llm_decision,
                size=spec.size, stop_distance=spec.stop_distance, entry_price=mid_price,
            )

        # LIVE execution
        try:
            confirm: DealConfirmation = self._ig.open_position(
                epic=epic,
                direction=direction,
                size=spec.size,
                stop_distance=spec.stop_distance,
            )
        except IGAPIError as e:
            logger.error("Order rejected by IG for %s: %s", epic, e)
            return self._outcome(
                epic, "BLOCKED",
                signal=signal_result.signal, ml_conf=signal_result.confidence,
                risk_passed=True, llm_decision=llm_decision,
            )

        if not confirm.accepted:
            logger.error("Deal not accepted for %s: %s", epic, confirm.reason)
            return self._outcome(
                epic, "BLOCKED",
                signal=signal_result.signal, ml_conf=signal_result.confidence,
                risk_passed=True, llm_decision=llm_decision,
            )

        logger.info(
            "LIVE TRADE OPENED: %s %s deal_id=%s level=%.2f size=%.1f",
            direction, epic, confirm.deal_id, confirm.level, spec.size,
        )
        self._record_live_outcome(epic, direction, confirm, spec.size)

        return self._outcome(
            epic, direction,
            signal=signal_result.signal, ml_conf=signal_result.confidence,
            risk_passed=True, llm_decision=llm_decision,
            deal_id=confirm.deal_id, entry_price=confirm.level,
            size=spec.size, stop_distance=spec.stop_distance,
        )

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _get_predictor(self, epic: str) -> SignalPredictor:
        if epic not in self._predictors:
            # Lazy-load if not pre-warmed
            self._predictors[epic] = SignalPredictor(epic)
        return self._predictors[epic]

    def _save_decision(
        self,
        epic: str,
        signal: SignalResult,
        llm: Optional[LLMDecision],
        risk_passed: bool,
        action_taken: str,
        size: Optional[float],
        stop_distance: Optional[float],
    ) -> int:
        record = {
            "epic": epic,
            "ml_signal": signal.signal,
            "ml_confidence": signal.confidence,
            "llm_action": llm.action if llm else None,
            "llm_confidence": llm.confidence if llm else None,
            "llm_reasoning": llm.reasoning if llm else None,
            "risk_checks_passed": "True" if risk_passed else "False",
            "action_taken": action_taken,
            "deal_id": None,
            "position_size": size,
            "stop_distance": stop_distance,
            "entry_price": None,
            "trading_mode": TRADING_MODE,
        }
        try:
            return save_trade_decision(record)
        except Exception as e:
            logger.error("Failed to save trade decision for %s: %s", epic, e)
            return -1

    def _record_paper_outcome(
        self,
        epic: str,
        direction: str,
        size: float,
        entry_price: float,
        stop_distance: float,
    ) -> None:
        try:
            save_trade_outcome({
                "deal_id": f"PAPER-{epic}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                "open_time": datetime.utcnow(),
                "epic": epic,
                "direction": direction,
                "entry_price": entry_price,
                "size": size,
                "close_reason": "PAPER",
            })
        except Exception as e:
            logger.warning("Failed to record paper outcome for %s: %s", epic, e)

    def _record_live_outcome(
        self,
        epic: str,
        direction: str,
        confirm: DealConfirmation,
        size: float,
    ) -> None:
        try:
            save_trade_outcome({
                "deal_id": confirm.deal_id,
                "open_time": datetime.utcnow(),
                "epic": epic,
                "direction": direction,
                "entry_price": confirm.level,
                "size": size,
            })
        except Exception as e:
            logger.warning("Failed to record live outcome for %s: %s", epic, e)

    @staticmethod
    def _outcome(
        epic: str,
        action_taken: str,
        signal: str = "HOLD",
        ml_conf: float = 0.0,
        risk_passed: bool = False,
        llm_decision: Optional[LLMDecision] = None,
        deal_id: Optional[str] = None,
        entry_price: Optional[float] = None,
        size: Optional[float] = None,
        stop_distance: Optional[float] = None,
    ) -> CycleOutcome:
        return CycleOutcome(
            epic=epic,
            action_taken=action_taken,
            ml_signal=signal,
            ml_confidence=ml_conf,
            llm_action=llm_decision.action if llm_decision else None,
            llm_confidence=llm_decision.confidence if llm_decision else None,
            llm_reasoning=llm_decision.reasoning if llm_decision else None,
            risk_passed=risk_passed,
            deal_id=deal_id,
            entry_price=entry_price,
            size=size,
            stop_distance=stop_distance,
        )
