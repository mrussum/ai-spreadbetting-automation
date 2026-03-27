"""Risk management layer — gates every trade before execution."""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from config.settings import (
    MAX_DAILY_LOSS,
    MAX_OPEN_POSITIONS,
    MAX_RISK_PER_TRADE,
    TRADING_MODE,
)
from database.crud import (
    get_daily_stats,
    get_kill_switch_status,
    set_kill_switch,
)

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    """Outcome of a full risk gate evaluation."""

    passed: bool
    reasons: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.passed = False
        self.reasons.append(reason)

    @property
    def summary(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "OK"


@dataclass
class PositionSpec:
    """Sizing output for a single trade."""

    size: float          # stake per point / contract units
    stop_distance: float # points from entry price
    risk_amount: float   # £ at risk on this trade


class RiskManager:
    """Evaluates whether a proposed trade is permissible under the current
    risk rules and computes position sizing.

    All checks are independent so that every failure reason is captured
    and logged, rather than short-circuiting at the first failure.

    Args:
        account_balance: current available account balance in £.
        open_positions: list of currently open :class:`broker.ig_models.Position`.
    """

    def __init__(
        self,
        account_balance: float,
        open_positions: list,
    ) -> None:
        self._balance = account_balance
        self._open_positions = open_positions

    # ─── Kill switch ─────────────────────────────────────────────────────────

    def is_kill_switch_active(self) -> bool:
        """Return True if the persistent kill switch is set."""
        return get_kill_switch_status()

    def trigger_kill_switch(self, reason: str) -> None:
        """Activate the kill switch and log the event.

        This is called automatically when the daily loss limit is breached.
        It can also be invoked externally (e.g. from the dashboard).
        """
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)
        set_kill_switch(active=True, reason=reason)

    # ─── Individual checks ───────────────────────────────────────────────────

    def _check_kill_switch(self, result: RiskCheckResult) -> None:
        if self.is_kill_switch_active():
            result.fail("Kill switch is active")

    def _check_daily_loss(self, result: RiskCheckResult) -> None:
        """Fail if today's realised loss exceeds MAX_DAILY_LOSS of starting balance."""
        stats = get_daily_stats(date.today())
        if stats is None:
            return  # No stats yet today — no trades taken, no loss

        if stats.starting_balance and stats.starting_balance > 0:
            loss_pct = -stats.total_pnl / stats.starting_balance
            if loss_pct >= MAX_DAILY_LOSS:
                reason = (
                    f"Daily loss limit reached: {loss_pct:.1%} loss "
                    f"(limit={MAX_DAILY_LOSS:.1%})"
                )
                result.fail(reason)
                self.trigger_kill_switch(reason)

    def _check_position_count(self, result: RiskCheckResult) -> None:
        n = len(self._open_positions)
        if n >= MAX_OPEN_POSITIONS:
            result.fail(
                f"Max open positions reached: {n}/{MAX_OPEN_POSITIONS}"
            )

    def _check_correlation(self, result: RiskCheckResult, epic: str) -> None:
        """Reject if we already hold a position in the same instrument."""
        for pos in self._open_positions:
            if pos.epic == epic:
                result.fail(
                    f"Already holding an open position in {epic}"
                )
                return

    def _check_balance(self, result: RiskCheckResult) -> None:
        if self._balance <= 0:
            result.fail(f"Account balance is zero or negative: £{self._balance:.2f}")

    # ─── Gate ────────────────────────────────────────────────────────────────

    def check_all(self, epic: str) -> RiskCheckResult:
        """Run every risk check for a proposed trade on *epic*.

        Args:
            epic: IG instrument epic being considered.

        Returns:
            :class:`RiskCheckResult` — inspect `.passed` and `.reasons`.
        """
        result = RiskCheckResult(passed=True)

        self._check_kill_switch(result)
        self._check_daily_loss(result)
        self._check_position_count(result)
        self._check_correlation(result, epic)
        self._check_balance(result)

        if result.passed:
            logger.info("Risk checks PASSED for %s", epic)
        else:
            logger.warning(
                "Risk checks FAILED for %s: %s", epic, result.summary
            )

        return result

    # ─── Position sizing ─────────────────────────────────────────────────────

    def size_position(self, atr: float, price: float) -> PositionSpec:
        """Compute stake size using fixed-fractional risk and ATR stop.

        Risk budget per trade = MAX_RISK_PER_TRADE × account_balance.
        Stop distance = 2 × ATR (in price points).
        Size = risk_budget / stop_distance.

        Args:
            atr: 14-period ATR value for the instrument (in price points).
            price: current mid price (used for logging/scaling only).

        Returns:
            :class:`PositionSpec` with size, stop_distance, and risk_amount.
        """
        risk_amount = MAX_RISK_PER_TRADE * self._balance
        stop_distance = max(2.0 * atr, 1.0)   # floor at 1 point to avoid division by zero
        size = risk_amount / stop_distance

        # Round down to 1 decimal place (IG minimum increment for most instruments)
        size = max(round(size * 10) / 10, 0.1)

        spec = PositionSpec(
            size=size,
            stop_distance=round(stop_distance, 1),
            risk_amount=round(risk_amount, 2),
        )

        logger.info(
            "Position sizing for price=%.2f, ATR=%.2f: "
            "size=%.1f, stop=%.1f pts, risk=£%.2f",
            price, atr, spec.size, spec.stop_distance, spec.risk_amount,
        )
        return spec
