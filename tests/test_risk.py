"""Tests for RiskManager — all DB calls are mocked."""

from unittest.mock import MagicMock, patch

import pytest

from risk.manager import PositionSpec, RiskCheckResult, RiskManager


def _make_position(epic: str = "IX.D.FTSE.DAILY.IP", direction: str = "BUY"):
    pos = MagicMock()
    pos.epic = epic
    pos.direction = direction
    return pos


def _make_manager(balance: float = 10_000.0, positions: list | None = None) -> RiskManager:
    return RiskManager(account_balance=balance, open_positions=positions or [])


# ─── RiskCheckResult ──────────────────────────────────────────────────────────

class TestRiskCheckResult:
    def test_starts_passing(self):
        r = RiskCheckResult(passed=True)
        assert r.passed is True
        assert r.reasons == []

    def test_fail_marks_not_passed(self):
        r = RiskCheckResult(passed=True)
        r.fail("reason A")
        assert r.passed is False
        assert "reason A" in r.reasons

    def test_multiple_failures_accumulated(self):
        r = RiskCheckResult(passed=True)
        r.fail("A")
        r.fail("B")
        assert len(r.reasons) == 2

    def test_summary_ok_when_no_reasons(self):
        r = RiskCheckResult(passed=True)
        assert r.summary == "OK"

    def test_summary_joins_reasons(self):
        r = RiskCheckResult(passed=True)
        r.fail("X")
        r.fail("Y")
        assert "X" in r.summary and "Y" in r.summary


# ─── Kill switch check ────────────────────────────────────────────────────────

class TestKillSwitchCheck:
    def test_fails_when_kill_switch_active(self):
        mgr = _make_manager()
        with (
            patch("risk.manager.get_kill_switch_status", return_value=True),
            patch("risk.manager.get_daily_stats", return_value=None),
        ):
            result = mgr.check_all("IX.D.FTSE.DAILY.IP")
        assert not result.passed
        assert any("kill switch" in r.lower() for r in result.reasons)

    def test_passes_when_kill_switch_inactive(self):
        mgr = _make_manager()
        with (
            patch("risk.manager.get_kill_switch_status", return_value=False),
            patch("risk.manager.get_daily_stats", return_value=None),
        ):
            result = mgr.check_all("IX.D.FTSE.DAILY.IP")
        assert result.passed


# ─── Daily loss check ─────────────────────────────────────────────────────────

class TestDailyLossCheck:
    def _fake_stats(self, starting: float, pnl: float):
        stats = MagicMock()
        stats.starting_balance = starting
        stats.total_pnl = pnl
        return stats

    def test_fails_when_loss_exceeds_limit(self):
        mgr = _make_manager()
        # 6% loss on 10k starting balance — exceeds MAX_DAILY_LOSS=5%
        stats = self._fake_stats(10_000.0, -600.0)
        with (
            patch("risk.manager.get_kill_switch_status", return_value=False),
            patch("risk.manager.get_daily_stats", return_value=stats),
            patch("risk.manager.set_kill_switch") as mock_ks,
        ):
            result = mgr.check_all("IX.D.FTSE.DAILY.IP")
        assert not result.passed
        mock_ks.assert_called_once_with(active=True, reason=pytest.approx(result.reasons[-1], abs=None))

    def test_passes_when_loss_within_limit(self):
        mgr = _make_manager()
        stats = self._fake_stats(10_000.0, -200.0)  # 2% loss — fine
        with (
            patch("risk.manager.get_kill_switch_status", return_value=False),
            patch("risk.manager.get_daily_stats", return_value=stats),
        ):
            result = mgr.check_all("IX.D.FTSE.DAILY.IP")
        assert result.passed

    def test_passes_when_no_stats_yet(self):
        mgr = _make_manager()
        with (
            patch("risk.manager.get_kill_switch_status", return_value=False),
            patch("risk.manager.get_daily_stats", return_value=None),
        ):
            result = mgr.check_all("IX.D.FTSE.DAILY.IP")
        assert result.passed


# ─── Position count check ─────────────────────────────────────────────────────

class TestPositionCountCheck:
    def test_fails_when_at_max_positions(self):
        positions = [_make_position(f"EPIC{i}") for i in range(3)]  # MAX_OPEN_POSITIONS=3
        mgr = _make_manager(positions=positions)
        with (
            patch("risk.manager.get_kill_switch_status", return_value=False),
            patch("risk.manager.get_daily_stats", return_value=None),
        ):
            result = mgr.check_all("NEW.EPIC")
        assert not result.passed
        assert any("max open positions" in r.lower() for r in result.reasons)

    def test_passes_when_below_max(self):
        positions = [_make_position("EPIC0"), _make_position("EPIC1")]
        mgr = _make_manager(positions=positions)
        with (
            patch("risk.manager.get_kill_switch_status", return_value=False),
            patch("risk.manager.get_daily_stats", return_value=None),
        ):
            result = mgr.check_all("NEW.EPIC")
        assert result.passed


# ─── Correlation / duplicate position check ───────────────────────────────────

class TestCorrelationCheck:
    def test_fails_when_same_epic_already_open(self):
        epic = "IX.D.FTSE.DAILY.IP"
        mgr = _make_manager(positions=[_make_position(epic)])
        with (
            patch("risk.manager.get_kill_switch_status", return_value=False),
            patch("risk.manager.get_daily_stats", return_value=None),
        ):
            result = mgr.check_all(epic)
        assert not result.passed
        assert any(epic in r for r in result.reasons)

    def test_passes_when_different_epic(self):
        mgr = _make_manager(positions=[_make_position("OTHER.EPIC")])
        with (
            patch("risk.manager.get_kill_switch_status", return_value=False),
            patch("risk.manager.get_daily_stats", return_value=None),
        ):
            result = mgr.check_all("IX.D.FTSE.DAILY.IP")
        assert result.passed


# ─── Balance check ────────────────────────────────────────────────────────────

class TestBalanceCheck:
    def test_fails_when_balance_zero(self):
        mgr = _make_manager(balance=0.0)
        with (
            patch("risk.manager.get_kill_switch_status", return_value=False),
            patch("risk.manager.get_daily_stats", return_value=None),
        ):
            result = mgr.check_all("IX.D.FTSE.DAILY.IP")
        assert not result.passed

    def test_fails_when_balance_negative(self):
        mgr = _make_manager(balance=-100.0)
        with (
            patch("risk.manager.get_kill_switch_status", return_value=False),
            patch("risk.manager.get_daily_stats", return_value=None),
        ):
            result = mgr.check_all("IX.D.FTSE.DAILY.IP")
        assert not result.passed


# ─── Position sizing ──────────────────────────────────────────────────────────

class TestSizePosition:
    def test_returns_position_spec(self):
        mgr = _make_manager(balance=10_000.0)
        spec = mgr.size_position(atr=20.0, price=7500.0)
        assert isinstance(spec, PositionSpec)

    def test_size_is_positive(self):
        mgr = _make_manager(balance=10_000.0)
        spec = mgr.size_position(atr=20.0, price=7500.0)
        assert spec.size > 0

    def test_stop_distance_is_2x_atr(self):
        mgr = _make_manager(balance=10_000.0)
        spec = mgr.size_position(atr=20.0, price=7500.0)
        assert spec.stop_distance == pytest.approx(40.0, abs=0.1)

    def test_risk_amount_equals_balance_times_max_risk(self):
        from config.settings import MAX_RISK_PER_TRADE
        mgr = _make_manager(balance=10_000.0)
        spec = mgr.size_position(atr=20.0, price=7500.0)
        assert spec.risk_amount == pytest.approx(10_000.0 * MAX_RISK_PER_TRADE, rel=0.01)

    def test_tiny_atr_uses_floor(self):
        """ATR near zero should not produce infinite size."""
        mgr = _make_manager(balance=10_000.0)
        spec = mgr.size_position(atr=0.0001, price=7500.0)
        assert spec.stop_distance >= 1.0
        assert spec.size < 10_000.0

    def test_size_minimum_is_0_1(self):
        """Even with a huge stop, minimum size is 0.1."""
        mgr = _make_manager(balance=1.0)   # tiny balance
        spec = mgr.size_position(atr=500.0, price=7500.0)
        assert spec.size >= 0.1
