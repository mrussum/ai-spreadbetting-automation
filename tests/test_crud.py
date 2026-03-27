"""Tests for database CRUD helpers using an in-memory SQLite database."""

from datetime import date, datetime
from unittest.mock import patch

import pytest

from database.models import (
    DailyStats,
    KillSwitchFlag,
    LLMLog,
    SessionLocal,
    TradeDecision,
    TradeOutcome,
)


# All tests patch SessionLocal to use the in-memory session from conftest.py
# We achieve isolation by pointing crud functions at the in-memory engine.

def _patch_session(in_memory_engine):
    """Return a context manager that redirects SessionLocal to the test engine."""
    from sqlalchemy.orm import sessionmaker
    TestSession = sessionmaker(bind=in_memory_engine)
    return patch("database.crud.SessionLocal", TestSession)


# ─── Kill switch ──────────────────────────────────────────────────────────────

class TestKillSwitch:
    def test_get_returns_false_when_no_row(self, in_memory_engine):
        from database.crud import get_kill_switch_status
        with _patch_session(in_memory_engine):
            assert get_kill_switch_status() is False

    def test_set_active_then_get_returns_true(self, in_memory_engine):
        from database.crud import get_kill_switch_status, set_kill_switch
        with _patch_session(in_memory_engine):
            set_kill_switch(active=True, reason="test")
            assert get_kill_switch_status() is True

    def test_deactivate_returns_false(self, in_memory_engine):
        from database.crud import get_kill_switch_status, set_kill_switch
        with _patch_session(in_memory_engine):
            set_kill_switch(active=True, reason="test")
            set_kill_switch(active=False)
            assert get_kill_switch_status() is False

    def test_set_twice_updates_single_row(self, in_memory_engine):
        """Calling set_kill_switch twice must not create two rows."""
        from sqlalchemy.orm import sessionmaker
        from database.crud import set_kill_switch
        TestSession = sessionmaker(bind=in_memory_engine)
        with _patch_session(in_memory_engine):
            set_kill_switch(active=True, reason="first")
            set_kill_switch(active=True, reason="second")
        with TestSession() as s:
            count = s.query(KillSwitchFlag).count()
        assert count == 1

    def test_activated_at_set_when_active(self, in_memory_engine):
        from sqlalchemy.orm import sessionmaker
        from database.crud import set_kill_switch
        TestSession = sessionmaker(bind=in_memory_engine)
        with _patch_session(in_memory_engine):
            set_kill_switch(active=True, reason="auto")
        with TestSession() as s:
            flag = s.query(KillSwitchFlag).first()
        assert flag.activated_at is not None

    def test_activated_at_cleared_when_deactivated(self, in_memory_engine):
        from sqlalchemy.orm import sessionmaker
        from database.crud import set_kill_switch
        TestSession = sessionmaker(bind=in_memory_engine)
        with _patch_session(in_memory_engine):
            set_kill_switch(active=True, reason="auto")
            set_kill_switch(active=False)
        with TestSession() as s:
            flag = s.query(KillSwitchFlag).first()
        assert flag.activated_at is None


# ─── Trade decisions ──────────────────────────────────────────────────────────

class TestTradeDecisions:
    def _decision(self, epic: str = "IX.D.FTSE.DAILY.IP") -> dict:
        return {
            "epic": epic,
            "ml_signal": "BUY",
            "ml_confidence": 0.75,
            "llm_action": "BUY",
            "llm_confidence": 0.82,
            "llm_reasoning": "Looks good.",
            "risk_checks_passed": "True",
            "action_taken": "PAPER_BUY",
            "deal_id": None,
            "position_size": 1.0,
            "stop_distance": 50.0,
            "entry_price": 7500.0,
            "trading_mode": "PAPER",
        }

    def test_save_returns_integer_id(self, in_memory_engine):
        from database.crud import save_trade_decision
        with _patch_session(in_memory_engine):
            record_id = save_trade_decision(self._decision())
        assert isinstance(record_id, int)
        assert record_id > 0

    def test_get_recent_decisions_returns_saved(self, in_memory_engine):
        from database.crud import get_recent_decisions, save_trade_decision
        with _patch_session(in_memory_engine):
            save_trade_decision(self._decision())
            decisions = get_recent_decisions(limit=10)
        assert len(decisions) == 1
        assert decisions[0].epic == "IX.D.FTSE.DAILY.IP"

    def test_get_recent_respects_limit(self, in_memory_engine):
        from database.crud import get_recent_decisions, save_trade_decision
        with _patch_session(in_memory_engine):
            for i in range(5):
                save_trade_decision(self._decision(f"EPIC{i}"))
            decisions = get_recent_decisions(limit=3)
        assert len(decisions) == 3


# ─── Daily stats ──────────────────────────────────────────────────────────────

class TestDailyStats:
    def test_get_returns_none_when_absent(self, in_memory_engine):
        from database.crud import get_daily_stats
        with _patch_session(in_memory_engine):
            result = get_daily_stats(date(2024, 1, 1))
        assert result is None

    def test_save_and_get_roundtrip(self, in_memory_engine):
        from database.crud import get_daily_stats, save_daily_stats
        target = date(2024, 1, 15)
        with _patch_session(in_memory_engine):
            save_daily_stats({
                "date": target,
                "starting_balance": 10000.0,
                "total_pnl": 150.0,
                "trades_taken": 2,
                "trades_won": 1,
                "max_drawdown": 0.02,
            })
            stats = get_daily_stats(target)
        assert stats is not None
        assert stats.total_pnl == pytest.approx(150.0)

    def test_save_twice_updates_not_duplicates(self, in_memory_engine):
        from sqlalchemy.orm import sessionmaker
        from database.crud import save_daily_stats
        TestSession = sessionmaker(bind=in_memory_engine)
        target = date(2024, 1, 20)
        with _patch_session(in_memory_engine):
            save_daily_stats({"date": target, "starting_balance": 10000.0, "total_pnl": 100.0, "trades_taken": 1, "trades_won": 1, "max_drawdown": 0.0})
            save_daily_stats({"date": target, "starting_balance": 10000.0, "total_pnl": 200.0, "trades_taken": 2, "trades_won": 1, "max_drawdown": 0.01})
        with TestSession() as s:
            count = s.query(DailyStats).count()
        assert count == 1


# ─── LLM logs ────────────────────────────────────────────────────────────────

class TestLLMLogs:
    def test_save_and_retrieve(self, in_memory_engine):
        from database.crud import get_recent_llm_logs, save_llm_log
        with _patch_session(in_memory_engine):
            save_llm_log({
                "epic": "IX.D.FTSE.DAILY.IP",
                "prompt_text": "Should I BUY?",
                "response_text": '{"action": "BUY", "confidence": 0.82, "reasoning": "OK."}',
                "tokens_used": 130,
                "latency_ms": 210,
            })
            logs = get_recent_llm_logs(limit=5)
        assert len(logs) == 1
        assert logs[0].epic == "IX.D.FTSE.DAILY.IP"
        assert logs[0].tokens_used == 130
