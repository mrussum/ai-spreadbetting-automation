"""Database read/write helpers."""

from datetime import date, datetime
from typing import Optional

from sqlalchemy import desc

from database.models import (
    DailyStats,
    KillSwitchFlag,
    LLMLog,
    SessionLocal,
    TradeDecision,
    TradeOutcome,
)


# ─── Trade Decisions ─────────────────────────────────────────


def save_trade_decision(decision: dict) -> int:
    """Save a trade decision record. Returns the record id."""
    with SessionLocal() as session:
        record = TradeDecision(**decision)
        session.add(record)
        session.commit()
        return record.id


def get_todays_decisions() -> list[TradeDecision]:
    """Get all trade decisions from today."""
    today = date.today()
    with SessionLocal() as session:
        return (
            session.query(TradeDecision)
            .filter(TradeDecision.timestamp >= datetime(today.year, today.month, today.day))
            .order_by(desc(TradeDecision.timestamp))
            .all()
        )


def get_recent_decisions(limit: int = 50) -> list[TradeDecision]:
    """Get the most recent trade decisions."""
    with SessionLocal() as session:
        return (
            session.query(TradeDecision)
            .order_by(desc(TradeDecision.timestamp))
            .limit(limit)
            .all()
        )


# ─── Trade Outcomes ──────────────────────────────────────────


def save_trade_outcome(outcome: dict) -> int:
    """Save a trade outcome record. Returns the record id."""
    with SessionLocal() as session:
        record = TradeOutcome(**outcome)
        session.add(record)
        session.commit()
        return record.id


def update_trade_outcome(deal_id: str, updates: dict) -> None:
    """Update a trade outcome by deal_id."""
    with SessionLocal() as session:
        session.query(TradeOutcome).filter(
            TradeOutcome.deal_id == deal_id
        ).update(updates)
        session.commit()


def get_open_outcomes() -> list[TradeOutcome]:
    """Get trade outcomes that haven't been closed yet."""
    with SessionLocal() as session:
        return (
            session.query(TradeOutcome)
            .filter(TradeOutcome.close_time.is_(None))
            .all()
        )


# ─── LLM Logs ────────────────────────────────────────────────


def save_llm_log(log: dict) -> int:
    """Save an LLM interaction log. Returns the record id."""
    with SessionLocal() as session:
        record = LLMLog(**log)
        session.add(record)
        session.commit()
        return record.id


def get_recent_llm_logs(limit: int = 20) -> list[LLMLog]:
    """Get the most recent LLM logs."""
    with SessionLocal() as session:
        return (
            session.query(LLMLog)
            .order_by(desc(LLMLog.timestamp))
            .limit(limit)
            .all()
        )


# ─── Daily Stats ─────────────────────────────────────────────


def save_daily_stats(stats: dict) -> None:
    """Save or update daily stats for a given date."""
    with SessionLocal() as session:
        existing = (
            session.query(DailyStats)
            .filter(DailyStats.date == stats["date"])
            .first()
        )
        if existing:
            for key, value in stats.items():
                setattr(existing, key, value)
        else:
            session.add(DailyStats(**stats))
        session.commit()


def get_daily_stats(target_date: Optional[date] = None) -> Optional[DailyStats]:
    """Get daily stats for a specific date (defaults to today)."""
    target_date = target_date or date.today()
    with SessionLocal() as session:
        return (
            session.query(DailyStats)
            .filter(DailyStats.date == target_date)
            .first()
        )


def get_stats_history(days: int = 30) -> list[DailyStats]:
    """Get the last N days of stats."""
    with SessionLocal() as session:
        return (
            session.query(DailyStats)
            .order_by(desc(DailyStats.date))
            .limit(days)
            .all()
        )


# ─── Kill Switch ─────────────────────────────────────────────


def get_kill_switch_status() -> bool:
    """Check if the kill switch is active. Returns True if active."""
    with SessionLocal() as session:
        flag = session.query(KillSwitchFlag).first()
        if flag is None:
            return False
        return flag.active == 1


def set_kill_switch(active: bool, reason: str = "") -> None:
    """Set the kill switch state."""
    with SessionLocal() as session:
        flag = session.query(KillSwitchFlag).first()
        if flag is None:
            flag = KillSwitchFlag(id=1)
            session.add(flag)
        flag.active = 1 if active else 0
        flag.activated_at = datetime.utcnow() if active else None
        flag.reason = reason
        session.commit()
