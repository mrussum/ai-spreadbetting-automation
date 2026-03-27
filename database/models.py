"""SQLAlchemy table definitions for the trading system."""

from datetime import datetime

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config.settings import DB_PATH

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class TradeDecision(Base):
    __tablename__ = "trade_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    epic = Column(String(50), nullable=False)
    ml_signal = Column(String(10), nullable=False)
    ml_confidence = Column(Float, nullable=False)
    llm_action = Column(String(10))
    llm_confidence = Column(Float)
    llm_reasoning = Column(Text)
    risk_checks_passed = Column(String(5), nullable=False)
    action_taken = Column(String(20), nullable=False)
    deal_id = Column(String(50))
    position_size = Column(Float)
    stop_distance = Column(Float)
    entry_price = Column(Float)
    trading_mode = Column(String(10), nullable=False)


class TradeOutcome(Base):
    __tablename__ = "trade_outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    deal_id = Column(String(50), nullable=False)
    open_time = Column(DateTime, nullable=False)
    close_time = Column(DateTime)
    epic = Column(String(50), nullable=False)
    direction = Column(String(10), nullable=False)
    entry_price = Column(Float, nullable=False)
    close_price = Column(Float)
    size = Column(Float, nullable=False)
    pnl = Column(Float)
    close_reason = Column(String(30))


class LLMLog(Base):
    __tablename__ = "llm_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    epic = Column(String(50), nullable=False)
    prompt_text = Column(Text, nullable=False)
    response_text = Column(Text, nullable=False)
    tokens_used = Column(Integer)
    latency_ms = Column(Integer)


class DailyStats(Base):
    __tablename__ = "daily_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True)
    starting_balance = Column(Float, nullable=False)
    ending_balance = Column(Float)
    trades_taken = Column(Integer, default=0)
    trades_won = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)


class KillSwitchFlag(Base):
    """Single-row table for the kill switch state."""

    __tablename__ = "kill_switch"

    id = Column(Integer, primary_key=True, default=1)
    active = Column(Integer, default=0)  # 0=off, 1=on
    activated_at = Column(DateTime)
    reason = Column(String(200))


def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)
