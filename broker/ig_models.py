"""Dataclasses for IG Group REST API responses."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AccountInfo:
    """Summary of the connected IG account."""

    account_id: str
    account_name: str
    account_type: str       # "SPREADBET", "CFD", etc.
    currency: str
    balance: float          # available to deal
    deposit: float          # total funds deposited
    profit_loss: float      # unrealised P&L
    available: float        # net liquidating value


@dataclass
class Position:
    """A single open position."""

    deal_id: str
    epic: str
    direction: str          # "BUY" or "SELL"
    size: float             # contract size / stake
    entry_price: float      # level at which position was opened
    current_price: float    # latest mid price
    unrealised_pnl: float
    currency: str
    created_date: str       # ISO-8601 string from API
    stop_level: Optional[float] = None
    limit_level: Optional[float] = None


@dataclass
class Price:
    """Current bid/offer snapshot for an instrument."""

    epic: str
    bid: float
    offer: float
    mid: float              # (bid + offer) / 2 — computed on construction
    status: str             # "TRADEABLE", "CLOSED", etc.
    update_time: str        # HH:MM:SS from API

    @classmethod
    def from_api(cls, epic: str, data: dict) -> "Price":
        """Construct from a raw IG /markets/{epic} API response."""
        snapshot = data.get("snapshot", {})
        bid = float(snapshot.get("bid", 0))
        offer = float(snapshot.get("offer", 0))
        return cls(
            epic=epic,
            bid=bid,
            offer=offer,
            mid=(bid + offer) / 2,
            status=snapshot.get("marketStatus", "UNKNOWN"),
            update_time=snapshot.get("updateTime", ""),
        )


@dataclass
class DealConfirmation:
    """Response from open/close position calls."""

    deal_reference: str
    deal_id: str
    epic: str
    direction: str          # "BUY" or "SELL"
    size: float
    level: float            # execution price
    status: str             # "ACCEPTED", "REJECTED"
    reason: str             # rejection reason or "SUCCESS"
    stop_level: Optional[float] = None
    limit_level: Optional[float] = None

    @property
    def accepted(self) -> bool:
        return self.status == "ACCEPTED"


@dataclass
class StreamPrice:
    """Real-time price tick from the Lightstreamer feed."""

    epic: str
    bid: float
    offer: float
    mid: float
    update_time: str
