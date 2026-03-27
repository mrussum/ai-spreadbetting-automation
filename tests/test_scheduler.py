"""Tests for scheduler market-hours logic."""

from datetime import datetime
from unittest.mock import patch

import pytz
import pytest

from scheduler.runner import _within_market_hours

LONDON = pytz.timezone("Europe/London")


def _london_dt(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """Construct a timezone-aware London datetime."""
    return LONDON.localize(datetime(year, month, day, hour, minute))


# 2024-01-08 is a Monday, 2024-01-13 is a Saturday
MONDAY = (2024, 1, 8)
FRIDAY = (2024, 1, 12)
SATURDAY = (2024, 1, 13)
SUNDAY = (2024, 1, 14)


def _run_at(dt: datetime) -> bool:
    with patch("scheduler.runner.datetime") as mock_dt:
        mock_dt.now.return_value = dt
        return _within_market_hours()


class TestWithinMarketHours:
    def test_monday_at_open_is_tradeable(self):
        assert _run_at(_london_dt(*MONDAY, 8, 0)) is True

    def test_monday_midday_is_tradeable(self):
        assert _run_at(_london_dt(*MONDAY, 12, 0)) is True

    def test_friday_at_close_is_tradeable(self):
        assert _run_at(_london_dt(*FRIDAY, 16, 30)) is True

    def test_friday_after_close_is_not_tradeable(self):
        assert _run_at(_london_dt(*FRIDAY, 16, 31)) is False

    def test_monday_before_open_is_not_tradeable(self):
        assert _run_at(_london_dt(*MONDAY, 7, 59)) is False

    def test_saturday_is_not_tradeable(self):
        assert _run_at(_london_dt(*SATURDAY, 12, 0)) is False

    def test_sunday_is_not_tradeable(self):
        assert _run_at(_london_dt(*SUNDAY, 10, 0)) is False

    def test_monday_at_midnight_is_not_tradeable(self):
        assert _run_at(_london_dt(*MONDAY, 0, 0)) is False

    def test_friday_lunchtime_is_tradeable(self):
        assert _run_at(_london_dt(*FRIDAY, 13, 15)) is True

    def test_monday_at_1600_is_tradeable(self):
        assert _run_at(_london_dt(*MONDAY, 16, 0)) is True

    def test_monday_at_1631_is_not_tradeable(self):
        assert _run_at(_london_dt(*MONDAY, 16, 31)) is False
