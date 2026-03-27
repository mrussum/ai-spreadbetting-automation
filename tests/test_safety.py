"""Safety property tests — assert non-negotiable system invariants.

These tests verify the rules listed in the project spec:
  1. TRADING_MODE defaults to PAPER
  2. Kill switch is independent of other components
  3. API keys/tokens are never logged
  4. Trade decisions saved to DB BEFORE execution (covered in test_trader.py)
  5. LLM failure → SKIP (covered in test_llm_analyst.py)
"""

import json
import logging
import re
from unittest.mock import MagicMock, patch

import pytest


# ─── Rule 1: TRADING_MODE defaults to PAPER ───────────────────────────────────

class TestTradingModeDefault:
    def test_default_trading_mode_is_paper(self):
        """If TRADING_MODE env var is unset, the system must default to PAPER."""
        import os
        import importlib

        # Reload settings without TRADING_MODE in the environment
        env_without_mode = {k: v for k, v in os.environ.items() if k != "TRADING_MODE"}
        with patch.dict(os.environ, env_without_mode, clear=True):
            import config.settings as settings
            importlib.reload(settings)
            mode = settings.TRADING_MODE

        assert mode == "PAPER", (
            f"TRADING_MODE defaulted to {mode!r} instead of 'PAPER'. "
            "This is a safety violation — the system must never default to LIVE."
        )

    def test_paper_mode_blocks_ig_open_position(self):
        """In PAPER mode, IGClient.open_position must never be called by Trader."""
        from broker.ig_models import AccountInfo, Price
        from execution.trader import Trader
        from llm.analyst import LLMDecision
        from models.predictor import SignalResult

        features = {col: 0.5 for col in __import__("data.features", fromlist=["FEATURE_COLUMNS"]).FEATURE_COLUMNS}
        features["atr_14"] = 20.0

        ig = MagicMock()
        ig.get_account_info.return_value = AccountInfo("A", "B", "C", "GBP", 10000.0, 10000.0, 0.0, 10000.0)
        ig.get_positions.return_value = []
        ig.get_price.return_value = Price("IX.D.FTSE.DAILY.IP", 7500.0, 7502.0, 7501.0, "TRADEABLE", "10:00")

        analyst = MagicMock()
        analyst.analyse.return_value = LLMDecision("BUY", 0.85, "OK", 100, 200)

        predictor = MagicMock()
        predictor.predict.return_value = SignalResult(
            epic="IX.D.FTSE.DAILY.IP", signal="BUY", confidence=0.75,
            buy_prob=0.75, sell_prob=0.25, features_snapshot=features,
        )

        trader = Trader(ig_client=ig, analyst=analyst, predictors={"IX.D.FTSE.DAILY.IP": predictor})
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
            trader.run_cycle("IX.D.FTSE.DAILY.IP")

        ig.open_position.assert_not_called()


# ─── Rule 2: Kill switch is independent ───────────────────────────────────────

class TestKillSwitchIndependence:
    def test_kill_switch_blocks_cycle_even_if_predictor_broken(self):
        """Kill switch must stop trading even if the ML predictor crashes."""
        from execution.trader import Trader

        broken_predictor = MagicMock()
        broken_predictor.predict.side_effect = RuntimeError("model corrupt")

        ig = MagicMock()
        analyst = MagicMock()
        trader = Trader(ig_client=ig, analyst=analyst, predictors={"IX.D.FTSE.DAILY.IP": broken_predictor})

        with patch("execution.trader.get_kill_switch_status", return_value=True):
            outcome = trader.run_cycle("IX.D.FTSE.DAILY.IP")

        assert outcome.action_taken == "BLOCKED"
        broken_predictor.predict.assert_not_called()

    def test_kill_switch_blocks_cycle_even_if_ig_client_broken(self):
        """Kill switch must stop trading even if the IG client crashes."""
        from execution.trader import Trader

        broken_ig = MagicMock()
        broken_ig.get_account_info.side_effect = RuntimeError("network error")

        analyst = MagicMock()
        predictor = MagicMock()
        trader = Trader(ig_client=broken_ig, analyst=analyst, predictors={"IX.D.FTSE.DAILY.IP": predictor})

        with patch("execution.trader.get_kill_switch_status", return_value=True):
            outcome = trader.run_cycle("IX.D.FTSE.DAILY.IP")

        assert outcome.action_taken == "BLOCKED"
        broken_ig.get_account_info.assert_not_called()

    def test_set_kill_switch_persists_to_db(self, in_memory_engine):
        """Kill switch state is stored in the DB, not in process memory."""
        from database.crud import get_kill_switch_status, set_kill_switch
        from sqlalchemy.orm import sessionmaker
        TestSession = sessionmaker(bind=in_memory_engine)

        with patch("database.crud.SessionLocal", TestSession):
            set_kill_switch(active=True, reason="safety test")
            # Read it back via a fresh query — not a cached value
            status = get_kill_switch_status()

        assert status is True


# ─── Rule 3: No secrets in logs ───────────────────────────────────────────────

class TestNoSecretsInLogs:
    """Verify that credential values never appear in log output."""

    SENSITIVE_PATTERNS = [
        r"sk-ant-",           # Anthropic key prefix
        r"password",          # password field (case-insensitive)
        r"X-SECURITY-TOKEN",  # IG session token header name in log lines
        r"api[_-]?key\s*=\s*\S{8,}",  # key=<value>
    ]

    def _capture_logs(self, func, *args, **kwargs):
        """Run func and return all log messages emitted."""
        handler = logging.handlers.MemoryHandler(capacity=1000, flushLevel=logging.CRITICAL)
        records: list[logging.LogRecord] = []

        class Collector(logging.Handler):
            def emit(self, record):
                records.append(record)

        collector = Collector()
        root = logging.getLogger()
        root.addHandler(collector)
        try:
            func(*args, **kwargs)
        except Exception:
            pass
        finally:
            root.removeHandler(collector)
        return [r.getMessage() for r in records]

    def test_ig_client_login_does_not_log_password(self):
        from broker.ig_client import IGClient
        client = IGClient()
        resp = MagicMock()
        resp.status_code = 200
        resp.ok = True
        resp.headers = {"CST": "test-cst", "X-SECURITY-TOKEN": "test-xst"}
        resp.json.return_value = {}

        import logging.handlers
        with patch.object(client._session, "post", return_value=resp):
            messages = self._capture_logs(client.login)

        full_output = "\n".join(messages).lower()
        assert "igtestdemo" not in full_output
        assert "password" not in full_output

    def test_ig_headers_builder_does_not_log_token(self):
        """_headers() must not log the security token value."""
        from broker.ig_client import IGClient
        client = IGClient()
        client._cst = "secret-cst-value"
        client._security_token = "secret-xst-value"
        client._logged_in = True

        import logging.handlers
        messages = self._capture_logs(client._headers)
        full_output = "\n".join(messages)
        assert "secret-cst-value" not in full_output
        assert "secret-xst-value" not in full_output

    def test_llm_prompt_does_not_contain_api_key(self):
        """The prompt sent to Claude must not include any API key."""
        from llm.analyst import _build_prompt
        features = {col: 0.5 for col in __import__("data.features", fromlist=["FEATURE_COLUMNS"]).FEATURE_COLUMNS}
        prompt = _build_prompt("IX.D.FTSE.DAILY.IP", "BUY", 0.72, features, 1, -50.0)
        assert "sk-ant" not in prompt
        assert "IG_API_KEY" not in prompt


# ─── Rule 5: LLM failure → SKIP (integration assertion) ──────────────────────

class TestLLMFailSafe:
    def test_any_llm_exception_produces_skip_not_trade(self):
        """No exception from the LLM layer should ever result in a trade."""
        from llm.analyst import LLMAnalyst
        features = {col: 0.5 for col in __import__("data.features", fromlist=["FEATURE_COLUMNS"]).FEATURE_COLUMNS}

        with patch("llm.analyst.ANTHROPIC_API_KEY", "test-key"):
            analyst = LLMAnalyst.__new__(LLMAnalyst)
            analyst._client = MagicMock()

        # Simulate every possible failure mode
        for side_effect in [
            Exception("timeout"),
            ConnectionError("network down"),
            ValueError("unexpected response"),
            json.JSONDecodeError("bad json", "", 0),
        ]:
            analyst._client.messages.create.side_effect = side_effect
            with patch("llm.analyst.save_llm_log"):
                result = analyst.analyse("IX.D.FTSE.DAILY.IP", "BUY", 0.75, features)
            assert result.action == "SKIP", (
                f"LLM raised {side_effect!r} but action was {result.action!r} instead of SKIP"
            )
