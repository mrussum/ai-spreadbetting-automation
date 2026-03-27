"""Tests for IGClient and ig_models — all HTTP calls are mocked."""

from unittest.mock import MagicMock, patch

import pytest

from broker.ig_client import IGAuthError, IGAPIError, IGClient
from broker.ig_models import AccountInfo, DealConfirmation, Position, Price, StreamPrice


# ─── ig_models ────────────────────────────────────────────────────────────────

class TestPriceFromApi:
    def test_constructs_from_snapshot(self):
        data = {
            "snapshot": {
                "bid": 7500.0,
                "offer": 7502.0,
                "marketStatus": "TRADEABLE",
                "updateTime": "10:00:00",
            }
        }
        price = Price.from_api("IX.D.FTSE.DAILY.IP", data)
        assert price.bid == 7500.0
        assert price.offer == 7502.0
        assert price.mid == pytest.approx(7501.0)
        assert price.status == "TRADEABLE"

    def test_mid_computed_correctly(self):
        data = {"snapshot": {"bid": 100.0, "offer": 102.0, "marketStatus": "TRADEABLE", "updateTime": ""}}
        price = Price.from_api("TEST", data)
        assert price.mid == pytest.approx(101.0)


class TestDealConfirmation:
    def test_accepted_true_when_status_accepted(self):
        confirm = DealConfirmation(
            deal_reference="REF1", deal_id="D1", epic="EP", direction="BUY",
            size=1.0, level=7500.0, status="ACCEPTED", reason="SUCCESS",
        )
        assert confirm.accepted is True

    def test_accepted_false_when_status_rejected(self):
        confirm = DealConfirmation(
            deal_reference="REF2", deal_id="", epic="EP", direction="BUY",
            size=1.0, level=0.0, status="REJECTED", reason="LIMIT_ERROR",
        )
        assert confirm.accepted is False


# ─── IGClient helpers ─────────────────────────────────────────────────────────

def _make_logged_in_client() -> IGClient:
    """Return an IGClient whose session tokens are already set."""
    client = IGClient()
    client._cst = "test-cst"
    client._security_token = "test-xst"
    client._logged_in = True
    return client


def _mock_response(status_code: int, json_data: dict, headers: dict | None = None):
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = status_code < 400
    resp.json.return_value = json_data
    resp.headers = headers or {}
    resp.text = str(json_data)
    return resp


# ─── IGClient.login ──────────────────────────────────────────────────────────

class TestLogin:
    def test_login_stores_tokens(self):
        client = IGClient()
        resp = _mock_response(
            200, {"accountType": "SPREADBET"},
            headers={"CST": "my-cst", "X-SECURITY-TOKEN": "my-xst"},
        )

        with patch.object(client._session, "post", return_value=resp):
            client.login()

        assert client._cst == "my-cst"
        assert client._security_token == "my-xst"
        assert client._logged_in is True

    def test_login_raises_on_bad_status(self):
        client = IGClient()
        resp = _mock_response(403, {"errorCode": "invalid.input"})

        with patch.object(client._session, "post", return_value=resp):
            with pytest.raises(IGAuthError):
                client.login()

    def test_login_raises_when_tokens_missing(self):
        client = IGClient()
        resp = _mock_response(200, {}, headers={})

        with patch.object(client._session, "post", return_value=resp):
            with pytest.raises(IGAuthError, match="tokens were missing"):
                client.login()


# ─── IGClient._request ───────────────────────────────────────────────────────

class TestRequest:
    def test_raises_when_not_logged_in(self):
        client = IGClient()
        with pytest.raises(IGAuthError, match="Not logged in"):
            client._request("GET", "/positions")

    def test_raises_ig_api_error_on_non_2xx(self):
        client = _make_logged_in_client()
        resp = _mock_response(400, {"errorCode": "error.invalid"})

        with patch.object(client._session, "request", return_value=resp):
            with pytest.raises(IGAPIError) as exc_info:
                client._request("GET", "/positions")
        assert exc_info.value.status_code == 400

    def test_returns_empty_dict_on_204(self):
        client = _make_logged_in_client()
        resp = _mock_response(204, {})
        resp.ok = True

        with patch.object(client._session, "request", return_value=resp):
            result = client._request("DELETE", "/session")
        assert result == {}


# ─── IGClient.get_account_info ───────────────────────────────────────────────

class TestGetAccountInfo:
    def test_parses_account_info(self):
        client = _make_logged_in_client()
        api_data = {
            "accounts": [
                {
                    "accountId": "ABC123",
                    "accountName": "Spreadbet",
                    "accountType": "SPREADBET",
                    "currency": "GBP",
                    "balance": {
                        "balance": 10000.0,
                        "deposit": 9000.0,
                        "profitLoss": 500.0,
                        "available": 8500.0,
                    },
                }
            ]
        }

        with patch.object(client, "_request", return_value=api_data):
            info = client.get_account_info()

        assert isinstance(info, AccountInfo)
        assert info.balance == 10000.0
        assert info.profit_loss == 500.0
        assert info.currency == "GBP"


# ─── IGClient.get_positions ──────────────────────────────────────────────────

class TestGetPositions:
    def test_returns_list_of_positions(self):
        client = _make_logged_in_client()
        api_data = {
            "positions": [
                {
                    "position": {
                        "dealId": "D1",
                        "direction": "BUY",
                        "size": 2.0,
                        "level": 7500.0,
                        "upl": 100.0,
                        "currency": "GBP",
                        "createdDateUTC": "2024-01-01T10:00:00",
                    },
                    "market": {
                        "epic": "IX.D.FTSE.DAILY.IP",
                        "bid": 7550.0,
                    },
                }
            ]
        }

        with patch.object(client, "_request", return_value=api_data):
            positions = client.get_positions()

        assert len(positions) == 1
        pos = positions[0]
        assert isinstance(pos, Position)
        assert pos.deal_id == "D1"
        assert pos.direction == "BUY"
        assert pos.size == 2.0

    def test_returns_empty_list_when_no_positions(self):
        client = _make_logged_in_client()
        with patch.object(client, "_request", return_value={"positions": []}):
            assert client.get_positions() == []


# ─── IGClient.get_price ──────────────────────────────────────────────────────

class TestGetPrice:
    def test_parses_price(self):
        client = _make_logged_in_client()
        api_data = {
            "snapshot": {
                "bid": 7400.0,
                "offer": 7402.0,
                "marketStatus": "TRADEABLE",
                "updateTime": "09:30:00",
            }
        }

        with patch.object(client, "_request", return_value=api_data):
            price = client.get_price("IX.D.FTSE.DAILY.IP")

        assert isinstance(price, Price)
        assert price.bid == 7400.0
        assert price.status == "TRADEABLE"


# ─── IGClient.open_position ──────────────────────────────────────────────────

class TestOpenPosition:
    def _confirm_data(self, status="ACCEPTED") -> dict:
        return {
            "dealId": "DEAL1",
            "epic": "IX.D.FTSE.DAILY.IP",
            "direction": "BUY",
            "size": 1.0,
            "level": 7500.0,
            "dealStatus": status,
            "reason": "SUCCESS" if status == "ACCEPTED" else "LIMIT_ERROR",
        }

    def test_open_position_returns_confirmation(self):
        client = _make_logged_in_client()

        def fake_request(method, path, **kwargs):
            if method == "POST":
                return {"dealReference": "REF1"}
            return self._confirm_data("ACCEPTED")

        with patch.object(client, "_request", side_effect=fake_request):
            confirm = client.open_position("IX.D.FTSE.DAILY.IP", "BUY", size=1.0)

        assert isinstance(confirm, DealConfirmation)
        assert confirm.accepted is True
        assert confirm.deal_id == "DEAL1"

    def test_open_position_raises_on_invalid_direction(self):
        client = _make_logged_in_client()
        with pytest.raises(ValueError, match="direction must be"):
            client.open_position("IX.D.FTSE.DAILY.IP", "HOLD", size=1.0)

    def test_open_position_raises_on_zero_size(self):
        client = _make_logged_in_client()
        with pytest.raises(ValueError, match="size must be positive"):
            client.open_position("IX.D.FTSE.DAILY.IP", "BUY", size=0.0)


# ─── IGClient.close_all ──────────────────────────────────────────────────────

class TestCloseAll:
    def test_close_all_closes_every_position(self):
        client = _make_logged_in_client()

        pos1 = Position("D1", "IX.D.FTSE.DAILY.IP", "BUY", 1.0, 7500.0, 7510.0, 10.0, "GBP", "")
        pos2 = Position("D2", "CS.D.GBPUSD.TODAY.IP", "SELL", 2.0, 1.25, 1.26, -20.0, "GBP", "")

        accepted = DealConfirmation("R", "D", "E", "BUY", 1.0, 7500.0, "ACCEPTED", "SUCCESS")

        with (
            patch.object(client, "get_positions", return_value=[pos1, pos2]),
            patch.object(client, "close_position", return_value=accepted),
        ):
            results = client.close_all(reason="test kill switch")

        assert len(results) == 2

    def test_close_all_returns_empty_when_no_positions(self):
        client = _make_logged_in_client()
        with patch.object(client, "get_positions", return_value=[]):
            results = client.close_all()
        assert results == []
