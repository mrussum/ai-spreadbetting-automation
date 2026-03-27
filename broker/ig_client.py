"""IG Group REST API client — authentication and trading operations."""

import logging
import time
from typing import Optional

import requests

from config.settings import (
    IG_ACC_ID,
    IG_ACC_TYPE,
    IG_API_KEY,
    IG_BASE_URL,
    IG_PASSWORD,
    IG_USERNAME,
)
from broker.ig_models import AccountInfo, DealConfirmation, Position, Price

logger = logging.getLogger(__name__)

# Seconds to wait before retrying a failed deal confirmation poll
_CONFIRM_POLL_INTERVAL = 1.0
_CONFIRM_MAX_ATTEMPTS = 10


class IGAuthError(Exception):
    """Raised when IG authentication fails."""


class IGAPIError(Exception):
    """Raised when the IG API returns an error response."""

    def __init__(self, status_code: int, error_code: str, message: str = "") -> None:
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(f"IG API error {status_code} [{error_code}]: {message}")


class IGClient:
    """Authenticated IG Group REST API client.

    Usage::

        client = IGClient()
        client.login()
        price = client.get_price("IX.D.FTSE.DAILY.IP")
        confirm = client.open_position("IX.D.FTSE.DAILY.IP", "BUY", size=1.0)
        client.logout()

    The client manages CST and X-Security-Token headers automatically after
    :meth:`login` and refreshes them on each request via the ``_request``
    helper.
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._cst: str = ""
        self._security_token: str = ""
        self._logged_in: bool = False

    # ─── Auth ────────────────────────────────────────────────────────────────

    def login(self) -> None:
        """Authenticate with IG and store session tokens.

        Uses version 2 of the /session endpoint to obtain CST and
        X-Security-Token headers required for all subsequent calls.

        Raises:
            IGAuthError: if credentials are missing or rejected.
        """
        if not all([IG_API_KEY, IG_USERNAME, IG_PASSWORD]):
            raise IGAuthError("IG_API_KEY, IG_USERNAME and IG_PASSWORD must be set")

        url = f"{IG_BASE_URL}/session"
        headers = {
            "X-IG-API-KEY": IG_API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json; charset=UTF-8",
            "Version": "2",
        }
        payload = {
            "identifier": IG_USERNAME,
            "password": IG_PASSWORD,
            "encryptedPassword": False,
        }

        resp = self._session.post(url, json=payload, headers=headers, timeout=10)

        if resp.status_code != 200:
            raise IGAuthError(
                f"Login failed ({resp.status_code}): {resp.text[:200]}"
            )

        self._cst = resp.headers.get("CST", "")
        self._security_token = resp.headers.get("X-SECURITY-TOKEN", "")

        if not self._cst or not self._security_token:
            raise IGAuthError("Login succeeded but tokens were missing from response headers")

        self._logged_in = True
        logger.info("Logged in to IG (%s account, type=%s)", IG_ACC_ID, IG_ACC_TYPE)

    def logout(self) -> None:
        """Invalidate the current IG session."""
        if not self._logged_in:
            return
        try:
            self._request("DELETE", "/session", version="1")
        except Exception as e:
            logger.warning("Logout request failed: %s", e)
        finally:
            self._cst = ""
            self._security_token = ""
            self._logged_in = False
            logger.info("Logged out of IG")

    # ─── Internal request helper ─────────────────────────────────────────────

    def _headers(self, version: str = "1") -> dict:
        """Build standard IG request headers."""
        return {
            "X-IG-API-KEY": IG_API_KEY,
            "CST": self._cst,
            "X-SECURITY-TOKEN": self._security_token,
            "Content-Type": "application/json",
            "Accept": "application/json; charset=UTF-8",
            "Version": version,
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        version: str = "1",
        payload: Optional[dict] = None,
        params: Optional[dict] = None,
        timeout: int = 10,
    ) -> dict:
        """Send an authenticated request and return the parsed JSON body.

        Args:
            method: HTTP verb ("GET", "POST", "PUT", "DELETE").
            path: API path relative to IG_BASE_URL (e.g. "/positions").
            version: IG API version header value.
            payload: JSON body for POST/PUT requests.
            params: URL query parameters.
            timeout: request timeout in seconds.

        Returns:
            Parsed JSON response as a dict (empty dict for 204 responses).

        Raises:
            IGAuthError: if the client is not logged in.
            IGAPIError: if the API returns a non-2xx status.
        """
        if not self._logged_in:
            raise IGAuthError("Not logged in — call login() first")

        url = f"{IG_BASE_URL}{path}"
        resp = self._session.request(
            method,
            url,
            headers=self._headers(version),
            json=payload,
            params=params,
            timeout=timeout,
        )

        # Update tokens if the API rotates them
        if "CST" in resp.headers:
            self._cst = resp.headers["CST"]
        if "X-SECURITY-TOKEN" in resp.headers:
            self._security_token = resp.headers["X-SECURITY-TOKEN"]

        if resp.status_code == 204:
            return {}

        if not resp.ok:
            try:
                body = resp.json()
                error_code = body.get("errorCode", "UNKNOWN")
            except Exception:
                error_code = "PARSE_ERROR"
            raise IGAPIError(resp.status_code, error_code, resp.text[:300])

        return resp.json()

    # ─── Account ─────────────────────────────────────────────────────────────

    def get_account_info(self) -> AccountInfo:
        """Return balance and P&L for the configured account.

        Raises:
            IGAPIError: if the account is not found.
        """
        data = self._request("GET", "/accounts", version="1")
        accounts = data.get("accounts", [])

        # Find the configured account ID; fall back to first
        account = next(
            (a for a in accounts if a.get("accountId") == IG_ACC_ID),
            accounts[0] if accounts else None,
        )
        if account is None:
            raise IGAPIError(404, "ACCOUNT_NOT_FOUND", f"Account {IG_ACC_ID} not found")

        balance_data = account.get("balance", {})
        return AccountInfo(
            account_id=account.get("accountId", ""),
            account_name=account.get("accountName", ""),
            account_type=account.get("accountType", ""),
            currency=account.get("currency", "GBP"),
            balance=float(balance_data.get("balance", 0)),
            deposit=float(balance_data.get("deposit", 0)),
            profit_loss=float(balance_data.get("profitLoss", 0)),
            available=float(balance_data.get("available", 0)),
        )

    # ─── Positions ───────────────────────────────────────────────────────────

    def get_positions(self) -> list[Position]:
        """Return all currently open positions."""
        data = self._request("GET", "/positions/otc", version="2")
        positions = []
        for item in data.get("positions", []):
            pos = item.get("position", {})
            market = item.get("market", {})
            positions.append(
                Position(
                    deal_id=pos.get("dealId", ""),
                    epic=market.get("epic", ""),
                    direction=pos.get("direction", ""),
                    size=float(pos.get("size", 0)),
                    entry_price=float(pos.get("level", 0)),
                    current_price=float(market.get("bid", 0)),
                    unrealised_pnl=float(pos.get("upl", 0)),
                    currency=pos.get("currency", "GBP"),
                    created_date=pos.get("createdDateUTC", ""),
                    stop_level=pos.get("stopLevel"),
                    limit_level=pos.get("limitLevel"),
                )
            )
        return positions

    # ─── Prices ──────────────────────────────────────────────────────────────

    def get_price(self, epic: str) -> Price:
        """Fetch the current bid/offer snapshot for an instrument."""
        data = self._request("GET", f"/markets/{epic}", version="3")
        return Price.from_api(epic, data)

    # ─── Order execution ─────────────────────────────────────────────────────

    def open_position(
        self,
        epic: str,
        direction: str,
        size: float,
        stop_distance: Optional[float] = None,
        limit_distance: Optional[float] = None,
    ) -> DealConfirmation:
        """Open a new OTC position and wait for confirmation.

        Args:
            epic: IG instrument epic.
            direction: "BUY" or "SELL".
            size: deal size (units / stake per point).
            stop_distance: points from entry for stop loss (optional).
            limit_distance: points from entry for limit (optional).

        Returns:
            :class:`DealConfirmation` — check `.accepted` before proceeding.
        """
        if direction not in ("BUY", "SELL"):
            raise ValueError(f"direction must be 'BUY' or 'SELL', got {direction!r}")
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")

        payload: dict = {
            "epic": epic,
            "expiry": "-",
            "direction": direction,
            "size": str(size),
            "orderType": "MARKET",
            "timeInForce": "FILL_OR_KILL",
            "guaranteedStop": False,
            "forceOpen": True,
            "currencyCode": "GBP",
        }
        if stop_distance is not None:
            payload["stopDistance"] = str(stop_distance)
        if limit_distance is not None:
            payload["limitDistance"] = str(limit_distance)

        resp = self._request("POST", "/positions/otc", version="2", payload=payload)
        deal_reference = resp.get("dealReference", "")
        logger.info("Deal reference received: %s", deal_reference)

        return self._poll_confirmation(deal_reference)

    def close_position(self, deal_id: str, direction: str, size: float, epic: str) -> DealConfirmation:
        """Close an existing position by deal ID.

        Args:
            deal_id: the deal ID of the open position.
            direction: closing direction ("BUY" to close a SELL, "SELL" to close a BUY).
            size: size to close (must match the open position size).
            epic: instrument epic.
        """
        payload = {
            "dealId": deal_id,
            "epic": epic,
            "expiry": "-",
            "direction": direction,
            "size": str(size),
            "orderType": "MARKET",
            "timeInForce": "FILL_OR_KILL",
        }
        # IG uses a custom DELETE-with-body via _method override
        payload["_method"] = "DELETE"
        resp = self._request("POST", "/positions/otc", version="1", payload=payload)
        deal_reference = resp.get("dealReference", "")
        logger.info("Close deal reference: %s for deal_id=%s", deal_reference, deal_id)
        return self._poll_confirmation(deal_reference)

    def close_all(self, reason: str = "kill switch") -> list[DealConfirmation]:
        """Close every open position immediately.

        This is the kill switch action — called regardless of other component
        state. Logs each close attempt individually so partial failures are
        visible.

        Args:
            reason: free-text reason logged alongside each close.

        Returns:
            List of :class:`DealConfirmation` for every close attempt.
        """
        logger.warning("KILL SWITCH: closing all open positions. Reason: %s", reason)
        positions = self.get_positions()
        if not positions:
            logger.info("Kill switch triggered but no open positions found")
            return []

        results = []
        for pos in positions:
            closing_direction = "SELL" if pos.direction == "BUY" else "BUY"
            try:
                confirm = self.close_position(
                    deal_id=pos.deal_id,
                    direction=closing_direction,
                    size=pos.size,
                    epic=pos.epic,
                )
                if confirm.accepted:
                    logger.warning(
                        "Kill switch closed %s %s (deal_id=%s)",
                        pos.epic, pos.direction, pos.deal_id,
                    )
                else:
                    logger.error(
                        "Kill switch FAILED to close %s deal_id=%s: %s",
                        pos.epic, pos.deal_id, confirm.reason,
                    )
                results.append(confirm)
            except Exception as e:
                logger.error(
                    "Kill switch exception closing %s deal_id=%s: %s",
                    pos.epic, pos.deal_id, e,
                )
        return results

    # ─── Deal confirmation polling ────────────────────────────────────────────

    def _poll_confirmation(self, deal_reference: str) -> DealConfirmation:
        """Poll /confirms/{deal_reference} until a final status is returned.

        IG processes deals asynchronously — the confirmation endpoint may return
        status "PENDING" for a short period before the deal settles.

        Args:
            deal_reference: the reference returned by the order endpoint.

        Returns:
            :class:`DealConfirmation` with the final deal status.
        """
        for attempt in range(1, _CONFIRM_MAX_ATTEMPTS + 1):
            try:
                data = self._request("GET", f"/confirms/{deal_reference}", version="1")
                status = data.get("dealStatus", "PENDING")
                if status != "PENDING":
                    return DealConfirmation(
                        deal_reference=deal_reference,
                        deal_id=data.get("dealId", ""),
                        epic=data.get("epic", ""),
                        direction=data.get("direction", ""),
                        size=float(data.get("size", 0)),
                        level=float(data.get("level", 0)),
                        status=status,
                        reason=data.get("reason", ""),
                        stop_level=data.get("stopLevel"),
                        limit_level=data.get("limitLevel"),
                    )
            except IGAPIError as e:
                logger.warning("Confirmation poll attempt %d failed: %s", attempt, e)

            time.sleep(_CONFIRM_POLL_INTERVAL)

        # If we exhaust all attempts, return a rejected placeholder
        logger.error("Deal confirmation timed out for reference %s", deal_reference)
        return DealConfirmation(
            deal_reference=deal_reference,
            deal_id="",
            epic="",
            direction="",
            size=0.0,
            level=0.0,
            status="REJECTED",
            reason="CONFIRMATION_TIMEOUT",
        )
