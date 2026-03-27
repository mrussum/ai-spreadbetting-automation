"""IG Group Lightstreamer streaming client for real-time price subscriptions."""

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Lightstreamer client is an optional dependency — import lazily so the rest
# of the system works without it installed (e.g. during backtesting).
try:
    from lightstreamer.client import LightstreamerClient, Subscription
    _LS_AVAILABLE = True
except ImportError:
    _LS_AVAILABLE = False
    logger.debug("lightstreamer-client not installed; IGStreamClient will be unavailable")

from broker.ig_models import StreamPrice

# Callback type: receives a StreamPrice whenever a tick arrives
PriceCallback = Callable[[StreamPrice], None]


class IGStreamClient:
    """Lightstreamer streaming client for IG live price feeds.

    Subscribes to MERGE-mode price items for a list of epics and calls a
    user-supplied callback on each tick.

    Usage::

        def on_price(p: StreamPrice):
            print(p.epic, p.bid, p.offer)

        stream = IGStreamClient(
            cst="...",
            security_token="...",
            lightstreamer_endpoint="https://push.lightstreamer.com",
            callback=on_price,
        )
        stream.connect(["IX.D.FTSE.DAILY.IP"])
        # ... stream runs in background thread ...
        stream.disconnect()

    Note:
        Requires ``lightstreamer-client-python`` to be installed.
        The IG Lightstreamer endpoint is returned in the /session response
        (``lightstreamerEndpoint`` field).
    """

    def __init__(
        self,
        cst: str,
        security_token: str,
        lightstreamer_endpoint: str,
        callback: PriceCallback,
    ) -> None:
        if not _LS_AVAILABLE:
            raise ImportError(
                "lightstreamer-client-python is required for streaming. "
                "Install it with: pip install lightstreamer-client-python"
            )

        self._cst = cst
        self._security_token = security_token
        self._endpoint = lightstreamer_endpoint
        self._callback = callback
        self._client: Optional["LightstreamerClient"] = None
        self._subscription: Optional["Subscription"] = None
        self._connected = False
        self._lock = threading.Lock()

    # ─── Lifecycle ───────────────────────────────────────────────────────────

    def connect(self, epics: list[str]) -> None:
        """Connect to Lightstreamer and subscribe to price feeds.

        Args:
            epics: list of IG instrument epics to subscribe to.
        """
        with self._lock:
            if self._connected:
                logger.warning("IGStreamClient already connected")
                return

            logger.info(
                "Connecting to Lightstreamer at %s for %d epics",
                self._endpoint, len(epics),
            )

            # IG uses the CST+SecurityToken as Lightstreamer password
            self._client = LightstreamerClient(self._endpoint, "DEFAULT")
            self._client.connectionDetails.setUser(self._cst)
            self._client.connectionDetails.setPassword(
                f"CST-{self._cst}|XST-{self._security_token}"
            )

            self._subscription = self._build_subscription(epics)
            self._client.subscribe(self._subscription)
            self._client.connect()
            self._connected = True
            logger.info("IGStreamClient connected and subscribed to %s", epics)

    def disconnect(self) -> None:
        """Unsubscribe and close the Lightstreamer connection."""
        with self._lock:
            if not self._connected or self._client is None:
                return
            try:
                if self._subscription is not None:
                    self._client.unsubscribe(self._subscription)
                self._client.disconnect()
            except Exception as e:
                logger.warning("Error during IGStreamClient disconnect: %s", e)
            finally:
                self._connected = False
                logger.info("IGStreamClient disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ─── Subscription setup ──────────────────────────────────────────────────

    def _build_subscription(self, epics: list[str]) -> "Subscription":
        """Build a MERGE subscription for bid/offer on the given epics.

        IG Lightstreamer items are prefixed with "MARKET:" and fields are
        BID, OFFER, UPDATE_TIME.
        """
        items = [f"MARKET:{epic}" for epic in epics]
        fields = ["BID", "OFFER", "UPDATE_TIME"]

        sub = Subscription(
            mode="MERGE",
            items=items,
            fields=fields,
        )
        sub.addlistener(_PriceListener(epics, self._callback))
        return sub


class _PriceListener:
    """Internal Lightstreamer subscription listener.

    Translates raw Lightstreamer field updates into :class:`StreamPrice`
    objects and forwards them to the user callback.
    """

    def __init__(self, epics: list[str], callback: PriceCallback) -> None:
        # Map "MARKET:{epic}" → epic for quick lookup
        self._item_to_epic = {f"MARKET:{e}": e for e in epics}
        self._callback = callback

    def onItemUpdate(self, update) -> None:  # noqa: N802 — Lightstreamer naming convention
        """Called by Lightstreamer on each field update."""
        item_name = update.getItemName()
        epic = self._item_to_epic.get(item_name, item_name)

        try:
            bid_raw = update.getValue("BID")
            offer_raw = update.getValue("OFFER")
            update_time = update.getValue("UPDATE_TIME") or ""

            if bid_raw is None or offer_raw is None:
                return

            bid = float(bid_raw)
            offer = float(offer_raw)
            mid = (bid + offer) / 2

            price = StreamPrice(
                epic=epic,
                bid=bid,
                offer=offer,
                mid=mid,
                update_time=update_time,
            )
            self._callback(price)
        except (ValueError, TypeError) as e:
            logger.debug("Could not parse stream update for %s: %s", epic, e)

    def onSubscriptionError(self, code, message) -> None:  # noqa: N802
        logger.error("Lightstreamer subscription error %s: %s", code, message)

    def onEndOfSnapshot(self, item_name, item_pos) -> None:  # noqa: N802
        logger.debug("End of snapshot for %s", item_name)
