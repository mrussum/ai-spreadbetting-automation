"""APScheduler-based trading loop — runs every 15 min during London market hours."""

import logging
import signal
import sys
from datetime import datetime

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import EPIC_TO_TICKER, TRADING_MODE
from database.models import init_db
from broker.ig_client import IGClient, IGAuthError
from execution.trader import Trader
from llm.analyst import LLMAnalyst
from models.predictor import SignalPredictor

logger = logging.getLogger(__name__)

LONDON_TZ = pytz.timezone("Europe/London")

# Market hours: Monday–Friday 08:00–16:30 London time
# APScheduler cron: run every 15 min from :00 to :15 to :30 to :45
# The job itself checks whether we're within the trading window before acting.
MARKET_OPEN_HOUR = 8
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 30


def _within_market_hours() -> bool:
    """Return True if the current London time is within Mon–Fri 08:00–16:30."""
    now = datetime.now(LONDON_TZ)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    open_time = now.replace(hour=MARKET_OPEN_HOUR, minute=0, second=0, microsecond=0)
    close_time = now.replace(
        hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0
    )
    return open_time <= now <= close_time


class TradingRunner:
    """Manages the APScheduler lifecycle and per-cycle orchestration.

    Instantiate once at startup, call :meth:`start` to block until shutdown.

    All components (IGClient, LLMAnalyst, predictors) are initialised once
    and shared across cycles to avoid repeated auth round-trips.
    """

    def __init__(self) -> None:
        self._scheduler = BlockingScheduler(timezone=LONDON_TZ)
        self._ig: IGClient | None = None
        self._analyst: LLMAnalyst | None = None
        self._trader: Trader | None = None
        self._running = False

    # ─── Lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialise everything, schedule the job, and block until shutdown."""
        logger.info("Trading runner starting [mode=%s]", TRADING_MODE)

        init_db()
        self._setup_components()
        self._register_job()
        self._register_signal_handlers()

        logger.info(
            "Scheduler started — running every 15 min, Mon–Fri %02d:00–%02d:%02d London",
            MARKET_OPEN_HOUR, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
        )
        self._running = True
        try:
            self._scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        """Gracefully stop the scheduler and log out of IG."""
        if not self._running:
            return
        self._running = False
        logger.info("Shutting down trading runner...")
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        if self._ig:
            try:
                self._ig.logout()
            except Exception as e:
                logger.warning("IG logout error during shutdown: %s", e)
        logger.info("Shutdown complete.")

    # ─── Setup ───────────────────────────────────────────────────────────────

    def _setup_components(self) -> None:
        """Authenticate with IG, warm up predictors and analyst."""
        self._ig = IGClient()
        try:
            self._ig.login()
        except IGAuthError as e:
            logger.critical("IG login failed: %s", e)
            sys.exit(1)

        self._analyst = LLMAnalyst()

        # Pre-load a predictor for each configured instrument
        predictors: dict[str, SignalPredictor] = {}
        for epic in EPIC_TO_TICKER:
            try:
                predictors[epic] = SignalPredictor(epic)
                logger.info("Loaded predictor for %s", epic)
            except FileNotFoundError:
                logger.warning(
                    "No trained model for %s — skipping. Run `python -m models.trainer` first.",
                    epic,
                )

        if not predictors:
            logger.critical("No predictors loaded — nothing to trade. Train models first.")
            sys.exit(1)

        self._trader = Trader(
            ig_client=self._ig,
            analyst=self._analyst,
            predictors=predictors,
        )

    def _register_job(self) -> None:
        """Schedule the trading cycle to run every 15 minutes."""
        trigger = CronTrigger(
            day_of_week="mon-fri",
            hour="8-16",
            minute="0,15,30,45",
            timezone=LONDON_TZ,
        )
        self._scheduler.add_job(
            func=self._run_cycle,
            trigger=trigger,
            id="trading_cycle",
            name="Trading cycle (all instruments)",
            max_instances=1,          # never overlap
            coalesce=True,            # skip missed fires rather than catching up
            misfire_grace_time=60,    # allow up to 60s late start
        )

    def _register_signal_handlers(self) -> None:
        """Handle SIGTERM and SIGINT for clean Docker/systemd shutdown."""
        def _handle(signum, frame):
            logger.info("Signal %d received — initiating shutdown", signum)
            self._shutdown()
            sys.exit(0)

        signal.signal(signal.SIGTERM, _handle)
        signal.signal(signal.SIGINT, _handle)

    # ─── Cycle ───────────────────────────────────────────────────────────────

    def _run_cycle(self) -> None:
        """Execute one trading cycle across all instruments.

        Called by APScheduler. Skips silently outside market hours (the cron
        trigger already limits to Mon–Fri 08:xx–16:xx but the close-minute
        boundary needs a runtime check).
        """
        if not _within_market_hours():
            logger.debug("Outside market hours — cycle skipped")
            return

        now = datetime.now(LONDON_TZ).strftime("%Y-%m-%d %H:%M")
        logger.info("─── Cycle at %s ───", now)

        for epic in list(self._trader._predictors.keys()):
            try:
                outcome = self._trader.run_cycle(epic)
                logger.info(
                    "%-30s → %-12s ml=%-4s conf=%.2f risk=%s llm=%s",
                    epic,
                    outcome.action_taken,
                    outcome.ml_signal,
                    outcome.ml_confidence,
                    "✓" if outcome.risk_passed else "✗",
                    outcome.llm_action or "-",
                )
            except Exception as e:
                # Never let a single instrument crash the whole cycle
                logger.error("Unhandled error in cycle for %s: %s", epic, e, exc_info=True)


def main() -> None:
    """Entry point: configure logging and start the runner."""
    import logging.config
    from config.settings import LOG_LEVEL, LOG_DIR

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "runner.log", encoding="utf-8"),
        ],
    )
    runner = TradingRunner()
    runner.start()


if __name__ == "__main__":
    main()
