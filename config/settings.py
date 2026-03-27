"""Configuration loaded from environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ─── IG Group API ────────────────────────────────────────────
IG_API_KEY: str = os.getenv("IG_API_KEY", "")
IG_USERNAME: str = os.getenv("IG_USERNAME", "")
IG_PASSWORD: str = os.getenv("IG_PASSWORD", "")
IG_ACC_TYPE: str = os.getenv("IG_ACC_TYPE", "DEMO")
IG_ACC_ID: str = os.getenv("IG_ACC_ID", "")

IG_BASE_URL: str = (
    "https://demo-api.ig.com/gateway/deal"
    if IG_ACC_TYPE == "DEMO"
    else "https://api.ig.com/gateway/deal"
)

# ─── Anthropic ───────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# ─── Trading Config ─────────────────────────────────────────
TRADING_MODE: str = os.getenv("TRADING_MODE", "PAPER")
MAX_RISK_PER_TRADE: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.01"))
MAX_DAILY_LOSS: float = float(os.getenv("MAX_DAILY_LOSS", "0.05"))
MAX_OPEN_POSITIONS: int = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
SIGNAL_THRESHOLD: float = float(os.getenv("SIGNAL_THRESHOLD", "0.65"))
LLM_CONFIDENCE_THRESHOLD: float = float(os.getenv("LLM_CONFIDENCE_THRESHOLD", "0.70"))

# ─── Instruments ─────────────────────────────────────────────
INSTRUMENTS: list[str] = [
    s.strip()
    for s in os.getenv("INSTRUMENTS", "IX.D.FTSE.DAILY.IP").split(",")
    if s.strip()
]

# ─── Epic → yfinance ticker mapping ─────────────────────────
EPIC_TO_TICKER: dict[str, str] = {
    "IX.D.FTSE.DAILY.IP": "^FTSE",
    "CS.D.GBPUSD.TODAY.IP": "GBPUSD=X",
    "CS.D.GOLD.TODAY.IP": "GC=F",
}

# ─── Paths ───────────────────────────────────────────────────
DB_PATH: str = str(PROJECT_ROOT / "trading.db")
MODELS_DIR: Path = PROJECT_ROOT / "models" / "saved"
LOG_DIR: Path = PROJECT_ROOT / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Logging ─────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


def validate_config() -> None:
    """Validate that all required configuration is present.

    Raises ValueError with a descriptive message if any required
    variable is missing or invalid.
    """
    errors: list[str] = []

    if not IG_API_KEY:
        errors.append("IG_API_KEY is required")
    if not IG_USERNAME:
        errors.append("IG_USERNAME is required")
    if not IG_PASSWORD:
        errors.append("IG_PASSWORD is required")
    if not IG_ACC_ID:
        errors.append("IG_ACC_ID is required")
    if IG_ACC_TYPE not in ("DEMO", "LIVE"):
        errors.append("IG_ACC_TYPE must be DEMO or LIVE")
    if not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY is required")
    if TRADING_MODE not in ("PAPER", "LIVE"):
        errors.append("TRADING_MODE must be PAPER or LIVE")
    if not (0 < MAX_RISK_PER_TRADE <= 0.05):
        errors.append("MAX_RISK_PER_TRADE must be between 0 and 0.05")
    if not (0 < MAX_DAILY_LOSS <= 0.20):
        errors.append("MAX_DAILY_LOSS must be between 0 and 0.20")
    if not INSTRUMENTS:
        errors.append("At least one instrument must be configured")

    if errors:
        raise ValueError(
            "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )
