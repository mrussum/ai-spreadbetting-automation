# Setup Guide — AI Spread Betting Trading Assistant

This document records every step taken to build and deploy this system, including all problems encountered and their solutions. It is written so that Claude Code on a new machine can replicate the setup end-to-end.

---

## System Overview

An automated spread betting assistant that:
- Downloads market data via yfinance (FTSE, GBPUSD, Gold)
- Trains an XGBoost model per instrument using 5-year daily data
- Generates BUY/SELL/HOLD signals every 15 minutes during London market hours
- Validates each signal with Claude (Haiku) as a second opinion
- Enforces risk management (daily loss limit, max positions, ATR-based stops)
- Executes paper or live trades via the IG Group REST API
- Displays a Streamlit dashboard at localhost:8501
- Defaults to PAPER mode — never risks real money unless explicitly switched

---

## Build Phases

The system was built in 6 phases. Each phase is a separate git commit on branch `claude/spread-betting-trading-bot-FZyj3`.

| Phase | Commit | What was built |
|-------|--------|----------------|
| 1 | `7023f9e` | Project structure, config, database models, data pipeline |
| 2 | `14353a2` | XGBoost trainer, SignalPredictor, vectorbt backtest |
| 3 | `3c9b1ba` | IGClient (auth + trading), ig_models dataclasses, IGStreamClient |
| 4 | `45007f2` | RiskManager, LLMAnalyst (Claude), Trader.run_cycle() orchestrator |
| 5 | `5d85b17` | APScheduler runner, Streamlit dashboard |
| 6 | `1f528ee` | Logging config, expanded tests (134 total), safety property tests |

---

## Prerequisites

- Python 3.11
- Git
- A Windows PC (commands below use PowerShell)
- An IG Group account with API demo access (see IG credentials section below)
- An Anthropic API key (console.anthropic.com)

---

## Step-by-Step Setup on a New Machine

### 1. Clone the repository

```powershell
git clone https://github.com/mrussum/ai-spreadbetting-automation.git
cd ai-spreadbetting-automation
git checkout claude/spread-betting-trading-bot-FZyj3
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
& venv\Scripts\Activate.ps1
```

If you see a script execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install dependencies

```powershell
python -m ensurepip --upgrade
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure credentials

Copy the example env file and fill in real values:
```powershell
copy .env.example .env
```

Open `.env` and set:
```
IG_API_KEY=<your IG demo API key>
IG_USERNAME=<your IG demo API username — see note below>
IG_PASSWORD=<your IG demo API password — see note below>
IG_ACC_TYPE=DEMO
IG_ACC_ID=<your IG account ID, e.g. Z69W1O>

ANTHROPIC_API_KEY=<your Anthropic API key>

TRADING_MODE=PAPER
MAX_RISK_PER_TRADE=0.01
MAX_DAILY_LOSS=0.05
MAX_OPEN_POSITIONS=3
SIGNAL_THRESHOLD=0.65
LLM_CONFIDENCE_THRESHOLD=0.70

INSTRUMENTS=IX.D.FTSE.DAILY.IP,CS.D.GBPUSD.TODAY.IP,CS.D.GOLD.TODAY.IP
```

### 5. Train the models

```powershell
python -m models.trainer
```

This downloads ~5 years of daily OHLCV data for each instrument and trains an XGBoost classifier. Takes 2–5 minutes. Saved models appear in `models/saved/`.

Expected output:
```
IX.D.FTSE.DAILY.IP: AUC=0.54xx, acc=0.50xx, rows=~1063
CS.D.GBPUSD.TODAY.IP: AUC=0.55xx, acc=0.53xx, rows=~1101
CS.D.GOLD.TODAY.IP: AUC=0.56xx, acc=0.50xx, rows=~1059
```

### 6. Start the scheduler (Terminal 1)

```powershell
cd C:\path\to\ai-spreadbetting-automation
& venv\Scripts\Activate.ps1
python -m scheduler.runner
```

The scheduler runs every 15 minutes, Monday–Friday 08:00–16:30 London time. It will wait silently outside market hours.

### 7. Start the dashboard (Terminal 2)

```powershell
cd C:\path\to\ai-spreadbetting-automation
& venv\Scripts\Activate.ps1
python -m streamlit run dashboard/app.py
```

Open http://localhost:8501 in a browser.

---

## IG Group API Credentials — Important Notes

IG has **two separate sets of credentials**:

| Type | Used for | Where to find |
|------|----------|---------------|
| Main IG login | Logging into ig.com website | Your existing IG account |
| API demo credentials | This application | labs.ig.com (set separately) |

**The API username is NOT your email address.** You must create dedicated API credentials:

1. Go to https://labs.ig.com and log in with your main IG credentials
2. Navigate to **API Demo** settings
3. Set a **username** (must be different from your live IG username — can be anything)
4. Set a **password**
5. Copy the **API key** shown on that page
6. Use these three values in your `.env` file

**Common error if you use the wrong credentials:**
```
Login failed (400): {"errorCode":"validation.pattern.invalid.authenticationRequest.identifier"}
```
This means the username/identifier is being rejected — you are using your email instead of the API-specific username, or the API key does not match.

---

## Problems Encountered and Solutions

### Problem 1: yfinance downloads failing with JSONDecodeError

**Error:**
```
Failed to get ticker '^FTSE' reason: Expecting value: line 1 column 1 (char 0)
JSONDecodeError('Expecting value: line 1 column 1 (char 0)')
Training failed for IX.D.FTSE.DAILY.IP: No data returned for ^FTSE
```

**Cause:** `requirements.txt` pinned `yfinance==0.2.36`. Yahoo Finance changed their API in late 2024 and this version is incompatible.

**Solution:** Upgrade yfinance:
```powershell
python -m ensurepip --upgrade
python -m pip install --upgrade yfinance
```
The working version is `yfinance==1.2.0`. This also installs `curl_cffi` and `websockets` as new dependencies, which are already pinned in `requirements.txt`.

---

### Problem 2: GBPUSD training produces 0 rows

**Error:**
```
Training data: 0 rows, 21 features, nan% positive
Training failed for CS.D.GBPUSD.TODAY.IP: Cannot have number of folds=6 greater than the number of samples=0
```

**Cause:** Forex pairs (GBPUSD=X) return `volume=0` for all rows from yfinance. The feature engineering code computes `volume_ratio = volume / volume_sma_20 = 0/0 = NaN`. Since `dropna()` is called after feature engineering, all rows are eliminated.

**Solution:** Already fixed in `data/features.py` — when all volume values are zero, `volume_ratio` is set to `1.0` (neutral) and `obv`/`volume_sma_20` are set to `0.0`, bypassing the division. No action needed on a fresh install.

---

### Problem 3: IG login error — wrong credentials format

**Error:**
```
CRITICAL IG login failed: Login failed (400): {"errorCode":"validation.pattern.invalid.authenticationRequest.identifier"}
```

**Cause:** Used the IG account email address as `IG_USERNAME` instead of the dedicated API demo username created at labs.ig.com.

**Solution:** Create separate API demo credentials at labs.ig.com (see IG credentials section above). Use the API-specific username (not the email) and its password in `.env`.

---

### Problem 4: `streamlit` command not found

**Error:**
```
streamlit : The term 'streamlit' is not recognized as the name of a cmdlet...
```

**Cause:** Running `streamlit run ...` without activating the venv, or running it from a terminal where the venv is not active.

**Solution:** Always run via the venv Python:
```powershell
python -m streamlit run dashboard/app.py
```
Or activate the venv first in that terminal before running `streamlit run dashboard/app.py`.

---

### Problem 5: `ModuleNotFoundError: No module named 'config'`

**Error appears in the Streamlit browser when the dashboard loads.**

**Cause:** Streamlit was launched from a directory other than the project root, so the project modules are not on `sys.path`.

**Solution:** Always `cd` to the project root before launching:
```powershell
cd C:\path\to\ai-spreadbetting-automation
python -m streamlit run dashboard/app.py
```

---

### Problem 6: pip not available in venv

**Error:**
```
No module named pip
```

**Cause:** The virtual environment was created without pip (can happen with some Python installations).

**Solution:**
```powershell
python -m ensurepip --upgrade
```
Then retry the pip command.

---

## Running Tests

```powershell
python -m pytest -v
```

Expected: **134 tests passing**. Coverage is ~69% (gaps are in network-dependent modules: `ig_stream.py`, `data/collector.py`, `scheduler/runner.py`).

---

## Safety Rules (non-negotiable, enforced in code)

1. `TRADING_MODE` defaults to `PAPER` — the system will never place real trades unless you explicitly set `TRADING_MODE=LIVE` in `.env`
2. The kill switch (`database/kill_switch` table) halts all trading independently of every other component
3. API keys and session tokens are never written to logs
4. Every trade decision is saved to the database **before** the order is sent to IG
5. If the LLM (Claude) fails for any reason, the trade is skipped — it never fails open

---

## Switching to Live Trading

When you have validated the system in PAPER mode and are ready to trade real money:

1. Change in `.env`:
   ```
   TRADING_MODE=LIVE
   IG_ACC_TYPE=LIVE
   ```
2. Update `IG_BASE_URL` in `config/settings.py` — this is handled automatically based on `IG_ACC_TYPE`
3. Create live API credentials at labs.ig.com (same process as demo)
4. Restart the scheduler

**Start with very small position sizes.** The default `MAX_RISK_PER_TRADE=0.01` risks 1% of account balance per trade.

---

## Directory Structure

```
├── config/
│   ├── settings.py          # All config loaded from .env
│   └── logging_config.py    # Centralised logging setup
├── database/
│   ├── models.py            # SQLAlchemy table definitions
│   └── crud.py              # DB read/write helpers
├── data/
│   ├── collector.py         # yfinance download + SQLite caching
│   └── features.py          # 21 technical indicators, FEATURE_COLUMNS
├── models/
│   ├── trainer.py           # XGBoost training with TimeSeriesSplit
│   ├── predictor.py         # SignalPredictor — live inference
│   └── saved/               # .pkl model files (gitignored)
├── broker/
│   ├── ig_models.py         # API response dataclasses
│   ├── ig_client.py         # IG REST API client
│   └── ig_stream.py         # Lightstreamer price feed
├── risk/
│   └── manager.py           # RiskManager — all trade gates + sizing
├── llm/
│   └── analyst.py           # LLMAnalyst — Claude trade validation
├── execution/
│   └── trader.py            # Trader.run_cycle() — full pipeline
├── scheduler/
│   └── runner.py            # APScheduler — 15min market hours loop
├── backtest/
│   └── run_backtest.py      # vectorbt walk-forward backtest
├── dashboard/
│   └── app.py               # Streamlit monitoring dashboard
├── tests/                   # 134 pytest tests
├── .env.example             # Credential template (never put real values here)
├── requirements.txt
└── README.md
```
