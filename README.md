# 🤖 Trading Assistant — AI-Powered Spread Betting System

> ⚠️ **Important:** This system defaults to PAPER MODE.
> Run on demo for a minimum of 90 days before considering
> live trading. Spread betting carries significant risk of loss.

---

## What This Does

An automated spread betting assistant for IG Group that:
- Analyses live market data using technical indicators
- Generates trade signals using a trained XGBoost model
- Validates every signal through a Claude AI reasoning layer
- Enforces strict risk management rules before any execution
- Runs continuously during UK market hours on a scheduler
- Provides a live Streamlit dashboard with full performance analytics

---

## ⚠️ Risk Warning

Spread betting is a leveraged product. The majority of retail spread betters lose money. This software is for educational purposes. Never trade with money you cannot afford to lose. Always use the DEMO account and PAPER trading mode first.

---

## Prerequisites

- Python 3.11 or later
- An IG Group account (start with a FREE demo: ig.com)
- An Anthropic API account (console.anthropic.com)
- Git

---

## Quick Start

### 1. Clone and Install
```bash
git clone <your-repo-url>
cd trading_assistant
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. Set Up Your Environment
```bash
cp .env.example .env
```

Open `.env` in any text editor and fill in:
- Your IG demo API key (from ig.com → MyIG → API)
- Your IG username and password
- Your IG account ID (shown in top-right corner of IG platform)
- Your Anthropic API key (from console.anthropic.com)

Leave `TRADING_MODE=PAPER` and `IG_ACC_TYPE=DEMO` — do not change these until you have completed 90 days of paper trading.

### 3. Download Historical Data and Train the Model
```bash
# Download 5 years of data for all configured instruments
python -m data.collector

# Engineer features and train XGBoost models
python -m models.trainer

# Review backtest results (do not proceed until results look healthy)
python -m backtest.run_backtest
```

**Before proceeding, check:**
- All model folds show accuracy > 52%
- Backtest shows 200+ trades
- Sharpe ratio > 0.5
- Maximum drawdown < 30%

### 4. Run a Single Test Cycle (Paper Mode)
```bash
python -m execution.trader --test-cycle --epic IX.D.FTSE.DAILY.IP
```

You should see a full cycle log: data → features → ML signal → LLM validation → risk check → "PAPER TRADE" logged to database. No real trades are placed.

### 5. Launch the Dashboard
```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

### 6. Start the Scheduler
```bash
python -m scheduler.runner
```

The scheduler runs every 15 minutes between 08:00–16:30, Monday to Friday (Europe/London time). All decisions are logged to the database and visible in the dashboard.

---

## Project Structure

```
trading_assistant/
├── config/         Configuration and environment variables
├── data/           Market data collection and feature engineering
├── models/         XGBoost model training and prediction
├── broker/         IG Group API client (REST + streaming)
├── risk/           Risk management rules and kill switch
├── llm/            Claude AI trade validation layer
├── execution/      Trade orchestration pipeline
├── scheduler/      APScheduler job runner
├── dashboard/      Streamlit monitoring dashboard
├── database/       SQLite database models and helpers
├── backtest/       Historical strategy backtesting
└── tests/          Pytest test suite
```

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/test_risk.py -v        # Risk rules only
pytest tests/test_features.py -v    # Feature pipeline only
```

---

## Retraining the Model

Retrain monthly or when performance degrades:
```bash
python -m data.collector --refresh  # Update historical data
python -m models.trainer            # Retrain all models
python -m backtest.run_backtest     # Validate before redeploying
```

---

## Safety Checklist Before Going Live

Complete every item before changing TRADING_MODE to LIVE:

- [ ] Running on DEMO account for 90+ days
- [ ] Paper trading results match backtest expectations
- [ ] All pytest tests passing
- [ ] Kill switch tested and confirmed working
- [ ] Dashboard showing correct data
- [ ] Daily loss limit tested (simulate a 5% loss day)
- [ ] Maximum position count limit tested
- [ ] You fully understand every indicator in the model
- [ ] You have read IG's API terms of service
- [ ] You have spoken to a financial adviser

---

## Emergency: Kill Switch

If something goes wrong, you have three ways to stop everything:

1. **Dashboard:** Click the red KILL SWITCH button (Tab: Live Overview)
2. **Command line:** `python -m risk.manager --kill-switch`
3. **Manual:** Log into IG platform directly and close positions

The kill switch closes all open positions immediately and stops the scheduler. It works independently of all other system components.

---

## Environment Variables Reference

| Variable | Description | Default |
|---|---|---|
| IG_API_KEY | Your IG API key | required |
| IG_USERNAME | Your IG username | required |
| IG_PASSWORD | Your IG password | required |
| IG_ACC_TYPE | DEMO or LIVE | DEMO |
| IG_ACC_ID | Your IG account ID | required |
| ANTHROPIC_API_KEY | Anthropic API key | required |
| TRADING_MODE | PAPER or LIVE | PAPER |
| MAX_RISK_PER_TRADE | Max % of account per trade | 0.01 (1%) |
| MAX_DAILY_LOSS | Stop trading if down this % | 0.05 (5%) |
| MAX_OPEN_POSITIONS | Max simultaneous positions | 3 |
| SIGNAL_THRESHOLD | Min ML confidence to signal | 0.65 |
| LLM_CONFIDENCE_THRESHOLD | Min Claude confidence to execute | 0.70 |
| INSTRUMENTS | Comma-separated IG epic codes | FTSE, GBPUSD, Gold |

---

## Disclaimer

This software is provided for educational purposes only. It does not constitute financial advice. Past performance does not guarantee future results. The authors accept no responsibility for financial losses incurred through use of this software.
