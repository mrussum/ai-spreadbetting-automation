"""Streamlit monitoring dashboard for the spread betting trading system."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import TRADING_MODE
from database.crud import (
    get_daily_stats,
    get_kill_switch_status,
    get_recent_decisions,
    get_recent_llm_logs,
    get_stats_history,
    set_kill_switch,
)
from database.models import init_db

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Spread Betting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure tables exist (safe to call multiple times)
init_db()

# Auto-refresh every 60 seconds
st.markdown(
    '<meta http-equiv="refresh" content="60">',
    unsafe_allow_html=True,
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Controls")

    mode_colour = "🟢" if TRADING_MODE == "PAPER" else "🔴"
    st.markdown(f"**Mode:** {mode_colour} `{TRADING_MODE}`")
    st.divider()

    # ── Kill switch ───────────────────────────────────────────────────────────
    kill_active = get_kill_switch_status()

    if kill_active:
        st.error("⛔ KILL SWITCH IS ACTIVE — all trading halted")
        if st.button("Deactivate Kill Switch", type="secondary"):
            set_kill_switch(active=False, reason="")
            st.rerun()
    else:
        st.success("✅ System running normally")
        with st.expander("Emergency stop"):
            reason = st.text_input("Reason (optional)", key="ks_reason")
            if st.button("🛑 Activate Kill Switch", type="primary"):
                set_kill_switch(active=True, reason=reason or "Manual dashboard activation")
                st.rerun()

    st.divider()
    if st.button("↻ Refresh now"):
        st.rerun()

# ─── Header ──────────────────────────────────────────────────────────────────

st.title("📈 Spread Betting — Live Monitor")

# ─── Account summary row ─────────────────────────────────────────────────────

today_stats = get_daily_stats()

col1, col2, col3, col4 = st.columns(4)

with col1:
    balance = today_stats.ending_balance if today_stats and today_stats.ending_balance else 0.0
    st.metric("Account Balance", f"£{balance:,.2f}")

with col2:
    pnl = today_stats.total_pnl if today_stats else 0.0
    pnl_pct = (pnl / today_stats.starting_balance * 100) if (today_stats and today_stats.starting_balance) else 0.0
    st.metric("Today's P&L", f"£{pnl:+,.2f}", delta=f"{pnl_pct:+.2f}%")

with col3:
    trades = today_stats.trades_taken if today_stats else 0
    won = today_stats.trades_won if today_stats else 0
    win_rate = f"{won/trades:.0%}" if trades else "—"
    st.metric("Trades Today", str(trades), delta=f"{won}W / {trades - won}L" if trades else None)

with col4:
    max_dd = today_stats.max_drawdown if today_stats else 0.0
    st.metric("Max Drawdown Today", f"{max_dd:.2f}%")

st.divider()

# ─── Main content: two columns ───────────────────────────────────────────────

left, right = st.columns([3, 2])

# ── Equity curve ─────────────────────────────────────────────────────────────

with left:
    st.subheader("Equity Curve (30 days)")
    history = get_stats_history(days=30)

    if history:
        hist_df = pd.DataFrame([
            {
                "date": str(s.date),
                "balance": s.ending_balance or s.starting_balance,
                "pnl": s.total_pnl or 0.0,
            }
            for s in reversed(history)
        ])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_df["date"],
            y=hist_df["balance"],
            mode="lines+markers",
            name="Balance",
            line=dict(color="#00cc88", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,204,136,0.1)",
        ))
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            height=280,
            xaxis_title=None,
            yaxis_title="£",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical stats yet — start trading to populate the equity curve.")

    # ── Recent decisions table ────────────────────────────────────────────────

    st.subheader("Recent Decisions")
    decisions = get_recent_decisions(limit=30)

    if decisions:
        rows = []
        for d in decisions:
            action_icon = {
                "BUY": "🟢 BUY", "SELL": "🔴 SELL",
                "PAPER_BUY": "🟦 PAPER BUY", "PAPER_SELL": "🟦 PAPER SELL",
                "SKIP": "⬜ SKIP", "BLOCKED": "⛔ BLOCKED",
            }.get(d.action_taken, d.action_taken)

            rows.append({
                "Time": str(d.timestamp)[:16],
                "Epic": d.epic,
                "Action": action_icon,
                "ML": f"{d.ml_signal} {d.ml_confidence:.0%}",
                "LLM": f"{d.llm_action} {d.llm_confidence:.0%}" if d.llm_action else "—",
                "Risk": "✓" if d.risk_checks_passed == "True" else "✗",
                "Mode": d.trading_mode,
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            height=280,
        )
    else:
        st.info("No decisions recorded yet.")

# ── LLM reasoning cards ───────────────────────────────────────────────────────

with right:
    st.subheader("LLM Reasoning (latest 5)")
    logs = get_recent_llm_logs(limit=5)

    if logs:
        for log in logs:
            # Extract action from the response JSON if possible
            import json
            action_label = "—"
            try:
                data = json.loads(log.response_text)
                action_label = data.get("action", "—")
                reasoning = data.get("reasoning", log.response_text[:200])
                confidence = data.get("confidence", None)
            except Exception:
                reasoning = log.response_text[:200]
                confidence = None

            colour = {"BUY": "green", "SELL": "red", "SKIP": "gray"}.get(action_label, "gray")
            badge = f":{colour}[**{action_label}**]"
            conf_str = f" ({confidence:.0%})" if confidence is not None else ""

            with st.expander(f"{str(log.timestamp)[:16]} — {log.epic} — {action_label}{conf_str}"):
                st.markdown(f"**Decision:** {badge}{conf_str}")
                st.markdown(f"**Reasoning:** {reasoning}")
                st.caption(f"Tokens: {log.tokens_used} | Latency: {log.latency_ms}ms")
    else:
        st.info("No LLM logs yet.")

    # ── Daily P&L bar chart ───────────────────────────────────────────────────

    st.subheader("Daily P&L (30 days)")
    history = get_stats_history(days=30)

    if history:
        bar_df = pd.DataFrame([
            {"date": str(s.date), "pnl": s.total_pnl or 0.0}
            for s in reversed(history)
        ])
        colours = ["#00cc88" if v >= 0 else "#ff4444" for v in bar_df["pnl"]]

        fig2 = go.Figure(go.Bar(
            x=bar_df["date"],
            y=bar_df["pnl"],
            marker_color=colours,
        ))
        fig2.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            height=220,
            xaxis_title=None,
            yaxis_title="£",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No P&L history yet.")

# ─── Footer ──────────────────────────────────────────────────────────────────

st.divider()
from datetime import datetime
import pytz
london = pytz.timezone("Europe/London")
now_london = datetime.now(london).strftime("%Y-%m-%d %H:%M:%S %Z")
st.caption(f"Last rendered: {now_london} | Mode: {TRADING_MODE} | Auto-refresh: 60s")
