"""LLM trade validation using Claude — fail-safe defaults to SKIP."""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import anthropic

from config.settings import ANTHROPIC_API_KEY, LLM_CONFIDENCE_THRESHOLD
from database.crud import save_llm_log

logger = logging.getLogger(__name__)

# Model to use for trade analysis
_MODEL = "claude-haiku-4-5-20251001"

# Hard cap on tokens to keep latency and cost predictable
_MAX_TOKENS = 512

_SYSTEM_PROMPT = """You are a disciplined quantitative trading risk analyst reviewing \
AI-generated trade signals for a spread betting account.

Your job is to validate or reject a proposed trade based on the market data provided. \
Be concise and conservative — when in doubt, SKIP.

You must respond with valid JSON only, in exactly this format:
{
  "action": "BUY" | "SELL" | "SKIP",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<one or two sentences>"
}

Rules:
- Only recommend BUY or SELL if the evidence clearly supports it.
- Return SKIP if conditions are ambiguous, contradictory, or high risk.
- Never recommend a position size or specific price levels.
- Respond with JSON only — no markdown, no preamble."""


def _build_prompt(
    epic: str,
    signal: str,
    ml_confidence: float,
    features: dict,
    open_positions: int,
    daily_pnl: float,
) -> str:
    """Construct the user-turn prompt for the trade review."""
    # Select the most interpretable features for the LLM
    readable = {
        "RSI (14)": round(features.get("rsi_14", 0), 1),
        "MACD histogram": round(features.get("macd_hist", 0), 4),
        "EMA 20 vs EMA 50": "above" if features.get("ema_20", 0) > features.get("ema_50", 0) else "below",
        "Bollinger band width": round(features.get("bb_width", 0), 4),
        "ATR (14)": round(features.get("atr_14", 0), 2),
        "Stochastic K": round(features.get("stoch_k", 0), 1),
        "Volume ratio vs 20d avg": round(features.get("volume_ratio", 0), 2),
        "1-day return": f"{features.get('returns_1', 0):.2%}",
        "5-day return": f"{features.get('returns_5', 0):.2%}",
    }

    feature_lines = "\n".join(f"  {k}: {v}" for k, v in readable.items())

    return (
        f"Instrument: {epic}\n"
        f"ML signal: {signal} (confidence: {ml_confidence:.1%})\n"
        f"Open positions: {open_positions}\n"
        f"Today's realised P&L: £{daily_pnl:+.2f}\n\n"
        f"Key technical indicators:\n{feature_lines}\n\n"
        f"Should this trade be executed? Respond with JSON only."
    )


@dataclass
class LLMDecision:
    """Parsed response from the LLM analyst."""

    action: str          # "BUY", "SELL", or "SKIP"
    confidence: float    # 0.0 – 1.0
    reasoning: str
    tokens_used: int
    latency_ms: int


class LLMAnalyst:
    """Calls Claude to validate a proposed trade signal.

    On any failure (network error, bad JSON, low confidence, timeout) the
    analyst returns SKIP — it never fails open to a trade.

    Usage::

        analyst = LLMAnalyst()
        decision = analyst.analyse(
            epic="IX.D.FTSE.DAILY.IP",
            signal="BUY",
            ml_confidence=0.72,
            features={...},
            open_positions=1,
            daily_pnl=-50.0,
        )
        if decision.action == "BUY" and decision.confidence >= 0.70:
            # execute trade
    """

    def __init__(self) -> None:
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY must be set to use LLMAnalyst")
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def analyse(
        self,
        epic: str,
        signal: str,
        ml_confidence: float,
        features: dict,
        open_positions: int = 0,
        daily_pnl: float = 0.0,
    ) -> LLMDecision:
        """Request a trade validation from Claude.

        Always returns a :class:`LLMDecision`. Returns SKIP on any error.

        Args:
            epic: IG instrument epic.
            signal: ML signal direction ("BUY" or "SELL").
            ml_confidence: model confidence (0–1).
            features: dict of feature name → value (from SignalResult.features_snapshot).
            open_positions: number of currently open positions.
            daily_pnl: today's cumulative realised P&L in £.

        Returns:
            :class:`LLMDecision` — check `.action` and `.confidence`.
        """
        prompt = _build_prompt(epic, signal, ml_confidence, features, open_positions, daily_pnl)
        start = time.monotonic()

        try:
            response = self._client.messages.create(
                model=_MODEL,
                max_tokens=_MAX_TOKENS,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            logger.error("LLM API call failed for %s: %s — defaulting to SKIP", epic, e)
            return self._skip(reason=f"API error: {e}", latency_ms=self._elapsed_ms(start))

        latency_ms = self._elapsed_ms(start)
        raw_text = response.content[0].text if response.content else ""
        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        decision = self._parse_response(raw_text, latency_ms, tokens_used)

        # Enforce LLM confidence threshold — low-confidence answers become SKIP
        if decision.action in ("BUY", "SELL") and decision.confidence < LLM_CONFIDENCE_THRESHOLD:
            logger.info(
                "LLM confidence %.2f below threshold %.2f for %s — overriding to SKIP",
                decision.confidence, LLM_CONFIDENCE_THRESHOLD, epic,
            )
            decision = LLMDecision(
                action="SKIP",
                confidence=decision.confidence,
                reasoning=(
                    f"Confidence {decision.confidence:.1%} below threshold "
                    f"{LLM_CONFIDENCE_THRESHOLD:.1%}. Original: {decision.reasoning}"
                ),
                tokens_used=tokens_used,
                latency_ms=latency_ms,
            )

        # Persist to DB regardless of outcome
        self._log(epic, prompt, raw_text, tokens_used, latency_ms)

        logger.info(
            "LLM decision for %s: %s (confidence=%.2f, latency=%dms)",
            epic, decision.action, decision.confidence, latency_ms,
        )
        return decision

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _parse_response(self, text: str, latency_ms: int, tokens_used: int) -> LLMDecision:
        """Parse the JSON response from Claude. Returns SKIP on any parse failure."""
        try:
            # Strip any accidental markdown fencing
            clean = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(clean)

            action = str(data.get("action", "SKIP")).upper()
            if action not in ("BUY", "SELL", "SKIP"):
                raise ValueError(f"Invalid action: {action!r}")

            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))

            reasoning = str(data.get("reasoning", ""))[:500]

            return LLMDecision(
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
            )
        except Exception as e:
            logger.warning("Failed to parse LLM response: %s | raw=%r", e, text[:200])
            return self._skip(reason=f"Parse error: {e}", latency_ms=latency_ms, tokens_used=tokens_used)

    def _skip(
        self,
        reason: str,
        latency_ms: int = 0,
        tokens_used: int = 0,
    ) -> LLMDecision:
        return LLMDecision(
            action="SKIP",
            confidence=0.0,
            reasoning=reason,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        return int((time.monotonic() - start) * 1000)

    def _log(
        self,
        epic: str,
        prompt: str,
        response: str,
        tokens: int,
        latency_ms: int,
    ) -> None:
        try:
            save_llm_log({
                "epic": epic,
                "prompt_text": prompt,
                "response_text": response,
                "tokens_used": tokens,
                "latency_ms": latency_ms,
            })
        except Exception as e:
            logger.warning("Failed to save LLM log: %s", e)
