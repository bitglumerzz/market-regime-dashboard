"""Trading-bot skeleton.

Architecture (the 5-stage design from the video):
  1. Brain         — the regime model (we reuse the dashboard's HMM)
  2. Strategy      — maps (regime, confidence) → target_position [0..1]
  3. Safety        — circuit breakers; if any trips, bot goes to cash
  4. Broker        — pluggable adapter; default is MockBroker (logs only)
  5. State / log   — what the bot did and why, for the dashboard

This file deliberately doesn't talk to a real broker. To wire up Alpaca
paper trading, implement an AlpacaBroker class with the same interface as
MockBroker and inject it via `Bot(broker=...)`.

NOTHING here places real orders. Even the "buy" call only mutates
in-memory state. The user must replace MockBroker with a real adapter
AND review their key management before any real money flows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Protocol


# ---------------------------------------------------------------------------
# Configurable defaults
# ---------------------------------------------------------------------------
DEFAULT_LONG_REGIMES: tuple[str, ...] = ("Low Vol", "Medium-Low Vol", "Medium Vol")
DEFAULT_CONFIDENCE_GATE: float = 0.60
DEFAULT_DAILY_LOSS_LIMIT: float = 0.02      # 2% in a single day → stop
DEFAULT_MAX_DRAWDOWN_LIMIT: float = 0.10    # 10% peak-to-trough → stop


# ---------------------------------------------------------------------------
# Broker interface
# ---------------------------------------------------------------------------
class BrokerProtocol(Protocol):
    """Minimal interface a broker adapter must implement."""

    def name(self) -> str: ...
    def is_live(self) -> bool: ...
    def get_equity(self) -> float: ...
    def get_position(self, ticker: str) -> float: ...
    def submit_target_position(
        self, ticker: str, target_fraction: float, price_hint: float,
    ) -> dict: ...


@dataclass
class MockBroker:
    """No-op broker that just tracks state in-memory and logs orders.

    Use this for testing and demos. Replace with a real adapter (Alpaca,
    IBKR, etc.) before connecting to real money.
    """
    starting_equity: float = 100_000.0
    equity: float = field(init=False)
    positions: dict[str, float] = field(default_factory=dict)
    last_prices: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.equity = self.starting_equity

    def name(self) -> str:
        return "MockBroker"

    def is_live(self) -> bool:
        return False

    def get_equity(self) -> float:
        # Mark-to-market with the last submitted price.
        equity = self.equity
        for tk, qty in self.positions.items():
            equity += qty * self.last_prices.get(tk, 0.0)
        return float(equity)

    def get_position(self, ticker: str) -> float:
        return float(self.positions.get(ticker, 0.0))

    def submit_target_position(
        self, ticker: str, target_fraction: float, price_hint: float,
    ) -> dict:
        target_fraction = max(0.0, min(1.0, float(target_fraction)))
        current_equity = self.get_equity()
        target_dollars = current_equity * target_fraction
        target_qty = target_dollars / price_hint if price_hint > 0 else 0.0
        current_qty = self.positions.get(ticker, 0.0)
        delta_qty = target_qty - current_qty
        # Settle the cash leg.
        self.equity -= delta_qty * price_hint
        self.positions[ticker] = target_qty
        self.last_prices[ticker] = price_hint
        return {
            "ticker": ticker,
            "side": "buy" if delta_qty > 0 else "sell" if delta_qty < 0 else "hold",
            "delta_qty": float(delta_qty),
            "new_qty": float(target_qty),
            "price": float(price_hint),
            "target_fraction": target_fraction,
        }


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
def decide_target_position(
    regime: str,
    confidence: float,
    *,
    long_regimes: Iterable[str] = DEFAULT_LONG_REGIMES,
    confidence_gate: float = DEFAULT_CONFIDENCE_GATE,
    sizing: str = "binary",
) -> tuple[float, str]:
    """Map (regime, confidence) → target position fraction [0..1].

    Returns (target_fraction, rationale_string).
    """
    long_set = set(long_regimes)
    if regime not in long_set:
        return 0.0, f"regime {regime!r} not in long set"
    if confidence < confidence_gate:
        return 0.0, (
            f"confidence {confidence:.0%} below gate "
            f"{confidence_gate:.0%}"
        )
    if sizing == "binary":
        return 1.0, f"binary long in {regime} at {confidence:.0%}"
    if sizing == "confidence":
        return float(confidence), f"confidence-sized long in {regime}"
    raise ValueError(f"Unknown sizing: {sizing!r}")


# ---------------------------------------------------------------------------
# Safety / circuit breakers
# ---------------------------------------------------------------------------
@dataclass
class SafetyState:
    """Tracks circuit-breaker conditions."""
    starting_equity: float
    peak_equity: float
    daily_anchor_equity: float
    daily_anchor_date: str
    tripped_daily: bool = False
    tripped_drawdown: bool = False
    tripped_manual: bool = False

    @property
    def any_tripped(self) -> bool:
        return self.tripped_daily or self.tripped_drawdown or self.tripped_manual

    def status(self) -> dict[str, bool]:
        return {
            "daily": self.tripped_daily,
            "drawdown": self.tripped_drawdown,
            "manual": self.tripped_manual,
        }


def evaluate_safety(
    state: SafetyState, current_equity: float, today_iso: str,
    *,
    daily_loss_limit: float = DEFAULT_DAILY_LOSS_LIMIT,
    max_drawdown_limit: float = DEFAULT_MAX_DRAWDOWN_LIMIT,
) -> SafetyState:
    """Update safety state and trip breakers if limits exceeded."""
    # Roll the daily anchor at the first call of a new date.
    if today_iso != state.daily_anchor_date:
        state.daily_anchor_date = today_iso
        state.daily_anchor_equity = current_equity
        state.tripped_daily = False  # daily breaker resets at session start

    # Track peak.
    if current_equity > state.peak_equity:
        state.peak_equity = current_equity

    # Daily loss check.
    if state.daily_anchor_equity > 0:
        daily_pnl = (current_equity / state.daily_anchor_equity) - 1.0
        if daily_pnl <= -daily_loss_limit:
            state.tripped_daily = True

    # Drawdown check.
    if state.peak_equity > 0:
        dd = (current_equity / state.peak_equity) - 1.0
        if dd <= -max_drawdown_limit:
            state.tripped_drawdown = True

    return state


# ---------------------------------------------------------------------------
# Bot orchestration
# ---------------------------------------------------------------------------
@dataclass
class BotDecision:
    """One decision made by the bot."""
    timestamp: str
    ticker: str
    regime: str
    confidence: float
    target_fraction: float
    rationale: str
    order: dict           # what was sent to the broker
    safety: dict[str, bool]
    equity_after: float


@dataclass
class Bot:
    """Lightweight bot: feed it (regime, confidence, price), it acts."""
    ticker: str
    broker: BrokerProtocol = field(default_factory=MockBroker)  # type: ignore[arg-type]
    long_regimes: tuple[str, ...] = DEFAULT_LONG_REGIMES
    confidence_gate: float = DEFAULT_CONFIDENCE_GATE
    sizing: str = "binary"
    daily_loss_limit: float = DEFAULT_DAILY_LOSS_LIMIT
    max_drawdown_limit: float = DEFAULT_MAX_DRAWDOWN_LIMIT
    safety: SafetyState | None = None
    log: list[BotDecision] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.safety is None:
            eq = self.broker.get_equity()
            today = datetime.utcnow().date().isoformat()
            self.safety = SafetyState(
                starting_equity=eq, peak_equity=eq,
                daily_anchor_equity=eq, daily_anchor_date=today,
            )

    def trip_manual(self, on: bool = True) -> None:
        assert self.safety is not None
        self.safety.tripped_manual = on

    def step(
        self, regime: str, confidence: float, price: float,
        *, today_iso: str | None = None,
    ) -> BotDecision:
        """Process one signal tick and submit a target position."""
        assert self.safety is not None
        today_iso = today_iso or datetime.utcnow().date().isoformat()

        current_equity = self.broker.get_equity()
        self.safety = evaluate_safety(
            self.safety, current_equity, today_iso,
            daily_loss_limit=self.daily_loss_limit,
            max_drawdown_limit=self.max_drawdown_limit,
        )

        if self.safety.any_tripped:
            target = 0.0
            rationale = (
                "safety tripped: "
                + ", ".join(k for k, v in self.safety.status().items() if v)
            )
        else:
            target, rationale = decide_target_position(
                regime, confidence,
                long_regimes=self.long_regimes,
                confidence_gate=self.confidence_gate,
                sizing=self.sizing,
            )

        order = self.broker.submit_target_position(self.ticker, target, price)
        decision = BotDecision(
            timestamp=datetime.utcnow().isoformat(timespec="seconds"),
            ticker=self.ticker, regime=regime, confidence=confidence,
            target_fraction=target, rationale=rationale,
            order=order, safety=self.safety.status(),
            equity_after=self.broker.get_equity(),
        )
        self.log.append(decision)
        return decision
