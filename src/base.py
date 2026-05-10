from __future__ import annotations
from collections import deque
from datamodel import Order, OrderDepth, TradingState


# ──────────────────────────────────────────────────────────────────────────────
# Rolling statistics — pure Python, no pandas dependency
# ──────────────────────────────────────────────────────────────────────────────

class RollingWindow:
    """
    Fixed-length rolling window with O(1) append and O(n) mean/std.

    Used for z-score-based pair trading and basket arbitrage.
    State is serialised as a plain list so it can be stored in traderData JSON.

    Lesson from nicolassinott: they used pandas Series with a 200-period window
    and a 5-period short-term smoother. We replicate this in pure Python.
    """

    def __init__(self, maxlen: int) -> None:
        self._maxlen = maxlen
        self._data: deque[float] = deque(maxlen=maxlen)

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_list(self) -> list[float]:
        return list(self._data)

    @classmethod
    def from_list(cls, data: list[float], maxlen: int) -> "RollingWindow":
        w = cls(maxlen)
        w._data = deque(data[-maxlen:], maxlen=maxlen)
        return w

    # ── Mutations ──────────────────────────────────────────────────────────

    def push(self, value: float) -> None:
        self._data.append(value)

    # ── Queries ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._data)

    def is_ready(self) -> bool:
        return len(self._data) == self._maxlen

    def mean(self) -> float:
        n = len(self._data)
        if n == 0:
            return 0.0
        return sum(self._data) / n

    def short_mean(self, n: int = 5) -> float:
        """Mean of the most recent n values (short-term smoother, like nicolassinott's spread_5)."""
        tail = list(self._data)[-n:]
        if not tail:
            return 0.0
        return sum(tail) / len(tail)

    def std(self) -> float:
        n = len(self._data)
        if n < 2:
            return 0.0
        m = self.mean()
        return (sum((x - m) ** 2 for x in self._data) / (n - 1)) ** 0.5

    def zscore(self, short_n: int = 5) -> float | None:
        """
        Z-score of the short-term mean relative to the long-term mean/std.
        Returns None until the window is populated.
        """
        if len(self._data) < max(short_n + 1, 20):
            return None
        s = self.std()
        if s == 0.0:
            return None
        return (self.short_mean(short_n) - self.mean()) / s


# ──────────────────────────────────────────────────────────────────────────────
# Base product trader
# ──────────────────────────────────────────────────────────────────────────────

class BaseProductTrader:
    """
    Shared utilities for all per-product traders.

    Interface:
      run(state, td) -> list[Order]
        state : current TradingState
        td    : already-parsed traderData dict (read + write; changes are
                persisted by Trader.run() after all sub-traders finish)

    Quantities: positive = buy, negative = sell.
    """

    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self._orders: list[Order] = []

    # ── Internal helpers ───────────────────────────────────────────────────

    def _reset(self) -> None:
        self._orders = []

    def _buy(self, price: int, qty: int) -> None:
        if qty > 0:
            self._orders.append(Order(self.symbol, int(price), int(qty)))

    def _sell(self, price: int, qty: int) -> None:
        if qty > 0:
            self._orders.append(Order(self.symbol, int(price), -int(qty)))

    # ── Position helpers ───────────────────────────────────────────────────

    def pos(self, state: TradingState) -> int:
        return state.position.get(self.symbol, 0)

    def buy_capacity(self, state: TradingState) -> int:
        return self.limit - self.pos(state)

    def sell_capacity(self, state: TradingState) -> int:
        return self.limit + self.pos(state)

    # ── Order-book helpers ─────────────────────────────────────────────────

    @staticmethod
    def best_bid(depth: OrderDepth) -> int | None:
        return max(depth.buy_orders) if depth.buy_orders else None

    @staticmethod
    def best_ask(depth: OrderDepth) -> int | None:
        return min(depth.sell_orders) if depth.sell_orders else None

    @staticmethod
    def bid_volume(depth: OrderDepth, price: int) -> int:
        return depth.buy_orders.get(price, 0)

    @staticmethod
    def ask_volume(depth: OrderDepth, price: int) -> int:
        return abs(depth.sell_orders.get(price, 0))

    @staticmethod
    def vwap_mid(depth: OrderDepth) -> float | None:
        """Volume-weighted mid using best bid and best ask."""
        bb = max(depth.buy_orders) if depth.buy_orders else None
        ba = min(depth.sell_orders) if depth.sell_orders else None
        if bb is None or ba is None:
            return float(bb or ba or 0) or None
        bv = depth.buy_orders[bb]
        av = abs(depth.sell_orders[ba])
        total = bv + av
        if total == 0:
            return (bb + ba) / 2.0
        return (bb * av + ba * bv) / total

    @staticmethod
    def mid(depth: OrderDepth) -> float | None:
        bb = max(depth.buy_orders) if depth.buy_orders else None
        ba = min(depth.sell_orders) if depth.sell_orders else None
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return float(bb or ba or 0) or None

    # ── Insider signal helpers ─────────────────────────────────────────────

    # Key insight from both Alpha Animals and Frankfurt Hedgehogs (P3 winners):
    # a named bot "Olivia" consistently traded at daily extremes, providing a
    # direction signal. We track any known insider name rather than hard-coding
    # one, so the framework generalises to new rounds.
    INSIDER_NAMES: frozenset[str] = frozenset({"Olivia", "Caesar", "Vladimir", "Camilla"})

    @classmethod
    def insider_direction(cls, state: TradingState, symbol: str) -> int:
        """
        +1 if an insider net-bought this tick, -1 if net-sold, 0 otherwise.
        """
        trades = state.market_trades.get(symbol, [])
        bought = sum(t.quantity for t in trades if t.buyer in cls.INSIDER_NAMES)
        sold = sum(t.quantity for t in trades if t.seller in cls.INSIDER_NAMES)
        if bought > sold:
            return 1
        if sold > bought:
            return -1
        return 0

    # ── Wash-trade guard ───────────────────────────────────────────────────

    def _no_wash_trades(self, orders: list[Order]) -> list[Order]:
        """
        Remove any buy order whose price ≥ a sell order's price in the same
        list. Such pairs would self-fill ('wash trade') — disqualifying under
        IMC rules.
        """
        buys = [(i, o) for i, o in enumerate(orders) if o.quantity > 0]
        sells = [(i, o) for i, o in enumerate(orders) if o.quantity < 0]
        bad_indices: set[int] = set()
        for bi, bo in buys:
            for si, so in sells:
                if bo.price >= so.price:
                    bad_indices.add(bi)
                    bad_indices.add(si)
        return [o for i, o in enumerate(orders) if i not in bad_indices]

    # ── Main interface ─────────────────────────────────────────────────────

    def run(self, state: TradingState, td: dict) -> list[Order]:
        raise NotImplementedError
