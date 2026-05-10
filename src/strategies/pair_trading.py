from __future__ import annotations
from datamodel import Order, TradingState
from src.base import BaseProductTrader, RollingWindow
from src.config import DEBUG


class PairTrader:
    """
    Statistical arbitrage on two correlated products using a z-score signal.

    Derived from nicolassinott's Round 2 (COCONUT/PINA_COLADAS) and Round 4
    (PICNIC_BASKET components) strategies, but generalised:

      spread = price_B - beta * price_A

    where `beta` is the regression coefficient between the two products.

    Entry:  |z-score| > ENTRY_Z   → trade the spread (long cheap / short rich)
    Exit:   |z-score| < EXIT_Z    → close the position

    Nicolassinott used:
      - Window: 200, short smoother: 5, threshold: 1.5σ (coconuts/pinas)
      - Window: 200, short smoother: 5, threshold: 2.0σ (basket components)
    We use the same defaults but expose them as constructor parameters.

    Usage:
      trader = PairTrader(
          sym_a="COCONUTS", limit_a=300,
          sym_b="PINA_COLADAS", limit_b=100,
          beta=1.551,
          entry_z=1.5, exit_z=0.5,
          trade_size_a=2, trade_size_b=1,
          td_key="coconuts_pinas",
      )
      orders = trader.run(state, td)
    """

    def __init__(
        self,
        sym_a: str,
        limit_a: int,
        sym_b: str,
        limit_b: int,
        beta: float,
        entry_z: float = 1.5,
        exit_z: float = 0.5,
        window: int = 200,
        short_n: int = 5,
        trade_size_a: int = 2,
        trade_size_b: int = 1,
        td_key: str = "pair",
    ) -> None:
        self.sym_a = sym_a
        self.limit_a = limit_a
        self.sym_b = sym_b
        self.limit_b = limit_b
        self.beta = beta
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.window = window
        self.short_n = short_n
        self.trade_size_a = trade_size_a
        self.trade_size_b = trade_size_b
        self.td_key = td_key
        self._helper_a = _OrderHelper(sym_a, limit_a)
        self._helper_b = _OrderHelper(sym_b, limit_b)

    def run(self, state: TradingState, td: dict) -> list[Order]:
        # Get mid prices for both legs
        depth_a = state.order_depths.get(self.sym_a)
        depth_b = state.order_depths.get(self.sym_b)
        if not depth_a or not depth_b:
            return []

        mid_a = _mid(depth_a)
        mid_b = _mid(depth_b)
        if mid_a is None or mid_b is None:
            return []

        # Load state
        key = self.td_key
        spread_history: list[float] = td.get(f"{key}_window", [])
        pair_pos: int = td.get(f"{key}_pos", 0)

        roll = RollingWindow.from_list(spread_history, self.window)

        # Update spread window
        spread = mid_b - self.beta * mid_a
        roll.push(spread)
        td[f"{key}_window"] = roll.to_list()

        z = roll.zscore(self.short_n)
        if z is None:
            return []

        orders: list[Order] = []
        pos_a = state.position.get(self.sym_a, 0)
        pos_b = state.position.get(self.sym_b, 0)

        # Long spread: sell A, buy B (spread is too low — B is cheap relative to A)
        if z < -self.entry_z and pair_pos <= 0:
            vol_a = min(self.trade_size_a, self.limit_a + pos_a)  # short A
            vol_b = min(self.trade_size_b, self.limit_b - pos_b)  # long B
            if vol_a > 0 and vol_b > 0:
                ba = BaseProductTrader.best_ask(depth_a)
                bb_b = BaseProductTrader.best_bid(depth_b)
                if ba and bb_b:
                    orders.append(Order(self.sym_a, ba + 1, -vol_a))  # sell A aggressively
                    orders.append(Order(self.sym_b, bb_b + 1, vol_b))  # buy B aggressively
                    td[f"{key}_pos"] = pair_pos - 1

        # Short spread: buy A, sell B (spread is too high — A is cheap relative to B)
        elif z > self.entry_z and pair_pos >= 0:
            vol_a = min(self.trade_size_a, self.limit_a - pos_a)   # long A
            vol_b = min(self.trade_size_b, self.limit_b + pos_b)   # short B
            if vol_a > 0 and vol_b > 0:
                bb_a = BaseProductTrader.best_bid(depth_a)
                ba_b = BaseProductTrader.best_ask(depth_b)
                if bb_a and ba_b:
                    orders.append(Order(self.sym_a, bb_a - 1, vol_a))   # buy A aggressively
                    orders.append(Order(self.sym_b, ba_b - 1, -vol_b))  # sell B aggressively
                    td[f"{key}_pos"] = pair_pos + 1

        # Exit: z-score has reverted
        elif abs(z) < self.exit_z and pair_pos != 0:
            if pair_pos < 0:  # was long spread: unwind (buy A, sell B)
                bb_a = BaseProductTrader.best_bid(depth_a)
                ba_b = BaseProductTrader.best_ask(depth_b)
                if bb_a and ba_b:
                    vol_a = min(self.trade_size_a, self.limit_a - pos_a)
                    vol_b = min(self.trade_size_b, self.limit_b + pos_b)
                    if vol_a > 0 and vol_b > 0:
                        orders.append(Order(self.sym_a, bb_a - 1, vol_a))
                        orders.append(Order(self.sym_b, ba_b - 1, -vol_b))
                        td[f"{key}_pos"] = 0
            else:  # was short spread: unwind (sell A, buy B)
                ba = BaseProductTrader.best_ask(depth_a)
                bb_b = BaseProductTrader.best_bid(depth_b)
                if ba and bb_b:
                    vol_a = min(self.trade_size_a, self.limit_a + pos_a)
                    vol_b = min(self.trade_size_b, self.limit_b - pos_b)
                    if vol_a > 0 and vol_b > 0:
                        orders.append(Order(self.sym_a, ba + 1, -vol_a))
                        orders.append(Order(self.sym_b, bb_b + 1, vol_b))
                        td[f"{key}_pos"] = 0

        if DEBUG:
            print(f"[PAIR {self.sym_a}/{self.sym_b}] z={z:.2f} pair_pos={pair_pos} spread={spread:.2f}")

        return orders


class BasketArb:
    """
    Basket arbitrage: trade a basket instrument vs. its synthetic replication.

    Derived from nicolassinott's Round 4 PICNIC_BASKET strategy:
      spread = price_basket - (w1*price_A + w2*price_B + w3*price_C)
      entry threshold: 2σ, window: 200, short_n: 5

    Usage:
      arb = BasketArb(
          basket_sym="PICNIC_BASKET", basket_limit=70,
          components=[("UKULELE",1,60), ("BAGUETTE",2,120), ("DIP",4,240)],
          entry_z=2.0, exit_z=0.5,
          trade_size=2,
          td_key="basket_arb",
      )
    """

    def __init__(
        self,
        basket_sym: str,
        basket_limit: int,
        components: list[tuple[str, float, int]],   # (symbol, weight, limit)
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        window: int = 200,
        short_n: int = 5,
        trade_size: int = 2,
        td_key: str = "basket",
    ) -> None:
        self.basket_sym = basket_sym
        self.basket_limit = basket_limit
        self.components = components
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.window = window
        self.short_n = short_n
        self.trade_size = trade_size
        self.td_key = td_key

    def _synthetic_price(self, state: TradingState) -> float | None:
        total = 0.0
        for sym, weight, _ in self.components:
            depth = state.order_depths.get(sym)
            if not depth:
                return None
            m = _mid(depth)
            if m is None:
                return None
            total += weight * m
        return total

    def run(self, state: TradingState, td: dict) -> list[Order]:
        basket_depth = state.order_depths.get(self.basket_sym)
        if not basket_depth:
            return []

        basket_mid = _mid(basket_depth)
        synth = self._synthetic_price(state)
        if basket_mid is None or synth is None:
            return []

        key = self.td_key
        history: list[float] = td.get(f"{key}_window", [])
        roll = RollingWindow.from_list(history, self.window)

        spread = basket_mid - synth
        roll.push(spread)
        td[f"{key}_window"] = roll.to_list()

        z = roll.zscore(self.short_n)
        if z is None:
            return []

        orders: list[Order] = []
        basket_pos = state.position.get(self.basket_sym, 0)

        # Basket is CHEAP (spread below mean): buy basket, sell components
        if z < -self.entry_z and abs(basket_pos) <= self.basket_limit - self.trade_size:
            ba = BaseProductTrader.best_ask(basket_depth)
            if ba:
                orders.append(Order(self.basket_sym, ba, self.trade_size))
            for sym, weight, lim in self.components:
                comp_pos = state.position.get(sym, 0)
                comp_depth = state.order_depths.get(sym)
                if comp_depth:
                    cb = BaseProductTrader.best_bid(comp_depth)
                    sell_vol = min(round(weight * self.trade_size), lim + comp_pos)
                    if cb and sell_vol > 0:
                        orders.append(Order(sym, cb, -sell_vol))

        # Basket is EXPENSIVE (spread above mean): sell basket, buy components
        elif z > self.entry_z and abs(basket_pos) <= self.basket_limit - self.trade_size:
            bb = BaseProductTrader.best_bid(basket_depth)
            if bb:
                orders.append(Order(self.basket_sym, bb, -self.trade_size))
            for sym, weight, lim in self.components:
                comp_pos = state.position.get(sym, 0)
                comp_depth = state.order_depths.get(sym)
                if comp_depth:
                    ca = BaseProductTrader.best_ask(comp_depth)
                    buy_vol = min(round(weight * self.trade_size), lim - comp_pos)
                    if ca and buy_vol > 0:
                        orders.append(Order(sym, ca, buy_vol))

        if DEBUG:
            print(f"[BASKET {self.basket_sym}] z={z:.2f} spread={spread:.2f} basket_pos={basket_pos}")

        return orders


# ── Internal helpers ───────────────────────────────────────────────────────────

class _OrderHelper(BaseProductTrader):
    """Thin wrapper used just to access BaseProductTrader statics."""
    def run(self, state, td):
        return []


def _mid(depth) -> float | None:
    bb = max(depth.buy_orders) if depth.buy_orders else None
    ba = min(depth.sell_orders) if depth.sell_orders else None
    if bb is not None and ba is not None:
        return (bb + ba) / 2.0
    return float(bb or ba or 0) or None
