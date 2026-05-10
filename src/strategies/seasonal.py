from __future__ import annotations
from datamodel import Order, TradingState
from src.base import BaseProductTrader
from src.config import DEBUG


class SeasonalTrader:
    """
    Timestamp-based seasonal trading for products with predictable price
    patterns within a day (e.g. BERRIES in P2 which peaked mid-day).

    Derived from nicolassinott's Round 3 BERRIES strategy:
      - Buy around timestamp 200 000 (± tolerance)
      - Sell around timestamp 500 000 (± tolerance)

    Enhanced with insider signal override (Olivia pattern from P3):
      If an insider is trading in the same direction as the seasonal signal,
      the position size is increased by INSIDER_BOOST_FACTOR.

    Usage:
      trader = SeasonalTrader(
          symbol="BERRIES",
          limit=250,
          buy_ts=200_000,
          sell_ts=500_000,
          tolerance=800,
          trade_size=40,
          td_key="berries_seasonal",
      )
    """

    INSIDER_BOOST_FACTOR: float = 1.5

    def __init__(
        self,
        symbol: str,
        limit: int,
        buy_ts: int,
        sell_ts: int,
        tolerance: int = 800,
        trade_size: int = 40,
        td_key: str = "seasonal",
    ) -> None:
        self.symbol = symbol
        self.limit = limit
        self.buy_ts = buy_ts
        self.sell_ts = sell_ts
        self.tolerance = tolerance
        self.trade_size = trade_size
        self.td_key = td_key
        self._helper = BaseProductTrader.__new__(BaseProductTrader)
        self._helper.symbol = symbol
        self._helper.limit = limit

    def run(self, state: TradingState, td: dict) -> list[Order]:
        depth = state.order_depths.get(self.symbol)
        if not depth:
            return []

        ts = state.timestamp
        orders: list[Order] = []
        pos = state.position.get(self.symbol, 0)
        insider = BaseProductTrader.insider_direction(state, self.symbol)

        in_buy_window = abs(ts - self.buy_ts) <= self.tolerance
        in_sell_window = abs(ts - self.sell_ts) <= self.tolerance

        if in_buy_window and pos < self.limit:
            size = self.trade_size
            if insider > 0:
                size = min(int(size * self.INSIDER_BOOST_FACTOR), self.limit - pos)
            ba = min(depth.sell_orders) if depth.sell_orders else None
            if ba:
                buy_vol = min(size, self.limit - pos)
                if buy_vol > 0:
                    orders.append(Order(self.symbol, ba, buy_vol))

        elif in_sell_window and pos > -self.limit:
            size = self.trade_size
            if insider < 0:
                size = min(int(size * self.INSIDER_BOOST_FACTOR), self.limit + pos)
            bb = max(depth.buy_orders) if depth.buy_orders else None
            if bb:
                sell_vol = min(size, self.limit + pos)
                if sell_vol > 0:
                    orders.append(Order(self.symbol, bb, -sell_vol))

        if DEBUG:
            print(f"[SEASONAL {self.symbol}] ts={ts} pos={pos} insider={insider} in_buy={in_buy_window} in_sell={in_sell_window}")

        return orders


class DerivativeSignalTrader:
    """
    Trade a product based on percentage changes in an external observation
    (e.g. DOLPHIN_SIGHTINGS predicts DIVING_GEAR in P2/P3).

    Derived from nicolassinott's Round 3 DIVING_GEAR strategy:
      - Long trigger: pct_change_dolphin > +0.002
      - Short trigger: pct_change_dolphin < -0.002
      - Minimum hold: 2000 timestamps
      - Close: 3 consecutive bars moving against position
    """

    def __init__(
        self,
        symbol: str,
        limit: int,
        observation_key: str,           # key in state.observations.plainValueObservations
        long_threshold: float = 0.002,
        short_threshold: float = -0.002,
        min_hold_ts: int = 2000,
        td_key: str = "deriv_signal",
    ) -> None:
        self.symbol = symbol
        self.limit = limit
        self.obs_key = observation_key
        self.long_thr = long_threshold
        self.short_thr = short_threshold
        self.min_hold_ts = min_hold_ts
        self.td_key = td_key

    def run(self, state: TradingState, td: dict) -> list[Order]:
        depth = state.order_depths.get(self.symbol)
        obs_val = state.observations.plainValueObservations.get(self.obs_key)
        if depth is None or obs_val is None:
            return []

        key = self.td_key
        last_obs: float = td.get(f"{key}_last_obs", float(obs_val))
        signal_ts: int = td.get(f"{key}_signal_ts", 0)
        signal_dir: int = td.get(f"{key}_signal_dir", 0)
        consec_against: int = td.get(f"{key}_consec", 0)

        pct_change = (float(obs_val) - last_obs) / max(abs(last_obs), 1e-6)

        # Detect new signal
        new_dir = 0
        if pct_change > self.long_thr:
            new_dir = 1
        elif pct_change < self.short_thr:
            new_dir = -1

        if new_dir != 0 and signal_dir == 0:
            signal_dir = new_dir
            signal_ts = state.timestamp
            consec_against = 0

        # Track consecutive adverse returns for exit
        if signal_dir != 0:
            if (signal_dir > 0 and pct_change < 0) or (signal_dir < 0 and pct_change > 0):
                consec_against += 1
            else:
                consec_against = 0

        td[f"{key}_last_obs"] = float(obs_val)
        td[f"{key}_signal_ts"] = signal_ts
        td[f"{key}_signal_dir"] = signal_dir
        td[f"{key}_consec"] = consec_against

        orders: list[Order] = []
        pos = state.position.get(self.symbol, 0)
        held_long_enough = (state.timestamp - signal_ts) >= self.min_hold_ts
        should_close = held_long_enough and consec_against >= 3

        bb = max(depth.buy_orders) if depth.buy_orders else None
        ba = min(depth.sell_orders) if depth.sell_orders else None

        if should_close and pos != 0:
            # Close position: place aggressive order 200 ticks away to ensure fill
            if pos > 0 and bb:
                orders.append(Order(self.symbol, bb - 200, -pos))
            elif pos < 0 and ba:
                orders.append(Order(self.symbol, ba + 200, -pos))
            td[f"{key}_signal_dir"] = 0

        elif signal_dir > 0 and pos < self.limit and not should_close:
            if ba:
                vol = min(self.limit - pos, self.limit)
                orders.append(Order(self.symbol, ba + 200, vol))

        elif signal_dir < 0 and pos > -self.limit and not should_close:
            if bb:
                vol = min(self.limit + pos, self.limit)
                orders.append(Order(self.symbol, bb - 200, -vol))

        if DEBUG:
            print(f"[DERIV {self.symbol}] dir={signal_dir} consec={consec_against} pos={pos} pct={pct_change:.4f}")

        return orders
