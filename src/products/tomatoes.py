from __future__ import annotations
import math
from datamodel import Order, TradingState
from src.base import BaseProductTrader, RollingWindow
from src.config import DEBUG


class TomatoesTrader(BaseProductTrader):
    """
    Market maker for TOMATOES — a slow random-walk product.

    Data observations (tutorial round):
      - No fixed fair value; mid-price drifts around ~5 000.
      - Bot walls: bid ~4 999-5 000, ask ~5 013-5 014 (≈13 tick spread).
      - No clear insider signal in tutorial data.

    Strategy:
      1. Estimate fair value via EMA of VWAP mid (smoothed to reduce adverse
         selection on noisy ticks).
      2. TAKE aggressively: only when market price has moved TAKE_EDGE ticks
         away from our EMA (i.e. clear dislocation). This avoids chasing noise.
      3. POST passively with POSITION-CONDITIONAL quoting — adopted from
         nicolassinott's Bananas strategy (91st, P1):
           - Neutral:  bid at floor(fv - BASE_SPREAD), ask at ceil(fv + BASE_SPREAD)
           - Long:     widen bid (don't want more), tighten ask (want to sell)
           - Short:    tighten bid (want to buy),   widen ask  (don't want to sell)
         This is more intuitive and transparent than a continuous skew formula.

    Lesson from nicolassinott:
      They used EMA α=0.5. We use α=0.3 (less reactive) to avoid
      chasing the noisy TOMATOES price. The wider spread (13 ticks) means
      the price has significant noise; a slower EMA is safer.

    State in traderData:
      "tomatoes_ema": float
    """

    EMA_ALPHA: float = 0.3
    BASE_SPREAD: int = 4    # Neutral half-spread
    WIDE_SPREAD: int = 6    # Wide side when holding inventory
    TAKE_EDGE: int = 2      # Take only when ask <= fv-TAKE_EDGE or bid >= fv+TAKE_EDGE

    TD_KEY: str = "tomatoes_ema"

    def run(self, state: TradingState, td: dict) -> list[Order]:
        self._reset()
        depth = state.order_depths.get(self.symbol)
        if not depth:
            return []

        raw_mid = self.vwap_mid(depth)
        if raw_mid is None:
            return []

        # ── Load / update EMA ────────────────────────────────────────────
        ema_fv: float = td.get(self.TD_KEY, raw_mid)
        ema_fv = self.EMA_ALPHA * raw_mid + (1 - self.EMA_ALPHA) * ema_fv
        td[self.TD_KEY] = ema_fv  # write back — caller persists td

        fv = ema_fv
        pos = self.pos(state)
        buy_cap = self.buy_capacity(state)
        sell_cap = self.sell_capacity(state)

        # ── 1. Aggressive takes (dislocation only) ──────────────────────
        for ask in sorted(depth.sell_orders):
            if buy_cap <= 0:
                break
            if ask <= fv - self.TAKE_EDGE:
                vol = min(self.ask_volume(depth, ask), buy_cap)
                self._buy(ask, vol)
                buy_cap -= vol

        for bid in sorted(depth.buy_orders, reverse=True):
            if sell_cap <= 0:
                break
            if bid >= fv + self.TAKE_EDGE:
                vol = min(self.bid_volume(depth, bid), sell_cap)
                self._sell(bid, vol)
                sell_cap -= vol

        # ── 2. Position-conditional passive quotes ───────────────────────
        # Inspired by nicolassinott's Bananas approach:
        #   neutral → symmetric tight quotes
        #   long    → wide bid (reduce buying), tight ask (encourage selling)
        #   short   → tight bid (encourage buying), wide ask (reduce selling)
        if pos == 0:
            bid_spread = self.BASE_SPREAD
            ask_spread = self.BASE_SPREAD
        elif pos > 0:
            bid_spread = self.WIDE_SPREAD
            ask_spread = self.BASE_SPREAD
        else:
            bid_spread = self.BASE_SPREAD
            ask_spread = self.WIDE_SPREAD

        post_bid = math.floor(fv) - bid_spread
        post_ask = math.ceil(fv) + ask_spread

        if buy_cap > 0:
            self._buy(post_bid, buy_cap)

        if sell_cap > 0:
            self._sell(post_ask, sell_cap)

        orders = self._no_wash_trades(self._orders)

        if DEBUG:
            print(
                f"[TOMATOES] pos={pos} ema={fv:.2f} raw={raw_mid:.2f} "
                f"bid={post_bid}(-{bid_spread}) ask={post_ask}(+{ask_spread}) "
                f"buy_cap={buy_cap} sell_cap={sell_cap}"
            )

        return orders
