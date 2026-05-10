from __future__ import annotations
from datamodel import Order, TradingState
from src.base import BaseProductTrader
from src.config import DEBUG


class EmeraldsTrader(BaseProductTrader):
    """
    Market maker for EMERALDS — a fixed fair-value product.

    Data observations (tutorial round):
      - Fair value is permanently 10 000.
      - Exchange bots post standing walls: bid at 9 992 (vol ~10-15),
        ask at 10 008 (vol ~10-15).
      - Occasional mid-book orders appear at exactly 10 000.

    Strategy (derived from Frankfurt Hedgehogs + Alpha Animals P3 approach):
      1. TAKE aggressively: buy every ask < FV; sell every bid > FV.
         At exactly FV, only trade to flatten existing inventory.
      2. POST passively:
         - Overbid the standing bot-bid by 1 tick (e.g. 9993 vs 9992),
           capped at FV-1. This captures any market sell order that would
           otherwise fill the bot's 9992 bid.
         - Underask the standing bot-ask by 1 tick (e.g. 10007 vs 10008),
           floored at FV+1.
      3. INVENTORY SKEW: if position > 0, shift both quotes downward to
         encourage position mean-reversion. Skew is bounded to MAX_SPREAD
         so quotes never cross the fair value.

    Lesson from nicolassinott (91st, P1):
      Their Pearls strategy posted at FV±1 only. Our approach is strictly
      superior because we overbid/underask the bot walls, gaining queue
      priority over them while staying inside the profitable range.
    """

    FAIR_VALUE: int = 10_000
    MAX_SPREAD: int = 7      # Quote range: [FV-7, FV-1] bid, [FV+1, FV+7] ask
    SKEW_DIVISOR: float = 4  # Larger → gentler skew

    def run(self, state: TradingState, td: dict) -> list[Order]:
        self._reset()
        depth = state.order_depths.get(self.symbol)
        if not depth:
            return []

        fv = self.FAIR_VALUE
        pos = self.pos(state)
        buy_cap = self.buy_capacity(state)
        sell_cap = self.sell_capacity(state)

        # ── 1. Aggressive takes ──────────────────────────────────────────
        for ask in sorted(depth.sell_orders):
            if buy_cap <= 0:
                break
            if ask < fv:
                vol = min(self.ask_volume(depth, ask), buy_cap)
                self._buy(ask, vol)
                buy_cap -= vol
            elif ask == fv and pos < 0:
                vol = min(self.ask_volume(depth, ask), buy_cap, -pos)
                self._buy(ask, vol)
                buy_cap -= vol

        for bid in sorted(depth.buy_orders, reverse=True):
            if sell_cap <= 0:
                break
            if bid > fv:
                vol = min(self.bid_volume(depth, bid), sell_cap)
                self._sell(bid, vol)
                sell_cap -= vol
            elif bid == fv and pos > 0:
                vol = min(self.bid_volume(depth, bid), sell_cap, pos)
                self._sell(bid, vol)
                sell_cap -= vol

        # ── 2. Passive quotes with inventory skew ───────────────────────
        skew = round(pos / self.SKEW_DIVISOR / self.limit * self.MAX_SPREAD)

        bb = self.best_bid(depth)
        raw_bid = (bb + 1) if bb is not None else (fv - self.MAX_SPREAD)
        post_bid = max(fv - self.MAX_SPREAD, min(fv - 1, raw_bid)) - skew

        ba = self.best_ask(depth)
        raw_ask = (ba - 1) if ba is not None else (fv + self.MAX_SPREAD)
        post_ask = min(fv + self.MAX_SPREAD, max(fv + 1, raw_ask)) - skew

        if buy_cap > 0 and post_bid < fv:
            self._buy(post_bid, buy_cap)

        if sell_cap > 0 and post_ask > fv:
            self._sell(post_ask, sell_cap)

        orders = self._no_wash_trades(self._orders)

        if DEBUG:
            print(
                f"[EMERALDS] pos={pos} skew={skew} "
                f"bid={post_bid} ask={post_ask} "
                f"buy_cap={buy_cap} sell_cap={sell_cap}"
            )

        return orders
