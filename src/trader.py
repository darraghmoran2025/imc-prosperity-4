from __future__ import annotations
import json
from datamodel import Order, TradingState
from src.config import LIMITS, MASTER_SWITCH
from src.products.emeralds import EmeraldsTrader
from src.products.tomatoes import TomatoesTrader


class Trader:
    """
    Exchange entry point — called once per tick.

    Design principles:
      - traderData is parsed ONCE here into a dict `td`.
      - `td` is passed by reference to every sub-trader's run(state, td).
        Sub-traders read and write directly into `td`; no monkey-patching.
      - After all sub-traders finish, `td` is re-serialised to compact JSON
        and returned as the new traderData string.
      - Conversions are returned as 0 (updated when conversion products arrive).
      - DEBUG = False eliminates all print() calls in competition submissions.
    """

    def __init__(self) -> None:
        self._emeralds = EmeraldsTrader("EMERALDS", LIMITS.get("EMERALDS", 50))
        self._tomatoes = TomatoesTrader("TOMATOES", LIMITS.get("TOMATOES", 50))
        # Round 2+ traders are added here when round data becomes available:
        # self._coconuts_pinas = PairTrader(...)
        # self._basket = BasketArb(...)
        # self._berries = SeasonalTrader(...)

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        # ── Parse persistent state ───────────────────────────────────────
        td: dict = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except Exception:
                pass

        # ── Run all sub-traders (td passed by reference) ─────────────────
        orders: dict[str, list[Order]] = {}

        if "EMERALDS" in state.order_depths:
            result = self._emeralds.run(state, td)
            if result:
                orders["EMERALDS"] = result

        if "TOMATOES" in state.order_depths:
            result = self._tomatoes.run(state, td)
            if result:
                orders["TOMATOES"] = result

        # Round 2+: uncomment when products become available
        # if "COCONUTS" in state.order_depths and "PINA_COLADAS" in state.order_depths:
        #     for o in self._coconuts_pinas.run(state, td):
        #         orders.setdefault(o.symbol, []).append(o)
        #
        # if "PICNIC_BASKET" in state.order_depths:
        #     for o in self._basket.run(state, td):
        #         orders.setdefault(o.symbol, []).append(o)

        # ── Re-serialise state (compact JSON, minimal bytes) ─────────────
        trader_data = json.dumps(td, separators=(",", ":"))

        return orders, 0, trader_data
