from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any
import math
import json


class Trader:

    POSITION_LIMIT = 10

    PEBBLES = ['PEBBLES_XS', 'PEBBLES_S', 'PEBBLES_M', 'PEBBLES_L', 'PEBBLES_XL']
    PEBBLES_SUM = 50000.0

    SNACKPACKS = ['SNACKPACK_CHOCOLATE', 'SNACKPACK_VANILLA', 'SNACKPACK_PISTACHIO',
                  'SNACKPACK_STRAWBERRY', 'SNACKPACK_RASPBERRY']

    WIDE_SPREAD_MM = [
        'UV_VISOR_YELLOW', 'UV_VISOR_AMBER', 'UV_VISOR_ORANGE',
        'UV_VISOR_RED', 'UV_VISOR_MAGENTA',
        'OXYGEN_SHAKE_MORNING_BREATH', 'OXYGEN_SHAKE_EVENING_BREATH',
        'OXYGEN_SHAKE_MINT', 'OXYGEN_SHAKE_CHOCOLATE', 'OXYGEN_SHAKE_GARLIC',
        'PANEL_1X2',
        'GALAXY_SOUNDS_DARK_MATTER', 'GALAXY_SOUNDS_BLACK_HOLES',
        'GALAXY_SOUNDS_PLANETARY_RINGS', 'GALAXY_SOUNDS_SOLAR_WINDS',
        'GALAXY_SOUNDS_SOLAR_FLAMES',
    ]

    MEDIUM_SPREAD_MM = [
        'SLEEP_POD_SUEDE', 'SLEEP_POD_LAMB_WOOL', 'SLEEP_POD_POLYESTER',
        'SLEEP_POD_NYLON', 'SLEEP_POD_COTTON',
        'TRANSLATOR_SPACE_GRAY', 'TRANSLATOR_ASTRO_BLACK',
        'TRANSLATOR_ECLIPSE_CHARCOAL', 'TRANSLATOR_GRAPHITE_MIST',
        'TRANSLATOR_VOID_BLUE',
        'PANEL_2X2', 'PANEL_1X4', 'PANEL_2X4', 'PANEL_4X4',
    ]

    TIGHT_SPREAD = [
        'MICROCHIP_CIRCLE', 'MICROCHIP_OVAL', 'MICROCHIP_SQUARE',
        'MICROCHIP_RECTANGLE', 'MICROCHIP_TRIANGLE',
        'ROBOT_VACUUMING', 'ROBOT_MOPPING', 'ROBOT_DISHES',
        'ROBOT_LAUNDRY', 'ROBOT_IRONING',
    ]

    def __init__(self):
        self.ema_fast: Dict[str, float] = {}
        self.ema_slow: Dict[str, float] = {}
        self.tick_count: Dict[str, int] = {}

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2.0
        return 0.0

    def get_wmid_price(self, order_depth: OrderDepth) -> float:
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            bid_vol = order_depth.buy_orders[best_bid]
            ask_vol = -order_depth.sell_orders[best_ask]
            total = bid_vol + ask_vol
            if total > 0:
                return (best_bid * ask_vol + best_ask * bid_vol) / total
            return (best_bid + best_ask) / 2.0
        return 0.0

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        if state.traderData:
            try:
                saved = json.loads(state.traderData)
                self.ema_fast = saved.get('ef', {})
                self.ema_slow = saved.get('es', {})
                self.tick_count = saved.get('tc', {})
            except:
                pass

        mid_prices: Dict[str, float] = {}
        for product in state.order_depths:
            mid = self.get_mid_price(state.order_depths[product])
            if mid > 0:
                mid_prices[product] = mid
                self.tick_count[product] = self.tick_count.get(product, 0) + 1
                if product in self.ema_fast:
                    self.ema_fast[product] = 0.15 * mid + 0.85 * self.ema_fast[product]
                    self.ema_slow[product] = 0.03 * mid + 0.97 * self.ema_slow[product]
                else:
                    self.ema_fast[product] = mid
                    self.ema_slow[product] = mid

        self.trade_pebbles(state, mid_prices, result)
        self.trade_snackpacks(state, mid_prices, result)

        # WIDE & MEDIUM: use v3 parameters (proven +18.6k)
        for product in self.WIDE_SPREAD_MM:
            if product in state.order_depths and product not in result:
                self.market_make_with_trend(state, product, mid_prices, result)

        for product in self.MEDIUM_SPREAD_MM:
            if product in state.order_depths and product not in result:
                self.market_make_with_trend(state, product, mid_prices, result)

        # TIGHT: new approach to actually get fills
        for product in self.TIGHT_SPREAD:
            if product in state.order_depths and product not in result:
                self.trade_tight(state, product, mid_prices, result)

        trader_data = json.dumps({
            'ef': self.ema_fast,
            'es': self.ema_slow,
            'tc': self.tick_count,
        })

        return result, 0, trader_data

    def get_trend(self, product: str, mid: float) -> float:
        ema_f = self.ema_fast.get(product, mid)
        ema_s = self.ema_slow.get(product, mid)
        ticks = self.tick_count.get(product, 0)
        if ticks < 50 or ema_s == 0:
            return 0.0
        return (ema_f - ema_s) / ema_s

    def trade_pebbles(self, state: TradingState, mid_prices: Dict[str, float],
                       result: Dict[str, List[Order]]):
        if not all(p in mid_prices for p in self.PEBBLES):
            return

        for product in self.PEBBLES:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)

            others = [p for p in self.PEBBLES if p != product]
            other_sum = sum(mid_prices[p] for p in others)
            fair = self.PEBBLES_SUM - other_sum

            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            if best_bid is None or best_ask is None:
                continue
            spread = best_ask - best_bid

            take_threshold = max(3, spread * 0.4)

            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price < fair - take_threshold:
                    ask_vol = -order_depth.sell_orders[ask_price]
                    buy_qty = min(ask_vol, self.POSITION_LIMIT - position)
                    if buy_qty > 0:
                        orders.append(Order(product, ask_price, buy_qty))
                        position += buy_qty

            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price > fair + take_threshold:
                    bid_vol = order_depth.buy_orders[bid_price]
                    sell_qty = min(bid_vol, self.POSITION_LIMIT + position)
                    if sell_qty > 0:
                        orders.append(Order(product, bid_price, -sell_qty))
                        position -= sell_qty

            pos_skew = -position * 0.5
            adj_fair = fair + pos_skew

            half_spread = max(3, math.floor(spread * 0.35))
            buy_price = math.floor(adj_fair - half_spread)
            sell_price = math.ceil(adj_fair + half_spread)

            buy_price = max(buy_price, best_bid + 1)
            sell_price = min(sell_price, best_ask - 1)

            if buy_price >= sell_price:
                buy_price = best_bid
                sell_price = best_ask

            buy_qty = self.POSITION_LIMIT - position
            sell_qty = self.POSITION_LIMIT + position

            if buy_qty > 0:
                orders.append(Order(product, buy_price, buy_qty))
            if sell_qty > 0:
                orders.append(Order(product, sell_price, -sell_qty))

            result[product] = orders

    def trade_snackpacks(self, state: TradingState, mid_prices: Dict[str, float],
                          result: Dict[str, List[Order]]):
        for product in self.SNACKPACKS:
            if product not in state.order_depths:
                continue

            order_depth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)

            mid = mid_prices.get(product)
            if not mid:
                continue

            wmid = self.get_wmid_price(order_depth)
            ema_f = self.ema_fast.get(product, mid)

            fair = 0.7 * wmid + 0.2 * ema_f + 0.1 * mid
            pos_skew = -position * 1.0
            adj_fair = fair + pos_skew

            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

            if best_bid is None or best_ask is None:
                continue

            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price < adj_fair - 1:
                    ask_vol = -order_depth.sell_orders[ask_price]
                    buy_qty = min(ask_vol, self.POSITION_LIMIT - position)
                    if buy_qty > 0:
                        orders.append(Order(product, ask_price, buy_qty))
                        position += buy_qty

            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price > adj_fair + 1:
                    bid_vol = order_depth.buy_orders[bid_price]
                    sell_qty = min(bid_vol, self.POSITION_LIMIT + position)
                    if sell_qty > 0:
                        orders.append(Order(product, bid_price, -sell_qty))
                        position -= sell_qty

            our_half = 2
            buy_price = math.floor(adj_fair - our_half)
            sell_price = math.ceil(adj_fair + our_half)

            buy_price = max(buy_price, best_bid + 1)
            sell_price = min(sell_price, best_ask - 1)

            if buy_price >= sell_price:
                buy_price = math.floor(adj_fair)
                sell_price = math.ceil(adj_fair)
                if buy_price == sell_price:
                    buy_price -= 1
                    sell_price += 1

            buy_qty = self.POSITION_LIMIT - position
            sell_qty = self.POSITION_LIMIT + position

            if buy_qty > 0:
                orders.append(Order(product, buy_price, buy_qty))
            if sell_qty > 0:
                orders.append(Order(product, sell_price, -sell_qty))

            result[product] = orders

    def market_make_with_trend(self, state: TradingState, product: str,
                                mid_prices: Dict[str, float], result: Dict[str, List[Order]]):
        """v3 parameters — proven at +18.6k. Don't touch."""
        order_depth = state.order_depths[product]
        orders: List[Order] = []
        position = state.position.get(product, 0)

        mid = mid_prices.get(product)
        if not mid:
            return

        wmid = self.get_wmid_price(order_depth)
        ema_f = self.ema_fast.get(product, mid)
        ema_s = self.ema_slow.get(product, mid)

        trend = self.get_trend(product, mid)

        fair = 0.5 * wmid + 0.3 * ema_f + 0.2 * ema_s

        # v3 trend skew: 5.0
        trend_skew = trend * mid * 5.0
        pos_skew = -position * 1.0

        adj_fair = fair + trend_skew + pos_skew

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is None or best_ask is None:
            return

        spread = best_ask - best_bid

        # v3 take edge: spread * 0.15
        take_edge = max(1, spread * 0.15)

        for ask_price in sorted(order_depth.sell_orders.keys()):
            if ask_price < adj_fair - take_edge:
                ask_vol = -order_depth.sell_orders[ask_price]
                buy_qty = min(ask_vol, self.POSITION_LIMIT - position)
                if buy_qty > 0:
                    orders.append(Order(product, ask_price, buy_qty))
                    position += buy_qty

        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if bid_price > adj_fair + take_edge:
                bid_vol = order_depth.buy_orders[bid_price]
                sell_qty = min(bid_vol, self.POSITION_LIMIT + position)
                if sell_qty > 0:
                    orders.append(Order(product, bid_price, -sell_qty))
                    position -= sell_qty

        our_half = max(1, math.floor(spread * 0.3))

        buy_price = math.floor(adj_fair - our_half)
        sell_price = math.ceil(adj_fair + our_half)

        buy_price = max(buy_price, best_bid + 1)
        sell_price = min(sell_price, best_ask - 1)

        if buy_price >= sell_price:
            buy_price = math.floor(adj_fair)
            sell_price = math.ceil(adj_fair)
            if buy_price == sell_price:
                buy_price -= 1
                sell_price += 1

        buy_qty = self.POSITION_LIMIT - position
        sell_qty = self.POSITION_LIMIT + position

        if buy_qty > 0:
            orders.append(Order(product, buy_price, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, sell_price, -sell_qty))

        result[product] = orders

    def trade_tight(self, state: TradingState, product: str,
                     mid_prices: Dict[str, float], result: Dict[str, List[Order]]):
        """Tight-spread products: improve spread by 1 tick, use trend to skew."""
        order_depth = state.order_depths[product]
        orders: List[Order] = []
        position = state.position.get(product, 0)

        mid = mid_prices.get(product)
        if not mid:
            return

        wmid = self.get_wmid_price(order_depth)
        ema_f = self.ema_fast.get(product, mid)

        trend = self.get_trend(product, mid)

        # Simple fair: lean on wmid and fast EMA
        fair = 0.6 * wmid + 0.4 * ema_f
        pos_skew = -position * 1.0
        adj_fair = fair + pos_skew

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is None or best_ask is None:
            return

        # Always improve the spread by 1 — this is what gets us fills
        buy_price = best_bid + 1
        sell_price = best_ask - 1

        # In a strong trend, lean harder on the trend side
        if trend > 0.005:
            buy_price = best_bid + 2
        elif trend < -0.005:
            sell_price = best_ask - 2

        # Safety: never cross ourselves
        if buy_price >= sell_price:
            buy_price = best_bid + 1
            sell_price = best_ask - 1
            if buy_price >= sell_price:
                buy_price = best_bid
                sell_price = best_ask

        buy_qty = self.POSITION_LIMIT - position
        sell_qty = self.POSITION_LIMIT + position

        if buy_qty > 0:
            orders.append(Order(product, buy_price, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, sell_price, -sell_qty))

        result[product] = orders