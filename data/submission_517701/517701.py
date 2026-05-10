"""
IMC Prosperity Round 4 - Trader v10 (The Anchor Vol Arb)
Strategy:
  THE ANCHOR      : Slow IV tracking (0.9/0.1) refuses to chase toxic market flow.
  FULL CAPACITY   : Unchained option limits (Max 300) to feast on ATM mispricings.
  DECOUPLED MM    : Pure Vol Arb on Options, Pure Market Making on Underlying.
  STRICT SKEW     : Linear defense to prevent toxic inventory buildup.
"""
from typing import List, Dict
from collections import defaultdict, deque
import math, statistics

from datamodel import OrderDepth, TradingState, Order

POS_LIMIT = {"HYDROGEL_PACK": 200, "VELVETFRUIT_EXTRACT": 200}
VOUCHER_LIMIT = 300
VOUCHER_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
ACTIVE_VOUCHERS = [5000, 5100, 5200, 5300, 5400, 5500] 

HYDRO_FAIR_CENTER = 10000.0
VOUCHER_TTE_DAYS  = 4

# ----- Black-Scholes Engine -----
def _N(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_d1(S, K, T_days, sigma):
    if sigma <= 0 or T_days <= 0: return 0.0
    T = T_days / 252.0
    return (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))

def bs_call(S, K, T_days, sigma):
    if sigma <= 0 or T_days <= 0: return max(S - K, 0.0)
    T = T_days / 252.0
    d1 = bs_d1(S, K, T_days, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * _N(d1) - K * _N(d2)

def implied_vol(S, K, T_days, price):
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-6 or price >= S - 1e-6: return None
    lo, hi = 0.001, 5.0
    for _ in range(40):
        m = 0.5 * (lo + hi)
        if bs_call(S, K, T_days, m) > price: hi = m
        else: lo = m
    return 0.5 * (lo + hi)

class Trader:
    def __init__(self):
        self.price_hist = defaultdict(lambda: deque(maxlen=300))
        self.last_seen = defaultdict(int)
        
        self.pending_eval = deque()  
        self.cp_pnl = defaultdict(float)         
        self.flow_imbalance = defaultdict(float) 
        self.current_iv = 0.15 

    @staticmethod
    def _mid(od):
        if not od.buy_orders or not od.sell_orders: return None
        return (max(od.buy_orders) + min(od.sell_orders)) / 2.0

    @staticmethod
    def _best(od):
        bb = max(od.buy_orders) if od.buy_orders else None
        ba = min(od.sell_orders) if od.sell_orders else None
        return bb, ba

    def _clip(self, val, limit):
        return max(-limit, min(limit, val))

    def _ingest(self, state: TradingState):
        while self.pending_eval and self.pending_eval[0][0] <= state.timestamp:
            eval_time, sym, cp, signed_qty, trade_price = self.pending_eval.popleft()
            od = state.order_depths.get(sym)
            if od:
                mid = self._mid(od)
                if mid is not None:
                    edge = signed_qty * (mid - trade_price)
                    per_unit_edge = self._clip(edge / max(1, abs(signed_qty)), 5.0)
                    self.cp_pnl[cp] = 0.98 * self.cp_pnl.get(cp, 0.0) + 0.02 * per_unit_edge

        for k in list(self.flow_imbalance.keys()):
            self.flow_imbalance[k] *= 0.98

        for sym, tlist in state.market_trades.items():
            for t in tlist:
                if t.timestamp <= self.last_seen[sym]: continue
                if t.buyer and t.buyer != "SUBMISSION":
                    self.pending_eval.append((state.timestamp + 500, sym, t.buyer, t.quantity, t.price))
                    self.flow_imbalance[sym] += self.cp_pnl.get(t.buyer, 0.0) * min(t.quantity, 10)
                if t.seller and t.seller != "SUBMISSION":
                    self.pending_eval.append((state.timestamp + 500, sym, t.seller, -t.quantity, t.price))
                    self.flow_imbalance[sym] -= self.cp_pnl.get(t.seller, 0.0) * min(t.quantity, 10)
            if tlist: self.last_seen[sym] = max(t.timestamp for t in tlist)

    def _quote_safe(self, sym, od, pos, limit, theo, skew, min_edge=1.0, max_qty=40):
        orders = []
        bb, ba = self._best(od)
        if bb is None or ba is None: return []

        if ba < theo - 2:
            qty = min(-od.sell_orders[ba], limit - pos)
            if qty > 0: orders.append(Order(sym, ba, qty)); pos += qty
        if bb > theo + 2:
            qty = min(od.buy_orders[bb], limit + pos)
            if qty > 0: orders.append(Order(sym, bb, -qty)); pos -= qty

        bid_px = bb + (1 if ba - bb >= 2 else 0)
        ask_px = ba - (1 if ba - bb >= 2 else 0)

        max_bid = int(math.floor(theo - min_edge + skew))
        min_ask = int(math.ceil(theo + min_edge + skew))

        bid_px = max(1, min(bid_px, max_bid))
        ask_px = max(ask_px, min_ask)
        if bid_px >= ask_px: ask_px = bid_px + 1

        bid_size = min(max_qty, max(0, limit - pos))
        ask_size = min(max_qty, max(0, limit + pos))

        if pos > limit * 0.8: bid_size = 0
        if pos < -limit * 0.8: ask_size = 0

        if bid_size > 0: orders.append(Order(sym, bid_px, bid_size))
        if ask_size > 0: orders.append(Order(sym, ask_px, -ask_size))
        return orders

    # -------------------- MAIN LOGIC --------------------
    def trade_hydrogel(self, state: TradingState):
        sym = "HYDROGEL_PACK"
        od = state.order_depths.get(sym)
        if not od: return []
        m = self._mid(od)
        if m is None: return []
        self.price_hist[sym].append(m)
        pos = state.position.get(sym, 0)
        
        rolling = statistics.fmean(list(self.price_hist[sym])[-50:]) if len(self.price_hist[sym]) >= 10 else m
        signal = self._clip(self.flow_imbalance[sym], 5.0)
        fair = 0.5 * HYDRO_FAIR_CENTER + 0.5 * rolling + signal
        
        return self._quote_safe(sym, od, pos, POS_LIMIT[sym], fair, skew=-0.02 * pos, min_edge=1.5, max_qty=50)

    def trade_velvetfruit(self, state: TradingState):
        sym = "VELVETFRUIT_EXTRACT"
        od = state.order_depths.get(sym)
        if not od: return []
        m = self._mid(od)
        if m is None: return []
        self.price_hist[sym].append(m)
        pos = state.position.get(sym, 0)
        
        rolling = statistics.fmean(list(self.price_hist[sym])[-50:]) if len(self.price_hist[sym]) >= 10 else m
        signal = self._clip(self.flow_imbalance[sym], 5.0)
        fair = rolling + signal
        
        return self._quote_safe(sym, od, pos, POS_LIMIT[sym], fair, skew=-0.05 * pos, min_edge=1.0, max_qty=40)

    def trade_vouchers(self, state: TradingState):
        out = defaultdict(list)
        
        od_und = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if not od_und: return out
        S = self._mid(od_und)
        if S is None: return out
        
        # The Anchor: 0.9/0.1 Slow IV Tracking (Prevents chasing toxic flow spikes)
        atm_strike = min(VOUCHER_STRIKES, key=lambda k: abs(k - S))
        od_atm = state.order_depths.get(f"VEV_{atm_strike}")
        if od_atm:
            atm_mid = self._mid(od_atm)
            if atm_mid:
                calc_iv = implied_vol(S, atm_strike, VOUCHER_TTE_DAYS, atm_mid)
                if calc_iv is not None:
                    self.current_iv = 0.9 * self.current_iv + 0.1 * calc_iv
                    
        for K in ACTIVE_VOUCHERS:
            sym = f"VEV_{K}"
            od = state.order_depths.get(sym)
            if not od: continue
            
            pos = state.position.get(sym, 0)
            theo = bs_call(S, K, VOUCHER_TTE_DAYS, self.current_iv)
            
            # FULL 300 LIMIT RESTORED
            orders = self._quote_safe(sym, od, pos, VOUCHER_LIMIT, theo, skew=-0.02 * pos, min_edge=1.5, max_qty=30)
            if orders: out[sym].extend(orders)

        return out

    def run(self, state: TradingState):
        self._ingest(state)
        result = {}
        
        hydro_orders = self.trade_hydrogel(state)
        if hydro_orders: result["HYDROGEL_PACK"] = hydro_orders
            
        velvet_orders = self.trade_velvetfruit(state)
        if velvet_orders: result["VELVETFRUIT_EXTRACT"] = velvet_orders
        
        for sym, ords in self.trade_vouchers(state).items():
            result[sym] = ords
            
        return result, 0, ""