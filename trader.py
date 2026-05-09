import json
from typing import List, Dict, Tuple, Any
import math
import numpy as np
from statistics import NormalDist
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

#[BASELINE CODE HERE]

class ProductTrader:
    def __init__(self, symbol:str, state: TradingState, new_traderData: dict):
        self.orders: list[Order] = []
        self.symbol = symbol
        self.state = state
        self.new_traderData = new_traderData

        self.position_limit = POS_LIMITS.get(symbol, 0)
        self.initial_position = self.state.position.get(symbol, 0)
        self.last_traderData = self._get_last_traderData()
        
        self.bid_orders, self.ask_orders = self._get_order_depth() 
        self.wall_ask = max(self.ask_orders.keys(), default=None)
        self.wall_bid = min(self.bid_orders.keys(), default=None)
        self.wall_mid = self._get_mid_price(self.wall_bid, self.wall_ask)
        self.best_ask, self.best_bid = self.get_new_best_prices()
        self.best_mid = self._get_mid_price(self.best_bid, self.best_ask)
        self.half_spread = (self.best_ask-self.best_bid)/2 if (self.best_ask is not None and self.best_bid is not None) else 0

    def _get_last_traderData(self):
        last_traderData = {}
        try:
            if self.state.traderData != '':
                last_traderData = json.loads(self.state.traderData)
        except: 
            pass
        return last_traderData

    def _get_order_depth(self):
        lob = self.state.order_depths.get(self.symbol, {})
        bid_orders = {p: abs(v) for p, v in sorted(lob.buy_orders.items(), reverse=True)}
        ask_orders = {p: abs(v) for p, v in sorted(lob.sell_orders.items())}
        return bid_orders, ask_orders
    
    def _get_mid_price(self, _bid, _ask):
        if _bid is not None and _ask is not None:
            return (_bid + _ask) / 2
        if _bid is not None:
            return _bid
        if _ask is not None:
            return _ask
        return None
    

    def get_new_best_prices(self):
        best_ask = min(self.ask_orders.keys(), default=None)
        best_bid = max(self.bid_orders.keys(), default=None)
        return best_ask, best_bid
    
    def update_prior(self, key,  new_obs, alpha=0.1):
        prior = self.last_traderData.get(key, new_obs)
        posterior = alpha*new_obs + (1-alpha)*prior
        self.new_traderData[key] = posterior
        return posterior

    def get_informed(self):
        return None

    def buy(self, price, volume):
        size = abs(int(volume))
        if size <=0: return 
        order = Order(self.symbol, int(price), size)
        self.orders.append(order)
        return 

    def sell(self, price, volume):
        size = abs(int(volume))
        if size <=0: return 
        order = Order(self.symbol, int(price), -size)
        self.orders.append(order)
        return 
    
    def get_orders(self, **kwargs): #return {self.symbol: self.orders}
        return {}

#Rounding Optimisation, Shared liquidity, Premiums?
class SimpleTrader(ProductTrader):
    """
    Neutralise, Make, Take around Fair Value subject to liquidity and final position limit constraints
    """
    def __init__(self, symbol:str, state: TradingState, new_traderData: dict):
        super().__init__(symbol, state, new_traderData)

    def neutralise(self, fair_value): #path dependent with w/o liquidity replacement
        net_position = self.initial_position
        
        if net_position>0:
            for bid in list(self.bid_orders.keys()):
                volume = self.bid_orders[bid]
                if bid < fair_value:
                    break
                size = min(volume, net_position)
                self.sell(bid, -size)
                net_position -= size
                
                self.bid_orders[bid] -= size
                if self.bid_orders[bid] <= 0:
                    del self.bid_orders[bid]

        elif net_position<0:
            for ask in list(self.ask_orders.keys()):
                volume = self.ask_orders[ask]
                if ask > fair_value:
                    break
                size = min(volume, -net_position)
                self.buy(ask, size)
                net_position += size
                
                self.ask_orders[ask] -= size
                if self.ask_orders[ask] <= 0:
                    del self.ask_orders[ask]
        
        taken = net_position - self.initial_position
        self.initial_position = net_position  
        return taken
    
    def take(self, fair_value):
        net_position = self.initial_position

        for bid in list(self.bid_orders.keys()):
            if bid <= fair_value:
                break
            volume = self.bid_orders[bid]
            sell_capacity = abs(self.position_limit + net_position)
            if sell_capacity <= 0:
                break
            size = min(volume, sell_capacity)
            self.sell(bid, -size)
            net_position -= size

            self.bid_orders[bid] -= size
            if self.bid_orders[bid] <= 0:
                del self.bid_orders[bid]

        for ask in list(self.ask_orders.keys()):
            if ask >= fair_value:
                break
            volume = self.ask_orders[ask]
            buy_capacity = abs(self.position_limit - net_position)
            if buy_capacity <= 0:
                break
            size = min(volume, buy_capacity)
            self.buy(ask, size)
            net_position += size

            self.ask_orders[ask] -= size
            if self.ask_orders[ask] <= 0:
                del self.ask_orders[ask]

        taken = net_position - self.initial_position
        self.initial_position = net_position
        return taken

    def make(self, fair_value): #path independent w/o liquidity constraints
        curr_best_ask, curr_best_bid = self.get_new_best_prices() #tested stale and extreme         
        ask = self.best_ask #ask = max(self.best_bid, ask)
        bid = self.best_bid #bid = min(self.best_ask, bid)

        net_position = self.initial_position
        if bid is not None:
            if bid == curr_best_bid:
                bid += 1
            if bid < fair_value:
                buy_capacity = abs(self.position_limit - net_position)
                if buy_capacity > 0:
                    size = buy_capacity
                    self.buy(bid, size)
                    net_position += size
        bids = net_position - self.initial_position

        net_position = self.initial_position
        if ask is not None:
            if ask == curr_best_ask:
                ask -= 1
            if ask > fair_value:
                sell_capacity = abs(self.position_limit + net_position)
                if sell_capacity > 0:
                    size = sell_capacity
                    self.sell(ask, -size)
                    net_position -= size
        asks = net_position - self.initial_position
        return bids, asks

    def track_missing(self):
        missing_bid = bool(self.best_bid is None)
        missing_ask = bool(self.best_ask is None)
        self.new_traderData[self.symbol] = {"bid": missing_bid, "ask": missing_ask}

        last_data = self.last_traderData.get(self.symbol, {})            
        return last_data.get("bid", False), last_data.get("ask", False)
    
    def get_fair_value(self, **kwargs):
        return None

    def get_orders(self):
        return {}

#----------------------------------------------------------------------------------------------------------------------------
#Round 1&2
class StaticTrader(SimpleTrader): 
    def __init__(self, symbol: str, state: TradingState, new_traderData: dict):
        super().__init__(symbol, state, new_traderData)

    def get_fair_value(self):
        return 10_000
    
    def get_orders(self):
        fair_value = self.get_fair_value()
        
        self.neutralise(fair_value)
        self.take(fair_value)
        self.make(fair_value)
        return {self.symbol: self.orders}
    
class TrendTrader(SimpleTrader): 
    def __init__(self, symbol: str, state: TradingState, new_traderData: dict):
        super().__init__(symbol, state, new_traderData)
        self.active_limit = 5
        self.passive_limit = self.position_limit - self.active_limit

    def get_fair_value(self):
        m = 0.001
        c = self.last_traderData.get("c", self.wall_mid)
        shift = self.last_traderData.get("shift", self.state.timestamp)
        self.new_traderData["c"] = c
        self.new_traderData["shift"] = shift
        if (c is None or shift is None):
            return 9999999999
        x = self.state.timestamp - shift 
        return math.ceil(m*x+c)
    
    def get_orders(self):
        net_position = self.initial_position
        taken = 0

        self.position_limit = self.passive_limit - self.active_limit
        self.initial_position = min(self.initial_position, self.position_limit)
        fair_value = 9999999999

        taken += self.neutralise(fair_value)
        taken += self.take(fair_value)
        self.make(fair_value)

        net_position = max(net_position + taken, self.position_limit) #if make then max is limit, if not it is net + takeng
        self.position_limit = self.active_limit
        self.initial_position = max(net_position - self.passive_limit, - self.active_limit)
        fair_value = self.get_fair_value() 

        self.neutralise(fair_value)
        self.take(fair_value)
        self.make(fair_value)
        return {self.symbol: self.orders}

#----------------------------------------------------------------------------------------------------------------------------
#Round 3&4
class InsiderTracker(ProductTrader):
    def __init__(self, symbol, state, new_traderData):
        super().__init__(symbol, state, new_traderData)
        self.informed = INFORMED.get(symbol, []) #Used Mark 38, ignored 55 cuz we have beef
        self.uninformed = UNINFORMED.get(symbol, [])

    def check_informed(self):
        bought, sold = False, False
        for trade in self.state.market_trades.get(self.symbol, []):
            if trade.buyer in self.informed:
                bought = True
            if trade.seller in self.informed:
                sold = True
        if bought and sold: return 0
        elif bought: return 1
        elif sold: return -1
        return 0

    def check_uninformed(self):
        bought, sold = False, False
        for trade in self.state.market_trades.get(self.symbol, []):
            if trade.buyer in self.uninformed:
                bought = True
            if trade.seller in self.uninformed:
                sold = True
        if bought and sold: return 0
        elif bought: return -1  
        elif sold: return 1     
        return 0

    def track_box(self):
        data = self.last_traderData.get(self.symbol, {})
        ub  = data.get("ub", self.wall_mid)
        lb  = data.get("lb", self.wall_mid)
        d1  = data.get("d1", 0)  # informed direction
        d2  = data.get("d2", 0)  # uninformed direction

        informed_direction   = self.check_informed()
        uninformed_direction = self.check_uninformed()

        if self.wall_mid > ub:
            ub = self.wall_mid
            if informed_direction == -1:
                d1 = -1
            if uninformed_direction == -1: 
                d2 = -1

        if self.wall_mid < lb:
            lb = self.wall_mid
            if informed_direction == 1:
                d1 = 1
            if uninformed_direction == 1:   
                d2 = 1

        self.new_traderData.setdefault(self.symbol, {})
        self.new_traderData[self.symbol].update({"ub": ub, "lb": lb, "d1": d1, "d2": d2})
        return d1, d2
    
class StaticTrader(SimpleTrader): 
    def __init__(self, symbol: str, state: TradingState, new_traderData: dict):
        super().__init__(symbol, state, new_traderData)
        self.insider_tracker = InsiderTracker(symbol, state, new_traderData)

    def get_fair_value(self):
        gamma = self.half_spread / 2
        adj = gamma * (self.initial_position / (self.position_limit + 1e-6))
        fair_value = int(self.wall_mid - adj)

        self.insider_tracker.track_box()
        direction = self.new_traderData.get(self.symbol, {}).get("d1", 0)
        degree = self.half_spread ##qty weighted?
        skew = direction * degree
        return max(0, fair_value + skew)
    
    def get_orders(self):
        fair_value = self.get_fair_value()
        if fair_value is None:
            return {}
        
        self.neutralise(fair_value)
        self.take(fair_value)
        self.make(fair_value)
        return {self.symbol: self.orders}

class Voucher(ProductTrader):
    def __init__(self, symbol, state, new_traderData):
        super().__init__(symbol, state, new_traderData)
        self.vouchers = {sym: SimpleTrader(sym, state, new_traderData) for sym in VEVS}
        self.underlying = StaticTrader(symbol, state, new_traderData)
        self.voucher_trackers = {sym: InsiderTracker(sym, state, new_traderData) for sym in VEVS}

    def _implied_vol(self, C_market, S, K, T, tol=1e-6, max_iter=100):
        if C_market <= max(0, S - K): return 0.15
        vol = 0.25  
        for _ in range(max_iter):
            C, greeks = self._BSM(S, K, T, vol)
            diff = C - C_market
            if abs(diff) < tol: break
            vega = max(greeks["vega"] * 100, 1e-6)
            vol -= diff / vega            
        return max(0.01, vol)

    def _BSM(self, S, K, T, vol, r=0):
        if T <= 0: return max(S - K, 0), {"delta": 1.0 if S > K else 0.0, "vega": 0}
        N = NormalDist() 
        total_vol = vol * math.sqrt(T)
        d1 = (math.log(S / K) + (r + (vol**2) / 2) * T) / total_vol
        d2 = d1 - total_vol
        delta = N.cdf(d1)
        C = S * delta - K * math.exp(-r * T) * N.cdf(d2)
        vega = S * N.pdf(d1) * math.sqrt(T) / 100 
        return C, {"delta": delta, "vega": vega}

    def _get_median(self, data):
        if not data: return None
        s = sorted(data)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0

    def _vega_weighted_mean(self, iv_vega_pairs):
        if not iv_vega_pairs: return None
        total_vega = sum(v for _, v in iv_vega_pairs)
        if total_vega == 0:
            return self._get_median([iv for iv, _ in iv_vega_pairs])
        return sum(iv * vega for iv, vega in iv_vega_pairs) / total_vega

    def get_orders(self):
        S = self.wall_mid
        T = (8 - DAY - self.state.timestamp / TICKS_PER_DAY) / DAYS_PER_YEAR
        
        if S <= 0 or T <= 0:
            return {}

        mm = self.underlying.get_orders()

        d1 = self.new_traderData.get(self.symbol, {}).get("d1", 0)
        d2 = self.new_traderData.get(self.symbol, {}).get("d2", 0)

        ivs = []
        valid_vouchers = []
        for sym, trader in self.vouchers.items():
            K = int(sym.split("_")[-1])
            market_price = trader.wall_mid
            if market_price > 0:
                iv = self._implied_vol(market_price, S, K, T)
                _, greeks = self._BSM(S, K, T, iv)
                delta = greeks["delta"]
                vega = greeks["vega"] * 100
                if not VEGA_WEIGHTED:
                    ivs.append(iv)
                else:
                    ivs.append((iv, vega))
                valid_vouchers.append((sym, trader, K, iv, delta))

        if not ivs:
            return mm

        median_vol = self._vega_weighted_mean(ivs) if VEGA_WEIGHTED else self._get_median(ivs)
        avg_iv_deviation = 0

        all_orders = {}
        for sym, trader, K, iv, delta in valid_vouchers:
            #if sym == "VEV_5400": median_vol = 0.2225
            fair_value, greeks = self._BSM(S, K, T, median_vol)
            vega = greeks["vega"] * 100
            threshold = avg_iv_deviation * vega

            if delta > 0.9:
                vd1, vd2 = self.voucher_trackers[sym].track_box()
                if vd1 != 0:
                    d1 = vd1
                if vd2 != 0:
                    d2 = vd2

                direction = d1             
                degree = self.half_spread
                fair_value = max(0, fair_value + direction * degree)

            trader.neutralise(fair_value)
            trader.take(fair_value, threshold)
            trader.make(fair_value, threshold)

            if trader.orders:
                all_orders[sym] = trader.orders

        self.new_traderData.setdefault(self.symbol, {}).update({"d1": d1, "d2": d2})

        return all_orders | mm


#----------------------------------------------------------------------------------------------------------------------------
#Round 5
class MarketMaker(SimpleTrader): 
    def __init__(self, symbol: str, state: TradingState, new_traderData: dict):
        super().__init__(symbol, state, new_traderData)

    def get_fair_value(self):
        gamma = self.half_spread /2
        adj = gamma * (self.initial_position / (self.position_limit + 1e-6))
        fair_value = self.wall_mid
        return int(fair_value - adj)
    
    def get_orders(self):
        fair_value = self.get_fair_value()
        if fair_value is None:
            return {}
        
        self.neutralise(fair_value)
        self.take(fair_value)
        self.make(fair_value)
        return {self.symbol: self.orders}


#----------------------------------------------------------------------------------------------------------------------------
#Naive ETF Arb for fun
class BasketTrader(SimpleTrader):
    def __init__(self, symbol: str, state: TradingState, new_traderData: dict):
        super().__init__(symbol, state, new_traderData)
        self.composition = ETF_ASSETS.get(self.symbol, {})
        self.part_traders = {sym: ProductTrader(sym, state, new_traderData) for sym in self.composition.keys()}
        
    def get_fair_value(self):
        nav = sum(self.part_traders[part].best_mid * qty for part, qty in self.composition.items())
        return nav
            
    def get_orders(self):
        fair_value = self.get_fair_value()
        if fair_value is None: 
            return {}
            
        fair_value = int(fair_value)
        #self.neutralise(fair_value) ##follow
        #self.take(fair_value)
        self.make(fair_value)
        return {self.symbol: self.orders} 

class HedgeTrader(SimpleTrader): 
    """
    Problem with hedging isn't the cost, its missing out on unhedged returns
    Thus, currently asset trades are not under hedge constraint
    """
    def __init__(self, symbol: str, state: TradingState, new_traderData: dict):
        super().__init__(symbol, state, new_traderData)
        self.composition = ETF_HEDGE.get(self.symbol, {})
        self.asset_traders = {sym: ProductTrader(sym, state, new_traderData) for sym in self.composition.keys()} 

    def get_fair_value(self):
        exposure = self.initial_position + sum(self.asset_traders[asset].initial_position * qty for asset, qty in self.composition.items())
        size = -exposure
        if size > 0:
            size = math.floor(size)
            fair_value = self.wall_ask + 1
        elif size < 0:
            size = math.ceil(size)
            fair_value = self.wall_bid - 1
        else:
            fair_value = None
        return fair_value, size

    def get_orders(self):
        fair_value, size = self.get_fair_value()
        if fair_value is None or size == 0:
            return {}

        ##[ADD LOGIC HERE]
        return {self.symbol: self.orders}  
