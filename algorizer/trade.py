from datetime import datetime, timezone

from .candle import candle_c
from .stream import getRealtimeCandle, createMarker, createLine, isInitializing, getCandle, getMintick, getPrecision
from .constants import constants as c
from . import active

EPSILON = 1e-9
COLOR_BULL = "#68b45c"
COLOR_BEAR = "#d44a4a"

def round_to_tick_size(value, tick_size):
    """Rounds a value to the nearest tick_size."""
    if tick_size == 0:
        return value
    return round(value / tick_size) * tick_size

def floor_to_tick_size(value, tick_size):
    import math
    """Rounds a value to the nearest tick_size."""
    if tick_size == 0:
        return value
    return math.floor(value / tick_size) * tick_size

def cleanFloatJunk(x:float, places=12):
    fmt = '{:.{}f}'.format(x, places)   # 12 digits after the point
    return float(fmt)

from dataclasses import dataclass
@dataclass
class strategy_stats_c:
    total_profit_loss: float = 0.0
    total_winning_positions: int = 0
    total_losing_positions: int = 0
    total_liquidated_positions: int = 0
    total_long_positions: int = 0
    total_winning_long_positions: int = 0
    total_losing_long_positions: int = 0
    total_short_positions: int = 0
    total_winning_short_positions: int = 0
    total_losing_short_positions: int = 0
    total_long_stoploss:int = 0
    total_short_stoploss:int = 0
    initial_liquidity:float = 0.0
    

class strategy_c:
    """
    Trading strategy: manages positions, stats, and capital.
    """
    def __init__(self, initial_liquidity: float = 10000.0, order_size: float = 100.0, max_position_size: float = 100.0, currency_mode: str = 'USD', leverage_long: float = 1.0, leverage_short: float = 1.0):
        self.positions = []
        self.order_size = min(order_size, max_position_size)
        self.max_position_size = max_position_size
        self.currency_mode = currency_mode.upper()
        self.leverage_long = leverage_long
        self.leverage_short = leverage_short
        self.liquidation_enabled = True
        self.maintenance_margin_rate = 0.0066
        self.hedged = True
        self.show_entry_markers = True
        self.liquidity = initial_liquidity
        self.fees_override_maker = 0.0
        self.fees_override_taker = 0.0
        self.stats: strategy_stats_c = strategy_stats_c()
        self.pnl_history: dict[str, list[float]] = {}

        if self.currency_mode not in ['USD', 'BASE']:
            raise ValueError(f"Invalid currency_mode: {currency_mode}. Must be 'USD' or 'BASE'.")

        if self.currency_mode == 'USD' and self.max_position_size > self.liquidity:
            raise ValueError(f"max_position_size ({self.max_position_size}) cannot be greater than initial_liquidity ({self.liquidity}) when currency_mode is 'USD'.")

        if self.liquidity < self.order_size:
            raise ValueError("Initial liquidity must be at least the order size.")

        if self.order_size <= 0:
            raise ValueError("order_size must be a positive value.")
        
        if self.leverage_long < 1 or self.leverage_short < 1:
            raise ValueError("Leverage values must be at least 1.")

    def order(self, order_type: int, pos_type: int, quantity: float, leverage: float, price: float = None)->'position_c':
        if not price:
            price = getRealtimeCandle().close

        pos = self.get_active_position(pos_type)
        if pos:
            return pos.execute_order(order_type, price, quantity, leverage)
        
        # create a new position
        pos = self._new_position(pos_type, leverage)
        order_info = pos.execute_order(order_type, price, quantity, leverage)

        if not order_info.get("error"):
            self.positions.append( pos )

        return order_info

    def _new_position(self, pos_type: int, leverage: float) -> 'position_c':
        # Set up stats if it's the first position ever opened
        if not self.positions:
            self.stats.initial_liquidity = self.liquidity

        pos = position_c(self)
        pos.type = pos_type
        pos.leverage = leverage
        return pos

    def _close_position(self, pos: 'position_c'):
        if pos.active:
            pos.active = False
            pos.stoploss_orders = []
            pos.takeprofit_orders = []

            # update stats
            if pos.type == c.LONG:
                self.stats.total_long_positions += 1
            elif pos.type == c.SHORT:
                self.stats.total_short_positions += 1

            if pos.realized_pnl_quantity > EPSILON:
                self.stats.total_winning_positions += 1
                if pos.type == c.LONG:
                    self.stats.total_winning_long_positions += 1
                elif pos.type == c.SHORT:
                    self.stats.total_winning_short_positions += 1
            elif pos.realized_pnl_quantity < -EPSILON:
                self.stats.total_losing_positions += 1
                if pos.type == c.LONG:
                    self.stats.total_losing_long_positions += 1
                elif pos.type == c.SHORT:
                    self.stats.total_losing_short_positions += 1

            if pos.was_liquidated:
                self.stats.total_liquidated_positions += 1

            self.stats.total_profit_loss += pos.realized_pnl_quantity

            # update monthly pnl.
            # The month includes only the positions closed during that month. Not still opened positions.
            # Use the timestamp of the last order (close) to determine the closing month/year
            if pos.order_history and 'timestamp' in pos.order_history[-1]:
                close_ts = pos.order_history[-1]['timestamp']
                dt = datetime.fromtimestamp(close_ts / 1000, tz=timezone.utc)
                year = str(dt.year)
                month = dt.month - 1  # 0-based index for months
                # Initialize year if not present
                if year not in self.pnl_history:
                    self.pnl_history[year] = [0.0 for _ in range(12)]
                # Add realized pnl to the correct month
                self.pnl_history[year][month] += pos.realized_pnl_quantity


    def get_active_position(self, pos_type: int = None) -> 'position_c':
        if not self.positions:
            return None
        for pos in reversed(self.positions):
            if pos.active:
                if pos_type is None or pos.type == pos_type:
                    return pos
        return None

    def price_update(self, candle: candle_c, realtime: bool = True):
        for pos in self.positions:
            if pos.active:
                pos.price_update(candle, realtime)

    def calculate_fee_taker(self, price: float, quantity: float) -> float:
        fee_taker = max(active.timeframe.stream.fee_taker, 0.0)
        if self.fees_override_taker > 0.0:
            fee_taker = self.fees_override_taker
        return abs(quantity) * price * fee_taker
    
    def calculate_fee_maker(self, price: float, quantity: float) -> float:
        fee_maker = max(active.timeframe.stream.fee_maker, 0.0)
        if self.fees_override_maker > 0.0:
            fee_maker = self.fees_override_maker
        return abs(quantity) * price * fee_maker

    def getMinOrder(self, at_price:float = None):
        ''' calculate minimum order allowed by the exchange. *without leverage*'''
        min_order = active.timeframe.stream.min_order
        # if active.timeframe.stream.min_order_usd:
        #     if at_price is None: 
        #         at_price = round_to_tick_size(getRealtimeCandle().close, getMintick())
        #     min_order = max( min_order, active.timeframe.stream.min_order_usd * at_price )
        return max( min_order, getPrecision() )
    
    def getMaxAvailableQuantity(self, price:float, leverage)->float:
        ''' calculate max quantity we can buy/sell in base currency from liquity'''
        assert(price != None)
        
        precision = max( getPrecision(), EPSILON )
        max_quantity = floor_to_tick_size(((self.liquidity / price) * leverage), precision)
        max_fee = self.calculate_fee_taker(price, max_quantity)
        cl = (max_quantity * price) / leverage # turn it back to dollars.
        while cl + max_fee > self.liquidity and max_quantity > precision:
            max_quantity -= precision
            max_fee = self.calculate_fee_taker(price, max_quantity)
            cl = (max_quantity * price) / leverage # turn it back to dollars.
        return max_quantity
    
    


class position_c:
    """
    Single trading position (long/short). Tracks orders, PnL, and state.
    """
    def __init__(self, strategy_instance: strategy_c):
        self.strategy_instance = strategy_instance
        self.active = False
        self.type = 0
        self.size = 0.0
        self.collateral = 0.0
        self.priceAvg = 0.0
        self.leverage = 1
        self.realized_pnl_quantity = 0.0
        self.order_history = []
        self.max_size_held = 0.0
        self.liquidation_price = 0.0
        self.was_liquidated = False
        self.stoploss_orders = []
        self.takeprofit_orders = []
        self.drawInfo = {
            'oldliquidation': 0,
            'liquidationLine': None
        }

    def calculate_collateral_from_history(self):
        collateral = 0.0
        if self.size < EPSILON:
            return 0.0
        for order_data in self.order_history:
            collateral += order_data.get('collateral_change', 0.0)
        return collateral
    
    def calculate_realized_pnl_from_history(self):
        # doesn't include fees
        pnl = 0.0
        for order_data in self.order_history:
            pnl += order_data.get('pnl', 0.0)
        return pnl
    
    def calculate_fees_from_history(self):
        fees = 0.0
        for order_data in self.order_history:
            fees += order_data.get('fees_cost', 0.0)
        return fees

    def calculate_pnl(self, current_price: float, quantity: float) -> float:
        '''return PnL in quote currency'''
        if quantity < EPSILON:
            return 0.0
        pnl = 0.0
        if self.type == c.LONG:
            pnl = (current_price - self.priceAvg) * quantity
        elif self.type == c.SHORT:
            pnl = (self.priceAvg - current_price) * quantity
        return pnl

    def calculate_liquidation_price(self) -> float:
        '''
        maintenance_margin_ratio: It is used to measure the user's position risk. 
        When it is equal to 100%, the position will be deleveraged or liquidated. 
        The margin ratio = maintenance margin / (position margin + unrealized profit and loss)
        '''
        if self.size < EPSILON or not self.active:
            return 0.0
        
        if not self.strategy_instance.liquidation_enabled or self.leverage < EPSILON :
            return 0.0
        
        MAINTENANCE_MARGIN_RATE = self.strategy_instance.maintenance_margin_rate
        position_value = abs(self.size * self.priceAvg)
        maintenance_margin = position_value * MAINTENANCE_MARGIN_RATE
        position_margin = self.collateral + self.calculate_realized_pnl_from_history() - self.calculate_fees_from_history()

        delta = (maintenance_margin - position_margin) / self.size
        if self.type == c.LONG:
            return round_to_tick_size(self.priceAvg + delta, getMintick())
        elif self.type == c.SHORT:
            return round_to_tick_size(self.priceAvg - delta, getMintick())
        return 0.0

    def execute_order(self, order_type: int, price: float, quantity: float, leverage: float, source:str= 'order'):
        price = round_to_tick_size(price, getMintick())
        if leverage > 1:
            quantity *= leverage
        
        # Determine if order increases or reduces position
        is_increasing = False
        if not self.active:
            self.type = c.LONG if order_type == c.BUY else c.SHORT
            is_increasing = True
        else:
            is_increasing = (order_type == c.BUY and self.type == c.LONG) or (order_type == c.SELL and self.type == c.SHORT)

        # Calculate vaues for the order
        fee = 0.0
        collateral_change = 0.0
        pnl = 0.0
        precision = getPrecision()
        if is_increasing:
            max_quantity = self.strategy_instance.getMaxAvailableQuantity(price, leverage)
            max_fee = self.strategy_instance.calculate_fee_taker(price, max_quantity)

            # we now have a max quantity we can purchase in base currency. Anything below is valid
            if quantity > max_quantity:
                quantity = max_quantity
                fee = max_fee
            else:
                quantity = floor_to_tick_size(quantity, precision)
                fee = self.strategy_instance.calculate_fee_taker(price, quantity)

            # clamp to max_position
            # I don't like that we are using dollars here. I'd like to keep this dollar free except for liquidity
            if self.strategy_instance.max_position_size > 0.0:
                if self.strategy_instance.currency_mode == 'USD':
                    value = (self.size * self.priceAvg) / leverage
                    cost = (quantity * price) / leverage
                    if value + cost > self.strategy_instance.max_position_size:
                        cost = self.strategy_instance.max_position_size - value
                        quantity = round_to_tick_size(((cost / price) * leverage), getPrecision())
                elif self.size + quantity > self.strategy_instance.max_position_size:
                    quantity = round_to_tick_size(self.strategy_instance.max_position_size - self.size, getPrecision())


            collateral_change = (quantity * price) / leverage

            if quantity < self.strategy_instance.getMinOrder():
                return {"error": "minorder"}
            
            pnl = 0.0
            
            # we have the data to create the order
        
        else:
            # Clamp quantity if reducing position
            quantity = min(quantity, self.size)
            if quantity != self.size:
                quantity = round_to_tick_size(quantity, getPrecision())
            collateral_change = (quantity * self.priceAvg) / self.leverage
            if collateral_change > self.collateral:
                collateral_change = self.collateral # Float ERROR
            collateral_change = -collateral_change

            # there is minimum when reducing other than the unit size
            if quantity < getPrecision():
                return {"error": "minorder"}
            
            fee = self.strategy_instance.calculate_fee_taker(price, quantity)
            pnl = self.calculate_pnl(price, quantity)

            # we have the data to create the order

        # I don't want to import decimal. I could, but I don't wanna
        quantity = cleanFloatJunk( quantity )
        collateral_change = cleanFloatJunk( collateral_change )
        fee = cleanFloatJunk( fee )
        pnl = cleanFloatJunk( pnl )
        
        # Store order in history
        order_info = {
            'type': order_type,
            'price': price,
            'quantity': quantity,
            'collateral_change': collateral_change,
            'leverage': leverage,
            'barindex': active.barindex,
            'timestamp': active.timeframe.timestamp,
            'fees_cost': fee,
            'pnl': pnl,
            'source': source
        }
        self.order_history.append(order_info)


        # Apply the order to the position status
        if is_increasing:
            new_size = self.size + order_info['quantity']
            self.priceAvg = ((self.priceAvg * self.size) + (price * order_info['quantity'])) / new_size
            self.size = new_size
        else:
            self.size -= order_info['quantity']
        self.collateral += order_info['collateral_change']
        self.strategy_instance.liquidity += -order_info['collateral_change'] # when reducing collateral_change is negative so it will be added
        self.strategy_instance.liquidity += order_info['pnl']
        self.strategy_instance.liquidity -= order_info['fees_cost']

        self.max_size_held = max(self.max_size_held, self.size)
        self.leverage = leverage if not self.active else self.leverage # FIXME: Allow to combine orders with different leverages

        # it's going to close the position after the broker event
        if self.size < EPSILON:
            self.size = 0.0
            self.collateral = 0.0
            self.was_liquidated = source == 'liquidation_trigger'

        # Broker event
        if not isInitializing():
            quantity_dollars = quantity * price
            position_size_base = self.size
            position_size_dollars = self.size * price
            info = {
                "order_type": order_type,
                "order_quantity": quantity,
                "order_quantity_dollars": quantity_dollars,
                "position_type": self.type,
                "position_size": position_size_base,
                "position_size_dollars": position_size_dollars,
                "leverage": leverage,
                "price": price,
                "source": source
            }
            active.timeframe.stream.broker_event( info )
        
        if self.size < EPSILON: # The order has emptied the position
            self.realized_pnl_quantity = self.calculate_realized_pnl_from_history() - self.calculate_fees_from_history()
            self.strategy_instance._close_position(self)
            return order_info
        
        self.active = True
        self.liquidation_price = self.calculate_liquidation_price()
        return order_info

    def close(self):
        if not self.active or self.size < EPSILON:
            return { "error": "noposition" }
        order_type = c.BUY if self.type == c.SHORT else c.SELL
        price = getRealtimeCandle().close
        order_info = self.execute_order(order_type, price, self.size, self.leverage)
        if not order_info.get("error"):
            marker(self)
        return order_info

    def check_stoploss(self, stoploss_order, candle:candle_c)->bool:
        price = stoploss_order.get('price')
        loss_pct = stoploss_order.get('loss_pct')
        if price:
            if self.type == c.LONG and candle.low > price:
                return False
            if self.type == c.SHORT and candle.high < price:
                return False
        if loss_pct:
            directional_price = candle.high if self.type == c.SHORT else candle.low
            pnl = self.calculate_pnl(directional_price, self.size)
            pnl = (pnl / abs(self.collateral)) * 100
            if pnl >= 0.0:
                return False
            if abs(pnl) < loss_pct:
                return False
        assert( price or loss_pct )

        if loss_pct:
            print( f"SL triggered: pnl:{pnl} trigger:{loss_pct} Entry:{self.priceAvg}")
        return True
    
    def check_takeprofit(self, tp_order, candle:candle_c)->bool:
        price = tp_order.get('price')
        win_pct = tp_order.get('win_pct')
        if price:
            if self.type == c.LONG and candle.high < price:
                return False
            if self.type == c.SHORT and candle.low > price:
                return False
        if win_pct:
            directional_price = candle.low if self.type == c.SHORT else candle.high
            pnl = self.calculate_pnl(directional_price, self.size)
            pnl = (pnl / abs(self.collateral)) * 100
            if pnl <= 0.0:
                return False
            if abs(pnl) < win_pct:
                return False
        assert( price or win_pct )

        if win_pct:
            print( f"SL triggered: pnl:{pnl} trigger:{win_pct} Entry:{self.priceAvg}")
        return True


    def price_update(self, candle:candle_c, realtime: bool = True):
        '''a tick with a price update has happened. Update the things to be updated in real time'''
        if not self.active:
            return
        
        # check take profit
        #
        triggered = []
        for tp_order in self.takeprofit_orders:
            if self.check_takeprofit( tp_order, candle ):
                order_type = c.BUY if self.type == c.SHORT else c.SELL
                quantity = tp_order.get('quantity')
                quantity_pct = tp_order.get('quantity_pct')

                closing_price = candle.close
                if not realtime and tp_order.get('price'):
                    closing_price = tp_order.get('price')
                
                if quantity:
                    self.execute_order(order_type, closing_price, quantity / self.leverage, self.leverage, source= 'takeprofit_trigger')
                else:
                    assert(quantity_pct)
                    quantity = self.size * (quantity_pct / 100)
                    self.execute_order(order_type, closing_price, quantity / self.leverage, self.leverage, source= 'takeprofit_trigger')
                marker( self, prefix=f'TP({quantity:.2f}):' )

                if not self.active:
                    break
                triggered.append( tp_order )
        
        if not self.active: # if the position was closed no need to continue
            return
        
        for s in triggered:
            self.takeprofit_orders.remove(s)

        # check stoploss
        #
        triggered = []
        for stoploss_order in self.stoploss_orders:
            if self.check_stoploss( stoploss_order, candle ):
                order_type = c.BUY if self.type == c.SHORT else c.SELL
                quantity = stoploss_order.get('quantity')
                quantity_pct = stoploss_order.get('quantity_pct')

                closing_price = candle.close
                if not realtime and stoploss_order.get('price'):
                    closing_price = stoploss_order.get('price')

                line = stoploss_order.get("line")
                if line and line.y1 == closing_price:
                    line.style = 'solid'
                    line.width = 3
                
                if quantity:
                    self.execute_order(order_type, closing_price, quantity / self.leverage, self.leverage, source= 'stoploss_trigger')
                else:
                    assert(quantity_pct)
                    quantity = self.size * (quantity_pct / 100)
                    self.execute_order(order_type, closing_price, quantity / self.leverage, self.leverage, source= 'stoploss_trigger')
                marker( self, prefix='Stoploss ⛔' )
                if self.type == c.SHORT:
                    self.strategy_instance.stats.total_short_stoploss += 1
                else:
                    self.strategy_instance.stats.total_long_stoploss += 1
                
                if not self.active:
                    break
                triggered.append( stoploss_order )

        if not self.active: # if the position was closed no need to continue
            return
        
        for s in triggered:
            self.stoploss_orders.remove(s)
        
        # check liquidation
        #
        
        self.liquidation_price = self.calculate_liquidation_price()
        if self.liquidation_price > EPSILON:
            if (self.type == c.LONG and candle.low < self.liquidation_price) or \
            (self.type == c.SHORT and candle.high > self.liquidation_price):
                line = self.drawInfo['liquidationLine']
                if line and line.y1 == self.liquidation_price:
                    line.style = 'solid'
                    line.width = 3
                order_type = c.BUY if self.type == c.SHORT else c.SELL
                self.execute_order(order_type, self.liquidation_price, self.size, self.leverage, source= 'liquidation_trigger')
                marker( self, prefix = '💀 ' )
                return # with the position liquidated there's no need to continue

    def get_unrealized_pnl(self) -> float:
        current_price = round_to_tick_size(getRealtimeCandle().close, getMintick())
        return self.calculate_pnl(current_price, self.size)

    def get_unrealized_pnl_percentage(self) -> float:
        unrealized_pnl_q = self.get_unrealized_pnl()
        if abs(unrealized_pnl_q) < EPSILON or abs(self.collateral) < EPSILON:
            return 0.0
        return (unrealized_pnl_q / abs(self.collateral)) * 100
        
    def get_order_by_direction(self, order_direction: int, older_than_bar_index: int = None) -> dict:
        for order_data in reversed(self.order_history):
            if order_data['type'] == order_direction:
                if older_than_bar_index is None or order_data['barindex'] < older_than_bar_index:
                    return order_data
        return None
    
    def createTakeprofit(self, price:float = None, quantity:float = None, win_pct:float = None, reduce_pct = None)->dict:
        ''' quantity is in *base currency*.
            quantity_pct is a percentage in a 0-100 scale'''
        if not price and not win_pct:
            print( "Warning: Stoploss order requires a price or a percentage. Ignoring")
            return None
        
        # if quantityUSDT and self.strategy_instance.currency_mode == 'USD': # convert it to base currency
        #     quantity = quantityUSDT / price
        #     reduce_pct = None
        
        if quantity:
            quantity = min(self.size, max(0, quantity))
            if quantity > EPSILON:
                reduce_pct = None
            else:
                quantity = None

        if reduce_pct and quantity == None:
            reduce_pct = min(100.0, max(1.0, reduce_pct)) if reduce_pct else 100.0
            if reduce_pct < 100.0:
                quantity = round_to_tick_size( self.size * (reduce_pct / 100.0), getPrecision() )
            else:
                quantity = self.size
        else:
            quantity = self.size

        

        # see if we have another TP in the same price
        # if we do we update it with the new values and return it
        for tp in self.takeprofit_orders:
            if abs( tp['price'] - price ) < EPSILON:
                tp['quantity'] = quantity
                tp['quantity_pct'] = reduce_pct
                tp ['win_pct'] = win_pct
                return tp

        # create the takeprofit item
        tp_order = {
            'price': price,
            'quantity': quantity,
            'quantity_pct': reduce_pct,
            'win_pct': win_pct,
            'line': None
        }

        self.takeprofit_orders.append( tp_order )

        # Broker event
        if not isInitializing():
            order_type = c.SELL if self.type == c.LONG else c.BUY
            quantity_dollars = quantity * price
            position_size_base = self.size
            position_size_dollars = self.size * price
            info = {
                "order_type": order_type,
                "order_quantity": quantity,
                "order_quantity_dollars": quantity_dollars,
                "position_type": self.type,
                "position_size": position_size_base,
                "position_size_dollars": position_size_dollars,
                "leverage": self.leverage,
                "price": price,
                "source": 'takeprofit_create'
            }
            active.timeframe.stream.broker_event( info )

        return tp_order
    
    def createStoploss(self, price:float = None, quantity:float = None, loss_pct:float = None, reduce_pct = None)->dict:
        ''' quantity is in base currency.
            quantity_pct is a percentage in a 0-100 scale'''
        if not price and not loss_pct:
            print( "Warning: Stoploss order requires a price or a percentage. Ignoring")
            return None
        
        # if quantityUSDT and self.strategy_instance.currency_mode == 'USD': # convert it to base currency
        #     quantity = quantityUSDT / price
        #     reduce_pct = None
        
        if quantity:
            quantity = min(self.size, max(0, quantity))
            if quantity > EPSILON:
                reduce_pct = None
            else:
                quantity = None

        if reduce_pct and quantity == None:
            reduce_pct = min(100.0, max(1.0, reduce_pct)) if reduce_pct else 100.0
            if reduce_pct < 100.0:
                quantity = round_to_tick_size( self.size * (reduce_pct / 100.0), getPrecision() )
            else:
                quantity = self.size
        else:
            quantity = self.size

        # see if we have another SL in the same price
        # if we do we update it with the new values and return it
        for sl in self.stoploss_orders:
            if abs( sl['price'] - price ) < EPSILON:
                sl['quantity'] = quantity
                sl['quantity_pct'] = reduce_pct
                sl ['loss_pct'] = loss_pct
                return sl

        # create the stoploss item
        stoploss_order = {
            'price': price,
            'quantity': quantity,
            'quantity_pct': reduce_pct,
            'loss_pct': loss_pct,
            'line': None
        }

        self.stoploss_orders.append( stoploss_order )

        # Broker event
        if not isInitializing():
            order_type = c.BUY if self.type == c.LONG else c.SELL
            quantity_dollars = quantity * price
            position_size_base = self.size
            position_size_dollars = self.size * price
            info = {
                "order_type": order_type,
                "order_quantity": quantity,
                "order_quantity_dollars": quantity_dollars,
                "position_type": self.type,
                "position_size": position_size_base,
                "position_size_dollars": position_size_dollars,
                "leverage": self.leverage,
                "price": price,
                "source": 'stoploss_create'
            }
            active.timeframe.stream.broker_event( info )

        return stoploss_order
    
    def drawStoploss( self, color= "#e38100", style = 'dotted', width= 2 ):
        if self.stoploss_orders:
            for sl in self.stoploss_orders: # the script doesn't do more than one at a time, but just in case I change it
                stoplossprice = sl.get("price")
                line = sl.get("line")
                if line is None:
                    line = createLine( active.barindex,
                                        stoplossprice, 
                                        active.barindex + 5,
                                        stoplossprice,
                                        color= color,
                                        style= style,
                                        width= width )
                    sl['line'] = line

                # keep updating it
                if line and active.barindex >= line.x2:
                    line.x2 = active.barindex + 1

    def drawTakeprofit( self, color= "#17c200", style = 'dotted', width= 2 ):
        if self.takeprofit_orders:
            for tp in self.takeprofit_orders: # the script doesn't do more than one at a time, but just in case I change it
                takeprofitprice = tp.get("price")
                line = tp.get("line")
                if line is None:
                    line = createLine( active.barindex,
                                        takeprofitprice, 
                                        active.barindex + 5,
                                        takeprofitprice,
                                        color= color,
                                        style= style,
                                        width= width )
                    tp['line'] = line

                # keep updating it
                if line and active.barindex >= line.x2:
                    line.x2 = active.barindex + 1

    def drawLiquidation(self, color= "#a00000", style = 'dotted', width= 2):
        if not self.active or not self.strategy_instance.liquidation_enabled:
            return
        
        if self.leverage > 0:
            liquidationLine = self.drawInfo['liquidationLine']
            oldLiquidationPrice = self.drawInfo['oldliquidation']

            if liquidationLine == None or oldLiquidationPrice != self.liquidation_price:
                liquidationLine = createLine( self.order_history[-1]['barindex'],
                                                    self.liquidation_price, 
                                                    active.barindex + 5,
                                                    self.liquidation_price,
                                                    color = "#a00000",
                                                    style='dotted',
                                                    width=2 )
                self.drawInfo['liquidationLine'] = liquidationLine
                self.drawInfo['oldliquidation'] = self.liquidation_price

            # keep updating it
            if active.barindex >= liquidationLine.x2:
                liquidationLine.x2 = active.barindex + 1


strategy = strategy_c(currency_mode='USD')



def newTick(candle: candle_c, realtime: bool = True):
    strategy.price_update(candle, realtime)

def marker( pos:position_c, message = None, prefix = '', reversal:bool = False ):
    if strategy.show_entry_markers and pos:
        order = pos.order_history[-1]
        if order['quantity'] < EPSILON:
            return
        newposition = len(pos.order_history) == 1
        closedposition = pos.active == False
        order_type = int(order['type'])
        order_cost = order['collateral_change']

        shape = 'arrow_up' if order_type == c.BUY else 'arrow_down'
        if newposition:
            shape = 'circle'
        elif closedposition == True:
            shape = 'square'

        if not message:
            if closedposition:
                pnl = pos.calculate_realized_pnl_from_history() - pos.calculate_fees_from_history()
                if prefix == '':
                    prefix = 'Close '
                if '💀' in prefix or '⛔' in prefix:
                    message = f" pnl:{pnl:.2f}"
                else:
                    message = f"{'🚩' if pnl < 0.0 else '💲'} pnl:{pnl:.2f}"
            else:
                order_name = 'buy' if order_type == c.BUY else 'sell'
                message = f"{order_name}:${abs(order_cost):.2f} (pos:{pos.size:.3f})"

        location = 'below' if order_type == c.BUY else 'above'
        if pos.was_liquidated or '💀' in prefix or '⛔' in prefix:
            location = 'below' if order_type == c.SHORT else 'above'
        
        createMarker( prefix + message,
                    location,
                    shape,
                    COLOR_BULL if pos.type == c.LONG else COLOR_BEAR
                    )

def getActivePosition(pos_type: int = None) -> 'position_c':
    return strategy.get_active_position(pos_type)


def order(cmd: str|int, target_position_type:int= None, quantity:float= None, leverage:float= None):
    if isinstance( cmd, str ):
        order_type = c.BUY if cmd.lower() == 'buy' else c.SELL if cmd.lower() == 'sell' else None
    elif isinstance( cmd, int ):
        order_type = cmd if cmd == c.BUY or c.SELL else None
    else:
        order_type = None

    if not order_type:
        raise ValueError(f"Invalid order command: {cmd}")
    
    if not target_position_type:
        if strategy.hedged:
            raise ValueError( f"in hedged mode orders must have a position type assigned" )
        active_pos = strategy.get_active_position()
        if active_pos:
            target_position_type = active_pos.type
        else:
            target_position_type = c.LONG if order_type == c.BUY else c.SHORT

    if target_position_type != c.LONG and target_position_type != c.SHORT:
        raise ValueError( f"Invalid position type: {target_position_type}" )

    selected_leverage = leverage if leverage is not None else (strategy.leverage_long if target_position_type == c.LONG else strategy.leverage_short)
    selected_leverage = max(selected_leverage, 1)
    current_price = round_to_tick_size(getRealtimeCandle().close, getMintick())

    if quantity is None:
        quantity = strategy.order_size

    # if we are trading in USDT convert the quantity to base units
    if strategy.currency_mode == 'USD':
        quantity = quantity/current_price # we don't adjust to precision yet

    actual_quantity_base_units = quantity

    # We have cleaned the inputs.
    # We now handle the oneway/hedged situation


    order = None

    if strategy.hedged: # 'HEDGE'
        order = strategy.order(order_type, target_position_type, actual_quantity_base_units, selected_leverage)
        assert order, "strategy.order must always return a dict"
        if not order.get("error"):
            marker( strategy.get_active_position(target_position_type) )
    else:  # ONEWAY
        active_pos = strategy.get_active_position()
        if not active_pos or active_pos.type == target_position_type:
            order = strategy.order(order_type, target_position_type, actual_quantity_base_units, selected_leverage)
            assert order, "strategy.order must always return a dict"
            if not order.get("error"):
                marker( strategy.get_active_position() )
        else:
            pos_size = active_pos.size
            if quantity == None or actual_quantity_base_units >= pos_size - EPSILON:
                active_pos.close()
                if active_pos.active == False:
                    marker(active_pos)
                
                if quantity == None:
                    remaining_quantity = actual_quantity_base_units
                else:
                    remaining_quantity = actual_quantity_base_units - pos_size
                if remaining_quantity > strategy.getMinOrder():
                    order = strategy.order(order_type, target_position_type, remaining_quantity, selected_leverage)
                    assert order, "strategy.order must always return a dict"
                    if not order.get("error"):
                        marker( strategy.get_active_position(), reversal= True )
            else:
                order = strategy.order(order_type, active_pos.type, actual_quantity_base_units, selected_leverage)
                assert order, "strategy.order must always return a dict"
                if not order.get("error"):
                    marker( strategy.get_active_position() )

    return order


def close(pos_type: int = None):
    mode = 'HEDGE' if strategy.hedged else 'ONEWAY'
    if mode == 'HEDGE' and not pos_type:
        raise ValueError("A position type is required in Hedge mode")
    if mode == 'ONEWAY':
        pos = strategy.get_active_position()
        if pos:
            pos.close()
    elif mode == 'HEDGE':
        if pos_type not in (c.LONG, c.SHORT):
            raise ValueError("Invalid position type")
        pos = strategy.get_active_position(pos_type)
        if pos:
            pos.close()


def createFakePosition( entry_price, position_type, quantity, leverage ):
    actual_quantity_base_units = quantity if quantity is not None else strategy.order_size
    if strategy.currency_mode == 'USD':
        actual_quantity_base_units = actual_quantity_base_units / entry_price if entry_price > EPSILON else 0.0
    strategy.order(position_type, position_type, actual_quantity_base_units, leverage, price=entry_price)
    marker( strategy.get_active_position() )


def print_summary_stats():
    """
    Print summary of strategy performance.
    """
    
    print(f"\n--- Strategy Summary Stats ---")
    
    # PnL percentage: percent change from initial_liquidity to current balance
    longpos = getActivePosition(c.LONG)
    shortpos = getActivePosition(c.SHORT)
    balance = strategy.liquidity
    balance += longpos.calculate_collateral_from_history() + longpos.get_unrealized_pnl() if longpos else 0.0
    balance += shortpos.calculate_collateral_from_history() + shortpos.get_unrealized_pnl() if shortpos else 0.0
    pnl_percentage_vs_liquidity = (balance - strategy.stats.initial_liquidity) / strategy.stats.initial_liquidity * 100 if strategy.stats.initial_liquidity != 0 else 0.0

    pnlStr = f"{balance - strategy.stats.initial_liquidity:.2f} ({pnl_percentage_vs_liquidity:.2f}%)" if strategy.liquidity > EPSILON else "Your account has been terminated."
    print(f"{active.timeframe.stream.symbol.split(':', 1)[0]:<10} {'Order size':<11} {'Max Position':<13} {'Initial Liquidity':<18} {'Final Liquidity':<16} {'Account PnL':<18}")
    print(f"{'':<10} {strategy.order_size:<11} {strategy.max_position_size:<13} {strategy.stats.initial_liquidity:<18} {strategy.liquidity:<16.2f} {pnlStr:<18}")
    print("------------------------------")

    # Calculate metrics
    total_closed_positions = strategy.stats.total_winning_positions + strategy.stats.total_losing_positions
    
    profitable_trades = strategy.stats.total_winning_positions
    losing_trades = strategy.stats.total_losing_positions
    
    percentage_profitable_trades = (profitable_trades / total_closed_positions) * 100 if total_closed_positions > 0 else 0.0

    long_win_ratio = (strategy.stats.total_winning_long_positions / strategy.stats.total_long_positions) * 100 if strategy.stats.total_long_positions > 0 else 0.0
    short_win_ratio = (strategy.stats.total_winning_short_positions / strategy.stats.total_short_positions) * 100 if strategy.stats.total_short_positions > 0 else 0.0

    long_sl_pct = (strategy.stats.total_long_stoploss / total_closed_positions) * 100 if total_closed_positions > 0 else 0.0
    short_sl_pct = (strategy.stats.total_short_stoploss / total_closed_positions) * 100 if total_closed_positions > 0 else 0.0
    liquidated_pct = (strategy.stats.total_liquidated_positions / total_closed_positions) * 100 if total_closed_positions > 0 else 0.0

    long_sl_str = f"{strategy.stats.total_long_stoploss} ({long_sl_pct:.1f}%)"
    short_sl_str = f"{strategy.stats.total_short_stoploss} ({short_sl_pct:.1f}%)"
    liquidated_str = f"{strategy.stats.total_liquidated_positions} ({liquidated_pct:.1f}%)"

    print(f"{'Trades':<8} {'Wins':<8} {'Losses':<8} {'Win Rate %':<12} {'Long Win %':<12} {'Short Win %':<12} {'Long SL':<15} {'Short SL':<15} {'Liquidated':<15}")
    print(f"{total_closed_positions:<8} {profitable_trades:<8} {losing_trades:<8} {percentage_profitable_trades:<12.2f} {long_win_ratio:<12.2f} {short_win_ratio:<12.2f} {long_sl_str:<15} {short_sl_str:<15} {liquidated_str:<15}")
    print("------------------------------")



def print_pnl_by_period_summary( quarter_pnl_relative_to_max_position = False ):
    """
    Print realized PnL by month and year using stats.pnl_history. Does not include unrealized PnL.
    quarter_pnl_relative_to_max_position : If false it will be relative to initial liquidity
    """
    from colorama import Fore, Style, init as colorama_init
    colorama_init() # Initialize Colorama for Windows console compatibility



    
    print("\n--- PnL By Period (Realized Only) ---")
    pnl_history = strategy.pnl_history
    if not pnl_history:
        print("No closed positions to report PnL by period.")
        return
    
    # Flatten all quarters into a list of (year, quarter_index, pnl, months)
    quarters = []
    for year in sorted(pnl_history.keys()):
        months = pnl_history[year]
        quarters.append((year, 1, sum(months[0:3]), months[0:3]))
        quarters.append((year, 2, sum(months[3:6]), months[3:6]))
        quarters.append((year, 3, sum(months[6:9]), months[6:9]))
        quarters.append((year, 4, sum(months[9:12]), months[9:12]))

    # Find first quarter with nonzero pnl
    first_trade_idx = None
    for i, (_, _, pnl, _) in enumerate(quarters):
        if abs(pnl) > EPSILON:
            first_trade_idx = i
            break

    # Find last quarter with any trade (nonzero pnl)
    last_trade_idx = None
    for i in range(len(quarters)-1, -1, -1):
        if abs(quarters[i][2]) > EPSILON:
            last_trade_idx = i
            break

    # If no trades, print nothing
    if first_trade_idx is None or last_trade_idx is None:
        print("No closed positions to report PnL by period.")
        return

    # Optionally skip last quarter from average unless it's in the third month
    last_q_months = quarters[last_trade_idx][3]
    
    # Check if the last quarter with trades is the current ongoing quarter
    is_ongoing_quarter = True
    try:
        if active.timeframe and active.timeframe.timestamp:
            dt_now = datetime.fromtimestamp(active.timeframe.timestamp / 1000, tz=timezone.utc)
            current_year = dt_now.year
            current_q = (dt_now.month - 1) // 3 + 1
            
            last_q_year = int(quarters[last_trade_idx][0])
            last_q_idx = quarters[last_trade_idx][1]
            
            if last_q_year < current_year or (last_q_year == current_year and last_q_idx < current_q):
                is_ongoing_quarter = False
    except:
        pass # Fallback to treating it as ongoing if we can't determine time

    last_quarter_incomplete = is_ongoing_quarter and abs(last_q_months[2]) < EPSILON
    last_idx_to_use = last_trade_idx

    # Print header
    print(f"{'Year':<5} {f'PnL {strategy.currency_mode}':>12} | {'Q1':>12} {'Q2':>12} {'Q3':>12} {'Q4':>12} ")

    # Print by year, quarters and total in one line
    # Also collect stats for average
    numQuarters = 0
    allQuarters = 0.0
    year_quarter_pnls = {}
    for i in range(first_trade_idx, last_idx_to_use+1):
        year, qidx, pnl, months = quarters[i]
        if year not in year_quarter_pnls:
            year_quarter_pnls[year] = [None, None, None, None]
        year_quarter_pnls[year][qidx-1] = pnl
        # Only include in average if not the last (incomplete) quarter
        if not (last_quarter_incomplete and i == last_idx_to_use):
            numQuarters += 1
            allQuarters += pnl

    # Print each year
    def c_pnl(val):
        s = f"{val:12.2f}"
        if val < -EPSILON:
            return f"{Fore.RED}{s}{Style.RESET_ALL}"
        return s

    for year in sorted(year_quarter_pnls.keys()):
        qpnls = year_quarter_pnls[year]
        total = sum([p if p is not None else 0.0 for p in qpnls])
        q1 = qpnls[0] if qpnls[0] is not None else 0.0
        q2 = qpnls[1] if qpnls[1] is not None else 0.0
        q3 = qpnls[2] if qpnls[2] is not None else 0.0
        q4 = qpnls[3] if qpnls[3] is not None else 0.0
        print(f"{year:<5} {c_pnl(total)} | {c_pnl(q1)} {c_pnl(q2)} {c_pnl(q3)} {c_pnl(q4)}")
    print("-----------------------------")
    avgq = allQuarters/numQuarters if numQuarters > 0 else 0.0
    base = strategy.max_position_size if quarter_pnl_relative_to_max_position else strategy.stats.initial_liquidity
    text = 'max_position_size' if quarter_pnl_relative_to_max_position else 'initial_liquidity'
    avgqpct = (avgq / base) * 100 if base > EPSILON else 0.0

    avgq_str = f"{avgq:.2f}"
    if avgq < -EPSILON:
        avgq_str = f"{Fore.RED}{avgq_str}{Style.RESET_ALL}"
    
    avgqpct_str = f"{avgqpct:.1f}%"
    if avgqpct < -EPSILON:
        avgqpct_str = f"{Fore.RED}{avgqpct_str}{Style.RESET_ALL}"

    print(f"Average PnL per quarter: {avgq_str} ({avgqpct_str} relative to {text})")
    print("-----------------------------")