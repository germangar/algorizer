# THIS FILE HAS BEEN WRITTEN BY GEMINI AI 
# I find this annoying to write so I just asked the AI to do it. 

from datetime import datetime, timezone

from .candle import candle_c
from .algorizer import getRealtimeCandle, createMarker, isInitializing, getCandle, getMintick, getPrecision, getFees
from .constants import c
from . import active # Corrected: Import active to get active.barindex

from datetime import datetime, timezone

from .candle import candle_c
from .algorizer import getRealtimeCandle, createMarker, isInitializing, getCandle, getMintick, getPrecision, getFees
from .constants import c
from . import active

EPSILON = 1e-9
COLOR_BULL = "#3bd818"
COLOR_BEAR = "#ed3535"

def round_to_tick_size(value, tick_size):
    """Rounds a value to the nearest tick_size."""
    if tick_size == 0:
        return value
    return round(value / tick_size) * tick_size

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
    

class strategy_c:
    """
    Trading strategy: manages positions, stats, and capital.
    """
    def __init__(self, initial_liquidity: float = 10000.0, order_size: float = 100.0, max_position_size: float = 100.0, currency_mode: str = 'USD', leverage_long: float = 1.0, leverage_short: float = 1.0):
        self.positions = []
        self.initial_liquidity = initial_liquidity
        self.order_size = min(order_size, max_position_size)
        self.max_position_size = max_position_size
        self.currency_mode = currency_mode.upper()
        self.leverage_long = leverage_long
        self.leverage_short = leverage_short
        self.maintenance_margin_rate = 0.0066
        self.hedged = True
        self.show_markers = True
        self.liquidity = initial_liquidity
        self.stats: strategy_stats_c = strategy_stats_c()
        self.pnl_history: dict[str, list[float]] = {}

        if self.currency_mode not in ['USD', 'BASE']:
            raise ValueError(f"Invalid currency_mode: {currency_mode}. Must be 'USD' or 'BASE'.")

        if self.currency_mode == 'USD' and self.max_position_size > self.liquidity:
            raise ValueError(f"max_position_size ({self.max_position_size}) cannot be greater than initial_liquidity ({self.liquidity}) when currency_mode is 'USD'.")

        if self.initial_liquidity < self.order_size:
            raise ValueError("Initial liquidity must be at least the order size.")

        if self.order_size <= 0:
            raise ValueError("order_size must be a positive value.")
        
        if self.leverage_long < 1 or self.leverage_short < 1:
            raise ValueError("Leverage values must be at least 1.")

    def order(self, order_type: int, pos_type: int, quantity: float, leverage: float, price: float = None)->'position_c':
        pos = self.get_active_position(pos_type)
        if not pos:
            pos = self._new_position(pos_type, leverage)
        if not price:
            price = getRealtimeCandle().close
        return pos.execute_order(order_type, price, quantity, leverage)

    def _new_position(self, pos_type: int, leverage: float) -> 'position_c':
        pos = position_c(self)
        pos.type = pos_type
        pos.leverage = leverage
        self.positions.append(pos)
        return pos

    def _close_position(self, pos: 'position_c'):
        if pos.active:
            pos.active = False

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

    def check_liquidation(self, candle: candle_c, realtime: bool = True):
        for pos in self.positions:
            if pos.active:
                pos.check_liquidation_and_close(candle.close, realtime)


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
        self.realized_pnl_percentage = 0.0
        self.order_history = []
        self.max_size_held = 0.0
        self.liquidation_price = 0.0
        self.was_liquidated = False

    def calculate_collateral_from_history(self):
        collateral = 0.0
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
        # doesn't include fees
        fees = 0.0
        for order_data in self.order_history:
            fees += order_data.get('fees_cost', 0.0)
        return fees

    def calculate_pnl(self, current_price: float, quantity: float = None) -> tuple[float, float]:
        if quantity is None:
            quantity = self.size
        if quantity < EPSILON:
            return 0.0, 0.0
        pnl_q = 0.0
        if self.type == c.LONG:
            pnl_q = (current_price - self.priceAvg) * quantity * self.leverage
        elif self.type == c.SHORT:
            pnl_q = (self.priceAvg - current_price) * quantity * self.leverage
        pnl_pct = (pnl_q / self.collateral) * 100 if self.collateral > EPSILON else 0.0
        return pnl_q, pnl_pct

    def calculate_fee_taker(self, price: float, quantity: float) -> float:
        _, taker_fee = getFees()
        return abs(quantity) * price * taker_fee
    
    def calculate_fee_maker(self, price: float, quantity: float) -> float:
        maker_fee, _ = getFees()
        return abs(quantity) * price * maker_fee

    def calculate_liquidation_price(self) -> float:
        '''
        maintenance_margin_ratio: It is used to measure the user's position risk. 
        When it is equal to 100%, the position will be deleveraged or liquidated. 
        The margin ratio = maintenance margin / (position margin + unrealized profit and loss)
        '''
        if self.size < EPSILON or not self.active:
            return 0.0
        
        if self.leverage <= 1: #FIXME? we assume leverage 1 is spot. It may not be correct, but who would use leverage 1 in futures?
            return 0.0
        
        MAINTENANCE_MARGIN_RATE = self.strategy_instance.maintenance_margin_rate

        position_value = abs(self.size * self.priceAvg)
        maintenance_margin = position_value * MAINTENANCE_MARGIN_RATE
        position_margin = self.collateral

        realized_pnl = self.calculate_realized_pnl_from_history()
        fees = self.calculate_fees_from_history()
        exit_fee = self.order_history[0].get('fees_cost', 0.0)
        position_margin = self.collateral + realized_pnl - fees - exit_fee

        delta = (maintenance_margin - position_margin) / (self.size)
        if self.type == c.LONG:
            return round_to_tick_size(self.priceAvg + delta, getMintick())
        elif self.type == c.SHORT:
            return round_to_tick_size(self.priceAvg - delta, getMintick())
        return 0.0

    def check_liquidation_and_close(self, current_price: float, realtime: bool = True):
        if not self.active or self.size < EPSILON:
            return
        self.liquidation_price = self.calculate_liquidation_price()
        if self.liquidation_price < EPSILON:
            return
        if (self.type == c.LONG and current_price <= self.liquidation_price) or \
           (self.type == c.SHORT and current_price >= self.liquidation_price):
            order_type = c.BUY if self.type == c.SHORT else c.SELL
            self.execute_order(order_type, self.liquidation_price, self.size, self.leverage, liquidation= True)
            marker( self, message = 'Liquidation' )

    def execute_order(self, order_type: int, price: float, quantity: float, leverage: float, liquidation:bool = False):
        price = round_to_tick_size(price, getMintick())
        if leverage > 1:
            quantity *= leverage
        quantity = round_to_tick_size(quantity, getPrecision())
        if quantity < EPSILON:
            return None

        # Determine if order increases or reduces position
        is_increasing = False
        if not self.active:
            self.type = c.LONG if order_type == c.BUY else c.SHORT
            is_increasing = True
        else:
            is_increasing = (order_type == c.BUY and self.type == c.LONG) or (order_type == c.SELL and self.type == c.SHORT)

        # Calculate collateral required
        collateral_change = 0.0
        if is_increasing:
            collateral_change = (quantity * price) / leverage
            if self.strategy_instance.currency_mode == 'USD' and collateral_change > self.strategy_instance.liquidity:
                quantity = (self.strategy_instance.liquidity * leverage) / price
                quantity = round_to_tick_size(quantity, getPrecision())
                collateral_change = (quantity * price) / leverage
        else:
            # Clamp quantity if reducing position
            quantity = min(quantity, self.size)
            collateral_change = (-quantity * price) / leverage

        if quantity < EPSILON:
            return None

        # Calculate PNL and fees
        fee = self.calculate_fee_taker(price, quantity)
        pnl_q, pnl_pct = self.calculate_pnl(price, quantity) if not is_increasing else (0.0, 0.0)

        # Update position state
        if is_increasing:
            new_size = self.size + quantity
            self.priceAvg = ((self.priceAvg * self.size) + (price * quantity)) / new_size if new_size > EPSILON else price
            self.size = new_size
            self.collateral += collateral_change
            self.max_size_held = max(self.max_size_held, self.size)
            self.leverage = leverage if not self.active else self.leverage # FIXME: Allow to combine orders with different leverages
        else:
            self.size -= quantity
            self.realized_pnl_quantity += pnl_q - fee
            self.realized_pnl_percentage += pnl_pct
            self.collateral += collateral_change
            if self.size < EPSILON:
                self.size = 0.0

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
            'pnl': pnl_q,
            'pnl_percentage': pnl_pct,
        }
        self.order_history.append(order_info)

        # Update strategy liquidity
        if is_increasing:
            self.strategy_instance.liquidity -= collateral_change + fee
        else:
            self.strategy_instance.liquidity += (pnl_q - fee)

        # Broker event
        if not isInitializing():
            quantity_dollars = quantity * price
            position_size_base = self.size
            position_size_dollars = self.size * price
            position_collateral_dollars = self.collateral
            active.timeframe.stream.broker_event(
                order_type=order_type,
                quantity=quantity,
                quantity_dollars=quantity_dollars,
                position_type=self.type,
                position_size_base=position_size_base,
                position_size_dollars=position_size_dollars,
                position_collateral_dollars=position_collateral_dollars,
                leverage=leverage
            )
        
        if self.size < EPSILON: # The order has emptied the position
            self.was_liquidated = liquidation
            self.size = 0.0
            self.collateral = 0.0
            self.realized_pnl_quantity = self.calculate_realized_pnl_from_history() - self.calculate_fees_from_history()
            self.strategy_instance._close_position(self)
            return order_info
        
        self.active = True
        self.liquidation_price = self.calculate_liquidation_price()
        return order_info

    def close(self):
        if not self.active or self.size < EPSILON:
            return
        order_type = c.BUY if self.type == c.SHORT else c.SELL
        price = getRealtimeCandle().close
        self.execute_order(order_type, price, self.size, self.leverage)

    def get_unrealized_pnl(self) -> float:
        if not self.active or self.size < EPSILON:
            return 0.0
        current_price = round_to_tick_size(getRealtimeCandle().close, getMintick())
        unrealized_pnl = 0.0
        if self.type == c.LONG:
            unrealized_pnl = (current_price - self.priceAvg) * self.size * self.leverage
        elif self.type == c.SHORT:
            unrealized_pnl = (self.priceAvg - current_price) * self.size * self.leverage
        return unrealized_pnl

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

strategy = strategy_c(currency_mode='USD')

def getActivePosition(pos_type: int = None) -> 'position_c':
    return strategy.get_active_position(pos_type)

def newTick(candle: candle_c, realtime: bool = True):
    strategy.check_liquidation(candle, realtime)

def marker( pos:position_c, message = None, reversal:bool = False ):
    if strategy.show_markers and pos:
        order = pos.order_history[-1]
        if order['quantity'] < EPSILON:
            return
        newposition = len(pos.order_history) == 1
        closedposition = pos.active == False
        order_type = int(order['type'])
        order_cost = order['collateral_change']

        shape = 'arrow_up' if order_type == c.BUY else 'arrow_down'
        if newposition:
            shape == 'circle'
        elif closedposition == True:
            shape = 'square'

        if not message:
            if closedposition:
                pnl = pos.calculate_realized_pnl_from_history() - pos.calculate_fees_from_history()
                message = f"pnl:{pnl:.2f}"
            else:
                order_name = 'buy' if order_type == c.BUY else 'sell'
                message = f"{order_name}:${abs(order_cost):.2f} (pos:{pos.size:.2f})"

        location = 'below' if order_type == c.BUY else 'above'
        if pos.was_liquidated:
            location = 'below' if order_type == c.SHORT else 'above'
        
        createMarker( message,
                    location,
                    shape,
                    COLOR_BULL if pos.type == c.LONG else COLOR_BEAR
                    )

def order(cmd: str, target_position_type: int = None, quantity: float = None, leverage: float = None):
    
    order_type = c.BUY if cmd.lower() == 'buy' else c.SELL if cmd.lower() == 'sell' else None
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

    selected_leverage = leverage if leverage is not None else (strategy.leverage_long if target_position_type == c.LONG else strategy.leverage_short)
    current_price = getRealtimeCandle().close

    # Convert quantity to base units. Not leveraged. Not rounded to tick size
    actual_quantity_base_units = quantity if quantity is not None else strategy.order_size
    if strategy.currency_mode == 'USD':
        actual_quantity_base_units = actual_quantity_base_units / current_price if current_price > EPSILON else 0.0

    # clamp only when not using a custom quantity
    if not quantity:
        active_pos = strategy.get_active_position()
        if active_pos:
            if strategy.currency_mode == 'BASE':
                if active_pos.type == order_type and active_pos.size + actual_quantity_base_units > strategy.max_position_size:
                    actual_quantity_base_units = strategy.max_position_size - active_pos.size
            else:
                q_dollars = actual_quantity_base_units * current_price
                if active_pos.type == order_type and active_pos.collateral + q_dollars > strategy.max_position_size:
                    actual_quantity_base_units = (strategy.max_position_size - active_pos.collateral) / current_price

    if actual_quantity_base_units < EPSILON:
        if not isInitializing():
            print(f"Order quantity too small: {actual_quantity_base_units}")
        return

    if strategy.hedged: # 'HEDGE'
        if strategy.order(order_type, target_position_type, actual_quantity_base_units, selected_leverage):
            marker( strategy.get_active_position(target_position_type) )
    else:  # ONEWAY
        active_pos = strategy.get_active_position()
        if not active_pos or active_pos.type == target_position_type:
            if strategy.order(order_type, target_position_type, actual_quantity_base_units, selected_leverage):
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
                if remaining_quantity > EPSILON:
                    if strategy.order(order_type, target_position_type, remaining_quantity, selected_leverage):
                        marker( strategy.get_active_position(), reversal= True )
            else:
                if strategy.order(order_type, active_pos.type, actual_quantity_base_units, selected_leverage):
                    marker( strategy.get_active_position() )



def close(pos_type: int = None):
    mode = 'HEDGE' if strategy.hedged else 'ONEWAY'
    if mode == 'HEDGE' and not pos_type:
        raise ValueError("A position type is required in Hedge mode")
    if mode == 'ONEWAY':
        pos = strategy.get_active_position()
        if pos:
            pos.close()
            marker(pos)
    elif mode == 'HEDGE':
        if pos_type not in (c.LONG, c.SHORT):
            raise ValueError("Invalid position type")
        pos = strategy.get_active_position(pos_type)
        if pos:
            pos.close()
            marker(pos)


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
    print("\n--- Strategy Summary Stats ---")
    
    # Calculate metrics
    total_closed_positions = strategy.stats.total_winning_positions + strategy.stats.total_losing_positions
    
    pnl_quantity = strategy.stats.total_profit_loss # Already in quote currency
    
    # PnL percentage compared to initial_liquidity (initial_liquidity is always in USD)
    pnl_percentage_vs_liquidity = (pnl_quantity / strategy.initial_liquidity) * 100 if strategy.initial_liquidity != 0 else 0.0
    
    # This calculation's meaning (PnL % vs Max Pos) depends on currency_mode for max_position_size
    # It's kept as is to match previous output structure.
    # It represents PnL as a percentage of the maximum capital allowed per position.
    pnl_percentage_vs_max_pos_size = (pnl_quantity / strategy.max_position_size) * 100 if strategy.max_position_size != 0 else 0.0

    profitable_trades = strategy.stats.total_winning_positions
    losing_trades = strategy.stats.total_losing_positions
    
    percentage_profitable_trades = (profitable_trades / total_closed_positions) * 100 if total_closed_positions > 0 else 0.0

    # Calculate long and short win ratios
    long_win_ratio = (strategy.stats.total_winning_long_positions / strategy.stats.total_long_positions) * 100 if strategy.stats.total_long_positions > 0 else 0.0
    short_win_ratio = (strategy.stats.total_winning_short_positions / strategy.stats.total_short_positions) * 100 if strategy.stats.total_short_positions > 0 else 0.0


    # Print header
    print(f"{'PnL %':<12} {'Total PnL':<12} {'Trades':<8} {'Wins':<8} {'Losses':<8} {'Win Rate %':<12} {'Long Win %':<12} {'Short Win %':<12} {'Liquidated':<12}")
    # Print values
    print(f"{pnl_percentage_vs_max_pos_size:<12.2f} {pnl_quantity:<12.2f} {total_closed_positions:<8} {profitable_trades:<8} {losing_trades:<8} {percentage_profitable_trades:<12.2f} {long_win_ratio:<12.2f} {short_win_ratio:<12.2f} {strategy.stats.total_liquidated_positions:<12}")
    print("------------------------------")
    if strategy.liquidity > 1:
        print(f"Final Account Liquidity: {strategy.liquidity:.2f} USD ({pnl_percentage_vs_liquidity:.2f}% PnL)"     )
    else:
        print("Your account has been terminated.")


def print_pnl_by_period_summary():
    """
    Print realized PnL by month and year using stats.pnl_history. Does not include unrealized PnL.
    """
    print("\n--- PnL By Period (Realized Only) ---")
    pnl_history = strategy.pnl_history
    if not pnl_history:
        print("No closed positions to report PnL by period.")
        return

    # Print by year, quarters and total in one line
    # Print header
    print(f"{'Year':<8} {f'PnL {strategy.currency_mode}':>12} {'Q1':>12} {'Q2':>12} {'Q3':>12} {'Q4':>12} ")
    for year in sorted(pnl_history.keys()):
        months = pnl_history[year]
        q1 = sum(months[0:3])
        q2 = sum(months[3:6])
        q3 = sum(months[6:9])
        q4 = sum(months[9:12])
        total = sum(months)
        print(f"{year:<8} {total:12.2f} {q1:12.2f} {q2:12.2f} {q3:12.2f} {q4:12.2f}")
    print("-----------------------------")