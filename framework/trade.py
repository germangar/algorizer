# THIS FILE HAS BEEN WRITTEN BY GEMINI AI 
# I find this annoying to write so I just asked the AI to do it. 

from datetime import datetime, timezone

from .candle import candle_c
from .algorizer import getRealtimeCandle, createMarker, isInitializing, getCandle, getMintick, getPrecision
from .constants import c
from . import active # Corrected: Import active to get active.barindex

# Define a small epsilon for floating point comparisons to determine if a size is effectively zero
EPSILON = 1e-9

def round_to_tick_size(value, tick_size):
    """Rounds a value to the nearest tick_size."""
    if tick_size == 0: # Avoid division by zero, though getMintick/getPrecision should ideally return non-zero
        return value
    return round(value / tick_size) * tick_size

class strategy_c:
    """
    Trading strategy: manages positions, stats, and capital.
    """
    def __init__(self, initial_liquidity: float = 10000.0, verbose: bool = False, order_size: float = 100.0, max_position_size: float = 100.0, currency_mode: str = 'USD', leverage_long: float = 1.0, leverage_short: float = 1.0):
        # List of all positions (active/closed, long/short)
        self.positions = []
        self.total_profit_loss = 0.0 # Cumulative PnL
        self.initial_liquidity = initial_liquidity
        self.total_winning_positions = 0
        self.total_losing_positions = 0
        # Long/short stats
        self.total_winning_long_positions = 0
        self.total_losing_long_positions = 0
        self.total_long_positions = 0
        self.total_winning_short_positions = 0
        self.total_losing_short_positions = 0
        self.total_short_positions = 0
        self.verbose = verbose
        self.hedged = False
        self.order_size = order_size
        self.max_position_size = max_position_size
        self.currency_mode = currency_mode.upper()
        self.first_order_timestamp = None
        self.leverage_long = leverage_long
        self.leverage_short = leverage_short
        # Liquidation stats
        self.total_liquidated_positions = 0
        self.total_liquidated_long_positions = 0
        self.total_liquidated_short_positions = 0

        # Validate currency_mode
        if self.currency_mode not in ['USD', 'BASE']:
            raise ValueError(f"Invalid currency_mode: {currency_mode}. Must be 'USD' or 'BASE'.")

        # Validate parameters based on currency_mode
        if self.currency_mode == 'USD' and self.max_position_size > self.initial_liquidity:
            # max_position_size is a per-position cap
            if self.max_position_size > self.initial_liquidity:
                raise ValueError(f"max_position_size ({self.max_position_size}) cannot be greater than initial_liquidity ({self.initial_liquidity}) when currency_mode is 'USD', as it represents a capital limit per position.")

        if self.order_size <= 0:
            raise ValueError("order_size must be a positive value.")
        if self.max_position_size <= 0:
            raise ValueError("max_position_size must be a positive value.")
        if self.leverage_long <= 0 or self.leverage_short <= 0:
            raise ValueError("Leverage values must be positive.")


    def open_position(self, pos_type: int, price: float, quantity: float, leverage: int) -> 'position_c':
        """
        Open a new position (base units). Returns position object.
        """
        pos = position_c(self) # Pass self (strategy_c instance) to position_c
        pos.leverage = leverage # Set the *position's* leverage based on the opening order
        pos.type = pos_type # Set the initial type of the position
        
        # Record the timestamp of the first order of the entire strategy
        if self.first_order_timestamp is None:
            self.first_order_timestamp = getCandle(active.barindex).timestamp

        # Add the initial order to history, including its leverage
        pos.order_history.append({
            'type': pos_type,
            'price': price, # Price in quote currency
            'quantity': quantity, # Quantity in base units
            'barindex': active.barindex,
            'pnl_quantity': 0.0,
            'pnl_percentage': 0.0,
            'leverage': leverage # Store leverage for this specific order
        })
        
        # Recalculate metrics based on this first order
        pos._recalculate_current_position_state()
        pos.active = True # Explicitly set to active as it's just opened
        pos.max_size_held = pos.size # Set max_size_held to initial size

        self.positions.append(pos) # Add the new position to the strategy's list
        # Use marker for open
        if pos_type == c.LONG:
            createMarker('🟢', location='below', shape='arrow_up', color='#00FF00')
        else:
            createMarker('🔴', location='above', shape='arrow_down', color='#FF0000')
        return pos

    def get_active_position(self, pos_type: int = None) -> 'position_c':
        """
        Get active position of given type (or any if None).
        """
        if not self.positions:
            return None
        # Iterate from the end to find the most recent active position
        for pos in reversed(self.positions):
            if pos.active:
                if pos_type is None: # If no specific type requested, return the first active
                    return pos
                elif pos.type == pos_type: # Return the active position of the specified type
                    return pos
        return None # No active position of the specified type found

    def get_direction(self) -> int:
        """
        Return direction of first active position (0 if none).
        """
        pos = self.get_active_position() # This will return either a c.LONG or c.SHORT active position if one exists
        if pos is None:
            return 0
        return pos.type

    def execute_order(self, cmd: str, target_position_type: int, quantity_in_base_units_input: float, leverage: int = 1):
        """
        Execute order (buy/sell) in base units. Handles position logic.
        """
        if cmd is None:
            return
        
        current_price = round_to_tick_size(getRealtimeCandle().close, getMintick()) # Round price using mintick
        cmd = cmd.lower()
        
        # Round incoming quantity to precision before any further processing
        actual_quantity_to_process_base_units = round_to_tick_size(quantity_in_base_units_input, getPrecision())

        # Quantity after initial filter for zero (before any clamping due to max_pos_size)
        if actual_quantity_to_process_base_units < EPSILON:
            if self.verbose or not isInitializing():
                print(f"Warning: Order quantity ({actual_quantity_to_process_base_units:.6f} base units) is effectively zero after precision rounding. No order placed.")
            return

        # Determine the direction of the order itself (BUY or SELL)
        order_direction = c.LONG if cmd == 'buy' else c.SHORT if cmd == 'sell' else 0
        if order_direction == 0:
            print(f"Error: Invalid command '{cmd}'. Must be 'buy' or 'sell'.")
            return

        # Initialize affected_pos to None; it will be set if an operation occurs
        affected_pos = None

        # Handle USD capital clamping if currency_mode is 'USD'
        if self.currency_mode == 'USD':
            current_active_pos = self.get_active_position() # Get the single active position in one-way mode

            cost_per_base_unit = current_price
            if cost_per_base_unit < EPSILON:
                cost_per_base_unit = EPSILON # Set to a small non-zero value

            # Clamp incoming order quantity based on max_position_size (USD collateral limit)
            available_liquidity = self.initial_liquidity
            max_cap = min(self.max_position_size, available_liquidity)
            if current_active_pos is None:
                # No active position, so max allowed collateral is min(max_position_size, initial_liquidity)
                max_allowed_collateral = max_cap
                clamped_quantity = min(actual_quantity_to_process_base_units, max_allowed_collateral / cost_per_base_unit)
                actual_quantity_to_process_base_units = round_to_tick_size(clamped_quantity, getPrecision())
            elif order_direction == current_active_pos.type:
                # Increasing existing position: only allow up to remaining collateral, but not more than available liquidity
                current_collateral = current_active_pos.priceAvg * current_active_pos.size
                remaining_collateral = max_cap - current_collateral
                if remaining_collateral > EPSILON:
                    clamped_quantity = min(actual_quantity_to_process_base_units, remaining_collateral / cost_per_base_unit)
                    actual_quantity_to_process_base_units = round_to_tick_size(clamped_quantity, getPrecision())
                else:
                    actual_quantity_to_process_base_units = 0.0
            # Note: For opposing orders (partial close/reversal), no collateral clamping is applied here.
            # They are meant to reduce/close the position, not increase capital exposure.
            # `actual_quantity_to_process_base_units` remains `quantity_in_base_units_input` in this case for opposing orders.


        # --- General logic that applies to both modes after `actual_quantity_to_process_base_units` is determined ---
        active_target_pos = self.get_active_position(target_position_type)
        if active_target_pos is None:
            if order_direction == target_position_type:
                clamped_quantity_final = actual_quantity_to_process_base_units
                if self.currency_mode == 'BASE':
                    clamped_quantity_final = round_to_tick_size(min(actual_quantity_to_process_base_units, self.max_position_size), getPrecision())
                if clamped_quantity_final > EPSILON:
                    affected_pos = self.open_position(target_position_type, current_price, clamped_quantity_final, leverage)
                elif self.verbose or not isInitializing():
                    print(f"Warning: Attempted to open new position, but calculated final quantity ({clamped_quantity_final:.2f} base units) is effectively zero. No position opened.")
            else:
                if self.verbose or not isInitializing():
                    print(f"Warning: In hedged mode, cannot open a {target_position_type} position with a {cmd} order without an active position. Please ensure order direction matches target position type for opening.")
        else:
            if order_direction == active_target_pos.type:
                clamped_quantity_final = actual_quantity_to_process_base_units
                if self.currency_mode == 'BASE':
                    available_space_base_units = self.max_position_size - active_target_pos.size
                    clamped_quantity_final = round_to_tick_size(min(actual_quantity_to_process_base_units, available_space_base_units), getPrecision())
                if clamped_quantity_final > EPSILON:
                    active_target_pos.update(order_direction, current_price, clamped_quantity_final, leverage)
                    affected_pos = active_target_pos
                elif self.verbose or not isInitializing():
                    print(f"Warning: Attempted to increase {active_target_pos.type} position but calculated final quantity ({clamped_quantity_final:.2f} base units) is effectively zero. No change to position.")
            else:
                if actual_quantity_to_process_base_units < active_target_pos.size - EPSILON:
                    active_target_pos.update(order_direction, current_price, actual_quantity_to_process_base_units, leverage)
                    affected_pos = active_target_pos
                elif actual_quantity_to_process_base_units >= active_target_pos.size - EPSILON:
                    closing_position_reference = active_target_pos
                    affected_pos = closing_position_reference
                    pos_size_to_close = closing_position_reference.size
                    closing_position_reference.close(current_price)
                    marker_text = 'W' if closing_position_reference.profit > EPSILON else ('L' if closing_position_reference.profit < -EPSILON else 'E')
                    marker_color = '#00CC00' if closing_position_reference.type == c.LONG else '#FF0000'
                    createMarker(marker_text, location='above', shape='square', color=marker_color)
                    if actual_quantity_to_process_base_units > pos_size_to_close + EPSILON:
                        if self.verbose or not isInitializing():
                            print(f"Warning: Attempted to close a {'LONG' if closing_position_reference.type == c.LONG else 'SHORT'} position with an oversized {cmd} order.")
                            print(f"Position was fully closed. Remaining quantity ({actual_quantity_to_process_base_units - pos_size_to_close:.2f} base units) was not used to open a new position.")


        # NEW: Call broker_event after order/position update
        if not isInitializing() and affected_pos is not None:
            final_position_type_for_broker_event = 0
            final_position_size_base_for_broker_event = 0.0
            final_position_size_dollars_for_broker_event = 0.0
            final_position_collateral_dollars_for_broker_event = 0.0 # NEW

            if affected_pos.active and affected_pos.size > EPSILON:
                final_position_type_for_broker_event = affected_pos.type
                final_position_size_base_for_broker_event = affected_pos.size * affected_pos.type # Signed
                final_position_size_dollars_for_broker_event = affected_pos.collateral * affected_pos.type # Signed (notional value including leverage)
                # Position collateral is priceAvg * size (no leverage applied here)
                final_position_collateral_dollars_for_broker_event = (affected_pos.priceAvg * affected_pos.size) * affected_pos.type # Signed
            # If position is inactive and size is effectively zero, it means it's flat, so leave values as 0.0

            # The order_type for broker event should be the direction of the order itself (BUY/SELL)
            order_type_for_broker_event = c.LONG if cmd == 'buy' else c.SHORT

            # The quantity_dollars for broker event refers to the *order's* notional value
            order_quantity_dollars = actual_quantity_to_process_base_units * current_price

            active.timeframe.stream.broker_event(
                order_type=order_type_for_broker_event,
                quantity=actual_quantity_to_process_base_units,
                quantity_dollars=order_quantity_dollars,
                position_type=final_position_type_for_broker_event, # The type of the *resulting* position
                position_size_base=final_position_size_base_for_broker_event,
                position_size_dollars=final_position_size_dollars_for_broker_event,
                position_collateral_dollars=final_position_collateral_dollars_for_broker_event,
                leverage=leverage
            )


    def close_position(self, pos_type: int = None):
        """
        Close active position of given type (or any if None).
        """
        if pos_type is None:
            pos_to_close = self.get_active_position()
        else:
            pos_to_close = self.get_active_position(pos_type)
        if pos_to_close is None or not pos_to_close.active or pos_to_close.size < EPSILON:
            if self.verbose or not isInitializing():
                type_str = f" { 'LONG' if pos_type == c.LONG else 'SHORT' }" if pos_type is not None else ""
                print(f"No active{type_str} position to close.")
            return
        close_cmd = 'sell' if pos_to_close.type == c.LONG else 'buy'
        self.execute_order(close_cmd, pos_to_close.type, pos_to_close.size, pos_to_close.leverage)

    def check_liquidation(self, candle:candle_c, realtime: bool = True):
        """
        Check and close liquidated positions on price update.
        """
        for pos in self.positions:
            if pos.active:
                pos.check_liquidation_and_close( candle.close, realtime )

    def get_total_profit_loss(self) -> float:
        """
        Return total realized PnL.
        """
        return self.total_profit_loss

    def get_average_winning_trade_pnl(self) -> float:
        """
        Average PnL of closed winning trades (0 if none).
        """
        total_winning_pnl = 0.0
        winning_trade_count = 0
        for pos in self.positions:
            if not pos.active and pos.profit > EPSILON: # Only consider closed and profitable positions
                total_winning_pnl += pos.profit
                winning_trade_count += 1
        
        return total_winning_pnl / winning_trade_count if winning_trade_count > 0 else 0.0

    def get_average_losing_trade_pnl(self) -> float:
        """
        Average PnL of closed losing trades (abs, 0 if none).
        """
        total_losing_pnl = 0.0 # Will sum up negative profits
        losing_trade_count = 0
        for pos in self.positions:
            if not pos.active and pos.profit < -EPSILON: # Only consider closed and losing positions
                total_losing_pnl += pos.profit # This will be a negative number
                losing_trade_count += 1
        
        # Return the absolute value to represent average loss as a positive number
        return abs(total_losing_pnl / losing_trade_count) if losing_trade_count > 0 else 0.0

    def print_detailed_stats(self):
        """
        Print all positions, order history, and summary.
        """
        print("\n--- Strategy Positions and Order History ---")
        if not self.positions:
            print("No positions have been opened yet.")
            return
        
        active_long_count = 0
        active_short_count = 0
        closed_long_count = 0
        closed_short_count = 0

        for i, pos in enumerate(self.positions):
            status = "Active" if pos.active else "Closed"
            position_type_str = "LONG" if pos.type == c.LONG else ("SHORT" if pos.type == c.SHORT else "N/A (Type not set)")
            
            if pos.active:
                if pos.type == c.LONG:
                    active_long_count += 1
                elif pos.type == c.SHORT:
                    active_short_count += 1
            else:
                if pos.type == c.LONG:
                    closed_long_count += 1
                elif pos.type == c.SHORT:
                    closed_short_count += 1

            # Get unrealized PnL only if active
            unrealized_pnl_qty = pos.get_unrealized_pnl_quantity() if pos.active else 0.0
            unrealized_pnl_pct = pos.get_unrealized_pnl_percentage() if pos.active else 0.0
            
            # Added collateral to print
            print(f"\nPosition #{i+1} (Status: {status}, Type: {position_type_str}, Current Size: {pos.size:.2f} base units, Avg Price: {pos.priceAvg:.2f}, Max Size Held: {pos.max_size_held:.2f} base units, Collateral: {pos.collateral:.2f} USD, Position Leverage: {pos.leverage})") 
            if pos.active:
                print(f"   Unrealized PnL: {unrealized_pnl_qty:.2f} (quote currency) ({unrealized_pnl_pct:.2f}%)")
            else:
                print(f"   Realized PnL: {pos.profit:.2f} (quote currency) ({pos.realized_pnl_percentage:.2f}%)")

            print("   Order History:")

            if not pos.order_history:
                print("     No orders in this position's history.")
            else:
                for j, order_data in enumerate(pos.order_history):
                    order_type_str = "BUY" if order_data['type'] == c.LONG else "SELL"
                    pnl_info = f" | Realized PnL: {order_data['pnl_quantity']:.2f} (quote currency) ({order_data['pnl_percentage']:.2f}%)" if abs(order_data['pnl_quantity']) > EPSILON or abs(order_data['pnl_percentage']) > EPSILON else ""
                    # Increased precision for quantity display, added leverage to print
                    print(f"     Order {j+1}: {order_type_str} {order_data['quantity']:.6f} base units at {order_data['price']:.2f} (Bar Index: {order_data['barindex']}, Leverage: {order_data.get('leverage', 'N/A')}){pnl_info}") 
        
        print("\n--- Position Summary ---")
        print(f"Active LONG positions: {active_long_count}")
        print(f"Active SHORT positions: {active_short_count}")
        print(f"Closed LONG positions: {closed_long_count}")
        print(f"Closed SHORT positions: {closed_short_count}")
        print(f"Total Strategy PnL: {self.total_profit_loss:.2f} (quote currency)")
        print("------------------------------------------")

    def print_summary_stats(self):
        """
        Print summary of strategy performance.
        """
        print("\n--- Strategy Summary Stats ---")
        
        # Calculate metrics
        total_closed_positions = self.total_winning_positions + self.total_losing_positions
        
        pnl_quantity = self.total_profit_loss # Already in quote currency
        
        # PnL percentage compared to initial_liquidity (initial_liquidity is always in USD)
        pnl_percentage_vs_liquidity = (pnl_quantity / self.initial_liquidity) * 100 if self.initial_liquidity != 0 else 0.0
        
        # This calculation's meaning (PnL % vs Max Pos) depends on currency_mode for max_position_size
        # It's kept as is to match previous output structure.
        # It represents PnL as a percentage of the maximum capital allowed per position.
        pnl_percentage_vs_max_pos_size = (pnl_quantity / self.max_position_size) * 100 if self.max_position_size != 0 else 0.0

        profitable_trades = self.total_winning_positions
        losing_trades = self.total_losing_positions
        
        percentage_profitable_trades = (profitable_trades / total_closed_positions) * 100 if total_closed_positions > 0 else 0.0

        avg_winning_pnl = self.get_average_winning_trade_pnl()
        avg_losing_pnl = self.get_average_losing_trade_pnl()

        # Calculate long and short win ratios
        long_win_ratio = (self.total_winning_long_positions / self.total_long_positions) * 100 if self.total_long_positions > 0 else 0.0
        short_win_ratio = (self.total_winning_short_positions / self.total_short_positions) * 100 if self.total_short_positions > 0 else 0.0


        # Print header
        print(f"{'PnL %':<12} {'Total PnL':<12} {'Trades':<8} {'Wins':<8} {'Losses':<8} {'Win Rate %':<12} {'Long Win %':<12} {'Short Win %':<12} {'Avg+ PnL':<12} {'Avg- PnL':<12} {'Acct PnL %':<12} {'Liquidated':<12}")
        # Print values
        print(f"{pnl_percentage_vs_max_pos_size:<12.2f} {pnl_quantity:<12.2f} {total_closed_positions:<8} {profitable_trades:<8} {losing_trades:<8} {percentage_profitable_trades:<12.2f} {long_win_ratio:<12.2f} {short_win_ratio:<12.2f} {avg_winning_pnl:<12.2f} {avg_losing_pnl:<12.2f} {pnl_percentage_vs_liquidity:<12.2f} {self.total_liquidated_positions:<12}")
        print("------------------------------")
        if self.initial_liquidity > 1:
            print(f"Final Account Liquidity: {self.initial_liquidity:.2f} USD")
        else:
            print("Your account has been terminated.")

    def print_pnl_by_period(self):
        """
        Print realized/unrealized PnL by month/quarter/year.
        """
        print("\n--- PnL By Period ---")

        if not self.positions or self.first_order_timestamp is None:
            print("No orders were processed during the strategy run to determine a period.")
            return

        # Get the timestamp of the last candle processed in the backtest
        last_candle_timestamp = getRealtimeCandle().timestamp # This is the current bar's timestamp at the end of the backtest

        # Aggregate realized PnL from closed positions
        pnl_by_month_realized = {} # Key: McClellan-MM, Value: PnL (quote currency)
        pnl_by_quarter_realized = {} # Key: McClellan-Qn, Value: PnL
        pnl_by_year_realized = {} # Key: McClellan, Value: PnL

        for pos in self.positions:
            if not pos.active and pos.close_timestamp is not None:
                # Convert milliseconds timestamp to datetime object (timezone-aware recommended for clarity)
                dt_obj = datetime.fromtimestamp(pos.close_timestamp / 1000, tz=timezone.utc)
                
                year = dt_obj.year
                month = dt_obj.month
                quarter = (dt_obj.month - 1) // 3 + 1 # Quarter (1, 2, 3, or 4)

                month_key = f"{year}-{month:02d}"
                quarter_key = f"{year}-Q{quarter}"
                year_key = f"{year}"

                pnl_by_month_realized.setdefault(month_key, 0.0)
                pnl_by_month_realized[month_key] += pos.profit

                pnl_by_quarter_realized.setdefault(quarter_key, 0.0)
                pnl_by_quarter_realized[quarter_key] += pos.profit

                pnl_by_year_realized.setdefault(year_key, 0.0)
                pnl_by_year_realized[year_key] += pos.profit
        
        # Add unrealized PnL from active positions to the final period
        final_dt_obj = datetime.fromtimestamp(last_candle_timestamp / 1000, tz=timezone.utc)
        final_month_key = f"{final_dt_obj.year}-{final_dt_obj.month:02d}"
        final_quarter_key = f"{final_dt_obj.year}-Q{(final_dt_obj.month - 1) // 3 + 1}"
        final_year_key = f"{final_dt_obj.year}"

        unrealized_pnl_total_active = 0.0
        for pos in self.positions:
            if pos.active:
                urpnl = pos.get_unrealized_pnl_quantity()
                unrealized_pnl_total_active += urpnl

                # Add unrealized PnL to the aggregated dicts for the final period
                pnl_by_month_realized.setdefault(final_month_key, 0.0)
                pnl_by_month_realized[final_month_key] += urpnl

                pnl_by_quarter_realized.setdefault(final_quarter_key, 0.0)
                pnl_by_quarter_realized[final_quarter_key] += urpnl

                pnl_by_year_realized.setdefault(final_year_key, 0.0)
                pnl_by_year_realized[final_year_key] += urpnl
        
        if abs(unrealized_pnl_total_active) > EPSILON:
             print(f"Note: Unrealized PnL ({unrealized_pnl_total_active:.2f} {self.currency_mode}) from active positions included in the final period's PnL.")

        # Determine the total duration of the strategy's activity for granularity
        start_dt = datetime.fromtimestamp(self.first_order_timestamp / 1000, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(last_candle_timestamp / 1000, tz=timezone.utc)

        # Calculate total months spanning the strategy's activity
        total_months_span = (end_dt.year - start_dt.year) * 12 + end_dt.month - start_dt.month + 1

        # Helper function to print a PnL dictionary for a given period type
        def _print_pnl_dict(title: str, pnl_dict: dict):
            print(f"\n--- {title} PnL ---")
            sorted_keys = sorted(pnl_dict.keys())
            if not sorted_keys:
                print(f"No data for {title.lower()} period.")
                return
            for key in sorted_keys:
                print(f"  {key}: {pnl_dict[key]:.2f} {self.currency_mode}") # Assuming quote currency is USD, using self.currency_mode for generality

        # Display based on calculated duration
        if total_months_span <= 3:
            _print_pnl_dict("Monthly", pnl_by_month_realized)
        elif total_months_span <= 24: # Up to 2 years (24 months)
            _print_pnl_dict("Quarterly", pnl_by_quarter_realized)
        else: # More than 2 years, show yearly and then quarterly
            _print_pnl_dict("Yearly", pnl_by_year_realized)
            _print_pnl_dict("Quarterly", pnl_by_quarter_realized) # Also show quarters for longer runs

        print("-----------------------------")


class position_c:
    """
    Single trading position (long/short). Tracks orders, PnL, and state.
    """
    def __init__(self, strategy_instance: strategy_c): # Accept strategy instance during initialization
        self.strategy_instance = strategy_instance # Store reference to the parent strategy
        self.active = False        # Is the position currently open?
        # self.type will be 1 for c.LONG or -1 for c.SHORT, even when closed.
        # It's initialized to 0, but will be set to c.LONG/c.SHORT on the first order.
        self.type = 0            
        self.size = 0.0          # Current size of the position (absolute value, always in base units)
        self.priceAvg = 0.0      # Average entry price of the position (in quote currency)
        self.leverage = 1        # Leverage applied to the position (set by opening order, assumed constant for position life)
        self.profit = 0.0        # Total realized PnL for this position when it closes (final value, in quote currency)
        self.realized_pnl_quantity = 0.0 # Cumulative realized PnL in quantity (in quote currency)
        self.realized_pnl_percentage = 0.0 # Cumulative realized PnL in percentage for this position (final value)
        self.order_history = []  # Stores {'type': c.LONG/c.SHORT, 'price': float, 'quantity': float, 'barindex': int, 'pnl_quantity': float, 'pnl_percentage': float, 'leverage': int}
        self.max_size_held = 0.0 # Variable to track maximum size held during the position's lifetime (in base units)
        self.collateral = 0.0 # Tracks the USD collateral (cost basis) currently tied up in the open position
        self.close_timestamp = None # NEW: Store timestamp when position is closed
        self.liquidation_price = 0.0
        self.was_liquidated = False

    def _recalculate_current_position_state(self):
        """
        Update size, avg price, collateral from order history.
        """
        net_long_quantity = 0.0
        net_long_value = 0.0 # Value in quote currency
        net_short_quantity = 0.0
        net_short_value = 0.0 # Value in quote currency

        for order_data in self.order_history:
            if order_data['type'] == c.LONG:
                net_long_quantity += order_data['quantity']
                net_long_value += order_data['price'] * order_data['quantity']
            elif order_data['type'] == c.SHORT:
                net_short_quantity += order_data['quantity']
                net_short_value += order_data['price'] * order_data['quantity']

        # Determine the net position in base units
        net_quantity = net_long_quantity - net_short_quantity
        net_value = net_long_value - net_short_value # Sum of value in quote currency

        if net_quantity > EPSILON: # Use EPSILON for comparison
            self.type = c.LONG
            self.size = net_quantity
            self.priceAvg = net_value / net_quantity
        elif net_quantity < -EPSILON: # Use EPSILON for comparison
            self.type = c.SHORT
            self.size = abs(net_quantity) # Size is always positive (in base units)
            self.priceAvg = abs(net_value / net_quantity) # Average price for short (in quote currency)
        else: # net_quantity is effectively zero, position is flat
            self.size = 0.0
            self.priceAvg = 0.0
            # self.type retains its last non-zero value, or remains 0 if no orders yet.
            # self.active will be set to False by the close method.

        # Calculate collateral based on the current (recalculated) state
        # This uses the position's overall leverage (self.leverage)
        if abs(self.size) > EPSILON: # Only if there's an active position size
            self.collateral = self.priceAvg * self.size
        else:
            self.collateral = 0.0
        self._update_liquidation_price()

    def _update_liquidation_price(self):
        if self.leverage == 1 or self.size == 0:
            self.liquidation_price = 0.0
            return
        if self.type == c.LONG:
            self.liquidation_price = self.priceAvg * (1 - 1.0 / self.leverage)
        elif self.type == c.SHORT:
            self.liquidation_price = self.priceAvg * (1 + 1.0 / self.leverage)
        else:
            self.liquidation_price = 0.0

    def get_average_entry_price_from_history(self) -> float:
        """
        Return avg entry price from order history.
        """
        net_long_quantity = 0.0
        net_long_value = 0.0
        net_short_quantity = 0.0
        net_short_value = 0.0

        for order_data in self.order_history:
            if order_data['type'] == c.LONG:
                net_long_quantity += order_data['quantity']
                net_long_value += order_data['price'] * order_data['quantity']
            elif order_data['type'] == c.SHORT:
                net_short_quantity += order_data['quantity']
                net_short_value += order_data['price'] * order_data['quantity']

        net_quantity = net_long_quantity - net_short_quantity
        net_value = net_long_value - net_short_value

        if abs(net_quantity) > EPSILON: # Use EPSILON for comparison
            return abs(net_value / net_quantity)
        return 0.0

    def update(self, op_type: int, price: float, quantity: float, leverage: int):
        """
        Add order to history, update state, handle markers.
        """
        # Store the state before the update for marker logic
        previous_active = self.active
        previous_type = self.type # Store previous type
        previous_size = self.size # Size in base units

        # Initialize PnL for this order
        pnl_q = 0.0
        pnl_pct = 0.0

        # Calculate PnL if this order reduces the position size
        # PnL for partial closes uses the *position's* overall leverage (self.leverage)
        if previous_active and op_type != self.type and quantity > EPSILON:
            # This order is opposing the current position type, thus reducing it
            reduced_quantity = min(quantity, previous_size) # The quantity being reduced (in base units)
            if self.type == c.LONG:
                pnl_q = (price - self.priceAvg) * reduced_quantity * self.leverage
            elif self.type == c.SHORT:
                pnl_q = (self.priceAvg - price) * reduced_quantity * self.leverage
            
            # Capital involved is in quote currency
            capital_involved = self.priceAvg * reduced_quantity
            if capital_involved > EPSILON:
                pnl_pct = (pnl_q / capital_involved) * 100
            else:
                pnl_pct = 0.0 # Avoid division by zero if capital involved is zero
            
            # Add this order's realized PnL to the position's cumulative realized PnL
            self.realized_pnl_quantity += pnl_q

        # Record the order in history with PnL and its specific leverage
        self.order_history.append({
            'type': op_type,
            'price': price, # Price in quote currency
            'quantity': quantity, # Quantity in base units
            'barindex': active.barindex,
            'pnl_quantity': pnl_q,
            'pnl_percentage': pnl_pct,
            'leverage': leverage # Store leverage for this specific order
        })

        # The position's overall leverage (self.leverage) is set only by the opening order
        # It is *not* updated by subsequent orders within the same position.
        # This ensures consistent PnL calculation for the entire position.

        # Recalculate metrics based on the updated history
        self._recalculate_current_position_state()

        # Update max_size_held if current size is greater
        if self.size > self.max_size_held:
            self.max_size_held = self.size

        # Determine marker color based on the *action* (buy/sell)
        marker_color = '#00FF00' if op_type == c.LONG else '#FF0000' # Green for Buy (c.LONG order type), Red for Sell (c.SHORT order type)
        marker_shape = 'arrow_up' if op_type == c.LONG else 'arrow_down'

        # Handle markers based on the *net* change in position
        if previous_active and self.active: # Position is still active
            # If the type has changed, it implies a reversal that didn't fully close the prior position
            if self.type != previous_type and self.size > EPSILON:
                createMarker('🔄', location='inside', shape='circle', color='#FFD700') # Gold circle for partial reversal
            elif self.size > previous_size + EPSILON: # Increasing position
                collateral = price * quantity
                marker_text = f"➕ ${collateral:.2f}"
                marker_color = '#00FF00' if op_type == c.LONG else '#FF0000' # Green for Buy, Red for Sell
                marker_shape = 'arrow_up' if op_type == c.LONG else 'arrow_down'
                createMarker(marker_text, location='below', shape=marker_shape, color=marker_color)
            elif self.size < previous_size - EPSILON: # Decreasing position (partial reduction)
                collateral = price * quantity
                marker_text = f"➖ ${collateral:.2f}"
                # Special logic for reduction markers:
                if previous_type == c.SHORT and op_type == c.LONG:
                    # Reducing a SHORT with a BUY: red, arrow_up
                    marker_color = '#FF0000'
                    marker_shape = 'arrow_up'
                elif previous_type == c.LONG and op_type == c.SHORT:
                    # Reducing a LONG with a SELL: green, arrow_down
                    marker_color = '#00FF00'
                    marker_shape = 'arrow_down'
                else:
                    # Fallback to default coloring
                    marker_color = '#00FF00' if op_type == c.LONG else '#FF0000'
                    marker_shape = 'arrow_up' if op_type == c.LONG else 'arrow_down'
                createMarker(marker_text, location='above', shape=marker_shape, color=marker_color)
        # If previous_active was True and self.active is now False, it means the position was closed.
        # This specific case is handled by the `close` method.

    def check_liquidation_and_close(self, current_price: float, realtime: bool = True):
        """
        Liquidate and close if loss equals collateral.
        """
        if not self.active or self.size < EPSILON or self.leverage <= 1:
            return
        self._update_liquidation_price()
        # Calculate current leveraged PnL
        if self.type == c.LONG:
            pnl = (current_price - self.priceAvg) * self.size * self.leverage
            collateral = self.priceAvg * self.size
            if pnl <= -collateral + EPSILON:
                self.was_liquidated = True
                close_price = current_price if realtime else self.liquidation_price
                self.close(close_price, liquidation_reason="LIQUIDATION")
                self.strategy_instance.total_liquidated_positions += 1
                self.strategy_instance.total_liquidated_long_positions += 1
        elif self.type == c.SHORT:
            pnl = (self.priceAvg - current_price) * self.size * self.leverage
            collateral = self.priceAvg * self.size
            if pnl <= -collateral + EPSILON:
                self.was_liquidated = True
                close_price = current_price if realtime else self.liquidation_price
                self.close(close_price, liquidation_reason="LIQUIDATION")
                self.strategy_instance.total_liquidated_positions += 1
                self.strategy_instance.total_liquidated_short_positions += 1

    def close(self, price: float, liquidation_reason: str = None):
        """
        Close position, realize PnL, update stats, handle markers.
        """
        if not self.active:
            return

        closing_quantity = self.size # This is in base units
        closing_op_type = c.SHORT if self.type == c.LONG else c.LONG # Use the position's current type to determine closing order type

        # Store previous state for PnL calculation on the closing order
        previous_avg_price = self.priceAvg
        previous_leverage = self.leverage # Use the position's effective leverage for calculation
        previous_position_type = self.type

        # Add the closing order to history, including its leverage
        self.order_history.append({
            'type': closing_op_type,
            'price': price, # Price in quote currency
            'quantity': closing_quantity, # Quantity in base units
            'barindex': active.barindex,
            'pnl_quantity': 0.0, # Will be updated after full PnL calculation
            'pnl_percentage': 0.0, # Will be updated after full PnL calculation
            'leverage': previous_leverage # Store leverage for this specific closing order
        })

        # Recalculate metrics based on the updated history
        self._recalculate_current_position_state()

        # If the position is now effectively closed (size is 0), calculate total PnL for this position object
        if abs(self.size) < EPSILON:
            # Add verbose logging for PnL calculation
            if self.strategy_instance.verbose or not isInitializing():
                print(f"DEBUG CLOSING PNL: prev_type={previous_position_type}, prev_avg_price={previous_avg_price:.6f}, close_price={price:.6f}, closing_qty={closing_quantity:.6f}, leverage={previous_leverage}")
                if previous_position_type == c.LONG:
                    print(f"DEBUG LONG PNL CALC: ({price:.6f} - {previous_avg_price:.6f}) * {closing_quantity:.6f} * {previous_leverage}")
                elif previous_position_type == c.SHORT:
                    print(f"DEBUG SHORT PNL CALC: ({previous_avg_price:.6f} - {price:.6f}) * {closing_quantity:.6f} * {previous_leverage}")

            final_close_pnl_q = 0.0
            if previous_position_type == c.LONG:
                final_close_pnl_q = (price - previous_avg_price) * closing_quantity * previous_leverage
            elif previous_position_type == c.SHORT:
                final_close_pnl_q = (previous_avg_price - price) * closing_quantity * previous_leverage
            
            # Capital involved is in quote currency
            final_close_capital_involved = previous_avg_price * closing_quantity
            final_close_pnl_pct = 0.0
            if final_close_capital_involved > EPSILON:
                final_close_pnl_pct = (final_close_pnl_q / final_close_capital_involved) * 100
            else:
                final_close_pnl_pct = 0.0
            
            # Update the last order in history with its PnL
            self.order_history[-1]['pnl_quantity'] = final_close_pnl_q
            self.order_history[-1]['pnl_percentage'] = final_close_pnl_pct

            # Add the PnL from the final closing order to the cumulative realized PnL
            self.realized_pnl_quantity += final_close_pnl_q
            
            self.profit = self.realized_pnl_quantity # Total position PnL is the cumulative realized PnL
            
            # Calculate final percentage PnL for the entire position
            total_entry_capital_for_position = 0.0
            for order_data in self.order_history:
                if order_data['pnl_quantity'] >= -EPSILON and order_data['pnl_quantity'] <= EPSILON:
                    order_leverage = order_data.get('leverage', 1.0)
                    total_entry_capital_for_position += order_data['price'] * order_data['quantity'] * order_leverage
            
            self.realized_pnl_percentage = (self.profit / total_entry_capital_for_position) * 100 if total_entry_capital_for_position != 0 else 0.0

            # Update strategy-level counters for winning/losing positions
            if self.profit > EPSILON:
                self.strategy_instance.total_winning_positions += 1
                if previous_position_type == c.LONG:
                    self.strategy_instance.total_winning_long_positions += 1
                elif previous_position_type == c.SHORT:
                    self.strategy_instance.total_winning_short_positions += 1
            elif self.profit < -EPSILON:
                self.strategy_instance.total_losing_positions += 1
                if previous_position_type == c.LONG:
                    self.strategy_instance.total_losing_long_positions += 1
                elif previous_position_type == c.SHORT:
                    self.strategy_instance.total_losing_short_positions += 1
            
            # Increment total long/short positions regardless of win/loss
            if previous_position_type == c.LONG:
                self.strategy_instance.total_long_positions += 1
            elif previous_position_type == c.SHORT:
                self.strategy_instance.total_short_positions += 1


            self.strategy_instance.total_profit_loss += self.profit # Update strategy's total PnL
            self.active = False # Explicitly set to inactive as it's fully closed
            self.collateral = 0.0 # Reset capital invested when position is closed
            self.close_timestamp = getRealtimeCandle().timestamp # NEW: Record closure timestamp

            # Update account liquidity with realized PnL (ensure float type and correct attribute)
            if hasattr(self.strategy_instance, 'initial_liquidity') and self.profit is not None:
                self.strategy_instance.initial_liquidity = float(self.strategy_instance.initial_liquidity) + float(self.profit)

            # Marker for liquidation or normal close
            if liquidation_reason is not None:
                marker_color = '#00CC00' if previous_position_type == c.LONG else '#FF0000'
                createMarker(liquidation_reason, location='above', shape='square', color=marker_color)
            # Print PnL to console
            if self.strategy_instance.verbose or not isInitializing():
                print(f"CLOSED POSITION ({'LONG' if previous_position_type == c.LONG else 'SHORT'}): PnL: {self.profit:.2f} (quote currency) | PnL %: {self.realized_pnl_percentage:.2f}% | Total Strategy PnL: {self.strategy_instance.total_profit_loss:.2f} (quote currency)")

    def get_unrealized_pnl_quantity(self) -> float:
        """
        Return unrealized PnL (quote currency) if active.
        """
        if not self.active or self.size < EPSILON:
            return 0.0

        current_price = round_to_tick_size(getRealtimeCandle().close, getMintick()) # Round current price using mintick
        unrealized_pnl = 0.0

        if self.type == c.LONG:
            unrealized_pnl = (current_price - self.priceAvg) * self.size * self.leverage
        elif self.type == c.SHORT:
            unrealized_pnl = (self.priceAvg - current_price) * self.size * self.leverage
        
        return unrealized_pnl

    def get_unrealized_pnl_percentage(self) -> float:
        """
        Return unrealized PnL as percent of collateral.
        """
        unrealized_pnl_q = self.get_unrealized_pnl_quantity()
        if abs(unrealized_pnl_q) < EPSILON and (not self.active or self.size < EPSILON):
            return 0.0

        # Capital involved in the active position based on entry price and quantity (in quote currency)
        # This is self.collateral
        capital_involved = self.collateral
        if abs(capital_involved) < EPSILON: # Avoid division by zero
            return 0.0
        
        return (unrealized_pnl_q / abs(capital_involved)) * 100

    def get_order_by_direction(self, order_direction: int, older_than_bar_index: int = None) -> dict:
        """
        Get last order of given direction (optionally before bar index).
        """
        # Iterate backward through the order history to find the most recent matching order
        for order_data in reversed(self.order_history):
            if order_data['type'] == order_direction:
                if older_than_bar_index is None:
                    # If no index is provided, return the very last order of this type
                    return order_data
                elif order_data['barindex'] < older_than_bar_index:
                    # If an index is provided, return the first order of this type that is older than the index
                    return order_data
        return None # No matching order found


class OrderManager:
    """
    Routes orders and enforces ONEWAY/HEDGE logic.
    """
    def __init__(self, strategy: strategy_c):
        self.strategy = strategy

    def mode(self):
        return "HEDGED" if self.strategy.hedged is True else "ONEWAY"

    def order(self, cmd: str, target_position_type: int, quantity: float = None, leverage: int = None):
        # Determine leverage to use for this order
        selected_leverage = leverage
        if selected_leverage is None:
            if target_position_type == c.LONG:
                selected_leverage = self.strategy.leverage_long
            elif target_position_type == c.SHORT:
                selected_leverage = self.strategy.leverage_short
            else:
                selected_leverage = 1

        # Convert quantity to base units if needed
        actual_quantity_base_units = 0.0
        current_price = getRealtimeCandle().close
        if self.strategy.currency_mode == 'USD':
            if quantity is not None:
                actual_quantity_base_units = quantity / current_price if current_price > EPSILON else 0.0
            else:
                actual_quantity_base_units = self.strategy.order_size / current_price if current_price > EPSILON else 0.0
            actual_quantity_base_units = round_to_tick_size(actual_quantity_base_units, getPrecision())
        else:  # BASE mode
            if quantity is not None:
                actual_quantity_base_units = quantity
            else:
                actual_quantity_base_units = self.strategy.order_size
            actual_quantity_base_units = round_to_tick_size(actual_quantity_base_units, getPrecision())

        if actual_quantity_base_units < EPSILON:
            if self.strategy.verbose or not isInitializing():
                print(f"Order quantity too small after conversion: {actual_quantity_base_units}")
            return

        if self.mode() == 'HEDGE':
            # Pass directly to the core
            self.strategy.execute_order(cmd, target_position_type, actual_quantity_base_units, selected_leverage)
        else:  # ONEWAY mode
            # Only one position at a time (either long or short)
            active_pos = self.strategy.get_active_position()
            order_direction = c.LONG if cmd == 'buy' else c.SHORT if cmd == 'sell' else 0
            if active_pos is None:
                # No active position, open new
                self.strategy.execute_order(cmd, target_position_type, actual_quantity_base_units, selected_leverage)
            elif order_direction == active_pos.type:
                # Increase existing position
                self.strategy.execute_order(cmd, target_position_type, actual_quantity_base_units, selected_leverage)
            else:
                # Opposing order: close/reverse logic
                if actual_quantity_base_units >= active_pos.size - EPSILON:
                    # Close existing and open new with remaining quantity
                    pos_size_to_close = active_pos.size
                    closing_position_reference = active_pos
                    closing_position_reference.close(current_price)
                    remaining_quantity = actual_quantity_base_units - pos_size_to_close
                    if remaining_quantity > EPSILON:
                        self.strategy.execute_order(cmd, target_position_type, remaining_quantity, selected_leverage)
                else:
                    # Partial close
                    self.strategy.execute_order(cmd, target_position_type, actual_quantity_base_units, selected_leverage)

    def close(self, pos_type: int = None):
        self.strategy.close_position(pos_type)

# Instantiate the core strategy always in HEDGE mode (no hedged flag)
strategy = strategy_c(currency_mode='USD')
# The order manager will enforce ONEWAY or HEDGE logic
order_manager = OrderManager(strategy)  # Default mode is 'ONEWAY'

# Update global functions to use the order manager

def openPosition(pos_type: int, price: float, quantity: float, leverage: int) -> 'position_c':
    return strategy.open_position(pos_type, price, quantity, leverage)

def getActivePosition(pos_type: int = None) -> 'position_c':
    return strategy.get_active_position(pos_type)

def direction() -> int:
    return strategy.get_direction()

def order(cmd: str, target_position_type: int, quantity: float = None, leverage: int = None):
    order_manager.order(cmd, target_position_type, quantity, leverage)

def close(pos_type: int = None):
    order_manager.close(pos_type)

def get_total_profit_loss() -> float:
    return strategy.get_total_profit_loss()

def print_strategy_stats():
    strategy.print_detailed_stats()

def print_summary_stats():
    strategy.print_summary_stats()

def print_pnl_by_period_summary():
    strategy.print_pnl_by_period()

def newTick(candle: candle_c, realtime: bool = True):
    strategy.check_liquidation(candle, realtime)




