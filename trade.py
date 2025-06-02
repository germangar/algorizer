

# THIS FILE HAS BEEN WRITTEN BY GEMINI AI 
# I find this annoying to write so I just asked the AI to do it. 
# The output doesn't seem correct, but I guess I'll work it from here.

from candle import candle_c
from algorizer import getRealtimeCandle, createMarker, isInitializing
import active # Import active to get active.barindex

# Define constants for position types
SHORT = -1
LONG = 1

# Define a small epsilon for floating point comparisons to determine if a size is effectively zero
EPSILON = 1e-9

class strategy_c:
    """
    Represents the overall trading strategy, managing positions and global statistics.
    """
    def __init__(self, initial_liquidity: float = 10000.0, verbose: bool = False, order_size: float = 100.0, max_position_size: float = 100.0, hedged: bool = False, currency_mode: str = 'USD'):
        self.positions = []   # List to hold all positions (both active and closed, Long and Short)
        self.total_profit_loss = 0.0 # Global variable to keep track of the total profit/loss for the entire strategy
        self.initial_liquidity = initial_liquidity # Starting capital for the strategy
        self.total_winning_positions = 0 # Counter for winning closed positions
        self.total_losing_positions = 0   # Counter for losing closed positions
        self.verbose = verbose # Controls whether warning prints are displayed
        self.order_size = order_size # Default quantity to use when none is provided (USD or BASE units depending on currency_mode)
        self.max_position_size = max_position_size # Maximum allowed total size for any single position (USD or BASE units)
        self.hedged = hedged # Controls whether multiple positions can be open simultaneously (True) or only one (False)
        self.currency_mode = currency_mode.upper() # NEW: 'USD' or 'BASE'

        # Validate currency_mode
        if self.currency_mode not in ['USD', 'BASE']:
            raise ValueError(f"Invalid currency_mode: {currency_mode}. Must be 'USD' or 'BASE'.")

        # Validate parameters based on currency_mode
        if self.currency_mode == 'USD' and self.max_position_size > self.initial_liquidity:
            raise ValueError(f"max_position_size ({self.max_position_size}) cannot be greater than initial_liquidity ({self.initial_liquidity}) when currency_mode is 'USD'.")
        
        if self.order_size <= 0:
            raise ValueError("order_size must be a positive value.")
        if self.max_position_size <= 0:
            raise ValueError("max_position_size must be a positive value.")


    def open_position(self, pos_type: int, price: float, quantity: float, leverage: int) -> 'position_c':
        """
        Creates and opens a new position, associated with this strategy instance.
        Quantity must always be in base units.
        """
        pos = position_c(self) # Pass self (strategy_c instance) to position_c
        pos.leverage = leverage # Set initial leverage for the position object
        pos.type = pos_type # Set the initial type of the position
        
        # Add the initial order to history with zero PnL
        pos.order_history.append({
            'type': pos_type,
            'price': price,
            'quantity': quantity,
            'barindex': active.barindex,
            'pnl_quantity': 0.0,
            'pnl_percentage': 0.0
        })
        
        # Recalculate metrics based on this first order
        pos._recalculate_current_position_state()
        pos.active = True # Explicitly set to active as it's just opened
        pos.max_size_held = pos.size # Set max_size_held to initial size

        self.positions.append(pos) # Add the new position to the strategy's list
        # Use specific marker for opening a position
        if pos_type == LONG:
            createMarker('🟢', location='below', shape='arrow_up', color='#00FF00') # Green arrow up for new LONG
        else: # SHORT
            createMarker('🔴', location='above', shape='arrow_down', color='#FF0000') # Red arrow down for new SHORT
        return pos

    def get_active_position(self, pos_type: int = None) -> 'position_c':
        """
        Retrieves the currently active position of a specific type (LONG or SHORT).
        If pos_type is None, it returns the first active position found (useful in one-way mode
        to get the single active position regardless of type).
        In hedged mode, if pos_type is None, it might return an arbitrary active position,
        so it's best to specify pos_type for clarity.
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
        Returns the direction of the first active position found.
        """
        pos = self.get_active_position() # This will return either a LONG or SHORT active position if one exists
        if pos is None:
            return 0
        return pos.type

    def execute_order(self, cmd: str, target_position_type: int, quantity_in_base_units_input: float, leverage: int = 1):
        """
        Executes a trading order. All internal calculations use base units.
        The price is retrieved from getRealtimeCandle().close internally.

        If hedged=True: Allows simultaneous LONG and SHORT positions.
        If hedged=False (One-Way Mode): Only one position (LONG or SHORT) can be active at a time.
            An oversized opposing order will close the current position and open a new one.

        Args:
            cmd (str): The command ('buy' or 'sell').
            target_position_type (int): Specifies which position (LONG or SHORT) this order targets.
            quantity_in_base_units_input (float): The quantity for the order, always in base currency units.
                                                This quantity might be clamped further inside based on strategy rules.
            leverage (int, optional): The leverage. Defaults to 1.
        """
        if cmd is None:
            return

        current_price = getRealtimeCandle().close
        cmd = cmd.lower()
        
        # Quantity after initial filter for zero (before any clamping due to max_pos_size)
        if quantity_in_base_units_input < EPSILON:
            if self.verbose and not isInitializing():
                print(f"Warning: Order quantity ({quantity_in_base_units_input:.6f} base units) is effectively zero. No order placed.")
            return

        # Determine the direction of the order itself (BUY or SELL)
        order_direction = LONG if cmd == 'buy' else SHORT if cmd == 'sell' else 0
        if order_direction == 0:
            print(f"Error: Invalid command '{cmd}'. Must be 'buy' or 'sell'.")
            return

        # This will be the actual quantity (in base units) that gets processed after all clamping
        actual_quantity_to_process_base_units = quantity_in_base_units_input

        # Handle USD capital clamping if currency_mode is 'USD'
        if self.currency_mode == 'USD':
            current_active_pos = self.get_active_position() # Get the single active position in one-way mode

            # Calculate potential USD cost/value of the incoming order, considering leverage
            # This is the cost basis for the amount of base units in this order.
            incoming_order_usd_cost = quantity_in_base_units_input * current_price * leverage

            if current_active_pos is None: # No active position, this is an opening order
                # The total capital for this initial order must not exceed max_position_size (USD)
                # Calculate max base units that can be bought given max_position_size USD capital
                if (current_price * leverage) == 0: # Avoid division by zero
                    max_base_units_from_usd_capital_limit = 0.0
                else:
                    max_base_units_from_usd_capital_limit = self.max_position_size / (current_price * leverage)
                
                # Clamp the incoming order quantity
                actual_quantity_to_process_base_units = min(quantity_in_base_units_input, max_base_units_from_usd_capital_limit)

                if actual_quantity_to_process_base_units < EPSILON:
                    if self.verbose and not isInitializing():
                        print(f"Warning: Calculated initial order quantity ({actual_quantity_to_process_base_units:.6f} base units) is effectively zero after USD capital clamping. No position opened.")
                    return # Exit if clamped to zero

            elif order_direction == current_active_pos.type: # Increasing an existing position
                # Calculate remaining USD capital capacity based on active_capital_invested
                remaining_usd_capital_capacity = self.max_position_size - current_active_pos.active_capital_invested

                if remaining_usd_capital_capacity <= EPSILON:
                    if self.verbose and not isInitializing():
                        print(f"Warning: Max USD position size ({self.max_position_size}) based on capital invested reached. No new {cmd} order placed.")
                    return # Exit if no more USD capacity

                # Calculate max base units that can be bought with remaining USD capacity
                if (current_price * leverage) == 0: # Avoid division by zero
                    max_base_units_from_usd_capital_capacity = 0.0
                else:
                    max_base_units_from_usd_capital_capacity = remaining_usd_capital_capacity / (current_price * leverage)
                
                # Clamp the incoming order quantity
                actual_quantity_to_process_base_units = min(quantity_in_base_units_input, max_base_units_from_usd_capital_capacity)

                if actual_quantity_to_process_base_units < EPSILON:
                    if self.verbose and not isInitializing():
                        print(f"Warning: Calculated order quantity ({actual_quantity_to_process_base_units:.6f} base units) is effectively zero after USD capital clamping. No order placed.")
                    return # Exit if clamped to zero

            # Note: For opposing orders (partial close/reversal), no capital clamping is applied here.
            # They are meant to reduce/close the position.
            # `actual_quantity_to_process_base_units` remains `quantity_in_base_units_input` in this case.


        # --- General logic that applies to both modes after `actual_quantity_to_process_base_units` is determined ---

        if self.hedged:
            active_target_pos = self.get_active_position(target_position_type)

            if active_target_pos is None:
                # No active position of the target_position_type, so open a new one
                if order_direction == target_position_type:
                    # In USD mode, `actual_quantity_to_process_base_units` already clamped.
                    # In BASE mode, clamp by `self.max_position_size`
                    clamped_quantity_final = actual_quantity_to_process_base_units
                    if self.currency_mode == 'BASE':
                        clamped_quantity_final = min(actual_quantity_to_process_base_units, self.max_position_size)

                    if clamped_quantity_final > EPSILON:
                        self.open_position(target_position_type, current_price, clamped_quantity_final, leverage)
                    elif self.verbose and not isInitializing():
                        print(f"Warning: Attempted to open new position, but calculated final quantity ({clamped_quantity_final:.2f} base units) is effectively zero. No position opened.")
                else:
                    if self.verbose and not isInitializing():
                        print(f"Warning: In hedged mode, cannot open a {target_position_type} position with a {cmd} order without an active position. Please ensure order direction matches target position type for opening.")
                    
            else: # An active position of the target_position_type exists
                if order_direction == active_target_pos.type:
                    # Order direction matches existing position type: increase position
                    # `actual_quantity_to_process_base_units` is already clamped by USD capital or initial quantity
                    # For BASE mode, it needs to be clamped against max_position_size_base_units (which is self.max_position_size)
                    clamped_quantity_final = actual_quantity_to_process_base_units
                    if self.currency_mode == 'BASE':
                        available_space_base_units = self.max_position_size - active_target_pos.size
                        clamped_quantity_final = min(actual_quantity_to_process_base_units, available_space_base_units)

                    if clamped_quantity_final > EPSILON:
                        active_target_pos.update(order_direction, current_price, clamped_quantity_final, leverage)
                    elif self.verbose and not isInitializing():
                        print(f"Warning: Attempted to increase {active_target_pos.type} position but calculated final quantity ({clamped_quantity_final:.2f} base units) is effectively zero. No change to position.")
                else:
                    # Order direction opposes existing position type: reduce or close
                    if actual_quantity_to_process_base_units < active_target_pos.size - EPSILON:
                        # Partial close: reduce the existing position
                        active_target_pos.update(order_direction, current_price, actual_quantity_to_process_base_units, leverage)
                    elif actual_quantity_to_process_base_units >= active_target_pos.size - EPSILON:
                        # Full close: close the existing position (or oversized order that just closes)
                        pos_type_being_closed = active_target_pos.type # Get type before closing
                        active_target_pos.close(current_price)
                        # Determine marker color based on the type of position that was closed
                        marker_color = '#00FF00' if pos_type_being_closed == LONG else '#FF0000'
                        createMarker('❌', location='above', shape='square', color=marker_color)
                        
                        if actual_quantity_to_process_base_units > active_target_pos.size + EPSILON: # Corrected check for oversized close
                            if self.verbose and not isInitializing():
                                print(f"Warning: Attempted to close a {active_target_pos.type} position with an oversized {cmd} order.")
                                print(f"Position was fully closed. Remaining quantity ({actual_quantity_to_process_base_units - active_target_pos.size:.2f} base units) was not used to open a new position.")

        else: # --- ONEWAY MODE LOGIC (Only one position at a time) ---
            current_overall_active_pos = self.get_active_position() # Get THE active position, if any type

            if current_overall_active_pos is None:
                # No active position, open a new one (only if order direction matches target type for consistency)
                if order_direction == target_position_type:
                    # `actual_quantity_to_process_base_units` already clamped by USD capital or initial quantity
                    # For BASE mode, it needs to be clamped against max_position_size
                    clamped_quantity_final = actual_quantity_to_process_base_units
                    if self.currency_mode == 'BASE':
                        clamped_quantity_final = min(actual_quantity_to_process_base_units, self.max_position_size)
                        
                    if clamped_quantity_final > EPSILON:
                        self.open_position(target_position_type, current_price, clamped_quantity_final, leverage)
                    elif self.verbose and not isInitializing():
                        print(f"Warning: Attempted to open new position, but calculated final quantity ({clamped_quantity_final:.2f} base units) is effectively zero. No position opened.")
                else:
                    if self.verbose and not isInitializing():
                        print(f"Warning: In one-way mode, no active position. Cannot {cmd} to target {target_position_type} type when opening a new position (order direction mismatch).")
            
            elif order_direction == current_overall_active_pos.type:
                # Order direction matches current overall position: increase existing position
                # `actual_quantity_to_process_base_units` already clamped by USD capital or initial quantity
                # For BASE mode, it needs to be clamped against max_position_size
                clamped_quantity_final = actual_quantity_to_process_base_units
                if self.currency_mode == 'BASE':
                    available_space_base_units = self.max_position_size - current_overall_active_pos.size
                    clamped_quantity_final = min(actual_quantity_to_process_base_units, available_space_base_units)

                if clamped_quantity_final > EPSILON:
                    current_overall_active_pos.update(order_direction, current_price, clamped_quantity_final, leverage)
                elif self.verbose and not isInitializing():
                    print(f"Warning: Attempted to increase {current_overall_active_pos.type} position but calculated final quantity ({clamped_quantity_final:.2f} base units) is effectively zero. No change to position.")

            elif order_direction != current_overall_active_pos.type:
                # Order direction opposes current overall position: close/reverse
                
                # Check if the order is large enough to close the existing position
                if actual_quantity_to_process_base_units >= current_overall_active_pos.size - EPSILON:
                    # Reversal: Close existing position and open new one with remaining quantity
                    pos_size_to_close = current_overall_active_pos.size
                    pos_type_being_closed = current_overall_active_pos.type # Get type before closing
                    
                    current_overall_active_pos.close(current_price)
                    # Determine marker color based on the type of position that was closed
                    marker_color = '#00FF00' if pos_type_being_closed == LONG else '#FF0000'
                    createMarker('❌', location='above', shape='square', color=marker_color)
                    
                    # Remaining quantity after closing the old position
                    remaining_quantity_base_units_for_new_pos = actual_quantity_to_process_base_units - pos_size_to_close
                    
                    if remaining_quantity_base_units_for_new_pos > EPSILON: 
                        # Open a new position with the remaining quantity, respecting max_position_size as a USD capital limit
                        clamped_new_pos_quantity_base_units = remaining_quantity_base_units_for_new_pos
                        if self.currency_mode == 'USD':
                             # Remaining USD capacity for a NEW position is `max_position_size`
                            if (current_price * leverage) == 0:
                                max_base_units_for_new_pos_from_usd_limit = 0.0
                            else:
                                max_base_units_for_new_pos_from_usd_limit = self.max_position_size / (current_price * leverage)
                            clamped_new_pos_quantity_base_units = min(remaining_quantity_base_units_for_new_pos, max_base_units_for_new_pos_from_usd_limit)
                        elif self.currency_mode == 'BASE':
                            clamped_new_pos_quantity_base_units = min(remaining_quantity_base_units_for_new_pos, self.max_position_size)

                        if clamped_new_pos_quantity_base_units > EPSILON:
                            self.open_position(order_direction, current_price, clamped_new_pos_quantity_base_units, leverage)
                            if self.verbose and not isInitializing():
                                print(f"Position reversed: Closed {pos_type_being_closed} position and opened new {order_direction} position with {clamped_new_pos_quantity_base_units:.2f} base units.")
                        elif self.verbose and not isInitializing():
                            print(f"Warning: Position closed. Remaining quantity ({remaining_quantity_base_units_for_new_pos:.2f} base units) clamped to 0 because it exceeds max_position_size or is too small. No new position opened.")
                    elif self.verbose and not isInitializing():
                        print(f"Position of type {'LONG' if pos_type_being_closed == LONG else 'SHORT'} was fully closed by an exact or slightly oversized order. No new position opened.")
                else:
                    # Partial close of the existing position
                    current_overall_active_pos.update(order_direction, current_price, actual_quantity_to_process_base_units, leverage)
                    if self.verbose and not isInitializing():
                        print(f"Partial close: Reduced {current_overall_active_pos.type} position by {actual_quantity_to_process_base_units:.2f} base units.")

    def close_position(self, pos_type: int = None): # pos_type is now optional
        """
        Closes a specific active position (LONG or SHORT) at the current realtime candle's close price.
        If hedged=False and pos_type is None, it closes the single active position, if any.

        Args:
            pos_type (int, optional): The type of position to close (LONG or SHORT).
                                      If hedged is False and this is None, closes any active position.
        """
        pos_to_close = None
        if self.hedged:
            if pos_type is None:
                if self.verbose and not isInitializing():
                    print("Warning: In hedged mode, 'close' requires a 'pos_type' (LONG or SHORT) to specify which position to close.")
                return
            pos_to_close = self.get_active_position(pos_type)
        else: # One-way mode
            if pos_type is None:
                pos_to_close = self.get_active_position() # Get the single active position, if any
            else:
                # If a type is specified, ensure it matches the actual active position in one-way mode
                current_active = self.get_active_position()
                if current_active and current_active.type == pos_type:
                    pos_to_close = current_active
                elif self.verbose and not isInitializing():
                    print(f"Warning: In one-way mode, attempted to close a { 'LONG' if pos_type == LONG else 'SHORT' } position, but the active position is {'LONG' if current_active.type == LONG else 'SHORT' if current_active.type == SHORT else 'None'}. No action taken.")
                    return


        if pos_to_close is None or not pos_to_close.active or pos_to_close.size < EPSILON:
            if self.verbose and not isInitializing():
                # Adjusted message for clarity based on whether a type was requested
                type_str = f" { 'LONG' if pos_type == LONG else 'SHORT' }" if pos_type is not None else ""
                print(f"No active{type_str} position to close.")
            return
        
        # Execute order to close the position
        # Determine the closing order type based on the position's type
        close_cmd = 'sell' if pos_to_close.type == LONG else 'buy'
        # Call execute_order with quantity_in_base_units (pos_to_close.size is already in base units)
        self.execute_order(close_cmd, pos_to_close.type, pos_to_close.size) 

    def get_total_profit_loss(self) -> float:
        """
        Returns the total accumulated profit or loss for the strategy.
        """
        return self.total_profit_loss

    def get_average_winning_trade_pnl(self) -> float:
        """
        Calculates the average realized PnL (in quote currency) for all closed winning trades.
        Returns 0.0 if there are no winning trades.
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
        Calculates the average realized PnL (in quote currency) for all closed losing trades.
        Returns 0.0 if there are no losing trades. The returned value is positive.
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
        Prints a list of all positions held by the strategy,
        including their status and the history of buy/sell orders within each.
        Also provides a summary of active and closed positions.
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
            position_type_str = "LONG" if pos.type == LONG else ("SHORT" if pos.type == SHORT else "N/A (Type not set)")
            
            # Get unrealized PnL only if active
            unrealized_pnl_qty = pos.get_unrealized_pnl_quantity() if pos.active else 0.0
            unrealized_pnl_pct = pos.get_unrealized_pnl_percentage() if pos.active else 0.0
            
            # Added active_capital_invested to print
            print(f"\nPosition #{i+1} (Status: {status}, Type: {position_type_str}, Current Size: {pos.size:.2f} base units, Avg Price: {pos.priceAvg:.2f}, Max Size Held: {pos.max_size_held:.2f} base units, Active Capital Invested: {pos.active_capital_invested:.2f} USD)") 
            if pos.active:
                print(f"   Unrealized PnL: {unrealized_pnl_qty:.2f} (quote currency) ({unrealized_pnl_pct:.2f}%)")
            else:
                print(f"   Realized PnL: {pos.profit:.2f} (quote currency) ({pos.realized_pnl_percentage:.2f}%)")

            print("   Order History:")

            if not pos.order_history:
                print("     No orders in this position's history.")
            else:
                for j, order_data in enumerate(pos.order_history):
                    order_type_str = "BUY" if order_data['type'] == LONG else "SELL"
                    pnl_info = f" | Realized PnL: {order_data['pnl_quantity']:.2f} (quote currency) ({order_data['pnl_percentage']:.2f}%)" if order_data['pnl_quantity'] != 0 or order_data['pnl_percentage'] != 0 else ""
                    # Increased precision for quantity display
                    print(f"     Order {j+1}: {order_type_str} {order_data['quantity']:.6f} base units at {order_data['price']:.2f} (Bar Index: {order_data['barindex']}){pnl_info}") 
        
        print("\n--- Position Summary ---")
        print(f"Active LONG positions: {active_long_count}")
        print(f"Active SHORT positions: {active_short_count}")
        print(f"Closed LONG positions: {closed_long_count}")
        print(f"Closed SHORT positions: {closed_short_count}")
        print(f"Total Strategy PnL: {self.total_profit_loss:.2f} (quote currency)")
        print("------------------------------------------")

    def print_summary_stats(self):
        """
        Prints a summary of the strategy's overall performance.
        """
        print("\n--- Strategy Summary Stats ---")
        
        # Calculate metrics
        total_closed_positions = self.total_winning_positions + self.total_losing_positions
        
        pnl_quantity = self.total_profit_loss # Already in quote currency
        
        # PnL percentage compared to initial_liquidity (initial_liquidity is always in USD)
        pnl_percentage_vs_liquidity = (pnl_quantity / self.initial_liquidity) * 100 if self.initial_liquidity != 0 else 0.0
        
        # This calculation's meaning (PnL % vs Max Pos) depends on currency_mode for max_position_size
        # It's kept as is to match previous output structure.
        pnl_percentage_vs_max_pos_size = (pnl_quantity / self.max_position_size) * 100 if self.max_position_size != 0 else 0.0

        profitable_trades = self.total_winning_positions
        losing_trades = self.total_losing_positions
        
        percentage_profitable_trades = (profitable_trades / total_closed_positions) * 100 if total_closed_positions > 0 else 0.0

        avg_winning_pnl = self.get_average_winning_trade_pnl()
        avg_losing_pnl = self.get_average_losing_trade_pnl()

        # Print header
        print(f"{'Pos PnL %':<12} {'Total PnL':<12} {'Trades':<8} {'Wins':<8} {'Losses':<8} {'Avg+ PnL':<12} {'Avg- PnL':<12} {'Win Rate %':<12} {'Acct PnL %':<12}")
        # Print values
        print(f"{pnl_percentage_vs_max_pos_size:<12.2f} {pnl_quantity:<12.2f} {total_closed_positions:<8} {profitable_trades:<8} {losing_trades:<8} {avg_winning_pnl:<12.2f} {avg_losing_pnl:<12.2f} {percentage_profitable_trades:<12.2f} {pnl_percentage_vs_liquidity:<12.2f}")
        print("------------------------------")


class position_c:
    """
    Represents an individual trading position (long or short).
    Handles opening, updating (increasing/reducing), and closing orders.
    Keeps a history of all orders made within this position.
    """
    def __init__(self, strategy_instance: strategy_c): # Accept strategy instance during initialization
        self.strategy_instance = strategy_instance # Store reference to the parent strategy
        self.active = False        # Is the position currently open?
        # self.type will be 1 for LONG or -1 for SHORT, even when closed.
        # It's initialized to 0, but will be set to LONG/SHORT on the first order.
        self.type = 0            
        self.size = 0.0          # Current size of the position (absolute value, always in base units)
        self.priceAvg = 0.0      # Average entry price of the position (in quote currency)
        self.leverage = 1        # Leverage applied to the position (assumed constant for this position object)
        self.profit = 0.0        # Total realized PnL for this position when it closes (final value, in quote currency)
        self.realized_pnl_quantity = 0.0 # Cumulative realized PnL in quantity (in quote currency)
        self.realized_pnl_percentage = 0.0 # Cumulative realized PnL in percentage for this position (final value)
        self.order_history = []  # Stores {'type': LONG/SHORT, 'price': float, 'quantity': float, 'barindex': int, 'pnl_quantity': float, 'pnl_percentage': float}
        self.max_size_held = 0.0 # Variable to track maximum size held during the position's lifetime (in base units)
        self.active_capital_invested = 0.0 # NEW: Tracks the USD capital (cost basis) currently tied up in the open position

    def _recalculate_current_position_state(self):
        """
        Recalculates the position's current size (in base units), average price (in quote currency),
        and active_capital_invested based on the accumulated order history.
        The 'type' is only set if the net quantity is non-zero,
        otherwise, it retains its last known direction or remains 0 if no orders.
        This method does not change `self.active`.
        """
        net_long_quantity = 0.0
        net_long_value = 0.0 # Value in quote currency
        net_short_quantity = 0.0
        net_short_value = 0.0 # Value in quote currency

        for order_data in self.order_history:
            if order_data['type'] == LONG:
                net_long_quantity += order_data['quantity']
                net_long_value += order_data['price'] * order_data['quantity']
            elif order_data['type'] == SHORT:
                net_short_quantity += order_data['quantity']
                net_short_value += order_data['price'] * order_data['quantity']

        # Determine the net position in base units
        net_quantity = net_long_quantity - net_short_quantity
        net_value = net_long_value - net_short_value # Sum of value in quote currency

        if net_quantity > EPSILON: # Use EPSILON for comparison
            self.type = LONG
            self.size = net_quantity
            self.priceAvg = net_value / net_quantity
        elif net_quantity < -EPSILON: # Use EPSILON for comparison
            self.type = SHORT
            self.size = abs(net_quantity) # Size is always positive (in base units)
            self.priceAvg = abs(net_value / net_quantity) # Average price for short (in quote currency)
        else: # net_quantity is effectively zero, position is flat
            self.size = 0.0
            self.priceAvg = 0.0
            # self.type retains its last non-zero value, or remains 0 if no orders yet.
            # self.active will be set to False by the close method.

        # Calculate active_capital_invested based on the current (recalculated) state
        if abs(self.size) > EPSILON: # Only if there's an active position size
            self.active_capital_invested = self.priceAvg * self.size * self.leverage
        else:
            self.active_capital_invested = 0.0


    def get_average_entry_price_from_history(self) -> float:
        """
        Calculates and returns the average entry price (in quote currency) based on the current order history.
        This is a public method for auditing/display.
        """
        net_long_quantity = 0.0
        net_long_value = 0.0
        net_short_quantity = 0.0
        net_short_value = 0.0

        for order_data in self.order_history:
            if order_data['type'] == LONG:
                net_long_quantity += order_data['quantity']
                net_long_value += order_data['price'] * order_data['quantity']
            elif order_data['type'] == SHORT:
                net_short_quantity += order_data['quantity']
                net_short_value += order_data['price'] * order_data['quantity']

        net_quantity = net_long_quantity - net_short_quantity
        net_value = net_long_value - net_short_value

        if abs(net_quantity) > EPSILON: # Use EPSILON for comparison
            return abs(net_value / net_quantity)
        return 0.0

    def update(self, op_type: int, price: float, quantity: float, leverage: int):
        """
        Records an order into the position's history and then recalculates
        the position's current state (type, size, average price, active_capital_invested).
        Handles markers based on the net change in position.
        This method is primarily for increasing or partially reducing a position.
        Quantity must always be in base units.
        """
        # Store the state before the update for marker logic
        previous_active = self.active
        previous_type = self.type # Store previous type
        previous_size = self.size # Size in base units

        # Initialize PnL for this order
        pnl_q = 0.0
        pnl_pct = 0.0

        # Calculate PnL if this order reduces the position size
        if previous_active and op_type != self.type and quantity > EPSILON:
            # This order is opposing the current position type, thus reducing it
            reduced_quantity = min(quantity, previous_size) # The quantity being reduced (in base units)
            if self.type == LONG:
                pnl_q = (price - self.priceAvg) * reduced_quantity * self.leverage
            elif self.type == SHORT:
                pnl_q = (self.priceAvg - price) * reduced_quantity * self.leverage
            
            # Capital involved is in quote currency
            capital_involved = self.priceAvg * reduced_quantity
            if capital_involved > EPSILON:
                pnl_pct = (pnl_q / capital_involved) * 100
            else:
                pnl_pct = 0.0 # Avoid division by zero if capital involved is zero
            
            # Add this order's realized PnL to the position's cumulative realized PnL
            self.realized_pnl_quantity += pnl_q

        # Record the order in history with PnL
        self.order_history.append({
            'type': op_type,
            'price': price, # Price in quote currency
            'quantity': quantity, # Quantity in base units
            'barindex': active.barindex,
            'pnl_quantity': pnl_q,
            'pnl_percentage': pnl_pct
        })

        # Set leverage only if it's the first order for this position object
        if not previous_active:
            self.leverage = leverage
            self.type = op_type # Set initial type when position becomes active

        # Recalculate metrics based on the updated history
        self._recalculate_current_position_state()

        # Update max_size_held if current size is greater
        if self.size > self.max_size_held:
            self.max_size_held = self.size

        # Determine marker color based on the *action* (buy/sell)
        marker_color = '#00FF00' if op_type == LONG else '#FF0000' # Green for Buy (LONG order type), Red for Sell (SHORT order type)
        marker_shape = 'arrow_up' if op_type == LONG else 'arrow_down'

        # Handle markers based on the *net* change in position
        # Markers for opening are handled in openPosition (already correct)
        if previous_active and self.active: # Position is still active
            # If the type has changed, it implies a reversal that didn't fully close the prior position
            if self.type != previous_type and self.size > EPSILON:
                createMarker('🔄', location='inside', shape='circle', color='#FFD700') # Gold circle for partial reversal
            elif self.size > previous_size + EPSILON: # Increasing position
                createMarker('➕', location='below', shape=marker_shape, color=marker_color)
            elif self.size < previous_size - EPSILON: # Decreasing position
                createMarker('➖', location='above', shape=marker_shape, color=marker_color)
        # If previous_active was True and self.active is now False, it means the position was closed.
        # This specific case is handled by the `close` method.

    def close(self, price: float):
        """
        Closes the active position by adding an opposing order that nets out the current size.
        Calculates the total realized profit/loss for this position and adds it to the global total.
        This method no longer adds the '❌' marker; that is handled by the 'execute_order' function for full closes.
        """
        if not self.active:
            return

        closing_quantity = self.size # This is in base units
        closing_op_type = SHORT if self.type == LONG else LONG # Use the position's current type to determine closing order type

        # Store previous state for PnL calculation on the closing order
        previous_avg_price = self.priceAvg
        previous_leverage = self.leverage
        previous_position_type = self.type

        # Add the closing order to history
        self.order_history.append({
            'type': closing_op_type,
            'price': price, # Price in quote currency
            'quantity': closing_quantity, # Quantity in base units
            'barindex': active.barindex,
            'pnl_quantity': 0.0, # Will be updated after full PnL calculation
            'pnl_percentage': 0.0 # Will be updated after full PnL calculation
        })

        # Recalculate metrics based on the updated history
        # This call should result in self.size == 0.0 if the closing order perfectly nets out.
        self._recalculate_current_position_state()

        # If the position is now effectively closed (size is 0), calculate total PnL for this position object
        if abs(self.size) < EPSILON:
            # Calculate PnL for the final closing order
            final_close_pnl_q = 0.0
            if previous_position_type == LONG:
                final_close_pnl_q = (price - previous_avg_price) * closing_quantity * previous_leverage
            elif previous_position_type == SHORT:
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
            # Sum only entry capital (orders where pnl_quantity is approx 0) to get total invested capital.
            total_entry_capital_for_position = sum(order_data['price'] * order_data['quantity'] for order_data in self.order_history if order_data['pnl_quantity'] >= -EPSILON and order_data['pnl_quantity'] <= EPSILON )
            self.realized_pnl_percentage = (self.profit / total_entry_capital_for_position) * 100 if total_entry_capital_for_position != 0 else 0.0

            # Update strategy-level counters for winning/losing positions
            if self.profit > EPSILON:
                self.strategy_instance.total_winning_positions += 1
            elif self.profit < -EPSILON:
                self.strategy_instance.total_losing_positions += 1

            self.strategy_instance.total_profit_loss += self.profit # Update strategy's total PnL
            self.active = False # Explicitly set to inactive as it's fully closed
            self.active_capital_invested = 0.0 # Reset capital invested when position is closed

            # Print PnL to console
            if self.strategy_instance.verbose and not isInitializing():
                print(f"CLOSED POSITION ({'LONG' if previous_position_type == LONG else 'SHORT'}): PnL: {self.profit:.2f} (quote currency) | PnL %: {self.realized_pnl_percentage:.2f}% | Total Strategy PnL: {self.strategy_instance.total_profit_loss:.2f} (quote currency)")

    def get_unrealized_pnl_quantity(self) -> float:
        """
        Calculates and returns the current unrealized Profit and Loss in quantity (quote currency)
        for this active position.
        Returns 0.0 if the position is not active.
        """
        if not self.active or self.size < EPSILON:
            return 0.0

        current_price = getRealtimeCandle().close # Price is always valid here
        unrealized_pnl = 0.0

        if self.type == LONG:
            unrealized_pnl = (current_price - self.priceAvg) * self.size * self.leverage
        elif self.type == SHORT:
            unrealized_pnl = (self.priceAvg - current_price) * self.size * self.leverage
        
        return unrealized_pnl

    def get_unrealized_pnl_percentage(self) -> float:
        """
        Calculates and returns the current unrealized Profit and Loss as a percentage
        of the capital invested for this active position.
        Returns 0.0 if the position is not active or if the average entry price is zero.
        """
        unrealized_pnl_q = self.get_unrealized_pnl_quantity()
        if unrealized_pnl_q == 0.0 and (not self.active or self.size < EPSILON):
            return 0.0

        # Capital involved in the active position based on entry price and quantity (in quote currency)
        # This should be self.active_capital_invested if it's correctly maintained as cost basis.
        capital_involved = self.active_capital_invested
        if abs(capital_involved) < EPSILON: # Avoid division by zero
            return 0.0
        
        return (unrealized_pnl_q / abs(capital_involved)) * 100


# Global instance of the strategy.
# Initialized with currency_mode='USD' as requested.
strategy = strategy_c(hedged=False, currency_mode='USD') 

# The following global functions will now call methods on the 'strategy' instance.
# This maintains compatibility with existing calls from other modules (e.g., algorizer.py).
def openPosition(pos_type: int, price: float, quantity: float, leverage: int) -> position_c:
    """
    Opens a new position. Quantity must be in base units.
    """
    return strategy.open_position(pos_type, price, quantity, leverage)

def getActivePosition(pos_type: int = None) -> position_c:
    return strategy.get_active_position(pos_type)

def direction() -> int:
    return strategy.get_direction()

def order(cmd: str, target_position_type: int, quantity: float = None, leverage: int = 1):
    """
    Places a trading order.
    The 'quantity' parameter's interpretation depends on strategy.currency_mode:
    - If currency_mode is 'USD', quantity is interpreted as a notional USD amount.
    - If currency_mode is 'BASE', quantity is interpreted as base currency units.
    """
    actual_quantity_base_units = 0.0
    current_price = getRealtimeCandle().close # Get price here for conversion if needed

    if strategy.currency_mode == 'USD':
        if quantity is None:
            # Use default order_size which is in USD
            actual_quantity_base_units = strategy.order_size / current_price
        else:
            # Convert provided USD quantity to base units
            actual_quantity_base_units = quantity / current_price

        # Safety check for extremely small quantities (if USD amount is too small for current price)
        if actual_quantity_base_units < EPSILON:
            if strategy.verbose and not isInitializing():
                print(f"Warning: Calculated order quantity ({actual_quantity_base_units:.6f} base units) is effectively zero based on current price ({current_price:.2f}). Order not placed.")
            return
    else: # strategy.currency_mode == 'BASE'
        if quantity is None:
            actual_quantity_base_units = strategy.order_size
        else:
            actual_quantity_base_units = quantity

    strategy.execute_order(cmd, target_position_type, actual_quantity_base_units, leverage)

def close(pos_type: int = None):
    strategy.close_position(pos_type)

def get_total_profit_loss() -> float:
    return strategy.get_total_profit_loss()

def print_strategy_stats(): # This will become print_detailed_stats
    strategy.print_detailed_stats()

def print_summary_stats(): # New function
    strategy.print_summary_stats()