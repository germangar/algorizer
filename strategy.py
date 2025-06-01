

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
    def __init__(self, initial_liquidity: float = 10000.0, verbose: bool = False, order_size: float = 100.0, max_position_size: float = 100.0, hedged: bool = False):
        self.positions = []   # List to hold all positions (both active and closed, Long and Short)
        self.total_profit_loss = 0.0 # Global variable to keep track of the total profit/loss for the entire strategy
        self.initial_liquidity = initial_liquidity # Starting capital for the strategy
        self.total_winning_positions = 0 # Counter for winning closed positions
        self.total_losing_positions = 0   # Counter for losing closed positions
        self.verbose = verbose # Controls whether warning prints are displayed
        self.order_size = order_size # Default quantity to use when none is provided
        self.max_position_size = max_position_size # Maximum allowed total size for any single position
        self.hedged = hedged # NEW: Controls whether multiple positions can be open simultaneously (True) or only one (False)

        # Validate parameters
        if self.max_position_size > self.initial_liquidity:
            raise ValueError(f"max_position_size ({self.max_position_size}) cannot be greater than initial_liquidity ({self.initial_liquidity}).")
        if self.order_size <= 0:
            raise ValueError("order_size must be a positive value.")
        if self.max_position_size <= 0:
            raise ValueError("max_position_size must be a positive value.")


    def open_position(self, pos_type: int, price: float, quantity: float, leverage: int) -> 'position_c':
        """
        Creates and opens a new position, associated with this strategy instance.
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

    def execute_order(self, cmd: str, target_position_type: int, quantity: float = None, leverage: int = 1): # Removed price parameter
        """
        Executes a trading order. Behavior depends on the 'hedged' flag.
        The price is retrieved from getRealtimeCandle().close internally.

        If hedged=True: Allows simultaneous LONG and SHORT positions.
        If hedged=False (One-Way Mode): Only one position (LONG or SHORT) can be active at a time.
            An oversized opposing order will close the current position and open a new one.

        Args:
            cmd (str): The command ('buy' or 'sell').
            target_position_type (int): Specifies which position (LONG or SHORT) this order targets.
            quantity (float, optional): The quantity for the order. If None or 0, uses self.order_size.
            leverage (int, optional): The leverage. Defaults to 1.
        """
        if cmd is None:
            return

        # Retrieve the current price directly
        current_price = getRealtimeCandle().close # Price is now retrieved here

        cmd = cmd.lower()
        
        # Use default order_size if quantity is not provided or zero
        if quantity is None or quantity <= EPSILON:
            quantity = self.order_size

        # Determine the direction of the order itself (BUY or SELL)
        order_direction = 0
        if cmd == 'buy':
            order_direction = LONG
        elif cmd == 'sell':
            order_direction = SHORT
        else:
            print(f"Error: Invalid command '{cmd}'. Must be 'buy' or 'sell'.")
            return

        if self.hedged:
            # --- HEDGED MODE LOGIC (Allows multiple positions) ---
            active_target_pos = self.get_active_position(target_position_type)

            if active_target_pos is None:
                # No active position of the target_position_type, so open a new one
                if order_direction == target_position_type:
                    # Clamp quantity to max_position_size if opening a new position
                    clamped_quantity = min(quantity, self.max_position_size)
                    if clamped_quantity > EPSILON:
                        self.open_position(target_position_type, current_price, clamped_quantity, leverage) # Pass current_price
                    elif self.verbose and not isInitializing():
                        print(f"Warning: Order quantity ({quantity:.2f}) clamped to 0 because it exceeds max_position_size ({self.max_position_size:.2f}) for opening a new position. No position opened.")
                else:
                    if self.verbose and not isInitializing():
                        print(f"Warning: In hedged mode, cannot open a {target_position_type} position with a {cmd} order without an active position. Please ensure order direction matches target position type for opening.")
                    
            else: # An active position of the target_position_type exists
                if order_direction == active_target_pos.type:
                    # Order direction matches existing position type: increase position
                    available_space = self.max_position_size - active_target_pos.size
                    clamped_quantity = min(quantity, available_space)

                    if clamped_quantity > EPSILON:
                        active_target_pos.update(order_direction, current_price, clamped_quantity, leverage) # Pass current_price
                    elif self.verbose and not isInitializing():
                        print(f"Warning: Attempted to increase {active_target_pos.type} position but order quantity ({quantity:.2f}) clamped to 0 because it would exceed max_position_size ({self.max_position_size:.2f}). No change to position.")
                else:
                    # Order direction opposes existing position type: reduce or close
                    if quantity < active_target_pos.size - EPSILON:
                        # Partial close: reduce the existing position
                        active_target_pos.update(order_direction, current_price, quantity, leverage) # Pass current_price
                    elif quantity >= active_target_pos.size - EPSILON:
                        # Full close: close the existing position (or oversized order that just closes)
                        active_target_pos.close(current_price) # Pass current_price
                        createMarker('❌', location='above', shape='square', color='#808080') # Grey square for full close
                        if quantity > active_target_pos.size + EPSILON:
                            if self.verbose and not isInitializing():
                                print(f"Warning: Attempted to close a {active_target_pos.type} position with an oversized {cmd} order.")
                                print(f"Position was fully closed. Remaining quantity ({quantity - active_target_pos.size:.2f}) was not used to open a new position.")

        else: # --- ONEWAY MODE LOGIC (Only one position at a time) ---
            current_overall_active_pos = self.get_active_position() # Get THE active position, if any type

            if current_overall_active_pos is None:
                # No active position, open a new one (only if order direction matches target type for consistency)
                if order_direction == target_position_type:
                    clamped_quantity = min(quantity, self.max_position_size)
                    if clamped_quantity > EPSILON:
                        self.open_position(target_position_type, current_price, clamped_quantity, leverage) # Pass current_price
                    elif self.verbose and not isInitializing():
                        print(f"Warning: Order quantity ({quantity:.2f}) clamped to 0 because it exceeds max_position_size ({self.max_position_size:.2f}) for opening a new position. No position opened.")
                else:
                    if self.verbose and not isInitializing():
                        print(f"Warning: In one-way mode, no active position. Cannot {cmd} to target {target_position_type} type when opening a new position (order direction mismatch).")
            
            elif order_direction == current_overall_active_pos.type:
                # Order direction matches current overall position: increase existing position
                available_space = self.max_position_size - current_overall_active_pos.size
                clamped_quantity = min(quantity, available_space)
                if clamped_quantity > EPSILON:
                    current_overall_active_pos.update(order_direction, current_price, clamped_quantity, leverage) # Pass current_price
                elif self.verbose and not isInitializing():
                    print(f"Warning: Attempted to increase {current_overall_active_pos.type} position but order quantity ({quantity:.2f}) clamped to 0 because it would exceed max_position_size ({self.max_position_size:.2f}). No change to position.")

            elif order_direction != current_overall_active_pos.type:
                # Order direction opposes current overall position: close/reverse
                
                # Check if the order is large enough to close the existing position
                if quantity >= current_overall_active_pos.size - EPSILON:
                    # Reversal: Close existing position and open new one with remaining quantity
                    pos_size_to_close = current_overall_active_pos.size
                    
                    # Close the existing position
                    # Note: pos.close() marks it inactive and calculates PnL
                    current_overall_active_pos.close(current_price) # Pass current_price
                    createMarker('❌', location='above', shape='square', color='#808080') # Marker for full close

                    remaining_quantity = quantity - pos_size_to_close
                    
                    if remaining_quantity > EPSILON:
                        # Open a new position in the opposite direction with the remaining quantity
                        clamped_new_pos_quantity = min(remaining_quantity, self.max_position_size)
                        if clamped_new_pos_quantity > EPSILON:
                            self.open_position(order_direction, current_price, clamped_new_pos_quantity, leverage) # Pass current_price
                            if self.verbose and not isInitializing():
                                print(f"Position reversed: Closed {current_overall_active_pos.type} position and opened new {order_direction} position with {clamped_new_pos_quantity:.2f} quantity.")
                        elif self.verbose and not isInitializing():
                            print(f"Warning: Position closed. Remaining quantity ({remaining_quantity:.2f}) clamped to 0 because it exceeds max_position_size ({self.max_position_size:.2f}) for opening a new position. No new position opened.")
                    elif self.verbose and not isInitializing():
                        print(f"Position of type {'LONG' if current_overall_active_pos.type == LONG else 'SHORT'} was fully closed by an exact or slightly oversized order. No new position opened.")
                else:
                    # Partial close of the existing position
                    current_overall_active_pos.update(order_direction, current_price, quantity, leverage) # Pass current_price
                    if self.verbose and not isInitializing():
                        print(f"Partial close: Reduced {current_overall_active_pos.type} position by {quantity:.2f}.")

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
        
        # We no longer need to retrieve getRealtimeCandle().close here,
        # as execute_order will retrieve it internally.
        
        # Execute order to close the position
        # Determine the closing order type based on the position's type
        close_cmd = 'sell' if pos_to_close.type == LONG else 'buy'
        # Call execute_order without price parameter
        self.execute_order(close_cmd, pos_to_close.type, pos_to_close.size) 

    def get_total_profit_loss(self) -> float:
        """
        Returns the total accumulated profit or loss for the strategy.
        """
        return self.total_profit_loss

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
            
            print(f"\nPosition #{i+1} (Status: {status}, Type: {position_type_str}, Current Size: {pos.size:.2f}, Avg Price: {pos.priceAvg:.2f}, Max Size Held: {pos.max_size_held:.2f})")
            if pos.active:
                print(f"   Unrealized PnL: {unrealized_pnl_qty:.2f} ({unrealized_pnl_pct:.2f}%)")
            else:
                print(f"   Realized PnL: {pos.profit:.2f} ({pos.realized_pnl_percentage:.2f}%)")

            print("   Order History:")

            if not pos.order_history:
                print("     No orders in this position's history.")
            else:
                for j, order_data in enumerate(pos.order_history):
                    order_type_str = "BUY" if order_data['type'] == LONG else "SELL"
                    pnl_info = f" | Realized PnL: {order_data['pnl_quantity']:.2f} ({order_data['pnl_percentage']:.2f}%)" if order_data['pnl_quantity'] != 0 or order_data['pnl_percentage'] != 0 else ""
                    print(f"     Order {j+1}: {order_type_str} {order_data['quantity']:.2f} at {order_data['price']:.2f} (Bar Index: {order_data['barindex']}){pnl_info}")
        
        print("\n--- Position Summary ---")
        print(f"Active LONG positions: {active_long_count}")
        print(f"Active SHORT positions: {active_short_count}")
        print(f"Closed LONG positions: {closed_long_count}")
        print(f"Closed SHORT positions: {closed_short_count}")
        print(f"Total Strategy PnL: {self.total_profit_loss:.2f}")
        print("------------------------------------------")

    def print_summary_stats(self):
        """
        Prints a summary of the strategy's overall performance.
        """
        print("\n--- Strategy Summary Stats ---")
        
        # Calculate metrics
        total_closed_positions = self.total_winning_positions + self.total_losing_positions
        
        pnl_quantity = self.total_profit_loss
        
        # PnL percentage compared to initial_liquidity
        pnl_percentage_vs_liquidity = (pnl_quantity / self.initial_liquidity) * 100 if self.initial_liquidity != 0 else 0.0
        
        # PnL percentage compared to max_position_size
        pnl_percentage_vs_max_pos_size = (pnl_quantity / self.max_position_size) * 100 if self.max_position_size != 0 else 0.0

        profitable_trades = self.total_winning_positions
        losing_trades = self.total_losing_positions
        
        percentage_profitable_trades = (profitable_trades / total_closed_positions) * 100 if total_closed_positions > 0 else 0.0

        # Print labels
        print(f"{'PnL %':<10} {'PnL (Qty)':<15} {'Trades':<10} {'Trades+':<10} {'Trades-':<10} {'Win Rate %':<15} {'Account PnL %':<20}")
        # Print values
        print(f"{pnl_percentage_vs_max_pos_size:<10.2f} {pnl_quantity:<15.2f} {total_closed_positions:<10} {profitable_trades:<10} {losing_trades:<10} {percentage_profitable_trades:<15.2f} {pnl_percentage_vs_liquidity:<20.2f}")
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
        self.size = 0.0          # Current size of the position (absolute value)
        self.priceAvg = 0.0      # Average entry price of the position
        self.leverage = 1        # Leverage applied to the position (assumed constant for this position object)
        self.profit = 0.0        # Total realized PnL for this position when it closes (final value)
        self.realized_pnl_quantity = 0.0 # Cumulative realized PnL in quantity for this position
        self.realized_pnl_percentage = 0.0 # Cumulative realized PnL in percentage for this position (final value)
        self.order_history = []  # Stores {'type': LONG/SHORT, 'price': float, 'quantity': float, 'barindex': int, 'pnl_quantity': float, 'pnl_percentage': float}
        self.max_size_held = 0.0 # Variable to track maximum size held during the position's lifetime

    def _recalculate_current_position_state(self):
        """
        Recalculates the position's current size and average price
        based on the accumulated order history.
        The 'type' is only set if the net quantity is non-zero,
        otherwise, it retains its last known direction or remains 0 if no orders.
        This method does not change `self.active`.
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

        # Determine the net position
        net_quantity = net_long_quantity - net_short_quantity
        net_value = net_long_value - net_short_value

        if net_quantity > EPSILON: # Use EPSILON for comparison
            self.type = LONG
            self.size = net_quantity
            self.priceAvg = net_value / net_quantity
        elif net_quantity < -EPSILON: # Use EPSILON for comparison
            self.type = SHORT
            self.size = abs(net_quantity) # Size is always positive
            self.priceAvg = abs(net_value / net_quantity) # Average price for short is also positive
        else: # net_quantity is effectively zero, position is flat
            self.size = 0.0
            self.priceAvg = 0.0
            # self.type retains its last non-zero value, or remains 0 if no orders yet.
            # self.active will be set to False by the close method.

    def get_average_entry_price_from_history(self) -> float:
        """
        Calculates and returns the average entry price based on the current order history.
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
        the position's current state (type, size, average price).
        Handles markers based on the net change in position.
        This method is primarily for increasing or partially reducing a position.
        Full closures and reversals are handled by the 'order' function.
        """
        # Store the state before the update for marker logic
        previous_active = self.active
        previous_type = self.type # Store previous type
        previous_size = self.size

        # Initialize PnL for this order
        pnl_q = 0.0
        pnl_pct = 0.0

        # Calculate PnL if this order reduces the position size
        if previous_active and op_type != self.type and quantity > EPSILON:
            # This order is opposing the current position type, thus reducing it
            reduced_quantity = min(quantity, previous_size) # The quantity being reduced
            if self.type == LONG:
                pnl_q = (price - self.priceAvg) * reduced_quantity * self.leverage
            elif self.type == SHORT:
                pnl_q = (self.priceAvg - price) * reduced_quantity * self.leverage
            
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
            'price': price,
            'quantity': quantity,
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

        # Determine marker shape based on the order type (op_type)
        marker_shape = 'arrow_up' if op_type == LONG else 'arrow_down'

        # Handle markers based on the *net* change in position
        # Markers for opening are handled in openPosition
        if previous_active and self.active: # Position is still active
            # If the type has changed, it implies a reversal that didn't fully close the prior position
            if self.type != previous_type and self.size > EPSILON:
                createMarker('🔄', location='inside', shape='circle', color='#FFD700') # Gold circle for partial reversal
            elif self.size > previous_size + EPSILON:
                createMarker('➕', location='below', shape=marker_shape, color='#00FF00') # Green based on order type
            elif self.size < previous_size - EPSILON:
                createMarker('➖', location='above', shape=marker_shape, color='#FF0000') # Red based on order type
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

        closing_quantity = self.size
        closing_op_type = SHORT if self.type == LONG else LONG # Use the position's current type to determine closing order type

        # Store previous state for PnL calculation on the closing order
        previous_avg_price = self.priceAvg
        previous_leverage = self.leverage
        previous_position_type = self.type

        # Add the closing order to history
        self.order_history.append({
            'type': closing_op_type,
            'price': price,
            'quantity': closing_quantity,
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
            
            final_close_capital_involved = previous_avg_price * closing_quantity
            final_close_pnl_pct = 0.0
            if final_close_capital_involved > EPSILON:
                final_close_pnl_pct = (final_close_pnl_q / final_close_capital_involved) * 100
            
            # Update the last order in history with its PnL
            self.order_history[-1]['pnl_quantity'] = final_close_pnl_q
            self.order_history[-1]['pnl_percentage'] = final_close_pnl_pct

            # Add the PnL from the final closing order to the cumulative realized PnL
            self.realized_pnl_quantity += final_close_pnl_q
            
            self.profit = self.realized_pnl_quantity # Total position PnL is the cumulative realized PnL
            
            # Calculate final percentage PnL for the entire position
            total_entry_capital_for_position = sum(order_data['price'] * order_data['quantity'] for order_data in self.order_history if order_data['pnl_quantity'] >= -EPSILON and order_data['pnl_quantity'] <= EPSILON ) # Sum only entry capital. If PnL is 0, it means it was an entry order.
            self.realized_pnl_percentage = (self.profit / total_entry_capital_for_position) * 100 if total_entry_capital_for_position != 0 else 0.0

            # Update strategy-level counters for winning/losing positions
            if self.profit > EPSILON:
                self.strategy_instance.total_winning_positions += 1
            elif self.profit < -EPSILON:
                self.strategy_instance.total_losing_positions += 1

            self.strategy_instance.total_profit_loss += self.profit # Update strategy's total PnL
            self.active = False # Explicitly set to inactive as it's fully closed

            # Print PnL to console
            if self.strategy_instance.verbose and not isInitializing():
                print(f"CLOSED POSITION ({'LONG' if previous_position_type == LONG else 'SHORT'}): PnL: {self.profit:.2f} | PnL %: {self.realized_pnl_percentage:.2f}% | Total Strategy PnL: {self.strategy_instance.total_profit_loss:.2f}")

    def get_unrealized_pnl_quantity(self) -> float:
        """
        Calculates and returns the current unrealized Profit and Loss in quantity for this active position.
        Returns 0.0 if the position is not active or if current price is not available.
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

        # Capital involved in the active position based on entry price and quantity
        capital_involved = self.priceAvg * self.size * self.leverage

        if abs(capital_involved) < EPSILON: # Avoid division by zero
            return 0.0
        
        return (unrealized_pnl_q / abs(capital_involved)) * 100


# Global instance of the strategy.
# You can now initialize it with hedged=True for hedged mode or hedged=False for one-way mode
strategy = strategy_c(hedged=False) # Defaulting to one-way mode as per request. Change to True for hedged mode.

# The following global functions will now call methods on the 'strategy' instance.
# This maintains compatibility with existing calls from other modules (e.g., algorizer.py).
def openPosition(pos_type: int, price: float, quantity: float, leverage: int) -> position_c:
    return strategy.open_position(pos_type, price, quantity, leverage)

def getActivePosition(pos_type: int = None) -> position_c:
    return strategy.get_active_position(pos_type)

def direction() -> int:
    return strategy.get_direction()

def order(cmd: str, target_position_type: int, quantity: float = None, leverage: int = 1): # Removed price parameter
    strategy.execute_order(cmd, target_position_type, quantity, leverage) # Removed price argument

def close(pos_type: int = None):
    strategy.close_position(pos_type)

def get_total_profit_loss() -> float:
    return strategy.get_total_profit_loss()

def print_strategy_stats(): # This will become print_detailed_stats
    strategy.print_detailed_stats()

def print_summary_stats(): # New function
    strategy.print_summary_stats()