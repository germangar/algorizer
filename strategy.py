

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
    def __init__(self, initial_liquidity: float = 10000.0, verbose: bool = False, order_size: float = 100.0, max_position_size: float = 100.0):
        self.positions = []  # List to hold all positions (both active and closed, Long and Short)
        self.total_profit_loss = 0.0 # Global variable to keep track of the total profit/loss for the entire strategy
        self.initial_liquidity = initial_liquidity # Starting capital for the strategy
        self.total_winning_positions = 0 # Counter for winning closed positions
        self.total_losing_positions = 0  # Counter for losing closed positions
        self.verbose = verbose # Controls whether warning prints are displayed
        self.order_size = order_size # Default quantity to use when none is provided
        self.max_position_size = max_position_size # Maximum allowed total size for any single position

        # Validate max_position_size
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
        If pos_type is None, it returns the first active position found.
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

    def execute_order(self, cmd: str, target_position_type: int, price: float, quantity: float = None, leverage: int = 1):
        """
        Executes a trading order in hedged mode.
        Handles opening, increasing, reducing, and closing specific LONG or SHORT positions.
        Reversals are not allowed; an oversized opposing order will simply close the position.

        Args:
            cmd (str): The command ('buy' or 'sell').
            target_position_type (int): Specifies which position (LONG or SHORT) this order targets.
            price (float): The price at which the order is executed.
            quantity (float, optional): The quantity for the order. If None or 0, uses self.order_size.
            leverage (int, optional): The leverage. Defaults to 1.
        """
        if cmd is None:
            return

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

        # Get the specific active position (LONG or SHORT) that this order is targeting
        active_target_pos = self.get_active_position(target_position_type)

        if active_target_pos is None:
            # No active position of the target_position_type, so open a new one
            # This implies order_direction must match target_position_type for opening
            if order_direction == target_position_type:
                # Clamp quantity to max_position_size if opening a new position
                clamped_quantity = min(quantity, self.max_position_size)
                if clamped_quantity > EPSILON:
                    self.open_position(target_position_type, price, clamped_quantity, leverage)
                elif self.verbose and not isInitializing():
                    print(f"Warning: Order quantity ({quantity:.2f}) clamped to 0 because it exceeds max_position_size ({self.max_position_size:.2f}) for opening a new position. No position opened.")
            else:
                print(f"Warning: Cannot open a {target_position_type} position with a {cmd} order without an active position.")
                print("To open a LONG position, use 'buy' with LONG target_position_type.")
                print("To open a SHORT position, use 'sell' with SHORT target_position_type.")
                print("To close an existing position, use an opposing order with the correct target_position_type.")
                
        else: # An active position of the target_position_type exists
            if order_direction == active_target_pos.type:
                # Order direction matches existing position type: increase position
                # Calculate available space up to max_position_size
                available_space = self.max_position_size - active_target_pos.size
                clamped_quantity = min(quantity, available_space)

                if clamped_quantity > EPSILON:
                    active_target_pos.update(order_direction, price, clamped_quantity, leverage)
                elif self.verbose and not isInitializing():
                    print(f"Warning: Attempted to increase {active_target_pos.type} position but order quantity ({quantity:.2f}) clamped to 0 because it would exceed max_position_size ({self.max_position_size:.2f}). No change to position.")
            else:
                # Order direction opposes existing position type: reduce or close
                # Compare with EPSILON to handle floating point inaccuracies
                if quantity < active_target_pos.size - EPSILON:
                    # Partial close: reduce the existing position
                    active_target_pos.update(order_direction, price, quantity, leverage)
                elif quantity >= active_target_pos.size - EPSILON: # quantity is greater than or effectively equal to position size
                    # Full close: close the existing position (or oversized order that just closes)
                    active_target_pos.close(price) # This calls position_c.close()
                    createMarker('❌', location='above', shape='square', color='#808080') # Grey square for full close
                    if quantity > active_target_pos.size + EPSILON: # If it was an oversized order
                        if self.verbose and not isInitializing():
                            print(f"Warning: Attempted to close a {active_target_pos.type} position with an oversized {cmd} order.")
                            print(f"Position was fully closed. Remaining quantity ({quantity - active_target_pos.size:.2f}) was not used to open a new position.")

    def close_position(self, pos_type: int):
        """
        Closes a specific active position (LONG or SHORT) at the current realtime candle's close price.

        Args:
            pos_type (int): The type of position to close (LONG or SHORT).
        """
        pos_to_close = self.get_active_position(pos_type)
        if pos_to_close is None or not pos_to_close.active or pos_to_close.size < EPSILON: # Use EPSILON
            if self.verbose and not isInitializing():
                print(f"No active { 'LONG' if pos_type == LONG else 'SHORT' } position to close.")
            return
        
        realtimeCandle = getRealtimeCandle()
        if realtimeCandle is not None:
            if pos_type == LONG:
                # To close a LONG position, we issue a SELL order targeting the LONG position
                self.execute_order('sell', LONG, realtimeCandle.close, pos_to_close.size)
            elif pos_type == SHORT:
                # To close a SHORT position, we issue a BUY order targeting the SHORT position
                self.execute_order('buy', SHORT, realtimeCandle.close, pos_to_close.size)

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
            
            print(f"\nPosition #{i+1} (Status: {status}, Type: {position_type_str}, Current Size: {pos.size:.2f}, Avg Price: {pos.priceAvg:.2f}, PnL: {pos.profit:.2f})")
            print("  Order History:")

            if pos.active:
                if pos.type == LONG:
                    active_long_count += 1
                elif pos.type == SHORT:
                    active_short_count += 1
            else:
                if pos.type == LONG:
                    closed_long_count += 1
                elif pos.type == SHORT:
                    closed_short_count += 1

            if not pos.order_history:
                print("    No orders in this position's history.")
            else:
                for j, order_data in enumerate(pos.order_history):
                    order_type_str = "BUY" if order_data['type'] == LONG else "SELL"
                    pnl_info = f" | PnL: {order_data['pnl_quantity']:.2f} ({order_data['pnl_percentage']:.2f}%)" if order_data['pnl_quantity'] != 0 or order_data['pnl_percentage'] != 0 else ""
                    print(f"    Order {j+1}: {order_type_str} {order_data['quantity']:.2f} at {order_data['price']:.2f} (Bar Index: {order_data['barindex']}){pnl_info}")
        
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
    Kee
    ps a history of all orders made within this position.
    """
    def __init__(self, strategy_instance: strategy_c): # Accept strategy instance during initialization
        self.strategy_instance = strategy_instance # Store reference to the parent strategy
        self.active = False      # Is the position currently open?
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
        This method is now primarily for increasing or partially reducing a position.
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
        if previous_active and op_type != self.type and quantity > EPSILON: # Use EPSILON
            # This order is opposing the current position type, thus reducing it
            reduced_quantity = min(quantity, previous_size) # The quantity being reduced
            if self.type == LONG:
                pnl_q = (price - self.priceAvg) * reduced_quantity * self.leverage
            elif self.type == SHORT:
                pnl_q = (self.priceAvg - price) * reduced_quantity * self.leverage
            
            capital_involved = self.priceAvg * reduced_quantity
            if capital_involved > EPSILON: # Use EPSILON
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

        # Determine marker shape based on the order type (op_type)
        marker_shape = 'arrow_up' if op_type == LONG else 'arrow_down'

        # Handle markers based on the *net* change in position
        # Markers for opening are handled in openPosition
        if previous_active and self.active: # Position is still active
            # If the type has changed, it implies a reversal that didn't fully close the prior position
            # This logic is mostly for partial reversals now that 'order' handles full reversals
            if self.type != previous_type and self.size > EPSILON: # Use EPSILON
                createMarker('🔄', location='inside', shape='circle', color='#FFD700') # Gold circle for partial reversal
            elif self.size > previous_size + EPSILON: # Increase - Use EPSILON
                createMarker('➕', location='below', shape=marker_shape, color='#00FF00') # Green based on order type
            elif self.size < previous_size - EPSILON: # Decrease (partial close) - Use EPSILON
                createMarker('➖', location='above', shape=marker_shape, color='#FF0000') # Red based on order type
        # If previous_active was True and self.active is now False, it means the position was closed.
        # This specific case is handled by the `close` method.

    def close(self, price: float):
        """
        Closes the active position by adding an opposing order that nets out the current size.
        Calculates the total realized profit/loss for this position and adds it to the global total.
        This method no longer adds the '❌' marker; that is handled by the 'order' function.
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
        if abs(self.size) < EPSILON: # Use EPSILON for comparison
            # Calculate PnL for the final closing order
            final_close_pnl_q = 0.0
            if previous_position_type == LONG:
                final_close_pnl_q = (price - previous_avg_price) * closing_quantity * previous_leverage
            elif previous_position_type == SHORT:
                final_close_pnl_q = (previous_avg_price - price) * closing_quantity * previous_leverage
            
            final_close_capital_involved = previous_avg_price * closing_quantity
            final_close_pnl_pct = 0.0
            if final_close_capital_involved > EPSILON: # Use EPSILON
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

            # The '❌' marker is now handled by the 'order' function for full closes.
            # createMarker('❌') # Removed from here

            # Print PnL to console
            if self.strategy_instance.verbose and not isInitializing():
                print(f"CLOSED POSITION ({'LONG' if previous_position_type == LONG else 'SHORT'}): PnL: {self.profit:.2f} | PnL %: {self.realized_pnl_percentage:.2f}% | Total Strategy PnL: {self.strategy_instance.total_profit_loss:.2f}")

        # If the position is not fully closed (e.g., partial close), self.active remains True
        # and no PnL is calculated for the position object yet.

# Global instance of the strategy.
strategy = strategy_c() # Initialized here, will manage all positions and stats

# The following global functions will now call methods on the 'strategy' instance.
# This maintains compatibility with existing calls from other modules (e.g., algorizer.py).
def openPosition(pos_type: int, price: float, quantity: float, leverage: int) -> position_c:
    return strategy.open_position(pos_type, price, quantity, leverage)

def getActivePosition(pos_type: int = None) -> position_c:
    return strategy.get_active_position(pos_type)

def direction() -> int:
    return strategy.get_direction()

def order(cmd: str, target_position_type: int, price: float, quantity: float = None, leverage: int = 1):
    strategy.execute_order(cmd, target_position_type, price, quantity, leverage)

def close(pos_type: int):
    strategy.close_position(pos_type)

def get_total_profit_loss() -> float:
    return strategy.get_total_profit_loss()

def print_strategy_stats(): # This will become print_detailed_stats
    strategy.print_detailed_stats()

def print_summary_stats(): # New function
    strategy.print_summary_stats()
