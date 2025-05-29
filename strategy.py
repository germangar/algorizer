

# THIS FILE HAS BEEN WRITTEN BY GEMINI AI 
# I find this annoying to write so I just asked the AI to do it. 
# The output doesn't seem correct, but I guess I'll work it from here.


from candle import candle_c
from algorizer import getRealtimeCandle, createMarker
import active # Import active to get active.barindex

p_verbose = False

# Define constants for position types
SHORT = -1
LONG = 1

# Define a small epsilon for floating point comparisons to determine if a size is effectively zero
EPSILON = 1e-9

# List to hold all positions (both active and closed, Long and Short)
positions = []

# Global variable to keep track of the total profit/loss for the entire strategy
total_profit_loss = 0.0

class strategy_c:
    """
    Represents the overall trading strategy.
    Currently, it's a placeholder, but can be extended for global strategy parameters or methods.
    """
    def __init__(self):
        return

class position_c:
    """
    Represents an individual trading position (long or short).
    Handles opening, updating (increasing/reducing), and closing orders.
    Keeps a history of all orders made within this position.
    """
    def __init__(self):
        self.active = False      # Is the position currently open?
        # self.type will be 1 for LONG or -1 for SHORT, even when closed.
        # It's initialized to 0, but will be set to LONG/SHORT on the first order.
        self.type = 0            
        self.size = 0.0          # Current size of the position (absolute value)
        self.priceAvg = 0.0      # Average entry price of the position
        self.leverage = 1        # Leverage applied to the position (assumed constant for this position object)
        self.profit = 0.0        # Total realized PnL for this position when it closes
        self.order_history = []  # Stores {'type': LONG/SHORT, 'price': float, 'quantity': float, 'barindex': int}

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

        # Record the order in history
        self.order_history.append({
            'type': op_type,
            'price': price,
            'quantity': quantity,
            'barindex': active.barindex
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

        # Add the closing order to history
        self.order_history.append({
            'type': closing_op_type,
            'price': price,
            'quantity': closing_quantity,
            'barindex': active.barindex
        })

        # Recalculate metrics based on the updated history
        # This call should result in self.size == 0.0 if the closing order perfectly nets out.
        self._recalculate_current_position_state()

        # If the position is now effectively closed (size is 0), calculate total PnL for this position object
        if abs(self.size) < EPSILON: # Use EPSILON for comparison
            # Calculate total realized PnL for this position's entire lifecycle
            total_realized_pnl = 0.0
            temp_long_orders = [] # list of [price, quantity]
            temp_short_orders = [] # list of [price, quantity]

            # Populate temporary lists with orders from history
            for order_data in self.order_history: # Renamed 'order' to 'order_data' to avoid conflict
                if order_data['type'] == LONG:
                    temp_long_orders.append([order_data['price'], order_data['quantity']])
                elif order_data['type'] == SHORT:
                    temp_short_orders.append([order_data['price'], order_data['quantity']])

            # Match long and short orders to calculate realized PnL (FIFO matching)
            while temp_long_orders and temp_short_orders:
                long_price, long_qty = temp_long_orders[0]
                short_price, short_qty = temp_short_orders[0]

                matched_qty = min(long_qty, short_qty)

                # PnL for a long position closed by a short (sell)
                total_realized_pnl += (short_price - long_price) * matched_qty * self.leverage

                # Update remaining quantities
                temp_long_orders[0][1] -= matched_qty
                temp_short_orders[0][1] -= matched_qty

                # Remove orders that are fully matched
                if temp_long_orders[0][1] < EPSILON: # Use EPSILON
                    temp_long_orders.pop(0)
                if temp_short_orders[0][1] < EPSILON: # Use EPSILON
                    temp_short_orders.pop(0)

            self.profit = total_realized_pnl
            global total_profit_loss
            total_profit_loss += self.profit
            self.active = False # Explicitly set to inactive as it's fully closed

            # The '❌' marker is now handled by the 'order' function for full closes.
            # createMarker('❌') # Removed from here

            # Print PnL to console
            # Calculate total capital involved in entry trades for percentage PnL
            # This is a simplified calculation for the total capital involved in the position's history
            total_capital_involved = sum(order_data['price'] * order_data['quantity'] for order_data in self.order_history)

            pnl_percentage = (self.profit / total_capital_involved) * 100 if total_capital_involved != 0 else 0

            if p_verbose : print(f"CLOSED POSITION ({'LONG' if self.type == LONG else 'SHORT'}): PnL: {self.profit:.2f} | PnL %: {pnl_percentage:.2f}% | Total Strategy PnL: {total_profit_loss:.2f}")

        # If the position is not fully closed (e.g., partial close), self.active remains True
        # and no PnL is calculated for the position object yet.

# Modify openPosition to use the new update logic
def openPosition(pos_type: int, price: float, quantity: float, leverage: int) -> position_c:
    """
    Creates and opens a new position.
    The position's initial order is added to its history, and metrics are recalculated.
    """
    pos = position_c()
    pos.leverage = leverage # Set initial leverage for the position object
    pos.type = pos_type # Set the initial type of the position
    
    # Add the initial order to history
    pos.order_history.append({
        'type': pos_type,
        'price': price,
        'quantity': quantity,
        'barindex': active.barindex
    })
    
    # Recalculate metrics based on this first order
    pos._recalculate_current_position_state()
    pos.active = True # Explicitly set to active as it's just opened

    positions.append(pos) # Add the new position to the global list
    # Use specific marker for opening a position
    if pos_type == LONG:
        createMarker('🟢', location='below', shape='arrow_up', color='#00FF00') # Green arrow up for new LONG
    else: # SHORT
        createMarker('🔴', location='above', shape='arrow_down', color='#FF0000') # Red arrow down for new SHORT
    return pos

def getActivePosition(pos_type: int = None) -> position_c:
    """
    Retrieves the currently active position of a specific type (LONG or SHORT).
    If pos_type is None, it returns the first active position found (legacy behavior,
    but in hedged mode, it's better to specify type).
    """
    if len(positions) == 0:
        return None
    # Iterate from the end to find the most recent active position
    for pos in reversed(positions):
        if pos.active:
            if pos_type is None: # If no specific type requested, return the first active
                return pos
            elif pos.type == pos_type: # Return the active position of the specified type
                return pos
    return None # No active position of the specified type found

def direction() -> int:
    """
    Returns the direction of the first active position found.
    In hedged mode, it's better to use getActivePosition(LONG).type or getActivePosition(SHORT).type.
    """
    pos = getActivePosition() # This will return either a LONG or SHORT active position if one exists
    if pos is None:
        return 0
    return pos.type

def order(cmd: str, target_position_type: int, price: float, quantity: float, leverage: int = 1):
    """
    Executes a trading order in hedged mode.
    Handles opening, increasing, reducing, and closing specific LONG or SHORT positions.
    Reversals are not allowed; an oversized opposing order will simply close the position.

    Args:
        cmd (str): The command ('buy' or 'sell').
        target_position_type (int): Specifies which position (LONG or SHORT) this order targets.
        price (float): The price at which the order is executed.
        quantity (float): The quantity for the order.
        leverage (int, optional): The leverage. Defaults to 1.
    """
    if cmd is None:
        return

    cmd = cmd.lower()
    
    # Determine the direction of the order itself (BUY or SELL)
    order_direction = 0
    if cmd == 'buy':
        order_direction = LONG
    elif cmd == 'sell':
        order_direction = SHORT
    else:
        print(f"Error: Invalid command '{cmd}'. Must be 'buy' or 'sell'.")
        return

    # Get the specific active position (LONG or SHORT) that this order is targeting
    active_target_pos = getActivePosition(target_position_type)

    if active_target_pos is None:
        # No active position of the target_position_type, so open a new one
        # This implies order_direction must match target_position_type for opening
        if order_direction == target_position_type:
            openPosition(target_position_type, price, quantity, leverage)
        else:
            print(f"Warning: Cannot open a {target_position_type} position with a {cmd} order without an active position.")
            print("To open a LONG position, use 'buy' with LONG target_position_type.")
            print("To open a SHORT position, use 'sell' with SHORT target_position_type.")
            print("To close an existing position, use an opposing order with the correct target_position_type.")
            
    else: # An active position of the target_position_type exists
        if order_direction == active_target_pos.type:
            # Order direction matches existing position type: increase position
            active_target_pos.update(order_direction, price, quantity, leverage)
        else:
            # Order direction opposes existing position type: reduce or close
            # Compare with EPSILON to handle floating point inaccuracies
            if quantity < active_target_pos.size - EPSILON:
                # Partial close: reduce the existing position
                active_target_pos.update(order_direction, price, quantity, leverage)
                # No marker needed here, '➖' is added by position_c.update
            elif quantity >= active_target_pos.size - EPSILON: # quantity is greater than or effectively equal to position size
                # Full close: close the existing position (or oversized order that just closes)
                active_target_pos.close(price)
                createMarker('❌', location='above', shape='square', color='#808080') # Grey square for full close
                if p_verbose and quantity > active_target_pos.size + EPSILON: # If it was an oversized order
                    print(f"Warning: Attempted to close a {active_target_pos.type} position with an oversized {cmd} order.")
                    print(f"Position was fully closed. Remaining quantity ({quantity - active_target_pos.size:.2f}) was not used to open a new position.")


def close(pos_type: int):
    """
    Closes a specific active position (LONG or SHORT) at the current realtime candle's close price.

    Args:
        pos_type (int): The type of position to close (LONG or SHORT).
    """
    pos_to_close = getActivePosition(pos_type)
    if pos_to_close is None or not pos_to_close.active or pos_to_close.size < EPSILON: # Use EPSILON
        print(f"No active { 'LONG' if pos_type == LONG else 'SHORT' } position to close.")
        return
    
    realtimeCandle = getRealtimeCandle()
    if realtimeCandle is not None:
        if pos_type == LONG:
            # To close a LONG position, we issue a SELL order targeting the LONG position
            order('sell', LONG, realtimeCandle.close, pos_to_close.size)
        elif pos_type == SHORT:
            # To close a SHORT position, we issue a BUY order targeting the SHORT position
            order('buy', SHORT, realtimeCandle.close, pos_to_close.size)


def get_total_profit_loss() -> float:
    """
    Returns the total accumulated profit or loss for the strategy.

    Returns:
        float: The total profit or loss.
    """
    return total_profit_loss

def print_strategy_stats():
    """
    Prints a list of all positions held by the strategy,
    including their status and the history of buy/sell orders within each.
    Also provides a summary of active and closed positions.
    """
    print("\n--- Strategy Positions and Order History ---")
    if not positions:
        print("No positions have been opened yet.")
        return
    
    active_long_count = 0
    active_short_count = 0
    closed_long_count = 0
    closed_short_count = 0

    for i, pos in enumerate(positions):
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
                print(f"    Order {j+1}: {order_type_str} {order_data['quantity']:.2f} at {order_data['price']:.2f} (Bar Index: {order_data['barindex']})")
    
    print("\n--- Position Summary ---")
    print(f"Active LONG positions: {active_long_count}")
    print(f"Active SHORT positions: {active_short_count}")
    print(f"Closed LONG positions: {closed_long_count}")
    print(f"Closed SHORT positions: {closed_short_count}")
    print("------------------------------------------")
