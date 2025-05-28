

# THIS FILE HAS BEEN WRITTEN BY GEMINI AI 
# I find this annoying to write so I just asked the AI to do it. 
# The output doesn't seem correct, but I guess I'll work it from here.




from candle import candle_c
from algorizer import getRealtimeCandle, createMarker
import active # Import active to get active.barindex

# Define constants for position types
SHORT = -1
LONG = 1

# List to hold active and closed positions
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

        for order in self.order_history:
            if order['type'] == LONG:
                net_long_quantity += order['quantity']
                net_long_value += order['price'] * order['quantity']
            elif order['type'] == SHORT:
                net_short_quantity += order['quantity']
                net_short_value += order['price'] * order['quantity']

        # Determine the net position
        net_quantity = net_long_quantity - net_short_quantity
        net_value = net_long_value - net_short_value

        if net_quantity > 0:
            self.type = LONG
            self.size = net_quantity
            self.priceAvg = net_value / net_quantity
        elif net_quantity < 0:
            self.type = SHORT
            self.size = abs(net_quantity) # Size is always positive
            self.priceAvg = abs(net_value / net_quantity) # Average price for short is also positive
        else: # net_quantity == 0, position is flat
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

        for order in self.order_history:
            if order['type'] == LONG:
                net_long_quantity += order['quantity']
                net_long_value += order['price'] * order['quantity']
            elif order['type'] == SHORT:
                net_short_quantity += order['quantity']
                net_short_value += order['price'] * order['quantity']

        net_quantity = net_long_quantity - net_short_quantity
        net_value = net_long_value - net_short_value

        if net_quantity != 0:
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

        # Handle markers based on the *net* change in position
        if not previous_active and self.active: # Opened a new position (should be handled by openPosition)
            createMarker('🟢' if self.type == LONG else '🔴')
        elif previous_active and self.active: # Position is still active
            # If the type has changed, it implies a reversal that didn't fully close the prior position
            # This logic is mostly for partial reversals now that 'order' handles full reversals
            if self.type != previous_type and self.size > 0:
                createMarker('🔄')
            elif self.size > previous_size: # Increase
                createMarker('➕')
            elif self.size < previous_size: # Decrease (partial close)
                createMarker('➖')
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
        if self.size == 0.0:
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
                if temp_long_orders[0][1] == 0:
                    temp_long_orders.pop(0)
                if temp_short_orders[0][1] == 0:
                    temp_short_orders.pop(0)

            self.profit = total_realized_pnl
            global total_profit_loss
            total_profit_loss += self.profit
            self.active = False # Explicitly set to inactive as it's fully closed

            # The '❌' marker is now handled by the 'order' function for full closes.
            # createMarker('❌') # Removed from here

            # Print PnL to console
            total_capital_involved = sum(order_data['price'] * order_data['quantity'] for order_data in self.order_history)
            pnl_percentage = (self.profit / total_capital_involved) * 100 if total_capital_involved != 0 else 0

            print(f"CLOSED POSITION ({'LONG' if self.type == LONG else 'SHORT'}): PnL: {self.profit:.2f} | PnL %: {pnl_percentage:.2f}% | Total Strategy PnL: {total_profit_loss:.2f}")

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
    createMarker('🟢' if pos_type == LONG else '🔴') # Mark the opening of the position on the chart
    return pos

def getActivePosition() -> position_c:
    """
    Retrieves the currently active position.

    Returns:
        position_c: The active position object, or None if no position is active.
    """
    if len(positions) == 0:
        return None
    # Iterate from the end to find the most recent active position
    for pos in reversed(positions):
        if pos.active:
            return pos
    return None # No active positions found

def direction() -> int:
    """
    Returns the direction of the active position.

    Returns:
        int: 1 for LONG, -1 for SHORT, 0 if no position is active.
    """
    pos = getActivePosition()
    if pos is None:
        return 0
    return pos.type

def order(cmd: str, price: float, quantity: float, leverage: int = 1):
    """
    Executes a trading order (buy, sell, or close).
    Handles opening, increasing, reducing, closing, and reversing positions.
    """
    if cmd is None:
        return

    cmd = cmd.lower()
    active_pos = getActivePosition() # Get the current active position

    if cmd == 'buy':
        op_type = LONG
    elif cmd == 'sell':
        op_type = SHORT
    elif cmd == 'close':
        if active_pos is None or not active_pos.active:
            return # Nothing to close
        active_pos.close(price) # Call the close method on the active position
        createMarker('❌') # Add X marker for explicit close
        return # Close command is handled, exit

    # Handle buy/sell commands
    if active_pos is None:
        # No active position, open a new one
        openPosition(op_type, price, quantity, leverage)
    else:
        # Active position exists, check interaction
        if active_pos.type == op_type:
            # Same direction, increase position
            active_pos.update(op_type, price, quantity, leverage)
        else:
            # Opposing direction, potentially reduce, close, or reverse
            if quantity < active_pos.size:
                # Partial close: reduce the existing position
                active_pos.update(op_type, price, quantity, leverage)
            elif quantity == active_pos.size:
                # Full close: close the existing position
                active_pos.close(price)
                createMarker('❌') # Add X marker for full close
            else: # quantity > active_pos.size
                # Reversal: Close existing position and and open a new one in the opposite direction
                size_to_close = active_pos.size

                # 1. Close the current position fully
                # The close method will handle PnL calculation and setting active=False
                # It no longer adds the '❌' marker here.
                active_pos.close(price)

                # 2. Open a new position with the remaining quantity in the new direction
                remaining_quantity_for_new_pos = quantity - size_to_close
                if remaining_quantity_for_new_pos > 0: # Only open if there's a remaining quantity
                    openPosition(op_type, price, remaining_quantity_for_new_pos, leverage)
                else:
                    # This case should ideally not happen if quantity > size_to_close
                    print("Warning: Reversal quantity calculation resulted in non-positive remaining quantity for new position.")


def close():
    """
    Closes the active position at the current realtime candle's close price.
    """
    pos = getActivePosition()
    if pos is None:
        return
    if not pos.active or pos.size == 0.0:
        return
    realtimeCandle = getRealtimeCandle()
    if realtimeCandle is not None:
        # Call the order function to handle the close, which will add the marker
        order('close', realtimeCandle.close, pos.size)

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
    """
    print("\n--- Strategy Positions and Order History ---")
    if not positions:
        print("No positions have been opened yet.")
        return
    
    activeCount = 0
    closedCount = 0

    for i, pos in enumerate(positions):
        status = "Active" if pos.active else "Closed"
        # Display the position's type (LONG/SHORT) even if it's closed
        position_type_str = "LONG" if pos.type == LONG else ("SHORT" if pos.type == SHORT else "N/A")
        
        print(f"\nPosition #{i+1} (Status: {status}, Type: {position_type_str}, Current Size: {pos.size:.2f}, Avg Price: {pos.priceAvg:.2f}, PnL: {pos.profit:.2f})")
        print("  Order History:")

        if( not pos.active ):
            closedCount += 1
        else:
            activeCount += 1
        if not pos.order_history:
            print("    No orders in this position's history.")
        else:
            for j, order_data in enumerate(pos.order_history):
                order_type_str = "BUY" if order_data['type'] == LONG else "SELL"
                print(f"    Order {j+1}: {order_type_str} {order_data['quantity']:.2f} at {order_data['price']:.2f} (Bar Index: {order_data['barindex']})")
    print( f"Closed positions:{closedCount} - Active positions:{activeCount}")
    print("------------------------------------------")


