

# THIS FILE HAS BEEN WRITTEN BY GEMINI AI 
# I find this annoying to write so I just asked the AI to do it. 
# The output doesn't seem correct, but I guess I'll work it from here.


from candle import candle_c
from algorizer import getRealtimeCandle, createMarker
import active # Import active to get active.barindex

# Define constants for position types
SHORT = -1
LONG = 1

class strategy_c:
    """
    Represents the overall trading strategy, managing positions and global statistics.
    """
    def __init__(self, initial_liquidity: float = 10000.0, default_position_size: float = 10.0):
        self.positions = []  # List to hold active and closed positions
        self.total_profit_loss = 0.0 # Global total profit/loss for the strategy
        self.initial_liquidity = initial_liquidity # Starting capital
        self.default_position_size = default_position_size # Default size for new orders
        self.total_winning_positions = 0
        self.total_losing_positions = 0

    def open_position(self, pos_type: int, price: float, quantity: float, leverage: int) -> 'position_c':
        """
        Creates and opens a new position, associated with this strategy instance.
        """
        pos = position_c(self) # Pass self (strategy_c instance) to position_c
        pos.leverage = leverage # Set initial leverage for the position object
        
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

        self.positions.append(pos) # Add the new position to the strategy's list
        createMarker('🟢' if pos_type == LONG else '🔴') # Mark the opening of the position on the chart
        return pos

    def get_active_position(self) -> 'position_c':
        """
        Retrieves the currently active position for this strategy.

        Returns:
            position_c: The active position object, or None if no position is active.
        """
        if not self.positions:
            return None
        # Iterate from the end to find the most recent active position
        for pos in reversed(self.positions):
            if pos.active:
                return pos
        return None # No active positions found

    def direction(self) -> int:
        """
        Returns the direction of the active position for this strategy.

        Returns:
            int: 1 for LONG, -1 for SHORT, 0 if no position is active.
        """
        pos = self.get_active_position()
        if pos is None:
            return 0
        return pos.type

    def order(self, cmd: str, price: float, quantity: float, leverage: int = 1):
        """
        Executes a trading order (buy, sell, or close) for this strategy.
        If there's an active position, it attempts to update it.
        If no active position and it's a buy/sell, it opens a new one.
        """
        if cmd is None:
            return

        cmd = cmd.lower()
        active_pos = self.get_active_position()

        if cmd == 'buy':
            if active_pos is None: # No active position, open a new LONG
                self.open_position(LONG, price, quantity, leverage)
            else: # Active position, update it
                active_pos.update(LONG, price, quantity, leverage)

        elif cmd == 'sell':
            if active_pos is None: # No active position, open a new SHORT
                self.open_position(SHORT, price, quantity, leverage)
            else: # Active position, update it
                active_pos.update(SHORT, price, quantity, leverage)

        elif cmd == 'close':
            if active_pos is None or not active_pos.active:
                return
            active_pos.close(price) # Close the active position

    def close_active_position(self):
        """
        Closes the active position for this strategy at the current realtime candle's close price.
        """
        pos = self.get_active_position()
        if pos is None:
            return
        if not pos.active or pos.size == 0.0:
            return
        realtimeCandle = getRealtimeCandle()
        if realtimeCandle is not None:
            pos.close(realtimeCandle.close)

    def get_total_profit_loss(self) -> float:
        """
        Returns the total accumulated profit or loss for the strategy.
        """
        return self.total_profit_loss

    def print_stats(self):
        """
        Prints various statistics about the strategy's performance.
        """
        print("\n--- Strategy Performance Statistics ---")
        print(f"Initial Liquidity: ${self.initial_liquidity:,.2f}")
        print(f"Current Total PnL: ${self.total_profit_loss:,.2f}")

        if self.initial_liquidity != 0:
            pnl_percentage = (self.total_profit_loss / self.initial_liquidity) * 100
            print(f"Total PnL Percentage: {pnl_percentage:,.2f}%")
        else:
            print("Total PnL Percentage: N/A (Initial Liquidity is zero)")

        total_closed_positions = self.total_winning_positions + self.total_losing_positions
        print(f"Total Positions Closed: {total_closed_positions}")
        print(f"  Winning Positions: {self.total_winning_positions}")
        print(f"  Losing Positions: {self.total_losing_positions}")

        # Calculate average profit/loss per trade
        avg_win = 0.0
        if self.total_winning_positions > 0:
            # Sum profit from all closed positions that were winners
            total_win_profit = sum(p.profit for p in self.positions if not p.active and p.profit > 0)
            avg_win = total_win_profit / self.total_winning_positions
        print(f"  Average Profit per Win: ${avg_win:,.2f}")

        avg_loss = 0.0
        if self.total_losing_positions > 0:
            # Sum loss from all closed positions that were losers
            total_loss_profit = sum(p.profit for p in self.positions if not p.active and p.profit < 0)
            avg_loss = total_loss_profit / self.total_losing_positions
        print(f"  Average Loss per Loss: ${avg_loss:,.2f}")

        # Win rate
        win_rate = (self.total_winning_positions / total_closed_positions) * 100 if total_closed_positions > 0 else 0
        print(f"Win Rate: {win_rate:,.2f}%")

        print("---------------------------------------")


class position_c:
    """
    Represents an individual trading position (long or short).
    Handles opening, updating (increasing/reducing), and closing orders.
    Keeps a history of all orders made within this position.
    """
    def __init__(self, strategy_instance: strategy_c): # Accept strategy instance during initialization
        self.strategy_instance = strategy_instance # Store reference to the parent strategy
        self.active = False      # Is the position currently open?
        self.type = 0            # -1 for SHORT, 1 for LONG, 0 for flat
        self.size = 0.0          # Current size of the position (absolute value)
        self.priceAvg = 0.0      # Average entry price of the position
        self.leverage = 1        # Leverage applied to the position (assumed constant for this position object)
        self.profit = 0.0        # Total realized PnL for this position when it closes
        self.order_history = []  # Stores {'type': LONG/SHORT, 'price': float, 'quantity': float, 'barindex': int}

    def _recalculate_current_position_state(self):
        """
        Recalculates the position's current type, size, and average price
        based on the accumulated order history. This method does not change `self.active`.
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
        else: # net_quantity == 0
            self.type = 0 # Flat
            self.size = 0.0
            self.priceAvg = 0.0

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
        """
        # Store the state before the update for marker logic
        previous_active = self.active
        previous_type = self.type
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

        # Recalculate metrics based on the updated history
        self._recalculate_current_position_state()

        # Handle markers based on the *net* change in position
        if not previous_active and self.active: # Opened a new position
            createMarker('🟢' if self.type == LONG else '🔴')
        elif previous_active and self.active: # Position is still active
            if self.type != previous_type: # Reversal
                createMarker('🔄')
            elif self.size > previous_size: # Increase
                createMarker('➕')
            elif self.size < previous_size: # Decrease
                createMarker('➖')
        # If previous_active was True and self.active is now False, it means the position was closed.
        # This specific case is handled by the `close` method.

    def close(self, price: float):
        """
        Closes the active position by adding an opposing order that nets out the current size.
        Calculates the total realized profit/loss for this position and adds it to the global total.
        """
        if not self.active:
            return

        closing_quantity = self.size
        closing_op_type = SHORT if self.type == LONG else LONG

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
            for order in self.order_history:
                if order['type'] == LONG:
                    temp_long_orders.append([order['price'], order['quantity']])
                elif order['type'] == SHORT:
                    temp_short_orders.append([order['price'], order['quantity']])

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
            
            # Update strategy-level statistics
            self.strategy_instance.total_profit_loss += self.profit
            if self.profit > 0:
                self.strategy_instance.total_winning_positions += 1
            elif self.profit < 0:
                self.strategy_instance.total_losing_positions += 1

            self.active = False # Explicitly set to inactive as it's fully closed

            createMarker('❌') # Mark position closed

            # Print PnL to console
            total_capital_involved = sum(order['price'] * order['quantity'] for order in self.order_history)
            pnl_percentage = (self.profit / total_capital_involved) * 100 if total_capital_involved != 0 else 0

            print(f"CLOSED POSITION: PnL: {self.profit:.2f} | PnL %: {pnl_percentage:.2f}% | Total Strategy PnL: {self.strategy_instance.total_profit_loss:.2f}")

        # If the position is not fully closed (e.g., partial close), self.active remains True
        # and no PnL is calculated for the position object yet.

# Create a global instance of the strategy. Other modules will import 'strategy' and use this instance.
strategy_instance = strategy_c(initial_liquidity=10000.0, default_position_size=10.0)

# Redefine the global functions to act as wrappers for the methods of strategy_instance.
# This maintains compatibility with existing calls from other modules (e.g., algorizer.py).
def openPosition(pos_type: int, price: float, quantity: float, leverage: int) -> position_c:
    return strategy_instance.open_position(pos_type, price, quantity, leverage)

def getActivePosition() -> position_c:
    return strategy_instance.get_active_position()

def direction() -> int:
    return strategy_instance.direction()

def order(cmd: str, price: float, quantity: float, leverage: int = 1):
    strategy_instance.order(cmd, price, quantity, leverage)

def close():
    strategy_instance.close_active_position()

def get_total_profit_loss() -> float:
    return strategy_instance.get_total_profit_loss()

def print_strategy_stats():
    strategy_instance.print_stats()
