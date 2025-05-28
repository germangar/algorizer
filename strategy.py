from candle import candle_c
from algorizer import getRealtimeCandle
from algorizer import createMarker

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
    Handles opening, updating (increasing/reducing), and closing positions.
    """
    def __init__(self):
        self.active = False      # Is the position currently open?
        self.type = 0            # -1 for SHORT, 1 for LONG
        self.size = 0.0          # Current size of the position (absolute value)
        self.priceAvg = 0.0      # Average entry price of the position
        self.leverage = 1        # Leverage applied to the position
        self.profit = 0.0        # Profit/Loss for this specific position when closed

    def update(self, op_type: int, price: float, quantity: float, leverage: int):
        """
        Updates an existing position or opens a new one.
        This method handles increasing, reducing, and reversing positions.

        Args:
            op_type (int): The type of operation (LONG or SHORT).
            price (float): The price at which the operation is executed.
            quantity (float): The quantity involved in the operation.
            leverage (int): The leverage for the operation.
        """
        global total_profit_loss # Declare intent to modify the global variable

        # If there's no active position, or the current operation is in the same direction,
        # or the current operation opens a new position after closing a previous one.
        if not self.active or self.type == op_type:
            # If the position is not active, initialize it
            if not self.active:
                self.active = True
                self.type = op_type
                self.size = quantity
                self.priceAvg = price
                self.leverage = leverage
                self.profit = 0.0 # Reset profit for a new position
                createMarker('🟢' if op_type == LONG else '🔴') # Mark new position
                return

            # If the position is active and in the same direction, increase size and update average price
            old_total_value = self.size * self.priceAvg
            new_total_value = quantity * price
            self.size += quantity
            self.priceAvg = (old_total_value + new_total_value) / self.size
            createMarker('➕') # Mark position increase
            return

        # Handling position reversal (e.g., LONG position and a SELL order)
        # Or reducing an existing position
        if self.type == LONG and op_type == SHORT:
            if self.size > quantity:
                # Reduce the long position
                self.size -= quantity
                createMarker('➖') # Mark position reduction
            elif self.size == quantity:
                # Close the long position
                self.profit = (price - self.priceAvg) * self.size * self.leverage
                total_profit_loss += self.profit
                pnl_percentage = (self.profit / (self.priceAvg * self.size)) * 100 if (self.priceAvg * self.size) != 0 else 0
                print(f"CLOSED LONG POSITION: PnL: {self.profit:.2f} ({pnl_percentage:.2f}%) | Total Strategy PnL: {total_profit_loss:.2f}")
                self.active = False
                self.size = 0.0
                createMarker('❌') # Mark position closed
            else: # self.size < quantity:
                # Reverse the position: close long and open short
                self.profit = (price - self.priceAvg) * self.size * self.leverage
                total_profit_loss += self.profit
                pnl_percentage = (self.profit / (self.priceAvg * self.size)) * 100 if (self.priceAvg * self.size) != 0 else 0
                print(f"REVERSED LONG TO SHORT: PnL: {self.profit:.2f} ({pnl_percentage:.2f}%) | Total Strategy PnL: {total_profit_loss:.2f}")
                remaining_quantity = quantity - self.size
                self.active = False # Close the current long position
                self.size = 0.0
                createMarker('🔄') # Mark position reversal
                openPosition(SHORT, price, remaining_quantity, leverage) # Open new short position
            return

        if self.type == SHORT and op_type == LONG:
            if self.size > quantity:
                # Reduce the short position
                self.size -= quantity
                createMarker('➖') # Mark position reduction
            elif self.size == quantity:
                # Close the short position
                self.profit = (self.priceAvg - price) * self.size * self.leverage
                total_profit_loss += self.profit
                pnl_percentage = (self.profit / (self.priceAvg * self.size)) * 100 if (self.priceAvg * self.size) != 0 else 0
                print(f"CLOSED SHORT POSITION: PnL: {self.profit:.2f} ({pnl_percentage:.2f}%) | Total Strategy PnL: {total_profit_loss:.2f}")
                self.active = False
                self.size = 0.0
                createMarker('❌') # Mark position closed
            else: # self.size < quantity:
                # Reverse the position: close short and open long
                self.profit = (self.priceAvg - price) * self.size * self.leverage
                total_profit_loss += self.profit
                pnl_percentage = (self.profit / (self.priceAvg * self.size)) * 100 if (self.priceAvg * self.size) != 0 else 0
                print(f"REVERSED SHORT TO LONG: PnL: {self.profit:.2f} ({pnl_percentage:.2f}%) | Total Strategy PnL: {total_profit_loss:.2f}")
                remaining_quantity = quantity - self.size
                self.active = False # Close the current short position
                self.size = 0.0
                createMarker('🔄') # Mark position reversal
                openPosition(LONG, price, remaining_quantity, leverage) # Open new long position
            return

    def close(self, price: float):
        """
        Closes the active position.

        Args:
            price (float): The price at which the position is closed.
        """
        if not self.active:
            return

        # To close a long position, issue a sell order of the same size
        if self.type == LONG:
            self.update(SHORT, price, self.size, self.leverage)
        # To close a short position, issue a buy order of the same size
        elif self.type == SHORT:
            self.update(LONG, price, self.size, self.leverage)

def openPosition(pos_type: int, price: float, quantity: float, leverage: int) -> position_c:
    """
    Creates and opens a new position.

    Args:
        pos_type (int): The type of position (LONG or SHORT).
        price (float): The entry price of the position.
        quantity (float): The size of the position.
        leverage (int): The leverage for the position.

    Returns:
        position_c: The newly created position object.
    """
    pos = position_c()
    pos.active = True
    pos.type = pos_type
    pos.size = quantity
    pos.leverage = leverage
    pos.priceAvg = price

    positions.append(pos) # Add the new position to the global list
    icon = '🔴' if pos.type == SHORT else '🟢'
    createMarker(icon) # Mark the opening of the position on the chart
    return pos

def getActivePosition() -> position_c:
    """
    Retrieves the currently active position.

    Returns:
        position_c: The active position object, or None if no position is active.
    """
    if len(positions) == 0:
        return None
    pos = positions[-1] # Get the last position added
    if not pos.active:
        return None
    return pos

def direction() -> int:
    """
    Returns the direction of the active position.

    Returns:
        int: 1 for LONG, -1 for SHORT, 0 if no position is active.
    """
    pos = getActivePosition()
    if pos is None or not pos.active:
        return 0
    return pos.type

def order(cmd: str, price: float, quantity: float, leverage: int = 1):
    """
    Executes a trading order (buy, sell, or close).

    Args:
        cmd (str): The command ('buy', 'sell', 'close').
        price (float): The price at which the order is executed.
        quantity (float): The quantity for the order.
        leverage (int, optional): The leverage. Defaults to 1.
    """
    if cmd is None:
        return

    cmd = cmd.lower()

    if cmd == 'buy':
        pos = getActivePosition()
        if pos is None: # No active position, open a new LONG
            openPosition(LONG, price, quantity, leverage)
        else: # Active position, update it
            pos.update(LONG, price, quantity, leverage)

    elif cmd == 'sell':
        pos = getActivePosition()
        if pos is None: # No active position, open a new SHORT
            openPosition(SHORT, price, quantity, leverage)
        else: # Active position, update it
            pos.update(SHORT, price, quantity, leverage)

    elif cmd == 'close':
        pos = getActivePosition()
        if pos is None or not pos.active:
            return
        pos.close(price) # Close the active position

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
        pos.close(realtimeCandle.close)

def get_total_profit_loss() -> float:
    """
    Returns the total accumulated profit or loss for the strategy.

    Returns:
        float: The total profit or loss.
    """
    return total_profit_loss
