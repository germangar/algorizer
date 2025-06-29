
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
    Represents the overall trading strategy, managing positions and global statistics.
    """
    def __init__(self, initial_liquidity: float = 10000.0, verbose: bool = False, order_size: float = 100.0, max_position_size: float = 100.0, hedged: bool = False, currency_mode: str = 'USD', leverage_long: float = 1.0, leverage_short: float = 1.0):
        self.positions = []   # List to hold all positions (both active and closed, Long and Short)
        self.total_profit_loss = 0.0 # Global variable to keep track of the total profit/loss for the entire strategy
        self.initial_liquidity = initial_liquidity # Starting capital for the strategy
        self.total_winning_positions = 0 # Counter for winning closed positions
        self.total_losing_positions = 0   # Counter for losing closed positions
        
        # NEW: Counters for long and short winning/total positions
        self.total_winning_long_positions = 0
        self.total_losing_long_positions = 0
        self.total_long_positions = 0

        self.total_winning_short_positions = 0
        self.total_losing_short_positions = 0
        self.total_short_positions = 0

        self.verbose = verbose # Controls whether warning prints are displayed
        self.order_size = order_size # Default quantity to use when none is provided (USD or BASE units depending on currency_mode)
        self.max_position_size = max_position_size # Maximum allowed total size for any single position (USD or BASE units)
        self.hedged = hedged # Controls whether multiple positions can be open simultaneously (True) or only one (False)
        self.currency_mode = currency_mode.upper() # NEW: 'USD' or 'BASE'
        self.first_order_timestamp = None # NEW: To track the very first order timestamp in the strategy
        self.leverage_long = leverage_long   # Default leverage for LONG trades
        self.leverage_short = leverage_short # Default leverage for SHORT trades

        # Validate currency_mode
        if self.currency_mode not in ['USD', 'BASE']:
            raise ValueError(f"Invalid currency_mode: {currency_mode}. Must be 'USD' or 'BASE'.")

        # Validate parameters based on currency_mode
        if self.currency_mode == 'USD' and self.max_position_size > self.initial_liquidity:
            # If max_position_size is meant as the *capital invested* in a single position, it should not exceed initial_liquidity.
            # If it's a notional size with leverage, it can be larger than initial_liquidity.
            # Given `active_capital_invested` is `price * size * leverage`, `max_position_size` is a limit on this `active_capital_invested`.
            # So, `max_position_size` as USD limit cannot be greater than `initial_liquidity` if it's supposed to represent a portion of the *account* capital for a single position.
            # If it's just a "target notional size", it can be anything.
            # Let's keep the user's intent: max_position_size acts as a capital limit per single position.
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
        Creates and opens a new position, associated with this strategy instance.
        Quantity must always be in base units.
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
        # Use specific marker for opening a position
        if pos_type == c.LONG:
            createMarker('🟢', location='below', shape='arrow_up', color='#00FF00') # Green arrow up for new LONG
        else: # SHORT
            createMarker('🔴', location='above', shape='arrow_down', color='#FF0000') # Red arrow down for new SHORT
        return pos

    def get_active_position(self, pos_type: int = None) -> 'position_c':
        """
        Retrieves the currently active position of a specific type (c.LONG or c.SHORT).
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
        pos = self.get_active_position() # This will return either a c.LONG or c.SHORT active position if one exists
        if pos is None:
            return 0
        return pos.type

    def execute_order(self, cmd: str, target_position_type: int, quantity_in_base_units_input: float, leverage: int = 1):
        """
        Executes a trading order. All internal calculations use base units.
        The price is retrieved from getRealtimeCandle().close internally.

        If hedged=True: Allows simultaneous c.LONG and c.SHORT positions.
        If hedged=False (One-Way Mode): Only one position (c.LONG or c.SHORT) can be active at a time.
            An oversized opposing order will close the current position and open a new one.

        Args:
            cmd (str): The command ('buy' or 'sell').
            target_position_type (int): Specifies which position (c.LONG or c.SHORT) this order targets.
            quantity_in_base_units_input (float): The quantity for the order, always in base currency units.
                                                This quantity might be clamped further inside based on strategy rules.
            leverage (int, optional): The leverage. Defaults to 1.
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

            # Calculate potential USD cost/value of the incoming order, considering leverage
            # This is the cost basis for the amount of base units in this order.
            # Using current_price * leverage for cost of 1 base unit for the order.
            cost_per_base_unit_with_leverage = current_price * leverage
            if cost_per_base_unit_with_leverage < EPSILON: # Avoid division by zero
                cost_per_base_unit_with_leverage = EPSILON # Set to a small non-zero value

            # Clamp incoming order quantity based on max_position_size (USD capital limit)
            if current_active_pos is None: # No active position, this is an opening order
                # The total capital for this initial order must not exceed max_position_size (USD)
                max_base_units_from_usd_capital_limit = self.max_position_size / cost_per_base_unit_with_leverage
                
                # Clamp the incoming order quantity and round to precision
                actual_quantity_to_process_base_units = round_to_tick_size(min(actual_quantity_to_process_base_units, max_base_units_from_usd_capital_limit), getPrecision())

                if actual_quantity_to_process_base_units < EPSILON:
                    if self.verbose or not isInitializing():
                        print(f"Warning: Calculated initial order quantity ({actual_quantity_to_process_base_units:.6f} base units) is effectively zero after USD capital clamping and precision rounding. No position opened.")
                    return # Exit if clamped to zero

            elif order_direction == current_active_pos.type: # Increasing an existing position
                # Calculate remaining USD capital capacity based on active_capital_invested
                remaining_usd_capital_capacity = self.max_position_size - current_active_pos.active_capital_invested

                if remaining_usd_capital_capacity <= EPSILON:
                    if self.verbose or not isInitializing():
                        print(f"Warning: Max USD position size ({self.max_position_size}) based on capital invested reached. No new {cmd} order placed.")
                    return # Exit if no more USD capacity

                # Calculate max base units that can be bought with remaining USD capacity
                max_base_units_from_usd_capital_capacity = remaining_usd_capital_capacity / cost_per_base_unit_with_leverage
                
                # Clamp the incoming order quantity and round to precision
                actual_quantity_to_process_base_units = round_to_tick_size(min(actual_quantity_to_process_base_units, max_base_units_from_usd_capital_capacity), getPrecision())

                if actual_quantity_to_process_base_units < EPSILON:
                    if self.verbose or not isInitializing():
                        print(f"Warning: Calculated order quantity ({actual_quantity_to_process_base_units:.6f} base units) is effectively zero after USD capital clamping and precision rounding. No order placed.")
                    return # Exit if clamped to zero

            # Note: For opposing orders (partial close/reversal), no capital clamping is applied here.
            # They are meant to reduce/close the position, not increase capital exposure.
            # `actual_quantity_to_process_base_units` remains `quantity_in_base_units_input` in this case for opposing orders.


        # --- General logic that applies to both modes after `actual_quantity_to_process_base_units` is determined ---

        if self.hedged:
            active_target_pos = self.get_active_position(target_position_type)

            if active_target_pos is None:
                # No active position of the target_position_type, so open a new one
                if order_direction == target_position_type:
                    # In USD mode, `actual_quantity_to_process_base_units` already clamped.
                    # In BASE mode, clamp by `self.max_position_size` and round
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
                    
            else: # An active position of the target_position_type exists
                if order_direction == active_target_pos.type:
                    # Order direction matches existing position type: increase position
                    # `actual_quantity_to_process_base_units` is already clamped by USD capital or initial quantity
                    # For BASE mode, it needs to be clamped against max_position_size (which is self.max_position_size) and round
                    clamped_quantity_final = actual_quantity_to_process_base_units
                    if self.currency_mode == 'BASE':
                        available_space_base_units = self.max_position_size - active_target_pos.size
                        clamped_quantity_final = round_to_tick_size(min(actual_quantity_to_process_base_units, available_space_base_units), getPrecision())

                    if clamped_quantity_final > EPSILON:
                        active_target_pos.update(order_direction, current_price, clamped_quantity_final, leverage)
                        affected_pos = active_target_pos # This position was affected
                    elif self.verbose or not isInitializing():
                        print(f"Warning: Attempted to increase {active_target_pos.type} position but calculated final quantity ({clamped_quantity_final:.2f} base units) is effectively zero. No change to position.")
                else:
                    # Order direction opposes existing position type: reduce or close
                    if actual_quantity_to_process_base_units < active_target_pos.size - EPSILON:
                        # Partial close: reduce the existing position
                        active_target_pos.update(order_direction, current_price, actual_quantity_to_process_base_units, leverage)
                        affected_pos = active_target_pos # This position was affected
                    elif actual_quantity_to_process_base_units >= active_target_pos.size - EPSILON:
                        # Full close: close the existing position (or oversized order that just closes)
                        
                        # Store current position reference before it gets closed for profit check
                        closing_position_reference = active_target_pos
                        affected_pos = closing_position_reference # This position will be affected
                        
                        # Get size BEFORE calling close for printing warnings
                        pos_size_to_close = closing_position_reference.size 

                        closing_position_reference.close(current_price) # This calculates and sets closing_position_reference.profit

                        # Determine marker based on the position's profit after it's closed
                        marker_text = 'W' if closing_position_reference.profit > EPSILON else ('L' if closing_position_reference.profit < -EPSILON else 'E') # 'E' for even/break-even
                        
                        # Determine marker color based on the TYPE of the position being closed
                        marker_color = '#00CC00' if closing_position_reference.type == c.LONG else '#FF0000' # Green for c.LONG, Red for c.SHORT

                        createMarker(marker_text, location='above', shape='square', color=marker_color)
                        
                        if actual_quantity_to_process_base_units > pos_size_to_close + EPSILON: 
                             if self.verbose or not isInitializing():
                                print(f"Warning: Attempted to close a {'LONG' if closing_position_reference.type == c.LONG else 'SHORT'} position with an oversized {cmd} order.")
                                print(f"Position was fully closed. Remaining quantity ({actual_quantity_to_process_base_units - pos_size_to_close:.2f} base units) was not used to open a new position.")

        else: # --- ONEWAY MODE LOGIC (Only one position at a time) ---
            current_overall_active_pos = self.get_active_position() # Get THE active position, if any type

            if current_overall_active_pos is None:
                # No active position, open a new one (only if order direction matches target type for consistency)
                if order_direction == target_position_type:
                    # `actual_quantity_to_process_base_units` already clamped by USD capital or initial quantity
                    # For BASE mode, it needs to be clamped against max_position_size and round
                    clamped_quantity_final = actual_quantity_to_process_base_units
                    if self.currency_mode == 'BASE':
                        clamped_quantity_final = round_to_tick_size(min(actual_quantity_to_process_base_units, self.max_position_size), getPrecision())
                        
                    if clamped_quantity_final > EPSILON:
                        affected_pos = self.open_position(target_position_type, current_price, clamped_quantity_final, leverage)
                    elif self.verbose or not isInitializing():
                        print(f"Warning: Attempted to open new position, but calculated final quantity ({clamped_quantity_final:.2f} base units) is effectively zero. No position opened.")
                else:
                    if self.verbose or not isInitializing():
                        print(f"Warning: In one-way mode, no active position. Cannot {cmd} to target {target_position_type} type when opening a new position (order direction mismatch).")
            
            elif order_direction == current_overall_active_pos.type:
                # Order direction matches current overall position: increase existing position
                # `actual_quantity_to_process_base_units` already clamped by USD capital or initial quantity
                # For BASE mode, it needs to be clamped against max_position_size and round
                clamped_quantity_final = actual_quantity_to_process_base_units
                if self.currency_mode == 'BASE':
                    available_space_base_units = self.max_position_size - current_overall_active_pos.size
                    clamped_quantity_final = round_to_tick_size(min(actual_quantity_to_process_base_units, available_space_base_units), getPrecision())

                if clamped_quantity_final > EPSILON:
                    current_overall_active_pos.update(order_direction, current_price, clamped_quantity_final, leverage)
                    affected_pos = current_overall_active_pos # This position was affected
                elif self.verbose or not isInitializing():
                    print(f"Warning: Attempted to increase {current_overall_active_pos.type} position but calculated final quantity ({clamped_quantity_final:.2f} base units) is effectively zero. No change to position.")

            elif order_direction != current_overall_active_pos.type:
                # Order direction opposes current overall position: close/reverse
                
                # Check if the order is large enough to close the existing position
                if actual_quantity_to_process_base_units >= current_overall_active_pos.size - EPSILON:
                    # Reversal: Close existing position and open new one with remaining quantity
                    pos_size_to_close = current_overall_active_pos.size
                    
                    # Store current position reference before it gets closed for profit check
                    closing_position_reference = current_overall_active_pos 

                    closing_position_reference.close(current_price) # This calculates and sets closing_position_reference.profit
                    # The closing part means this position is affected
                    affected_pos = closing_position_reference

                    # Determine marker based on the position's profit after it's closed
                    marker_text = 'W' if closing_position_reference.profit > EPSILON else ('L' if closing_position_reference.profit < -EPSILON else 'E')
                    
                    # Determine marker color based on the TYPE of the position being closed
                    marker_color = '#00CC00' if closing_position_reference.type == c.LONG else '#FF0000' # Green for c.LONG, Red for c.SHORT

                    createMarker(marker_text, location='above', shape='square', color=marker_color)
                    
                    # Remaining quantity after closing the old position
                    remaining_quantity_base_units_for_new_pos = actual_quantity_to_process_base_units - pos_size_to_close
                    
                    if remaining_quantity_base_units_for_new_pos > EPSILON: 
                        # Open a new position with the remaining quantity, respecting max_position_size as a USD capital limit
                        clamped_new_pos_quantity_base_units = remaining_quantity_base_units_for_new_pos
                        if self.currency_mode == 'USD':
                             # Remaining USD capacity for a NEW position is `max_position_size`
                            cost_per_base_unit_with_leverage = current_price * leverage
                            if cost_per_base_unit_with_leverage < EPSILON:
                                cost_per_base_unit_with_leverage = EPSILON # Avoid division by zero
                            max_base_units_for_new_pos_from_usd_limit = self.max_position_size / cost_per_base_unit_with_leverage
                            clamped_new_pos_quantity_base_units = round_to_tick_size(min(remaining_quantity_base_units_for_new_pos, max_base_units_for_new_pos_from_usd_limit), getPrecision())
                        elif self.currency_mode == 'BASE':
                            clamped_new_pos_quantity_base_units = round_to_tick_size(min(remaining_quantity_base_units_for_new_pos, self.max_position_size), getPrecision())

                        if clamped_new_pos_quantity_base_units > EPSILON:
                            # This open_position call will trigger another execute_order, which will trigger broker_event.
                            # So, we should *not* call broker_event here again for the opening part.
                            # We just need to ensure the `affected_pos` points to the *new* position if one is opened.
                            new_pos_obj = self.open_position(order_direction, current_price, clamped_new_pos_quantity_base_units, leverage)
                            affected_pos = new_pos_obj # Set affected_pos to the newly opened position
                            if self.verbose or not isInitializing():
                                print(f"Position reversed: Closed {'LONG' if closing_position_reference.type == c.LONG else 'SHORT'} position and opened new {order_direction} position with {clamped_new_pos_quantity_base_units:.2f} base units.")
                        elif self.verbose or not isInitializing():
                            print(f"Warning: Position closed. Remaining quantity ({remaining_quantity_base_units_for_new_pos:.2f} base units) clamped to 0 because it exceeds max_position_size or is too small. No new position opened.")
                    elif self.verbose or not isInitializing():
                        print(f"Position of type {'LONG' if closing_position_reference.type == c.LONG else 'SHORT'} was fully closed by an exact or slightly oversized order. No new position opened.")
                else:
                    # Partial close of the existing position
                    current_overall_active_pos.update(order_direction, current_price, actual_quantity_to_process_base_units, leverage)
                    affected_pos = current_overall_active_pos # This position was affected
                    if self.verbose or not isInitializing():
                        print(f"Partial close: Reduced {current_overall_active_pos.type} position by {actual_quantity_to_process_base_units:.2f} base units.")

        # Call the broker_event after the order is processed and position state is updated
        if not isInitializing() and affected_pos is not None:
            final_position_type_for_broker_event = 0
            final_position_size_base_for_broker_event = 0.0
            final_position_size_dollars_for_broker_event = 0.0
            final_position_collateral_dollars_for_broker_event = 0.0 # NEW

            if affected_pos.active and affected_pos.size > EPSILON:
                final_position_type_for_broker_event = affected_pos.type
                final_position_size_base_for_broker_event = affected_pos.size * affected_pos.type # Signed
                final_position_size_dollars_for_broker_event = affected_pos.active_capital_invested * affected_pos.type # Signed (notional value including leverage)
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


    def close_position(self, pos_type: int = None): # pos_type is now optional
        """
        Closes a specific active position (c.LONG or c.SHORT) at the current realtime candle's close price.
        If hedged=False and pos_type is None, it closes the single active position, if any.

        Args:
            pos_type (int, optional): The type of position to close (c.LONG or c.SHORT).
                                      If hedged is False and this is None, closes any active position.
        """
        pos_to_close = None
        if self.hedged:
            if pos_type is None:
                if self.verbose or not isInitializing():
                    print("Warning: In hedged mode, 'close' requires a 'pos_type' (c.LONG or c.SHORT) to specify which position to close.")
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
                elif self.verbose or not isInitializing():
                    print(f"Warning: In one-way mode, attempted to close a { 'LONG' if pos_type == c.LONG else 'SHORT' } position, but the active position is {'LONG' if current_active.type == c.LONG else 'SHORT' if current_active.type == c.SHORT else 'None'}. No action taken.")
                    return


        if pos_to_close is None or not pos_to_close.active or pos_to_close.size < EPSILON:
            if self.verbose or not isInitializing():
                # Adjusted message for clarity based on whether a type was requested
                type_str = f" { 'LONG' if pos_type == c.LONG else 'SHORT' }" if pos_type is not None else ""
                print(f"No active{type_str} position to close.")
            return
        
        # Execute order to close the position
        # Determine the closing order type based on the position's type
        close_cmd = 'sell' if pos_to_close.type == c.LONG else 'buy'
        # Call execute_order with quantity_in_base_units (pos_to_close.size is already in base units)
        # Use the position's effective leverage for closing order
        self.execute_order(close_cmd, pos_to_close.type, pos_to_close.size, pos_to_close.leverage) 

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
            
            # Added active_capital_invested to print
            print(f"\nPosition #{i+1} (Status: {status}, Type: {position_type_str}, Current Size: {pos.size:.2f} base units, Avg Price: {pos.priceAvg:.2f}, Max Size Held: {pos.max_size_held:.2f} base units, Active Capital Invested: {pos.active_capital_invested:.2f} USD, Position Leverage: {pos.leverage})") 
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
        print(f"{'PnL %':<12} {'Total PnL':<12} {'Trades':<8} {'Wins':<8} {'Losses':<8} {'Win Rate %':<12} {'Long Win %':<12} {'Short Win %':<12} {'Avg+ PnL':<12} {'Avg- PnL':<12} {'Acct PnL %':<12}")
        # Print values
        print(f"{pnl_percentage_vs_max_pos_size:<12.2f} {pnl_quantity:<12.2f} {total_closed_positions:<8} {profitable_trades:<8} {losing_trades:<8} {percentage_profitable_trades:<12.2f} {long_win_ratio:<12.2f} {short_win_ratio:<12.2f} {avg_winning_pnl:<12.2f} {avg_losing_pnl:<12.2f} {pnl_percentage_vs_liquidity:<12.2f}")
        
        print("------------------------------")

    def print_pnl_by_period(self):
        """
        Calculates and prints the realized PnL of closed positions by month, quarter, and year.
        Includes unrealized PnL from active positions in the final period.
        The granularity of the output depends on the total duration of the strategy's activity.
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
    Represents an individual trading position (long or short).
    Handles opening, updating (increasing/reducing), and closing orders.
    Keeps a history of all orders made within this position.
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
        self.active_capital_invested = 0.0 # NEW: Tracks the USD capital (cost basis) currently tied up in the open position
        self.close_timestamp = None # NEW: Store timestamp when position is closed
        self.liquidation_price = 0.0

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

        # Calculate active_capital_invested based on the current (recalculated) state
        # This uses the position's overall leverage (self.leverage)
        if abs(self.size) > EPSILON: # Only if there's an active position size
            self.active_capital_invested = self.priceAvg * self.size * self.leverage
        else:
            self.active_capital_invested = 0.0

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
        Calculates and returns the average entry price (in quote currency) based on the current order history.
        This is a public method for auditing/display.
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
        # This call should result in self.size == 0.0 if the closing order perfectly nets out.
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
            # Sum only entry capital (orders where pnl_quantity is approx 0) to get total invested capital.
            # This is the sum of (price * quantity * leverage) for opening/increasing orders.
            # For simplicity, if total_entry_capital_for_position is 0, the percentage is 0.
            total_entry_capital_for_position = 0.0
            for order_data in self.order_history:
                # Assuming entry orders have pnl_quantity close to zero
                if order_data['pnl_quantity'] >= -EPSILON and order_data['pnl_quantity'] <= EPSILON:
                    # Use the leverage stored with the order
                    order_leverage = order_data.get('leverage', 1.0) # Default to 1.0 if not found for old data
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
            self.active_capital_invested = 0.0 # Reset capital invested when position is closed
            self.close_timestamp = getRealtimeCandle().timestamp # NEW: Record closure timestamp

            # Print PnL to console
            if self.strategy_instance.verbose or not isInitializing():
                print(f"CLOSED POSITION ({'LONG' if previous_position_type == c.LONG else 'SHORT'}): PnL: {self.profit:.2f} (quote currency) | PnL %: {self.realized_pnl_percentage:.2f}% | Total Strategy PnL: {self.strategy_instance.total_profit_loss:.2f} (quote currency)")

    def get_unrealized_pnl_quantity(self) -> float:
        """
        Calculates and returns the current unrealized Profit and Loss in quantity (quote currency)
        for this active position.
        Returns 0.0 if the position is not active.
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
        Calculates and returns the current unrealized Profit and Loss as a percentage
        of the capital invested for this active position.
        Returns 0.0 if the position is not active or if the average entry price is zero.
        """
        unrealized_pnl_q = self.get_unrealized_pnl_quantity()
        if abs(unrealized_pnl_q) < EPSILON and (not self.active or self.size < EPSILON):
            return 0.0

        # Capital involved in the active position based on entry price and quantity (in quote currency)
        # This is self.active_capital_invested
        capital_involved = self.active_capital_invested
        if abs(capital_involved) < EPSILON: # Avoid division by zero
            return 0.0
        
        return (unrealized_pnl_q / abs(capital_involved)) * 100

    def get_order_by_direction(self, order_direction: int, older_than_bar_index: int = None) -> dict:
        """
        Retrieves the last order of a specific direction (c.LONG or c.SHORT).
        If older_than_bar_index is provided, it finds the first order of that direction
        that occurred *before* the given bar index (i.e., closest older order).

        Args:
            order_direction (int): The type of order to find (c.LONG or c.SHORT).
            older_than_bar_index (int, optional): If provided, returns the first order
                                                of the specified direction that occurred
                                                before this bar index. Defaults to None,
                                                which returns the very last order of that direction.

        Returns:
            dict: The order dictionary if found, otherwise None.
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


# Global instance of the strategy.
# Initialized with currency_mode='USD' as requested.
strategy = strategy_c(hedged=False, currency_mode='USD') 

# The following global functions will now call methods on the 'strategy' instance.
# This maintains compatibility with existing calls from other modules (e.g., algorizer.py).
def openPosition(pos_type: int, price: float, quantity: float, leverage: int) -> 'position_c':
    """
    Opens a new position. Quantity must be in base units.
    """
    return strategy.open_position(pos_type, price, quantity, leverage)

def getActivePosition(pos_type: int = None) -> 'position_c':
    return strategy.get_active_position(pos_type)

def direction() -> int:
    return strategy.get_direction()

def order(cmd: str, target_position_type: int, quantity: float = None, leverage: int = None): # leverage parameter is now optional
    """
    Places a trading order.
    The 'quantity' parameter's interpretation depends on strategy.currency_mode:
    - If currency_mode is 'USD', quantity is interpreted as a notional USD amount.
    - If currency_mode is 'BASE', quantity is interpreted as base currency units.
    """
    actual_quantity_base_units = 0.0 # Initialize here to ensure scope
    current_price = getRealtimeCandle().close # Get price here for conversion if needed

    # Determine leverage to use for this order
    selected_leverage = leverage
    if selected_leverage is None:
        if target_position_type == c.LONG:
            selected_leverage = strategy.leverage_long
        elif target_position_type == c.SHORT:
            selected_leverage = strategy.leverage_short
        else:
            # Fallback if target_position_type is neither LONG nor SHORT
            # This case ideally shouldn't happen with proper usage of `order`
            selected_leverage = 1.0 
            if strategy.verbose or not isInitializing():
                print(f"Warning: Could not determine default leverage for target_position_type {target_position_type}. Using default leverage 1.0.")

    if strategy.currency_mode == 'USD':
        if quantity is None:
            # Use default order_size which is in USD
            # Ensure price is not zero to avoid division by zero
            if current_price < EPSILON: 
                if strategy.verbose or not isInitializing():
                    print(f"Warning: Current price ({current_price:.2f}) is too low for USD quantity conversion. Order not placed.")
                return
            actual_quantity_base_units = strategy.order_size / current_price
        else:
            # Convert provided USD quantity to base units
            if current_price < EPSILON:
                if strategy.verbose or not isInitializing():
                    print(f"Warning: Current price ({current_price:.2f}) is too low for USD quantity conversion. Order not placed.")
                return
            actual_quantity_base_units = quantity / current_price

        # Safety check for extremely small quantities (if USD amount is too small for current price)
        # Apply precision rounding here as well, since `actual_quantity_base_units` was just calculated
        actual_quantity_base_units = round_to_tick_size(actual_quantity_base_units, getPrecision())

        if actual_quantity_base_units < EPSILON:
            if strategy.verbose or not isInitializing():
                print(f"Warning: Calculated order quantity ({actual_quantity_base_units:.6f} base units) is effectively zero based on current price ({current_price:.2f}). Order not placed.")
            return
    else: # strategy.currency_mode == 'BASE'
        if quantity is None:
            actual_quantity_base_units = strategy.order_size
        else:
            actual_quantity_base_units = quantity
        
        # Ensure base quantity is rounded to precision before executing order
        actual_quantity_base_units = round_to_tick_size(actual_quantity_base_units, getPrecision())


    strategy.execute_order(cmd, target_position_type, actual_quantity_base_units, selected_leverage) 

def close(pos_type: int = None):
    strategy.close_position(pos_type)

def get_total_profit_loss() -> float:
    return strategy.get_total_profit_loss()

def print_strategy_stats(): # This will become print_detailed_stats
    strategy.print_detailed_stats()

def print_summary_stats(): # New function
    strategy.print_summary_stats()

def print_pnl_by_period_summary(): # New global function for PnL by period
    strategy.print_pnl_by_period()

def priceUpdate( candle:candle_c, realtime:bool ):
    pass
