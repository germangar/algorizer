
import numpy as np
from .nameseries import series_c
from .constants import c
from . import active


from dataclasses import dataclass
@dataclass
class pivot_c:
    index: int
    type: int
    price: float
    timestamp: int

class pivots_c:
    def __init__(self, min_range_pct: float = 5.0, reversal_pct: float = 30.0):
        # Configuration
        self.min_range_pct = min_range_pct
        self.reversal_pct = reversal_pct
        self.barindex = -1
        
        # State variables
        self.trend = c.LONG  # Start assuming uptrend
        self._current_trend_high_extrema = None
        self._current_trend_low_extrema = None
        self._current_trend_high_extrema_index = None
        self._current_trend_low_extrema_index = None
        self._last_confirmed_pivot_price = None
        self._last_confirmed_pivot_reversal_magnitude = None
        self.isNewPivot = False # a new pivot was created in the last update
        self.pivots:list[pivot_c] = []
        self.temp_pivot: pivot_c = None # Stores the potential pivot in progress
        self._current_reversal_percentage: float = 0.0 # Stores the current reversal percentage of the temp_pivot
        
    def process_candle(self, index: int, high: float, low: float)->bool:
        self.isNewPivot = False
        # Do NOT reset _current_reversal_percentage here. It's tied to the current WIP pivot.
        
        if self.barindex >= index:
            return False
        self.barindex = index

        # Initialize on first candle
        if self._current_trend_high_extrema is None:
            self._current_trend_high_extrema = high
            self._current_trend_low_extrema = low
            self._current_trend_high_extrema_index = index
            self._current_trend_low_extrema_index = index
            
            # Initialize temp_pivot for the first time
            if self.trend > 0: # Long trend implies potential high pivot
                self.temp_pivot = pivot_c(index=self._current_trend_high_extrema_index, type=c.PIVOT_HIGH, price=self._current_trend_high_extrema, timestamp=active.timeframe.timestampAtIndex(self._current_trend_high_extrema_index))
            else: # Short trend implies potential low pivot
                self.temp_pivot = pivot_c(index=self._current_trend_low_extrema_index, type=c.PIVOT_LOW, price=self._current_trend_low_extrema, timestamp=active.timeframe.timestampAtIndex(self._current_trend_low_extrema_index))
            self._current_reversal_percentage = 0.0 # It starts at 0 since there's no reversal yet
            return False
            
        if self.trend > 0: # Currently in an uptrend (looking for high pivot)
            # Track new high if we make one
            if high >= self._current_trend_high_extrema:
                self._current_trend_high_extrema = high
                self._current_trend_high_extrema_index = index
                self.temp_pivot = pivot_c(index=self._current_trend_high_extrema_index, type=c.PIVOT_HIGH, price=self._current_trend_high_extrema, timestamp=active.timeframe.timestampAtIndex(self._current_trend_high_extrema_index))
                self._current_reversal_percentage = 0.0 # Reset as new high sets a new "base" for potential reversal
                return False
            
            # Calculate current reversal if price drops from HH (this is the key calculation for the user's request)
            current_reversal = self._current_trend_high_extrema - low
            if self._current_trend_high_extrema != 0: # Avoid division by zero
                self._current_reversal_percentage = (current_reversal / self._current_trend_high_extrema) * 100
            else:
                self._current_reversal_percentage = 0.0 # Handle case where extrema is 0 (unlikely for prices, but good practice)

            # Check for potential reversal
            min_range_threshold = self._current_trend_high_extrema * (1 - self.min_range_pct * 0.01)
            
            if low < min_range_threshold:
                
                # If we have a previous pivot to compare to
                if self._last_confirmed_pivot_price is not None:
                    reversal_threshold = self._last_confirmed_pivot_reversal_magnitude * (self.reversal_pct * 0.01)
                    if current_reversal >= reversal_threshold:
                        # Confirmed down pivot
                        self.addPivot(self._current_trend_high_extrema_index, c.PIVOT_HIGH, self._current_trend_high_extrema) # Add the high pivot
                        self.trend = c.SHORT
                        self._last_confirmed_pivot_price = self._current_trend_high_extrema
                        self._last_confirmed_pivot_reversal_magnitude = current_reversal
                        self._current_trend_low_extrema = low
                        self._current_trend_low_extrema_index = index
                        self.temp_pivot = pivot_c(index=self._current_trend_low_extrema_index, type=c.PIVOT_LOW, price=self._current_trend_low_extrema, timestamp=active.timeframe.timestampAtIndex(self._current_trend_low_extrema_index))
                        self._current_reversal_percentage = 0.0 # Reset for new trend
                        return True
                else:
                    # First pivot, only use min_range
                    self.addPivot( self._current_trend_high_extrema_index, c.PIVOT_HIGH, self._current_trend_high_extrema) # Add the high pivot
                    self.trend = c.SHORT
                    self._last_confirmed_pivot_price = self._current_trend_high_extrema
                    self._last_confirmed_pivot_reversal_magnitude = current_reversal
                    self._current_trend_low_extrema = low
                    self._current_trend_low_extrema_index = index
                    self.temp_pivot = pivot_c(index=self._current_trend_low_extrema_index, type=c.PIVOT_LOW, price=self._current_trend_low_extrema, timestamp=active.timeframe.timestampAtIndex(self._current_trend_low_extrema_index))
                    self._current_reversal_percentage = 0.0 # Reset for new trend
                    return True
                    
        else:  # Currently in a downtrend (looking for low pivot)
            # Track new low if we make one
            if low <= self._current_trend_low_extrema:
                self._current_trend_low_extrema = low
                self._current_trend_low_extrema_index = index
                self.temp_pivot = pivot_c(index=self._current_trend_low_extrema_index, type=c.PIVOT_LOW, price=self._current_trend_low_extrema, timestamp=active.timeframe.timestampAtIndex(self._current_trend_low_extrema_index))
                self._current_reversal_percentage = 0.0 # Reset reversal percentage as a new low is made
                return False
                
            # Calculate current reversal if price rises from LL (this is the key calculation for the user's request)
            current_reversal = high - self._current_trend_low_extrema
            if self._current_trend_low_extrema != 0: # Avoid division by zero
                self._current_reversal_percentage = (current_reversal / self._current_trend_low_extrema) * 100
            else:
                self._current_reversal_percentage = 0.0 # Handle case where extrema is 0 (unlikely for prices, but good practice)

            # Check for potential reversal
            min_range_threshold = self._current_trend_low_extrema * (1 + self.min_range_pct * 0.01)
            
            if high > min_range_threshold:
                
                # If we have a previous pivot to compare to
                if self._last_confirmed_pivot_price is not None:
                    reversal_threshold = self._last_confirmed_pivot_reversal_magnitude * (self.reversal_pct * 0.01)
                    if current_reversal >= reversal_threshold:
                        # Confirmed up pivot
                        self.addPivot( self._current_trend_low_extrema_index, c.PIVOT_LOW, self._current_trend_low_extrema) # Add the low pivot
                        self.trend = c.LONG
                        self._last_confirmed_pivot_price = self._current_trend_low_extrema
                        self._last_confirmed_pivot_reversal_magnitude = current_reversal
                        self._current_trend_high_extrema = high
                        self._current_trend_high_extrema_index = index
                        self.temp_pivot = pivot_c(index=self._current_trend_high_extrema_index, type=c.PIVOT_HIGH, price=self._current_trend_high_extrema, timestamp=active.timeframe.timestampAtIndex(self._current_trend_high_extrema_index))
                        self._current_reversal_percentage = 0.0 # Reset for new trend
                        return True
                else:
                    # First pivot, only use min_range
                    self.addPivot( self._current_trend_low_extrema_index, c.PIVOT_LOW, self._current_trend_low_extrema) # Add the low pivot
                    self.trend = c.LONG
                    self._last_confirmed_pivot_price = self._current_trend_low_extrema
                    self._last_confirmed_pivot_reversal_magnitude = current_reversal
                    self._current_trend_high_extrema = high
                    self._current_trend_high_extrema_index = index
                    self.temp_pivot = pivot_c(index=self._current_trend_high_extrema_index, type=c.PIVOT_HIGH, price=self._current_trend_high_extrema, timestamp=active.timeframe.timestampAtIndex(self._current_trend_high_extrema_index))
                    self._current_reversal_percentage = 0.0 # Reset for new trend
                    return True
        
        return False

    def addPivot(self, index, type, price):
        pivot = pivot_c(
                index=index,
                type=type,
                price=price,
                timestamp=active.timeframe.timestampAtIndex(index)
                )
        if len(self.pivots) >= 500:
            self.pivots = self.pivots[1:] + [pivot]  # Slicing out the oldest element
        else:
            self.pivots.append(pivot)
        self.isNewPivot = True # Set to True when a pivot is successfully added


    def getLast(self, type:int = None, since:int = None)->pivot_c|None:
        if since is None:since = active.barindex
        for pivot in reversed(self.pivots):
            if pivot.index >= since:
                continue
            if type is not None and type != pivot.type:
                continue
            return pivot
        return None
    
    def update(self, high: series_c, low: series_c):
        if( not active.timeframe.jumpstart ):
            self.isNewPivot = self.process_candle(active.barindex, high[active.barindex], low[active.barindex])