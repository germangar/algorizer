
import pandas as pd
import pandas_ta as pt
import numpy as np
import time
import active
import tools


# Dynamically set __all__ to include all names that don't start with '_' and are not in _exclude
_exclude = ['active']
__all__ = [name for name in globals() if not (name.startswith('_') or name in _exclude)]




# #
# # GENERATED SERIES : These are series of values that are calculated always using the same formula
# #

def _generatedseries_calculate_highest(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    values = series.to_numpy()
    length = len(values)
    result = np.full(length, np.nan)
    
    if length >= period:
        # Create strided array view for rolling windows
        strides = np.lib.stride_tricks.sliding_window_view(values, period)
        
        # Calculate max for each window
        result[period-1:] = np.max(strides, axis=1)

    return pd.Series(result, index=series.index)

def _generatedseries_calculate_lowest(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    values = series.to_numpy()
    length = len(values)
    result = np.full(length, np.nan)
    
    if length >= period:
        # Create strided array view for rolling windows
        strides = np.lib.stride_tricks.sliding_window_view(values, period)
        
        # Calculate min for each window
        result[period-1:] = np.min(strides, axis=1)

    return pd.Series(result, index=series.index)

def _generatedseries_calculate_highestbars(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    values = series.to_numpy()
    length = len(values)
    result = np.full(length, np.nan)
    
    # Only process when we have enough data
    if length >= period:
        # Create strided array view for rolling windows
        strides = np.lib.stride_tricks.sliding_window_view(values, period)
        
        # Calculate argmax for each window
        indices = np.argmax(strides, axis=1)
        
        # Convert to offset from current position
        result[period-1:] = (period - 1) - indices

    return pd.Series(result, index=series.index)

def _generatedseries_calculate_lowestbars(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    values = series.to_numpy()
    length = len(values)
    result = np.full(length, np.nan)
    
    # Only process when we have enough data
    if length >= period:
        # Create strided array view for rolling windows
        strides = np.lib.stride_tricks.sliding_window_view(values, period)
        
        # Calculate argmin for each window
        indices = np.argmin(strides, axis=1)
        
        # Convert to offset from current position
        result[period-1:] = (period - 1) - indices

    return pd.Series(result, index=series.index)

def _generatedseries_calculate_sma(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.sma( series, period )

def _generatedseries_calculate_ema(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.ema( series, period )

def _generatedseries_calculate_dema(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.dema( series, period )

def _generatedseries_calculate_linreg(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.linreg( series, period )

def _generatedseries_calculate_rma(series: pd.Series, length: int, df:pd.DataFrame, param=None) -> pd.Series:
    return series.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

def _generatedseries_calculate_stdev(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.stdev( series, period )

def _generatedseries_calculate_bias(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.bias( series, period )

def _generatedseries_calculate_cfo(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.cfo( series, period )

def _generatedseries_calculate_fwma(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.fwma( series, period )

def _generatedseries_calculate_dev(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    if 1:
        return pt.mad( series, period )
    else:
        # Calculate the average deviation over a given rolling window in a pandas Series.
        # Initialize a list to hold the deviation values
        deviations = [pd.NA] * (period - 1)  # Start with NA values for the initial periods
        # Iterate over each rolling window
        for i in range(period - 1, len(series)):
            rolwindow = series[i - period + 1:i + 1]
            mean = rolwindow.mean()
            deviation = (rolwindow - mean).abs().sum() / period
            deviations.append(deviation)
        return pd.Series(deviations, index=series.index).dropna()

def _generatedseries_calculate_williams_r(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    """
        Calculate Williams %R for a given series using OHLC data from df over a period.

        Args:
        - series: pd.Series, typically a placeholder, but required for compatibility with generatedSeries_c.
        - period: int, the period/window for the Williams %R calculation.

        Returns:
        - pd.Series, the calculated Williams %R values.
        """
    if 1:
        return pt.willr( df['high'], df['low'], df['close'], length=period )
    else:
        # Ensure the DataFrame has the required columns
        # if not all(col in df.columns for col in ['high', 'low', 'close']):
        #     raise ValueError("The global DataFrame must contain 'high', 'low', and 'close' columns")

        if len(df) < period:
            return pd.Series([pd.NA] * len(df), index=df.index)  # Not enough data to calculate Williams %R

        # Initialize a list to hold the Williams %R values
        williams_r_values = [pd.NA] * (period - 1)  # NA for the initial period

        # Calculate Williams %R for each rolling window
        for i in range(period - 1, len(df)):
            highest_high = df['high'].iloc[i - period + 1:i + 1].max()
            lowest_low = df['low'].iloc[i - period + 1:i + 1].min()
            current_close = df['close'].iloc[i]

            if highest_high == lowest_low:  # Prevent division by zero
                williams_r_values.append(pd.NA)
            else:
                williams_r = (highest_high - current_close) / (highest_high - lowest_low) * -100
                williams_r_values.append(williams_r)

        return pd.Series(williams_r_values, index=df.index)

def _generatedseries_calculate_rsi(series, period, df:pd.DataFrame, param=None) -> pd.Series:
    if 1:
        # Convert to numpy array for faster operations
        values = series.to_numpy()
        deltas = np.diff(values, prepend=np.nan)
        length = len(values)
        result = np.full(length, np.nan)
        
        if length >= period:
            # Separate gains and losses
            gains = np.where(deltas > 0, deltas, 0)
            losses = -np.where(deltas < 0, deltas, 0)
            
            # Create strided views for rolling windows
            gains_windows = np.lib.stride_tricks.sliding_window_view(gains, period)
            losses_windows = np.lib.stride_tricks.sliding_window_view(losses, period)
            
            # Calculate means for each window
            avg_gains = np.mean(gains_windows, axis=1)
            avg_losses = np.mean(losses_windows, axis=1)
            
            # Calculate RS and RSI
            # Add small epsilon to avoid division by zero
            rs = avg_gains / (avg_losses + 1e-10)  
            rsi = 100 - (100 / (1 + rs))
            
            # Assign results
            result[period-1:] = rsi

        return pd.Series(result, index=series.index)
    else:
        deltas = series.diff()
        gain = deltas.where(deltas > 0, 0).rolling(window=period).mean()
        loss = -deltas.where(deltas < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


def _generatedseries_calculate_tr(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    """
    Calculate the True Range (TR) for a given series.

    Args:
    - series: pd.Series, the input series (only used to align with generatedSeries_c interface).
    - period: int, the period for the True Range calculation.

    Returns:
    - pd.Series, the calculated True Range series.
    """
    if 1:
        if len(series) < period:
            return pd.Series( [pd.NA] * len(series), index=series.index )  # Not enough data to calculate the slope
        return pt.true_range( df['high'], df['low'], df['close'], length=period )
    else:
        
        high = df['high']
        low = df['low']
        close = df['close']

        high_low = high - low
        high_close_prev = (high - close.shift()).abs()
        low_close_prev = (low - close.shift()).abs()

        tr = high_low.combine(high_close_prev, max).combine(low_close_prev, max)
        return tr

def _generatedseries_calculate_atr(series, period, df:pd.DataFrame, param=None) -> pd.Series:
    if len(series) < period:
        return pd.Series( [pd.NA] * len(series), index=series.index )  # Not enough data to calculate the slope
    return pt.atr( df['high'], df['low'], df['close'], length=period )
    

def _generatedseries_calculate_rising(series: pd.Series, length: int, df:pd.DataFrame, param=None) -> pd.Series:
    """
    Check if the series has been rising for the given length of time.

    Args:
    - series: pd.Series, the input series.
    - length: int, the number of periods to check.

    Returns:
    - pd.Series, a boolean series indicating where the series is rising.
    """
    if len(series) < length:
        return pd.Series([pd.NA] * len(series), index=series.index)  # Not enough data to perform the check

    # Calculate the difference between consecutive elements
    diff_series = series.diff().dropna()
    
    # Create a boolean series indicating whether each rolling window is rising
    is_rising = diff_series.rolling(window=length-1).apply(lambda x: (x > 0).all(), raw=True).astype(bool)
    is_rising = pd.concat([pd.Series([pd.NA] * (length-1), index=series.index[:length-1]), is_rising])

    return is_rising

def _generatedseries_calculate_falling(series: pd.Series, length: int, df:pd.DataFrame, param=None) -> pd.Series:
    """
    Check if the series has been falling for the given length of time.

    Args:
    - series: pd.Series, the input series.
    - length: int, the number of periods to check.

    Returns:
    - pd.Series, a boolean series indicating where the series is falling.
    """
    if len(series) < length:
        return pd.Series([pd.NA] * len(series), index=series.index)  # Not enough data to perform the check
    
    # Calculate the difference between consecutive elements
    diff_series = series.diff().dropna()

    # Create a boolean series indicating whether each rolling window is falling
    is_falling = diff_series.rolling(window=length-1).apply(lambda x: (x < 0).all(), raw=True).astype(bool)
    is_falling = pd.concat([pd.Series([pd.NA] * (length-1), index=series.index[:length-1]), is_falling])
    return is_falling

def _generatedseries_calculate_wma(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    """
    Calculate the Weighted Moving Average (WMA) for a given series and length.
    
    Args:
    - series: pd.Series, the input series.
    - length: int, the period/window for the WMA calculation.
    
    Returns:
    - pd.Series, the calculated WMA series.
    """
    if 1:
        if len(series) < period:
            return pd.Series([pd.NA] * len(series), index=series.index)  # Not enough data to calculate the slope
        return pt.wma( series, period )
    else:
        
        weights = pd.Series(range(1, period + 1))
        wma = series.rolling(period).apply(lambda prices: (prices * weights).sum() / weights.sum(), raw=True)
        return wma

def _generatedseries_calculate_hma(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    """
    Calculate the Hull Moving Average (HMA) for a given series and length.
    
    Args:
    - series: pd.Series, the input series.
    - length: int, the period/window for the HMA calculation.
    
    Returns:
    - pd.Series, the calculated HMA series.
    """
    if 1:
        if len(series) < period:
            return pd.Series([pd.NA] * len(series), index=series.index)  # Not enough data to calculate the slope
        return pt.hma( series, period )
    else:
        if len(series) < period:
            return pd.Series([pd.NA] * len(series), index=series.index)  # Not enough data to calculate the HMA
        
        half_length = int(period / 2)
        sqrt_length = int(period ** 0.5)
        
        wma_half_length = pt.wma(series, half_length)
        wma_full_length = pt.wma(series, period)
        
        diff_wma = 2 * wma_half_length - wma_full_length
        
        hma = pt.wma(diff_wma, sqrt_length)
        
        return hma

def _generatedseries_calculate_slope(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    """
    Calculate the slope of a rolling window for a given length in a pandas Series without using numpy.

    Args:
    - series: pd.Series, the input series.
    - length: int, the period/window for the slope calculation.

    Returns:
    - pd.Series, the calculated slope series.
    """
    if len(series) < period:
        return pd.Series([pd.NA] * len(series), index=series.index)  # Not enough data to calculate the slope
    
    if 1: 
        return pt.slope( series, period )
    else:
        # this one doesn't fail on single candle updates but it's slower than recalculating it all using pandas_ta
        def slope_calc(y):
            x = range(len(y))
            n = len(y)
            x_mean = sum(x) / n
            y_mean = sum(y) / n

            num = sum((x_i - x_mean) * (y_i - y_mean) for x_i, y_i in zip(x, y))
            den = sum((x_i - x_mean) ** 2 for x_i in x)

            if den == 0:
                return 0  # Prevent division by zero

            slope = num / den
            return slope

        # Apply the slope calculation to each rolling window
        slope_series = series.rolling(window=period).apply(slope_calc, raw=False)

        return slope_series

def _generatedseries_calculate_vhma(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    """
    Calculate the Vertical Horizontal Moving Average (VHMA) for a given series.

    Args:
        series (pd.Series): The input series (e.g., close price).
        period (int): The period for the VHMA calculation.
        df (pd.DataFrame): The DataFrame containing the data.
        param (None): Not used.

    Returns:
        pd.Series: The calculated VHMA series.
    """
    highest = series.rolling(window=period).max()
    lowest = series.rolling(window=period).min()
    R = highest - lowest
    change = series.diff().abs()
    vhf = R / change.rolling(window=period).sum()
    vhf = vhf.fillna(0)  # Replace NaN with 0 to avoid issues in calculation

    vhma = np.zeros(len(series))
    vhma[:] = np.nan # Initialize with NaN

    for i in range(1, len(series)):
        if np.isnan(vhma[i-1]):
            vhma[i] = series.iloc[i]
        else:
            vhma[i] = vhma[i-1] + (vhf.iloc[i]**2) * (series.iloc[i] - vhma[i-1])
    
    vhma = pd.Series(vhma, index=series.index)
    return vhma
    
def _generatedseries_calculate_cci(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.cci( df['high'], df['low'], df['close'], period )

def _generatedseries_calculate_bbupper(series, period, df: pd.DataFrame, param=None) -> pd.Series:
    BBmult = param
    sma_name = tools.generatedSeriesNameFormat('sma', series, period)
    stdev_name = tools.generatedSeriesNameFormat('stdev', series, period)

    if len(series) == period:
        # Real-time update case — return just last value
        upper = df[sma_name].iloc[-1] + BBmult * df[stdev_name].iloc[-1]
        return pd.Series([upper], index=[df.index[-1]])

    # Initialization case — full series calculation
    return df[sma_name] + (BBmult * df[stdev_name])

def _generatedseries_calculate_bblower(series, period, df: pd.DataFrame, param=None) -> pd.Series:
    BBmult = param
    sma_name = tools.generatedSeriesNameFormat('sma', series, period)
    stdev_name = tools.generatedSeriesNameFormat('stdev', series, period)

    if len(series) == period:
        # Real-time update case — compute only the last value
        lower = df[sma_name].iloc[-1] - BBmult * df[stdev_name].iloc[-1]
        return pd.Series([lower], index=[df.index[-1]])

    # Initialization case — full series
    return df[sma_name] - (BBmult * df[stdev_name])

def _generatedseries_calculate_inverse_fisher_rsi(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    rsi = df[tools.generatedSeriesNameFormat("rsi", series, period)]
    v1 = 0.1 * (rsi - 50)

    weights = np.arange(1, period + 1)
    wma = v1.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    # Clip v2 before exponentiation to prevent overflow
    v2_clipped = wma.clip(lower=-10, upper=10)
    exp_val = np.exp(2 * v2_clipped)
    iftrsi = (exp_val - 1) / (exp_val + 1)
    return iftrsi

def _generatedseries_calculate_fisher(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    high = df["high"]
    low = df["low"]
    fisher_df = pt.fisher(high, low, length=period, signal=1)  # Minimal signal to isolate Fisher line
    return fisher_df.iloc[:, 0]  # Main Fisher line

def _generatedseries_calculate_fisher_signal(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    high = df["high"]
    low = df["low"]
    fisher_df = pt.fisher(high, low, length=period, signal=param if param else 9)
    return fisher_df.iloc[:, 1]  # Signal line

def _generatedseries_calculate_ao(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    """
    Calculate Awesome Oscillator: SMA(median_price, 5) - SMA(median_price, 34)
    `period` is ignored; `param` can optionally override the two SMA lengths as a tuple: (fast, slow)
    """
    if 'high' not in df.columns or 'low' not in df.columns:
        raise ValueError("DataFrame must contain 'high' and 'low' columns for AO calculation.")
    
    fast, slow = (5, 34)
    if isinstance(param, tuple) and len(param) == 2:
        fast, slow = param

    median_price = (df['high'] + df['low']) / 2
    sma_fast = median_price.rolling(window=fast).mean()
    sma_slow = median_price.rolling(window=slow).mean()

    ao = sma_fast - sma_slow
    return ao

def _generatedseries_calculate_br(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    brar_df = pt.brar(high=df['high'], low=df['low'], close=df['close'], length=period)
    return brar_df['BR']

def _generatedseries_calculate_ar(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    brar_df = pt.brar(high=df['high'], low=df['low'], close=df['close'], length=period)
    return brar_df['AR']

def _generatedseries_calculate_cg(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.cg( series, period )

def _generatedseries_calculate_barssince(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    # Get array of indices where condition is True
    true_indices = np.where(series)[0]
    
    if len(true_indices) == 0:
        return pd.Series(np.nan, index=series.index)
    
    # Create array of all indices
    all_indices = np.arange(len(series))
    
    # Find the closest previous True index for each position
    insertions = np.searchsorted(true_indices, all_indices, side='right') - 1
    
    # Create result array
    result = np.full(len(series), np.nan)
    
    # Calculate distances where we have a previous True index
    valid_mask = insertions >= 0
    result[valid_mask] = all_indices[valid_mask] - true_indices[insertions[valid_mask]]
    
    # Filter by period if specified
    if period is not None:
        result[result > period] = np.nan
    
    return pd.Series(result, index=series.index)


def _generatedseries_calculate_indexwhentrue(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    last_true = -1
    out = []
    for i, val in enumerate(series):
        if val:
            last_true = i
        out.append(last_true if last_true != -1 else np.nan)
    return pd.Series(out, index=series.index)

def _generatedseries_calculate_indexwhenfalse(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    last_false = -1
    out = []
    for i, val in enumerate(series):
        if not val:
            last_false = i
        out.append(last_false if last_false != -1 else np.nan)
    return pd.Series(out, index=series.index)

def _generatedseries_calculate_barswhiletrue(series: pd.Series, period: int = None, df: pd.DataFrame = None, param=None) -> pd.Series:
    length = len(series)
    max_lookback = period if (period is not None and period <= length) else length
    out = []
    count = 0

    for i in range(length):
        val = series.iat[i]
        if val:
            count += 1
        else:
            count = 0

        if period:
            count = min(count, period)

        out.append(count)

    return pd.Series(out, index=series.index)

def _generatedseries_calculate_barswhilefalse(series: pd.Series, period: int = None, df: pd.DataFrame = None, param=None) -> pd.Series:
    length = len(series)
    max_lookback = period if (period is not None and period <= length) else length
    out = []
    count = 0

    for i in range(length):
        val = series.iat[i]
        if not val:
            count += 1
        else:
            count = 0

        if period:
            count = min(count, period)

        out.append(count)

    return pd.Series(out, index=series.index)



class generatedSeries_c:
    def __init__( self, type:str, source:pd.Series, period:int, func = None, param=None, always_reset:bool = False, timeframe = None ):
        self.name = tools.generatedSeriesNameFormat( type, source, period )
        self.sourceName = source.name
        self.period = period
        self.param = param
        self.func = func
        self.timeframe = timeframe
        self.timestamp = 0
        self.alwaysReset = always_reset

        if( self.timeframe == None ):
            raise SystemError( f"Generated Series has no assigned timeframe [{self.name}]")
        
        if( self.timeframe.shadowcopy ):
            raise SystemError( f'Tried to create series [{self.name}] while shadowcopying.' )

        if( self.func == None ):
            raise SystemError( f"Generated Series without a func [{self.name}]")
        
        if( self.period is None ):
            self.period = len(source)

        if( self.period < 1 ):
            raise SystemError( f"Generated Series  with invalid period [{period}]")
        

    def initialize( self, source:pd.Series ):
        if( len(source) >= self.period and ( not self.name in self.timeframe.df.columns or self.alwaysReset ) ):
            if( self.timeframe.shadowcopy ):
                raise SystemError( f"[{self.name}] tried to initialize as shadowcopy" )
            start_time = time.time()
            self.timeframe.df[self.name] = self.func(source, self.period, self.timeframe.df, self.param).dropna()
            self.timestamp = self.timeframe.df['timestamp'].iloc[self.timeframe.barindex]
            if( self.timeframe.stream.initializing ):
                print( f"Initialized {self.name}." + " Elapsed time: {:.2f} seconds".format(time.time() - start_time))


    def update( self, source:pd.Series ):
        if( self.timeframe.shadowcopy ):
            return

        # has this row already been updated?
        if( self.timestamp >= self.timeframe.df.iat[self.timeframe.barindex, 0] ): # same as self.timeframe.df['timestamp'].iloc[self.timeframe.barindex]
            return

        # if non existant try to create new. A few need to be made new every time
        if( self.timestamp == 0 or self.alwaysReset ):
            self.initialize( source )
            return
        
        
        if( len(self.timeframe.df) < self.period ):
            return
        
        # realtime updates

        # slice the required block of candles to calculate the current value of the generated series
        newval = self.func(source[-self.period:], self.period, self.timeframe.df, self.param).loc[self.timeframe.barindex]
        self.timeframe.df.loc[self.timeframe.df.index[-1], self.name] = newval
        self.timestamp = self.timeframe.timestamp

    def value( self, backindex = 0 ):
        if( backindex >= len(self.timeframe.df) ):
            raise SystemError( "generatedseries_c.value() : backindex out of bounds")

        return self.timeframe.df[self.name].iloc[self.timeframe.barindex - backindex]

    def series( self ):
        return self.timeframe.df[self.name]
    
    # direct operations will always operate the last value.
    # If you want to operate the whole series do it with the .series() method.
    def __add__(self, other):
        if isinstance(other, generatedSeries_c):
            return self.value() + other.value()
        if isinstance(other, pd.Series):
            return self.value() + other.iloc[active.barindex]
        return self.value() + other
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, generatedSeries_c):
            return self.value() - other.value()
        if isinstance(other, pd.Series):
            return self.value() - other.iloc[active.barindex]
        return self.value() - other
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __mul__(self, other):
        if isinstance(other, generatedSeries_c):
            return self.value() * other.value()
        elif isinstance(other, pd.Series):
            return self.value() * other.iloc[active.barindex]
        else:
            return self.value() * other
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, generatedSeries_c):
            return self.value() / other.value()
        if isinstance(other, pd.Series):
            return self.value() / other.iloc[active.barindex]
        return self.value() / other
    
    def __rtruediv__(self, other):
        return self.__truediv__(other)
    
    def __lt__(self, other):
        if isinstance(other, generatedSeries_c):
            return self.value() < other.value()
        if isinstance(other, pd.Series):
            return self.value() < other.iloc[active.barindex]
        return self.value() < other
    
    def __rlt__(self, other):
        return self.__lt__(other)
    
    def __gt__(self, other):
        if isinstance(other, generatedSeries_c):
            return self.value() > other.value()
        if isinstance(other, pd.Series):
            return self.value() > other.iloc[active.barindex]
        return self.value() > other
    
    def __rgt__(self, other):
        return self.__gt__(other)
    
    def __eq__(self, other):
        if other is None:
            return False
        if isinstance(other, generatedSeries_c):
            return self.value() == other.value()
        if isinstance(other, pd.Series):
            return self.value() == other.iloc[active.barindex]
        return self.value() == other
    
    # def __req__(self, other):
    #     return self.__eq__(other)

    def plot( self, chart = None ):
        '''it returns the generatedSeries even when calling plot from the timeframe and the function returns the plot_c'''
        if( self.timestamp > 0 ):
            self.timeframe.plot( self.series(), self.name, chart )
            return self
    
    def crossingUp( self, other ):
        df = self.timeframe.df
        if( self.timestamp == 0 or len(df)<2 or self.value() == None or self.value(1) == None ):
            return False
        if isinstance( other, generatedSeries_c ):
            if( other.timestamp == 0  or other.value() == None or other.value(1) == None ):
                return False
            return ( self.value(1) <= other.value(1) and self.value() >= other.value() and self.value() != self.value(1) )
        if isinstance( other, pd.Series ):
            if( len(other) < 2 ):
                return False
            if pd.isna(other.iloc[self.timeframe.barindex-1]) or pd.isna(other.iloc[self.timeframe.barindex]) :
                return False
            return ( self.value(1) <= other.iloc[self.timeframe.barindex-1] and self.value() >= other.iloc[self.timeframe.barindex] and self.value() != self.value(1) )
        try:
            float(other)
        except ValueError:
            return False
        else:
            return ( self.value(1) <= float(other) and self.value() >= float(other) and self.value() != self.value(1) )
    
    def crossingDown( self, other ):
        df = self.timeframe.df
        if( self.timestamp == 0 or len(df)<2 or self.value() == None or self.value(1) == None ):
            return False
        if isinstance( other, generatedSeries_c ):
            if( other.timestamp == 0  or other.value() == None or other.value(1) == None ):
                return False
            return ( self.value(1) >= other.value(1) and self.value() <= other.value() and self.value() != self.value(1) )
        if isinstance( other, pd.Series ):
            if( len(other) < 2 ):
                return False
            if pd.isna(other.iloc[self.timeframe.barindex-1]) or pd.isna(other.iloc[self.timeframe.barindex]) :
                return False
            return ( self.value(1) >= other.iloc[self.timeframe.barindex-1] and self.value() <= other.iloc[self.timeframe.barindex] and self.value() != self.value(1) )
        try:
            float(other)
        except ValueError:
            return False
        else:
            return ( self.value(1) >= float(other) and self.value() <= float(other) and self.value() != self.value(1) )
    
    def crossing( self, other ):
        return self.crossingUp(other) or self.crossingDown(other)
    
    
    


def highest(source: pd.Series, period: int, timeframe=None) -> generatedSeries_c:
    if timeframe is None: timeframe = active.timeframe
    return timeframe.calcGeneratedSeries('highest', source, period, _generatedseries_calculate_highest)

def lowest(source: pd.Series, period: int, timeframe=None) -> generatedSeries_c:
    if timeframe is None: timeframe = active.timeframe
    return timeframe.calcGeneratedSeries('lowest', source, period, _generatedseries_calculate_lowest)

def highestbars(source: pd.Series, period: int, timeframe=None) -> generatedSeries_c:
    if timeframe is None: timeframe = active.timeframe
    return timeframe.calcGeneratedSeries('highestbars', source, period, _generatedseries_calculate_highestbars)

def lowestbars(source: pd.Series, period: int, timeframe=None) -> generatedSeries_c:
    if timeframe is None: timeframe = active.timeframe
    return timeframe.calcGeneratedSeries('lowestbars', source, period, _generatedseries_calculate_lowestbars)

def falling( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'falling', source, period, _generatedseries_calculate_falling )

def rising( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'rising', source, period, _generatedseries_calculate_rising )

def SMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'sma', source, period, _generatedseries_calculate_sma )

def EMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( "ema", source, period, _generatedseries_calculate_ema )

def DEMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( "dema", source, period, _generatedseries_calculate_dema, always_reset=True )

def WMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( "wma", source, period, _generatedseries_calculate_wma )

def HMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( "hma", source, period, _generatedseries_calculate_hma, always_reset=True )

# def JMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
#     if timeframe == None : timeframe = active.timeframe
#     return timeframe.calcGeneratedSeries( "jma", source, period, pt.jma )

# def KAMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
#     if timeframe == None : timeframe = active.timeframe
#     return timeframe.calcGeneratedSeries( "kama", source, period, pt.kama )

def LINREG( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( "linreg", source, period, _generatedseries_calculate_linreg )

def RSI( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'rsi', source, period, _generatedseries_calculate_rsi )

def DEV( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'dev', source, period, _generatedseries_calculate_dev )

def STDEV( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'stdev', source, period, _generatedseries_calculate_stdev )

def RMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'rma', source, period, _generatedseries_calculate_rma, always_reset=True )

def WPR( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'wpr', source, period, _generatedseries_calculate_williams_r )

def TR( period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'tr', pd.Series([pd.NA] * period, name = 'tr'), period, _generatedseries_calculate_tr )

def ATR2( period:int, timeframe = None )->generatedSeries_c: # The other one using pt is much faster
    if timeframe == None : timeframe = active.timeframe
    tr = timeframe.calcGeneratedSeries( 'tr', timeframe.df['close'], period, _generatedseries_calculate_tr )
    return timeframe.calcGeneratedSeries( 'atr', tr.series(), period, _generatedseries_calculate_rma )

def ATR( period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'atr', timeframe.df['close'], period, _generatedseries_calculate_atr )

def SLOPE( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'slope', source, period, _generatedseries_calculate_slope, always_reset=True )

def VHMA(source: pd.Series, period: int, timeframe=None)->generatedSeries_c:
    if timeframe is None:
        timeframe = active.timeframe
    return timeframe.calcGeneratedSeries('vhma', source, period, _generatedseries_calculate_vhma, always_reset = True)

def BIAS( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'bias', source, period, _generatedseries_calculate_bias )

def CCI( period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'cci', timeframe.df['close'], period, _generatedseries_calculate_cci )

def CFO( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'cfo', source, period, _generatedseries_calculate_cfo )

def FWMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'fwma', source, period, _generatedseries_calculate_fwma )

def BBu( source:pd.Series, period:int, mult:float, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'bbu', source, period, _generatedseries_calculate_bbupper, mult )

def BBl( source:pd.Series, period:int, mult:float, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'bbl', source, period, _generatedseries_calculate_bblower, mult )

def IFTrsi( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    rsi = timeframe.calcGeneratedSeries( 'rsi', source, period, _generatedseries_calculate_rsi )
    return timeframe.calcGeneratedSeries( 'iftrsi', timeframe.df['close'], period, _generatedseries_calculate_inverse_fisher_rsi )

def Fisher( period:int, signal:float=None, timeframe = None )->tuple[generatedSeries_c, generatedSeries_c]:
    if timeframe == None : timeframe = active.timeframe
    fish = timeframe.calcGeneratedSeries( 'fisher', timeframe.df['close'], period, _generatedseries_calculate_fisher )
    sig = timeframe.calcGeneratedSeries( 'fishersig', timeframe.df['close'], period, _generatedseries_calculate_fisher_signal, signal )
    return fish, sig
    
def AO( period: int = 0, fast: int = 5, slow: int = 34, timeframe = None ) -> generatedSeries_c:
    if timeframe is None: timeframe = active.timeframe
    param = (fast, slow)
    return timeframe.calcGeneratedSeries('ao', timeframe.df['close'], period, _generatedseries_calculate_ao, param)

def BRAR( period:int, signal:float=None, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    br = timeframe.calcGeneratedSeries( 'br', timeframe.df['close'], period, _generatedseries_calculate_br )
    ar = timeframe.calcGeneratedSeries( 'ar', timeframe.df['close'], period, _generatedseries_calculate_ar )
    return br, ar

def CG( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'cg', source, period, _generatedseries_calculate_cg )

def barsSinceSeries(source: pd.Series, period: int, timeframe=None) -> generatedSeries_c:
    import inspect
    # Get caller info by going up 2 levels in the stack
    caller_frame = inspect.currentframe().f_back.f_back
    frame_info = inspect.getframeinfo(caller_frame)
    caller_id = f"{frame_info.function}_{frame_info.lineno}"
    if timeframe is None:
        timeframe = active.timeframe
    return timeframe.calcGeneratedSeries('barsSince'+caller_id, source, period, _generatedseries_calculate_barssince)

def barsWhileTrueSeries(source: pd.Series, period: int = None, timeframe=None) -> generatedSeries_c:
    if timeframe is None: timeframe = active.timeframe
    return timeframe.calcGeneratedSeries('barsWhileTrue', source, period, _generatedseries_calculate_barswhiletrue)

def barsWhileFalseSeries(source: pd.Series, period: int = None, timeframe=None) -> generatedSeries_c:
    if timeframe is None: timeframe = active.timeframe
    return timeframe.calcGeneratedSeries('barsWhileFalse', source, period, _generatedseries_calculate_barswhilefalse)

def indexWhenTrueSeries(source: pd.Series, period: int = None, timeframe=None) -> generatedSeries_c:
    if timeframe is None:
        timeframe = active.timeframe
    return timeframe.calcGeneratedSeries('indexwhentrue_series', source, period, _generatedseries_calculate_indexwhentrue)

def indexWhenFalseSeries(source: pd.Series, period: int, timeframe=None) -> generatedSeries_c:
    if timeframe is None:
        timeframe = active.timeframe
    return timeframe.calcGeneratedSeries('indexwhenfalse_series', source, period, _generatedseries_calculate_indexwhenfalse)


# #
# # OTHER NOT GENERATED SERIES
# #

def BollingerBands( source:pd.Series, period:int, mult:float = 2.0 )->tuple[generatedSeries_c, generatedSeries_c, generatedSeries_c]:
    """
    Returns the Bollinger Bands (basis, upper, lower) for the given source series and period.

    Args:
        source (pd.Series): The input pandas Series to calculate the Bollinger Bands on.
        period (int): The period/window for the Bollinger Bands calculation.

    Returns:
        Tuple[generatedSeries_c, generatedSeries_c, generatedSeries_c]: The basis (SMA), upper band, and lower band as generatedSeries_c objects.
    """
    BBbasis = SMA(source, period)
    STDEV(source, period)
    BBupper = active.timeframe.calcGeneratedSeries( 'bbu', source, period, _generatedseries_calculate_bbupper, mult )
    BBlower = active.timeframe.calcGeneratedSeries( 'bbl', source, period, _generatedseries_calculate_bblower, mult )
    return BBbasis, BBupper, BBlower


def indexWhenTrue( source ):
    """
    Returns the index of the most recent (from the end) True value in the given series.

    Args:
        source: The input series or generatedSeries_c to inspect.

    Returns:
        int or None: The index position of the most recent True value, or None if not found.
    """
    if( not isinstance(source, pd.Series ) ):
        if( isinstance( source, generatedSeries_c) ):
            source = source.series()
        else:
            raise ValueError( "calcIndexWhenTrue must be called with a series" )
    boolean_source = source.astype(bool) if source.dtype != bool else source
    return boolean_source[::-1].idxmax() if source.any() else None


def barsSince( source ):
    """
    Returns the number of bars since the last True value in the given series.

    Args:
        source: The input series or generatedSeries_c to inspect.

    Returns:
        int or None: The number of bars since the last True value, or None if not found.
    """
    index_when_true = indexWhenTrue( source )
    if( index_when_true == None ):
        return None
    return active.barindex - index_when_true


def indexWhenFalse( source ):
    """
    Returns the index of the most recent (from the end) False value in the given series.

    Args:
        source: The input series or generatedSeries_c to inspect.

    Returns:
        int: The index position of the most recent False value, or 0 if not found or series is empty.
    """
    if( not isinstance(source, pd.Series ) ):
        if( isinstance( source, generatedSeries_c) ):
            source = source.series()
        else:
            raise ValueError( "calcIndexWhenFalse must be called with a series" )
    if not source.empty and source.any():
        # Ensure the source is boolean
        boolean_source = source.astype(bool) if source.dtype != bool else source
        
        # Reverse the boolean source and Calculate the cumulative sum of the negated values
        cumsum_negated = (~boolean_source[::-1]).cumsum()
        first_false_index = cumsum_negated[cumsum_negated == 1].index.min() # Find the first 1 in the cumulative sum (which means the first False in the original series from the end)

        return first_false_index
    else:
        return 0  # Return 0 if the series is empty or has no True values''


def barsWhileTrue( source ):
    """
    Returns the number of bars for which the given series has continuously been True, up to the current barindex.

    Args:
        barindex: The current bar index.
        source: The input series or generatedSeries_c to inspect.

    Returns:
        int or None: The number of bars while True, or None if not found.
    """
    index_when_false = indexWhenFalse( source )
    if( index_when_false == None ):
        return None
    return active.barindex - index_when_false

class pivots_c:
    def __init__(self, high: pd.Series, low: pd.Series, depth: int = 64, deviation: float = 2, backstep: int = 5):
        self.depth = depth
        self.deviation = deviation
        self.backstep = backstep
        self.precision = 100
        
        # State variables
        self.direction = 0
        self.heightDiffTopEnough = False
        self.heightDiffBottomEnough = False

        # Initialize tracking columns and get their indices
        self.initializeTrackingColumns()

        # State variables
        self.zindex = active.barindex
        self.z1index = active.barindex
        self.z2index = active.barindex
        self.last_update = -1
        self.last_direction = 0

        # confirmed pivots
        self.new = 0
        self.last_pivot_index = 0
        self.last_pivot_price = 0.0
        self.last_pivot_timestamp = 0
        self.high_pivots = []
        self.low_pivots = []

    def makeName(self):
        return f'OHDTE{active.timeframe.timeframeStr}{self.depth}{self.deviation}{self.backstep}'

    def update2(self, high: pd.Series, low: pd.Series):
        self.new = 0
        barindex = active.barindex
        if barindex < self.depth:
            return (self.direction, self.z1index, self.z2index)
        
        hbGenSeries = highestbars(high, self.depth)
        lbGenSeries = lowestbars(low, self.depth)
        hb_offset = hbGenSeries.value()
        lb_offset = lbGenSeries.value()

        # store in dataframe columns the old results from heighDiffTopEnough and heightDiffBottomEnough for vectorized comparison
        if( not active.timeframe.shadowcopy ):
            active.timeframe.df.iat[barindex, self.OHDTEcolumnindex] = not self.heightDiffTopEnough
            active.timeframe.df.iat[barindex, self.OHDBEcolumnindex] = not self.heightDiffBottomEnough

        self.heightDiffTopEnough = high[barindex-hb_offset] - high[barindex] > self.deviation * self.precision
        self.heightDiffBottomEnough = low[barindex] - low[barindex-lb_offset] > self.deviation * self.precision

        hr_gs = barsSinceSeries(active.timeframe.df.iloc[:, self.OHDTEcolumnindex], self.depth)
        lr_gs = barsSinceSeries(active.timeframe.df.iloc[:, self.OHDBEcolumnindex], self.depth)

        condition = barsSinceSeries(hr_gs.series() <= lr_gs.series(), self.depth + 1).value()
        if not condition:
            condition = 0

        new_direction = -1 if condition >= self.backstep else 1
        
        # print( f"HR:{hr_gs.value()} LR:{lr_gs.value()} direction:{self.direction}")

        if new_direction != self.direction:
            # self.newPivot(self.direction, high, low)
            self.z1index = self.z2index
            self.z2index = self.zindex
            # self.zindex = active.barindex

        high_now = high.at[barindex]
        low_now = low.at[barindex]

        if new_direction > 0:
            if high_now > high.at[self.z2index]:
                self.z2index = barindex
                self.zindex = barindex
            if low_now < low.at[self.zindex]:
                self.zindex = barindex

        if new_direction < 0:
            if low_now < low.at[self.z2index]:
                self.z2index = barindex
                self.zindex = barindex
            if high_now > high.at[self.zindex]:
                self.zindex = barindex

        self.direction = new_direction
        return (self.direction, self.z1index, self.z2index)
    
    def update(self, high: pd.Series, low: pd.Series):
        direction, confirmed, current = self.update2(high, low)
        if( confirmed != self.last_pivot_index ):
            self.last_pivot_index = confirmed
            self.last_pivot_price = high[confirmed]
            self.last_pivot_timestamp = active.timeframe.df['timestamp'].at[confirmed]
            # if direction == -1 :
            #     self.high_pivots.append((confirmed, high[confirmed]))
            # elif self.direction == 1:
            #     self.low_pivots.append((confirmed, low[confirmed]))
            # self.confirmedLast = confirmed
            self.new = direction

    def initializeTrackingColumns(self):
        """Initialize columns for tracking height differences and store their indices"""
        df = active.timeframe.df
        
        # Initialize result arrays
        height_diff_top = pd.Series(False, index=df.index)
        height_diff_bottom = pd.Series(False, index=df.index)
        
        # Calculate only from depth onwards
        for i in range(self.depth, len(df)):
            window = slice(i - self.depth + 1, i + 1)
            window_high = df.high.iloc[window]
            window_low = df.low.iloc[window]
            
            # Find highest and lowest in window
            highest_idx = i - self.depth + 1 + window_high.argmax()
            lowest_idx = i - self.depth + 1 + window_low.argmin()
            
            # Calculate differences
            height_diff_top.iloc[i] = df.high.iloc[highest_idx] - df.high.iloc[i] > self.deviation * self.precision
            height_diff_bottom.iloc[i] = df.low.iloc[i] - df.low.iloc[lowest_idx] > self.deviation * self.precision

        # Create and store columns
        active.timeframe.df[self.makeName()+'T'] = ~height_diff_top
        active.timeframe.df[self.makeName()+'B'] = ~height_diff_bottom
        self.OHDTEcolumnindex = active.timeframe.df.columns.get_loc(self.makeName()+'T')
        self.OHDBEcolumnindex = active.timeframe.df.columns.get_loc(self.makeName()+'B')

'''
    heighDiffTopEnough = (_high[-ta.highestbars(depth)] - _high > deviation*syminfo.mintick)
    hr = ta.barssince(not heighDiffTopEnough[1] )
    heightDiffBottomEnough = (_low - _low[-ta.lowestbars(depth)] > deviation*syminfo.mintick)
    lr = ta.barssince(not heightDiffBottomEnough[1] )
         
    direction = ta.barssince(hr <= lr) >= backstep? -1: 1
'''
'''
var last_h = 1
last_h += 1
var last_l = 1
last_l += 1
var lw = 1
var hg = 1
lw += 1
hg += 1
p_lw = -ta.lowestbars(bottom, Depth)
p_hg = -ta.highestbars(top, Depth)
lowing = lw == p_lw or bottom - bottom[p_lw] > Deviation * syminfo.mintick
highing = hg == p_hg or top[p_hg] - top > Deviation * syminfo.mintick
lh = ta.barssince(not highing[1])
ll = ta.barssince(not lowing[1])
down = ta.barssince(not(lh > ll)) >= Backstep
lower = bottom[lw] > bottom[p_lw]
higher = top[hg] < top[p_hg]
if lw != p_lw and (not down[1] or lower)
    lw := p_lw < hg ? p_lw : 0
    lw
if hg != p_hg and (down[1] or higher)
    hg := p_hg < lw ? p_hg : 0
    hg

line zz = na
label point = na
x1 = down ? lw : hg
y1 = down ? bottom[lw] : top[hg]'''


pivotsNow:pivots_c = None
def pivots( high:pd.Series, low:pd.Series )->pivots_c:
    global pivotsNow
    if pivotsNow == None:
        pivotsNow = pivots_c(high, low)

    pivotsNow.update(high, low)
    return pivotsNow



def crossingUp( self, other ):
    """
    Determines if 'self' crosses up over 'other' between the previous and current bar.

    Args:
        self: The first value/series.
        other: The second value/series to compare against.

    Returns:
        bool: True if a crossing up occurred, False otherwise.
    """
    if isinstance( self, generatedSeries_c ):
        return self.crossingUp( other )
    
    if isinstance( self, int ):
        self = float(self)

    if isinstance( other, int ):
        other = float(other)

    if( isinstance(self, float) and isinstance(other, float) ):
        print( "* WARNING: crossinUp: Two static values can never cross" )
        return False
    
    self_old = 0
    self_new = 0
    other_old = 0
    other_new = 0
    if isinstance( self, pd.Series ):
        if( len(self) < 2 or active.barindex < 1 ):
            return False
        self_old = self.iloc[active.barindex-1]
        self_new = self.iloc[active.barindex]
        if isinstance( other, pd.Series ):
            if( len(other) < 2 ):
                return False
            other_old = other.iloc[active.barindex-1]
            other_new = other.iloc[active.barindex]
        elif isinstance( other, generatedSeries_c ):
            if( other.timestamp == 0 or len(other.series()) < 2 or active.barindex < 1 ):
                return False
            other_old = other.value(1)
            other_new = other.value()
        else:
            try:
                float(other)
            except ValueError:
                return False
            else:
                other_old = float(other)
                other_new = float(other)
    else:
        try:
            float(self)
        except ValueError:
            print( "crossinUp: Unsupported type", type(self) )
            return False
        else:
            return crossingDown( other, self )

    return ( self_old <= other_old and self_new >= other_new and self_old != self_new )


def crossingDown( self, other ):
    """
    Determines if 'self' crosses down below 'other' between the previous and current bar.

    Args:
        self: The first value/series.
        other: The second value/series to compare against.

    Returns:
        bool: True if a crossing down occurred, False otherwise.
    """
    if isinstance( self, generatedSeries_c ):
        return self.crossingDown( other )
    
    if isinstance( self, int ):
        self = float(self)

    if isinstance( other, int ):
        other = float(other)

    if( isinstance(self, float) and isinstance(other, float) ):
        print( "* WARNING: crossinDown: Two static values can never cross" )
        return False
    
    self_old = 0
    self_new = 0
    other_old = 0
    other_new = 0
    if isinstance( self, pd.Series ):
        if( len(self) < 2 or active.barindex < 1 ):
            return False
        self_old = self.iloc[active.barindex-1]
        self_new = self.iloc[active.barindex]
        if isinstance( other, pd.Series ):
            if( len(other) < 2 ):
                return False
            other_old = other.iloc[active.barindex-1]
            other_new = other.iloc[active.barindex]
        elif isinstance( other, generatedSeries_c ):
            if( other.timestamp == 0 or len(other.series()) < 2 or active.barindex < 1 ):
                return False
            other_old = other.value(1)
            other_new = other.value()
        else:
            try:
                float(other)
            except ValueError:
                return False
            else:
                other_old = float(other)
                other_new = float(other)
    else:
        try:
            float(self)
        except ValueError:
            print( "crossinDown: Unsupported type", type(self) )
            return False
        else:
            return crossingUp( other, self )

    return ( self_old >= other_old and self_new <= other_new and self_old != self_new )

def crossing( self, other ):
    """
    Determines if 'self' crosses either up or down with respect to 'other' between the previous and current bar.

    Args:
        self: The first value/series.
        other: The second value/series to compare against.

    Returns:
        bool: True if a crossing (up or down) occurred, False otherwise.
    """
    return crossingUp( other, self ) or crossingDown( other, self )
