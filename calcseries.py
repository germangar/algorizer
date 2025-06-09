
import pandas as pd
try:
    import talib
    talib_available = True
except ImportError:
    talib_available = False
    print("talib import failed")
import pandas_ta as pt
import numpy as np
import time
from constants import c
import active
import tools

if pt.Imports["talib"]:
    print("pandas_ta is using talib")
else:
    print("pandas_ta is not using talib")
    
# from talib import abstract
# from pprint import pprint
# pprint( abstract.Function("SUB").info )
    

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

def _generatedseries_calculate_binary_operator(dummy_source: pd.Series, period: int, df: pd.DataFrame, param: tuple) -> pd.Series:
    """
    Apply a custom binary operation between two columns in the dataframe.

    Args:
        dummy_source (pd.Series): Ignored, only used for naming compatibility.
        period (int): Minimum required length for initialization (usually 1).
        df (pd.DataFrame): The dataframe containing the operands.
        param (tuple): (colA, colB, op_func), where:
            - colA (str): Name of first column.
            - colB (str): Name of second column.
            - op_func (Callable): A lambda or function taking two Series and returning a Series.

    Returns:
        pd.Series: Resulting series from applying op_func to colA and colB.
    """
    if not isinstance(param, tuple) or len(param) != 3:
        raise ValueError("binary_operator series requires param=(colA, colB)")
    
    colA, colB, op_func = param

    if colA not in df.columns or colB not in df.columns:
        raise ValueError(f"binary_operator series: Columns '{colA}' or '{colB}' not found in DataFrame")

    seriesA = df[colA]
    seriesB = df[colB]
    result = op_func(seriesA, seriesB)
    return result

def _generatedseries_calculate_subtract_series(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    if not isinstance(param, tuple) or len(param) != 2:
        raise ValueError("subtract_series requires param=(colA, colB)")
    
    colA, colB = param
    if colA not in df.columns or colB not in df.columns:
        raise KeyError(f"Columns {colA} or {colB} not found in DataFrame")

    return df[colA] - df[colB]

def _generatedseries_calculate_add_series(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    if not isinstance(param, tuple) or len(param) != 2:
        raise ValueError("subtract_series requires param=(colA, colB)")
    
    colA, colB = param
    if colA not in df.columns or colB not in df.columns:
        raise KeyError(f"Columns {colA} or {colB} not found in DataFrame")

    return df[colA] + df[colB]

def _generatedseries_calculate_multiply_series(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    if not isinstance(param, tuple) or len(param) != 2:
        raise ValueError("subtract_series requires param=(colA, colB)")
    
    colA, colB = param
    if colA not in df.columns or colB not in df.columns:
        raise KeyError(f"Columns {colA} or {colB} not found in DataFrame")

    return df[colA] * df[colB]

def _generatedseries_calculate_divide_series(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    if not isinstance(param, tuple) or len(param) != 2:
        raise ValueError("subtract_series requires param=(colA, colB)")
    
    colA, colB = param
    if colA not in df.columns or colB not in df.columns:
        raise KeyError(f"Columns {colA} or {colB} not found in DataFrame")

    return df[colA] / df[colB]

# def _generatedseries_calculate_series_const_lambda( series: pd.Series, period: int, df: pd.DataFrame, param: tuple[float, object] ) -> pd.Series:
#     if not isinstance(param, tuple) or len(param) != 2:
#         raise ValueError("series_const_operator requires param=(constant, op_func)")
#     constant, op_func = param
#     if not callable(op_func):
#         raise ValueError("The second item in param must be a callable (e.g., lambda function)")

#     return op_func(series, constant)

def _generatedseries_calculate_series_const_lambda(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    if not isinstance(param, tuple) or len(param) != 2:
        raise ValueError("SeriesConstOp param must be a tuple: (constant, op_func)")
    constant, op_func = param
    if not callable(op_func):
        raise ValueError("The second item in param must be a callable (e.g., lambda function)")
    
    # Apply the operation safely with try/except in case of numeric errors
    try:
        result = op_func(series, constant)
    except Exception as e:
        raise RuntimeError(f"Error applying operation to Series: {e}")

    # Replace inf with NaN to prevent polluting the result
    return pd.Series(result).replace([np.inf, -np.inf], np.nan)

# REVERSED!
def _generatedseries_calculate_const_series_lambda(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    if not isinstance(param, tuple) or len(param) != 2:
        raise ValueError("ConstSeriesOp param must be a tuple: (constant, op_func)")

    constant, op_func = param
    if not callable(op_func):
        raise ValueError("The second item in param must be a callable (e.g., lambda function)")

    try:
        result = op_func(constant, series)
    except Exception as e:
        raise RuntimeError(f"Error applying const-series operation: {e}")

    return pd.Series(result).replace([np.inf, -np.inf], np.nan)



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


def _generatedseries_calculate_indexwhentrue(
    series: pd.Series, period: int, df: pd.DataFrame, param=None
) -> pd.Series:
    length = len(series)
    out = np.full(length, np.nan)
    last_true = -1
    for i, val in enumerate(series):
        if val:
            last_true = i
        if last_true != -1:
            out[i] = last_true
    return pd.Series(out, index=series.index)


def _generatedseries_calculate_indexwhenfalse(
    series: pd.Series, period: int, df: pd.DataFrame, param=None
) -> pd.Series:
    length = len(series)
    out = np.full(length, np.nan)
    last_false = -1
    for i, val in enumerate(series):
        if not val:
            last_false = i
        if last_false != -1:
            out[i] = last_false
    return pd.Series(out, index=series.index)


def _generatedseries_calculate_barswhiletrue(
    series: pd.Series, period: int = None, df: pd.DataFrame = None, param=None
) -> pd.Series:
    arr = series.values.astype(bool)
    counts = np.zeros_like(arr, dtype=int)
    c = 0
    for i, val in enumerate(arr):
        c = c + 1 if val else 0
        if period:
            c = min(c, period)
        counts[i] = c
    return pd.Series(counts, index=series.index)


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
        
        timeframe = self.timeframe

        # has this row already been updated?
        if( self.timestamp >= timeframe.df.iat[timeframe.barindex, 0] ): # same as self.timeframe.df['timestamp'].iloc[self.timeframe.barindex]
            return

        # if non existant try to create new. A few need to be made new every time
        if( self.timestamp == 0 or self.alwaysReset ):
            self.initialize( source )
            return

        # realtime updates

        # slice the required block of candles to calculate the current value of the generated series
        newval = self.func(source[-self.period:], self.period, timeframe.df, self.param).loc[timeframe.barindex]
        timeframe.df.loc[timeframe.df.index[-1], self.name] = newval
        self.timestamp = timeframe.timestamp

    def value( self, backindex = 0 ):
        if( backindex >= len(self.timeframe.df) ):
            raise SystemError( "generatedseries_c.value() : backindex out of bounds")

        return self.timeframe.df[self.name].iloc[self.timeframe.barindex - backindex]

    def series( self ):
        return self.timeframe.df[self.name]
    

    def __add__(self, other):
        if isinstance(other, (generatedSeries_c, pd.Series)):
            return addSeries(self.name, other.name)
        if isinstance(other, (float, int)):
            return addSeriesConst(self, other)
        raise ValueError("WTF def __add__(self, other)")

    def __radd__(self, other):
        if isinstance(other, (float, int)):
            return addToConst(other, self)
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (generatedSeries_c, pd.Series)):
            return subtractSeries(self.name, other.name)
        if isinstance(other, (float, int)):
            return subtractSeriesConst(self, other)
        raise ValueError("WTF def __sub__(self, other)")

    def __rsub__(self, other):
        if isinstance(other, (float, int)):
            return subtractFromConst(other, self)
        raise ValueError("rsub only defined for const - series")

    def __mul__(self, other):
        if isinstance(other, (generatedSeries_c, pd.Series)):
            return multiplySeries(self.name, other.name)
        if isinstance(other, (float, int)):
            return mulSeriesConst(self, other)
        raise ValueError("WTF def __mul__(self, other)")

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return multFromConst(other, self)
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (generatedSeries_c, pd.Series)):
            return divideSeries(self.name, other.name)
        if isinstance(other, (float, int)):
            return divSeriesConst(self, other)
        raise ValueError("WTF def __truediv__(self, other)")

    def __rtruediv__(self, other):
        if isinstance(other, (float, int)):
            return divFromConst(other, self)
        raise ValueError("rtruediv only defined for const / series")
    




    def __lt__(self, other):
        if isinstance(other, (generatedSeries_c, pd.Series)):
            return compareSeries(self.name, other.name, lambda a, b: a < b, "<")
        if isinstance(other, (float, int)):
            return SeriesConstOp(self, other, lambda a, b: a < b, "<")
        raise ValueError("Unsupported operand type for <")
    
    def __rlt__(self, other):
        if isinstance(other, (float, int)):
            return SeriesConstOp(self, other, lambda a, b: b < a, "r<")
        raise ValueError("Unsupported reversed operand for <")

    def __le__(self, other):
        if isinstance(other, (generatedSeries_c, pd.Series)):
            return compareSeries(self.name, other.name, lambda a, b: a <= b, "<=")
        if isinstance(other, (float, int)):
            return SeriesConstOp(self, other, lambda a, b: a <= b, "<=")
        raise ValueError("Unsupported operand type for <=")
    
    def __rle__(self, other):
        if isinstance(other, (float, int)):
            return SeriesConstOp(self, other, lambda a, b: b <= a, "r<=")
        raise ValueError("Unsupported reversed operand for <=")

    def __gt__(self, other):
        if isinstance(other, (generatedSeries_c, pd.Series)):
            return compareSeries(self.name, other.name, lambda a, b: a > b, ">")
        if isinstance(other, (float, int)):
            return SeriesConstOp(self, other, lambda a, b: a > b, ">")
        raise ValueError("Unsupported operand type for >")
    
    def __rgt__(self, other):
        if isinstance(other, (float, int)):
            return SeriesConstOp(self, other, lambda a, b: b > a, "r>")
        raise ValueError("Unsupported reversed operand for >")

    def __ge__(self, other):
        if isinstance(other, (generatedSeries_c, pd.Series)):
            return compareSeries(self.name, other.name, lambda a, b: a >= b, ">=")
        if isinstance(other, (float, int)):
            return SeriesConstOp(self, other, lambda a, b: a >= b, ">=")
        raise ValueError("Unsupported operand type for >=")
    
    def __rge__(self, other):
        if isinstance(other, (float, int)):
            return SeriesConstOp(self, other, lambda a, b: b >= a, "r>=")
        raise ValueError("Unsupported reversed operand for >=")

    def __eq__(self, other):
        if isinstance(other, (generatedSeries_c, pd.Series)):
            return compareSeries(self.name, other.name, lambda a, b: a == b, "==")
        if isinstance(other, (float, int)):
            return SeriesConstOp(self, other, lambda a, b: a == b, "==")
        return NotImplemented
    
    def __req__(self, other):
        if isinstance(other, (float, int)):
            return SeriesConstOp(self, other, lambda a, b: b == a, "r==")
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, (generatedSeries_c, pd.Series)):
            return compareSeries(self.name, other.name, lambda a, b: a != b, "!=")
        if isinstance(other, (float, int)):
            return SeriesConstOp(self, other, lambda a, b: a != b, "!=")
        return NotImplemented

    def __rne__(self, other):
        if isinstance(other, (float, int)):
            return SeriesConstOp(self, other, lambda a, b: b != a, "r!=")
        return NotImplemented



    def plot( self, chart_name = None, color = "#8FA7BBAA", style = 'solid', width = 1  ):
        '''* it returns the generatedSeries. Calling plot from the timeframe and the function returns the plot_c but not here*

        source: can either be a series or a value. A series can only be plotted when it is in the dataframe. When plotting a value a series will be automatically created in the dataframe.
        chart_name: Leave empty for the main panel. Use 'panel' for plotting in the subpanel.
        color: in a string. Can be hexadecial '#DADADADA' or rgba format 'rgba(255,255,255,1.0)'
        style: LINE_STYLE = Literal['solid', 'dotted', 'dashed', 'large_dashed', 'sparse_dotted']
        width: int
        '''
        if( self.timestamp > 0 ):
            self.timeframe.plot( self.series(), self.name, chart_name, color, style, width )
            return self
    def histogram( self, chart_name = None, color = "#4A545D", margin_top = 0.0, margin_bottom = 0.0 ):
        '''* it returns the generatedSeries. Calling plot from the timeframe and the function returns the plot_c but not here*

        source: can either be a series or a value. A series can only be plotted when it is in the dataframe. When plotting a value a series will be automatically created in the dataframe.
        chart_name: Leave empty for the main panel. Use 'panel' for plotting in the subpanel.
        color: in a string. Can be hexadecial '#DADADADA' or rgba format 'rgba(255,255,255,1.0)'
        style: LINE_STYLE = Literal['solid', 'dotted', 'dashed', 'large_dashed', 'sparse_dotted']
        width: int
        '''
        if( self.timestamp > 0 ):
            self.timeframe.histogram( self.series(), self.name, chart_name, color, margin_top, margin_bottom )
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
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('highest', source, period, _generatedseries_calculate_highest)

def lowest(source: pd.Series, period: int, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('lowest', source, period, _generatedseries_calculate_lowest)

def highestbars(source: pd.Series, period: int, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('highestbars', source, period, _generatedseries_calculate_highestbars)

def lowestbars(source: pd.Series, period: int, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('lowestbars', source, period, _generatedseries_calculate_lowestbars)

def falling( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'falling', source, period, _generatedseries_calculate_falling )

def rising( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'rising', source, period, _generatedseries_calculate_rising )

def binaryOperationSeries(colA: str|pd.Series|generatedSeries_c, colB: str|pd.Series|generatedSeries_c, op_func, timeframe=None) -> generatedSeries_c:
    """
    op_func: a lambda or function taking (Series a, Series b) and returning a Series. Example: 'lambda a, b: a + b'
    """
    if isinstance(colA,(pd.Series, generatedSeries_c)):
        colA = colA.name
    if isinstance(colB,(pd.Series, generatedSeries_c)):
        colB = colB.name
    import inspect
    timeframe = timeframe or active.timeframe
    # Get caller info by going up 2 levels in the stack
    caller_frame = inspect.currentframe().f_back
    frame_info = inspect.getframeinfo(caller_frame)
    caller_id = f"{frame_info.function[:3]}{str(frame_info.lineno)[-3:]}"
    indexA = timeframe.df.columns.get_loc(colA)
    indexB = timeframe.df.columns.get_loc(colB)
    dummy_name = f"{caller_id}{indexA}{indexB}"
    dummy_source = pd.Series([np.nan], name=dummy_name) # must have len of at least 1
    return timeframe.calcGeneratedSeries( 'biop', dummy_source, 1, _generatedseries_calculate_binary_operator, param=(colA, colB, op_func) )

def subtractSeries(colA: str|pd.Series|generatedSeries_c, colB: str|pd.Series|generatedSeries_c, timeframe=None) -> generatedSeries_c:
    # return binaryOperationSeries(colA, colB, lambda a, b: a - b) # for testing binaryOperationSeries
    if isinstance(colA,(pd.Series, generatedSeries_c)):
        colA = colA.name
    if isinstance(colB,(pd.Series, generatedSeries_c)):
        colB = colB.name
    timeframe = timeframe or active.timeframe
    indexA = timeframe.df.columns.get_loc(colA)
    indexB = timeframe.df.columns.get_loc(colB)
    dummy_source = pd.Series([np.nan], name=f"{indexA}&{indexB}") # must have len of at least 1
    return timeframe.calcGeneratedSeries( 'sub', dummy_source, 1, _generatedseries_calculate_subtract_series, param=(colA, colB))

def addSeries(colA: str|pd.Series|generatedSeries_c, colB: str|pd.Series|generatedSeries_c, timeframe=None) -> generatedSeries_c:
    if isinstance(colA,(pd.Series, generatedSeries_c)):
        colA = colA.name
    if isinstance(colB,(pd.Series, generatedSeries_c)):
        colB = colB.name
    timeframe = timeframe or active.timeframe
    indexA = timeframe.df.columns.get_loc(colA)
    indexB = timeframe.df.columns.get_loc(colB)
    dummy_source = pd.Series([np.nan], name=f"{indexA}&{indexB}") # must have len of at least 1
    return timeframe.calcGeneratedSeries( 'add', dummy_source, 1, _generatedseries_calculate_add_series, param=(colA, colB))

def multiplySeries(colA: str|pd.Series|generatedSeries_c, colB: str|pd.Series|generatedSeries_c, timeframe=None) -> generatedSeries_c:
    if isinstance(colA,(pd.Series, generatedSeries_c)):
        colA = colA.name
    if isinstance(colB,(pd.Series, generatedSeries_c)):
        colB = colB.name
    timeframe = timeframe or active.timeframe
    indexA = timeframe.df.columns.get_loc(colA)
    indexB = timeframe.df.columns.get_loc(colB)
    dummy_source = pd.Series([np.nan], name=f"{indexA}&{indexB}") # must have len of at least 1
    return timeframe.calcGeneratedSeries( 'mult', dummy_source, 1, _generatedseries_calculate_multiply_series, param=(colA, colB))

def divideSeries(colA: str|pd.Series|generatedSeries_c, colB: str|pd.Series|generatedSeries_c, timeframe=None) -> generatedSeries_c:
    if isinstance(colA,(pd.Series, generatedSeries_c)):
        colA = colA.name
    if isinstance(colB,(pd.Series, generatedSeries_c)):
        colB = colB.name
    timeframe = timeframe or active.timeframe
    indexA = timeframe.df.columns.get_loc(colA)
    indexB = timeframe.df.columns.get_loc(colB)
    dummy_source = pd.Series([np.nan], name=f"{indexA}&{indexB}") # must have len of at least 1
    return timeframe.calcGeneratedSeries( 'divi', dummy_source, 1, _generatedseries_calculate_divide_series, param=(colA, colB))

def SeriesConstOp(col: str|pd.Series|generatedSeries_c, constant: float, op_func, op_label: str = "", timeframe=None) -> generatedSeries_c:
    """
    Apply a custom binary operation between a column and a constant.
    
    Args:
        col (str): Name of the column (Series) in the DataFrame.
        constant (float): The constant value.
        op_func (Callable): The binary operation function, e.g., lambda s, c: s + c.
        op_label (str): Optional label to differentiate multiple uses of same op_func.
        timeframe: The timeframe context (defaults to active.timeframe).

    Returns:
        generatedSeries_c
    """
    if isinstance(col,(pd.Series, generatedSeries_c)):
        col = col.name
    timeframe = timeframe or active.timeframe
    source = timeframe.df[col]
    name_suffix = f"{col}{op_label or 'op'}_{str(constant).replace('.', '_')}"
    return timeframe.calcGeneratedSeries(f"scop{name_suffix}", source, 1, _generatedseries_calculate_series_const_lambda, (constant, op_func))

def addSeriesConst(col:str|pd.Series|generatedSeries_c, constant:(float|int), timeframe=None)->generatedSeries_c:
    return SeriesConstOp(col, constant, lambda s, c: s + c, "add", timeframe)

def subtractSeriesConst(col:str|pd.Series|generatedSeries_c, constant:(float|int), timeframe=None)->generatedSeries_c:
    return SeriesConstOp(col, constant, lambda s, c: s - c, "sub", timeframe)

def mulSeriesConst(col:str|pd.Series|generatedSeries_c, constant:(float|int), timeframe=None)->generatedSeries_c:
    return SeriesConstOp(col, constant, lambda s, c: s * c, "mul", timeframe)

def divSeriesConst(col:str|pd.Series|generatedSeries_c, constant:(float|int), timeframe=None)->generatedSeries_c:
    return SeriesConstOp(col, constant, lambda s, c: s / c if c != 0 else np.nan, "div", timeframe)

def powSeriesConst(col:str|pd.Series|generatedSeries_c, constant:(float|int), timeframe=None)->generatedSeries_c:
    return SeriesConstOp(col, constant, lambda s, c: np.power(s, c), "pow", timeframe)

def compareSeries(colA: str, colB: str, op_func, op_label: str = "", timeframe=None):
    timeframe = timeframe or active.timeframe
    df = timeframe.df

    if colA not in df.columns or colB not in df.columns:
        raise ValueError(f"compareSeries: Columns '{colA}' or '{colB}' not found")

    indexA = df.columns.get_loc(colA)
    indexB = df.columns.get_loc(colB)
    name = f"{indexA}-{indexB}"

    return timeframe.calcGeneratedSeries(
        f"cmp{name}_{op_label}",
        df[colA],
        1,
        _generatedseries_calculate_binary_operator,
        (colA, colB, op_func)
    )

# REVERSED!
def ConstSeriesOp(
    col: str | pd.Series | generatedSeries_c,
    constant: float,
    op_func,
    op_label: str = "",
    timeframe=None
) -> generatedSeries_c:
    """
    Apply a binary operation between a constant and a series, where constant is left operand.

    Args:
        col: Column name, Series, or generatedSeries_c.
        constant: The constant left operand.
        op_func: Callable like lambda c, s: ...
        op_label: Label to distinguish operations.
        timeframe: Optional TimeFrame context (defaults to active.timeframe).

    Returns:
        generatedSeries_c
    """
    if isinstance(col, (pd.Series, generatedSeries_c)):
        col = col.name

    timeframe = timeframe or active.timeframe

    if col not in timeframe.df.columns:
        raise ValueError(f"ConstSeriesOp: Column '{col}' not found in DataFrame.")

    source = timeframe.df[col]
    clean_constant = str(constant).replace(".", "_")
    name_suffix = f"{op_label}_{clean_constant}_{col}"

    return timeframe.calcGeneratedSeries(
        f"csop_{name_suffix}",
        source,
        1,
        _generatedseries_calculate_const_series_lambda,
        (constant, op_func)
    )

def subtractFromConst(constant:(float|int), col:str|pd.Series|generatedSeries_c, timeframe=None)->generatedSeries_c:
    return ConstSeriesOp(col, constant, lambda c, s: c - s, "subFrom", timeframe)

def divFromConst(constant:(float|int), col:str|pd.Series|generatedSeries_c, timeframe=None)->generatedSeries_c:
    return ConstSeriesOp(col, constant, lambda c, s: c / s.replace(0, np.nan), "divFrom", timeframe)

def constPowSeries(constant:(float|int), col:str|pd.Series|generatedSeries_c, timeframe=None)->generatedSeries_c:
    return ConstSeriesOp(col, constant, lambda c, s: np.power(c, s), "powFrom", timeframe)

def constMinusLog(col:str|pd.Series|generatedSeries_c, constant=np.e, timeframe=None)->generatedSeries_c:
    return ConstSeriesOp(col, constant, lambda c, s: np.log(c / s.replace(0, np.nan)), "logFrom", timeframe)

def addToConst(constant:(float|int), col:str|pd.Series|generatedSeries_c, timeframe=None)->generatedSeries_c:
    return ConstSeriesOp(col, constant, lambda c, s: c + s, "addFrom", timeframe)

def multFromConst(constant:(float|int), col:str|pd.Series|generatedSeries_c, timeframe=None)->generatedSeries_c:
    return ConstSeriesOp(col, constant, lambda c, s: c * s, "mulFrom", timeframe)

def compareSeriesConst(col: str | pd.Series | generatedSeries_c, constant: float, op_func, op_label: str = "", timeframe=None):
    if isinstance(col, (pd.Series, generatedSeries_c)):
        col = col.name
    timeframe = timeframe or active.timeframe
    df = timeframe.df

    if col not in df.columns:
        raise ValueError(f"compareSeriesConst: Column '{col}' not found")

    name_suffix = f"{col}{op_label}_{str(constant).replace('.', '_')}"

    return timeframe.calcGeneratedSeries(
        f"cmpc{name_suffix}",
        df[col],
        1,
        _generatedseries_calculate_series_const_lambda,
        (constant, op_func)
    )



def SMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'sma', source, period, _generatedseries_calculate_sma )

def EMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( "ema", source, period, _generatedseries_calculate_ema )

def DEMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( "dema", source, period, _generatedseries_calculate_dema, always_reset=True )

def WMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( "wma", source, period, _generatedseries_calculate_wma )

def HMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( "hma", source, period, _generatedseries_calculate_hma, always_reset=True )

# def JMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
#     timeframe = timeframe or active.timeframe
#     return timeframe.calcGeneratedSeries( "jma", source, period, pt.jma )

# def KAMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
#     timeframe = timeframe or active.timeframe
#     return timeframe.calcGeneratedSeries( "kama", source, period, pt.kama )

def LINREG( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( "linreg", source, period, _generatedseries_calculate_linreg )

def RSI( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'rsi', source, period, _generatedseries_calculate_rsi )

def DEV( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'dev', source, period, _generatedseries_calculate_dev )

def STDEV( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'stdev', source, period, _generatedseries_calculate_stdev )

def RMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'rma', source, period, _generatedseries_calculate_rma, always_reset=True )

def WPR( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'wpr', source, period, _generatedseries_calculate_williams_r )

def TR( period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'tr', pd.Series([pd.NA] * period, name = 'tr'), period, _generatedseries_calculate_tr )

def ATR2( period:int, timeframe = None )->generatedSeries_c: # The other one using pt is much faster
    timeframe = timeframe or active.timeframe
    tr = timeframe.calcGeneratedSeries( 'tr', timeframe.df['close'], period, _generatedseries_calculate_tr )
    return timeframe.calcGeneratedSeries( 'atr', tr.series(), period, _generatedseries_calculate_rma )

def ATR( period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'atr', timeframe.df['close'], period, _generatedseries_calculate_atr )

def SLOPE( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'slope', source, period, _generatedseries_calculate_slope, always_reset=True )

def VHMA(source: pd.Series, period: int, timeframe=None)->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('vhma', source, period, _generatedseries_calculate_vhma, always_reset = True)

def BIAS( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'bias', source, period, _generatedseries_calculate_bias )

def CCI( period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'cci', timeframe.df['close'], period, _generatedseries_calculate_cci )

def CFO( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'cfo', source, period, _generatedseries_calculate_cfo )

def FWMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'fwma', source, period, _generatedseries_calculate_fwma )

def BBu( source:pd.Series, period:int, mult:float, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'bbu', source, period, _generatedseries_calculate_bbupper, mult )

def BBl( source:pd.Series, period:int, mult:float, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'bbl', source, period, _generatedseries_calculate_bblower, mult )

def IFTrsi( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    rsi = timeframe.calcGeneratedSeries( 'rsi', source, period, _generatedseries_calculate_rsi )
    return timeframe.calcGeneratedSeries( 'iftrsi', timeframe.df['close'], period, _generatedseries_calculate_inverse_fisher_rsi )

def Fisher( period:int, signal:float=None, timeframe = None )->tuple[generatedSeries_c, generatedSeries_c]:
    timeframe = timeframe or active.timeframe
    fish = timeframe.calcGeneratedSeries( 'fisher', timeframe.df['close'], period, _generatedseries_calculate_fisher )
    sig = timeframe.calcGeneratedSeries( 'fishersig', timeframe.df['close'], period, _generatedseries_calculate_fisher_signal, signal )
    return fish, sig
    
def AO( period: int = 0, fast: int = 5, slow: int = 34, timeframe = None ) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    param = (fast, slow)
    return timeframe.calcGeneratedSeries('ao', timeframe.df['close'], period, _generatedseries_calculate_ao, param)

def BRAR( period:int, signal:float=None, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    br = timeframe.calcGeneratedSeries( 'br', timeframe.df['close'], period, _generatedseries_calculate_br )
    ar = timeframe.calcGeneratedSeries( 'ar', timeframe.df['close'], period, _generatedseries_calculate_ar )
    return br, ar

def CG( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'cg', source, period, _generatedseries_calculate_cg )

def barsSinceSeries(source: pd.Series, period: int, timeframe=None) -> generatedSeries_c:
    import inspect
    # Get caller info by going up 2 levels in the stack
    caller_frame = inspect.currentframe().f_back
    frame_info = inspect.getframeinfo(caller_frame)
    caller_id = f"{frame_info.function[:5]}{frame_info.lineno}"

    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('barsSince'+caller_id, source, period, _generatedseries_calculate_barssince)

def barsWhileTrueSeries(source: pd.Series, period: int = None, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('barsWhileTrue', source, period, _generatedseries_calculate_barswhiletrue)

def barsWhileFalseSeries(source: pd.Series, period: int = None, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('barsWhileFalse', source, period, _generatedseries_calculate_barswhilefalse)

def indexWhenTrueSeries(source: pd.Series, period: int = None, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('indexwhentrue_series', source, period, _generatedseries_calculate_indexwhentrue)

def indexWhenFalseSeries(source: pd.Series, period: int, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
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


def MACD(source: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9, timeframe=None) -> tuple[generatedSeries_c, generatedSeries_c, generatedSeries_c]:
    """
    Returns the MACD line, Signal line, and Histogram for given source and periods.
    Args:
        source (pd.Series): The price series (e.g. close).
        fast (int): Fast EMA period.
        slow (int): Slow EMA period.
        signal (int): Signal EMA period.
        timeframe: The timeframe context (defaults to active.timeframe).
    Returns:
        Tuple of (MACD line, Signal line, Histogram) as generatedSeries_c objects.
    """
    if timeframe is None:
        timeframe = active.timeframe
    # Calculate the fast and slow EMAs
    fast_ema = EMA(source, fast, timeframe)
    slow_ema = EMA(source, slow, timeframe)

    # MACD line: difference between fast and slow EMA
    fastminusslow = fast_ema - slow_ema # Same as # fastminusslow = subtractSeries(fast_ema.name, slow_ema.name)
    macd_line = timeframe.calcGeneratedSeries('macd_line', fastminusslow.series(), 1, lambda s, p, d, x: s)

    # Signal line: EMA of the MACD line
    signal_line = EMA(macd_line.series(), signal, timeframe)

    # Histogram: MACD line - Signal line
    macdminussignal = macd_line - signal_line # Same as # macdminussignal = subtractSeries(macd_line.name, signal_line.name)
    hist = timeframe.calcGeneratedSeries('macd_hist', macdminussignal.series(), 1, lambda s, p, d, x: s)

    return macd_line, signal_line, hist



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
        self.current_HH = None
        self.current_LL = None
        self.current_HH_index = None
        self.current_LL_index = None
        self.last_pivot_price = None
        self.last_pivot_move = None

        self.is_pivot = False
        self.pivots:list[pivot_c] = []
        
    def process_candle(self, index: int, high: float, low: float)->bool:
        self.is_pivot = False
        if self.barindex >= index:
            return False
        self.barindex = index

        # Initialize on first candle
        if self.current_HH is None:
            self.current_HH = high
            self.current_LL = low
            self.current_HH_index = index
            self.current_LL_index = index
            return False
            
        if self.trend > 0:
            # Track new high if we make one
            if high >= self.current_HH:
                self.current_HH = high
                self.current_HH_index = index
                return False
            
            # Check for potential reversal
            min_range_threshold = self.current_HH * (1 - self.min_range_pct * 0.01)
            
            if low < min_range_threshold:
                current_reversal = self.current_HH - low
                
                # If we have a previous pivot to compare to
                if self.last_pivot_price is not None:
                    reversal_threshold = self.last_pivot_move * (self.reversal_pct * 0.01)
                    if current_reversal >= reversal_threshold:
                        # Confirmed down pivot
                        self.newPivot(self.current_HH_index, self.trend, self.current_HH)
                        self.trend = c.SHORT
                        self.last_pivot_price = self.current_HH
                        self.last_pivot_move = current_reversal
                        self.current_LL = low
                        self.current_LL_index = index
                        return True
                else:
                    # First pivot, only use min_range
                    self.newPivot( self.current_HH_index, self.trend, self.current_HH)
                    self.trend = c.SHORT
                    self.last_pivot_price = self.current_HH
                    self.last_pivot_move = current_reversal
                    self.current_LL = low
                    self.current_LL_index = index
                    return True
                    
        else:  # Downtrend
            # Track new low if we make one
            if low <= self.current_LL:
                self.current_LL = low
                self.current_LL_index = index
                return False
                
            # Check for potential reversal
            min_range_threshold = self.current_LL * (1 + self.min_range_pct * 0.01)
            
            if high > min_range_threshold:
                current_reversal = high - self.current_LL
                
                # If we have a previous pivot to compare to
                if self.last_pivot_price is not None:
                    reversal_threshold = self.last_pivot_move * (self.reversal_pct * 0.01)
                    if current_reversal >= reversal_threshold:
                        # Confirmed up pivot
                        self.newPivot( self.current_LL_index, self.trend, self.current_LL)
                        self.trend = c.LONG
                        self.last_pivot_price = self.current_LL
                        self.last_pivot_move = current_reversal
                        self.current_HH = high
                        self.current_HH_index = index
                        return True
                else:
                    # First pivot, only use min_range
                    self.newPivot( self.current_LL_index, self.trend, self.current_LL)
                    self.trend = c.LONG
                    self.last_pivot_price = self.current_LL
                    self.last_pivot_move = current_reversal
                    self.current_HH = high
                    self.current_HH_index = index
                    return True
        
        return False

    def newPivot(self, index, type, price):
        pivot = pivot_c(
                index=index,
                type=type,
                price=price,
                timestamp=int(active.timeframe.df.iat[index, c.DF_TIMESTAMP])
                )
        if len(self.pivots) >= 500:
            self.pivots = self.pivots[1:] + [pivot]  # Slicing out the oldest element
        else:
            self.pivots.append(pivot)

    def getLast(self, type:int = None, since:int = None)->pivot_c|None:
        if since is None:since = active.barindex
        for pivot in reversed(self.pivots):
            if pivot.index >= since:
                continue
            if type is not None and type != pivot.type:
                continue
            return pivot
        return None
    
    def update(self, high: pd.Series, low: pd.Series):
        if( not active.timeframe.jumpstart ):
            self.is_pivot = self.process_candle(active.barindex, high[active.barindex], low[active.barindex])

        
pivotsNow:pivots_c = None
def pivots( high:pd.Series, low:pd.Series, amplitude: float = 1.0, reversal_percent: float = 32.0 )->pivots_c:
    global pivotsNow
    if pivotsNow == None:
        pivotsNow = pivots_c(amplitude, reversal_percent)

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
