
from typing import Union
talib_available = False
talib = None
# try:
#     import talib
#     talib_available = True
# except ImportError:
#     talib_available = False
#     print("Talib not available")

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import time

from .nameseries import series_c
from .constants import c
from . import active
from . import tools
    

# Dynamically set __all__ to include all names that don't start with '_' and are not in _exclude
_exclude = ['active']
__all__ = [name for name in globals() if not (name.startswith('_') or name in _exclude)]




# #
# # GENERATED SERIES : These are series of values that are calculated always using the same formula
# #



# --- Optimized Helper for rolling window operations ---
def _rolling_window_apply_optimized(arr: np.ndarray, window: int, func) -> np.ndarray:
    """
    Applies a function over a rolling window of a 1D NumPy array using sliding_window_view.
    Pads the beginning with NaNs to match the input array's length.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=np.float64)
    
    n = len(arr)
    if window < 1 or window > n:
        return np.full_like(arr, np.nan)

    windows = sliding_window_view(arr, window_shape=window)
    applied_values = func(windows) 
    
    result = np.concatenate((np.full(window - 1, np.nan), applied_values))
    
    return result



NumericScalar = Union[float, int]
OperandType = Union[np.ndarray, NumericScalar]

def _generatedseries_calculate_add_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return source + param

def _generatedseries_calculate_subtract_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return source - param

def _generatedseries_calculate_multiply_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return source * param

def _generatedseries_calculate_divide_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return source / param

def _generatedseries_calculate_power_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.power(source, param)

def _generatedseries_calculate_min_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.minimum(source, param)

def _generatedseries_calculate_max_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.maximum(source, param)

def _generatedseries_calculate_equal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return source == param

def _generatedseries_calculate_notequal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return source != param

def _generatedseries_calculate_greater_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return source > param

def _generatedseries_calculate_greaterorequal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return source >= param

def _generatedseries_calculate_less_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return source < param

def _generatedseries_calculate_lessequal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return source <= param

def _generatedseries_calculate_logical_not(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    return ~source


##### scalars by series


def _generatedseries_calculate_scalar_add_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param + source # Note: param is the scalar, source is the series

def _generatedseries_calculate_scalar_subtract_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param - source

def _generatedseries_calculate_scalar_multiply_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param * source

def _generatedseries_calculate_scalar_divide_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param / source

def _generatedseries_calculate_scalar_power_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return np.power(param, source) # Note the order: scalar (param) first, then series (source)

def _generatedseries_calculate_scalar_min_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return np.minimum(param, source)

def _generatedseries_calculate_scalar_max_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return np.maximum(param, source)

def _generatedseries_calculate_scalar_equal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param == source

def _generatedseries_calculate_scalar_notequal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param != source

def _generatedseries_calculate_scalar_greater_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param > source

def _generatedseries_calculate_scalar_greaterorequal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param >= source

def _generatedseries_calculate_scalar_less_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param < source

def _generatedseries_calculate_scalar_lessequal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param <= source




################################ ANALYSIS TOOLS #####################################


# _highest250. Elapsed time: 0.00 seconds
def _generatedseries_calculate_highest(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    """
    Calculates the highest value over a specified period using NumPy.
    """
    if talib_available:
        return talib.MAX(source, period)
    source = np.asarray(source, dtype=np.float64)
    return _rolling_window_apply_optimized(source, period, lambda x: np.max(x, axis=1))

def _generatedseries_calculate_lowest(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    """
    Calculates the lowest value over a specified period using NumPy.
    """
    if talib_available:
        return talib.MIN(source, period)
    source = np.asarray(source, dtype=np.float64)
    return _rolling_window_apply_optimized(source, period, lambda x: np.min(x, axis=1))

# _highestbars250. Elapsed time: 0.01 seconds
def _generatedseries_calculate_highestbars(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    # if talib_available:
    #     return talib.MAXINDEX(source, period)
    source = np.asarray(source, dtype=np.float64)

    return _rolling_window_apply_optimized(source, period, lambda x: (period - 1) - np.argmax(x, axis=1))

# _lowestbars250. Elapsed time: 0.01 seconds
def _generatedseries_calculate_lowestbars(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    # if talib_available:
    #     return talib.MININDEX(source, period)
    source = np.asarray(source, dtype=np.float64)

    return _rolling_window_apply_optimized(source, period, lambda x: (period - 1) - np.argmin(x, axis=1))

# _falling250. Elapsed time: 0.01 seconds
def _generatedseries_calculate_falling(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    n = len(source)

    if period < 1 or period > n:
        return np.full_like(source, np.nan, dtype=bool)

    diffs = np.concatenate(([np.nan], np.diff(source)))
    
    window_for_diffs = period - 1

    if window_for_diffs < 1: # If period is 1, a single value is trivially "falling" if not NaN
        result = ~np.isnan(source) # If period is 1, it's falling if it's not NaN
        return result.astype(bool)

    if len(diffs[1:]) < window_for_diffs:
        return np.full_like(source, np.nan, dtype=bool)

    windows_of_diffs = sliding_window_view(diffs[1:], window_shape=window_for_diffs)

    # Check if all elements in each window are strictly negative
    all_negative = np.all(windows_of_diffs < 0, axis=1)

    result_array = np.full(n, np.nan)
    result_array[period - 1:] = all_negative

    return result_array.astype(bool)

# _rising250. Elapsed time: 0.01 seconds
def _generatedseries_calculate_rising(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    """
    Calculates a boolean series indicating if the source has been strictly rising
    over the given period.
    """
    source = np.asarray(source, dtype=np.float64)
    n = len(source)

    if period < 1 or period > n:
        return np.full_like(source, np.nan, dtype=bool) # Use bool dtype for boolean results

    diffs = np.concatenate(([np.nan], np.diff(source)))

    window_for_diffs = period - 1
    
    if window_for_diffs < 1: # If period is 1, a single value is trivially "rising" if not NaN
        result = ~np.isnan(source) # If period is 1, it's rising if it's not NaN
        return result.astype(bool)

    # Create sliding window view on `diffs` starting from the second element
    if len(diffs[1:]) < window_for_diffs:
        return np.full_like(source, np.nan, dtype=bool)

    windows_of_diffs = sliding_window_view(diffs[1:], window_shape=window_for_diffs)
    all_positive = np.all(windows_of_diffs > 0, axis=1)

    result_array = np.full(n, np.nan)
    result_array[period - 1:] = all_positive

    # Convert to boolean, NaNs will remain as NaN, althought they will be converted to float64 in the dataset
    return result_array.astype(bool)

#
def _generatedseries_calculate_barssince(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    # Get array of indices where condition is True
    true_indices = np.where(series)[0]
    if len(true_indices) == 0:
        return np.full_like(series, np.nan, dtype=np.float64)

    all_indices = np.arange(len(series))
    insertions = np.searchsorted(true_indices, all_indices, side='right') - 1
    result = np.full(len(series), np.nan, dtype=np.float64)
    valid_mask = insertions >= 0
    result[valid_mask] = all_indices[valid_mask] - true_indices[insertions[valid_mask]]

    if period is not None:
        result[result > period] = np.nan

    return result

#
def _generatedseries_calculate_indexwhentrue(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    length = len(series)
    out = np.full(length, np.nan, dtype=np.float64)
    last_true = -1
    for i, val in enumerate(series):
        if val:
            last_true = i
        if last_true != -1:
            out[i] = last_true
    return out

#
def _generatedseries_calculate_indexwhenfalse(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    length = len(series)
    out = np.full(length, np.nan, dtype=np.float64)
    last_false = -1
    for i, val in enumerate(series):
        if not val:
            last_false = i
        if last_false != -1:
            out[i] = last_false
    return out

#
def _generatedseries_calculate_barswhiletrue(series: np.ndarray, period: int = None, dataset: np.ndarray = None, cindex:int = None, param=None) -> np.ndarray:
    arr = series.astype(bool)
    counts = np.zeros_like(arr, dtype=int)
    c = 0
    for i, val in enumerate(arr):
        c = c + 1 if val else 0
        if period:
            c = min(c, period)
        counts[i] = c
    return counts.astype(np.float64)  # for consistency with other outputs

#
def _generatedseries_calculate_barswhilefalse(series: np.ndarray, period: int = None, dataset: np.ndarray = None, cindex:int = None, param=None) -> np.ndarray:
    length = len(series)
    max_lookback = period if (period is not None and period <= length) else length
    out = np.zeros(length, dtype=int)
    count = 0
    for i in range(length):
        val = series[i]
        if not val:
            count += 1
        else:
            count = 0
        if period:
            count = min(count, period)
        out[i] = count
    return out.astype(np.float64)



########################### INDICATORS #################################

#
def _generatedseries_calculate_sma(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    if talib_available:
        return talib.SMA(source, period)
    
    source = np.asarray(source, dtype=np.float64)
    if period < 1 or period > source.shape[0]:
        return np.full_like(source, np.nan)

    sma = np.full_like(source, np.nan)
    cumsum = np.cumsum(np.insert(source, 0, 0))
    sma[period-1:] = (cumsum[period:] - cumsum[:-period]) / period
    return sma

# _ema_250. Elapsed time: 0.03 seconds (a little slow, but it's the only reliable one. Talib is also unreliable)
def _generatedseries_calculate_ema(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    if talib_available:
        return talib.EMA(series, period)
    length = len(series)
    if length == 0 or period < 1:
        return np.array([], dtype=np.float64)

    # Initialize output array
    result = np.full(length, np.nan, dtype=np.float64)

    # Find first non-NaN value
    valid_idx = np.where(~np.isnan(series))[0]
    if len(valid_idx) == 0:
        return result
    start_idx = valid_idx[0]

    # Set initial EMA to first non-NaN value
    result[start_idx] = series[start_idx]

    # Smoothing factor
    alpha = 2 / (period + 1)
    beta = 1 - alpha

    # Compute EMA iteratively
    for i in range(start_idx + 1, length):
        if not np.isnan(series[i]):
            result[i] = alpha * series[i] + beta * result[i - 1]
        else:
            result[i] = np.nan

    return result

#
def _generatedseries_calculate_dema(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    length = len(series)
    if length < period:
        return np.full(length, np.nan)

    # Calculate first EMA
    ema1 = _generatedseries_calculate_ema(series, period, dataset)

    # Calculate EMA of EMA
    ema2 = _generatedseries_calculate_ema(ema1, period, dataset)

    # Calculate DEMA: 2 * EMA1 - EMA2
    dema = 2 * ema1 - ema2
    return dema

# _rma250. Elapsed time: 0.01 seconds
def _generatedseries_calculate_rma(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    length = len(series)
    if length < period:
        return np.full(length, np.nan)

    # Initialize output array
    rma = np.full(length, np.nan)

    # Compute initial SMA using sliding_window_view
    windows = sliding_window_view(series, window_shape=period)
    rma[period - 1] = np.mean(windows[0], axis=-1)

    # Compute RMA iteratively
    alpha = 1.0 / period
    one_minus_alpha = 1.0 - alpha
    for i in range(period, length):
        rma[i] = alpha * series[i] + one_minus_alpha * rma[i - 1]

    return rma

# _wma250. Elapsed time: 0.02 seconds (talib 00.00 seconds)
def _generatedseries_calculate_wma(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    if talib_available:
        return talib.WMA(series, period)
    
    length = len(series)
    if length < period:
        return np.full(length, np.nan)

    # Precompute weights and their sum
    weights = np.arange(1, period + 1, dtype=np.float64)
    weight_sum = period * (period + 1) / 2  # Sum of weights: 1 + 2 + ... + period

    # Create rolling windows
    windows = sliding_window_view(series, window_shape=period)

    # Compute WMA for all windows
    weighted_sums = np.sum(windows * weights, axis=1)  # Element-wise multiplication and sum
    wma = weighted_sums / weight_sum

    # Pad with NaNs for the first period - 1 values
    result = np.full(length, np.nan)
    result[period - 1:] = wma

    return result

# _linreg250. Elapsed time: 0.02 seconds (talib 0.01 seconds)
def _generatedseries_calculate_linreg(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    if talib_available:
        return talib.LINEARREG(series, period)

    length = len(series)
    if length < period:
        return np.full(length, np.nan)

    # Initialize output array
    linreg = np.full(length, np.nan)

    # Create sliding windows
    windows = sliding_window_view(series, window_shape=period)

    # Time indices for regression
    t = np.arange(period, dtype=np.float64)
    t_sum = np.sum(t)
    t_sq_sum = np.sum(t * t)
    n = period

    # Compute sums for regression
    y_sum = np.sum(windows, axis=1)
    ty_sum = np.sum(windows * t, axis=1)

    # Compute slope (b) and intercept (a) vectorized
    denominator = n * t_sq_sum - t_sum ** 2
    b = (n * ty_sum - t_sum * y_sum) / denominator
    a = (y_sum - b * t_sum) / n

    # Forecasted price at t = period - 1
    linreg[period - 1:] = a + b * (period - 1)

    return linreg

# _bias250. Elapsed time: 0.00 seconds
def _generatedseries_calculate_bias(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    n = len(source)

    if period < 1 or period > n:
        return np.full_like(source, np.nan)

    # Calculate the Simple Moving Average (SMA) of the source price
    sma_values = _generatedseries_calculate_sma(source, period, dataset)

    # Initialize bias array with NaNs
    bias = np.full_like(source, np.nan)

    # Identify valid indices where SMA is not NaN and not zero to avoid division by zero
    valid_indices = np.where((~np.isnan(sma_values)) & (sma_values != 0))

    # Apply the BIAS formula # bias = ((source - sma_values) / sma_values) * 100
    if len(valid_indices[0]) > 0:
        bias[valid_indices] = ((source[valid_indices] - sma_values[valid_indices]) / sma_values[valid_indices]) * 100

    return bias

# _cci250. Elapsed time: 0.04 seconds (talib 0.01 seconds)
def _generatedseries_calculate_cci(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    length = len(series)
    if length < period:
        return np.full(length, np.nan)
    
    if talib_available:
        return talib.CCI(dataset[:, c.DF_HIGH], dataset[:, c.DF_LOW], dataset[:, c.DF_CLOSE], period)

    # Compute Typical Price
    tp = (dataset[:, c.DF_HIGH] + dataset[:, c.DF_LOW] + dataset[:, c.DF_CLOSE]) / 3.0

    # Create sliding windows
    tp_windows = sliding_window_view(tp, window_shape=period)

    # Compute SMA
    sma = np.mean(tp_windows, axis=1)

    # Compute MAD
    mad = np.mean(np.abs(tp_windows - sma[:, np.newaxis]), axis=1)

    # Compute CCI
    cci = np.full(length, np.nan)
    denominator = 0.015 * mad
    cci[period - 1:] = np.where(denominator > 1e-10, (tp[period - 1:] - sma) / denominator, np.nan)

    return cci

# 0.02
def _generatedseries_calculate_cfo(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    length = len(series)
    if length < period:
        return np.full(length, np.nan)

    # Initialize output array
    cfo = np.full(length, np.nan)

    # Create sliding windows
    windows = sliding_window_view(series, window_shape=period)  # Shape: (length - period + 1, period)

    # Time indices for regression
    t = np.arange(period, dtype=np.float64)
    t_sum = np.sum(t)
    t_sq_sum = np.sum(t * t)
    n = period

    # Compute sums for regression
    y_sum = np.sum(windows, axis=1)  # Sum of y_i for each window
    ty_sum = np.sum(windows * t, axis=1)  # Sum of t_i * y_i for each window

    # Compute slope (b) and intercept (a) vectorized
    denominator = n * t_sq_sum - t_sum ** 2
    b = (n * ty_sum - t_sum * y_sum) / denominator  # Slope
    a = (y_sum - b * t_sum) / n  # Intercept

    # Forecasted price at t = period - 1
    forecasts = a + b * (period - 1)

    # Current close prices for valid indices
    closes = series[period - 1:]

    # Compute CFO: ((close - forecast) * 100) / close
    valid_closes = np.abs(closes) > 1e-10
    cfo[period - 1:] = np.where(valid_closes, ((closes - forecasts) * 100) / closes, np.nan)

    return cfo

# _cmo250. Elapsed time: 0.01 seconds
def _generatedseries_calculate_cmo(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    if talib_available:
        return talib.CMO( source, period )

    source = np.asarray(source, dtype=np.float64)
    n = len(source)

    if period < 1 or period > n:
        return np.full_like(source, np.nan)

    # Calculate price changes (diff). Prepend a NaN.
    changes = np.concatenate(([np.nan], np.diff(source)))

    # Separate positive and negative changes
    sum_up_values = np.where(changes > 0, changes, 0.0)
    sum_down_values = np.where(changes < 0, np.abs(changes), 0.0)
    # CORRECTED LINES: Wrap np.sum(x, axis=1) in a lambda function
    rolling_sum_up = _rolling_window_apply_optimized(np.nan_to_num(sum_up_values, nan=0.0), period, lambda x: np.sum(x, axis=1))
    rolling_sum_down = _rolling_window_apply_optimized(np.nan_to_num(sum_down_values, nan=0.0), period, lambda x: np.sum(x, axis=1))
    sum_total = rolling_sum_up + rolling_sum_down

    # Calculate CMO
    cmo = np.full_like(source, np.nan)
    non_zero_total_idx = np.where(sum_total != 0)

    # Apply CMO formula: 100 * ((Sum_Up - Sum_Down) / (Sum_Up + Sum_Down))
    cmo[non_zero_total_idx] = 100 * ((rolling_sum_up[non_zero_total_idx] - rolling_sum_down[non_zero_total_idx]) / sum_total[non_zero_total_idx])
    zero_total_idx = np.where(sum_total == 0)
    cmo[zero_total_idx] = np.where(~np.isnan(sum_total[zero_total_idx]), 0.0, np.nan)

    return cmo

#
def _generatedseries_calculate_fwma(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    length = len(series)
    if length < period:
        return np.full(length, np.nan)

    # Generate Fibonacci weights
    fib = np.zeros(period, dtype=np.float64)
    fib[0] = 1
    if period > 1:
        fib[1] = 1
        for i in range(2, period):
            fib[i] = fib[i-1] + fib[i-2]
    weights = fib[::-1]  # Reverse: [F_n, F_{n-1}, ..., F_1]
    weight_sum = np.sum(weights)

    # Create rolling windows
    windows = sliding_window_view(series, window_shape=period)

    # Compute FWMA: Σ(x_j * w_j) / Σ(w_j)
    weighted_sums = np.sum(windows * weights, axis=1)
    fwma = weighted_sums / weight_sum

    # Pad with NaNs for the first period - 1 values
    result = np.full(length, np.nan)
    result[period - 1:] = fwma

    return result

# _stdev250. Elapsed time: 0.02 seconds (talib 0.00 seconds)
def _generatedseries_calculate_stdev(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    if talib_available:
        return talib.STDDEV(series, period)

    length = len(series)
    if length < period:
        return np.full(length, np.nan)

    # Create rolling windows
    windows = sliding_window_view(series, window_shape=period)

    # Compute sample standard deviation (ddof=1) for each window
    stdev = np.std(windows, axis=1, ddof=1)

    # Pad with NaNs for the first period - 1 values
    result = np.full(length, np.nan)
    result[period - 1:] = stdev

    return result

# _dev250. Elapsed time: 0.03 seconds
def _generatedseries_calculate_dev(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    length = len(series)
    if length < period:
        return np.full(length, np.nan)

    # Create rolling windows
    windows = sliding_window_view(series, window_shape=period)

    # Compute mean for each window
    means = np.mean(windows, axis=1)

    # Compute mean absolute deviation: Σ(|x - mean|) / period
    abs_deviations = np.abs(windows - means[:, np.newaxis])
    dev = np.sum(abs_deviations, axis=1) / period

    # Pad with NaNs for the first period - 1 values
    result = np.full(length, np.nan)
    result[period - 1:] = dev

    return result

# _wpr250. Elapsed time: 0.01 seconds (talib 0.0 secods)
def _generatedseries_calculate_williams_r(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    if talib_available:
        return talib.WILLR(dataset[:, c.DF_HIGH], dataset[:, c.DF_LOW], series, period)
    
    length = dataset.shape[0]
    if length < period:
        return np.full(length, np.nan)

    # Extract high, low, close from dataset
    high = dataset[:, c.DF_HIGH]
    low = dataset[:, c.DF_LOW]
    close = series

    # Compute rolling highest high and lowest low
    high_windows = sliding_window_view(high, window_shape=period)
    low_windows = sliding_window_view(low, window_shape=period)
    highest_high = np.max(high_windows, axis=1)
    lowest_low = np.min(low_windows, axis=1)

    # Compute Williams %R
    numerator = highest_high - close[period - 1:]  # Align close with window ends
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, (numerator / denominator) * -100, np.nan)

    # Pad with NaNs for the first period - 1 values
    result = np.full(length, np.nan)
    result[period - 1:] = williams_r

    return result

# _tr250. Elapsed time: 0.00 seconds 
def _generatedseries_calculate_tr(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    if talib_available:
        return talib.TRANGE(dataset[:, c.DF_HIGH], dataset[:, c.DF_LOW], dataset[:, c.DF_CLOSE])
    
    length = dataset.shape[0]
    if length < 1:
        return np.array([])

    # Extract high, low, close from dataset
    high = dataset[:, c.DF_HIGH]
    low = dataset[:, c.DF_LOW]
    close = dataset[:, c.DF_CLOSE]

    # Compute high - low
    high_low = high - low

    # Compute |high - close_prev| and |low - close_prev|
    close_prev = np.roll(close, 1)  # Shift close by 1
    close_prev[0] = close[0]  # Set first value to avoid undefined close[-1]
    high_close_prev = np.abs(high - close_prev)
    low_close_prev = np.abs(low - close_prev)

    # Compute TR as max(high_low, high_close_prev, low_close_prev)
    tr = np.maximum.reduce([high_low, high_close_prev, low_close_prev])

    return tr

# _atr250. Elapsed time: 0.01 seconds
def _generatedseries_calculate_atr(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    if talib_available:
        return talib.ATR(dataset[:, c.DF_HIGH], dataset[:, c.DF_LOW], dataset[:, c.DF_CLOSE], period)
    
    # Compute RMA of True Range
    tr = _generatedseries_calculate_tr(series, period, dataset, param)
    atr = _generatedseries_calculate_rma(tr, period, dataset, param)

    return atr

# _slope250. Elapsed time: 0.03 seconds (talib 0.01 seconds)
def _generatedseries_calculate_slope(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    if talib_available:
        return talib.LINEARREG_SLOPE(series, period)

    length = len(series)
    if length < period:
        return np.full(length, np.nan)

    # Precompute x and constants
    x = np.arange(period, dtype=np.float64)
    x_mean = (period - 1) / 2  # Mean of 0, 1, ..., period-1
    x_centered = x - x_mean
    denominator = np.sum(x_centered ** 2)  # Σ((x_i - x_mean)^2), constant for all windows

    # Create rolling windows
    windows = sliding_window_view(series, window_shape=period)

    # Compute slopes for all windows
    y = windows  # Shape: (length - period + 1, period)
    y_mean = np.mean(y, axis=1)[:, np.newaxis]  # Shape: (length - period + 1, 1)
    y_centered = y - y_mean  # Shape: (length - period + 1, period)
    numerator = np.sum(y_centered * x_centered, axis=1)  # Shape: (length - period + 1,)
    
    # Compute slopes, handle division by zero
    slopes = np.where(denominator != 0, numerator / denominator, 0.0)

    # Pad with NaNs for the first period - 1 values
    result = np.full(length, np.nan)
    result[period - 1:] = slopes

    return result

# _vhma250. Elapsed time: 0.04 seconds - Needs reset
def _generatedseries_calculate_vhma(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    length = len(series)
    if length < period:
        return np.full(length, np.nan)

    # Step 1: Compute rolling maximum and minimum
    windows = sliding_window_view(series, window_shape=period)
    highest = np.max(windows, axis=1)
    lowest = np.min(windows, axis=1)

    # Pad with NaNs at the beginning to match original length
    highest_padded = np.concatenate([np.full(period - 1, np.nan), highest])
    lowest_padded = np.concatenate([np.full(period - 1, np.nan), lowest])

    # Step 2: Calculate R
    R = highest_padded - lowest_padded

    # Step 3: Compute absolute change
    change = np.abs(np.diff(series, prepend=series[0]))  # Prepend first value to maintain length

    # Step 4: Compute rolling sum of change and vhf
    change_windows = sliding_window_view(change, window_shape=period)
    rolling_sum_change = np.sum(change_windows, axis=1)
    rolling_sum_change_padded = np.concatenate([np.full(period - 1, np.nan), rolling_sum_change])
    
    vhf = R / rolling_sum_change_padded
    vhf = np.nan_to_num(vhf, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN and inf with 0

    # Step 5: Compute vhma iteratively
    vhma = np.full(length, np.nan)
    for i in range(1, length):
        if np.isnan(vhma[i - 1]):
            vhma[i] = series[i]
        else:
            vhma[i] = vhma[i - 1] + (vhf[i] ** 2) * (series[i] - vhma[i - 1])

    return vhma

# _rsi14. Elapsed time: 0.02 seconds
def _generatedseries_calculate_rsi(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    if talib_available:
        return talib.RSI(series, period)

    length = len(series)
    if length < period + 1:
        return np.full(length, np.nan)

    # Step 1: Compute price changes
    delta = np.diff(series, prepend=series[0])  # Prepend first value to maintain length

    # Step 2: Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    # Step 3: Initialize SMMA with simple moving average for first period
    gains_windows = sliding_window_view(gains, window_shape=period)
    losses_windows = sliding_window_view(losses, window_shape=period)
    
    avg_gain_initial = np.mean(gains_windows[0], axis=-1)
    avg_loss_initial = np.mean(losses_windows[0], axis=-1)

    # Initialize arrays for SMMA
    avg_gains = np.full(length, np.nan)
    avg_losses = np.full(length, np.nan)
    
    # Set first valid SMMA value
    avg_gains[period] = avg_gain_initial
    avg_losses[period] = avg_loss_initial

    # Step 4: Compute SMMA iteratively
    alpha = 1.0 / period
    for i in range(period + 1, length):
        avg_gains[i] = alpha * gains[i] + (1 - alpha) * avg_gains[i - 1]
        avg_losses[i] = alpha * losses[i] + (1 - alpha) * avg_losses[i - 1]

    # Step 5: Compute RS and RSI
    rs = np.where(avg_losses > 0, avg_gains / avg_losses, np.inf)  # Handle division by zero
    rsi = 100 - (100 / (1 + rs))
    
    # Ensure NaNs for first period - 1 values
    rsi[:period] = np.nan
    rsi = np.nan_to_num(rsi, nan=np.nan, posinf=np.nan, neginf=np.nan)  # Clean up infs

    return rsi

#
def _generatedseries_calculate_inverse_fisher_rsi(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    # series is already the RSI as np.ndarray
    rsi = series.astype(np.float64)
    v1 = 0.1 * (rsi - 50)

    # Weighted Moving Average (WMA)
    def wma(arr, window):
        weights = np.arange(1, window + 1)
        ret = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            windowed = arr[i - window + 1:i + 1]
            if np.any(np.isnan(windowed)):
                continue
            ret[i] = np.dot(windowed, weights) / weights.sum()
        return ret

    wma_v1 = wma(v1, period)
    v2_clipped = np.clip(wma_v1, -10, 10)
    exp_val = np.exp(2 * v2_clipped)
    iftrsi = (exp_val - 1) / (exp_val + 1)
    return iftrsi

#
def _generatedseries_calculate_fisher(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    """
    Fisher Transform (main line)
    """
    high = dataset[:, c.DF_HIGH]
    low = dataset[:, c.DF_LOW]

    # Use median price for normalization
    med = (high + low) / 2
    length = len(med)
    value = np.full(length, np.nan, dtype=np.float64)
    fish = np.full(length, np.nan, dtype=np.float64)

    for i in range(period-1, length):
        window = med[i - period + 1:i + 1]
        min_ = np.min(window)
        max_ = np.max(window)
        if max_ == min_:
            norm = 0
        else:
            norm = 2 * ((med[i] - min_) / (max_ - min_) - 0.5)
            norm = np.clip(norm, -0.999, 0.999)
        value[i] = norm

    # Fisher Transform
    for i in range(period-1, length):
        prev = fish[i-1] if i > 0 else 0
        fish[i] = 0.5 * np.log((1 + value[i]) / (1 - value[i])) + 0.5 * prev

    return fish

#
def _generatedseries_calculate_fisher_signal(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    """
    Fisher Transform signal line: usually an EMA of Fisher line, default length 9 if not provided
    """
    signal_period = param if param else 9
    fish = _generatedseries_calculate_fisher(series, period, dataset)
    length = len(fish)
    sig = np.full(length, np.nan, dtype=np.float64)
    alpha = 2 / (signal_period + 1)
    for i in range(period-1, length):
        if i == period-1:
            sig[i] = fish[i]
        else:
            sig[i] = alpha * fish[i] + (1 - alpha) * sig[i-1]
    return sig

#
def _generatedseries_calculate_ao(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    """
    Awesome Oscillator: SMA(median_price, fast) - SMA(median_price, slow)
    `param` can optionally override the two SMA lengths as a tuple: (fast, slow)
    """
    fast, slow = (5, 34)
    if isinstance(param, tuple) and len(param) == 2:
        fast, slow = param

    # Use column indices for high and low
    high = dataset[:, c.DF_HIGH]
    low = dataset[:, c.DF_LOW]
    median_price = (high + low) / 2

    def sma(arr, window):
        ret = np.full_like(arr, np.nan, dtype=np.float64)
        if window > len(arr):
            return ret
        cumsum = np.cumsum(np.insert(arr, 0, 0))
        ret[window-1:] = (cumsum[window:] - cumsum[:-window]) / window
        return ret

    sma_fast = sma(median_price, fast)
    sma_slow = sma(median_price, slow)
    ao = sma_fast - sma_slow
    return ao

#
def _generatedseries_calculate_br(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    """
    BR (Buying Pressure Ratio) -- NumPy implementation
    BR = SUM(MAX(high - prev_close, 0), N) / SUM(MAX(prev_close - low, 0), N) * 100
    """
    high = dataset[:, c.DF_HIGH]
    low = dataset[:, c.DF_LOW]
    close = dataset[:, c.DF_CLOSE]

    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan  # No previous close for the first row

    br_num = np.maximum(high - prev_close, 0)
    br_den = np.maximum(prev_close - low, 0)

    def rolling_sum(arr, window):
        ret = np.full_like(arr, np.nan, dtype=np.float64)
        if window > len(arr):
            return ret
        cumsum = np.cumsum(np.insert(arr, 0, 0))
        ret[window-1:] = cumsum[window:] - cumsum[:-window]
        return ret

    sum_num = rolling_sum(br_num, period)
    sum_den = rolling_sum(br_den, period)
    br = (sum_num / sum_den) * 100
    return br

#
def _generatedseries_calculate_ar(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    """
    AR (Active Ratio) -- NumPy implementation
    AR = SUM(high - open, N) / SUM(open - low, N) * 100
    """
    high = dataset[:, c.DF_HIGH]
    low = dataset[:, c.DF_LOW]
    open_ = dataset[:, c.DF_OPEN]

    ar_num = high - open_
    ar_den = open_ - low

    def rolling_sum(arr, window):
        ret = np.full_like(arr, np.nan, dtype=np.float64)
        if window > len(arr):
            return ret
        cumsum = np.cumsum(np.insert(arr, 0, 0))
        ret[window-1:] = cumsum[window:] - cumsum[:-window]
        return ret

    sum_num = rolling_sum(ar_num, period)
    sum_den = rolling_sum(ar_den, period)
    ar = (sum_num / sum_den) * 100
    return ar

#
def _generatedseries_calculate_cg(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    """
    Center of Gravity (CG) oscillator (NumPy version).
    Formula: CG = sum(i * price[i]) / sum(price[i]), over the lookback window.
    """
    arr = series.astype(np.float64)
    length = len(arr)
    cg = np.full(length, np.nan, dtype=np.float64)
    for i in range(period - 1, length):
        window = arr[i - period + 1:i + 1]
        if np.all(np.isnan(window)):
            continue
        weights = np.arange(1, period + 1)[::-1]  # period .. 1
        denominator = np.sum(window)
        if denominator == 0:
            cg[i] = np.nan
        else:
            cg[i] = np.sum(window * weights) / denominator
    return cg

#
def _generatedseries_calculate_stoch_k(source_close: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    """
    Calculates the Stochastic %K line. It adapts to operate on either the full
    historical data (during initialization) or on a recent slice (during updates).

    Args:
        source_close (np.ndarray): The close price series. This can be the full series
                                   during initialization or a recent slice during update.
                                   Its length determines the scope of calculation.
        k_period (int): The lookback period for the %K calculation (e.g., 14 bars).
        dataset (np.ndarray): The full 2D dataset array (e.g., timeframe.dataset).
                              Used to access the high and low columns for the
                              corresponding slice of data being processed.
        param (Any): Unused for standard %K, but kept for signature consistency.

    Returns:
        np.ndarray: The calculated %K line as a 1D NumPy array.
                    Its length will match the length of `source_close`.
                    Returns NaNs if `k_period` cannot be met.
    """
    if talib_available:
        high = dataset[:, c.DF_HIGH]
        low = dataset[:, c.DF_LOW]
        close = dataset[:, c.DF_CLOSE] # HACK: it overrides the 'close' in the source
        k, d = talib.STOCH( high, low, close, fastk_period=period, slowk_period=1 )
        return k

    source_close = np.asarray(source_close, dtype=np.float64)
    current_input_len = len(source_close) # This will be full_len during initialize, and k_period_slice_len during update

    # Validate period relative to the length of the input data being processed
    if period < 1 or current_input_len < period:
        # If the input data is shorter than the period, we cannot calculate a valid window.
        # This handles early bars or very short slices during incremental updates.
        return np.full_like(source_close, np.nan)

    # --- Determine the relevant slice of high/low values from the full dataset ---
    full_dataset_len = len(dataset)
    
    # Calculate the starting index in the full `dataset` that corresponds to
    # the beginning of the `source_close` array currently being processed.
    # We assume `source_close` is either the complete column or the latest `current_input_len` elements.
    start_index_in_dataset = full_dataset_len - current_input_len
    
    # Defensive check: if `current_input_len` somehow exceeds `full_dataset_len`
    # (shouldn't happen in typical use with `period_slice`), reset `start_index`
    if start_index_in_dataset < 0:
        start_index_in_dataset = 0 

    # Extract the corresponding slices for high and low from the full `dataset`.
    # These slices (`high_values_slice`, `low_values_slice`) will now have
    # the exact same length as `source_close` (`current_input_len`).
    high_values_slice = dataset[start_index_in_dataset : start_index_in_dataset + current_input_len, c.DF_HIGH]
    low_values_slice = dataset[start_index_in_dataset : start_index_in_dataset + current_input_len, c.DF_LOW]

    # --- Calculations now operate on consistently sized (and potentially shorter) arrays ---

    # Calculate Highest High (HH) and Lowest Low (LL) over the `k_period`
    # `_rolling_window_apply_optimized` will produce an array of length `current_input_len`
    hh_values = _rolling_window_apply_optimized(high_values_slice, period, lambda x: np.max(x, axis=1))
    ll_values = _rolling_window_apply_optimized(low_values_slice, period, lambda x: np.min(x, axis=1))

    # Initialize the %K array with NaNs, matching the `source_close` length
    k_line = np.full_like(source_close, np.nan)

    # Calculate %K: ((Close - LL) / (HH - LL)) * 100
    # Calculate the range (difference between HH and LL)
    diff_hl = hh_values - ll_values

    # Find valid indices where the calculation can be performed:
    # 1. `diff_hl` is not zero (to avoid division by zero).
    # 2. `diff_hl` is not NaN.
    # 3. `source_close` is not NaN.
    # 4. `ll_values` is not NaN.
    # All operands now have the same length (`current_input_len`), resolving the `ValueError`.
    valid_indices = np.where(
        (diff_hl != 0) & 
        (~np.isnan(diff_hl)) & 
        (~np.isnan(source_close)) & 
        (~np.isnan(ll_values))
    )

    if valid_indices[0].size > 0:
        k_line[valid_indices] = (
            (source_close[valid_indices] - ll_values[valid_indices]) / diff_hl[valid_indices]
        ) * 100

    # Handle cases where the high-low range (diff_hl) is zero.
    # Typically, if the range is zero and the close is at the low, %K is 0.
    # If range is zero and close is at the high, %K is 100 (which it must be if close == low).
    zero_range_indices = np.where(
        (diff_hl == 0) & 
        (~np.isnan(source_close)) & 
        (~np.isnan(ll_values)) # Check if ll_values is valid here too
    )
    if zero_range_indices[0].size > 0:
        # If the range is zero, the close, high, and low are all the same value.
        # Conventionally, %K is often set to 0.0 or 100.0, or even NaN.
        # Setting to 0.0 if close is equal to LL (which it must be if diff_hl is 0)
        k_line[zero_range_indices] = 0.0 
        # Alternatively, you might set it to 100.0 if you consider it at the top of a zero-range.
        # Some implementations prefer NaN in this specific case. For most common trading, 0 or 100 is seen.
        # np.where(source_close[zero_range_indices] == ll_values[zero_range_indices], 0.0, 100.0) # If close could be different, but diff_hl=0 makes this unlikely

    return k_line


def _generatedseries_calculate_obv(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    """
    Calculate On-Balance Volume (OBV) using numpy, optimized for full calculation.

    Args:
        series (np.ndarray): Placeholder (unused, for compatibility).
        period (int): Unused (OBV is cumulative, no lookback).
        dataset (np.ndarray): 2D array with columns [..., c.DF_CLOSE, ..., c.DF_VOLUME, ...].
        param (tuple): (close_col_idx, volume_col_idx) for column indices.

    Returns:
        np.ndarray: OBV values, cumulative sum of signed volume.
    """
    if dataset.shape[0] == 0:
        return np.array([], dtype=np.float64)

    try:
        close = source # dataset[:, c.DF_CLOSE]
        volume = dataset[:, c.DF_VOLUME]
        length = len(close)

        # Initialize output
        result = np.full(length, np.nan, dtype=np.float64)
        if length < 1:
            return result

        # Set initial OBV to 0
        result[0] = 0 if not np.isnan(volume[0]) and not np.isnan(close[0]) else np.nan

        if length < 2:
            return result

        # Compute price direction: sign(close[i] - close[i-1])
        close_diff = np.diff(close)
        direction = np.sign(close_diff)

        # Compute signed volume
        signed_volume = np.zeros(length, dtype=np.float64)
        signed_volume[1:] = direction * volume[1:]

        # Handle NaNs in close or volume
        valid_mask = ~np.isnan(close) & ~np.isnan(volume)
        signed_volume[~valid_mask] = 0  # Set invalid to 0 for cumsum

        # Cumulative sum for OBV
        result = np.cumsum(signed_volume, dtype=np.float64)

        # Set NaN where close or volume is invalid
        result[~valid_mask] = np.nan

        return result
    except (IndexError, ValueError):
        return np.full(len(dataset), np.nan, dtype=np.float64)
    



class generatedSeries_c:
    def __init__(self, type: str, source: np.ndarray, period: int, func=None, param=None, always_reset: bool = False, timeframe=None):
        """
        Faithful conversion for numpy-based storage.
        source: 1D numpy array (column slice of timeframe.dataset)
        """
        # Find the column index and name for the passed source array
        if timeframe is None:
            raise SystemError(f"Generated Series has no assigned timeframe [unknown]")

        if isinstance( source, series_c ):
            source_col_idx = source.index
            source_col_name = source.name
        else:
            raise ValueError( "Source must be 'series_c' type" )
            # source_col_idx = tools.get_column_index_from_array(timeframe.dataset, source)
            # if source_col_idx is None:
            #     raise ValueError("Column not found in dataset")
            # source_col_name = timeframe.columns[source_col_idx]

        self.name = tools.generatedSeriesNameFormat(type, source, period)
        self.sourceName = source_col_name
        self.sourceIndex = source_col_idx
        self.column_index = -1
        self.period = period if period is not None else len(source)
        self.param = param
        self.func = func
        self.timeframe = timeframe
        self.lastUpdatedTimestamp = 0
        self.alwaysReset = always_reset
        self.__current_cache = np.nan # will update on demand when calling iloc. Don't use from here.
        self.__cached_barindex = -1

        if self.func is None:
            raise SystemError(f"Generated Series without a func [{self.name}]")

        if self.period < 1:
            raise SystemError(f"Generated Series with invalid period [{period}]")


    def initialize( self, source: series_c ):
        assert isinstance(source, series_c), "Source must be series_c type"  # temporary while make sure everything is
        if len(source) >= self.period and (self.name not in self.timeframe.columns or self.alwaysReset):
            timeframe = self.timeframe
            if timeframe.backtesting and not timeframe.jumpstart:
                raise SystemError(f"[{self.name}] tried to initialize as backtesting")
            
            barindex = len(source) - 1
            start_time = time.time()

            # Call the func, which must now accept a 1D numpy array as the source and the 2D array as "dataset"
            # Expect func to return a 1D numpy array of values, aligned with the full dataset length
            values = self.func(source, self.period, timeframe.dataset, self.column_index, self.param)
            if isinstance(values, (list, tuple)):
                values = np.array(values, dtype=np.float64)

            # Add the new column if necessary
            if self.name not in timeframe.columns:
                series = timeframe.createColumn( self.name )
                self.column_index = series.index

                # n_rows = timeframe.dataset.shape[0]
                # new_col = np.full((n_rows, 1), np.nan, dtype=np.float64)
                # timeframe.dataset = np.hstack([timeframe.dataset, new_col])
                # self.column_index = timeframe.dataset.shape[1] - 1
                # timeframe.columns.append(self.name)
                # timeframe.registeredSeries[self.name] = series_c(timeframe.dataset[:,self.column_index], self.name)

            # Only assign values where not nan (mimicking dropna)
            mask = ~np.isnan(values)
            timeframe.dataset[mask, self.column_index] = values[mask]

            # Find the timestamp column index
            self.lastUpdatedTimestamp = int(timeframe.dataset[barindex, c.DF_TIMESTAMP])

            if timeframe.stream.initializing:
                print(f"Initialized {self.name}. Elapsed time: {time.time() - start_time:.2f} seconds")

    def update( self, source:series_c ):
        if self.timeframe.backtesting:
            return
        
        assert isinstance(source, series_c), "Source must be series_c type"  # temporary while make sure everything is
        
        tf = self.timeframe

        # if non existent or needs reset, initialize
        if self.alwaysReset or self.lastUpdatedTimestamp == 0:
            self.initialize(source)
            return

        # has this row already been updated?
        if self.lastUpdatedTimestamp >= tf.timestamp:
            return

        # slice the required block for current calculation
        barindex = tf.barindex
        period_slice = source[-self.period:]
        # func should return a 1D array or scalar; we want the most recent value
        newval = self.func(period_slice, self.period, tf.dataset, self.column_index, self.param)
        if isinstance(newval, (np.ndarray, series_c, list, tuple)):
            newval = newval[-1]
        tf.dataset[barindex, self.column_index] = newval
        self.lastUpdatedTimestamp = tf.timestamp

    def iloc( self, index = -1 ):
        barindex = self.timeframe.barindex

        if self.timeframe != active.timeframe :
            timestamp = active.timeframe.timestamp + ( (index+1) * self.timeframe.timeframeMsec )
            return self.timeframe.valueAtTimestamp( self.name, timestamp )

        # Handle lazy-loading cache for the current bar (index -1)
        if index == -1:
            # Check if the cache is valid for the current active.barindex
            if self.__cached_barindex == barindex and not np.isnan(self.__current_cache):
                return self.__current_cache
            else:
                # If cache is invalid or not yet populated, fetch from DataFrame
                if barindex >= 0 and barindex < len(self.timeframe.dataset):
                    value = self.timeframe.dataset[barindex, self.column_index]
                    self.__current_cache = value
                    self.__cached_barindex = barindex
                    return value
                else:
                    # Handle out-of-bounds access for current bar
                    return np.nan # Or raise an error, depending on desired behavior
        
        # Original iloc logic for other indices
        if index < 0:
            index = barindex + 1 + index
        
        # Ensure the index is within valid bounds after translation/clamping
        if index < 0 or index >= len(self.timeframe.dataset):
            return np.nan # Return NaN for out-of-bounds access
            
        return self.timeframe.dataset[index, self.column_index]
    iat = iloc # alias for the same method
    value = iloc # alias for the same method
    
    
    def current( self ):
        '''returns the last value in the series'''
        return self.iloc(-1)
    

    def series(self):
        try:
            series = self.timeframe.registeredSeries[self.name]
        except Exception as e:
            series = self.timeframe.dataset[:, self.column_index]
        return series
    

    def __add__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, series_c, NumericScalar)):
            return addSeries(self, other)
        raise ValueError("WTF def __add__(self, other)")

    def __radd__(self, other):
        if isinstance(other, (float, int)):
            return addScalar(other, self)
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, series_c, NumericScalar)):
            return subtractSeries(self, other)
        raise ValueError("WTF def __sub__(self, other)")

    def __rsub__(self, other):
        if isinstance(other, (float, int)):
            return subtractScalar(other, self)
        raise ValueError("rsub only defined for const - series")

    def __mul__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, series_c, NumericScalar)):
            return multiplySeries(self, other)
        raise ValueError("WTF def __mul__(self, other)")

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return multiplyScalar(other, self)
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, series_c, NumericScalar)):
            return divideSeries(self, other)
        raise ValueError("WTF def __truediv__(self, other)")

    def __rtruediv__(self, other):
        if isinstance(other, (float, int)):
            return divideScalar(other, self)
        raise ValueError("rtruediv only defined for const / series")
    
    def __neg__(self):
        return multiplySeries(self, -1)

    def __lt__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, series_c, NumericScalar)):
            return lessSeries(self, other)
        raise ValueError("Unsupported operand type for <")
    
    def __rlt__(self, other):
        if isinstance(other, (float, int)):
            return lessScalar(self, other)
        raise ValueError("Unsupported reversed operand for <")

    def __le__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, series_c, NumericScalar)):
            return lessOrEqualSeries(self, other)
        raise ValueError("Unsupported operand type for <=")
    
    def __rle__(self, other):
        if isinstance(other, (float, int)):
            return lessOrEqualScalar(self, other)
        raise ValueError("Unsupported reversed operand for <=")

    def __gt__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, series_c, NumericScalar)):
            return greaterSeries(self, other)
        raise ValueError("Unsupported operand type for >")
    
    def __rgt__(self, other):
        if isinstance(other, (float, int)):
            return greaterScalar(self, other)
        raise ValueError("Unsupported reversed operand for >")

    def __ge__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, series_c, NumericScalar)):
            return greaterOrEqualSeries(self, other)
        raise ValueError("Unsupported operand type for >=")
    
    def __rge__(self, other):
        if isinstance(other, (float, int)):
            return greaterOrEqualScalar(self, other)
        raise ValueError("Unsupported reversed operand for >=")

    def __eq__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, series_c, NumericScalar)):
            return equalSeries(self, other)
        return NotImplemented
    
    def __req__(self, other):
        if isinstance(other, (float, int)):
            return equalScalar(self, other)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, series_c, NumericScalar)):
            return notequalSeries(self, other)
        return NotImplemented

    def __rne__(self, other):
        if isinstance(other, (float, int)):
            return notEqualScalar(self, other)
        return NotImplemented
    
    def __invert__(self):
        return notSeries(self)
    
    '''
    def __and__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, NumericScalar)):
            return andSeries(self, other)
        if isinstance(other, (bool, int, float)):
            return andSeriesConst(self, bool(other))
        raise ValueError("Unsupported operand for &")

    def __rand__(self, other):
        if isinstance(other, (bool, int, float)):
            return andConstSeries(bool(other), self)
        raise ValueError("Unsupported operand for &")

    def __or__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray)):
            return orSeries(self, other)
        if isinstance(other, (bool, int, float)):
            return orSeriesConst(self, bool(other))
        raise ValueError("Unsupported operand for |")

    def __ror__(self, other):
        if isinstance(other, (bool, int, float)):
            return orConstSeries(bool(other), self)
        raise ValueError("Unsupported operand for |")

    def __xor__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray)):
            return xorSeries(self, other)
        if isinstance(other, (bool, int, float)):
            return xorSeriesConst(self, bool(other))
        raise ValueError("Unsupported operand for ^")

    def __rxor__(self, other):
        if isinstance(other, (bool, int, float)):
            return xorConstSeries(bool(other), self)
        raise ValueError("Unsupported operand for ^")
    '''
    



    def plot( self, chart_name = None, color = "#8FA7BBAA", style = 'solid', width = 1  ):
        '''* it returns the generatedSeries. Calling plot from the timeframe and the function returns the plot_c but not here*

        source: can either be a series or a value. A series can only be plotted when it is in the dataframe. When plotting a value a series will be automatically created in the dataframe.
        chart_name: Leave empty for the main panel. Use 'panel' for plotting in the subpanel.
        color: in a string. Can be hexadecial '#DADADADA' or rgba format 'rgba(255,255,255,1.0)'
        style: LINE_STYLE = Literal['solid', 'dotted', 'dashed', 'large_dashed', 'sparse_dotted']
        width: int
        '''
        if( self.lastUpdatedTimestamp > 0 ):
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
        if( self.lastUpdatedTimestamp > 0 ):
            self.timeframe.histogram( self.series(), self.name, chart_name, color, margin_top, margin_bottom )
            return self
    
    def crossingUp( self, other ):
        current_self_val = self.iloc(-1)
        previous_self_val = self.iloc(-2)

        if np.isnan(current_self_val) or np.isnan(previous_self_val):
            return False

        if isinstance( other, generatedSeries_c ):
            current_other_val = other.iloc(-1)
            previous_other_val = other.iloc(-2)
            if np.isnan(current_other_val) or np.isnan(previous_other_val):
                return False
            return ( previous_self_val <= previous_other_val and current_self_val >= current_other_val and current_self_val != previous_self_val )
        elif isinstance( other, (series_c,np.ndarray) ):
            # Use iloc directly from the pd.Series
            if len(other) < 2 or active.barindex < 1 or np.isnan(other[active.barindex-1]) or np.isnan(other[active.barindex]):
                return False
            return ( previous_self_val <= other[active.barindex-1] and current_self_val >= other[active.barindex] and current_self_val != previous_self_val )
        else: # assuming float or int
            try:
                float_other = float(other)
            except ValueError:
                return False
            # Corrected line: Use previous_self_val in the last condition
            return ( previous_self_val <= float_other and current_self_val >= float_other and current_self_val != previous_self_val )
    
    def crossingDown( self, other ):
        current_self_val = self.iloc(-1)
        previous_self_val = self.iloc(-2)
        if np.isnan(current_self_val) or np.isnan(previous_self_val):
            return False

        if isinstance( other, generatedSeries_c ):
            current_other_val = other.iloc(-1)
            previous_other_val = other.iloc(-2)
            if np.isnan(current_other_val) or np.isnan(previous_other_val):
                return False
            return ( previous_self_val >= previous_other_val and current_self_val <= current_other_val and current_self_val != previous_self_val )
        elif isinstance( other, (series_c,np.ndarray) ):
            if len(other) < 2 or active.barindex < 1 or np.isnan(other[active.barindex-1]) or np.isnan(other[active.barindex]):
                return False
            return ( previous_self_val >= other[active.barindex-1] and current_self_val <= other[active.barindex] and current_self_val != previous_self_val )
        else: 
            try:
                float_other = float(other)
            except ValueError:
                return False
            return ( previous_self_val >= float_other and current_self_val <= float_other and current_self_val != previous_self_val )
    
    def crossing( self, other ):
        return self.crossingUp(other) or self.crossingDown(other)
    


def _ensure_object_array( data: series_c|generatedSeries_c )-> series_c:
    """
    Helper function to ensure the input is a NumPy array.
    This replaces the initial type checking and conversion logic.
    """
    if isinstance(data, series_c):
        return data
    elif isinstance(data, generatedSeries_c):
        return data.series()
    elif isinstance(data, np.ndarray):
        # try to guess its index but we won't allow it anyway. We want to get rid of this option
        index = tools.get_column_index_from_array( self.dataset, source )
        if index:
            raise ValueError( f"_ensure_object_array: Numpy np.ndarray is not a valid object, but array index found [{index}]. Name: [{self.columns[index]}]" )
        raise ValueError( "_ensure_object_array: Numpy np.ndarray is not a valid object" )
    else:
        raise TypeError(f"Unsupported input type: {type(data)}. Expected np.ndarray, pd.Series, or generatedSeries_c.")



def addSeries(colA: str | generatedSeries_c | series_c, colB: str | generatedSeries_c | series_c | NumericScalar, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe

    colA = timeframe.seriesFromMultiObject( colA )
    if isinstance( colB, NumericScalar ):
        name = f"add_{colA.index}_{colB}"
    else:
        colB = timeframe.seriesFromMultiObject( colB )
        name = f"add_{colA.index}_{colB.index}" # Using resolved indices/names for consistent naming
        if colA.shape != colB.shape: # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")

    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_add_series, colB )

def subtractSeries(colA: str | generatedSeries_c | series_c, colB: str | generatedSeries_c | series_c | NumericScalar, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe

    colA = timeframe.seriesFromMultiObject( colA )
    if isinstance( colB, NumericScalar ):
        name = f"sub_{colA.index}_{colB}"
    else:
        colB = timeframe.seriesFromMultiObject( colB )
        name = f"sub_{colA.index}_{colB.index}" # Using resolved indices/names for consistent naming
        if colA.shape != colB.shape: # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries( name, colA, 1, _generatedseries_calculate_subtract_series, colB )

def multiplySeries(colA: str | generatedSeries_c | series_c, colB: str | generatedSeries_c | series_c | NumericScalar, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe

    colA = timeframe.seriesFromMultiObject( colA )
    if isinstance( colB, NumericScalar ):
        name = f"mul_{colA.index}_{colB}"
    else:
        colB = timeframe.seriesFromMultiObject( colB )
        name = f"mul_{colA.index}_{colB.index}" # Using resolved indices/names for consistent naming
        if colA.shape != colB.shape: # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_multiply_series, colB)

def divideSeries(colA: str | generatedSeries_c | series_c, colB: str | generatedSeries_c | series_c | NumericScalar, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe

    colA = timeframe.seriesFromMultiObject( colA )
    if isinstance( colB, NumericScalar ):
        name = f"div_{colA.index}_{colB}"
    else:
        colB = timeframe.seriesFromMultiObject( colB )
        name = f"div_{colA.index}_{colB.index}" # Using resolved indices/names for consistent naming
        if colA.shape != colB.shape: # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_divide_series, colB)

def powerSeries(colA: str | generatedSeries_c | series_c, colB: str | generatedSeries_c | series_c | NumericScalar, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe

    colA = timeframe.seriesFromMultiObject( colA )
    if isinstance( colB, NumericScalar ):
        name = f"pow_{colA.index}_{colB}"
    else:
        colB = timeframe.seriesFromMultiObject( colB )
        name = f"pow_{colA.index}_{colB.index}" # Using resolved indices/names for consistent naming
        if colA.shape != colB.shape: # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_power_series, colB)

def minSeries(colA: str | generatedSeries_c | series_c, colB: str | generatedSeries_c | series_c | NumericScalar, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe

    colA = timeframe.seriesFromMultiObject( colA )
    if isinstance( colB, NumericScalar ):
        name = f"min_{colA.index}_{colB}"
    else:
        colB = timeframe.seriesFromMultiObject( colB )
        name = f"min_{colA.index}_{colB.index}" # Using resolved indices/names for consistent naming
        if colA.shape != colB.shape: # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_min_series, colB)

def maxSeries(colA: str | generatedSeries_c | series_c, colB: str | generatedSeries_c | series_c | NumericScalar, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe

    colA = timeframe.seriesFromMultiObject( colA )
    if isinstance( colB, NumericScalar ):
        name = f"max_{colA.index}_{colB}"
    else:
        colB = timeframe.seriesFromMultiObject( colB )
        name = f"max_{colA.index}_{colB.index}" # Using resolved indices/names for consistent naming
        if colA.shape != colB.shape: # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_max_series, colB)

def equalSeries(colA: str | generatedSeries_c | series_c, colB: str | generatedSeries_c | series_c | NumericScalar, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe

    colA = timeframe.seriesFromMultiObject( colA )
    if isinstance( colB, NumericScalar ):
        name = f"eq_{colA.index}_{colB}"
    else:
        colB = timeframe.seriesFromMultiObject( colB )
        name = f"eq_{colA.index}_{colB.index}" # Using resolved indices/names for consistent naming
        if colA.shape != colB.shape: # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_equal_series, colB)

def notequalSeries(colA: str | generatedSeries_c | series_c, colB: str | generatedSeries_c | series_c | NumericScalar, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe

    colA = timeframe.seriesFromMultiObject( colA )
    if isinstance( colB, NumericScalar ):
        name = f"neq_{colA.index}_{colB}"
    else:
        colB = timeframe.seriesFromMultiObject( colB )
        name = f"neq_{colA.index}_{colB.index}" # Using resolved indices/names for consistent naming
        if colA.shape != colB.shape: # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_notequal_series, colB)

def greaterSeries(colA: str | generatedSeries_c | series_c, colB: str | generatedSeries_c | series_c | NumericScalar, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe

    colA = timeframe.seriesFromMultiObject( colA )
    if isinstance( colB, NumericScalar ):
        name = f"gr_{colA.index}_{colB}"
    else:
        colB = timeframe.seriesFromMultiObject( colB )
        name = f"gr_{colA.index}_{colB.index}" # Using resolved indices/names for consistent naming
        if colA.shape != colB.shape: # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_greater_series, colB)

def greaterOrEqualSeries(colA: str | generatedSeries_c | series_c, colB: str | generatedSeries_c | series_c | NumericScalar, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe

    colA = timeframe.seriesFromMultiObject( colA )
    if isinstance( colB, NumericScalar ):
        name = f"gre_{colA.index}_{colB}"
    else:
        colB = timeframe.seriesFromMultiObject( colB )
        name = f"gre_{colA.index}_{colB.index}" # Using resolved indices/names for consistent naming
        if colA.shape != colB.shape: # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_greaterorequal_series, colB)

def lessSeries(colA: str | generatedSeries_c | series_c, colB: str | generatedSeries_c | series_c | NumericScalar, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe

    colA = timeframe.seriesFromMultiObject( colA )
    if isinstance( colB, NumericScalar ):
        name = f"lt_{colA.index}_{colB}"
    else:
        colB = timeframe.seriesFromMultiObject( colB )
        name = f"lt_{colA.index}_{colB.index}" # Using resolved indices/names for consistent naming
        if colA.shape != colB.shape: # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_less_series, colB)

def lessOrEqualSeries(colA: str | generatedSeries_c | series_c, colB: str | generatedSeries_c | series_c | NumericScalar, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe

    colA = timeframe.seriesFromMultiObject( colA )
    if isinstance( colB, NumericScalar ):
        name = f"le_{colA.index}_{colB}"
    else:
        colB = timeframe.seriesFromMultiObject( colB )
        name = f"le_{colA.index}_{colB.index}" # Using resolved indices/names for consistent naming
        if colA.shape != colB.shape: # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_lessequal_series, colB)

def notSeries(source: str | generatedSeries_c | series_c, timeframe=None) ->generatedSeries_c:
    timeframe = timeframe or active.timeframe

    source = timeframe.seriesFromMultiObject( source )
    name = f"not_{source.index}"
    return timeframe.calcGeneratedSeries( name, source, 1, _generatedseries_calculate_logical_not, None )

#
########## SCALARS By SERIES
#


def addScalar(scalar: NumericScalar, series: str | generatedSeries_c | series_c, timeframe=None) -> generatedSeries_c:
    """
    Factory function for scalar + series.
    """
    timeframe = timeframe or active.timeframe
    series = timeframe.seriesFromMultiObject( series )
    name = f"add_{scalar}_{series.index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries( name, series, 1, _generatedseries_calculate_scalar_add_series, scalar )

def subtractScalar(scalar: NumericScalar, series: str | generatedSeries_c | series_c, timeframe=None) -> generatedSeries_c:
    """
    Factory function for scalar - series.
    """
    timeframe = timeframe or active.timeframe
    series = timeframe.seriesFromMultiObject( series )
    name = f"sub_{scalar}_{series.index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_subtract_series, scalar)

def multiplyScalar(scalar: NumericScalar, series: str | generatedSeries_c | series_c, timeframe=None) -> generatedSeries_c:
    """
    Factory function for scalar * series.
    """
    timeframe = timeframe or active.timeframe
    series = timeframe.seriesFromMultiObject( series )
    name = f"mul_{scalar}_{series.index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_multiply_series, scalar)

def divideScalar(scalar: NumericScalar, series: str | generatedSeries_c | series_c, timeframe=None) -> generatedSeries_c:
    """
    Factory function for scalar / series.
    """
    timeframe = timeframe or active.timeframe
    series = timeframe.seriesFromMultiObject( series )
    name = f"div_{scalar}_{series.index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_divide_series, scalar)

def powerScalar(scalar: NumericScalar, series: str | generatedSeries_c | series_c, timeframe=None) -> generatedSeries_c:
    """
    Factory function for scalar ** series.
    """
    timeframe = timeframe or active.timeframe
    series = timeframe.seriesFromMultiObject( series )
    name = f"pow_{scalar}_{series.index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_power_series, scalar)

def minScalar(scalar: NumericScalar, series: str | generatedSeries_c | series_c, timeframe=None) -> generatedSeries_c:
    """
    Factory function for min(scalar, series).
    """
    timeframe = timeframe or active.timeframe
    series = timeframe.seriesFromMultiObject( series )
    name = f"min_{scalar}_{series.index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_min_series, scalar)

def maxScalar(scalar: NumericScalar, series: str | generatedSeries_c | series_c, timeframe=None) -> generatedSeries_c:
    """
    Factory function for max(scalar, series).
    """
    timeframe = timeframe or active.timeframe
    series = timeframe.seriesFromMultiObject( series )
    name = f"max_{scalar}_{series.index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_max_series, scalar)

def equalScalar(scalar: NumericScalar, series: str | generatedSeries_c | series_c, timeframe=None) -> generatedSeries_c:
    """Factory for scalar == series."""
    timeframe = timeframe or active.timeframe
    series = timeframe.seriesFromMultiObject( series )
    name = f"eq_{scalar}_{series.index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_equal_series, scalar)

def notEqualScalar(scalar: NumericScalar, series: str | generatedSeries_c | series_c, timeframe=None) -> generatedSeries_c:
    """Factory for scalar != series."""
    timeframe = timeframe or active.timeframe
    series = timeframe.seriesFromMultiObject( series )
    name = f"neq_{scalar}_{series.index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_notequal_series, scalar)

def greaterScalar(scalar: NumericScalar, series: str | generatedSeries_c | series_c, timeframe=None) -> generatedSeries_c:
    """Factory for scalar > series."""
    timeframe = timeframe or active.timeframe
    series = timeframe.seriesFromMultiObject( series )
    name = f"gt_{scalar}_{series.index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_greater_series, scalar)

def greaterOrEqualScalar(scalar: NumericScalar, series: str | generatedSeries_c | series_c, timeframe=None) -> generatedSeries_c:
    """Factory for scalar >= series."""
    timeframe = timeframe or active.timeframe
    series = timeframe.seriesFromMultiObject( series )
    name = f"ge_{scalar}_{series.index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_greaterorequal_series, scalar)

def lessScalar(scalar: NumericScalar, series: str | generatedSeries_c | series_c, timeframe=None) -> generatedSeries_c:
    """Factory for scalar < series."""
    timeframe = timeframe or active.timeframe
    series = timeframe.seriesFromMultiObject( series )
    name = f"lt_{scalar}_{series.index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_less_series, scalar)

def lessOrEqualScalar(scalar: NumericScalar, series: str | generatedSeries_c | series_c, timeframe=None) -> generatedSeries_c:
    """Factory for scalar <= series."""
    timeframe = timeframe or active.timeframe
    series = timeframe.seriesFromMultiObject( series )
    name = f"le_{scalar}_{series.index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_lessequal_series, scalar)



###################### ANALITIC TOOLS #################################

def highest(source: series_c|generatedSeries_c, period: int, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('highest', _ensure_object_array(source), period, _generatedseries_calculate_highest)

def lowest(source: series_c|generatedSeries_c, period: int, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('lowest', _ensure_object_array(source), period, _generatedseries_calculate_lowest)

def highestbars(source: series_c|generatedSeries_c, period: int, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('highestbars', _ensure_object_array(source), period, _generatedseries_calculate_highestbars)

def lowestbars(source: series_c|generatedSeries_c, period: int, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('lowestbars', _ensure_object_array(source), period, _generatedseries_calculate_lowestbars)

def falling( source: series_c|generatedSeries_c, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'falling', _ensure_object_array(source), period, _generatedseries_calculate_falling )

def rising( source: series_c|generatedSeries_c, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'rising', _ensure_object_array(source), period, _generatedseries_calculate_rising )

def barsSinceSeries(source: series_c|generatedSeries_c, period: int, timeframe=None) -> generatedSeries_c:
    import inspect
    # Get caller info by going up 2 levels in the stack
    caller_frame = inspect.currentframe().f_back
    frame_info = inspect.getframeinfo(caller_frame)
    caller_id = f"{frame_info.function[:5]}{frame_info.lineno}"

    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('barsSince'+caller_id, _ensure_object_array(source), period, _generatedseries_calculate_barssince)

def indexWhenTrueSeries(source: series_c|generatedSeries_c, period: int = None, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('indexwhentrue_series', _ensure_object_array(source), period, _generatedseries_calculate_indexwhentrue)

def indexWhenFalseSeries(source: series_c|generatedSeries_c, period: int, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('indexwhenfalse_series', _ensure_object_array(source), period, _generatedseries_calculate_indexwhenfalse)

def barsWhileTrueSeries(source: series_c|generatedSeries_c, period: int = None, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('barsWhileTrue', _ensure_object_array(source), period, _generatedseries_calculate_barswhiletrue)

def barsWhileFalseSeries(source: series_c|generatedSeries_c, period: int = None, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('barsWhileFalse', _ensure_object_array(source), period, _generatedseries_calculate_barswhilefalse)




########################## INDICATORS #################################

def SMA( source: series_c|generatedSeries_c, period: int, timeframe=None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('sma', _ensure_object_array(source), period, _generatedseries_calculate_sma)


def EMA( source: series_c|generatedSeries_c, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( "ema", _ensure_object_array(source), period, _generatedseries_calculate_ema, always_reset=True )

def DEMA( source: series_c|generatedSeries_c, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( "dema", _ensure_object_array(source), period, _generatedseries_calculate_dema, always_reset=True )


def RMA( source:series_c|generatedSeries_c, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'rma', _ensure_object_array(source), period, _generatedseries_calculate_rma, always_reset=True )

def WMA( source:series_c|generatedSeries_c, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( "wma", _ensure_object_array(source), period, _generatedseries_calculate_wma )

def HMA( source:series_c|generatedSeries_c, period:int, timeframe = None )->generatedSeries_c:
    """Hull Moving Average implementation using multiple calculation steps
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    timeframe = timeframe or active.timeframe

    source = _ensure_object_array(source)
    
    # First calculate WMA with half period
    half_length = int(period / 2) 
    wma_half = timeframe.calcGeneratedSeries( "wma", source, half_length, _generatedseries_calculate_wma )
    
    # Calculate WMA with full period
    wma_full = timeframe.calcGeneratedSeries( "wma", source, period,  _generatedseries_calculate_wma )
    
    # Calculate 2 * WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA with sqrt(period)
    sqrt_period = int(np.sqrt(period))
    return timeframe.calcGeneratedSeries( "hma", raw_hma.series(), sqrt_period, _generatedseries_calculate_wma )

# # def JMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
# #     timeframe = timeframe or active.timeframe
# #     return timeframe.calcGeneratedSeries( "jma", source, period, pt.jma )

# # def KAMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
# #     timeframe = timeframe or active.timeframe
# #     return timeframe.calcGeneratedSeries( "kama", source, period, pt.kama )


def STDEV( source:series_c|generatedSeries_c, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'stdev', _ensure_object_array(source), period, _generatedseries_calculate_stdev )

def DEV( source:series_c|generatedSeries_c, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'dev', _ensure_object_array(source), period, _generatedseries_calculate_dev )

def WILLR( period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    source = timeframe.dataset[:, c.DF_CLOSE]
    return timeframe.calcGeneratedSeries( 'wpr', source, period, _generatedseries_calculate_williams_r )

def TR( period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    source = timeframe.dataset[:, c.DF_CLOSE] # pass through the checks
    return timeframe.calcGeneratedSeries( 'tr', source, period, _generatedseries_calculate_tr )

def ATR( period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    source = timeframe.dataset[:, c.DF_CLOSE]
    return timeframe.calcGeneratedSeries( 'atr', source, period, _generatedseries_calculate_atr )

def SLOPE( source:series_c|generatedSeries_c, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'slope', _ensure_object_array(source), period, _generatedseries_calculate_slope )

def VHMA(source: series_c|generatedSeries_c, period: int, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('vhma', _ensure_object_array(source), period, _generatedseries_calculate_vhma, always_reset= True)

def BIAS( source:series_c|generatedSeries_c, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'bias', _ensure_object_array(source), period, _generatedseries_calculate_bias )

def LINREG( source:series_c|generatedSeries_c, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( "linreg", _ensure_object_array(source), period, _generatedseries_calculate_linreg )

def CCI(period: int = 20, timeframe=None) -> generatedSeries_c:
    if not isinstance(period, int ):
        raise ValueError( "CCI requires only a period argument" )
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('cci', timeframe.dataset[:, c.DF_CLOSE], period, _generatedseries_calculate_cci, always_reset= True)

def CFO( source:series_c|generatedSeries_c, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'cfo', _ensure_object_array(source), period, _generatedseries_calculate_cfo )

def CMO(source: series_c|generatedSeries_c, period: int, timeframe=None) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries('cmo', _ensure_object_array(source), period, _generatedseries_calculate_cmo, always_reset= True)

def FWMA( source:series_c|generatedSeries_c, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'fwma', _ensure_object_array(source), period, _generatedseries_calculate_fwma )

def RSI( source:series_c|generatedSeries_c, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'rsi', _ensure_object_array(source), period, _generatedseries_calculate_rsi, always_reset=True )

def IFTrsi( source:series_c|generatedSeries_c, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    rsi = timeframe.calcGeneratedSeries( 'rsi', _ensure_object_array(source), period, _generatedseries_calculate_rsi, always_reset=True )
    return timeframe.calcGeneratedSeries( 'iftrsi', rsi.series(), period, _generatedseries_calculate_inverse_fisher_rsi )

def Fisher( period:int, signal:float=None, timeframe = None )->tuple[generatedSeries_c, generatedSeries_c]:
    timeframe = timeframe or active.timeframe
    fish = timeframe.calcGeneratedSeries( 'fisher', timeframe.dataset[:, c.DF_CLOSE], period, _generatedseries_calculate_fisher )
    sig = timeframe.calcGeneratedSeries( 'fishersig', timeframe.dataset[:, c.DF_CLOSE], period, _generatedseries_calculate_fisher_signal, signal )
    return fish, sig

def AO( fast: int = 5, slow: int = 34, timeframe = None ) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    param = (fast, slow)
    return timeframe.calcGeneratedSeries('ao', timeframe.dataset[:,c.DF_CLOSE], max(fast,slow), _generatedseries_calculate_ao, param)

def BR( period:int, timeframe = None )->tuple[generatedSeries_c, generatedSeries_c]:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'br', timeframe.dataset[:,c.DF_CLOSE], period, _generatedseries_calculate_br )

def AR( period:int, timeframe = None )->tuple[generatedSeries_c, generatedSeries_c]:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'ar', timeframe.dataset[:,c.DF_CLOSE], period, _generatedseries_calculate_ar )

def BRAR( period:int, timeframe = None )->tuple[generatedSeries_c, generatedSeries_c]:
    timeframe = timeframe or active.timeframe
    br = BR(period, timeframe)
    ar = AR(period, timeframe)
    return br, ar

def CG( source:series_c|generatedSeries_c, period:int, timeframe = None )->generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'cg', _ensure_object_array(source), period, _generatedseries_calculate_cg )

def STOCHk( source: series_c|generatedSeries_c, period:int, timeframe=None )-> tuple[generatedSeries_c, generatedSeries_c]:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( "stochk", _ensure_object_array(source), period, _generatedseries_calculate_stoch_k )


def OBV( source: series_c|generatedSeries_c, timeframe=None ) -> generatedSeries_c:
    timeframe = timeframe or active.timeframe
    return timeframe.calcGeneratedSeries( 'obv', source, 1, _generatedseries_calculate_obv, always_reset=True )



# # #
# # # OTHER NOT GENERATED SERIES
# # #


def Stochastic(source: series_c|generatedSeries_c, k_period: int = 14, d_period: int = 3, timeframe=None)-> tuple[generatedSeries_c, generatedSeries_c]:
    """
    Calculates the Stochastic Oscillator (%K and %D lines).

    Args:
        close_series_input: The close price series. Can be a string (column name),
                             a generatedSeries_c object, or a direct NumPy array.
        k_period (int): The period for the %K calculation (e.g., 14).
        d_period (int): The period for the %D calculation (SMA of %K, e.g., 3).
        timeframe: The timeframe context (defaults to active.timeframe).

    Returns:
        Tuple[generatedSeries_c, generatedSeries_c]: A tuple containing the %K line
        and the %D line as generatedSeries_c objects.
    """
    timeframe = timeframe or active.timeframe

    # Create the %K line generatedSeries_c
    k_line_series = STOCHk(source, k_period)

    # Create the %D line generatedSeries_c (SMA of %K)
    d_line_series = SMA( k_line_series, d_period )
    return k_line_series, d_line_series


def BollingerBands( source:series_c|generatedSeries_c, period:int, mult:float = 2.0, timeframe=None )->tuple[generatedSeries_c, generatedSeries_c, generatedSeries_c]:
    """
    Returns the Bollinger Bands (basis, upper, lower) for the given source series and period.

    Args:
        source (pd.Series): The input pandas Series to calculate the Bollinger Bands on.
        period (int): The period/window for the Bollinger Bands calculation.

    Returns:
        Tuple[generatedSeries_c, generatedSeries_c, generatedSeries_c]: The basis (SMA), upper band, and lower band as generatedSeries_c objects.
    """
    timeframe = timeframe or active.timeframe
    BBbasis = SMA(source, period)
    stdev = STDEV(source, period)
    BBupper = BBbasis + (stdev * mult)
    BBlower = BBbasis - (stdev * mult)
    # BBupper = active.timeframe.calcGeneratedSeries( 'bbu', source, period, _generatedseries_calculate_bbupper, mult )
    # BBlower = active.timeframe.calcGeneratedSeries( 'bbl', source, period, _generatedseries_calculate_bblower, mult )
    return BBbasis, BBupper, BBlower


def MACD( source:series_c|generatedSeries_c, fast: int = 12, slow: int = 26, signal: int = 9, timeframe=None) -> tuple[generatedSeries_c, generatedSeries_c, generatedSeries_c]:
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
    timeframe = timeframe or active.timeframe
    # Calculate the fast and slow EMAs
    fast_ema = EMA(source, fast, timeframe)
    slow_ema = EMA(source, slow, timeframe)

    # MACD line: difference between fast and slow EMA
    macd_line = fast_ema - slow_ema

    # Signal line: EMA of the MACD line
    signal_line = EMA(macd_line, signal, timeframe)

    # Histogram: MACD line - Signal line
    hist = macd_line - signal_line

    return macd_line, signal_line, hist


################ Helpers. Not series #########################

def indexWhenTrue(source: series_c|generatedSeries_c)-> Union[int, None]:
    """
    Finds the 0-based positional index of the last True value in a boolean-coercible array.

    Args:
        source: A NumPy array, pandas Series, or generatedSeries_c containing boolean or
                values that can be coerced to boolean (e.g., 0/1 integers).

    Returns:
        int: The 0-based index of the last True value, or None if no True values are found.
    """
    source_array = _ensure_object_array(source)
    
    # Ensure the array is boolean. This handles cases where source might be 0s and 1s.
    boolean_source = source_array.astype(bool)

    # Find all indices where the condition is True
    true_indices = np.where(boolean_source)[0]

    if true_indices.size > 0:
        # Return the last (most recent) index where the condition was True
        return int(true_indices[-1])
    else:
        return None

def indexWhenFalse(source: series_c|generatedSeries_c)-> Union[int, None]:
    """
    Finds the 0-based positional index of the last False value in a boolean-coercible array.

    Args:
        source: A NumPy array, pandas Series, or generatedSeries_c containing boolean or
                values that can be coerced to boolean.

    Returns:
        int: The 0-based index of the last False value, or None if no False values are found.
    """
    source_array = _ensure_object_array(source)
    
    # Ensure the array is boolean
    boolean_source = source_array.astype(bool)
    
    # Find all indices where the condition is False (using logical NOT on the boolean array)
    false_indices = np.where(~boolean_source)[0]

    if false_indices.size > 0:
        # Return the last (most recent) index where the condition was False
        return int(false_indices[-1])
    else:
        return None
    
def barsSince( source ):
    index_when_true = indexWhenTrue( source )
    if index_when_true is None: 
        return None
    return active.barindex - index_when_true

def barsWhileTrue( source ):
    index_when_false = indexWhenFalse( source )
    if index_when_false is None: 
        return None
    return active.barindex - index_when_false

def crossingUp( self:series_c|generatedSeries_c|float, other:series_c|generatedSeries_c|float ):
    """
    Determines if 'self' crosses up over 'other' between the previous and current bar.

    Args:
        self: The first value/series.
        other: The second value/series to compare against.

    Returns:
        bool: True if a crossing up occurred, False otherwise.
    """
    if isinstance( self, generatedSeries_c ):
        # Directly use self.iloc(-1) and self.iloc(-2) for current and previous values
        current_self_val = self.iloc(-1)
        previous_self_val = self.iloc(-2)
        if np.isnan(current_self_val) or np.isnan(previous_self_val):
            return False

        if isinstance( other, generatedSeries_c ):
            current_other_val = other.iloc(-1)
            previous_other_val = other.iloc(-2)
            if np.isnan(current_other_val) or np.isnan(previous_other_val):
                return False
            return ( previous_self_val <= previous_other_val and current_self_val >= current_other_val and current_self_val != previous_self_val )
        elif isinstance( other, (np.ndarray, series_c) ):
            if len(other) < 2 or active.barindex < 1 or np.isnan(other[active.barindex-1]) or np.isnan(other[active.barindex]):
                return False
            return ( previous_self_val <= other[active.barindex-1] and current_self_val >= other[active.barindex] and current_self_val != previous_self_val )
        else: # assuming float or int
            try:
                float_other = float(other)
            except ValueError:
                return False
            # Corrected line: Use previous_self_val in the last condition
            return ( previous_self_val <= float_other and current_self_val >= float_other and current_self_val != previous_self_val )
    
    # Original logic for pd.Series and float/int (unchanged, but might need similar iloc(-1) and iloc(-2) adaptations for clarity if it's not already doing that)
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
    if isinstance( self, (np.ndarray, series_c) ):
        if( len(self) < 2 or active.barindex < 1 ):
            return False
        self_old = self[active.barindex-1]
        self_new = self[active.barindex]
        if isinstance( other, (np.ndarray, series_c) ):
            if( len(other) < 2 ):
                return False
            other_old = other[active.barindex-1]
            other_new = other[active.barindex]
        elif isinstance( other, generatedSeries_c ):
            # Directly use other.iloc(-1) and other.iloc(-2)
            if np.isnan(other.lastUpdatedTimestamp) or len(other.series()) < 2 or active.barindex < 1 :
                return False
            other_old = other.iloc(-2)
            other_new = other.iloc(-1) 
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


def crossingDown( self:series_c|generatedSeries_c|float, other:series_c|generatedSeries_c|float ):
    """
    Determines if 'self' crosses down below 'other' between the previous and current bar.

    Args:
        self: The first value/series.
        other: The second value/series to compare against.

    Returns:
        bool: True if a crossing down occurred, False otherwise.
    """
    if isinstance( self, generatedSeries_c ):
        current_self_val = self.iloc(-1)
        previous_self_val = self.iloc(-2)
        if np.isnan(current_self_val) or np.isnan(previous_self_val):
            return False

        if isinstance( other, generatedSeries_c ):
            current_other_val = other.iloc(-1)
            previous_other_val = other.iloc(-2)
            if np.isnan(current_other_val) or np.isnan(previous_other_val):
                return False
            return ( previous_self_val >= previous_other_val and current_self_val <= current_other_val and current_self_val != previous_self_val )
        elif isinstance( other, (np.ndarray, series_c) ):
            if len(other) < 2 or active.barindex < 1 or np.isnan(other[active.barindex-1]) or np.isnan(other[active.barindex]):
                return False
            return ( previous_self_val >= other[active.barindex-1] and current_self_val <= other[active.barindex] and current_self_val != previous_self_val )
        else: 
            try:
                float_other = float(other)
            except ValueError:
                return False
            # Corrected: Changed `previous_other_val` to `previous_self_val` for constant comparison.
            return ( previous_self_val >= float_other and current_self_val <= float_other and current_self_val != previous_self_val )
    
    # Original logic for pd.Series and float/int (unchanged, but might need similar iloc(-1) and iloc(-2) adaptations for clarity if it's not already doing that)
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
    if isinstance( self, (np.ndarray, series_c) ):
        if( len(self) < 2 or active.barindex < 1 ):
            return False
        self_old = self[active.barindex-1]
        self_new = self[active.barindex]
        if isinstance( other, (np.ndarray, series_c) ):
            if( len(other) < 2 ):
                return False
            other_old = other[active.barindex-1]
            other_new = other[active.barindex]
        elif isinstance( other, generatedSeries_c ):
            if np.isnan(other.lastUpdatedTimestamp) or len(other.series()) < 2 or active.barindex < 1 :
                return False
            other_old = other.iloc(-2)
            other_new = other.iloc(-1) 
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
    return crossingUp( self, other ) or crossingDown( self, other )


from .pivots import pivots_c, pivot_c
pivotsNow:pivots_c = None
def pivots( high:series_c, low:series_c, amplitude: float = 1.0, reversal_percent: float = 32.0 )->pivots_c:
    global pivotsNow
    if pivotsNow == None:
        pivotsNow = pivots_c(amplitude, reversal_percent)

    pivotsNow.update(high, low)
    return pivotsNow