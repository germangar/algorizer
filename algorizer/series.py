
from typing import Union, TYPE_CHECKING
talib_available = False
talib = None
try:
    import talib
    talib_available = True
except ImportError:
    talib_available = False
    print("Talib not available")

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import time

from .constants import c
from . import active
from . import tools


if TYPE_CHECKING:
    from .stream import timeframe_c
    

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
    if window < 1:
        return np.full_like(arr, np.nan)
    
    if n == 0:
        return np.array([], dtype=np.float64)

    if window > n:
        return np.full_like(arr, np.nan)

    windows = sliding_window_view(arr, window_shape=window)
    
    try:
        applied_values = func(windows, axis=-1)
    except TypeError:
        applied_values = np.array([func(w) for w in windows], dtype=np.float64)
        
    return np.concatenate((np.full(window - 1, np.nan, dtype=np.float64), applied_values))


def _rolling_window_apply_optimized_axis(arr: np.ndarray, window: int, func) -> np.ndarray:
    """
    Applies a function over a rolling window of a 1D NumPy array using sliding_window_view.
    Pads the beginning with NaNs to match the input array's length.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=np.float64)
    
    n = len(arr)
    if window < 1:
        return np.full_like(arr, np.nan) # Should technically return all NaNs for length < 1
    
    # Check if we have enough data to form at least one window
    if n < window:
        return np.full_like(arr, np.nan)
        
    # Create the rolling window view
    # The view will have shape (n - window + 1, window)
    window_view = sliding_window_view(arr, window)
    
    # Apply the function across the last axis (the window dimension)
    # The result will have shape (n - window + 1,)
    calculated_values = func(window_view)
    
    # Pad the beginning with NaNs to match the original array size
    # We need (window - 1) NaNs at the start
    padding = np.full(window - 1, np.nan, dtype=np.float64)
    
    # Concatenate the padding and the calculated values
    result = np.concatenate((padding, calculated_values))
    
    # Ensure the result has the same length as the input
    if len(result) > n:
        return result[:n]
    elif len(result) < n:
        # This case should ideally not happen if n >= window
        return np.pad(result, (0, n - len(result)), constant_values=np.nan)

    return result


NumericScalar = Union[float, int]
OperandType = Union[np.ndarray, NumericScalar]

def _prepare_param_for_op(param, src_len: int, dataset) -> np.ndarray:
    """
    Normalize 'param' into a 1-D np.float64 array of length src_len aligned to the tail.

    Rules:
    - Python or NumPy scalar -> full array filled with scalar.
    - or 1-D ndarray -> tail-aligned: last src_len values (left-pad with NaNs if shorter).
    - None or empty -> full NaN array.
    - 0-D np.ndarray (numpy scalar) treated like scalar.
    - 2-D arrays raise ValueError.
    """

    # Scalar (covers Python and NumPy scalars)
    if np.isscalar(param):
        val = float(param)
        return np.full(src_len, val, dtype=np.float64)

    # None or empty
    if param is None:
        return np.full(src_len, np.nan, dtype=np.float64)
    
    if isinstance(param, generatedSeries_c):
        param = param.series()
        if len(param) >= src_len:
            return param[-src_len:].astype(np.float64, copy=False)
        return np.asarray(param, dtype=np.float64)

    # coerce to ndarray
    param_arr = np.asarray(param, dtype=np.float64)

    # If param_arr ended up 0-D (numpy scalar), treat like scalar
    if param_arr.ndim == 0:
        return np.full(src_len, float(param_arr), dtype=np.float64)

    # Reject 2-D or higher arrays — callers should pass a column or 1-D array
    if param_arr.ndim != 1:
        raise ValueError("param must be scalar or 1-D array/series")

    p_len = param_arr.shape[0]
    # If param has at least src_len elements, take last src_len
    if p_len >= src_len:
        return param_arr[-src_len:].astype(np.float64, copy=False)

    # p_len < src_len -> left-pad with NaNs so tails align
    pad = np.full(src_len - p_len, np.nan, dtype=np.float64)
    return np.concatenate((pad, param_arr)).astype(np.float64)


def _generatedseries_calculate_add_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) + _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_subtract_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) - _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_multiply_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) * _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_divide_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) / _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_power_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.power(np.asarray(source, dtype=np.float64), _prepare_param_for_op( param, source.shape[0], dataset ))

def _generatedseries_calculate_equal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) == _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_notequal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) != _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_greater_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) > _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_greaterorequal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) >= _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_less_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) < _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_lessequal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) <= _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_bitwise_and_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param= None) -> np.ndarray:
    param_array = _prepare_param_for_op( param, source.shape[0], dataset )
    int_result = np.bitwise_and(source.astype(np.int64), param_array.astype(np.int64)) # to int
    return int_result.astype(np.float64) # back to float

def _generatedseries_calculate_bitwise_or_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param) -> np.ndarray:
    param_array = _prepare_param_for_op( param, source.shape[0], dataset )
    int_result = np.bitwise_or(source.astype(np.int64), param_array.astype(np.int64))
    return int_result.astype(np.float64)

def _generatedseries_calculate_bitwise_xor_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param) -> np.ndarray:
    param_array = _prepare_param_for_op( param, source.shape[0], dataset )
    int_result = np.bitwise_xor(source.astype(np.int64), param_array.astype(np.int64))
    return int_result.astype(np.float64)

def _generatedseries_calculate_bitwise_not_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param) -> np.ndarray:
    int_result = np.bitwise_not(source.astype(np.int64))
    return int_result.astype(np.float64)

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

def _generatedseries_calculate_scalar_bitwise_and_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    """
    Performs element-wise bitwise AND operation: scalar & series.
    Casts result to np.float64 for storage compatibility.
    """
    int_result = np.bitwise_and(param, source.astype(np.int64)) # to int
    return int_result.astype(np.float64) # back to float

def _generatedseries_calculate_scalar_bitwise_or_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    """
    Performs element-wise bitwise OR operation: scalar | series.
    Casts operands to np.int64 for the bitwise operation, and the result back to np.float64.
    """
    int_result = np.bitwise_or(param, source.astype(np.int64))
    return int_result.astype(np.float64)

def _generatedseries_calculate_scalar_bitwise_xor_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    """
    Performs element-wise bitwise XOR operation: scalar ^ series.
    Casts operands to np.int64 for the bitwise operation, and the result back to np.float64.
    """
    int_result = np.bitwise_xor(param, source.astype(np.int64))
    return int_result.astype(np.float64)

def _generatedseries_calculate_abs_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: None) -> np.ndarray:
    return np.abs(source)



################################ ANALYSIS TOOLS #####################################


# _highest250. Elapsed time: 0.00 seconds
def _generatedseries_calculate_highest(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    """
    Calculates the highest value over a specified period using NumPy.
    """
    if talib_available:
        return talib.MAX(source, period)
    source = np.asarray(source, dtype=np.float64)
    return _rolling_window_apply_optimized(source, period, lambda x: np.nanmax(x))

def _generatedseries_calculate_lowest(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    """
    Calculates the lowest value over a specified period using NumPy.
    """
    if talib_available:
        return talib.MIN(source, period)
    source = np.asarray(source, dtype=np.float64)
    return _rolling_window_apply_optimized(source, period, lambda x: np.nanmin(x))

# _highestbars250. Elapsed time: 0.01 seconds
def _generatedseries_calculate_highestbars(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)

    def nan_safe_argmax(a, window_len):
        if np.all(np.isnan(a)):
            return np.nan
        return (window_len - 1) - np.nanargmax(a)

    return _rolling_window_apply_optimized(source, period, lambda x: nan_safe_argmax(x, period))

# _lowestbars250. Elapsed time: 0.01 seconds
def _generatedseries_calculate_lowestbars(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)

    def nan_safe_argmin(a, window_len):
        if np.all(np.isnan(a)):
            return np.nan
        return (window_len - 1) - np.nanargmin(a)

    return _rolling_window_apply_optimized(source, period, lambda x: nan_safe_argmin(x, period))

# _falling250. Elapsed time: 0.01 seconds
def _generatedseries_calculate_falling(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    n = len(source)

    if period < 1 or period > n:
        return np.full_like(source, np.nan, dtype=np.float64)

    diffs = np.concatenate(([np.nan], np.diff(source)))
    
    window_for_diffs = period - 1

    if window_for_diffs < 1: # If period is 1, a single value is trivially "falling" if not NaN
        result = ~np.isnan(source) # If period is 1, it's falling if it's not NaN
        return result.astype(np.float64)

    if len(diffs[1:]) < window_for_diffs:
        return np.full_like(source, np.nan, dtype=np.float64)

    windows_of_diffs = sliding_window_view(diffs[1:], window_shape=window_for_diffs)

    # Check if all elements in each window are strictly negative
    all_negative = np.all(windows_of_diffs < 0, axis=1)

    result_array = np.full(n, np.nan, dtype=np.float64)
    result_array[period - 1:] = all_negative.astype(np.float64)

    return result_array

# _rising250. Elapsed time: 0.01 seconds
def _generatedseries_calculate_rising(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    """
    Calculates a boolean series indicating if the source has been strictly rising
    over the given period.
    """
    source = np.asarray(source, dtype=np.float64)
    n = len(source)

    if period < 1 or period > n:
        return np.full_like(source, np.nan, dtype=np.float64)

    diffs = np.concatenate(([np.nan], np.diff(source)))

    window_for_diffs = period - 1
    
    if window_for_diffs < 1: # If period is 1, a single value is trivially "rising" if not NaN
        result = ~np.isnan(source) # If period is 1, it's rising if it's not NaN
        return result.astype(np.float64)

    # Create sliding window view on `diffs` starting from the second element
    if len(diffs[1:]) < window_for_diffs:
        return np.full_like(source, np.nan, dtype=np.float64)

    windows_of_diffs = sliding_window_view(diffs[1:], window_shape=window_for_diffs)
    all_positive = np.all(windows_of_diffs > 0, axis=1)

    result_array = np.full(n, np.nan, dtype=np.float64)
    result_array[period - 1:] = all_positive.astype(np.float64)

    # Convert to boolean, NaNs will remain as NaN, althought they will be converted to float64 in the dataset
    return result_array

#
def _generatedseries_calculate_barssince(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    # Get array of indices where condition is True. A "True" value is non-zero and not NaN.
    true_mask = (series != 0) & ~np.isnan(series)
    true_indices = np.where(true_mask)[0]
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
    # A "True" value is non-zero and not NaN. `if val:` is truthy for np.nan, which is a bug.
    true_mask = (series != 0) & ~np.isnan(series)
    true_indices = np.where(true_mask)[0]

    if len(true_indices) == 0:
        return np.full_like(series, np.nan, dtype=np.float64)

    all_indices = np.arange(len(series))
    # Find the index of the last `true_index` that is <= each `all_indices`.
    insertion_indices = np.searchsorted(true_indices, all_indices, side='right')
    indices_into_true_indices = insertion_indices - 1

    out = np.full(len(series), np.nan, dtype=np.float64)
    valid_mask = indices_into_true_indices >= 0
    out[valid_mask] = true_indices[indices_into_true_indices[valid_mask]]
    return out

#
def _generatedseries_calculate_indexwhenfalse(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    # A "False" value is 0. NaN is not considered False. `series == 0` correctly handles NaN.
    false_indices = np.where(series == 0)[0]

    if len(false_indices) == 0:
        return np.full_like(series, np.nan, dtype=np.float64)

    all_indices = np.arange(len(series))
    # Find the index of the last `false_index` that is <= each `all_indices`.
    insertion_indices = np.searchsorted(false_indices, all_indices, side='right')
    indices_into_false_indices = insertion_indices - 1

    out = np.full(len(series), np.nan, dtype=np.float64)
    valid_mask = indices_into_false_indices >= 0
    out[valid_mask] = false_indices[indices_into_false_indices[valid_mask]]
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

def _generatedseries_calculate_min_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.minimum(np.asarray(source, dtype=np.float64), _prepare_param_for_op( param, source.shape[0], dataset ))

def _generatedseries_calculate_max_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.maximum(np.asarray(source, dtype=np.float64), _prepare_param_for_op( param, source.shape[0], dataset ))

def _generatedseries_calculate_shift(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None)->np.ndarray:
    """
    Calculates the element-wise shifted (lagged/leaded) version of the source series.

    Args:
        source (np.ndarray): The base series data to be shifted.
        period (int): The number of bars to shift.
                      Positive (e.g., 1): Lag (looks to the past, fills start with NaN).
                      Negative (e.g., -1): Lead (looks to the future, fills end with NaN).
        dataset, cindex, param: Standard parameters (unused for a simple shift).
    
    Returns:
        np.ndarray: The resulting shifted series.
    """
    shift = period
    n = len(source)
    # Initialize the result array with NaNs, same shape as source
    result = np.full_like(source, np.nan, dtype=np.float64)

    if shift == 0:
        return source.copy()
    
    # Positive shift (Lag: C[i] = C[i-shift])
    if shift > 0:
        result[shift:] = source[:-shift]
    
    # Negative shift (Lead: C[i] = C[i-shift]) where shift is negative
    elif shift < 0:
        abs_shift = abs(shift)
        result[:-abs_shift] = source[abs_shift:]
        
    return result

def _generatedseries_calculate_sum(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    return _rolling_window_apply_optimized(source, period, np.nansum)

#
def _generatedseries_calculate_sma(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    if period < 1 or period > source.shape[0]:
        return np.full_like(source, np.nan)

    sma = np.full_like(source, np.nan)
    cumsum = np.nancumsum(np.insert(source, 0, 0))
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
            if np.isnan(result[i - 1]):
                result[i] = series[i] # Restart EMA if previous value was NaN
            else:
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
    ema1 = _generatedseries_calculate_ema(series, period, cindex, dataset)

    # Calculate EMA of EMA
    ema2 = _generatedseries_calculate_ema(ema1, period, cindex, dataset)

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
    
    # Find the first window with enough data
    first_valid_window_idx = -1
    for i in range(period - 1, length):
        window = series[i - period + 1 : i + 1]
        if not np.all(np.isnan(window)):
            rma[i] = np.nanmean(window)
            first_valid_window_idx = i
            break
            
    if first_valid_window_idx == -1:
        return rma # Return all NaNs if no valid window found

    # Compute RMA iteratively
    alpha = 1.0 / period
    one_minus_alpha = 1.0 - alpha
    for i in range(first_valid_window_idx + 1, length):
        if np.isnan(series[i]):
            rma[i] = np.nan
            continue
        
        if np.isnan(rma[i - 1]):
            # Previous RMA is NaN, try to re-seed with a new SMA
            window = series[i - period + 1 : i + 1]
            rma[i] = np.nanmean(window)
        else:
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
    weighted_sums = np.nansum(windows * weights, axis=1)  # Element-wise multiplication and sum
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
    y_sum = np.nansum(windows, axis=1)
    ty_sum = np.nansum(windows * t, axis=1)

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
    sma_values = _generatedseries_calculate_sma(source, period, -1, dataset)

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
    current_input_len = len(series)
    
    if current_input_len < period: 
        return np.full(current_input_len, np.nan)
    
    # Derive corresponding slices for high, low from the full dataset
    full_dataset_len = dataset.shape[0]
    # Calculate the start index in the full dataset for the current series slice
    start_index_in_dataset = max(0, full_dataset_len - current_input_len)
    end_index_in_dataset = start_index_in_dataset + current_input_len

    high_slice = dataset[start_index_in_dataset:end_index_in_dataset, c.DF_HIGH]
    low_slice = dataset[start_index_in_dataset:end_index_in_dataset, c.DF_LOW]
    close_slice = series # 'series' here is already the slice of close price

    if talib_available:
        # Pass the sliced data to talib. It should produce the correct result for this slice.
        # If talib.CCI behaves unexpectedly with slices (e.g., assumes full history),
        # this might still behave like a full recalculation internally within talib,
        # but the wrapper passes only the relevant data.
        return talib.CCI(high_slice, low_slice, close_slice, period)

    # Compute Typical Price using the slices
    tp_slice = (high_slice + low_slice + close_slice) / 3.0

    # Create sliding windows on the derived tp_slice
    # The output of sliding_window_view will have length (len(tp_slice) - period + 1)
    tp_windows = sliding_window_view(tp_slice, window_shape=period)

    # Compute SMA and MAD over these windows
    sma_values = np.nanmean(tp_windows, axis=1)
    mad_values = np.nanmean(np.abs(tp_windows - sma_values[:, np.newaxis]), axis=1)

    # Compute CCI on the calculated rolling values
    cci_calculated = np.full(len(sma_values), np.nan) # Initialize with NaN, length (current_input_len - period + 1)
    denominator = 0.015 * mad_values
    
    # Perform element-wise division. np.where handles division by zero or NaN denominator.
    cci_calculated = np.where(denominator > 1e-10, (tp_slice[period - 1:] - sma_values) / denominator, np.nan)

    # Pad with NaNs at the beginning to match the original input slice length
    result_with_padding = np.concatenate((np.full(period - 1, np.nan), cci_calculated))

    return result_with_padding

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
    y_sum = np.nansum(windows, axis=1)  # Sum of y_i for each window
    ty_sum = np.nansum(windows * t, axis=1)  # Sum of t_i * y_i for each window

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
def _generatedseries_calculate_cmo(source: np.ndarray, period: int, dataset: np.ndarray, cindex: int, param) -> np.ndarray:
    # if talib_available:
    #     return talib.CMO( source, period ) # It returns a different result
    
    src = np.asarray(source, dtype=np.float64)
    length = int(period) # Ensure length is an integer

    n = len(src)
    if n == 0:
        return np.array([], dtype=np.float64)

    # 1. Calculate momentum (momm = src - src[1])
    # np.diff returns an array of length N-1. Prepend a NaN to align with the original series length.
    momm = np.concatenate(([np.nan], np.diff(src)))

    # 2. Calculate m1 (positive momentum) and m2 (absolute negative momentum)
    m1 = np.maximum(momm, 0.0)
    m2 = np.maximum(-momm, 0.0) # Using -momm to get the positive value of negative changes

    # 3. Calculate rolling sums sm1 and sm2 over 'length' periods
    # _rolling_window_apply_optimized handles NaN padding at the start.
    sm1 = _rolling_window_apply_optimized(m1, length, lambda x, axis: np.nansum(x, axis=axis))
    sm2 = _rolling_window_apply_optimized(m2, length, lambda x, axis: np.nansum(x, axis=axis))

    # 4. Calculate Chande Momentum Oscillator: 100 * (sm1 - sm2) / (sm1 + sm2)
    denominator = sm1 + sm2
    
    # Use np.where to handle division by zero or where denominator is NaN:
    # If denominator is 0.0 or NaN, the result for that point will be NaN.
    cmo_series = np.where(
        (denominator == 0.0) | np.isnan(denominator), # Check for zero OR NaN in denominator
        np.nan, # Result is NaN if denominator is zero or NaN
        100.0 * (sm1 - sm2) / denominator # Otherwise, perform the calculation
    )

    return cmo_series

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
def _generatedseries_calculate_stdev(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param:float=1.0) -> np.ndarray:
    if talib_available:
        return talib.STDDEV(series, period) * param

    length = len(series)
    if length < period:
        return np.full(length, np.nan)
    
    ddof = 0
    
    # Single window case (incremental updates)
    if length == period:
        # Calculate std dev directly without rolling windows
        return np.nanstd(series, ddof=ddof) * param

    # Create rolling windows (full dataset initialization case)
    windows = sliding_window_view(series, window_shape=period)

    # Compute sample standard deviation (ddof=0) for each window
    # Use nan-aware std to be robust to NaNs in incremental/update slices
    try:
        stdev = np.nanstd(windows, axis=1, ddof=ddof) * param
    except Exception:
        # Fallback: compute per-window to avoid shape issues
        stdev = np.array([np.nanstd(w, ddof=ddof) * param for w in windows], dtype=np.float64)

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
    means = np.nanmean(windows, axis=1)

    # Compute mean absolute deviation: Σ(|x - mean|) / period
    abs_deviations = np.abs(windows - means[:, np.newaxis])
    dev = np.nansum(abs_deviations, axis=1) / period

    # Pad with NaNs for the first period - 1 values
    result = np.full(length, np.nan)
    result[period - 1:] = dev

    return result

# _wpr250. Elapsed time: 0.01 seconds (talib 0.0 secods)
def _generatedseries_calculate_williams_r(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    current_input_len = len(series)
    
    if current_input_len < period:
        return np.full(current_input_len, np.nan)

    full_dataset_len = dataset.shape[0]
    start_index_in_dataset = max(0, full_dataset_len - current_input_len)
    end_index_in_dataset = start_index_in_dataset + current_input_len

    high_slice = dataset[start_index_in_dataset:end_index_in_dataset, c.DF_HIGH]
    low_slice = dataset[start_index_in_dataset:end_index_in_dataset, c.DF_LOW]
    close_slice = series # 'series' is already the correct close price slice

    if talib_available:
        return talib.WILLR(high_slice, low_slice, close_slice, period)
    
    # Compute rolling highest high and lowest low over the slices
    high_windows = sliding_window_view(high_slice, window_shape=period)
    low_windows = sliding_window_view(low_slice, window_shape=period)
    highest_high = np.nanmax(high_windows, axis=1)
    lowest_low = np.nanmin(low_windows, axis=1)

    # Compute Williams %R using the derived slices
    # Align close_slice with window ends for calculation
    # The length of highest_high and lowest_low is (current_input_len - period + 1)
    # So close_slice also needs to be sliced to align for element-wise operations.
    numerator = highest_high - close_slice[period - 1:]
    denominator = highest_high - lowest_low
    
    williams_r_calculated = np.full(len(numerator), np.nan) # length (current_input_len - period + 1)
    williams_r_calculated = np.where( (denominator != 0) & (~np.isnan(denominator)), (numerator / denominator) * -100, np.nan)

    # Pad with NaNs at the beginning to match the original input slice length
    result_with_padding = np.concatenate((np.full(period - 1, np.nan), williams_r_calculated))

    return result_with_padding

# _tr250. Elapsed time: 0.00 seconds 
def _generatedseries_calculate_tr(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    length = dataset.shape[0]
    if length < period:
        return np.full(length, np.nan)
    
    high = None; low = None
    if isinstance(param, tuple) and len(param) == 2:
        high, low = param

    if high == None: high = dataset[:, c.DF_HIGH]
    if low == None: low = dataset[:, c.DF_LOW]
    high = _prepare_param_for_op( high, len(series), dataset )
    low = _prepare_param_for_op( low, len(series), dataset )
    close = series

    if talib_available:
        return talib.TRANGE(high, low, close)

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
    length = dataset.shape[0]
    if length < period:
        return np.full(length, np.nan)
    
    if talib_available:
        if isinstance(param, tuple) and len(param) == 2:
            high, low = param
            assert(type(high)==generatedSeries_c and type(low)==generatedSeries_c)
            high = dataset[:, high.column_index]
            low = dataset[:, low.column_index]
            series = dataset[:, c.DF_CLOSE]
        else:
            high = dataset[:, c.DF_HIGH]
            low = dataset[:, c.DF_LOW]
            series = dataset[:, c.DF_CLOSE]
        return talib.ATR(high, low, series, period)
    
    # Compute RMA of True Range
    tr = _generatedseries_calculate_tr(series, period, dataset, cindex, param)
    atr = _generatedseries_calculate_rma(tr, period, dataset, cindex)

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
    y_mean = np.nanmean(y, axis=1)[:, np.newaxis]  # Shape: (length - period + 1, 1)
    y_centered = y - y_mean  # Shape: (length - period + 1, period)
    numerator = np.nansum(y_centered * x_centered, axis=1)  # Shape: (length - period + 1,)
    
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
    highest = np.nanmax(windows, axis=1)
    lowest = np.nanmin(windows, axis=1)

    # Pad with NaNs at the beginning to match original length
    highest_padded = np.concatenate([np.full(period - 1, np.nan), highest])
    lowest_padded = np.concatenate([np.full(period - 1, np.nan), lowest])

    # Step 2: Calculate R
    R = highest_padded - lowest_padded

    # Step 3: Compute absolute change
    change = np.abs(np.diff(series, prepend=series[0]))  # Prepend first value to maintain length

    # Step 4: Compute rolling sum of change and vhf
    change_windows = sliding_window_view(change, window_shape=period)
    rolling_sum_change = np.nansum(change_windows, axis=1)
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
    length = len(series)
    if length < period + 1:
        return np.full(length, np.nan)
    
    if talib_available:
        return talib.RSI(series, period)

    # Step 1: Compute price changes
    delta = np.diff(series, prepend=series[0])  # Prepend first value to maintain length

    # Step 2: Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    # Step 3: Initialize SMMA with simple moving average for first period
    gains_windows = sliding_window_view(gains, window_shape=period)
    losses_windows = sliding_window_view(losses, window_shape=period)
    
    avg_gain_initial = np.nanmean(gains_windows[0], axis=-1)
    avg_loss_initial = np.nanmean(losses_windows[0], axis=-1)

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
        min_ = np.nanmin(window)
        max_ = np.nanmax(window)
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
    fish = _generatedseries_calculate_fisher(series, period, cindex, dataset)
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
        cumsum = np.nancumsum(np.insert(arr, 0, 0))
        ret[window-1:] = (cumsum[window:] - cumsum[:-window]) / window
        return ret

    sma_fast = sma(median_price, fast)
    sma_slow = sma(median_price, slow)
    ao = sma_fast - sma_slow
    return ao

#
def _generatedseries_calculate_br(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    """
    BR (Buying Pressure Ratio) -- NumPy implementation, optimized for incremental updates.
    BR = SUM(MAX(high - prev_close, 0), N) / SUM(MAX(prev_close - low, 0), N) * 100
    """
    current_input_len = len(series)

    # We need 'period' bars for rolling sum, and 1 extra for prev_close if starting from 0,
    # but the rolling_sum helper handles `window > len(arr)` by returning NaNs.
    # The 'series' itself is the current close slice.
    
    # Derive corresponding slices for high, low from the full dataset
    full_dataset_len = dataset.shape[0]
    start_index_in_dataset = max(0, full_dataset_len - current_input_len)
    end_index_in_dataset = start_index_in_dataset + current_input_len

    high_slice = dataset[start_index_in_dataset:end_index_in_dataset, c.DF_HIGH]
    low_slice = dataset[start_index_in_dataset:end_index_in_dataset, c.DF_LOW]
    close_slice = series # This is already the close price slice

    # Construct prev_close_slice
    prev_close_slice = np.full_like(close_slice, np.nan)
    if start_index_in_dataset > 0:
        # The first element of prev_close_slice is the close of the bar just before current_input_len started
        prev_close_slice[0] = dataset[start_index_in_dataset - 1, c.DF_CLOSE]
    # The rest of prev_close_slice comes from shifting the current close_slice
    prev_close_slice[1:] = close_slice[:-1]
    
    br_num = np.maximum(high_slice - prev_close_slice, 0)
    br_den = np.maximum(prev_close_slice - low_slice, 0)

    # Re-use the existing rolling_sum helper function (defined locally for now)
    def rolling_sum(arr, window):
        ret = np.full_like(arr, np.nan, dtype=np.float64)
        if window > len(arr):
            return ret
        cumsum = np.nancumsum(np.insert(arr, 0, 0))
        ret[window-1:] = cumsum[window:] - cumsum[:-window]
        return ret

    # These rolling sums will be calculated over the derived slices
    sum_num = rolling_sum(br_num, period)
    sum_den = rolling_sum(br_den, period)
    
    # Calculate BR. The length of sum_num and sum_den is (current_input_len - period + 1)
    br_calculated = np.full(len(sum_num), np.nan)
    # Handle division by zero or NaN denominator
    br_calculated = np.where( (sum_den != 0) & (~np.isnan(sum_den)), (sum_num / sum_den) * 100, np.nan)

    # Pad with NaNs at the beginning to match the original input slice length
    # The padding length should be period - 1 to align with rolling window output
    result_with_padding = np.concatenate((np.full(period - 1, np.nan), br_calculated))

    return result_with_padding

#
def _generatedseries_calculate_ar(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    """
    AR (Active Ratio) -- NumPy implementation, optimized for incremental updates.
    AR = SUM(high - open, N) / SUM(open - low, N) * 100
    """
    current_input_len = len(series) # This is the length of the slice (e.g., self.period)

    if current_input_len < period:
        return np.full(current_input_len, np.nan)

    full_dataset_len = dataset.shape[0]
    start_index_in_dataset = max(0, full_dataset_len - current_input_len)
    end_index_in_dataset = start_index_in_dataset + current_input_len

    high_slice = dataset[start_index_in_dataset:end_index_in_dataset, c.DF_HIGH]
    low_slice = dataset[start_index_in_dataset:end_index_in_dataset, c.DF_LOW]
    open_slice = dataset[start_index_in_dataset:end_index_in_dataset, c.DF_OPEN]

    ar_num_slice = high_slice - open_slice
    ar_den_slice = open_slice - low_slice

    # Re-use the existing rolling_sum helper function (defined locally for now)
    def rolling_sum(arr, window):
        ret = np.full_like(arr, np.nan, dtype=np.float64)
        if window > len(arr):
            return ret
        cumsum = np.nancumsum(np.insert(arr, 0, 0))
        ret[window-1:] = cumsum[window:] - cumsum[:-window]
        return ret

    # These rolling sums will be calculated over the derived slices
    sum_num = rolling_sum(ar_num_slice, period)
    sum_den = rolling_sum(ar_den_slice, period)
    
    # Calculate AR. The length of sum_num and sum_den is (current_input_len - period + 1)
    ar_calculated = np.full(len(sum_num), np.nan)
    # Handle division by zero or NaN denominator
    ar_calculated = np.where( (sum_den != 0) & (~np.isnan(sum_den)), (sum_num / sum_den) * 100, np.nan)

    # Pad with NaNs at the beginning to match the original input slice length
    # The padding length should be period - 1 to align with rolling window output
    result_with_padding = np.concatenate((np.full(period - 1, np.nan), ar_calculated))

    return result_with_padding

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
        denominator = np.nansum(window)
        if denominator == 0:
            cg[i] = np.nan
        else:
            cg[i] = np.nansum(window * weights) / denominator
    return cg

#
def _generatedseries_calculate_stoch_k(source_close: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
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
    hh_values = _rolling_window_apply_optimized(high_values_slice, period, lambda x: np.nanmax(x))
    ll_values = _rolling_window_apply_optimized(low_values_slice, period, lambda x: np.nanmin(x))

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

#
def _generatedseries_calculate_obv(source: np.ndarray, period: int, dataset: np.ndarray, cindex: int, param=None) -> np.ndarray:
    """
    Calculate On-Balance Volume (OBV) using numpy, optimized for full and incremental updates.

    Args:
        source (np.ndarray): Close prices (dataset[:, c.DF_CLOSE] or slice [-2:]).
        period (int): Set to 2 (for close[i], close[i-1]).
        dataset (np.ndarray): 2D array with columns [..., c.DF_VOLUME, ...].
        cindex (int): Output column index (-1 for init, >=0 for update).
        param (int): Volume column index (c.DF_VOLUME).

    Returns:
        np.ndarray: OBV values, full array or single value.
    """
    # oddly enough, talib is slower on this one
    # if talib_available:
    #     return talib.OBV( dataset[:, c.DF_CLOSE], dataset[:, c.DF_VOLUME] )

    if dataset.shape[0] == 0 or len(source) == 0:
        return np.array([], dtype=np.float64)

    try:
        # Get volume column
        volume = dataset[:, c.DF_VOLUME]
        length = dataset.shape[0]
        source_length = len(source)

        # Initialize output
        result = np.full(length, np.nan, dtype=np.float64)

        # Determine if update mode (source is slice, period=2)
        is_update = source_length >= 2 and cindex >= 0 and cindex < dataset.shape[1]
        barindex = length - 1

        # Update mode: compute single new OBV value
        if is_update:
            if barindex < 0 or source_length < 2:
                return np.array([np.nan], dtype=np.float64)
            if np.any(np.isnan(source)) or np.isnan(volume[barindex]):
                return np.array([np.nan], dtype=np.float64)
            
            # Get previous OBV
            prev_obv = 0.0 if barindex == 0 else dataset[barindex-1, cindex]
            if barindex > 0 and np.isnan(prev_obv):
                return np.array([np.nan], dtype=np.float64)

            # Compute direction and new OBV
            direction = np.sign(source[-1] - source[-2])
            signed_volume = direction * volume[barindex]
            new_obv = prev_obv + signed_volume
            return np.array([new_obv], dtype=np.float64)

        # Full calculation: initialize or recompute
        if source_length != length:
            return np.full(length, np.nan, dtype=np.float64)

        # Set initial OBV
        result[0] = 0 if not np.isnan(source[0]) and not np.isnan(volume[0]) else np.nan
        if length == 1:
            return result

        # Determine start index for partial recalculation
        start_idx = 0
        prev_obv = 0.0
        if cindex >= 0 and cindex < dataset.shape[1] and length > 1:
            valid_obv = dataset[:-1, cindex]
            valid_mask = ~np.isnan(valid_obv)
            if np.any(valid_mask):
                start_idx = np.where(valid_mask)[0][-1] + 1
                prev_obv = valid_obv[start_idx - 1]

        # Compute signed volume
        close_diff = np.diff(source[start_idx:])
        direction = np.sign(close_diff)
        signed_volume = np.zeros(length - start_idx, dtype=np.float64)
        signed_volume[1:] = direction * volume[start_idx + 1:]

        # Handle NaNs
        valid_mask = ~np.isnan(source[start_idx:]) & ~np.isnan(volume[start_idx:])
        signed_volume[~valid_mask] = 0

        # Cumulative sum
        result[start_idx:] = np.cumsum(signed_volume, dtype=np.float64) + prev_obv
        result[start_idx:][~valid_mask] = np.nan

        return result
    except (IndexError, ValueError):
        return np.full(length, np.nan, dtype=np.float64)
    

# ... (your existing _generatedseries_calculate_ functions) ...

# --- Laguerre Oscillator Calculation ---
def _generatedseries_calculate_laguerre(source: np.ndarray, period: int, dataset: np.ndarray, cindex: int, param: float) -> np.ndarray:
    """
    Calculates the Laguerre Oscillator.

    Args:
        source (np.ndarray): The input price series (e.g., close prices).
        period (int): A dummy period, not directly used in Laguerre calculation,
                      but kept for consistent signature.
        dataset (np.ndarray): The full 2D dataset (timeframe.dataset).
                              Not directly used for Laguerre calculation but required by signature.
        cindex (int): The column index of the output series in the dataset.
                      Not directly used in this calculation, but required by signature.
        param (float): The 'gamma' factor for the Laguerre filter (0 to 1).

    Returns:
        np.ndarray: The calculated Laguerre Oscillator series (0 to 1).
    """
    price = np.asarray(source, dtype=np.float64)
    gamma = float(param) # The gamma factor is passed as 'param'
    n = len(price)

    if n == 0:
        return np.array([], dtype=np.float64)

    # Initialize Laguerre filter components
    L0 = np.full(n, np.nan, dtype=np.float64)
    L1 = np.full(n, np.nan, dtype=np.float64)
    L2 = np.full(n, np.nan, dtype=np.float64)
    L3 = np.full(n, np.nan, dtype=np.float64)
    
    laguerre_oscillator = np.full(n, np.nan, dtype=np.float64)

    # Find the first valid price to start calculation
    first_valid_idx = np.where(~np.isnan(price))[0]
    if len(first_valid_idx) == 0:
        return laguerre_oscillator # All NaNs

    start_idx = first_valid_idx[0]

    # Initialize L0, L1, L2, L3 at the first valid point
    L0[start_idx] = (1 - gamma) * price[start_idx]
    L1[start_idx] = -gamma * L0[start_idx] + L0[start_idx] 
    L2[start_idx] = -gamma * L1[start_idx] + L1[start_idx]
    L3[start_idx] = -gamma * L2[start_idx] + L2[start_idx]

    # Calculate for subsequent bars
    for i in range(start_idx + 1, n):
        if np.isnan(price[i]):
            # If current price is NaN, propagate NaNs for L0-L3 and Laguerre
            L0[i], L1[i], L2[i], L3[i] = np.nan, np.nan, np.nan, np.nan
            laguerre_oscillator[i] = np.nan
            continue

        # Safely get previous values, re-seed if NaN
        l0_prev = L0[i-1]
        l1_prev = L1[i-1]
        l2_prev = L2[i-1]
        l3_prev = L3[i-1]

        # Calculate L0
        if np.isnan(l0_prev):
            L0[i] = (1 - gamma) * price[i]
        else:
            L0[i] = (1 - gamma) * price[i] + gamma * l0_prev

        # Calculate L1
        if np.isnan(l0_prev) or np.isnan(l1_prev):
            L1[i] = -gamma * L0[i] + L0[i]
        else:
            L1[i] = -gamma * L0[i] + l0_prev + gamma * l1_prev

        # Calculate L2
        if np.isnan(l1_prev) or np.isnan(l2_prev):
            L2[i] = -gamma * L1[i] + L1[i]
        else:
            L2[i] = -gamma * L1[i] + l1_prev + gamma * l2_prev

        # Calculate L3
        if np.isnan(l2_prev) or np.isnan(l3_prev):
            L3[i] = -gamma * L2[i] + L2[i]
        else:
            L3[i] = -gamma * L2[i] + l2_prev + gamma * l3_prev

        # Calculate CU and CD
        cu = 0.0
        cd = 0.0

        if L0[i] > L1[i]: cu += L0[i] - L1[i]
        else: cd += L1[i] - L0[i]

        if L1[i] > L2[i]: cu += L1[i] - L2[i]
        else: cd += L2[i] - L1[i]
        
        if L2[i] > L3[i]: cu += L2[i] - L3[i]
        else: cd += L3[i] - L2[i]
        
        # Calculate Laguerre Oscillator
        if (cu + cd) == 0:
            laguerre_oscillator[i] = 0.0
        else:
            laguerre_oscillator[i] = cu / (cu + cd)

    return laguerre_oscillator



class generatedSeries_c:
    
    def __init__(self, name: str, source: np.ndarray, period:int= 1, func=None, param=None, always_reset:bool= False):

        self.timeframe: 'timeframe_c' = active.timeframe
        timeframe = self.timeframe
        
        # These are generated series created by the user, used by plots, or the built in (ohlcv) columns
        if func is None:
            self.name = name
            self.source_name = name
            self.column_index = -1
            self.period = 1 # ignore the period

            self.param = None
            self.func = None
            self.lastUpdatedTimestamp = timeframe.timestamp
            self.alwaysReset = False
            self._is_generated_series = False # do not touch. 
            
            # ohlcv (and top/bottom) are special. They don't need a column created.
            if self.name == 'timestamp' : self.column_index = c.DF_TIMESTAMP
            if self.name == 'open' : self.column_index = c.DF_OPEN
            if self.name == 'high' : self.column_index = c.DF_HIGH
            if self.name == 'low' : self.column_index = c.DF_LOW
            if self.name == 'close' : self.column_index = c.DF_CLOSE
            if self.name == 'volume' : self.column_index = c.DF_VOLUME
            if self.name == 'top' : self.column_index = c.DF_TOP
            if self.name == 'bottom' : self.column_index = c.DF_BOTTOM

            # create a column
            if self.column_index == -1:
                assert name not in timeframe.generatedSeries.keys(), f"A generatedSeries_c with the name '{name}' already exists"
                self.column_index = timeframe.dataset_createColumn()

            # register the column
            timeframe.generatedSeries[self.name] = self
            return
        
        if not isinstance( source, generatedSeries_c ):
            raise ValueError( f"Source must be 'generatedSeries_c' type [{name}] for series with a func" )

        testname = tools.generatedSeriesNameFormat(name, source, period)
        if testname in timeframe.generatedSeries.keys():
            raise ValueError( f"A generatedSeries_c with the name '{testname}' already exists" )
        self.name = testname
        self.column_index = -1
        self.source_name = source.name
        self.period = max(period, 1) if period is not None else len(source)
        self.param = param
        self.func = func
        self.lastUpdatedTimestamp = 0
        self.alwaysReset = always_reset
        self._is_generated_series = True # do not touch. 
        
        # create a column and register it
        self.column_index = timeframe.dataset_createColumn()
        timeframe.generatedSeries[self.name] = self
        
        
    def calculate_full( self, source ):
        if not self.func:
            raise ValueError( f"Tried to initialize {self.name} without a func" )
        
        # FIXME: All of these are overdone. They're temporary. Remove them
        assert isinstance(source, generatedSeries_c), f"Source {source.name} must be generatedSeries_c type"  # temporary while make sure everything is
        assert self.timeframe == active.timeframe
        assert self.column_index != -1
        assert self.name in self.timeframe.generatedSeries.keys()
        assert self.source_name in self.timeframe.generatedSeries.keys()
        
        if len(source) < self.period:
            return
        
        timeframe = self.timeframe

        if self.lastUpdatedTimestamp >= timeframe.timestamp:
            return

        start_time = time.time()

        # Call the func, which must now accept a 1D numpy array as the source and the 2D array as "dataset"
        # Expect func to return a 1D numpy array of values, aligned with the full dataset length
        array = timeframe.dataset[:,source.column_index]
        values = self.func(array, self.period, timeframe.dataset, self.column_index, self.param)
        if isinstance(values, (list, tuple)):
            values = np.array(values, dtype=np.float64)

        timeframe.dataset[:, self.column_index] = values

        # Update the timestamp from the last row
        barindex = len(timeframe.dataset) - 1
        self.lastUpdatedTimestamp = int(timeframe.dataset[barindex, c.DF_TIMESTAMP])

        if timeframe.stream.initializing:
            print(f"Initialized {self.name} ({self.column_index}). Elapsed time: {time.time() - start_time:.2f} seconds")

    def update( self, source ):
        if not self.func:
            self.lastUpdatedTimestamp = self.timeframe.timestamp
            return

        assert isinstance(source, generatedSeries_c), "Source must be generatedSeries_c type"  # temporary while make sure everything is
        assert self.timeframe == active.timeframe
        
        timeframe = self.timeframe

        # has this row already been updated?
        if self.lastUpdatedTimestamp >= timeframe.timestamp:
            return

        # if non existent or needs reset, initialize
        if self.alwaysReset or self.lastUpdatedTimestamp == 0:
            self.calculate_full(source)
            return

        # slice the required block for current calculation
        period_slice = timeframe.dataset[-self.period:, source.column_index]

        # func should return a 1D array or scalar; we want the most recent value
        newval = self.func(period_slice, self.period, timeframe.dataset, self.column_index, self.param)
        
        if isinstance(newval, (np.ndarray, list, tuple)):
            newval = newval[-1]
            
        timeframe.dataset[timeframe.barindex, self.column_index] = newval
        self.lastUpdatedTimestamp = timeframe.timestamp


    def iloc( self, index = -1 ):
        barindex = self.timeframe.barindex

        if self.timeframe != active.timeframe :
            timestamp = active.timeframe.timestamp + ( (index+1) * self.timeframe.timeframeMsec )
            return self.timeframe.valueAtTimestamp( self.name, timestamp )

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
        return self.__getitem__(self.timeframe.barindex)
    
    def series(self)->np.ndarray:
        try:
            return self.timeframe.dataset[:,self.column_index]
        except Exception as e:
            raise ValueError( "series() method couldn't produce an array" )
        
    def tolist(self)->list:
        return self.timeframe.dataset[:,self.column_index].tolist()
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            key = slice(start, stop, step)
        else:
            if key < 0:
                key = self.timeframe.barindex + 1 + key
            if key < 0 or key >= len(self.timeframe.dataset):
                return np.nan # Return NaN for out-of-bounds access
        
        return self.timeframe.dataset[key, self.column_index]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            # Handle slice keys like [0:5]
            start, stop, step = key.indices(len(self))
            key = slice(start, stop, step)
        elif key < 0:
            key = self.timeframe.barindex + 1 + key
        
        # Allow both scalar and array assignment
        self.timeframe.dataset[key, self.column_index] = value

    def __len__(self):
        return self.timeframe.dataset.shape[0]
    
    def _lenError(self, other):
        if isinstance(other, generatedSeries_c):
            if self.timeframe != other.timeframe:
                raise ValueError( f"Can't operate on series from different timeframes. {self.name} [{self.timeframe.timeframeStr}] - {other.name} [{other.timeframe.timeframeStr}]" )
        raise ValueError( f"Can't operate on series of different lengths. {self.name} != {other.name}" )
    
    def __add__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, NumericScalar)):
            if not isinstance(other, NumericScalar) and len(self) != len(other):
                self._lenError(other)
            return addSeries(self, other)
        raise ValueError( f"Can't add type {type(other)} to generatedSeries_c)" )

    def __radd__(self, other):
        if isinstance(other, (float, int)):
            return addScalar(other, self)
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, NumericScalar)):
            if not isinstance(other, NumericScalar) and len(self) != len(other):
                self._lenError(other)
            return subtractSeries(self, other)
        raise ValueError( f"Can't subtract type {type(other)} to generatedSeries_c)" )

    def __rsub__(self, other):
        if isinstance(other, (float, int)):
            return subtractScalar(other, self)
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, NumericScalar)):
            if not isinstance(other, NumericScalar) and len(self) != len(other):
                self._lenError(other)
            return multiplySeries(self, other)
        raise ValueError( f"Can't multiply type {type(other)} to generatedSeries_c)" )

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return multiplyScalar(other, self)
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, NumericScalar)):
            if not isinstance(other, NumericScalar) and len(self) != len(other):
                self._lenError(other)
            return divideSeries(self, other)
        raise ValueError( f"Can't divide type {type(other)} to generatedSeries_c)" )

    def __rtruediv__(self, other):
        if isinstance(other, (float, int)):
            return divideScalar(other, self)
        raise ValueError("rtruediv only defined for const / series")
    
    def __neg__(self):
        return multiplySeries(self, -1)

    def __lt__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, NumericScalar)):
            if not isinstance(other, NumericScalar) and len(self) != len(other):
                self._lenError(other)
            return lessSeries(self, other)
        raise ValueError("Unsupported operand type for <")
    
    def __rlt__(self, other):
        if isinstance(other, (float, int)):
            return lessScalar(other, self)
        raise ValueError("Unsupported reversed operand for <")

    def __le__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, NumericScalar)):
            if not isinstance(other, NumericScalar) and len(self) != len(other):
                self._lenError(other)
            return lessOrEqualSeries(self, other)
        raise ValueError("Unsupported operand type for <=")
    
    def __rle__(self, other):
        if isinstance(other, (float, int)):
            return lessOrEqualScalar(other, self)
        raise ValueError("Unsupported reversed operand for <=")

    def __gt__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, NumericScalar)):
            if not isinstance(other, NumericScalar) and len(self) != len(other):
                self._lenError(other)
            return greaterSeries(self, other)
        raise ValueError("Unsupported operand type for >")
    
    def __rgt__(self, other):
        if isinstance(other, (float, int)):
            return greaterScalar(self, other)
        raise ValueError("Unsupported reversed operand for >")

    def __ge__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, NumericScalar)):
            if not isinstance(other, NumericScalar) and len(self) != len(other):
                self._lenError(other)
            return greaterOrEqualSeries(self, other)
        raise ValueError("Unsupported operand type for >=")
    
    def __rge__(self, other):
        if isinstance(other, (float, int)):
            return greaterOrEqualScalar(self, other)
        raise ValueError("Unsupported reversed operand for >=")

    def __eq__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, NumericScalar)):
            if not isinstance(other, NumericScalar) and len(self) != len(other):
                self._lenError(other)
            return equalSeries(self, other)
        return NotImplemented
    
    def __req__(self, other):
        if isinstance(other, (float, int)):
            return equalScalar(self, other)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, NumericScalar)):
            if not isinstance(other, NumericScalar) and len(self) != len(other):
                self._lenError(other)
            return notequalSeries(self, other)
        return NotImplemented

    def __rne__(self, other):
        if isinstance(other, (float, int)):
            return notEqualScalar(self, other)
        return NotImplemented
    
    def __and__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, NumericScalar)):
            if not isinstance(other, NumericScalar) and len(self) != len(other):
                self._lenError(other) # Assuming _lenError is available
            return bitwiseAndSeries(self, other)
        # Delegate to standard Python if 'other' has a higher precedence __rand__ method
        return NotImplemented 

    def __rand__(self, other):
        if isinstance(other, NumericScalar):
            # Only handle if 'other' is a simple scalar, otherwise NotImplemented
            return bitwiseAndScalar(other, self)
        return NotImplemented
    
    def __or__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, NumericScalar)):
            if not isinstance(other, NumericScalar) and len(self) != len(other):
                self._lenError(other) # Assuming _lenError is available
            return bitwiseOrSeries(self, other)
        # Delegate to standard Python if 'other' has a higher precedence __rand__ method
        return NotImplemented 

    def __ror__(self, other):
        if isinstance(other, NumericScalar):
            # Only handle if 'other' is a simple scalar, otherwise NotImplemented
            return bitwiseOrScalar(other, self)
        return NotImplemented
    
    def __xor__(self, other):
        if isinstance(other, (generatedSeries_c, np.ndarray, NumericScalar)):
            if not isinstance(other, NumericScalar) and len(self) != len(other):
                self._lenError(other) # Assuming _lenError is available
            return bitwiseXorSeries(self, other)
        # Delegate to standard Python if 'other' has a higher precedence __rand__ method
        return NotImplemented 

    def __rxor__(self, other):
        if isinstance(other, NumericScalar):
            # Only handle if 'other' is a simple scalar, otherwise NotImplemented
            return bitwiseXorScalar(other, self)
        return NotImplemented

    def __invert__(self):
        return bitwiseNotSeries(self)
    
    def __abs__(self):
        return ABS(self)

    
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
            return ( previous_self_val <= previous_other_val and current_self_val >= current_other_val and not (previous_self_val == previous_other_val and current_self_val == current_other_val) )
        elif isinstance( other, (np.ndarray) ):
            # Use iloc directly from the pd.Series
            if len(other) < 2 or active.barindex < 1 or np.isnan(other[active.barindex-1]) or np.isnan(other[active.barindex]):
                return False
            return ( previous_self_val <= other[active.barindex-1] and current_self_val >= other[active.barindex] and not (previous_self_val == other[active.barindex-1] and current_self_val == other[active.barindex]) )
        else: # assuming float or int
            try:
                float_other = float(other)
            except ValueError:
                return False
            # Corrected line: Use previous_self_val in the last condition
            return ( previous_self_val <= float_other and current_self_val >= float_other and not (previous_self_val == float_other and current_self_val == float_other) )
    
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
            return ( previous_self_val >= previous_other_val and current_self_val <= current_other_val and not (previous_self_val == previous_other_val and current_self_val == current_other_val) )
        elif isinstance( other, (np.ndarray) ):
            if len(other) < 2 or active.barindex < 1 or np.isnan(other[active.barindex-1]) or np.isnan(other[active.barindex]):
                return False
            return ( previous_self_val >= other[active.barindex-1] and current_self_val <= other[active.barindex] and not (previous_self_val == other[active.barindex-1] and current_self_val == other[active.barindex]) )
        else: 
            try:
                float_other = float(other)
            except ValueError:
                return False
            return ( previous_self_val >= float_other and current_self_val <= float_other and not (previous_self_val == float_other and current_self_val == float_other) )
    
    def crossing( self, other ):
        return self.crossingUp(other) or self.crossingDown(other)
    


def _ensure_object_array( data: generatedSeries_c )-> generatedSeries_c:
    """
    Helper function to ensure the input is a series object.
    This replaces the initial type checking and conversion logic.
    """
    if isinstance(data, generatedSeries_c):
        return data
    elif isinstance(data, np.ndarray):
        raise TypeError( "_ensure_object_array: Numpy ndarray is not a valid object" )
    else:
        raise TypeError(f"Unsupported input type: {type(data)}. Expected generatedSeries_c.")



def addSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"add_{colA.column_index}_s{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"add_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")

    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_add_series, param= colB )

def subtractSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"sub_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"sub_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries( name, colA, 1, _generatedseries_calculate_subtract_series, param= colB )

def multiplySeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"mul_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"mul_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_multiply_series, param= colB)

def divideSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"div_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"div_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_divide_series, param= colB)

def powerSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"pow_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"pow_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_power_series, param= colB)

def equalSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"eq_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"eq_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_equal_series, param= colB)

def notequalSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"neq_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"neq_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_notequal_series, param= colB)

def greaterSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"gr_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"gr_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_greater_series, param= colB)

def greaterOrEqualSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"gre_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"gre_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_greaterorequal_series, param= colB)

def lessSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"lt_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"lt_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_less_series, param= colB)

def lessOrEqualSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"le_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"le_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_lessequal_series, param= colB)

def bitwiseAndSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if np.isscalar( colB ):
        name = f"and_{colA.column_index}_s{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"and_{colA.column_index}_{colB.column_index}"
        if len(colA) != len(colB):
            raise ValueError("Operands must have the same shape for element-wise bitwise AND.")

    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_bitwise_and_series, param= colB )

def bitwiseOrSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if np.isscalar( colB ):
        name = f"or_{colA.column_index}_s{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"or_{colA.column_index}_{colB.column_index}"
        if len(colA) != len(colB):
            raise ValueError("Operands must have the same shape for element-wise bitwise OR.")

    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_bitwise_or_series, param= colB )


def bitwiseXorSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if np.isscalar( colB ):
        name = f"xor_{colA.column_index}_s{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"xor_{colA.column_index}_{colB.column_index}"
        if len(colA) != len(colB):
            raise ValueError("Operands must have the same shape for element-wise bitwise XOR.")

    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_bitwise_xor_series, param= colB )

def bitwiseNotSeries(colA: generatedSeries_c) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    # The name only needs the index of the single operand
    name = f"not_{colA.column_index}"
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_bitwise_not_series, param= None )


#
########## SCALARS By SERIES
#


def addScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"add_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries( name, series, 1, _generatedseries_calculate_scalar_add_series, param= scalar )

def subtractScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"sub_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_subtract_series, param= scalar)

def multiplyScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"mul_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_multiply_series, param= scalar)

def divideScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"div_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_divide_series, param= scalar)

def powerScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"pow_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_power_series, param= scalar)

def equalScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"eq_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_equal_series, param= scalar)

def notEqualScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"neq_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_notequal_series, param= scalar)

def greaterScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"gt_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_greater_series, param= scalar)

def greaterOrEqualScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"ge_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_greaterorequal_series, param= scalar)

def lessScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"lt_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_less_series, param= scalar)

def lessOrEqualScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"le_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_lessequal_series, param= scalar)

def bitwiseAndScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"and_{scalar}_{series.column_index}" 
    return timeframe.calcGeneratedSeries( name, series, 1, _generatedseries_calculate_scalar_bitwise_and_series, param= scalar )

def bitwiseOrScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"or_{scalar}_{series.column_index}" 
    return timeframe.calcGeneratedSeries( name, series, 1, _generatedseries_calculate_scalar_bitwise_or_series, param= scalar )


def bitwiseXorScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"xor_{scalar}_{series.column_index}" 
    return timeframe.calcGeneratedSeries( name, series, 1, _generatedseries_calculate_scalar_bitwise_xor_series, param= scalar )

def ABS(source: generatedSeries_c)->generatedSeries_c:
    source = _ensure_object_array( source )
    timeframe = source.timeframe
    return timeframe.calcGeneratedSeries( "abs", source, 1, _generatedseries_calculate_abs_series )


###################### ANALITIC TOOLS #################################


def highest(source: generatedSeries_c, period: int) -> generatedSeries_c:
    return source.timeframe.calcGeneratedSeries('highest', source, period, _generatedseries_calculate_highest)

def lowest(source: generatedSeries_c, period: int) -> generatedSeries_c:
    return source.timeframe.calcGeneratedSeries('lowest', _ensure_object_array(source), period, _generatedseries_calculate_lowest)

def highestbars(source: generatedSeries_c, period: int) -> generatedSeries_c:
    return source.timeframe.calcGeneratedSeries('highestbars', _ensure_object_array(source), period, _generatedseries_calculate_highestbars)

def lowestbars(source: generatedSeries_c, period: int) -> generatedSeries_c:
    return source.timeframe.calcGeneratedSeries('lowestbars', _ensure_object_array(source), period, _generatedseries_calculate_lowestbars)

def falling( source: generatedSeries_c, period:int )->generatedSeries_c:
    return source.timeframe.calcGeneratedSeries( 'falling', _ensure_object_array(source), period, _generatedseries_calculate_falling )

def rising( source: generatedSeries_c, period:int )->generatedSeries_c:
    return source.timeframe.calcGeneratedSeries( 'rising', _ensure_object_array(source), period, _generatedseries_calculate_rising )

def barsSinceSeries(source: generatedSeries_c, period: int) -> generatedSeries_c:
    import inspect
    # Get caller info by going up 2 levels in the stack
    caller_frame = inspect.currentframe().f_back
    frame_info = inspect.getframeinfo(caller_frame)
    caller_id = f"{frame_info.function[:5]}{frame_info.lineno}"
    return source.timeframe.calcGeneratedSeries('barsSince'+caller_id, _ensure_object_array(source), period, _generatedseries_calculate_barssince)

def indexWhenTrueSeries(source: generatedSeries_c, period: int = None) -> generatedSeries_c:
    return source.timeframe.calcGeneratedSeries('indexwhentrue_series', _ensure_object_array(source), period, _generatedseries_calculate_indexwhentrue)

def indexWhenFalseSeries(source: generatedSeries_c, period: int) -> generatedSeries_c:
    return source.timeframe.calcGeneratedSeries('indexwhenfalse_series', _ensure_object_array(source), period, _generatedseries_calculate_indexwhenfalse)

def barsWhileTrueSeries(source: generatedSeries_c, period: int = None) -> generatedSeries_c:
    return source.timeframe.calcGeneratedSeries('barsWhileTrue', _ensure_object_array(source), period, _generatedseries_calculate_barswhiletrue)

def barsWhileFalseSeries(source: generatedSeries_c, period: int = None) -> generatedSeries_c:
    return source.timeframe.calcGeneratedSeries('barsWhileFalse', _ensure_object_array(source), period, _generatedseries_calculate_barswhilefalse)




########################## INDICATORS #################################

def MIN( colA: generatedSeries_c | NumericScalar, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    if isinstance( colA, NumericScalar ) and isinstance( colB, NumericScalar ):
        return min( colA, colB )
    
    if isinstance( colA, NumericScalar ): # swap them if the scalar is first
        scalar = colA
        colA = _ensure_object_array(colB)
        colB = scalar
    
    if isinstance( colB, NumericScalar ):
        name = f"min_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array(colB)
        name = f"min_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise operations.")
    
    timeframe = colA.timeframe
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_min_series, param= colB)

def MAX( colA: generatedSeries_c | NumericScalar, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    if isinstance( colA, NumericScalar ) and isinstance( colB, NumericScalar ):
        return min( colA, colB )
    
    if isinstance( colA, NumericScalar ): # swap them if the scalar is first
        scalar = colA
        colA = _ensure_object_array(colB)
        colB = scalar
    
    if isinstance( colB, NumericScalar ):
        name = f"min_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array(colB)
        name = f"min_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise operations.")
    
    timeframe = colA.timeframe
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_max_series, param= colB)

def SHIFT(source:generatedSeries_c, offset: int)->generatedSeries_c:
    """
    A positive 'offset' (e.g., 1) lags the series (C[i] takes the value of C[i-1]).
    A negative 'offset' (e.g., -1) leads the series (C[i] takes the value of C[i+1]).
    
    Args:
        source (generatedSeries_c): The input series to shift.
        offset (int): The number of bars to shift.
    
    Returns:
        generatedSeries_c: A new series representing the shifted values.
    """
    if offset == 0:
        return source
    return source.timeframe.calcGeneratedSeries( 'shift', _ensure_object_array(source), offset, _generatedseries_calculate_shift )

def SUM( source:generatedSeries_c, period:int )->generatedSeries_c:
    return source.timeframe.calcGeneratedSeries( 'sum', _ensure_object_array(source), period, _generatedseries_calculate_sum )

def SMA( source: generatedSeries_c, period: int )->generatedSeries_c:
    return source.timeframe.calcGeneratedSeries('sma', _ensure_object_array(source), period, _generatedseries_calculate_sma )

def EMA( source: generatedSeries_c, period:int )->generatedSeries_c:
    return source.timeframe.calcGeneratedSeries( "ema", _ensure_object_array(source), period, _generatedseries_calculate_ema, always_reset=True )

def DEMA( source: generatedSeries_c, period:int )->generatedSeries_c:
    return source.timeframe.calcGeneratedSeries( "dema", _ensure_object_array(source), period, _generatedseries_calculate_dema, always_reset=True )

def RMA( source:generatedSeries_c, period:int )->generatedSeries_c:
    return source.timeframe.calcGeneratedSeries( 'rma', _ensure_object_array(source), period, _generatedseries_calculate_rma, always_reset=True )

def WMA( source:generatedSeries_c, period:int )->generatedSeries_c:
    return source.timeframe.calcGeneratedSeries( "wma", _ensure_object_array(source), period, _generatedseries_calculate_wma )

def HMA( source:generatedSeries_c, period:int )->generatedSeries_c:
    """Hull Moving Average implementation using multiple calculation steps
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """

    source = _ensure_object_array(source)
    timeframe = source.timeframe
    
    # First calculate WMA with half period
    half_length = int(period / 2) 
    wma_half = timeframe.calcGeneratedSeries( "wma", source, half_length, _generatedseries_calculate_wma )
    
    # Calculate WMA with full period
    wma_full = timeframe.calcGeneratedSeries( "wma", source, period,  _generatedseries_calculate_wma )
    
    # Calculate 2 * WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA with sqrt(period)
    sqrt_period = int(np.sqrt(period))
    return timeframe.calcGeneratedSeries( "hma", raw_hma, sqrt_period, _generatedseries_calculate_wma )

def STDEV( source:generatedSeries_c, period:int, scalar:float = 1.0 )->generatedSeries_c:
    return source.timeframe.calcGeneratedSeries( 'stdev', _ensure_object_array(source), period, _generatedseries_calculate_stdev, param = scalar, always_reset= talib_available )

def DEV( source:generatedSeries_c, period:int )->generatedSeries_c:
    return source.timeframe.calcGeneratedSeries( 'dev', _ensure_object_array(source), period, _generatedseries_calculate_dev )

def WILLR( period:int )->generatedSeries_c:
    timeframe = active.timeframe
    source = timeframe.generatedSeries['close']
    return timeframe.calcGeneratedSeries( 'wpr', source, period, _generatedseries_calculate_williams_r )

def TR( period:int, high:generatedSeries_c= None, low:generatedSeries_c= None )->generatedSeries_c:
    timeframe = active.timeframe
    source = timeframe.generatedSeries['close']
    if high is None: high = timeframe.generatedSeries["high"]
    if low is None: low = timeframe.generatedSeries["low"]
    name = 'tr'
    if high.column_index != c.DF_HIGH or low.column_index != c.DF_LOW:
        name += f"{high.column_index}{low.column_index}"
    return timeframe.calcGeneratedSeries( name, source, period, _generatedseries_calculate_tr, param= (high, low) )

def ATR( period:int, high:generatedSeries_c= None, low:generatedSeries_c= None )->generatedSeries_c:
    timeframe = active.timeframe
    source = timeframe.generatedSeries['close']
    if high is None: high = timeframe.generatedSeries["high"]
    if low is None: low = timeframe.generatedSeries["low"]
    name = 'atr'
    if high.column_index != c.DF_HIGH or low.column_index != c.DF_LOW:
        name += f"{high.column_index}{low.column_index}"
    return timeframe.calcGeneratedSeries( name, source, period, _generatedseries_calculate_atr, param= (high, low), always_reset= True )  # rma requires always_reset, so atr also must

def SLOPE( source:generatedSeries_c, period:int )->generatedSeries_c:
    return source.timeframe.calcGeneratedSeries( 'slope', _ensure_object_array(source), period, _generatedseries_calculate_slope )

def VHMA(source: generatedSeries_c, period: int) -> generatedSeries_c:
    return source.timeframe.calcGeneratedSeries('vhma', _ensure_object_array(source), period, _generatedseries_calculate_vhma, always_reset= True)

def BIAS( source:generatedSeries_c, period:int )->generatedSeries_c:
    return source.timeframe.calcGeneratedSeries( 'bias', _ensure_object_array(source), period, _generatedseries_calculate_bias )

def LINREG( source:generatedSeries_c, period:int )->generatedSeries_c:
    return source.timeframe.calcGeneratedSeries( "linreg", _ensure_object_array(source), period, _generatedseries_calculate_linreg )

def CCI(period: int = 20) -> generatedSeries_c:
    if not isinstance(period, int ):
        raise ValueError( "CCI requires only a period argument" )
    timeframe = active.timeframe
    source = timeframe.generatedSeries['close']
    return timeframe.calcGeneratedSeries('cci', _ensure_object_array(source), period, _generatedseries_calculate_cci, always_reset= True)

def CFO( source:generatedSeries_c, period:int )->generatedSeries_c:
    return source.timeframe.calcGeneratedSeries( 'cfo', _ensure_object_array(source), period, _generatedseries_calculate_cfo )

def CMO(source: generatedSeries_c, period: int = 9) -> generatedSeries_c:
    return source.timeframe.calcGeneratedSeries('cmo', _ensure_object_array(source), period, _generatedseries_calculate_cmo)

def FWMA( source:generatedSeries_c, period:int )->generatedSeries_c:
    return source.timeframe.calcGeneratedSeries( 'fwma', _ensure_object_array(source), period, _generatedseries_calculate_fwma )

def RSI( source:generatedSeries_c, period:int )->generatedSeries_c:
    return source.timeframe.calcGeneratedSeries( 'rsi', _ensure_object_array(source), period, _generatedseries_calculate_rsi, always_reset=True )

def IFTrsi( source:generatedSeries_c, period:int )->generatedSeries_c:
    timeframe = _ensure_object_array(source).timeframe
    rsi = timeframe.calcGeneratedSeries( 'rsi', source, period, _generatedseries_calculate_rsi, always_reset=True )
    return timeframe.calcGeneratedSeries( 'iftrsi', rsi, period, _generatedseries_calculate_inverse_fisher_rsi )

def Fisher( period:int, signal:float=None )->tuple[generatedSeries_c, generatedSeries_c]:
    timeframe = active.timeframe
    source = timeframe.generatedSeries['close']
    fish = timeframe.calcGeneratedSeries( 'fisher', source, period, _generatedseries_calculate_fisher )
    sig = timeframe.calcGeneratedSeries( 'fishersig', source, period, _generatedseries_calculate_fisher_signal, signal )
    return fish, sig

def AO( fast: int = 5, slow: int = 34 ) -> generatedSeries_c:
    timeframe = active.timeframe
    param = (fast, slow)
    source = timeframe.generatedSeries['close']
    return timeframe.calcGeneratedSeries('ao', source, max(fast,slow), _generatedseries_calculate_ao, param)

def BR( period:int )->tuple[generatedSeries_c, generatedSeries_c]:
    timeframe = active.timeframe
    source = timeframe.generatedSeries['close']
    return timeframe.calcGeneratedSeries( 'br', source, period, _generatedseries_calculate_br )

def AR( period:int )->tuple[generatedSeries_c, generatedSeries_c]:
    timeframe = active.timeframe
    source = timeframe.generatedSeries['close']
    return timeframe.calcGeneratedSeries( 'ar', source, period, _generatedseries_calculate_ar )

def BRAR( period:int )->tuple[generatedSeries_c, generatedSeries_c]:
    timeframe = active.timeframe
    br = BR(period, timeframe)
    ar = AR(period, timeframe)
    return br, ar

def CG( source:generatedSeries_c, period:int )->generatedSeries_c:
    return source.timeframe.calcGeneratedSeries( 'cg', _ensure_object_array(source), period, _generatedseries_calculate_cg )

def STOCHk( source: generatedSeries_c, period:int )-> tuple[generatedSeries_c, generatedSeries_c]:
    return source.timeframe.calcGeneratedSeries( "stochk", _ensure_object_array(source), period, _generatedseries_calculate_stoch_k )

def OBV( timeframe=None ) -> generatedSeries_c:
    # period=2 because obv reads the previous value of close. It can not be anything else.
    source = timeframe.generatedSeries['close']
    return source.timeframe.calcGeneratedSeries( 'obv', source, 2, _generatedseries_calculate_obv )

def LAGUERRE(source: Union[str, generatedSeries_c], gamma: float = 0.7)->generatedSeries_c:
    return source.timeframe.calcGeneratedSeries( 'lagerre', _ensure_object_array(source), 1, _generatedseries_calculate_laguerre, gamma, always_reset=True )




# # #
# # # OTHER NOT GENERATED SERIES
# # #


def Stochastic(source: generatedSeries_c, k_period: int = 14, d_period: int = 3)-> tuple[generatedSeries_c, generatedSeries_c]:
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
    # Create the %K line generatedSeries_c
    k_line_series = STOCHk(source, k_period)

    # Create the %D line generatedSeries_c (SMA of %K)
    d_line_series = SMA( k_line_series, d_period )
    return k_line_series, d_line_series


def BollingerBands( source:generatedSeries_c, period:int, mult:float = 2.0 )->tuple[generatedSeries_c, generatedSeries_c, generatedSeries_c]:
    """
    Returns the Bollinger Bands (basis, upper, lower) for the given source series and period.

    Args:
        source (pd.Series): The input pandas Series to calculate the Bollinger Bands on.
        period (int): The period/window for the Bollinger Bands calculation.

    Returns:
        Tuple[generatedSeries_c, generatedSeries_c, generatedSeries_c]: The basis (SMA), upper band, and lower band as generatedSeries_c objects.
    """
    BBbasis = SMA(source, period)
    stdev = STDEV(source, period, mult)
    BBupper = BBbasis + stdev
    BBlower = BBbasis - stdev
    return BBbasis, BBupper, BBlower


def MACD( source:generatedSeries_c, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[generatedSeries_c, generatedSeries_c, generatedSeries_c]:
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
    # Calculate the fast and slow EMAs
    fast_ema = EMA(source, fast)
    slow_ema = EMA(source, slow)

    # MACD line: difference between fast and slow EMA
    macd_line = fast_ema - slow_ema

    # Signal line: EMA of the MACD line
    signal_line = EMA(macd_line, signal)

    # Histogram: MACD line - Signal line
    hist = macd_line - signal_line

    return macd_line, signal_line, hist


################ Helpers. Not series #########################

def indexWhenTrue(source: generatedSeries_c, lookback_period: int = None, since: int = None) -> Union[int, None]:
    """
    Finds the index of the last True value, searching backwards from a given point.
    A value is considered True if it's non-zero and not NaN.

    Args:
        source: A generatedSeries_c object.
        lookback_period (int, optional): The maximum number of bars to look back from the 'since' index.
                                         If None, searches the entire history up to 'since'. Defaults to None.
        since (int, optional): The index to start searching backwards from.
                               If None, starts from the end of the series. Defaults to None.

    Returns:
        int: The index of the last True value found, or None if not found.
    """
    source_array = _ensure_object_array(source).series()
    
    # Determine the end index of our search window
    end_index = (len(source_array) - 1) if since is None else since
    if end_index < 0 or end_index >= len(source_array):
        return None # 'since' index is out of bounds

    # Determine the start index of our search window
    if lookback_period is None:
        start_index = 0
    else:
        start_index = max(0, end_index - lookback_period + 1)
    
    # Create the slice to search within
    search_slice = source_array[start_index : end_index + 1]

    # Find True values within the slice (non-zero and not NaN)
    true_mask = (search_slice != 0) & ~np.isnan(search_slice)
    slice_true_indices = np.where(true_mask)[0]

    if slice_true_indices.size > 0:
        # Get the last true index within the slice, which is the first one found searching backwards
        last_true_index_in_slice = slice_true_indices[-1]
        # Convert it to an absolute index relative to the original source_array
        return start_index + last_true_index_in_slice
    else:
        return None

def indexWhenFalse(source: generatedSeries_c, lookback_period: int = None, since: int = None) -> Union[int, None]:
    """
    Finds the index of the last False value, searching backwards from a given point.
    A value is considered False if it is zero; NaN is not considered False.

    Args:
        source: A generatedSeries_c object.
        lookback_period (int, optional): The maximum number of bars to look back from the 'since' index. 
                                         If None, searches the entire history up to 'since'. Defaults to None.
        since (int, optional): The index to start searching backwards from. 
                               If None, starts from the end of the series. Defaults to None.

    Returns:
        int: The index of the last False value found, or None if not found.
    """
    source_array = _ensure_object_array(source).series()

    # Determine the end index of our search window
    end_index = (len(source_array) - 1) if since is None else since
    if end_index < 0 or end_index >= len(source_array):
        return None # 'since' index is out of bounds

    # Determine the start index of our search window
    if lookback_period is None:
        start_index = 0
    else:
        start_index = max(0, end_index - lookback_period + 1)
        
    # Create the slice to search within
    search_slice = source_array[start_index : end_index + 1]

    # A "False" value is 0. `search_slice == 0` correctly evaluates to False for NaN entries.
    false_indices_in_slice = np.where(search_slice == 0)[0]

    if false_indices_in_slice.size > 0:
        # Get the last false index within the slice, which is the first one found searching backwards
        last_false_index_in_slice = false_indices_in_slice[-1]
        # Convert it to an absolute index relative to the original source_array
        return start_index + last_false_index_in_slice
    else:
        return None
    
def barsSince( source, lookback: int = None ):
    """
    Calculates the number of bars that have passed since the last True condition.

    Args:
        source: A generatedSeries_c object.
        lookback (int, optional): The maximum number of bars to look back for the True condition. 
                                  If None, searches the entire available history. Defaults to None.

    Returns:
        int: The number of bars since the last True condition, or None if no True condition is found.
    """
    index_when_true = indexWhenTrue( source, lookback_period=lookback )
    if index_when_true is None: 
        return None
    return active.barindex - index_when_true

def barsWhileTrue( source, lookback: int = None ):
    """
    Calculates the number of consecutive bars (including the current one)
    for which the source condition has been True, looking back up to `lookback` bars.

    Args:
        source: A generatedSeries_c object.
        lookback (int, optional): The maximum number of bars to look back for a False condition.
                                  If None, searches the entire available history. Defaults to None.

    Returns:
        int: The number of consecutive True bars ending at the current position.
             Returns None if the source array is empty or the lookback period is invalid.
    """
    source_array = _ensure_object_array(source).series()
    current_bar_index = active.barindex # Assuming active.barindex is the current bar's index

    # indexWhenFalse will search backwards from current_bar_index
    index_of_last_false_in_window = indexWhenFalse( source, lookback_period=lookback, since=current_bar_index )

    if index_of_last_false_in_window is None:
        # If no False is found in the specified lookback window, it means the condition has been True
        # for the entire window up to the current bar.
        if lookback is not None:
            # The number of bars that were True is exactly the length of the effective search window.
            # This is min(lookback, current_bar_index + 1) because the lookback cannot exceed the actual history.
            return min(lookback, current_bar_index + 1)
        else:
            # If lookback is None, means it checked the whole history.
            # If no False was found, all bars up to current_bar_index were True.
            return current_bar_index + 1
    else:
        # If a False is found, count the bars from that False occurrence to the current bar.
        return current_bar_index - index_of_last_false_in_window

def crossingUp( self:generatedSeries_c|float, other:generatedSeries_c|float ):
    """
    Determines if 'self' crosses up over 'other' between the previous and current bar.

    Args:
        self: The first value/series.
        other: The second value/series to compare against.

    Returns:
        bool: True if a crossing up occurred, False otherwise.
    """
    if isinstance( self, generatedSeries_c ):
        return self.crossingUp(other)
    
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
    if isinstance( self, np.ndarray ):
        if( len(self) < 2 or active.barindex < 1 ):
            return False
        self_old = self[active.barindex-1]
        self_new = self[active.barindex]
        if isinstance( other, np.ndarray ):
            if( len(other) < 2 ):
                return False
            other_old = other[active.barindex-1]
            other_new = other[active.barindex]
        elif isinstance( other, generatedSeries_c ):
            # Directly use other.iloc(-1) and other.iloc(-2)
            if np.isnan(other.lastUpdatedTimestamp) or len(other) < 2 or active.barindex < 1 :
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

    return ( self_old <= other_old and self_new >= other_new and not (self_old == other_old and self_new == other_new) )


def crossingDown( self:generatedSeries_c|float, other:generatedSeries_c|float ):
    """
    Determines if 'self' crosses down below 'other' between the previous and current bar.

    Args:
        self: The first value/series.
        other: The second value/series to compare against.

    Returns:
        bool: True if a crossing down occurred, False otherwise.
    """
    if isinstance( self, generatedSeries_c ):
        return self.crossingDown(other)
    
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
    if isinstance( self, np.ndarray ):
        if( len(self) < 2 or active.barindex < 1 ):
            return False
        self_old = self[active.barindex-1]
        self_new = self[active.barindex]
        if isinstance( other, np.ndarray ):
            if( len(other) < 2 ):
                return False
            other_old = other[active.barindex-1]
            other_new = other[active.barindex]
        elif isinstance( other, generatedSeries_c ):
            if np.isnan(other.lastUpdatedTimestamp) or len(other) < 2 or active.barindex < 1 :
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

    return ( self_old >= other_old and self_new <= other_new and not (self_old == other_old and self_new == other_new) )

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
def pivots( high:generatedSeries_c, low:generatedSeries_c, amplitude: float = 1.0, reversal_percent: float = 32.0 )->pivots_c:
    global pivotsNow
    if pivotsNow == None:
        pivotsNow = pivots_c(amplitude, reversal_percent)

    pivotsNow.update(high, low)
    return pivotsNow