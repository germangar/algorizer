import platform

_talib_available = False
talib = None
if platform.python_implementation() == 'CPython':
    try:
        import talib
        _talib_available = True
    except ImportError:
        _talib_available = False
        print("Talib not available")
else:
    print("Talib not loaded (not running on CPython)")

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from typing import Union
from .series import generatedSeries_c, _prepare_param_for_op, _ensure_object_array, NumericScalar, OperandType
from .constants import constants as c
from . import active


################################ ANALYSIS TOOLS #####################################


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



# #
# # CALCULATION FUNCTIONS : calculate the series slice or full
# # 


# _highest250. Elapsed time: 0.00 seconds
def _generatedseries_calculate_highest(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    if _talib_available:
        return talib.MAX(source, period)
    source = np.asarray(source, dtype=np.float64)
    return _rolling_window_apply_optimized(source, period, lambda x: np.nanmax(x))

def _generatedseries_calculate_lowest(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    if _talib_available:
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


def _generatedseries_calculate_crossing_up(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    series1 = np.asarray(source, dtype=np.float64)
    
    # If the input array is too short for comparison with a previous bar, return NaNs.
    # We need at least two bars (current and previous) to detect a cross.
    if len(series1) < 2:
        return np.full_like(series1, np.nan)

    # Prepare the 'param' (second series) for vectorized operations.
    # This handles scalars, NumPy arrays, and other generatedSeries_c objects.
    series2 = _prepare_param_for_op(param, series1.shape[0], dataset)

    # Get the previous values for both series by rolling the array.
    # The first element becomes the last element, so we set it to NaN for correctness.
    series1_prev = np.roll(series1, 1)
    series1_prev[0] = np.nan
    
    series2_prev = np.roll(series2, 1)
    series2_prev[0] = np.nan

    # Condition for crossing up:
    # 1. The previous value of series1 was less than or equal to series2.
    # 2. The current value of series1 is strictly greater than series2.
    # We also ensure that neither of the involved values are NaN for a valid cross.
    crossed_up = (series1_prev <= series2_prev) & (series1 > series2) & \
                 (~np.isnan(series1_prev)) & (~np.isnan(series2_prev)) & \
                 (~np.isnan(series1)) & (~np.isnan(series2))

    # Convert the boolean result to float (1.0 for True, 0.0 for False)
    result = crossed_up.astype(np.float64)
    
    # The first element will always be NaN because there's no prior bar for comparison.
    # This is implicitly handled by np.roll and the NaN assignment, but explicit is clear. 
    result[0] = np.nan 
    
    return result


def _generatedseries_calculate_crossing_down(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    series1 = np.asarray(source, dtype=np.float64)
    
    # If the input array is too short for comparison with a previous bar, return NaNs.
    if len(series1) < 2:
        return np.full_like(series1, np.nan)

    # Prepare the 'param' (second series) for vectorized operations.
    series2 = _prepare_param_for_op(param, series1.shape[0], dataset)

    # Get the previous values for both series by rolling the array.
    series1_prev = np.roll(series1, 1)
    series1_prev[0] = np.nan
    
    series2_prev = np.roll(series2, 1)
    series2_prev[0] = np.nan

    # Condition for crossing down:
    # 1. The previous value of series1 was greater than or equal to series2.
    # 2. The current value of series1 is strictly less than series2.
    # We also ensure that neither of the involved values are NaN for a valid cross.
    crossed_down = (series1_prev >= series2_prev) & (series1 < series2) & \
                   (~np.isnan(series1_prev)) & (~np.isnan(series2_prev)) & \
                   (~np.isnan(series1)) & (~np.isnan(series2))

    # Convert the boolean result to float (1.0 for True, 0.0 for False)
    result = crossed_down.astype(np.float64)
    
    # The first element will always be NaN.
    result[0] = np.nan 
    
    return result


def _generatedseries_calculate_crossing(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    series1 = np.asarray(source, dtype=np.float64)
    
    # If the input array is too short for comparison with a previous bar, return NaNs.
    if len(series1) < 2:
        return np.full_like(series1, np.nan)

    # Prepare the 'param' (second series) for vectorized operations.
    series2 = _prepare_param_for_op(param, series1.shape[0], dataset)

    # Get the previous values for both series by rolling the array.
    series1_prev = np.roll(series1, 1)
    series1_prev[0] = np.nan
    
    series2_prev = np.roll(series2, 1)
    series2_prev[0] = np.nan

    # Condition for crossing up: (series1_prev <= series2_prev) & (series1 > series2)
    # Condition for crossing down: (series1_prev >= series2_prev) & (series1 < series2)
    # Ensure all involved values are not NaN for a valid cross.
    valid_mask = (~np.isnan(series1_prev)) & (~np.isnan(series2_prev)) & \
                 (~np.isnan(series1)) & (~np.isnan(series2))

    crossed_up = (series1_prev <= series2_prev) & (series1 > series2)
    crossed_down = (series1_prev >= series2_prev) & (series1 < series2)
    
    # Combine both conditions and apply the valid_mask
    crossed = (crossed_up | crossed_down) & valid_mask

    # Convert the boolean result to float (1.0 for True, 0.0 for False)
    result = crossed.astype(np.float64)
    
    # The first element will always be NaN.
    result[0] = np.nan 
    
    return result

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
    if _talib_available:
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
    if _talib_available:
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
    if _talib_available:
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

    if _talib_available:
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
    if _talib_available:
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

    if _talib_available:
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

    if _talib_available:
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
    
    if _talib_available:
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
    if _talib_available:
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
    
    if _talib_available:
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

    # These rolling sums will be calculated over the derived slices
    sum_num = _rolling_window_apply_optimized(br_num, period, np.nansum)
    sum_den = _rolling_window_apply_optimized(br_den, period, np.nansum)
    
    # Calculate BR.
    # Handle division by zero or NaN denominator
    br_calculated = np.where( (sum_den != 0) & (~np.isnan(sum_den)), (sum_num / sum_den) * 100, np.nan)

    return br_calculated

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

    sum_num = _rolling_window_apply_optimized(ar_num_slice, period, np.nansum)
    sum_den = _rolling_window_apply_optimized(ar_den_slice, period, np.nansum)
    
    # Calculate AR.
    # Handle division by zero or NaN denominator
    ar_calculated = np.where( (sum_den != 0) & (~np.isnan(sum_den)), (sum_num / sum_den) * 100, np.nan)

    return ar_calculated

#
def _generatedseries_calculate_cg(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
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
    if _talib_available:
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



# #
# # FACTORY FUNCTIONS: Set up the generated series to be calculated
# #


def highest(source: generatedSeries_c, period: int) -> generatedSeries_c:
    """
    Calculates the highest value of a source series over a given period.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The number of bars to look back.

    Returns:
        generatedSeries_c: A new series representing the highest value in the lookback period.
    """
    return source.timeframe.calcGeneratedSeries('highest', source, period, _generatedseries_calculate_highest)

def lowest(source: generatedSeries_c, period: int) -> generatedSeries_c:
    """
    Calculates the lowest value of a source series over a given period.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The number of bars to look back.

    Returns:
        generatedSeries_c: A new series representing the lowest value in the lookback period.
    """
    return source.timeframe.calcGeneratedSeries('lowest', _ensure_object_array(source), period, _generatedseries_calculate_lowest)

def highestbars(source: generatedSeries_c, period: int) -> generatedSeries_c:
    """
    Calculates the number of bars since the highest value in a given period.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The number of bars to look back.

    Returns:
        generatedSeries_c: A new series representing the number of bars since the highest value.
    """
    return source.timeframe.calcGeneratedSeries('highestbars', _ensure_object_array(source), period, _generatedseries_calculate_highestbars)

def lowestbars(source: generatedSeries_c, period: int) -> generatedSeries_c:
    """
    Calculates the number of bars since the lowest value in a given period.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The number of bars to look back.

    Returns:
        generatedSeries_c: A new series representing the number of bars since the lowest value.
    """
    return source.timeframe.calcGeneratedSeries('lowestbars', _ensure_object_array(source), period, _generatedseries_calculate_lowestbars)

def falling( source: generatedSeries_c, period:int )->generatedSeries_c:
    """
    Checks if a series has been falling for a given period.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The number of bars to check for a falling trend.

    Returns:
        generatedSeries_c: A boolean series (1.0 for True, 0.0 for False) indicating if the condition is met.
    """
    return source.timeframe.calcGeneratedSeries( 'falling', _ensure_object_array(source), period, _generatedseries_calculate_falling )

def rising( source: generatedSeries_c, period:int )->generatedSeries_c:
    """
    Checks if a series has been rising for a given period.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The number of bars to check for a rising trend.

    Returns:
        generatedSeries_c: A boolean series (1.0 for True, 0.0 for False) indicating if the condition is met.
    """
    return source.timeframe.calcGeneratedSeries( 'rising', _ensure_object_array(source), period, _generatedseries_calculate_rising )


def crossingUp(source1: generatedSeries_c, source2: Union[generatedSeries_c, NumericScalar]) -> generatedSeries_c:
    """
    Detects when one series crosses above another.

    This returns a boolean series (1.0 for True, 0.0 for False) that is True on the bar where
    source1's value becomes strictly greater than source2's value, after being less than or
    equal to it on the previous bar.

    Args:
        source1 (generatedSeries_c): The series that is crossing.
        source2 (Union[generatedSeries_c, NumericScalar]): The series or scalar being crossed.

    Returns:
        generatedSeries_c: A boolean series indicating the crossover event.
    """
    source1 = _ensure_object_array(source1) # Ensure source1 is a generatedSeries_c
    
    # Create a unique name for the generated series to prevent collisions
    if np.isscalar(source2):
        name = f"crossingUp_{source1.name}_{str(source2)}"
    else:
        source2 = _ensure_object_array(source2) # Ensure source2 is a generatedSeries_c
        name = f"crossingUp_{source1.name}_{source2.name}"

    # Period of 2 is needed to compare current and previous bars
    return source1.timeframe.calcGeneratedSeries( name, source1, 2, _generatedseries_calculate_crossing_up, param= source2 )

def crossingDown(source1: generatedSeries_c, source2: Union[generatedSeries_c, NumericScalar]) -> generatedSeries_c:
    """
    Detects when one series crosses below another.

    This returns a boolean series (1.0 for True, 0.0 for False) that is True on the bar where
    source1's value becomes strictly less than source2's value, after being greater than or
    equal to it on the previous bar.

    Args:
        source1 (generatedSeries_c): The series that is crossing.
        source2 (Union[generatedSeries_c, NumericScalar]): The series or scalar being crossed.

    Returns:
        generatedSeries_c: A boolean series indicating the cross-under event.
    """
    source1 = _ensure_object_array(source1) 
    
    # Create a unique name for the generated series to prevent collisions
    if np.isscalar(source2):
        name = f"crossingDown_{source1.name}_{str(source2)}"
    else:
        source2 = _ensure_object_array(source2) # Ensure source2 is a generatedSeries_c
        name = f"crossingDown_{source1.name}_{source2.name}"

    # Period of 2 is needed to compare current and previous bars
    return source1.timeframe.calcGeneratedSeries( name, source1, 2, _generatedseries_calculate_crossing_down, param=source2 )

def crossing(source1: generatedSeries_c, source2: Union[generatedSeries_c, NumericScalar]) -> generatedSeries_c:
    """
    Detects when one series crosses another in either direction.

    This is true when `crossingUp` or `crossingDown` would be true.

    Args:
        source1 (generatedSeries_c): The first series.
        source2 (Union[generatedSeries_c, NumericScalar]): The second series or scalar.

    Returns:
        generatedSeries_c: A boolean series indicating a cross in either direction.
    """
    source1 = _ensure_object_array(source1)
    
    # Create a unique name for the generated series to prevent collisions
    if np.isscalar(source2):
        name = f"crossing_{source1.name}_{str(source2)}"
    else:
        source2 = _ensure_object_array(source2) # Ensure source2 is a generatedSeries_c
        name = f"crossing_{source1.name}_{source2.name}"

    # Period of 2 is needed to compare current and previous bars
    return source1.timeframe.calcGeneratedSeries( name, source1, 2, _generatedseries_calculate_crossing, param=source2 )


def barsSinceSeries(source: generatedSeries_c, period: int) -> generatedSeries_c:
    """
    Counts the number of bars since a condition was last true.

    The condition is considered true for any non-zero, non-NaN value in the source series.
    The count resets to 0 on the bar where the condition is true.

    Args:
        source (generatedSeries_c): A boolean-like series where non-zero values are considered True.
        period (int): The lookback period. If the event has not occurred within this period, the result is NaN.

    Returns:
        generatedSeries_c: A series with the count of bars since the last true event.
    """
    import inspect
    # Get caller info by going up 2 levels in the stack
    caller_frame = inspect.currentframe().f_back
    frame_info = inspect.getframeinfo(caller_frame)
    caller_id = f"{frame_info.function[:5]}{frame_info.lineno}"
    return source.timeframe.calcGeneratedSeries('barsSince'+caller_id, _ensure_object_array(source), period, _generatedseries_calculate_barssince)

def indexWhenTrueSeries(source: generatedSeries_c, period: int = None) -> generatedSeries_c:
    """
    Returns the bar index of the last time a condition was true.

    For each bar, this series holds the absolute bar index of the most recent prior bar
    where the source series was true (non-zero, non-NaN).

    Args:
        source (generatedSeries_c): The boolean-like source series.
        period (int, optional): Not used in the calculation logic. Defaults to None.

    Returns:
        generatedSeries_c: A series containing the bar index of the last true occurrence.
    """
    return source.timeframe.calcGeneratedSeries('indexwhentrue_series', _ensure_object_array(source), period, _generatedseries_calculate_indexwhentrue)

def indexWhenFalseSeries(source: generatedSeries_c, period: int) -> generatedSeries_c:
    """
    Returns the bar index of the last time a condition was false.

    For each bar, this series holds the absolute bar index of the most recent prior bar
    where the source series was false (equal to 0). NaN values are not considered false.

    Args:
        source (generatedSeries_c): The boolean-like source series.
        period (int): Not used in the calculation logic.

    Returns:
        generatedSeries_c: A series containing the bar index of the last false occurrence.
    """
    return source.timeframe.calcGeneratedSeries('indexwhenfalse_series', _ensure_object_array(source), period, _generatedseries_calculate_indexwhenfalse)

def barsWhileTrueSeries(source: generatedSeries_c, period: int = None) -> generatedSeries_c:
    """
    Counts the number of consecutive bars a condition has been true.

    The count increments for each consecutive bar where the source is true (non-zero, non-NaN)
    and resets to zero when it becomes false.

    Args:
        source (generatedSeries_c): The boolean-like source series.
        period (int, optional): The maximum value the count can reach. Defaults to None (no limit).

    Returns:
        generatedSeries_c: A series with the count of consecutive true bars.
    """
    return source.timeframe.calcGeneratedSeries('barsWhileTrue', _ensure_object_array(source), period, _generatedseries_calculate_barswhiletrue)

def barsWhileFalseSeries(source: generatedSeries_c, period: int = None) -> generatedSeries_c:
    """
    Counts the number of consecutive bars a condition has been false.

    The count increments for each consecutive bar where the source is false (zero)
    and resets to zero when it becomes true.

    Args:
        source (generatedSeries_c): The boolean-like source series.
        period (int, optional): The maximum value the count can reach. Defaults to None (no limit).

    Returns:
        generatedSeries_c: A series with the count of consecutive false bars.
    """
    return source.timeframe.calcGeneratedSeries('barsWhileFalse', _ensure_object_array(source), period, _generatedseries_calculate_barswhilefalse)




########################## INDICATORS #################################

def MIN( colA: generatedSeries_c | NumericScalar, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    """
    Calculates the element-wise minimum of two series or a series and a scalar.

    Args:
        colA (Union[generatedSeries_c, NumericScalar]): The first series or scalar.
        colB (Union[generatedSeries_c, NumericScalar]): The second series or scalar.

    Returns:
        generatedSeries_c: A new series containing the element-wise minimum.
    """
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
    """
    Calculates the element-wise maximum of two series or a series and a scalar.

    Args:
        colA (Union[generatedSeries_c, NumericScalar]): The first series or scalar.
        colB (Union[generatedSeries_c, NumericScalar]): The second series or scalar.

    Returns:
        generatedSeries_c: A new series containing the element-wise maximum.
    """
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
    Shifts a series by a given offset, providing access to past or future values.

    A positive 'offset' (e.g., 1) lags the series, so the value at the current bar `i`
    is taken from the previous bar `i-1`. This is equivalent to `source[1]` in Pine Script.
    A negative 'offset' (e.g., -1) leads the series, so the value at `i` is from `i+1`.

    Args:
        source (generatedSeries_c): The input series to shift.
        offset (int): The number of bars to shift. Positive for lag, negative for lead.

    Returns:
        generatedSeries_c: A new series representing the shifted values.
    """
    if offset == 0:
        return source
    return source.timeframe.calcGeneratedSeries( 'shift', _ensure_object_array(source), offset, _generatedseries_calculate_shift )

def SUM( source:generatedSeries_c, period:int )->generatedSeries_c:
    """
    Calculates the rolling sum of a series over a specified period.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The number of bars to include in the sum.

    Returns:
        generatedSeries_c: A new series representing the rolling sum.
    """
    return source.timeframe.calcGeneratedSeries( 'sum', _ensure_object_array(source), period, _generatedseries_calculate_sum )

def SMA( source: generatedSeries_c, period: int )->generatedSeries_c:
    """
    Simple Moving Average (SMA).

    Calculates the arithmetic mean of a series over a specified period.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The moving average period.

    Returns:
        generatedSeries_c: A new series representing the SMA.
    """
    return source.timeframe.calcGeneratedSeries('sma', _ensure_object_array(source), period, _generatedseries_calculate_sma )

def EMA( source: generatedSeries_c, period:int )->generatedSeries_c:
    """
    Exponential Moving Average (EMA).

    Calculates a weighted moving average where more recent values are given more weight.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The moving average period.

    Returns:
        generatedSeries_c: A new series representing the EMA.
    """
    return source.timeframe.calcGeneratedSeries( "ema", _ensure_object_array(source), period, _generatedseries_calculate_ema, always_reset=True )

def DEMA( source: generatedSeries_c, period:int )->generatedSeries_c:
    """
    Double Exponential Moving Average (DEMA).

    A more responsive moving average that reduces lag by combining a single and double EMA.
    Formula: DEMA = 2 * EMA(source, period) - EMA(EMA(source, period), period)

    Args:
        source (generatedSeries_c): The input series.
        period (int): The moving average period for both EMAs.

    Returns:
        generatedSeries_c: A new series representing the DEMA.
    """
    return source.timeframe.calcGeneratedSeries( "dema", _ensure_object_array(source), period, _generatedseries_calculate_dema, always_reset=True )

def RMA( source:generatedSeries_c, period:int )->generatedSeries_c:
    """
    Relative Moving Average (RMA) or Running Moving Average.

    A type of moving average that is smoothed differently from a simple or exponential one.
    It's also known as Wilder's Smoothing Average.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The smoothing period.

    Returns:
        generatedSeries_c: A new series representing the RMA.
    """
    return source.timeframe.calcGeneratedSeries( 'rma', _ensure_object_array(source), period, _generatedseries_calculate_rma, always_reset=True )

def WMA( source:generatedSeries_c, period:int )->generatedSeries_c:
    """
    Weighted Moving Average (WMA).

    Calculates a moving average where earlier values in the period have linearly decreasing weights.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The moving average period.

    Returns:
        generatedSeries_c: A new series representing the WMA.
    """
    return source.timeframe.calcGeneratedSeries( "wma", _ensure_object_array(source), period, _generatedseries_calculate_wma )

def HMA( source:generatedSeries_c, period:int )->generatedSeries_c:
    """
    Hull Moving Average (HMA).

    A fast and smooth moving average that minimizes lag while maintaining curve smoothness.
    Formula: HMA = WMA(2 * WMA(source, period/2) - WMA(source, period), sqrt(period))

    Args:
        source (generatedSeries_c): The input series.
        period (int): The HMA period.

    Returns:
        generatedSeries_c: A new series representing the HMA.
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
    """
    Standard Deviation.

    Calculates the rolling standard deviation of a series over a given period.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The calculation period.
        scalar (float, optional): A multiplier for the final result. Defaults to 1.0.

    Returns:
        generatedSeries_c: A new series representing the rolling standard deviation.
    """
    return source.timeframe.calcGeneratedSeries( 'stdev', _ensure_object_array(source), period, _generatedseries_calculate_stdev, param = scalar, always_reset= _talib_available )

def DEV( source:generatedSeries_c, period:int )->generatedSeries_c:
    """
    Mean Absolute Deviation (DEV).

    Calculates the average of the absolute deviations from the mean over a given period.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The calculation period.

    Returns:
        generatedSeries_c: A new series representing the mean absolute deviation.
    """
    return source.timeframe.calcGeneratedSeries( 'dev', _ensure_object_array(source), period, _generatedseries_calculate_dev )

def WILLR( period:int )->generatedSeries_c:
    """
    Williams %R.

    A momentum indicator that measures overbought and oversold levels. It uses the 'close', 'high',
    and 'low' series from the active timeframe.

    Args:
        period (int): The lookback period.

    Returns:
        generatedSeries_c: A new series representing the Williams %R values.
    """
    timeframe = active.timeframe
    source = timeframe.generatedSeries['close']
    return timeframe.calcGeneratedSeries( 'wpr', source, period, _generatedseries_calculate_williams_r )

def TR( period:int, high:generatedSeries_c= None, low:generatedSeries_c= None )->generatedSeries_c:
    """
    True Range (TR).

    Calculates the true range, which is the greatest of the following:
    - Current high minus the current low
    - Absolute value of the current high minus the previous close
    - Absolute value of the current low minus the previous close

    Args:
        period (int): The period (used for consistency, but TR is calculated from bar to bar).
        high (generatedSeries_c, optional): The high series. Defaults to active timeframe's high.
        low (generatedSeries_c, optional): The low series. Defaults to active timeframe's low.

    Returns:
        generatedSeries_c: A new series representing the True Range.
    """
    timeframe = active.timeframe
    source = timeframe.generatedSeries['close']
    if high is None: high = timeframe.generatedSeries["high"]
    if low is None: low = timeframe.generatedSeries["low"]
    name = 'tr'
    if high.column_index != c.DF_HIGH or low.column_index != c.DF_LOW:
        name += f"{high.column_index}{low.column_index}"
    return timeframe.calcGeneratedSeries( name, source, period, _generatedseries_calculate_tr, param= (high, low) )

def ATR( period:int, high:generatedSeries_c= None, low:generatedSeries_c= None )->generatedSeries_c:
    """
    Average True Range (ATR).

    A volatility indicator that shows how much an asset moves on average. It is an RMA of the True Range (TR).

    Args:
        period (int): The smoothing period for the RMA.
        high (generatedSeries_c, optional): The high series. Defaults to active timeframe's high.
        low (generatedSeries_c, optional): The low series. Defaults to active timeframe's low.

    Returns:
        generatedSeries_c: A new series representing the Average True Range.
    """
    timeframe = active.timeframe
    source = timeframe.generatedSeries['close']
    if high is None: high = timeframe.generatedSeries["high"]
    if low is None: low = timeframe.generatedSeries["low"]
    name = 'atr'
    if high.column_index != c.DF_HIGH or low.column_index != c.DF_LOW:
        name += f"{high.column_index}{low.column_index}"
    return timeframe.calcGeneratedSeries( name, source, period, _generatedseries_calculate_atr, param= (high, low), always_reset= True )  # rma requires always_reset, so atr also must

def SLOPE( source:generatedSeries_c, period:int )->generatedSeries_c:
    """
    Linear Regression Slope.

    Calculates the slope of a linear regression line over a specified period.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The period for the linear regression calculation.

    Returns:
        generatedSeries_c: A new series representing the slope.
    """
    return source.timeframe.calcGeneratedSeries( 'slope', _ensure_object_array(source), period, _generatedseries_calculate_slope )

def VHMA(source: generatedSeries_c, period: int) -> generatedSeries_c:
    """
    Variable-length Hull Moving Average (VHMA).

    An adaptive version of the Hull Moving Average where the smoothing period is
    dynamically adjusted based on market volatility.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The base period for the calculation.

    Returns:
        generatedSeries_c: A new series representing the VHMA.
    """
    return source.timeframe.calcGeneratedSeries('vhma', _ensure_object_array(source), period, _generatedseries_calculate_vhma, always_reset= True)

def BIAS( source:generatedSeries_c, period:int )->generatedSeries_c:
    """
    Bias (BIAS).

    Measures the percentage deviation of a series from its Simple Moving Average.
    Formula: ((source - SMA(source, period)) / SMA(source, period)) * 100

    Args:
        source (generatedSeries_c): The input series.
        period (int): The period for the SMA calculation.

    Returns:
        generatedSeries_c: A new series representing the BIAS oscillator.
    """
    return source.timeframe.calcGeneratedSeries( 'bias', _ensure_object_array(source), period, _generatedseries_calculate_bias )

def LINREG( source:generatedSeries_c, period:int )->generatedSeries_c:
    """
    Linear Regression Curve (LINREG).

    Calculates the ending value of a linear regression line over a specified period.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The period for the linear regression calculation.

    Returns:
        generatedSeries_c: A new series representing the linear regression value.
    """
    return source.timeframe.calcGeneratedSeries( "linreg", _ensure_object_array(source), period, _generatedseries_calculate_linreg )

def CCI(period: int = 20) -> generatedSeries_c:
    """
    Commodity Channel Index (CCI).

    A momentum-based oscillator used to help determine when an investment vehicle is
    reaching a condition of being overbought or oversold. It uses the 'close', 'high',
    and 'low' series from the active timeframe.

    Args:
        period (int, optional): The lookback period. Defaults to 20.

    Returns:
        generatedSeries_c: A new series representing the CCI.
    """
    if not isinstance(period, int ):
        raise ValueError( "CCI requires only a period argument" )
    timeframe = active.timeframe
    source = timeframe.generatedSeries['close']
    return timeframe.calcGeneratedSeries('cci', _ensure_object_array(source), period, _generatedseries_calculate_cci, always_reset= True)

def CFO( source:generatedSeries_c, period:int )->generatedSeries_c:
    """
    Chande Forecast Oscillator (CFO).

    Measures the percentage difference between the closing price and its n-period
    linear regression forecast.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The period for the linear regression forecast.

    Returns:
        generatedSeries_c: A new series representing the CFO.
    """
    return source.timeframe.calcGeneratedSeries( 'cfo', _ensure_object_array(source), period, _generatedseries_calculate_cfo )

def CMO(source: generatedSeries_c, period: int = 9) -> generatedSeries_c:
    """
    Chande Momentum Oscillator (CMO).

    A momentum oscillator that measures the strength of a trend. It is calculated by
    taking the sum of all positive momentum over a period and subtracting the sum of
    all negative momentum, then dividing by the sum of all momentum.

    Args:
        source (generatedSeries_c): The input series.
        period (int, optional): The lookback period. Defaults to 9.

    Returns:
        generatedSeries_c: A new series representing the CMO, oscillating between -100 and +100.
    """
    return source.timeframe.calcGeneratedSeries('cmo', _ensure_object_array(source), period, _generatedseries_calculate_cmo)

def FWMA( source:generatedSeries_c, period:int )->generatedSeries_c:
    """
    Fibonacci Weighted Moving Average (FWMA).

    A moving average where the weights are based on the Fibonacci sequence.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The moving average period.

    Returns:
        generatedSeries_c: A new series representing the FWMA.
    """
    return source.timeframe.calcGeneratedSeries( 'fwma', _ensure_object_array(source), period, _generatedseries_calculate_fwma )

def RSI( source:generatedSeries_c, period:int )->generatedSeries_c:
    """
    Relative Strength Index (RSI).

    A momentum oscillator that measures the speed and change of price movements.
    RSI oscillates between zero and 100.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The lookback period.

    Returns:
        generatedSeries_c: A new series representing the RSI.
    """
    return source.timeframe.calcGeneratedSeries( 'rsi', _ensure_object_array(source), period, _generatedseries_calculate_rsi, always_reset=True )

def IFTrsi( source:generatedSeries_c, period:int )->generatedSeries_c:
    """
    Inverse Fisher Transform on RSI (IFTrsi).

    Applies an Inverse Fisher Transform to the RSI values to help identify
    overbought and oversold conditions with sharper, more defined signals.

    Args:
        source (generatedSeries_c): The input series for the RSI calculation.
        period (int): The period for the underlying RSI calculation.

    Returns:
        generatedSeries_c: A new series representing the IFTrsi.
    """
    timeframe = _ensure_object_array(source).timeframe
    rsi = timeframe.calcGeneratedSeries( 'rsi', source, period, _generatedseries_calculate_rsi, always_reset=True )
    return timeframe.calcGeneratedSeries( 'iftrsi', rsi, period, _generatedseries_calculate_inverse_fisher_rsi )

def Fisher( period:int, signal:float=None )->tuple[generatedSeries_c, generatedSeries_c]:
    """
    Fisher Transform.

    An indicator that converts prices into a Gaussian normal distribution, helping to
    identify trend reversals with sharper signals. It uses 'high' and 'low' from the active timeframe.

    Args:
        period (int): The lookback period for the transform.
        signal (float, optional): The period for the signal line (an EMA of the Fisher line).
                                  Defaults to 9 if not provided.

    Returns:
        tuple[generatedSeries_c, generatedSeries_c]: A tuple containing the Fisher Transform line
                                                     and its signal line.
    """
    timeframe = active.timeframe
    source = timeframe.generatedSeries['close']
    fish = timeframe.calcGeneratedSeries( 'fisher', source, period, _generatedseries_calculate_fisher )
    sig = timeframe.calcGeneratedSeries( 'fishersig', source, period, _generatedseries_calculate_fisher_signal, signal )
    return fish, sig

def AO( fast: int = 5, slow: int = 34 ) -> generatedSeries_c:
    """
    Awesome Oscillator (AO).

    A momentum indicator that calculates the difference between a 34-period and 5-period
    Simple Moving Average of the median price (High+Low)/2.

    Args:
        fast (int, optional): The period for the fast SMA. Defaults to 5.
        slow (int, optional): The period for the slow SMA. Defaults to 34.

    Returns:
        generatedSeries_c: A new series representing the Awesome Oscillator.
    """
    timeframe = active.timeframe
    param = (fast, slow)
    source = timeframe.generatedSeries['close']
    return timeframe.calcGeneratedSeries('ao', source, max(fast,slow), _generatedseries_calculate_ao, param)

def BR( period:int )->generatedSeries_c:
    """
    Willingness to Buy Ratio (BR).

    A market sentiment indicator that measures the strength of buying pressure versus selling pressure.

    Args:
        period (int): The lookback period.

    Returns:
        generatedSeries_c: A new series representing the BR.
    """
    timeframe = active.timeframe
    source = timeframe.generatedSeries['close']
    return timeframe.calcGeneratedSeries( 'br', source, period, _generatedseries_calculate_br )

def AR( period:int )->generatedSeries_c:
    """
    Momentum Ratio (AR).

    A market sentiment indicator that measures the strength of the current trend by comparing
    the difference between the high and open to the difference between the open and low.

    Args:
        period (int): The lookback period.

    Returns:
        generatedSeries_c: A new series representing the AR.
    """
    timeframe = active.timeframe
    source = timeframe.generatedSeries['close']
    return timeframe.calcGeneratedSeries( 'ar', source, period, _generatedseries_calculate_ar )

def BRAR( period:int )->tuple[generatedSeries_c, generatedSeries_c]:
    """
    Calculates both the Willingness to Buy Ratio (BR) and Momentum Ratio (AR).

    Args:
        period (int): The lookback period for both indicators.

    Returns:
        tuple[generatedSeries_c, generatedSeries_c]: A tuple containing the BR and AR series.
    """
    br = BR(period)
    ar = AR(period)
    return br, ar

def CG( source:generatedSeries_c, period:int )->generatedSeries_c:
    """
    Center of Gravity (CG) Oscillator.

    An oscillator that helps identify turning points with minimal lag.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The lookback period.

    Returns:
        generatedSeries_c: A new series representing the Center of Gravity oscillator.
    """
    return source.timeframe.calcGeneratedSeries( 'cg', _ensure_object_array(source), period, _generatedseries_calculate_cg )

def STOCHk( source: generatedSeries_c, period:int )-> tuple[generatedSeries_c, generatedSeries_c]:
    """
    Stochastic Oscillator %K line.

    This is the main line for the Stochastic Oscillator. It is often smoothed to create the %D line.

    Args:
        source (generatedSeries_c): The input series (typically 'close').
        period (int): The lookback period for the stochastic calculation.

    Returns:
        generatedSeries_c: A new series representing the %K line.
    """
    return source.timeframe.calcGeneratedSeries( "stochk", _ensure_object_array(source), period, _generatedseries_calculate_stoch_k )

def OBV( timeframe=None ) -> generatedSeries_c:
    """
    On-Balance Volume (OBV).

    A momentum indicator that uses volume flow to predict changes in stock price.
    It uses the 'close' and 'volume' series from the specified timeframe.

    Args:
        timeframe (Timeframe, optional): The timeframe context. Defaults to the active timeframe.

    Returns:
        generatedSeries_c: A new series representing the OBV.
    """
    # period=2 because obv reads the previous value of close. It can not be anything else.
    source = timeframe.generatedSeries['close']
    return source.timeframe.calcGeneratedSeries( 'obv', source, 2, _generatedseries_calculate_obv )

def LAGUERRE(source: Union[str, generatedSeries_c], gamma: float = 0.7)->generatedSeries_c:
    """
    Laguerre Oscillator.

    A responsive oscillator that uses a Laguerre filter to reduce lag and provide clear signals.

    Args:
        source (Union[str, generatedSeries_c]): The input series.
        gamma (float, optional): The smoothing factor for the Laguerre filter. Defaults to 0.7.

    Returns:
        generatedSeries_c: A new series representing the Laguerre Oscillator (values 0 to 1).
    """
    return source.timeframe.calcGeneratedSeries( 'lagerre', _ensure_object_array(source), 1, _generatedseries_calculate_laguerre, gamma, always_reset=True )


# # #
# # # COMPOSITE OF SEVERAL SERIES
# # #

def Stochastic(source: generatedSeries_c, k_period: int = 14, d_period: int = 3)-> tuple[generatedSeries_c, generatedSeries_c]:
    """
    Calculates the Stochastic Oscillator (%K and %D lines).

    Args:
        source (generatedSeries_c): The input price series (typically 'close').
        k_period (int): The period for the %K calculation (e.g., 14).
        d_period (int): The period for the %D calculation (SMA of %K, e.g., 3).

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
    Calculates Bollinger Bands (basis, upper, lower).

    Bollinger Bands consist of a middle band being an N-period simple moving average (SMA),
    an upper band at K standard deviations above the middle band, and a lower band at K
    standard deviations below the middle band.

    Args:
        source (generatedSeries_c): The input series.
        period (int): The period for the SMA and standard deviation calculation.
        mult (float, optional): The number of standard deviations. Defaults to 2.0.

    Returns:
        Tuple[generatedSeries_c, generatedSeries_c, generatedSeries_c]: The basis (SMA), upper band, and lower band.
    """
    BBbasis = SMA(source, period)
    stdev = STDEV(source, period, mult)
    BBupper = BBbasis + stdev
    BBlower = BBbasis - stdev
    return BBbasis, BBupper, BBlower


def MACD( source:generatedSeries_c, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[generatedSeries_c, generatedSeries_c, generatedSeries_c]:
    """
    Moving Average Convergence Divergence (MACD).

    A trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.

    Args:
        source (generatedSeries_c): The input price series (e.g. close).
        fast (int, optional): The period for the fast EMA. Defaults to 12.
        slow (int, optional): The period for the slow EMA. Defaults to 26.
        signal (int, optional): The period for the signal line EMA. Defaults to 9.

    Returns:
        Tuple[generatedSeries_c, generatedSeries_c, generatedSeries_c]: A tuple containing the MACD line, Signal line, and Histogram.
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


# # #
# # # OTHER TA STUFF NOT GENERATED SERIES BUT SINGLE SCALAR RETURNS
# # #


def highestBarSingle(source:generatedSeries_c, lookback_period:int = None, since:int= None)-> Union[int, None]:
    """
    Finds the absolute bar index of the highest value within a specified lookback window.

    Args:
        source (generatedSeries_c): The input series to search.
        lookback_period (int, optional): The number of bars to look back from the 'since' bar.
                                         If None, searches from the beginning of the series. Defaults to None.
        since (int, optional): The ending bar index (inclusive) for the lookback.
                               If None, uses the current active bar index. Defaults to None.

    Returns:
        Union[int, None]: The absolute bar index of the highest value, or None if no valid
                          (non-NaN) highest value is found within the window.
    """
    if since is None:
        since = active.barindex
    
    if lookback_period is None:
        start_index = 0
    else:
        start_index = max(0, since - lookback_period + 1)
    end_index = since + 1
    
    if start_index >= end_index:
        return None

    # Slice the underlying numpy array
    series_slice_values = source.series()[start_index:end_index]
    
    if len(series_slice_values) == 0:
        return None

    try:
        # nanargmax returns the index within the slice where the max value is.
        relative_index = np.nanargmax(series_slice_values)
        # The absolute index in the full series is start_index + relative_index.
        absolute_index = start_index + relative_index
        return absolute_index
    except ValueError: # This happens if the slice contains only NaNs
        return None

def highestSingle(source:generatedSeries_c, lookback_period:int, since:int= None)-> Union[float, None]:
    """
    Finds the highest value within a specified lookback window.

    Args:
        source (generatedSeries_c): The input series to search.
        lookback_period (int): The number of bars to look back from the 'since' bar.
        since (int, optional): The ending bar index (inclusive) for the lookback.
                               If None, uses the current active bar index. Defaults to None.

    Returns:
        Union[float, None]: The highest value found within the window, or None if no valid
                            (non-NaN) highest value is found.
    """
    index = highestBarSingle(source, lookback_period, since)
    if index is None:
        return None
    return source[index]

def lowestBarSingle(source:generatedSeries_c, lookback_period:int= None, since:int= None)-> Union[int, None]:
    """
    Finds the absolute bar index of the lowest value within a specified lookback window.

    Args:
        source (generatedSeries_c): The input series to search.
        lookback_period (int, optional): The number of bars to look back from the 'since' bar.
                                         If None, searches from the beginning of the series. Defaults to None.
        since (int, optional): The ending bar index (inclusive) for the lookback.
                               If None, uses the current active bar index. Defaults to None.

    Returns:
        Union[int, None]: The absolute bar index of the lowest value, or None if no valid
                          (non-NaN) lowest value is found within the window.
    """
    if since is None:
        since = active.barindex
    if lookback_period is None:
        start_index = 0
    else:
        start_index = max(0, since - lookback_period + 1)
    end_index = since + 1

    if start_index >= end_index:
        return None

    series_slice_values = source.series()[start_index:end_index]
    
    if len(series_slice_values) == 0:
        return None

    try:
        relative_index = np.nanargmin(series_slice_values)
        absolute_index = start_index + relative_index
        return absolute_index
    except ValueError: # All-NaN slice
        return None

def lowestSingle(source:generatedSeries_c, lookback_period:int, since:int= None)-> Union[float, None]:
    """
    Finds the lowest value within a specified lookback window.

    Args:
        source (generatedSeries_c): The input series to search.
        lookback_period (int): The number of bars to look back from the 'since' bar.
        since (int, optional): The ending bar index (inclusive) for the lookback.
                               If None, uses the current active bar index. Defaults to None.

    Returns:
        Union[float, None]: The lowest value found within the window, or None if no valid
                            (non-NaN) lowest value is found.
    """
    index = lowestBarSingle(source, lookback_period, since)
    if index is None:
        return None
    return source[index]

def indexWhenTrueSingle(source: generatedSeries_c, lookback_period: int = None, since: int = None)-> Union[int, None]:
    """
    Finds the absolute bar index of the most recent occurrence where the source series was "True"
    within a specified lookback period.

    A value is considered "True" if it is non-zero and not NaN.

    Args:
        source (generatedSeries_c): The boolean-like input series.
        lookback_period (int, optional): The number of bars to look back from the 'since' bar.
                                         If None, searches from the beginning of the series. Defaults to None.
        since (int, optional): The ending bar index (inclusive) for the search.
                               If None, uses the current active bar index. Defaults to None.

    Returns:
        Union[int, None]: The absolute bar index of the last "True" occurrence, or None if no
                          "True" value is found within the specified window.
    """
    source = _ensure_object_array(source)
    full_source_array = source.series()
    
    # Determine the end of the search window (exclusive for slicing)
    end_index = (since if since is not None else active.barindex) + 1

    # Determine the start of the search window
    if lookback_period is not None and lookback_period > 0:
        start_index = max(0, end_index - lookback_period)
    else:
        start_index = 0
        
    # Slice the array to the search window
    source_array_slice = full_source_array[start_index:end_index]

    # Create a boolean mask for "True" values
    true_mask = (source_array_slice != 0) & (~np.isnan(source_array_slice))
    
    # Find all true indices *relative to the slice*
    relative_true_indices = np.where(true_mask)[0]

    if relative_true_indices.size > 0:
        # Get the last relative index and convert it to an absolute index
        last_relative_index = relative_true_indices[-1]
        absolute_index = start_index + last_relative_index
        return absolute_index
            
    return None # No True value found


def indexWhenFalseSingle(source: generatedSeries_c, lookback_period: int = None, since: int = None)-> Union[int, None]:
    """
    Finds the absolute bar index of the most recent occurrence where the source series was "False"
    within a specified lookback period.

    A value is considered "False" if it is exactly 0. NaN values are explicitly not considered "False".

    Args:
        source (generatedSeries_c): The boolean-like input series.
        lookback_period (int, optional): The number of bars to look back from the 'since' bar.
                                         If None, searches from the beginning of the series. Defaults to None.
        since (int, optional): The ending bar index (inclusive) for the search.
                               If None, uses the current active bar index. Defaults to None.

    Returns:
        Union[int, None]: The absolute bar index of the last "False" occurrence, or None if no
                          "False" value is found within the specified window.
    """
    source_series = _ensure_object_array(source)
    full_source_array = source_series.series()

    # Determine the end of the search window (exclusive for slicing)
    end_index = (since if since is not None else active.barindex) + 1

    # Determine the start of the search window
    if lookback_period is not None and lookback_period > 0:
        start_index = max(0, end_index - lookback_period)
    else:
        start_index = 0
        
    # Slice the array to the search window
    source_array_slice = full_source_array[start_index:end_index]

    # A "False" value is 0. `series == 0` correctly handles NaN.
    relative_false_indices = np.where(source_array_slice == 0)[0]

    if relative_false_indices.size > 0:
        # Get the last relative index and convert it to an absolute index
        last_relative_index = relative_false_indices[-1]
        absolute_index = start_index + last_relative_index
        return absolute_index
            
    return None # No False value found
    

def barsSinceSingle( source:generatedSeries_c, lookback_period:int = None )-> Union[int, None]:
    """
    Calculates the number of bars that have passed since the last "True" condition
    occurred within a given lookback period, up to the current bar.

    A value is considered "True" if it is non-zero and not NaN.

    Args:
        source (generatedSeries_c): The boolean-like input series.
        lookback_period (int, optional): The maximum number of bars to look back from the current bar
                                         to find the last "True" condition. If None, searches the
                                         entire history. Defaults to None.

    Returns:
        Union[int, None]: The number of bars elapsed since the last "True" condition,
                          or None if no "True" condition is found within the lookback period.
    """
    # Find the last true index, but only look back 'lookback_period' bars.
    index_when_true = indexWhenTrueSingle( source, lookback_period=lookback_period )

    if index_when_true is None:
        return None # No True condition was found within the lookback period.

    return active.barindex - index_when_true


def barsWhileTrueSingle( source:generatedSeries_c, lookback_period:int = None )-> Union[int, None]:
    """
    Calculates the number of consecutive bars (including the current one)
    for which the source condition has been "True", optionally capped by `lookback_period`.

    A value is considered "True" if it is non-zero and not NaN.

    Args:
        source (generatedSeries_c): The boolean-like input series.
        lookback_period (int, optional): The maximum number of consecutive "True" bars to count.
                                         If the actual streak is longer, the function returns this cap.
                                         If None, there is no limit. Defaults to None.

    Returns:
        Union[int, None]: The count of consecutive "True" bars ending at the current position,
                          or None if the series is empty or the condition is never met.
    """
    current_bar_index = active.barindex

    # Find the last False, searching the whole history up to the current bar.
    index_of_last_false = indexWhenFalseSingle( source, lookback_period=None, since=current_bar_index )

    count = 0
    if index_of_last_false is None:
        # No False found, so the streak is the entire history up to this point.
        count = current_bar_index + 1
    else:
        # Streak is the distance from the last False.
        count = current_bar_index - index_of_last_false

    # Apply the lookback_period as a cap on the result.
    if lookback_period is not None and count > lookback_period:
        return lookback_period

    return count

def crossingUpSingle( series1:generatedSeries_c|float, series2:generatedSeries_c|float )-> bool:
    """
    Determines if `series1` crosses up over `series2` between the previous and current bar.

    This function checks if, on the previous bar, `series1` was less than or equal to `series2`,
    and on the current bar, `series1` is strictly greater than `series2`.

    Args:
        series1 (Union[generatedSeries_c, float]): The first series or a scalar value.
        series2 (Union[generatedSeries_c, float]): The second series or a scalar value to compare against.

    Returns:
        bool: True if a crossing up occurred, False otherwise.
    """
    if isinstance( series1, generatedSeries_c ):
        return series1.crossingUp(series2)
    
    # Original logic for pd.Series and float/int (unchanged, but might need similar iloc(-1) and iloc(-2) adaptations for clarity if it's not already doing that)
    if isinstance( series1, int ):
        series1 = float(series1)

    if isinstance( series2, int ):
        series2 = float(series2)

    if( isinstance(series1, float) and isinstance(series2, float) ):
        print( "* WARNING: crossinUp: Two static values can never cross" )
        return False
    
    series1_old = 0
    series1_new = 0
    series2_old = 0
    series2_new = 0
    if isinstance( series1, np.ndarray ):
        if( len(series1) < 2 or active.barindex < 1 ):
            return False
        series1_old = series1[active.barindex-1]
        series1_new = series1[active.barindex]
        if isinstance( series2, np.ndarray ):
            if( len(series2) < 2 ):
                return False
            series2_old = series2[active.barindex-1]
            series2_new = series2[active.barindex]
        elif isinstance( series2, generatedSeries_c ):
            # Directly use series2.iloc(-1) and series2.iloc(-2)
            if np.isnan(series2.lastUpdatedTimestamp) or len(series2) < 2 or active.barindex < 1 :
                return False
            series2_old = series2.iloc(-2)
            series2_new = series2.iloc(-1) 
        else: # assuming float or int
            try:
                float_series2 = float(series2)
            except ValueError:
                return False
            else:
                series2_old = float(series2)
                series2_new = float(series2)
    else:
        try:
            float(series1)
        except ValueError:
            print( "crossinUp: Unsupported type", type(series1) )
            return False
        else:
            return crossingDownSingle( series2, series1 )

    return ( series1_old <= series2_old and series1_new >= series2_new and not (series1_old == series2_old and series1_new == series2_new) )


def crossingDownSingle( series1:generatedSeries_c|float, series2:generatedSeries_c|float )-> bool:
    """
    Determines if `series1` crosses down below `series2` between the previous and current bar.

    This function checks if, on the previous bar, `series1` was greater than or equal to `series2`,
    and on the current bar, `series1` is strictly less than `series2`.

    Args:
        series1 (Union[generatedSeries_c, float]): The first series or a scalar value.
        series2 (Union[generatedSeries_c, float]): The second series or a scalar value to compare against.

    Returns:
        bool: True if a crossing down occurred, False otherwise.
    """
    if isinstance( series1, generatedSeries_c ):
        return series1.crossingDown(series2)
    
    # Original logic for pd.Series and float/int (unchanged, but might need similar iloc(-1) and iloc(-2) adaptations for clarity if it's not already doing that)
    if isinstance( series1, int ):
        series1 = float(series1)

    if isinstance( series2, int ):
        series2 = float(series2)

    if( isinstance(series1, float) and isinstance(series2, float) ):
        print( "* WARNING: crossinDown: Two static values can never cross" )
        return False
    
    series1_old = 0
    series1_new = 0
    series2_old = 0
    series2_new = 0
    if isinstance( series1, np.ndarray ):
        if( len(series1) < 2 or active.barindex < 1 ):
            return False
        series1_old = series1[active.barindex-1]
        series1_new = series1[active.barindex]
        if isinstance( series2, np.ndarray ):
            if( len(series2) < 2 ):
                return False
            series2_old = series2[active.barindex-1]
            series2_new = series2[active.barindex]
        elif isinstance( series2, generatedSeries_c ):
            if np.isnan(series2.lastUpdatedTimestamp) or len(series2) < 2 or active.barindex < 1 :
                return False
            series2_old = series2.iloc(-2)
            series2_new = series2.iloc(-1) 
        else:
            try:
                float(series2)
            except ValueError:
                return False
            else:
                series2_old = float(series2)
                series2_new = float(series2)
    else:
        try:
            float(series1)
        except ValueError:
            print( "crossinDown: Unsupported type", type(series1) )
            return False
        else:
            return crossingUpSingle( series2, series1 )

    return ( series1_old >= series2_old and series1_new <= series2_new and not (series1_old == series2_old and series1_new == series2_new) )

def crossingSingle( series1:generatedSeries_c|float, series2:generatedSeries_c|float )-> bool:
    """
    Determines if `series1` crosses `series2` (either up or down) between the previous and current bar.

    This function returns True if either `crossingUpSingle(series1, series2)` or
    `crossingDownSingle(series1, series2)` is True.

    Args:
        series1 (Union[generatedSeries_c, float]): The first series or a scalar value.
        series2 (Union[generatedSeries_c, float]): The second series or a scalar value to compare against.

    Returns:
        bool: True if a crossing (up or down) occurred, False otherwise.
    """
    return crossingUpSingle( series1, series2 ) or crossingDownSingle( series1, series2 )



##
## Pivots is here so it shares the 'ta' name. Doesn't really belong
##

from .pivots import pivots_c, pivot_c
pivotsNow:pivots_c = None
def pivots( high:generatedSeries_c, low:generatedSeries_c, amplitude: float = 1.0, reversal_percent: float = 32.0 )->pivots_c:
    """
    Calculates and manages pivot points (e.g., swing highs and lows).

    This function initializes a `pivots_c` object if not already present and updates it
    with the latest high and low series data to identify pivot points.

    Args:
        high (generatedSeries_c): The high price series.
        low (generatedSeries_c): The low price series.
        amplitude (float, optional): The minimum amplitude (in price units) required for a swing.
                                     Defaults to 1.0.
        reversal_percent (float, optional): The percentage of price movement required for a reversal
                                            to confirm a pivot. Defaults to 32.0.

    Returns:
        pivots_c: An instance of the `pivots_c` class, updated with the latest pivot data.
    """
    global pivotsNow
    if pivotsNow == None:
        pivotsNow = pivots_c(amplitude, reversal_percent)

    pivotsNow.update(high, low)
    return pivotsNow


# Dynamically set __all__ to include all functions defined in this module
# that do not start with an underscore. 
# (I keep trying to clean up the private stuff from showing up in autocompletion
# but I don't seem to be getting it to work)
import types
__all__ = [
    name for name, val in globals().items()
    if isinstance(val, types.FunctionType) and val.__module__ == __name__ and not name.startswith('_')
]