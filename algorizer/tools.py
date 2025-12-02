import numpy as np
from .series import generatedSeries_c

def get_column_index_from_array( dataset:np.ndarray, candidate_col:np.ndarray ):
    # Try to find the index of the candidate_col in the dataset columns
    for idx in range(dataset.shape[1]):
        if np.shares_memory(dataset[:, idx], candidate_col) or np.all(dataset[:, idx] == candidate_col):
            return idx
    return None

def stringToValue( arg )->float:
    try:
        float(arg)
    except ValueError:
        value = None
    else:
        value = float(arg)
    return value


# used to standarize the name given to a generated series (calcseries.py)
# I probably should defined type pd.series for 'source' but I don't feel like importing pandas here
def generatedSeriesNameFormat( type, source, period:int ):
    if isinstance(source, generatedSeries_c):
        return f'_{type}_{period}{source.name}'
    return f'_{type}_{period}'

def hx2rgba(hex_color):
    """Converts a hex color code (with or without alpha) to an RGBA string for CSS."""
    hex_color = hex_color.lstrip('#')
    hex_length = len(hex_color)
    if hex_length not in (6, 8):
        return None  # Invalid hex code length

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    if hex_length == 6:
        a = 1.0
    else:
        a = round(int(hex_color[6:8], 16) / 255, 3)

    return f'rgba({r}, {g}, {b}, {a})'
    #return (r, g, b, a)


def emptyFunction(func):
    return func.__code__.co_consts == (None,)


''' # CCXT timeframe conventions
def parse_timeframe(timeframe):
        amount = int(timeframe[0:-1])
        unit = timeframe[-1]
        if 'y' == unit:
            scale = 60 * 60 * 24 * 365
        elif 'M' == unit:
            scale = 60 * 60 * 24 * 30
        elif 'w' == unit:
            scale = 60 * 60 * 24 * 7
        elif 'd' == unit:
            scale = 60 * 60 * 24
        elif 'h' == unit:
            scale = 60 * 60
        elif 'm' == unit:
            scale = 60
        elif 's' == unit:
            scale = 1
        else:
            raise NotSupported('timeframe unit {} is not supported'.format(unit))
        return amount * scale
'''


timeframeSufffixes = [ 'm', 'h', 'd', 'w', 'M', 'y' ]

def validateTimeframeName( timeframeName ):
        if not isinstance( timeframeName, str ):
            print( "validateTimeframeName: Timeframe was not a string" )
            return False

        amount = stringToValue( timeframeName[:-1] )
        if( amount == None ):
            print( f"validateTimeframeName: Timeframe string didn't produce a value '{timeframeName}'" )
            return False

        unit = timeframeName[-1]

        if unit not in timeframeSufffixes:
            print( f"validateTimeframeName: Unknown timeframe suffix '{timeframeName}'. Valid suffixes:" )
            print( timeframeSufffixes )
            return False
        
        return True


# ccxt.bitget.parse_timeframe(timeframe) * 1000
            
def timeframeInt( timeframeName )->int:
    '''Returns timeframe as integer in minutes'''
    if( not validateTimeframeName(timeframeName) ):
        raise SystemError( f"timeframeInt: {timeframeName} is not a valid timeframe name" )
    
    amount = int(timeframeName[0:-1])
    unit = timeframeName[-1]
    if 'y' == unit:
        scale = 60 * 24 * 365
    elif 'M' == unit:
        scale = 60 * 24 * 30
    elif 'w' == unit:
        scale = 60 * 24 * 7
    elif 'd' == unit:
        scale = 60 * 24
    elif 'h' == unit:
        scale = 60
    elif 'm' == unit:
        scale = 1

    return int( amount * scale )

def timeframeMsec( timeframeName )->int:
    return int( timeframeInt( timeframeName ) * 60 * 1000 )

def timeframeSec( timeframeName )->int:
    return int( timeframeInt( timeframeName ) * 60 )

def timeframeString( timeframe )->str:
    if( type(timeframe) != int ):
        if( validateTimeframeName(timeframe) ):
            return timeframe
        SystemError( f"timeframeNameToMinutes: Timeframe was not an integer nor a valid format: {timeframe}" )

    name = 'invalid'
    
    if( timeframe < 60 and timeframe >= 1 ):
        name =  f'{timeframe}m'
    elif( timeframe < 1440 ):
        name =  f'{int(timeframe/60)}h'
    elif( timeframe < 10080 ):
        name =  f'{int(timeframe/1440)}d'
    elif( timeframe < 604800 ):
        name =  f'{int(timeframe/10080)}w'
    elif( timeframe < 2592000 ):
        name =  f'{int(timeframe/604800)}M'
    
    return name


def resample_ohlcv_np(dataset: np.ndarray, target_timeframe) -> np.ndarray:
    """
    Resample OHLCV dataset to a higher timeframe using numpy.

    Args:
        dataset (np.ndarray): 2D array with columns [timestamp, open, high, low, close, volume].
        target_timeframe: String (e.g., '15m', '1h') or int (minutes, e.g., 15, 60).

    Returns:
        np.ndarray: Resampled OHLCV array with same column structure.
    """
    if dataset.shape[0] == 0 or dataset.shape[1] < 6:
        return np.empty((0, 6), dtype=np.float64)

    # Convert target_timeframe to milliseconds
    if isinstance(target_timeframe, str):
        target_timeframe_ms = timeframeMsec(target_timeframe)
    else:
        target_timeframe_ms = target_timeframe * 60 * 1000

    # Extract timestamps
    timestamps = dataset[:, 0]

    # Compute bucket indices (floor division by timeframe)
    bucket_indices = (timestamps // target_timeframe_ms).astype(np.int64)

    # Find unique buckets and their counts
    unique_buckets, bucket_counts = np.unique(bucket_indices, return_counts=True)
    n_buckets = len(unique_buckets)

    # Initialize output array
    resampled = np.empty((n_buckets, 6), dtype=np.float64)

    # Process each bucket
    for i, bucket in enumerate(unique_buckets):
        # Get rows in this bucket
        mask = bucket_indices == bucket
        bucket_data = dataset[mask]

        # Set timestamp (left edge of bucket)
        resampled[i, 0] = bucket * target_timeframe_ms

        # Aggregations
        bucket_open = bucket_data[:, 1]  # open
        bucket_high = bucket_data[:, 2]  # high
        bucket_low = bucket_data[:, 3]   # low
        bucket_close = bucket_data[:, 4] # close
        bucket_volume = bucket_data[:, 5] # volume

        # open: first non-NaN
        valid_open = bucket_open[~np.isnan(bucket_open)]
        resampled[i, 1] = valid_open[0] if len(valid_open) > 0 else np.nan

        # high: max non-NaN
        resampled[i, 2] = np.nanmax(bucket_high) if np.any(~np.isnan(bucket_high)) else np.nan

        # low: min non-NaN
        resampled[i, 3] = np.nanmin(bucket_low) if np.any(~np.isnan(bucket_low)) else np.nan

        # close: last non-NaN
        valid_close = bucket_close[~np.isnan(bucket_close)]
        resampled[i, 4] = valid_close[-1] if len(valid_close) > 0 else np.nan

        # volume: sum non-NaN
        resampled[i, 5] = np.nansum(bucket_volume) if np.any(~np.isnan(bucket_volume)) else np.nan

    # Filter out rows with any NaN values
    valid_rows = ~np.any(np.isnan(resampled), axis=1)
    resampled = resampled[valid_rows]

    return resampled

