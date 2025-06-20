

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
    return f'_{type}{period}{source.name}'

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


def resample_ohlcv(df, target_timeframe):
    """
    Resample OHLCV dataframe to a higher timeframe.
    Accepts target_timeframe as number of minutes (e.g., 15, 60, 1440).
    Keeps timestamp in milliseconds, no datetime column is returned.
    """

    import pandas as pd

    def map_minutes_to_pandas_freq(minutes: int) -> str:
        if minutes % 1440 == 0:
            return f"{minutes // 1440}D"
        elif minutes % 60 == 0:
            return f"{minutes // 60}H"
        else:
            return f"{minutes}min"

    # If target_timeframe is a string like '15', convert to int
    if isinstance(target_timeframe, str):
        target_timeframe = int(timeframeInt(target_timeframe))

    pandas_freq = map_minutes_to_pandas_freq(target_timeframe)

    df = df.copy()
    df.index = pd.to_datetime(df['timestamp'], unit='ms')

    resampled = df.resample(pandas_freq, label='left', closed='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    resampled['timestamp'] = (resampled.index.astype('int64') // 10**6)
    return resampled.reset_index(drop=True)[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

