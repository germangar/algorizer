

def stringToValue( arg )->float:
    try:
        float(arg)
    except ValueError:
        value = None
    else:
        value = float(arg)
    return value

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

# def replaceValueByTimestamp( df, timestamp, key:str, value ):
#     if( key == 'open' or key == 'high' or key == 'low' or key == 'close' ):
#         df.loc[df['timestamp'] == timestamp, f'{key}'] = value

