




def stringToValue( arg )->float:
    try:
        float(arg)
    except ValueError:
        value = None
    else:
        value = float(arg)
    return value
            
def timeframeInt( timeframeName )->int:
    if( type(timeframeName) != str ):
        SystemError( "timeframeNameToMinutes: Timeframe was not a string" )

    if( timeframeName[-1:].lower()  == "m" ):
        if( stringToValue(timeframeName[:-1]) != None ):
            return int( stringToValue( timeframeName[:-1] ) )
    elif( timeframeName[-1:].lower()  == "h" ):
        value = stringToValue( timeframeName[:-1] )
        if( value != None ):
            return int( value * 60 )
    elif( timeframeName[-1:].lower()  == "d" ):
        value = stringToValue( timeframeName[:-1] )
        if( value != None ):
            return int( value * 60 * 24 )
    elif( timeframeName[-1:].lower()  == "w" ):
        value = stringToValue( timeframeName[:-1] )
        if( value != None ):
            return int( value * 60 * 24 * 7 )
        
    # fall through if unsucessful
    SystemError( "timeframeNameToMinutes: Invalid timeframe string:", timeframeName )

def timeframeString( timeframe )->str:
    if( type(timeframe) != int ):
        SystemError( "timeframeNameToMinutes: Timeframe was not an integer" )
    
    if( timeframe < 60 and timeframe >= 1 ):
        return f'{timeframe}m'
    elif( timeframe < 1440 ):
        return f'{int(timeframe/60)}h'
    elif( timeframe < 10080 ):
        return f'{int(timeframe/1440)}d'
    elif( timeframe == 10080 ):
        return f'{int(timeframe/10080)}w'
    
    # fall through if unsucessful
    SystemError( "timeframeNameToMinutes: Unsupported timeframe:", timeframe )