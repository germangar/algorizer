
import pandas as pd

# concat a dictionary to a dataframe
# new_row = { 'timestamp': timestamp, self.name : newval }
# self.dataFrame = pd.concat( [self.dataFrame, pd.DataFrame(new_row, index=[0])], ignore_index=False )


# Extract the last 'num_rows' rows of the specified column into a new DataFrame
# sdf = df[self.source].tail(self.period).to_frame()


# append a row series/dictionary/dataframe to a dataframe
def df_append( df, row ):
    if isinstance( row, pd.Series ):
        newrow = row
    elif isinstance( row, dict ):
        newrow = pd.Series(row)
    elif isinstance( row, pd.DataFrame ):
        if len( row ) != 1:
            raise ValueError( "DataFrame for new row should have only one row" )
        newrow = row.iloc[0]
    else:
        raise ValueError( type(row), "Unsupported data type for new row" )
    
    if not all(col in df.columns for col in newrow.index):
        raise ValueError( "Column names in new row do not match DataFrame columns" )
    
    return pd.concat( [df, newrow.to_frame().T], ignore_index=True )



def stringToValue( arg )->float:
    try:
        float(arg)
    except ValueError:
        value = None
    else:
        value = float(arg)
    return value



timeframeNames = [ '1m', '5m', '15m', '30m', '45m', '1h', '2h', '3h', '4h', '1d', '1w' ]


# ccxt.bitget.parse_timeframe(timeframe) * 1000
            
def timeframeInt( timeframeName )->int:
    '''Returns timeframe as integer in minutes'''
    if( type(timeframeName) != str ):
        raise SystemError( "timeframeInt: Timeframe was not a string" )

    if( timeframeName not in timeframeNames ):
        raise SystemError( f"timeframeInt: {timeframeName} is not a valid timeframe name" )

    scale = timeframeName[-1:].lower()
    value = stringToValue( timeframeName[:-1] )
    if( value != None ):
        if( scale  == "m" ):
            return int( value )
        elif( scale  == "h" ):
            return int( value * 60 )
        elif( scale  == "d" ):
            return int( value * 60 * 24 )
        elif( scale  == "w" ):
            return int( value * 60 * 24 * 7 )
        
    # fall through if unsucessful
    SystemError( "timeframeNameToMinutes: Invalid timeframe string:", timeframeName )

def timeframeMsec( timeframeName )->int:
    return int( timeframeInt( timeframeName ) * 60 * 1000 )

def timeframeSec( timeframeName )->int:
    return int( timeframeInt( timeframeName ) * 60 )

def timeframeString( timeframe )->str:
    if( type(timeframe) != int ):
        if( timeframe in timeframeNames ):
            return timeframe
        SystemError( "timeframeNameToMinutes: Timeframe was not an integer" )

    name = 'invalid'
    
    if( timeframe < 60 and timeframe >= 1 ):
        name =  f'{timeframe}m'
    elif( timeframe < 1440 ):
        name =  f'{int(timeframe/60)}h'
    elif( timeframe < 10080 ):
        name =  f'{int(timeframe/1440)}d'
    elif( timeframe == 10080 ):
        name =  f'{int(timeframe/10080)}w'
    
    if( name not in timeframeNames ):
    # fall through if unsucessful
        raise SystemError( "timeframeString: Unsupported timeframe:", timeframe )
    
    return name

def replaceValueByTimestamp( df, timestamp, key:str, value ):
    if( key == 'open' or key == 'high' or key == 'low' or key == 'close' ):
        df.loc[df['timestamp'] == timestamp, f'{key}'] = value

