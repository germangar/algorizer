
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

def replaceValueByTimestamp( df, timestamp, key:str, value ):
    if( key == 'open' or key == 'high' or key == 'low' or key == 'close' ):
        df.loc[df['timestamp'] == timestamp, f'{key}'] = value

'''
class sma_c:
    def __init__( self, source:str, period ):
        self.period = period
        self.source = source
        self.name = f'sma {source} {period}'
        self.initialized = False

        if( not self.source in df.columns ):
            raise SystemError( f"SMA with unknown source [{source}]")

        if( self.period < 1 ):
            raise SystemError( f"SMA with invalid period [{period}]")

    def update( self ):

        #if non existant try to create new
        if( not self.initialized ):
            if( len(df) >= self.period and not self.name in df.columns ):
                df[self.name] = df[self.source].rolling(window=self.period).mean()
                self.initialized = True
            return self.initialized
        
        # check if this row has already been updated
        if( not pd.isna(df[self.name].iloc[-1]) ):
            return True
        
        # isolate only the required block of candles to calculate the current value of the SMA
        # Extract the last 'num_rows' rows of the specified column into a new DataFrame
        sdf = df[self.source].tail(self.period).to_frame(name=self.source)
        if( len(sdf) < self.period ):
            return False 
        
        newval = sdf[self.source].rolling(window=self.period).mean().dropna().iloc[-1]
        df.loc[df.index[-1], self.name] = newval # the new row is already created

        return True
    
    def plotData( self ):
        return pd.DataFrame({'timestamp': df['timestamp'], self.name: df[self.name]}).dropna()

registeredSMAs = []

def calcSMA( source:str, period ):
    name = f'sma {source} {period}'
    sma = None
    # find if there's a SMA already created for this series
    for thisSMA in registeredSMAs:
        if thisSMA.name == name:
            sma = thisSMA
            #print( 'found', name )
            break
    if sma == None:
        sma = sma_c( source, period )
        registeredSMAs.append(sma)

    sma.update()
    return sma
'''

