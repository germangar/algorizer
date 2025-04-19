#lightweight-charts simple test

import pandas as pd
import pandas_ta as pt
import math
from lightweight_charts import Chart
import asyncio
import ccxt.pro as ccxt
import time
from pprint import pprint

from window import window_c

import tasks

import tools
from tools import df_append
from fetcher import candles_c

import strategy


SHOW_VOLUME = False
verbose = False


def crossingUp( self, other ):
    if isinstance( self, generatedSeries_c ):
        return self.crossingUp( other )
    
    self_old = 0
    self_new = 0
    other_old = 0
    other_new = 0
    if isinstance( self, pd.Series ):
        if( len(self) < 2 ):
            return False
        self_old = self.iloc[-2]
        self_new = self.iloc[-1]
        if isinstance( other, pd.Series ):
            if( len(other) < 2 ):
                return False
            other_old = other.iloc[-2]
            self_new = other.iloc[-1]
        elif isinstance( other, generatedSeries_c ):
            if( other.timestamp == 0 or len(other.source) < 2 ):
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
    if isinstance( self, generatedSeries_c ):
        return self.crossingDown( other )
    
    self_old = 0
    self_new = 0
    other_old = 0
    other_new = 0
    if isinstance( self, pd.Series ):
        if( len(self) < 2 ):
            return False
        self_old = self.iloc[-2]
        self_new = self.iloc[-1]
        if isinstance( other, pd.Series ):
            if( len(other) < 2 ):
                return False
            other_old = other.iloc[-2]
            self_new = other.iloc[-1]
        elif isinstance( other, generatedSeries_c ):
            if( other.timestamp == 0 or len(other.source) < 2 ):
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
    return crossingUp( other, self ) or crossingDown( other, self )


class plot_c:
    def __init__( self, name:str, chart_name = None ):
        self.name = name
        self.chartName = chart_name
        self.source = None
        self.line = None
        self.initialized = False

    def update( self, source, timeframe ):

        if( timeframe.window is None ):
            # make a backup of the source series only on the last bar of the initialization
            # which will be used to jump-start the plots at opening the chart
            if( timeframe.shadowcopy ):
                if( not isinstance(source, pd.Series) ):
                    source = pd.Series([source] * len(timeframe.df), index=timeframe.df.index)
                if( len( timeframe.initdata ) <= len(timeframe.df) + 1 ): 
                    self.source = source
            return
        
        
        if( not isinstance(source, pd.Series) ):
            source = pd.Series([source] * len(timeframe.df), index=timeframe.df.index)
        
        if( not self.initialized ):
            if( len(source)<1 ):
                return
            chart = timeframe.window.bottomPanel if( self.chartName == 'panel' ) else timeframe.window.chart
            self.line = chart.create_line( self.name, price_line=False, price_label=False )
            self.line.set( pd.DataFrame({'time': pd.to_datetime( timeframe.df['timestamp'], unit='ms' ), self.name: source}) )
            self.initialized = True
            return
        
        source = source.dropna()
        if( len(source) < 1 ):
            return
        
        # it's initalized so only update the new line
        newval = source.iloc[-1]
        self.line.update( pd.Series( {'time': pd.to_datetime( timeframe.timestamp, unit='ms' ), 'value': newval } ) )


def plot( timeframe, name, source, chart_name = None ):
    timeframe.plot( name, source, chart_name )

'''
class markers_c:
    def __init__( self, text:str, timestamp:int, chart_name:str = None ):
        self.timestamp = timestamp
        self.text = text
        self.chartName = chart_name
        self.chart = None
        self.marker = None
    
    def __del__( self ):
        self.remove()

    def remove( self ):
        if( self.marker != None ):
            if( self.chart ):
                self.chart.remove_marker( self.marker )
                self.marker = None

    def refreshInChart( self ):
        if( self.chart ):
            self.remove()
            self.marker = self.chart.marker( time = pd.to_datetime( self.timestamp, unit='ms' ), text = self.text )
'''


def resample_ohlcv(df, target_timeframe):
    """
    Resample OHLCV dataframe to a higher timeframe.
    Accepts target_timeframe as number of minutes (e.g., 15, 60, 1440).
    Keeps timestamp in milliseconds, no datetime column is returned.
    """

    def map_minutes_to_pandas_freq(minutes: int) -> str:
        if minutes % 1440 == 0:
            return f"{minutes // 1440}D"
        elif minutes % 60 == 0:
            return f"{minutes // 60}H"
        else:
            return f"{minutes}T"

    # If target_timeframe is a string like '15', convert to int
    if isinstance(target_timeframe, str):
        target_timeframe = int(tools.timeframeInt(target_timeframe))

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


class timeframe_c:
    def __init__( self, stream, timeframeStr ):
        self.stream = stream
        self.timeframeStr = tools.timeframeString( timeframeStr )
        self.timeframe = tools.timeframeInt(self.timeframeStr) # in minutes
        self.timeframeMsec = tools.timeframeMsec(self.timeframeStr)
        self.callback = None
        self.barindex = -1
        self.timestamp = 0
        self.shadowcopy = False
        self.window = None

        self.df:pd.DataFrame = []
        self.initdata:pd.DataFrame = []
        self.generatedSeries: dict[str, generatedSeries_c] = {}
        self.registeredPlots: dict[str, plot_c] = {}

    def initDataframe( self, ohlcvDF ):
        print( "=================" )
        print( f"Creating dataframe {self.timeframeStr}" )

        # take out the last row to jumpstart the generatedSeries later
        self.df = ohlcvDF.iloc[:-1].copy()

        print( f"Calculating generated series {self.timeframeStr}" )

        # do the jump-starting with the last row of the dataframe
        start_time = time.time()
        self.parseCandleUpdate(ohlcvDF.iloc[[-1]])
        print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))

        print( f"Computing script logic {self.timeframeStr}" )
        # at this point we have the generatedSeries initialized for the whole dataframe
        # move the dataframe to use it as source for running the script logic.
        # Start with a new dataframe with only the first row copied from the precomputed dataframe.
        # The precomputed data will be (shadow)copied into the new dataframe as we progress
        # through the bars.
        ###############################################################################
        self.initdata = self.df
        self.df = pd.DataFrame( pd.DataFrame(self.initdata.iloc[0]).T, columns=self.initdata.columns )
        
        # run the script logic accross all the rows
        self.shadowcopy = True
        self.barindex = -1
        self.timestamp = int(self.df.iloc[0]['timestamp'])
        self.parseCandleUpdate(self.initdata)
        self.shadowcopy = False
        ###############################################################################

        print( len(self.df), "candles processed. Total time: {:.2f} seconds".format(time.time() - start_time))
        self.initdata = [] # free memory

    def parseCandleUpdate( self, rows ):

        for newrow in rows.itertuples(index=False):
            newTimestamp = int(newrow.timestamp)

            oldTimestamp = int(self.df.iloc[-1]['timestamp']) if len(self.df) > 1 else 0
            if oldTimestamp > newTimestamp:  # the server has sent a bunch of older candles
                continue

            if oldTimestamp + self.timeframeMsec > newTimestamp :
                # update the realtime candle
                if( self.timeframeStr == self.stream.timeframeFetch ):
                    self.df.loc[self.df.index[-1], 'open'] = newrow.open
                    self.df.loc[self.df.index[-1], 'high'] = newrow.high
                    self.df.loc[self.df.index[-1], 'low'] = newrow.low
                    self.df.loc[self.df.index[-1], 'close'] = newrow.close
                    self.df.loc[self.df.index[-1], 'volume'] = newrow.volume
                else:
                    self.df.loc[self.df.index[-1], 'high'] = max(newrow.high, self.df.iloc[-1]['high'])
                    self.df.loc[self.df.index[-1], 'low'] = min(newrow.low, self.df.iloc[-1]['low'])
                    self.df.loc[self.df.index[-1], 'close'] = newrow.close
                    # add the volume of the smallest candles
                    fecthTF = self.stream.timeframes[self.stream.timeframeFetch]
                    self.df.loc[self.df.index[-1], 'volume'] = fecthTF.df.loc[fecthTF.df['timestamp'] >= oldTimestamp, 'volume'].sum()

                # update the chart
                if self.window is not None:
                    self.window.updateChart(self)

            else:
                if not self.stream.initializing:
                    print( f'NEW {self.timeframeStr} CANDLE', newrow )

                # CLOSE REALTIME CANDLE 
                self.barindex = self.df.iloc[-1].name
                self.timestamp = self.df['timestamp'].iloc[-1]
                new_row_index = self.barindex + 1

                if( self.callback != None ):
                    self.callback( self, self.df['open'], self.df['high'], self.df['low'], self.df['close'] )

                # if( not self.shadowcopy ):
                #     self.updateAllGeneratedSeries() # update all calculated series regardless if they are called or not

                # OPEN A NEW CANDLE
                if self.shadowcopy:
                    # row_to_append = self.initdata.iloc[new_row_index].to_frame().T
                    # self.df = pd.concat( [self.df, row_to_append], ignore_index=True )
                    self.df = pd.concat( [self.df, self.initdata.iloc[new_row_index].to_frame().T], ignore_index=True )
                    
                else:
                    self.df.loc[new_row_index, 'timestamp'] = newTimestamp
                    self.df.loc[new_row_index, 'open'] = newrow.open
                    self.df.loc[new_row_index, 'high'] = newrow.high
                    self.df.loc[new_row_index, 'low'] = newrow.low
                    self.df.loc[new_row_index, 'close'] = newrow.close
                    self.df.loc[new_row_index, 'volume'] = newrow.volume

                # update the chart
                if( self.window != None ):
                    self.window.updateChart(self)

                if self.shadowcopy and new_row_index % 5000 == 0:
                    print( new_row_index, "candles processed." )

                if not self.shadowcopy and verbose:
                    print( self.df )


    def calcGeneratedSeries( self, type:str, source:pd.Series, period:int, func, always_reset:bool = False ):
        name = generatedSeriesNameFormat( type, source, period )

        gs = self.generatedSeries.get( name )
        if( gs == None ):
            gs = generatedSeries_c( type, source, period, func, always_reset, self )
            self.generatedSeries[name] = gs

        gs.update( source )
        return gs
    
    '''
    def createMarker( self, text:str = '', timestamp:int = -1, chart_name:str = None ):
        if timestamp == -1:
            timestamp = self.timestamp
        self.markers.append( markers_c( text, timestamp, chart_name ) )
    '''

    def plot( self, name, source, chart_name = None ):
        plot = self.registeredPlots.get( name )

        if( plot == None ):
            plot = plot_c( name, chart_name )
            self.registeredPlots[name] = plot
        
        plot.update( source, self )
        return plot
    
    def jumpstartPlots( self ):
        for plot in self.registeredPlots.values():
            if not plot.initialized:
                plot.update( plot.source, self )
    
def getTimeframeObject( name ):
    name = tools.timeframeString( name ) #it validates de name
    return stream.timeframes[name]

class stream_c:
    def __init__( self, symbol, exchangeID:str, timeframeList, max_amount = 5000 ):
        self.symbol = symbol # FIXME: add verification
        self.initializing = True
        self.timeframeFetch = None
        self.timeframes: dict[str, timeframe_c] = {}

        '''
        self.markers:markers_c = []
        '''

        #################################################
        # Validate de timeframes list and find 
        # the smallest for fetching the data
        #################################################
        if not isinstance(timeframeList, list) :
            timeframeList = [tools.timeframeString( timeframeList )]

        smallest = -1
        for t in timeframeList:
            # timeframeSec validates all the names. It will drop with a error if not valid.
            if tools.timeframeSec(t) < smallest or smallest < 0 :
                smallest = tools.timeframeSec(t)
                self.timeframeFetch = t

        if self.timeframeFetch == None :
            raise SystemError( f"stream_c->Init: timeframeList doesn't contain a valid timeframe name ({timeframeList})" )
        
        # the amount of candles to fetch are defined by the last timeframe on the list
        scale = int( tools.timeframeSec(timeframeList[-1]) / tools.timeframeSec(self.timeframeFetch) )
        
        
        #################################################
        # Fetch the candle history and update the cache
        #################################################

        fetcher = candles_c( exchangeID, self.symbol )
        ohlcvs = fetcher.loadCacheAndFetchUpdate( self.symbol, self.timeframeFetch, max_amount * scale )
        if( len(ohlcvs) == 0 ):
            raise SystemExit( f'No candles available in {exchangeID}. Aborting')
        ohlcvDF = pd.DataFrame( ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'] )
        ohlcvs = []

        #################################################
        # Create the timeframe sets with their dataframes
        #################################################

        for t in timeframeList:
            if t == self.timeframeFetch:
                candles = ohlcvDF
            else:
                candles = resample_ohlcv( ohlcvDF, t )

            timeframe = timeframe_c( self, t )

            func_name = f'runCloseCandle_{t}'
            timeframe.callback = globals().get(func_name)
            if timeframe.callback is None:
                print( f"*** WARNING: Timeframe {t} doesn't have a closeCandle function. Create a 'closeCandle_{t}( timeframe:timeframe_c, open:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series )' function in your script")

            timeframe.initDataframe( candles )
            self.timeframes[t] = timeframe
            candles = []

        ohlcvDF = []

        #################################################

        self.initializing = False

        #################################################

        # connect to ccxt.pro (FIXME? This is probably redundant with the fetcher)
        try:
            self.exchange = getattr(ccxt, exchangeID)({
                    "options": {'defaultType': 'swap', 'adjustForTimeDifference' : True},
                    "enableRateLimit": False
                    }) 
        except Exception as e:
            raise SystemExit( "Couldn't initialize exchange:", exchangeID )
        
        
        # We're done. Start fetching
        tasks.registerTask( fetchCandleUpdates( self ) )


    def parseCandleUpdateMulti( self, rows ):
        for timeframe in self.timeframes.values():
            timeframe.parseCandleUpdate(rows)

    def createWindow( self, timeframeStr ):
        # FIXME: Add proper checks
        timeframe = self.timeframes[tools.timeframeString( timeframeStr )]
        timeframe.window = window_c( timeframe )
        timeframe.jumpstartPlots()





def generatedseries_calculate_sma(series: pd.Series, period: int, df:pd.DataFrame) -> pd.Series:
    return pt.sma( series, period )

def generatedseries_calculate_ema(series: pd.Series, period: int, df:pd.DataFrame) -> pd.Series:
    return pt.ema( series, period )

def generatedseries_calculate_dema(series: pd.Series, period: int, df:pd.DataFrame) -> pd.Series:
    return pt.dema( series, period )

def generatedseries_calculate_linreg(series: pd.Series, period: int, df:pd.DataFrame) -> pd.Series:
    return pt.linreg( series, period )

def generatedseries_calculate_rma(series: pd.Series, length: int, df:pd.DataFrame) -> pd.Series:
    return series.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

def generatedseries_calculate_stdev(series: pd.Series, period: int, df:pd.DataFrame) -> pd.Series:
    return pt.stdev( series, period )

def generatedseries_calculate_bias(series: pd.Series, period: int, df:pd.DataFrame) -> pd.Series:
    return pt.bias( series, period )

def generatedseries_calculate_cfo(series: pd.Series, period: int, df:pd.DataFrame) -> pd.Series:
    return pt.cfo( series, period )

def generatedseries_calculate_fwma(series: pd.Series, period: int, df:pd.DataFrame) -> pd.Series:
    return pt.fwma( series, period )

def generatedseries_calculate_dev(series: pd.Series, period: int, df:pd.DataFrame) -> pd.Series:
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

def generatedseries_calculate_williams_r(series: pd.Series, period: int, df:pd.DataFrame) -> pd.Series:
    if 1:
        return pt.willr( df['high'], df['low'], df['close'], length=period )
    else:
        """
        Calculate Williams %R for a given series using OHLC data from df over a period.

        Args:
        - series: pd.Series, typically a placeholder, but required for compatibility with generatedSeries_c.
        - period: int, the period/window for the Williams %R calculation.

        Returns:
        - pd.Series, the calculated Williams %R values.
        """

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

def generatedseries_calculate_rsi(series, period, df:pd.DataFrame) -> pd.Series:
    deltas = series.diff()
    gain = deltas.where(deltas > 0, 0).rolling(window=period).mean()
    loss = -deltas.where(deltas < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi #.iloc[-1]  # Returning the last value of RSI


def generatedseries_calculate_tr(series: pd.Series, period: int, df:pd.DataFrame) -> pd.Series:
    if 1:
        if len(series) < period:
            return pd.Series( [pd.NA] * len(series), index=series.index )  # Not enough data to calculate the slope
        return pt.true_range( df['high'], df['low'], df['close'], length=period )
    else:
        """
        Calculate the True Range (TR) for a given series.

        Args:
        - series: pd.Series, the input series (only used to align with generatedSeries_c interface).
        - period: int, the period for the True Range calculation.

        Returns:
        - pd.Series, the calculated True Range series.
        """
        high = df['high']
        low = df['low']
        close = df['close']

        high_low = high - low
        high_close_prev = (high - close.shift()).abs()
        low_close_prev = (low - close.shift()).abs()

        tr = high_low.combine(high_close_prev, max).combine(low_close_prev, max)
        return tr

def generatedseries_calculate_atr(series, period, df:pd.DataFrame) -> pd.Series:
    if len(series) < period:
        return pd.Series( [pd.NA] * len(series), index=series.index )  # Not enough data to calculate the slope
    return pt.atr( df['high'], df['low'], df['close'], length=period )
    

def generatedseries_calculate_rising(series: pd.Series, length: int, df:pd.DataFrame) -> pd.Series:
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

def generatedseries_calculate_falling(series: pd.Series, length: int, df:pd.DataFrame) -> pd.Series:
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

def generatedseries_calculate_wma(series: pd.Series, period: int, df:pd.DataFrame) -> pd.Series:
    if 1:
        if len(series) < period:
            return pd.Series([pd.NA] * len(series), index=series.index)  # Not enough data to calculate the slope
        return pt.wma( series, period )
    else:
        """
        Calculate the Weighted Moving Average (WMA) for a given series and length.
        
        Args:
        - series: pd.Series, the input series.
        - length: int, the period/window for the WMA calculation.
        
        Returns:
        - pd.Series, the calculated WMA series.
        """
        weights = pd.Series(range(1, period + 1))
        wma = series.rolling(period).apply(lambda prices: (prices * weights).sum() / weights.sum(), raw=True)
        return wma

def generatedseries_calculate_hma(series: pd.Series, period: int, df:pd.DataFrame) -> pd.Series:
    if 1:
        if len(series) < period:
            return pd.Series([pd.NA] * len(series), index=series.index)  # Not enough data to calculate the slope
        return pt.hma( series, period )
    else:
        """
        Calculate the Hull Moving Average (HMA) for a given series and length.
        
        Args:
        - series: pd.Series, the input series.
        - length: int, the period/window for the HMA calculation.
        
        Returns:
        - pd.Series, the calculated HMA series.
        """
        if len(series) < period:
            return pd.Series([pd.NA] * len(series), index=series.index)  # Not enough data to calculate the HMA
        
        half_length = int(period / 2)
        sqrt_length = int(period ** 0.5)
        
        wma_half_length = pt.wma(series, half_length)
        wma_full_length = pt.wma(series, period)
        
        diff_wma = 2 * wma_half_length - wma_full_length
        
        hma = pt.wma(diff_wma, sqrt_length)
        
        return hma

def generatedseries_calculate_slope(series: pd.Series, period: int, df:pd.DataFrame) -> pd.Series:
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
    
def generatedseries_calculate_cci(series: pd.Series, period: int, df:pd.DataFrame ) -> pd.Series:
    return pt.cci( df['high'], df['low'], df['close'], period )


def generatedSeriesNameFormat( type, source:pd.Series, period:int ):
    if( source.name == None ):
        raise SystemError( f"Generated Series has no valid name [{type}{period} {source.name}]")
    return f'{type}{period} {source.name}'

class generatedSeries_c:
    def __init__( self, type:str, source:pd.Series, period:int, func = None, always_reset:bool = False, timeframe:timeframe_c = None ):
        self.name = generatedSeriesNameFormat( type, source, period )
        self.sourceName = source.name
        self.period = period
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

        if( self.period < 1 ):
            raise SystemError( f"Generated Series  with invalid period [{period}]")
        

    def initialize( self, source:pd.Series ):
        if( len(source) >= self.period and ( not self.name in self.timeframe.df.columns or self.alwaysReset ) ):
            if( self.timeframe.shadowcopy ):
                raise SystemError( f"[{self.name}] tried to initialize as shadowcopy" )
            start_time = time.time()
            self.timeframe.df[self.name] = self.func(source, self.period, self.timeframe.df).dropna()
            self.timestamp = self.timeframe.df['timestamp'].iloc[-1]
            if( self.timeframe.stream.initializing ):
                print( f"Initialized {self.name}." + " Elapsed time: {:.2f} seconds".format(time.time() - start_time))


    def update( self, source:pd.Series ):
        if( self.timeframe.shadowcopy ):
            return

        # has this row already been updated?
        if( self.timestamp >= self.timeframe.df['timestamp'].iloc[-1] ):
            return

        # if non existant try to create new. A few need to be made new every time
        if( self.timestamp == 0 or self.alwaysReset ):
            self.initialize( source )
            return
        
        # this happens when making the shadow copy
        # if( not pd.isna( self.stream.df[self.name].iloc[-1] ) ):
        #     return
        
        if( len(self.timeframe.df) < self.period ):
            return
        
        # realtime updates

        # slice the required block of candles to calculate the current value of the generated series
        newval = self.func(source[-self.period:], self.period, self.timeframe.df).iloc[-1]
        self.timeframe.df.loc[self.timeframe.df.index[-1], self.name] = newval
        self.timestamp = self.timeframe.timestamp
        
    def plot( self, chart = None ):
        if( self.timestamp > 0 ):
            self.timeframe.plot( self.name, self.series(), chart )
    
    def series( self ):
        return self.timeframe.df[self.name]
    
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
            if pd.isna(other.iloc[-2]) or pd.isna(other.iloc[-1]) :
                return False
            return ( self.value(1) <= other.iloc[-2] and self.value() >= other.iloc[-1] and self.value() != self.value(1) )
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
            if pd.isna(other.iloc[-2]) or pd.isna(other.iloc[-1]) :
                return False
            return ( self.value(1) >= other.iloc[-2] and self.value() <= other.iloc[-1] and self.value() != self.value(1) )
        try:
            float(other)
        except ValueError:
            return False
        else:
            return ( self.value(1) >= float(other) and self.value() <= float(other) and self.value() != self.value(1) )
    
    def crossing( self, other ):
        return self.crossingUp(other) or self.crossingDown(other)
    
    def value( self, backindex = 0 ):
        df = self.timeframe.df
        if( backindex < 0 or backindex + 1 > len(df) ):
            return None
            #raise KeyError( 'Invalid backindex. It must be 0 or more. Maybe you wanted to use iloc(index)')
        if( self.timestamp == 0 or pd.isna(df[self.name].iloc[-(backindex + 1)]) ):
            return None
        return df[self.name].iloc[-(backindex + 1)]
    
    def bool( self, backindex = 0 ):
        df = self.timeframe.df
        if( backindex < 0 or backindex + 1 > len(df) ):
            return None
            #raise KeyError( 'Invalid backindex. It must be 0 or more. Maybe you wanted to use iloc(index)')
        if( self.timestamp == 0 or pd.isna(df[self.name].iloc[-(backindex + 1)]) ):
            return None
        return df[self.name].iloc[-(backindex + 1)] != 0
        
    def loc( self, index = 0 ):
        df = self.timeframe.df
        if( self.timestamp == 0 ):
            print( f"Warning: {self.name} has not yet produced any value")
            return 0 # let's do this just to avoid crashes
        
        if( index < 0 or index > len(df) ):
            raise KeyError( 'Invalid index. It must be 0 or more')
        return df[self.name].loc[index]
    
    def iloc( self, index = -1 ):
        df = self.timeframe.df
        if( self.timestamp == 0 ):
            print( f"Warning: {self.name} has not yet produced any value")
            return 0 # let's do this just to avoid crashes
        
        return df[self.name].iloc[index]
    


def findIndexWhenTrue( source ):
    if( not isinstance(source, pd.Series ) ):
        if( isinstance( source, generatedSeries_c) ):
            source = source.series()
        else:
            raise ValueError( "calcIndexWhenTrue must be called with a series" )
    boolean_source = source.astype(bool) if source.dtype != bool else source
    return boolean_source[::-1].idxmax() if source.any() else None


def calcBarsSince( barindex, source ):
    index_when_true = findIndexWhenTrue( source )
    if( index_when_true == None ):
        return None
    return barindex - index_when_true
    # return activeStream.barindex - index_when_true


def findIndexWhenFalse( source ):
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


def calcBarsWhileTrue( barindex, source ):
    index_when_false = findIndexWhenFalse( source )
    if( index_when_false == None ):
        return None
    return barindex - index_when_false
    # return activeStream.barindex - index_when_false


# this can be done to any pandas_ta function that returns a series and takes as arguments a series and a period.
def falling( timeframe:timeframe_c, source:pd.Series, period:int ):
    return timeframe.calcGeneratedSeries( 'falling', source, period, generatedseries_calculate_falling )

def rising( timeframe:timeframe_c, source:pd.Series, period:int ):
    return timeframe.calcGeneratedSeries( 'rising', source, period, generatedseries_calculate_rising )

def calcSMA( timeframe:timeframe_c, source:pd.Series, period:int ):
    return timeframe.calcGeneratedSeries( 'sma', source, period, generatedseries_calculate_sma )

def calcEMA( timeframe:timeframe_c, source:pd.Series, period:int ):
    return timeframe.calcGeneratedSeries( "ema", source, period, generatedseries_calculate_ema )

def calcDEMA( timeframe:timeframe_c, source:pd.Series, period:int ):
    return timeframe.calcGeneratedSeries( "dema", source, period, generatedseries_calculate_dema, always_reset=True )

def calcWMA( timeframe:timeframe_c, source:pd.Series, period:int ):
    return timeframe.calcGeneratedSeries( "wma", source, period, generatedseries_calculate_wma )

def calcHMA( timeframe:timeframe_c, source:pd.Series, period:int ):
    return timeframe.calcGeneratedSeries( "hma", source, period, generatedseries_calculate_hma, always_reset=True )

# def calcJMA( timeframe:timeframe_c, source:pd.Series, period:int ):
#     return timeframe.calcGeneratedSeries( "jma", source, period, pt.jma )

# def calcKAMA( timeframe:timeframe_c, source:pd.Series, period:int ):
#     return timeframe.calcGeneratedSeries( "kama", source, period, pt.kama )

def calcLINREG( timeframe:timeframe_c, source:pd.Series, period:int ):
    return timeframe.calcGeneratedSeries( "linreg", source, period, generatedseries_calculate_linreg )

def calcRSI( timeframe:timeframe_c, source:pd.Series, period:int ):
    return timeframe.calcGeneratedSeries( 'rsi', source, period, generatedseries_calculate_rsi )

def calcDEV( timeframe:timeframe_c, source:pd.Series, period:int ):
    return timeframe.calcGeneratedSeries( 'dev', source, period, generatedseries_calculate_dev )

def calcSTDEV( timeframe:timeframe_c, source:pd.Series, period:int ):
    return timeframe.calcGeneratedSeries( 'stdev', source, period, generatedseries_calculate_stdev )

def calcRMA( timeframe:timeframe_c, source:pd.Series, period:int ):
    return timeframe.calcGeneratedSeries( 'rma', source, period, generatedseries_calculate_rma, always_reset=True )

def calcWPR( timeframe:timeframe_c, source:pd.Series, period:int ):
    return timeframe.calcGeneratedSeries( 'wpr', source, period, generatedseries_calculate_williams_r )

def calcATR2( timeframe:timeframe_c, period:int ): # The other one using pt is much faster
    tr = timeframe.calcGeneratedSeries( 'tr', pd.Series([pd.NA] * period, name = 'tr'), period, generatedseries_calculate_tr )
    return timeframe.calcGeneratedSeries( 'atr', tr.series(), period, generatedseries_calculate_rma )

def calcTR( timeframe:timeframe_c, period:int ):
    return timeframe.calcGeneratedSeries( 'tr', pd.Series([pd.NA] * period, name = 'tr'), period, generatedseries_calculate_tr )

def calcATR( timeframe:timeframe_c, period:int ):
    return timeframe.calcGeneratedSeries( 'atr', pd.Series([pd.NA] * period, name = 'atr'), period, generatedseries_calculate_atr )

def calcSLOPE( timeframe:timeframe_c, source:pd.Series, period:int ):
    return timeframe.calcGeneratedSeries( 'slope', source, period, generatedseries_calculate_slope, always_reset=True )

def calcBIAS( timeframe:timeframe_c, source:pd.Series, period:int ):
    return timeframe.calcGeneratedSeries( 'bias', source, period, generatedseries_calculate_bias )

def calcCCI( timeframe:timeframe_c, period:int ):
    return timeframe.calcGeneratedSeries( 'cci', pd.Series([pd.NA] * period, name = 'cci'), period, generatedseries_calculate_cci )

def calcCFO( timeframe:timeframe_c, source:pd.Series, period:int ):
    return timeframe.calcGeneratedSeries( 'cfo', source, period, generatedseries_calculate_cfo )

def calcFWMA( timeframe:timeframe_c, source:pd.Series, period:int ):
    return timeframe.calcGeneratedSeries( 'fwma', source, period, generatedseries_calculate_fwma )


def runCloseCandle_5m( timeframe:timeframe_c, open:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series ):
    sma = calcSMA( timeframe, close, 350 )
    sma.plot()
    rsi = calcRSI( timeframe, close, 14 )
    rsiplot = plot( timeframe, rsi.name, rsi.series(), 'panel' )
    return

def runCloseCandle_15m( timeframe:timeframe_c, open:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series ):
    sma = calcSMA( timeframe, close, 350 )
    sma.plot()
    # calcCCI( timeframe, 20 ).plot('panel')

    slope1000 = calcSMA( timeframe, calcSLOPE( timeframe, close, 200 ).series() * 500000, 14 )
    plot( timeframe, slope1000.name, slope1000.series(), 'panel' )
    return

def runCloseCandle_1m( timeframe:timeframe_c, open:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series ):

    ###########################
    # strategy code goes here #
    ###########################
    sma = calcSMA( timeframe, close, 350 )
    sma.plot()

    ema = calcEMA( timeframe, close, 4 )
    ema.plot()

    lr = calcLINREG( timeframe, close, 300 )
    lr.plot()

    rsi = calcRSI( timeframe, close, 14 )
    rsiplot = plot( timeframe, rsi.name, rsi.series(), 'panel' )
    
    # plot( "lazyline", 0, 'panel' )

    # FIXME: It crashes when calling to plot the same series
    # atr = calcATR( stream, 14 )
    # plot( stream, atr.name, atr.series(), 'panel' )

    # calcTR(14).plot('panel')

    

    # # sma_rising = rising( stream, sma.name, 10 )

    # cfo = calcCFO( stream, close, 20 )
    # cfo.plot('panel')

    # dev = calcDEV( stream, close, 30 )
    # # plot( stream, dev.name, dev.series(), 'panel' )

    # rma = calcRMA( stream, close, 90 )
    # rma.plot()

    stdev = calcSTDEV( timeframe, close, 350 )

    # willr = calcWPR( stream, close, 32 ).plot('panel')
    # calcBIAS( stream, close, 32 ).plot('panel')

    # hma = calcHMA( stream, close, 150 )
    # hma.plot()
    # r = rising( stream, hma.series(), 10 )
    # f = falling( stream, hma.series(), 10 )
    # if( not stream.initializing ):
    #     print( (hma.series() > 1.7) )


    # calcBarsSince( stream.barindex, r )
    # calcBarsWhileTrue( hma.series() > 1.7 )

    # calcCCI( stream, 20 )

    # slope1000 = calcSMA( stream, calcSLOPE( stream, close, 200 ).series() * 500000, 14 )
    # plot( stream, slope1000.name, slope1000.series(), 'panel' )

    '''
    if( sma.crossingUp(stream, close) ):
        stream.createMarker( '🔷' )

    if crossingDown( stream, sma, lr ):
        stream.createMarker( '🔺' )
    '''
    



    return


async def fetchCandleUpdates( stream:stream_c ):

    maxRows = 10
    while True:
        try:
            response = await stream.exchange.watch_ohlcv( stream.symbol, stream.timeframeFetch, limit = maxRows )
            #print(response)

        except Exception as e:
            print( 'Exception raised at fetchCandleupdates: Reconnecting', e, type(e) )
            await stream.exchange.close()
            await asyncio.sleep(1.0)
            continue
        
            
        # extract the data

        if( len(response) > maxRows ):
            response = response[len(response)-maxRows:]

        #pprint( response )
        if( len(response) ):
            stream.parseCandleUpdateMulti( pd.DataFrame( response, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'] ) )
        
        await asyncio.sleep(0.005)

    await exchange.close()

async def on_timeframe_selection(chart):
    print( f'Getting data with a {chart.topbar["my_switcher"].value} timeframe.' )

from datetime import datetime
async def update_clock(stream):
    #FIXME: Find the timeframe with a window
    # if( window == None ):
    #     return
    # while window.chart.is_alive:
    #     await asyncio.sleep(1-(datetime.now().microsecond/1_000_000))
    #     window.chart.legend( visible=True, ohlc=False, percent=False, font_size=18, text=stream.symbol + ' - ' + stream.timeframeStr + ' - ' + stream.exchange.id + ' - ' + f'candles:{len(stream.df)}' + ' - ' + datetime.now().strftime('%H:%M:%S') )
    return


import aioconsole
async def cli_task(stream):
    while True:
        command = await aioconsole.ainput()  # <-- Use aioconsole for non-blocking input

        if command.lower() == 'chart':
            print( 'opening chart' )
            window = stream.createWindow( stream.timeframeFetch )
        
        await asyncio.sleep(0.05)

if __name__ == '__main__':

    stream = stream_c( 'LDO/USDT:USDT', 'bitget', ['1m', '5m'], 1000 )

    # tasks.registerTask( update_clock(stream) )
    tasks.registerTask( cli_task(stream) )

    stream.createWindow( '5m' )

    asyncio.run( tasks.runTasks() )


    



