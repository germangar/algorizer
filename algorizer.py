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
from window import createWindow

import tasks

import tools
from tools import df_append
from fetcher import candles_c

import strategy


window = None
SHOW_VOLUME = False
verbose = True


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

    def update( self, source, stream, window = None ):

        if( window == None ): 
            # make a backup of the source series only on the last bar of the initialization
            # which will be used to jump-start the plots at opening the chart
            if( stream.shadowcopy ):
                if( not isinstance(source, pd.Series) ):
                    source = pd.Series([source] * len(stream.df), index=stream.df.index)
                if( len( stream.initdata ) <= len(stream.df) + 1 ): 
                    self.source = source
            return
        
        if( not isinstance(source, pd.Series) ):
            source = pd.Series([source] * len(stream.df), index=stream.df.index)
        
        if( not self.initialized ):
            if( len(source)<1 ):
                return
            chart = window.bottomPanel if( self.chartName == 'panel' ) else window.chart
            self.line = chart.create_line( self.name, price_line=False, price_label=False )
            self.line.set( pd.DataFrame({'time': pd.to_datetime( stream.df['timestamp'], unit='ms' ), self.name: source}) )
            self.initialized = True
            return
        
        source = source.dropna()
        if( len(source) < 1 ):
            return
        
        # it's initalized so only update the new line
        newval = source.iloc[-1]
        self.line.update( pd.Series( {'time': pd.to_datetime( stream.timestamp, unit='ms' ), 'value': newval } ) )


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


class stream_c:
    def __init__( self, symbol, exchangeID:str, timeframe, max_amount = 5000 ):
        self.symbol = symbol # FIXME: add verification
        self.market = None
        self.timeframe = timeframe if( type(timeframe) == int ) else tools.timeframeInt(timeframe)
        self.timeframeStr = tools.timeframeString( self.timeframe )
        self.barindex = -1
        self.timestamp = 0
        self.initializing = True
        self.shadowcopy = False

        self.generatedSeries:generatedSeries_c = []
        self.markers:markers_c = []
        self.registeredPlots:plot_c = []

        self.df:pd.DataFrame = []
        self.initdata:pd.DataFrame = []

        try:
            self.exchange = getattr(ccxt, exchangeID)({
                    "options": {'defaultType': 'swap', 'adjustForTimeDifference' : True},
                    "enableRateLimit": False
                    }) 
        except Exception as e:
            raise SystemExit( "Couldn't initialize exchange:", exchangeID )
        
        #################################################

        # the fetcher will be inside the stream
        fetcher = candles_c( self.exchange.id, self.symbol )

        #ohlcvs = fetcher.fetchAmount( stream.symbol, stream.timeframeStr, amount=10000 )
        ohlcvs = fetcher.loadCacheAndFetchUpdate( self.symbol, self.timeframeStr, max_amount )
        if( len(ohlcvs) == 0 ):
            raise SystemExit( f'No candles available in {stream.exchange.id}. Aborting')
        
        print( "Creating dataframe" )

        # take out the last row to jumpstart the generatedSeries later
        self.df = pd.DataFrame( ohlcvs[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'] )

        print( "Calculating generated series" )

        start_time = time.time()
        self.parseCandleUpdate( [ohlcvs[-1]] ) # de the jump-starting
        print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))


        print( "Computing script logic" )
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
        self.timestamp = 0
        self.barindex = -1
        self.parseCandleUpdate( ohlcvs )
        self.shadowcopy = False
        ###############################################################################

        print( len(self.df), "candles processed. Total time: {:.2f} seconds".format(time.time() - start_time))
        self.initializing = False
        self.initdata = [] # free memory
        ohlcvs = None

        tasks.registerTask( fetchCandleUpdates( self ) )

        
    def parseCandleUpdate( self, rows ):
        global activeStream
        activeStream = self
        for newrow in rows:
            newTimestamp = int(newrow[0])
            if( newTimestamp == None ):
                break
            
            oldTimestamp = int(self.df.iloc[-1]['timestamp']) if len(self.df) > 1 else 0
            if( oldTimestamp > newTimestamp ): # the server has sent a bunch of older candles
                continue

            if( oldTimestamp == newTimestamp ):

                # update the realtime candle
                self.df.loc[self.df.index[-1], 'open'] = newrow[1]
                self.df.loc[self.df.index[-1], 'high'] = newrow[2]
                self.df.loc[self.df.index[-1], 'low'] = newrow[3]
                self.df.loc[self.df.index[-1], 'close'] = newrow[4]
                self.df.loc[self.df.index[-1], 'volume'] = newrow[5]

                #update the chart
                if( window != None ):
                    window.updateChart()
                    
            else:
                if( not self.initializing ):
                    print( 'NEW CANDLE', newrow )

                # CLOSE REALTIME CANDLE

                self.barindex = self.df.iloc[-1].name
                self.timestamp = self.df['timestamp'].iloc[-1]
                new_row_index = self.barindex + 1

                runCloseCandle( self, self.df['open'], self.df['high'], self.df['low'], self.df['close'] )

                # if( not self.shadowcopy ):
                #     self.updateAllGeneratedSeries() # update all calculated series regardless if they are called or not

                # OPEN A NEW CANDLE

                if( self.shadowcopy ):
                    # row_to_append = self.initdata.iloc[new_row_index].to_frame().T
                    # self.df = pd.concat( [self.df, row_to_append], ignore_index=True )
                    self.df = pd.concat( [self.df, self.initdata.iloc[new_row_index].to_frame().T], ignore_index=True )

                else:
                    self.df.loc[new_row_index, 'timestamp'] = newTimestamp
                    self.df.loc[new_row_index, 'open'] = newrow[1]
                    self.df.loc[new_row_index, 'high'] = newrow[2]
                    self.df.loc[new_row_index, 'low'] = newrow[3]
                    self.df.loc[new_row_index, 'close'] = newrow[4]
                    self.df.loc[new_row_index, 'volume'] = newrow[5]

                # update the chart
                if( window != None ):
                    window.updateChart()

                if( self.shadowcopy and new_row_index % 5000 == 0 ):
                    print( new_row_index, "candles processed." )

                if( not self.shadowcopy and verbose ):
                    print( self.df )


    def calcGeneratedSeries( self, type:str, source:pd.Series, period:int, func, always_reset:bool = False ):
        name = generatedSeriesNameFormat( type, source, period )
        cseries = None
        # find if there's a item already created for this series
        for cs in self.generatedSeries:
            if cs.name == name:
                cseries = cs
                # print( 'found', name )
                break
        if cseries == None:
            cseries = generatedSeries_c( type, source, period, func, always_reset, self )
            self.generatedSeries.append(cseries)
        cseries.update( source )
        return cseries
    
    def createMarker( self, text:str, timestamp:int, chart_name:str = None ):
        if timestamp == -1:
            timestamp = self.timestamp
        self.markers.append( markers_c( text, timestamp, chart_name ) )

    def plot( self, name, source, chart_name = None ):
        plot = None
        for p in self.registeredPlots:
            if( name == p.name ):
                plot = p
                break

        if( plot == None ):
            plot = plot_c( name, chart_name )
            self.registeredPlots.append( plot )

        
        plot.update( source, self, window )
        return plot
    
    def jumpstartPlots( self, window ):
        for plot in self.registeredPlots:
            if not plot.initialized:
                plot.update( plot.source, self, window )

        


def createMarker( text:str = '', timestamp:int = -1, chart_name:str = None ):
    activeStream.createMarker( text, timestamp, chart_name )

def plot( name, source, chart_name = None ):
    activeStream.plot( name, source, chart_name )

registeredStreams:stream_c = []
activeStream:stream_c = None



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
            window = series[i - period + 1:i + 1]
            mean = window.mean()
            deviation = (window - mean).abs().sum() / period
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
    def __init__( self, type:str, source:pd.Series, period:int, func = None, always_reset:bool = False, stream:stream_c = None ):
        self.name = generatedSeriesNameFormat( type, source, period )
        self.sourceName = source.name
        self.period = period
        self.func = func
        self.stream = stream
        self.timestamp = 0
        self.alwaysReset = always_reset

        if( self.stream == None ):
            raise SystemError( f"Generated Series has no assigned stream [{self.name}]")
        
        if( self.stream.shadowcopy ):
            raise SystemError( f'Tried to create series [{self.name}] while shadowcopying.' )

        if( self.func == None ):
            raise SystemError( f"Generated Series without a func [{self.name}]")

        if( self.period < 1 ):
            raise SystemError( f"Generated Series  with invalid period [{period}]")
        

    def initialize( self, source:pd.Series ):
        if( len(source) >= self.period and ( not self.name in self.stream.df.columns or self.alwaysReset ) ):
            if( self.stream.shadowcopy ):
                raise SystemError( f"[{self.name}] tried to initialize as shadowcopy" )
            start_time = time.time()
            self.stream.df[self.name] = self.func(source, self.period, self.stream.df).dropna()
            self.timestamp = self.stream.df['timestamp'].iloc[-1]
            if( self.stream.initializing ):
                print( f"Initialized {self.name}." + " Elapsed time: {:.2f} seconds".format(time.time() - start_time))


    def update( self, source:pd.Series ):
        if( self.stream.shadowcopy ):
            return
        
        df = self.stream.df

        # has this row already been updated?
        if( self.timestamp >= df['timestamp'].iloc[-1] ):
            return

        # if non existant try to create new
        # the rma needs to be recalculated in full for every new candle
        if( self.timestamp == 0 or self.alwaysReset ):
            self.initialize( source )
            return
        
        # this happens when making the shadow copy
        if( not pd.isna( df[self.name].iloc[-1] ) ):
            return
            raise ValueError( f"generatedSeries {self.name} had a value with a outdated timestamp" )
        
        if( len(self.stream.df) < self.period ):
            return
        
        # realtime updates

        # slice the required block of candles to calculate the current value of the generated series
        newval = self.func(source[-self.period:], self.period, self.stream.df).iloc[-1]
        df.loc[df.index[-1], self.name] = newval
        self.timestamp = self.stream.timestamp
        
    def plot( self, chart = None ):
        if( self.timestamp > 0 ):
            plot( self.name, self.series(), chart )
    
    def series( self ):
        return self.stream.df[self.name]
    
    def crossingUp( self, other ):
        df = self.stream.df
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
        df = self.stream.df
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
        df = self.stream.df
        if( backindex < 0 or backindex + 1 > len(df) ):
            return None
            #raise KeyError( 'Invalid backindex. It must be 0 or more. Maybe you wanted to use iloc(index)')
        if( self.timestamp == 0 or pd.isna(df[self.name].iloc[-(backindex + 1)]) ):
            return None
        return df[self.name].iloc[-(backindex + 1)]
    
    def bool( self, backindex = 0 ):
        df = self.stream.df
        if( backindex < 0 or backindex + 1 > len(df) ):
            return None
            #raise KeyError( 'Invalid backindex. It must be 0 or more. Maybe you wanted to use iloc(index)')
        if( self.timestamp == 0 or pd.isna(df[self.name].iloc[-(backindex + 1)]) ):
            return None
        return df[self.name].iloc[-(backindex + 1)] != 0
        
    def loc( self, index = 0 ):
        df = self.stream.df
        if( self.timestamp == 0 ):
            print( f"Warning: {self.name} has not yet produced any value")
            return 0 # let's do this just to avoid crashes
        
        if( index < 0 or index > len(df) ):
            raise KeyError( 'Invalid index. It must be 0 or more')
        return df[self.name].loc[index]
    
    def iloc( self, index = -1 ):
        df = self.stream.df
        if( self.timestamp == 0 ):
            print( f"Warning: {self.name} has not yet produced any value")
            return 0 # let's do this just to avoid crashes
        
        return df[self.name].iloc[index]
    


def calcIndexWhenTrue( source ):
    if( not isinstance(source, pd.Series ) ):
        if( isinstance( source, generatedSeries_c) ):
            source = source.series()
        else:
            raise ValueError( "calcIndexWhenTrue must be called with a series" )
    boolean_source = source.astype(bool) if source.dtype != bool else source
    return boolean_source[::-1].idxmax() if source.any() else None


def calcBarsSince( source ):
    index_when_true = calcIndexWhenTrue( source )
    if( index_when_true == None ):
        return None
    return activeStream.barindex - index_when_true


def calcIndexWhenFalse( source ):
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


def calcBarsWhileTrue( source ):
    index_when_false = calcIndexWhenFalse( source )
    if( index_when_false == None ):
        return None
    return activeStream.barindex - index_when_false


# this can be done to any pandas_ta function that returns a series and takes as arguments a series and a period.
def falling( source:pd.Series, period:int ):
    return activeStream.calcGeneratedSeries( 'falling', source, period, generatedseries_calculate_falling )

def rising( source:pd.Series, period:int ):
    return activeStream.calcGeneratedSeries( 'rising', source, period, generatedseries_calculate_rising )

def calcSMA( source:pd.Series, period:int ):
    return activeStream.calcGeneratedSeries( 'sma', source, period, generatedseries_calculate_sma )

def calcEMA( source:pd.Series, period:int ):
    return activeStream.calcGeneratedSeries( "ema", source, period, generatedseries_calculate_ema )

def calcDEMA( source:pd.Series, period:int ):
    return activeStream.calcGeneratedSeries( "dema", source, period, generatedseries_calculate_dema, always_reset=True )

def calcWMA( source:pd.Series, period:int ):
    return activeStream.calcGeneratedSeries( "wma", source, period, generatedseries_calculate_wma )

def calcHMA( source:pd.Series, period:int ):
    return activeStream.calcGeneratedSeries( "hma", source, period, generatedseries_calculate_hma, always_reset=True )

# def calcJMA( source:pd.Series, period:int ):
#     return activeStream.calcGeneratedSeries( "jma", source, period, pt.jma )

# def calcKAMA( source:pd.Series, period:int ):
#     return activeStream.calcGeneratedSeries( "kama", source, period, pt.kama )

def calcLINREG( source:pd.Series, period:int ):
    return activeStream.calcGeneratedSeries( "linreg", source, period, generatedseries_calculate_linreg )

def calcRSI( source:pd.Series, period:int ):
    return activeStream.calcGeneratedSeries( 'rsi', source, period, generatedseries_calculate_rsi )

def calcDEV( source:pd.Series, period:int ):
    return activeStream.calcGeneratedSeries( 'dev', source, period, generatedseries_calculate_dev )

def calcSTDEV( source:pd.Series, period:int ):
    return activeStream.calcGeneratedSeries( 'stdev', source, period, generatedseries_calculate_stdev )

def calcRMA( source:pd.Series, period:int ):
    return activeStream.calcGeneratedSeries( 'rma', source, period, generatedseries_calculate_rma, always_reset=True )

def calcWPR( source:pd.Series, period:int ):
    return activeStream.calcGeneratedSeries( 'wpr', source, period, generatedseries_calculate_williams_r )

def calcATR2( period:int ): # The other one using pt is much faster
    tr = activeStream.calcGeneratedSeries( 'tr', pd.Series([pd.NA] * period, name = 'tr'), period, generatedseries_calculate_tr )
    return activeStream.calcGeneratedSeries( 'atr', tr.series(), period, generatedseries_calculate_rma )

def calcTR( period:int ):
    return activeStream.calcGeneratedSeries( 'tr', pd.Series([pd.NA] * period, name = 'tr'), period, generatedseries_calculate_tr )

def calcATR( period:int ):
    return activeStream.calcGeneratedSeries( 'atr', pd.Series([pd.NA] * period, name = 'atr'), period, generatedseries_calculate_atr )

def calcSLOPE( source:pd.Series, period:int ):
    return activeStream.calcGeneratedSeries( 'slope', source, period, generatedseries_calculate_slope, always_reset=True )

def calcBIAS( source:pd.Series, period:int ):
    return activeStream.calcGeneratedSeries( 'bias', source, period, generatedseries_calculate_bias )

def calcCCI( period:int ):
    return activeStream.calcGeneratedSeries( 'cci', pd.Series([pd.NA] * period, name = 'cci'), period, generatedseries_calculate_cci )

def calcCFO( source:pd.Series, period:int ):
    return activeStream.calcGeneratedSeries( 'cfo', source, period, generatedseries_calculate_cfo )

def calcFWMA( source:pd.Series, period:int ):
    return activeStream.calcGeneratedSeries( 'fwma', source, period, generatedseries_calculate_fwma )



def runCloseCandle( stream:stream_c, open:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series ):
    barindex = stream.barindex

    ###########################
    # strategy code goes here #
    ###########################

    sma = calcSMA( close, 350 )
    sma.plot()

    ema = calcEMA( close, 4 )
    ema.plot()

    calcLINREG( close, 300 ).plot()

    # rsi = calcRSI( close, 14 )
    # rsiplot = plot( rsi.name, rsi.series(), 'panel' )

    atr = calcATR( 14 )
    plot( atr.name, atr.series(), 'panel' )

    calcTR(14).plot('panel')

    # # sma_rising = rising( sma.name, 10 )

    # cfo = calcCFO( close, 20 )
    # cfo.plot(window.bottomPanel)

    # dev = calcDEV( close, 30 )
    # # plot( dev.name, dev.series(), window.bottomPanel )

    # rma = calcRMA( close, 90 )
    # rma.plot()

    stdev = calcSTDEV( close, 350 )

    # willr = calcWPR( close, 32 ).plot('panel')
    # calcBIAS( close, 32 ).plot(window.bottomPanel)

    # hma = calcHMA( close, 150 )
    # hma.plot()
    # r = rising( hma.series(), 10 )
    # f = falling( hma.series(), 10 )
    # if( not activeStream.initializing ):
    #     print( (hma.series() > 1.7) )


    # calcBarsSince( r )
    # calcBarsWhileTrue( hma.series() > 1.7 )

    # calcCCI( 20 )

    # slope1000 = calcSMA( calcSLOPE( close, 200 ).series() * 500000, 14 )
    # plot( slope1000.name, slope1000.series(), window.bottomPanel )


    if( sma.crossingUp(close) ):
        createMarker( '🔷' )

    if crossingDown( sma, ema ):
        createMarker( '🔺' )

    # plot( "lazyline", 30, window.bottomPanel )



    return


async def fetchCandleUpdates( stream:stream_c ):

    maxRows = 100
    while True:
        try:
            response = await stream.exchange.watch_ohlcv( stream.symbol, stream.timeframeStr, limit = maxRows )
            #print(response)

        except Exception as e:
            print( 'Exception raised at fetchCandleupdates: Reconnecting', e, type(e) )
            await stream.exchange.close()
            await asyncio.sleep(1.0)
            continue
        
            
        # extract the data

        if( len(response) > maxRows ):
            response = response[len(response)-maxRows:]

        stream.parseCandleUpdate( response )
        
        await asyncio.sleep(0.005)

    await exchange.close()


from datetime import datetime
async def update_clock(stream):
    if( window == None ):
        return
    while window.chart.is_alive:
        await asyncio.sleep(1-(datetime.now().microsecond/1_000_000))
        window.chart.legend( visible=True, ohlc=False, percent=False, font_size=18, text=stream.symbol + ' - ' + stream.timeframeStr + ' - ' + stream.exchange.id + ' - ' + f'candles:{len(stream.df)}' + ' - ' + datetime.now().strftime('%H:%M:%S') )


async def on_timeframe_selection(chart):
    print( f'Getting data with a {chart.topbar["my_switcher"].value} timeframe.' )



if __name__ == '__main__':

    # WIP stream
    stream = stream_c( 'LDO/USDT:USDT', 'bitmart', '5m', 10000 )
    registeredStreams.append( stream )

    
    tasks.registerTask( update_clock(stream) )

    window = createWindow( stream )

    #stream.parseCandleUpdate( [last_ohlcv] ) # jump-start the chart plots
    
    asyncio.run( tasks.runTasks() )


    



