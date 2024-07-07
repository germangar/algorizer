#lightweight-charts simple test

import pandas as pd
import pandas_ta as ta
import math
from lightweight_charts import Chart
import asyncio
import ccxt.pro as ccxt
import time
from pprint import pprint

import tools
from tools import df_append
from fetcher import candles_c


SHOW_VOLUME = False
chart_opened = False



def crossingUp( self, other ):
    if isinstance( self, customSeries_c ):
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
        elif isinstance( other, customSeries_c ):
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
    if isinstance( self, customSeries_c ):
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
        elif isinstance( other, customSeries_c ):
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
    def __init__( self, name:str, source:pd.DataFrame, chart = None ):
        self.name = name
        self.chart = chart
        self.line = None
        self.initialized = False

    def update( self, source:pd.DataFrame, chart = None ):
        if( not chart_opened ):
            return

        if( self.chart == None ):
            if( chart != None ):
                self.chart = chart
                print( "Plot assigned a chart")
            else:
                print( "Can't initialize a plot not associated with a chart")
                return
        
        if( not self.initialized ):
            if( len(source)<1 ): # pd.isna(source)
                return
            self.line = chart.create_line( self.name, price_line=False, price_label=False )
            self.line.set( pd.DataFrame({'time': pd.to_datetime( activeContext.df['timestamp'], unit='ms' ), self.name: source}).dropna() )
            self.initialized = True
            return

        # it's initalized so only update the new line
        newval = source.iloc[-1]
        timestamp = int(activeContext.df.iloc[-1]['timestamp'])
        self.line.update( pd.Series( {'time': pd.to_datetime( timestamp, unit='ms' ), 'value': newval } ) )


registeredPlots:plot_c = []

def plot( name, source:pd.DataFrame, chart ):
    if( chart == None ):
        return
    plot = None
    for thisPlot in registeredPlots:
        if( name == thisPlot.name ):
            plot = thisPlot
            #print( 'found Plot' )
            break

    if( plot == None ):
        plot = plot_c( name, source, chart )
        registeredPlots.append( plot )

    plot.update( source, chart )

class markers_c:
    def __init__( self, text:str, timestamp, chart = None ):
        self.timestamp = timestamp
        self.text = text
        self.chart = chart
        self.marker = None
    
    def __del__( self ):
        self.remove()

    def remove( self ):
        if( self.marker != None ):
            if( self.chart ):
                self.chart.remove_marker( self.marker )
            self.marker = None

    def refreshInChart( self, chart ):
        self.remove()
        if( self.chart == None ):
            self.chart = chart
        self.marker = self.chart.marker( time = pd.to_datetime( self.timestamp, unit='ms' ), text = self.text )


class context_c:
    def __init__( self, symbol, exchangeID:str, timeframe ):
        self.symbol = symbol # FIXME: add verification
        self.market = None
        self.timeframe = timeframe if( type(timeframe) == int ) else tools.timeframeInt(timeframe)
        self.timeframeStr = tools.timeframeString( self.timeframe )
        self.barindex = -1
        self.timestamp = 0
        self.initializing = True
        self.shadowcopy = False
        self.chart = None

        self.markers:markers_c = []
        self.customSeries:customSeries_c = []

        self.df:pd.DataFrame = []
        self.initdata:pd.DataFrame = []

        try:
            self.exchange = getattr(ccxt, exchangeID)({
                    "options": {'defaultType': 'swap', 'adjustForTimeDifference' : True},
                    "enableRateLimit": False
                    }) 
        except Exception as e:
            raise SystemExit( "Couldn't initialize exchange:", exchangeID )
        
    def parseCandleUpdate( self, rows ):
        global activeContext
        activeContext = self
        chart = self.chart
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
                if( chart != None and not self.initializing ):
                    data_dict = {'time': pd.to_datetime( self.df['timestamp'].iloc[-1], unit='ms' ), 'open': self.df['open'].iloc[-1], 'high': self.df['high'].iloc[-1], 'low': self.df['low'].iloc[-1], 'close': self.df['close'].iloc[-1] }
                    if SHOW_VOLUME:
                        data_dict['volume'] = self.df['volume'].iloc[-1]
                    chart.update( pd.Series(data_dict) )

            else:
                if( not self.initializing ):
                    print( 'NEW CANDLE', newrow )

                # CLOSE REALTIME CANDLE

                self.barindex = self.df.iloc[-1].name
                self.timestamp = self.df['timestamp'].iloc[-1]
                new_row_index = self.barindex + 1

                # if( self.shadowcopy and self.barindex + 1 != len(self.df) ):
                #     raise SystemError( f"self.barindex [{self.barindex}] != len(self.df [{len(self.df)}])" )

                if( not self.shadowcopy ):
                    self.updateAllCustomSeries() # update all calculated series regardless if they are called or not

                runCloseCandle( self, self.df['open'], self.df['high'], self.df['low'], self.df['close'] )

                # OPEN A NEW CANDLE

                if( self.shadowcopy ):
                    row_to_append = self.initdata.iloc[new_row_index].to_frame().T
                    self.df = pd.concat( [self.df, row_to_append], ignore_index=True )
                else:
                    self.df.loc[new_row_index, 'timestamp'] = newTimestamp
                    self.df.loc[new_row_index, 'open'] = newrow[1]
                    self.df.loc[new_row_index, 'high'] = newrow[2]
                    self.df.loc[new_row_index, 'low'] = newrow[3]
                    self.df.loc[new_row_index, 'close'] = newrow[4]
                    self.df.loc[new_row_index, 'volume'] = newrow[5]

                # update the chart
                if( chart != None and not self.initializing ):
                    data_dict = {'time': pd.to_datetime( self.df['timestamp'].iloc[-1], unit='ms' ), 'open': self.df['open'].iloc[-1], 'high': self.df['high'].iloc[-1], 'low': self.df['low'].iloc[-1], 'close': self.df['close'].iloc[-1] }
                    if SHOW_VOLUME:
                        data_dict['volume'] = self.df['volume'].iloc[-1]
                    chart.update( pd.Series(data_dict) )

                    chart.legend( visible=True, ohlc=False, percent=False, font_size=18, text=self.symbol + ' - ' + self.timeframeStr + ' - ' + self.exchange.id + ' - ' + f'candles:{len(self.df)}' )

                runOpenCandle( self )
                if( self.shadowcopy and new_row_index % 5000 == 0 ):
                    print( new_row_index, "candles processed." )

                # if( not self.shadowcopy ):
                #     print( self.df )

    def updateAllCustomSeries( self ):
        for cseries in self.customSeries:
            cseries.update()

    def calcCustomSeries( self, type:str, source:str, period:int, func ):
        name = f'{type} {source} {period}'
        cseries = None
        # find if there's a item already created for this series
        for thisCS in self.customSeries:
            if thisCS.name == name:
                cseries = thisCS
                # print( 'found', name )
                break
        if cseries == None:
            cseries = customSeries_c( type, source, period, func, self )
            self.customSeries.append(cseries)
        cseries.update()
        return cseries


    def createMarker( self, text:str ):
        self.markers.append( markers_c( text, self.timestamp, self.chart ) )

    

registeredContexts:context_c = []
activeContext:context_c = None

def customseries_calculate_rma(series: pd.Series, length: int) -> pd.Series:
    # RMA needs to be recalculated in full every time
    return series.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

def customseries_calculate_dev(series: pd.Series, period: int) -> pd.Series:
    if 1:
        return ta.mad( series, period )
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

def customseries_calculate_williams_r(series: pd.Series, period: int) -> pd.Series:
    if 0:
        """
        Calculate Williams %R for a given series using OHLC data from activeContext.df over a period.

        Args:
        - series: pd.Series, typically a placeholder, but required for compatibility with customSeries_c.
        - period: int, the period/window for the Williams %R calculation.

        Returns:
        - pd.Series, the calculated Williams %R values.
        """
        # global activeContext

        # Ensure activeContext and its DataFrame are accessible
        if 'activeContext' not in globals():
            raise ValueError("activeContext is not defined in the global scope")

        df = activeContext.df

        # Ensure the DataFrame has the required columns
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            raise ValueError("The global DataFrame must contain 'high', 'low', and 'close' columns")

        if len(df) < period:
            return pd.Series([pd.NA] * len(df), index=df.index)  # Not enough data to calculate Williams %R

        # Initialize a list to hold the Williams %R values
        williams_r_values = [pd.NA] * (period - 1)  # NA for the initial period
    else:
        return ta.willr( activeContext.df['high'], activeContext.df['low'], activeContext.df['close'], length=period )

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

def customseries_calculate_rsi(series, period) -> pd.Series:
    deltas = series.diff()
    gain = deltas.where(deltas > 0, 0).rolling(window=period).mean()
    loss = -deltas.where(deltas < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi #.iloc[-1]  # Returning the last value of RSI

class customSeries_c:
    def __init__( self, type:str, source:str, period:int, func = None, context:context_c = None ):
        self.name = f'{type} {source} {period}'
        self.source = source
        self.period = period
        self.func = func
        self.context = context
        self.timestamp = 0
        self.alwaysReset = True if self.func == customseries_calculate_rma or self.func == ta.rma else False

        if( self.context == None ):
            raise SystemError( f"Custom Series has no assigned context [{self.name}]")

        if( self.func == None ):
            raise SystemError( f"Custom Series without a func [{self.name}]")

        if( not self.source in context.df.columns ):
            raise SystemError( f"Custom Series  with unknown source [{source}]")

        if( self.period < 1 ):
            raise SystemError( f"Custom Series  with invalid period [{period}]")
        
    def initialize( self ):
        if( len(self.context.df) >= self.period and ( not self.name in self.context.df.columns or self.alwaysReset ) ):
            if( self.context.shadowcopy ):
                raise SystemError( f"[{self.name}] tried to initialize as shadowcopy" )
            start_time = time.time()
            self.context.df[self.name] = self.func(self.context.df[self.source], self.period)
            self.timestamp = self.context.df['timestamp'].iloc[-1]
            if( self.context.initializing ):
                print( f"Initialized {self.name}." + " Elapsed time: {:.2f} seconds".format(time.time() - start_time))

    def update( self ):
        if( self.context.shadowcopy ):
            return
        
        df = self.context.df

        # has this row already been updated?
        if( self.timestamp >= df['timestamp'].iloc[-1] ):
            # print( self.name, self.timestamp, "datafrne:", df['timestamp'].iloc[-1] )
            return

        # if non existant try to create new
        # the rma needs to be recalculated in full for every new candle
        if( self.timestamp == 0 or self.alwaysReset ):
            self.initialize()
            return
        
        # this happens when making the shadow copy
        if( not pd.isna( df[self.name].iloc[-1] ) ):
            return
            raise ValueError( f"customSeries {self.name} had a value with a outdated timestamp" )
        
        if( len(self.context.df) < self.period ):
            return
        
        # This should only happen in realtime updates
        # print( f"Updating {self.name}" )
        
        # isolate only the required block of candles to calculate the current value of the custom series
        # Extract the last 'num_rows' rows of the specified column into a new DataFrame
        #sdf = df[self.source].tail(self.period).to_frame(name=self.source)
        #newval = self.func( sdf[self.source], self.period ).iloc[-1]
        #sdf = df.iloc[-self.period:, :][self.source]
        newval = self.func( df.iloc[-self.period:, :][self.source], self.period ).iloc[-1]
        
        # with full dataframe: newval = self.func( df.iloc[-self.period:, :], self.period ).iloc[-1]
        #newval = self.func( df.iloc[-self.period:, :][self.name], self.period ).iloc[-1]
        df.loc[df.index[-1], self.name] = newval
        self.timestamp = df['timestamp'].iloc[-1]
        
    def plot( self ):
        chart = self.context.chart
        if( self.timestamp > 0 or chart != None ):
            plot( self.name, self.plotData(), chart )
    
    def plotData( self ):
        df = self.context.df
        if( self.timestamp == 0 ):
            return pd.NA
            #return pd.DataFrame( columns = ['timestamp', self.name] )
        #return pd.DataFrame({'timestamp': df['timestamp'], self.name: df[self.name]}).dropna()
        return df[self.name]
    
    def crossingUp( self, other ):
        df = self.context.df
        if( self.timestamp == 0 or len(df)<1 ):
            return False
        if isinstance( other, customSeries_c ):
            if( other.timestamp == 0 ):
                return False
            return ( self.value(1) <= other.value(1) and self.value() >= other.value() and self.value() != self.value(1) )
        if isinstance( other, pd.Series ):
            if( len(other) < 2 ):
                return False
            return ( self.value(1) <= other.iloc[-2] and self.value() >= other.iloc[-1] and self.value() != self.value(1) )
        try:
            float(other)
        except ValueError:
            return False
        else:
            return ( self.value(1) <= float(other) and self.value() >= float(other) and self.value() != self.value(1) )
    
    def crossingDown( self, other ):
        df = self.context.df
        if( self.timestamp == 0 or len(df)<1 ):
            return False
        if isinstance( other, customSeries_c ):
            if( other.timestamp == 0 ):
                return False
            return ( self.value(1) >= other.value(1) and self.value() <= other.value() and self.value() != self.value(1) )
        if isinstance( other, pd.Series ):
            if( len(other) < 2 ):
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
        df = self.context.df
        if( self.timestamp == 0 ):
            #print( f"Warning: {self.name} has not yet produced any value")
            return 0 # let's do this just to avoid crashes
        
        if( backindex < 0 ):
            raise KeyError( 'Invalid backindex. It must be 0 or more. Maybe you wanted to use iloc(index)')
        return df[self.name].iloc[-(backindex + 1)]
        
    def loc( self, index = 0 ):
        df = self.context.df
        if( self.timestamp == 0 ):
            print( f"Warning: {self.name} has not yet produced any value")
            return 0 # let's do this just to avoid crashes
        
        if( index < 0 or index > len(df) ):
            raise KeyError( 'Invalid index. It must be 0 or more')
        return df[self.name].loc[index]
    
    def iloc( self, index = -1 ):
        df = self.context.df
        if( self.timestamp == 0 ):
            print( f"Warning: {self.name} has not yet produced any value")
            return 0 # let's do this just to avoid crashes
        
        return df[self.name].iloc[index]
    

# this can be done to any pandas_ta function that returns a series and takes as arguments a series and a period.
def calcSMA( source:str, period:int ):
    return activeContext.calcCustomSeries( 'sma', source, period, ta.sma )

def calcEMA( source:str, period:int ):
    return activeContext.calcCustomSeries( "ema", source, period, ta.ema )

def calcRSI( source:str, period:int ):
    return activeContext.calcCustomSeries( 'rsi', source, period, customseries_calculate_rsi )

def calcDEV( source:str, period:int ):
    return activeContext.calcCustomSeries( 'dev', source, period, customseries_calculate_dev )

def calcSTDEV( source:str, period:int ):
    return activeContext.calcCustomSeries( 'stdev', source, period, ta.stdev )

def calcRMA( source:str, period:int ):
    return activeContext.calcCustomSeries( 'rma', source, period, customseries_calculate_rma )

def calcWPR( source:str, period:int ):
    return activeContext.calcCustomSeries( 'wpr', source, period, customseries_calculate_williams_r )


def runOpenCandle( context:context_c ):
    return


def runCloseCandle( context:context_c, open:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series ):
    barindex = context.barindex

    ###########################
    # strategy code goes here #
    ###########################

    sma = calcSMA( 'close', 90 )
    sma.plot()

    ema = calcEMA( 'close', 4 )
    plot( ema.name, ema.plotData(), context.chart )

    rsi = calcRSI( 'close', 14 )

    dev = calcDEV( 'close', 30 )
    #plot( dev.name, dev.plotData(), context.chart )

    rma = calcRMA( 'close', 500 )
    rma.plot()

    stdev = calcSTDEV( 'close', 350 )

    willr = calcWPR( 'close', 32)

    calcSMA( 'close', 500 )

    # create_histogram(name: str, color: COLOR, price_line: bool, price_label: bool, scale_margin_top: float, scale_margin_bottom: float)
    # if( context.chart != None and rsi.timestamp != 0 ):
    #     hist = context.chart.create_histogram( rsi.name, price_line = False, price_label = True )
    #     hist.set( rsi.plotData() )

    if( sma.crossingUp(close) ):
        context.createMarker( text='🔷' )

    if( crossingDown( sma, close ) ):
        context.createMarker( text='🔺' )

    

    return


async def fetchCandleUpdates( context:context_c ):

    maxRows = 100
    while True:
        response = await context.exchange.watch_ohlcv( context.symbol, context.timeframeStr, limit = maxRows )
        #print(response)

        # extract the data

        if( len(response) > maxRows ):
            response = response[len(response)-maxRows:]

        #print( f"FETCHED {len(response)} CANDLES")
        context.parseCandleUpdate( response )
        
        await asyncio.sleep(0.003)

    await exchange.close()


async def doNothing():
    while True:
        # doing nothing yet
        await asyncio.sleep(1)

from datetime import datetime
async def update_clock(context):
    if( context.chart == None ):
        return
    while context.chart.is_alive:
        await asyncio.sleep(1-(datetime.now().microsecond/1_000_000))
        context.chart.legend( visible=True, ohlc=False, percent=False, font_size=18, text=context.symbol + ' - ' + context.timeframeStr + ' - ' + context.exchange.id + ' - ' + f'candles:{len(context.df)}' + ' - ' + datetime.now().strftime('%H:%M:%S') )

async def runTasks(context):
    task1 = asyncio.create_task(fetchCandleUpdates(context))
    task2 = asyncio.create_task(update_clock(context))
    task3 = context.chart.show_async() if context.chart is not None else None

    tasks = [task for task in [task1, task2, task3] if task is not None]
    await asyncio.gather(*tasks)

    

async def on_timeframe_selection(chart):
    print( f'Getting data with a {chart.topbar["my_switcher"].value} timeframe.' )

async def on_button_press(chart):
    new_button_value = 'On' if chart.topbar['my_button'].value == 'Off' else 'Off'
    chart.topbar['my_button'].set(new_button_value)
    print(f'Turned something {new_button_value.lower()}.')

# def on_horizontal_line_move(chart, line):
#     print(f'Horizontal line moved to: {line.price}')

def launchChart( context:context_c, last_ohlcv ):
    global chart_opened
    ##########################
    #### Set up the chart ####
    ##########################
    
    tmpdf = pd.DataFrame( { 'time':pd.to_datetime( context.df['timestamp'], unit='ms' ), 'open':context.df['open'], 'high':context.df['high'], 'low':context.df['low'], 'close':context.df['close']} )
    if( SHOW_VOLUME ):
        tmpdf['volume'] = context.df['volume']

    chart = Chart( toolbox = False )
    chart.legend( visible=True, ohlc=False, percent=False, font_size=18, text=context.symbol + ' - ' + context.timeframeStr + ' - ' + context.exchange.id + ' - ' + f'candles:{len(context.df)}' )
    chart.precision(4)
    #chart.watermark("Hello World")
    # chart.layout( background_color='rgb(249, 250, 246)', text_color='rgb(54, 71, 77)', font_size=14 )
    # chart.grid(vert_enabled: bool, horz_enabled: bool, color: COLOR, style: LINE_STYLE)
    chart.layout( font_size=14 )

    # chart.topbar.switcher(
    #     name='my_switcher',
    #     options=('1min', '5min', '30min'),
    #     default='5min',
    #     func=on_timeframe_selection)
    
    chart.topbar.button('my_button', 'Off', func=on_button_press)

    # chart.horizontal_line(1.6, func=on_horizontal_line_move)
        
    chart.set(tmpdf)
    #chart.show( block=False )
    chart_opened = True

    # dump all the collected markers into the chart
    for marker in context.markers:
        marker.refreshInChart( chart )

    context.chart = chart
    context.parseCandleUpdate( last_ohlcv )

    # chart.show( block=False )
    return chart

if __name__ == '__main__':

    # WIP context
    context = context_c( 'LDO/USDT:USDT', 'bitmart', '1m' )
    registeredContexts.append( context )

    # the fetcher will be inside the context
    fetcher = candles_c( context.exchange.id, context.symbol )

    # filename = f'stuff/{exchangeName}-{coin}-USDT-{timeframe}.csv'
    # df = pd.read_csv( filename )
    # print( 'Loading', filename )
    
    ohlcvs = fetcher.fetchAmount( context.symbol, context.timeframeStr, amount=10000 )


    context.df = pd.DataFrame( ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'] )

    total_start_time = time.time()
    # delete the last row in the dataframe and extract the last row in ohlcvs.
    print( "Creating dataframe" )
    context.df.drop( context.df.tail(1).index, inplace=True )
    last_ohlcv = ohlcvs[-1]
    ohlcvs = ohlcvs[:-1]
    print( "Calculating custom series" )
    start_time = time.time()
    context.parseCandleUpdate( [last_ohlcv] )
    print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))

    ###############################################################################
    # at this point we have the customSeries initialized for the whole dataframe
    # move the dataframe to use it as source for the initialization with precomputed data
    print( "Computing script logic" )

    context.timestamp = 0
    context.barindex = -1
    context.initdata = context.df
    context.shadowcopy = True

    start_time = time.time()
    # start with a blank dataframe with only the first row copied from the precomputed dataframe
    context.df = pd.DataFrame( pd.DataFrame(context.initdata.iloc[0]).T, columns=context.initdata.columns )
    
    # run the script
    context.parseCandleUpdate( ohlcvs )
    context.shadowcopy = False
    print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))
    ###############################################################################

    # jump-start the chart by running the last row as if it was a update
    start_time = time.time()
    print( len(context.df), "candles processed. Total time: {:.2f} seconds".format(time.time() - total_start_time))
    context.initializing = False
    context.initdata = None # free memory
    ohlcvs = None

    launchChart( context, [last_ohlcv] )
    asyncio.run( runTasks(context) )



