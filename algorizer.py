#lightweight-charts simple test

import pandas as pd
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
            if( other.timestamp == 0 or len(df) < 2 ):
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
            if( other.timestamp == 0 or len(df) < 2 ):
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
            self.line.set( pd.DataFrame({'time': pd.to_datetime( source['timestamp'], unit='ms' ), self.name: source[self.name]}).dropna() )
            self.initialized = True
            return

        # it's initalized so only update the new line
        newval = source.iloc[-1][self.name]
        timestamp = int(source.iloc[-1]['timestamp'])
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
        global activeConext
        activeConext = self
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
activeConext:context_c = None


# Define the function for RSI calculation using apply
def customseries_calculate_rsi(series, period):
    deltas = series.diff()
    gain = deltas.where(deltas > 0, 0).rolling(window=period).mean()
    loss = -deltas.where(deltas < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]  # Returning the last value of RSI

def customseries_calculate_sma(series: pd.Series, period: int) -> float:
    if len(series) < period:
        return pd.NA  # Not enough data to calculate the SMA
    return series.mean()

def customseries_calculate_ema(series: pd.Series, period: int) -> float:
    if len(series) < period:
        return pd.NA  # Not enough data to calculate the EMA
    alpha = 2 / (period + 1)
    ema = series.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    return ema

def customseries_calculate_stdev(series: pd.Series, length: int) -> float:
    if len(series) < length:
        return pd.NA  # Not enough data to calculate the standard deviation

    avg = series.rolling(window=length).mean().iloc[-1]  # Calculate the SMA
    sum_of_square_deviations = 0.0

    for i in range(len(series) - length, len(series)):
        deviation = series.iloc[i] - avg
        sum_of_square_deviations += deviation ** 2

    stdev = math.sqrt(sum_of_square_deviations / length)
    return stdev

def customseries_calculate_rma(series: pd.Series, length: int) -> float:
    if len(series) < length:
        return pd.NA  # Not enough data to calculate the RMA

    alpha = 1 / length

    # Initialize sum with SMA if previous sum is NaN, otherwise calculate RMA recursively
    sum_value = series.rolling(window=length).mean().iloc[length - 1]

    for i in range(length, len(series)):
        if pd.isna(sum_value):
            sum_value = series.iloc[:length].mean()

        sum_value = alpha * series.iloc[i] + (1 - alpha) * sum_value

    return sum_value

def customseries_calculate_dev(series: pd.Series, length: int) -> float:
    """
    Calculate the average deviation over a given rolling window in a pandas Series.
    """
    if len(series) < length:
        return pd.NA  # Not enough data to calculate the deviation

    mean = series.rolling(window=length).mean().iloc[-1]  # Calculate the SMA
    deviation_sum = 0.0

    for i in range(len(series) - length, len(series)):
        val = series.iloc[i]
        deviation_sum += abs(val - mean)

    average_deviation = deviation_sum / length
    return average_deviation

class customSeries_c:
    def __init__( self, type:str, source:str, period:int, func = None, context:context_c = None ):
        self.name = f'{type} {source} {period}'
        self.source = source
        self.period = period
        self.func = func
        self.context = context
        self.timestamp = 0

        if( self.context == None ):
            raise SystemError( f"Custom Series has no assigned context [{self.name}]")

        if( self.func == None ):
            raise SystemError( f"Custom Series without a func [{self.name}]")

        if( not self.source in context.df.columns ):
            raise SystemError( f"Custom Series  with unknown source [{source}]")

        if( self.period < 1 ):
            raise SystemError( f"Custom Series  with invalid period [{period}]")
        
    def initialize( self ):
        if( len(self.context.df) >= self.period and not self.name in self.context.df.columns ):
            if( self.context.shadowcopy ):
                raise SystemError( f"[{self.name}] tried to initialize as shadowcopy" )
            start_time = time.time()
            self.context.df[self.name] = self.context.df[self.source].rolling(window=self.period).apply(lambda x: self.func(x, self.period))
            self.timestamp = self.context.df['timestamp'].iloc[-1]
            print( f"Initialized {self.name}." + " Elapsed time: {:.2f} seconds".format(time.time() - start_time))

    def update( self ):
        if( self.context.shadowcopy ):
            return
        
        df = self.context.df
        #if non existant try to create new
        if( self.timestamp == 0 ):
            self.initialize()
            return
        
        # has this row already been updated?
        if( self.timestamp >= df['timestamp'].iloc[-1] ):
            # print( self.name, self.timestamp, "datafrne:", df['timestamp'].iloc[-1] )
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
        sdf = df[self.source].tail(self.period).to_frame(name=self.source)
        newval = self.func( sdf[self.source], self.period )
        df.loc[df.index[-1], self.name] = newval
        self.timestamp = df['timestamp'].iloc[-1]
        
    def plot( self ):
        chart = self.context.chart
        if( self.timestamp > 0 or chart != None ):
            plot( self.name, self.plotData(), chart )
    
    def plotData( self ):
        df = self.context.df
        if( self.timestamp == 0 ):
            return pd.DataFrame( columns = ['timestamp', self.name] )
        return pd.DataFrame({'timestamp': df['timestamp'], self.name: df[self.name]}).dropna()
    
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

def calcSMA( source:str, period:int ):
    return activeConext.calcCustomSeries( 'sma', source, period, customseries_calculate_sma )

def calcEMA( source:str, period:int ):
    return activeConext.calcCustomSeries( "ema", source, period, customseries_calculate_ema )

def calcRSI( source:str, period:int ):
    return activeConext.calcCustomSeries( 'rsi', source, period, customseries_calculate_rsi )





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


async def otherstuff():
    while True:
        # doing nothing yet
        await asyncio.sleep(1)

# from datetime import datetime
# async def update_clock(chart):
#     if( chart == None ):
#         return
#     while chart.is_alive:
#         await asyncio.sleep(1-(datetime.now().microsecond/1_000_000))
#         chart.legend( visible=True, ohlc=False, percent=False, font_size=18, text=symbol + ' - ' + timeframe + ' - ' + exchangeName + ' - ' + f'candles:{len(ohlcvs)}' + ' - ' + datetime.now().strftime('%H:%M:%S') )

async def runTasks( context ):
    # Start the fetchCandleUpdates function
    task1 = asyncio.create_task( fetchCandleUpdates(context) )
    
    # Start the monitorOtherUpdates function
    task2 = asyncio.create_task( otherstuff() )

    # clock
    # task3 = asyncio.create_task( update_clock(context.chart) )

    # Run functions concurrently
    await asyncio.gather( task1, task2 )

# async def on_timeframe_selection(chart):
#     print( f'Getting data with a {chart.topbar["my_switcher"].value} timeframe.' )

# async def on_button_press(chart):
#     new_button_value = 'On' if chart.topbar['my_button'].value == 'Off' else 'Off'
#     chart.topbar['my_button'].set(new_button_value)
#     print(f'Turned something {new_button_value.lower()}.')

# def on_horizontal_line_move(chart, line):
#     print(f'Horizontal line moved to: {line.price}')

def launchChart( df ):
    global chart_opened
    ##########################
    #### Set up the chart ####
    ##########################
    
    tmpdf = pd.DataFrame( { 'time':pd.to_datetime( df['timestamp'], unit='ms' ), 'open':df['open'], 'high':df['high'], 'low':df['low'], 'close':df['close']} )
    if( SHOW_VOLUME ):
        tmpdf['volume'] = df['volume']

    chart = Chart( toolbox = False )
    chart.legend( visible=True, ohlc=False, percent=False, font_size=18, text=context.symbol + ' - ' + context.timeframeStr + ' - ' + context.exchange.id + ' - ' + f'candles:{len(df)}' )
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
    
    # chart.topbar.button('my_button', 'Off', func=on_button_press)

    # chart.horizontal_line(2.080, func=on_horizontal_line_move)
        
    chart.set(tmpdf)
    chart.show( block=False )
    chart_opened = True

    # dump all the collected markers into the chart
    for marker in context.markers:
        marker.refreshInChart( chart )

    return chart

if __name__ == '__main__':

    # WIP context
    context = context_c( 'LDO/USDT:USDT', 'bitget', '1m' )
    registeredContexts.append( context )

    # the fetcher will be inside the context
    fetcher = candles_c( context.exchange.id, context.symbol )

    # filename = f'stuff/{exchangeName}-{coin}-USDT-{timeframe}.csv'
    # df = pd.read_csv( filename )
    # print( 'Loading', filename )
    
    ohlcvs = fetcher.fetchAmount( context.symbol, context.timeframeStr, amount=2000 )

    if 1:
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
        print( "Finishing" )
        context.chart = launchChart( context.df )
        context.parseCandleUpdate( [last_ohlcv] ) # parse the last candle so the chart graphics are jumpstarted
        print( len(context.df), "candles processed. Total time: {:.2f} seconds".format(time.time() - total_start_time))
        context.initializing = False
        context.initdata = None # free memory
        ohlcvs = None

    ##########################
    #### Set up the data and perform the backtest ####
    ##########################
    else:
        #   extract the last ohlcv on the list
        last_ohlcv = ohlcvs[-1]
        ohlcvs = ohlcvs[:-1]
        context.df = pd.DataFrame( [ohlcvs[0]], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'] )

        print('---------------------------------------------------')
        print('  Processing. This may take a while. Please wait')

        start_time = time.time()

        context.parseCandleUpdate( ohlcvs )
        context.initializing = False

        ohlcvs.append( last_ohlcv ) # this is not really needed, but... restore the ohlcvs list to its original form

        print('                   Done.')
        print("  Elapsed time: {:.2f} seconds".format(time.time() - start_time))
        print('---------------------------------------------------')

        print( "OHLCVs PARSED" )

        context.chart = launchChart( context.df )

        #jump start it with the last candle
        print( "PARSING LAST OHLCV" )
        context.parseCandleUpdate( [last_ohlcv] )
        print( "DONE LAST OHLCV" )

    #########################################################

    asyncio.run( runTasks(context) )

    

