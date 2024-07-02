#lightweight-charts simple test

import pandas as pd
# import math
from lightweight_charts import Chart
import asyncio
import ccxt.pro as ccxt
import time
from pprint import pprint

import tools
from tools import df_append
from fetcher import candles_c


SHOW_VOLUME = False



class context_c:
    def __init__( self, symbol, exchangeID:str, timeframe ):
        self.symbol = symbol # FIXME: add verification
        self.timeframe = timeframe if( type(timeframe) == int ) else tools.timeframeInt(timeframe)
        self.timeframeName = tools.timeframeString( self.timeframe )
        self.timestamp = 0

        self.exchange = getattr(ccxt, exchangeID)({
                "options": {'defaultType': 'swap', 'adjustForTimeDifference' : True},
                "enableRateLimit": False
                }) 

        ###################################
        #### Initialize the dataframe #####
        ###################################
        # load dataframe from cache (to do)
        ohlcvs = candles.fetchAmount( symbol, timeframe=timeframe, amount=1000 )
        self.df  = pd.DataFrame( ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'] )

        # delete the last row in the dataframe and extract the last row in ohlcvs.
        self.df .drop( self.df .tail(1).index, inplace=True )
        last_ohlcv = [ohlcvs[-1]]

        # jump-start the series and plots calculation by running the last row as if it was a update
        parseCandleUpdate( last_ohlcv, chart )


    # update dataframe from 
    
    def newCandle():
        return
    def updateRealtimeCandle():
        return
    


df:pd.DataFrame = []

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


class customSeries_c:
    def __init__( self, type:str, source:str, period:int, func = None ):
        self.name = f'{type} {source} {period}'
        self.source = source
        self.period = period
        self.func = func
        self.initialized = False

        if( self.func == None ):
            raise SystemError( f"Custom Series without a func [{self.name}]")

        if( not self.source in df.columns ):
            raise SystemError( f"SMA with unknown source [{source}]")

        if( self.period < 1 ):
            raise SystemError( f"SMA with invalid period [{period}]")

    def update( self ):

        #if non existant try to create new
        if( not self.initialized ):
            if( len(df) >= self.period and not self.name in df.columns ):
                df[self.name] = df[self.source].rolling(window=self.period).apply(lambda x: self.func(x, self.period))
                self.initialized = True
            return self.initialized
        
        # check if this row has already been updated
        if( not pd.isna(df[self.name].iloc[-1]) ):
            return True
        
        # isolate only the required block of candles to calculate the current value of the custom series
        # Extract the last 'num_rows' rows of the specified column into a new DataFrame
        sdf = df[self.source].tail(self.period).to_frame(name=self.source)
        if( len(sdf) < self.period ):
            return False 
        
        newval = self.func( sdf[self.source], self.period )
        df.loc[df.index[-1], self.name] = newval
        return True
    
    def plotData( self ):
        return pd.DataFrame({'timestamp': df['timestamp'], self.name: df[self.name]}).dropna()
    
    def value( self, backindex = None, index = None ):
        if( backindex != None ):
            if( backindex < 0 ):
                raise KeyError( 'Invalid backindex. It must be 0 or more')
            return df[self.name].iloc[-(backindex + 1)]
        if( index != None ):
            if( index < 0 or index > len(df) ):
                raise KeyError( 'Invalid index. It must be 0 or more')
            return df[self.name].loc[index]
        return df[self.name].iloc[-1]


registeredCustomSeries = []

def calcCustomSeries( type:str, source:str, period:int, func ):
    name = f'{type} {source} {period}'
    cseries = None
    # find if there's a item already created for this series
    for thisCS in registeredCustomSeries:
        if thisCS.name == name:
            cseries = thisCS
            # print( 'found', name )
            break
    if cseries == None:
        cseries = customSeries_c( type, source, period, func )
        registeredCustomSeries.append(cseries)

    cseries.update()
    return cseries

def calcSMA( source:str, period:int ):
    return calcCustomSeries( 'sma', source, period, customseries_calculate_sma )

def calcEMA( source:str, period:int ):
    return calcCustomSeries( "ema", source, period, customseries_calculate_ema )

def calcRSI( source:str, period:int ):
    return calcCustomSeries( 'rsi', source, period, customseries_calculate_rsi )




class plot_c:
    def __init__( self, name:str, source:pd.DataFrame, chart = None ):
        self.name = name
        self.chart = chart
        self.line = None
        self.initialized = False

    def update( self, source:pd.DataFrame, chart = None ):
        if( self.chart == None ):
            if( chart != None ):
                self.chart = chart
            else:
                print( "Can't initialize a plot not associated with a chart")
                return
        
        if( not self.initialized ):
            self.line = chart.create_line( self.name, price_line=False, price_label=False )
            self.line.set( pd.DataFrame({'time': pd.to_datetime( source['timestamp'], unit='ms' ), self.name: source[self.name]}).dropna() )
            self.initialized = True
            return

        # it's initalized so only update the new line
        newval = source.iloc[-1][self.name]
        timestamp = int(source.iloc[-1]['timestamp'])
        self.line.update( pd.Series( {'time': pd.to_datetime( timestamp, unit='ms' ), 'value': newval } ) )



registeredPlots = []
def plot( name, source:pd.DataFrame, chart ):
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


coin = 'LDO'
timeframe = '1m'
exchangeName = 'bitget'

symbol = f'{coin}/USDT:USDT'
candles = candles_c( exchangeName, symbol )


def replaceValueByTimestamp( df, timestamp, key:str, value ):
    if( key == 'open' or key == 'high' or key == 'low' or key == 'close' ):
        df.loc[df['timestamp'] == timestamp, f'{key}'] = value



def runOpenCandle( chart ):

    return


def runCloseCandle( chart ):

    ###########################
    # strategy code goes here #
    ###########################

    sma = calcSMA( 'close', 90 )
    if( sma != None ):
        plot( sma.name, sma.plotData(), chart )

    ema = calcEMA( 'close', 4 )
    if( ema != None ):
        plot( ema.name, ema.plotData(), chart )

    # rsi = calcRSI( 'close', 14 )
    # if( rsi != None ):
    #     plot( rsi.name, rsi.plotData(), chart )

    # plotData = pd.DataFrame({'timestamp': df['timestamp'], 'low': df['low'] - 0.01}).dropna()
    # plot( "low", plotData, chart )

    return


def parseCandleUpdate( rows, chart = None ):
    for newrow in rows:
            newTimestamp = int(newrow[0])
            if( newTimestamp == None ):
                break
            
            oldTimestamp = int(df.iloc[-1]['timestamp'])
            if( oldTimestamp > newTimestamp ):
                continue

            if( oldTimestamp == newTimestamp ):
                #print( 'same timestamp', int(oldTimestamp), '=', newrow[0] )

                # update the realtime candle
                df.loc[df.index[-1], 'open'] = newrow[1]
                df.loc[df.index[-1], 'high'] = newrow[2]
                df.loc[df.index[-1], 'low'] = newrow[3]
                df.loc[df.index[-1], 'close'] = newrow[4]
                df.loc[df.index[-1], 'volume'] = newrow[5]

                #update the chart
                if( chart != None ):
                    data_dict = {'time': pd.to_datetime( newrow[0], unit='ms' ), 'open': newrow[1], 'high': newrow[2], 'low': newrow[3], 'close': newrow[4]}
                    if SHOW_VOLUME:
                        data_dict['volume'] = newrow[5]
                    chart.update( pd.Series(data_dict) )

            else:
                print( 'NEW CANDLE', newrow )

                # the realtime candle is now closed
                runCloseCandle( chart )

                # OPEN A NEW CANDLE
                new_row_index = len(df)
                df.loc[new_row_index, 'time'] = pd.to_datetime( newrow[0], unit='ms' )
                df.loc[new_row_index, 'timestamp'] = newTimestamp
                df.loc[new_row_index, 'open'] = newrow[1]
                df.loc[new_row_index, 'high'] = newrow[2]
                df.loc[new_row_index, 'low'] = newrow[3]
                df.loc[new_row_index, 'close'] = newrow[4]
                df.loc[new_row_index, 'volume'] = newrow[5]

                # update the chart
                if( chart != None ):
                    data_dict = {'time': pd.to_datetime( newrow[0], unit='ms' ), 'open': newrow[1], 'high': newrow[2], 'low': newrow[3], 'close': newrow[4]}
                    if SHOW_VOLUME:
                        data_dict['volume'] = newrow[5]
                    chart.update( pd.Series(data_dict) )

                    chart.legend( visible=True, ohlc=False, percent=False, font_size=18, text=symbol + ' - ' + timeframe + ' - ' + exchangeName + ' - ' + f'candles:{len(df)}' )

                runOpenCandle( chart )

# get new bars
async def fetchCandleUpdates(chart):
    # exchange = ccxt.bitget({
    exchange = getattr(ccxt, exchangeName)({
                "options": {'defaultType': 'swap', 'adjustForTimeDifference' : True},
                "enableRateLimit": False
                }) 

    subscriptions = f'{coin}/USDT:USDT'
    maxRows = 100
    while True:
        response = await exchange.watch_ohlcv( subscriptions, timeframe, limit = maxRows )
        #print(response)

        # extract the data

        if( len(response) > maxRows ):
            response = response[len(response)-maxRows:]

        parseCandleUpdate( response, chart )
        
        await asyncio.sleep(0.003)

    await exchange.close()


async def otherstuff():
    while True:
        # doing nothing yet
        await asyncio.sleep(1)

from datetime import datetime
async def update_clock(chart):
    while chart.is_alive:
        await asyncio.sleep(1-(datetime.now().microsecond/1_000_000))
        chart.legend( visible=True, ohlc=False, percent=False, font_size=18, text=symbol + ' - ' + timeframe + ' - ' + exchangeName + ' - ' + f'candles:{len(ohlcvs)}' + ' - ' + datetime.now().strftime('%H:%M:%S') )

async def runTasks( chart ):
    # Start the fetchCandleUpdates function
    task1 = asyncio.create_task( fetchCandleUpdates(chart) )
    
    # Start the monitorOtherUpdates function
    task2 = asyncio.create_task( otherstuff() )

    # clock
    task3 = asyncio.create_task( update_clock(chart) )

    # Run functions concurrently
    await asyncio.gather( task1, task2, task3 )

async def on_timeframe_selection(chart):
    print( f'Getting data with a {chart.topbar["my_switcher"].value} timeframe.' )

async def on_button_press(chart):
    new_button_value = 'On' if chart.topbar['my_button'].value == 'Off' else 'Off'
    chart.topbar['my_button'].set(new_button_value)
    print(f'Turned something {new_button_value.lower()}.')

def on_horizontal_line_move(chart, line):
    print(f'Horizontal line moved to: {line.price}')

if __name__ == '__main__':
    # filename = f'stuff/{exchangeName}-{coin}-USDT-{timeframe}.csv'
    # df = pd.read_csv( filename )
    # print( 'Loading', filename )

    # Define column names and meanings
    # column_meanings = {
    #     0: 'time',
    #     1: 'open',
    #     2: 'high',
    #     3: 'low',
    #     4: 'close',
    #     5: 'volume'
    # }
    
    # df.columns = column_meanings.values()

    ########################################################
    # Columns: time | open | high | high | close | volume
    
    ohlcvs = candles.fetchAmount( symbol, timeframe=timeframe, amount=1000 )

    chart = Chart( toolbox = False )
    chart.legend( visible=True, ohlc=False, percent=False, font_size=18, text=symbol + ' - ' + timeframe + ' - ' + exchangeName + ' - ' + f'candles:{len(ohlcvs)}' )
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


    df = pd.DataFrame( ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'] )

    # delete the last row in the dataframe and extract the last row in ohlcvs.
    df.drop( df.tail(1).index, inplace=True )
    last_ohlcv = [ohlcvs[-1]]
    
    tmpdf = pd.DataFrame( { 'time':pd.to_datetime( df['timestamp'], unit='ms' ), 'open':df['open'], 'high':df['high'], 'low':df['low'], 'close':df['close']} )
    if( SHOW_VOLUME ):
        tmpdf['volume'] = df['volume']
    
    chart.set(tmpdf)

    # jump-start the series and plots calculation by running the last row as if it was a update
    parseCandleUpdate( last_ohlcv, chart ) 
        
    #########################################################
    # print( df )

    chart.show( block=False )

    asyncio.run( runTasks(chart) )

    

