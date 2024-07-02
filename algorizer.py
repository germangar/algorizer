#lightweight-charts simple test

import pandas as pd
from lightweight_charts import Chart
import asyncio
import ccxt.pro as ccxt
import time
from pprint import pprint

import tools
from fetcher import candles_c


SHOW_VOLUME = False



class context_c:
    def __init__( self, symbol, exchangeID:str, timeframe, dataFrame ):
        self.symbol = symbol # FIXME: add verification
        self.exchangeID = exchangeID
        #self.exchange = LOAD THE EXCHANGE
        self.timeframe = timeframe if( type(timeframe) == int ) else tools.timeframeInt(timeframe)
        self.timeframeName = tools.timeframeString( self.timeframe )
        self.candles = dataFrame
        self.timestamp = 0

    # load dataframe from cache

    # update dataframe from 
    
    def newCandle():
        return
    def updateRealtimeCandle():
        return
    


class sma_c:
    def __init__( self, source:str, period, df ):
        self.period = period
        self.source = source
        self.name = f'{source} {period}'
        self.initialized = False
        self.dataFrame = None

        if( self.period < 1 ):
            SystemError( f"SMA with invalid period [{period}]")

        if( len(df) < period ):
            return
        
        # self.dataFrame = pd.DataFrame({ 'timestamp': df['timestamp'], self.name : df[source].rolling(window=period).mean() }).dropna()
        # self.initialized = True

    def update( self, df ):

        #if non existant try to create new
        if( not self.initialized ):
            if( len(df) >= self.period ):
                self.dataFrame = pd.DataFrame({ 'timestamp': df['timestamp'], self.name : df[self.source].rolling(window=self.period).mean() }).dropna()
                self.initialized = True
                return True
            return False
        
        #update from the existing one
        sdf = df.tail(self.period)
        if( len(sdf) < 1 ):
            return False 
        
        # isolate only the required block of candles to calculate the current value of the SMA
        newval = sdf[self.source].rolling(window=self.period).mean().dropna()
        if( len(newval) < 1 ):
            return False
        newval = newval.iloc[-1]
        timestamp = int( df.iloc[-1]['timestamp'] )
        self.dataFrame = tools.df_append( self.dataFrame, {'timestamp': timestamp, self.name : newval} )
        # new_row = { 'timestamp': timestamp, self.name : newval }
        # self.dataFrame = pd.concat( [self.dataFrame, pd.DataFrame(new_row, index=[0])], ignore_index=False )

        return True
    
registeredSMAs = []
        
def calcSMA( df, source:str, period ):
    name = f'{source} {period}'
    sma = None
    # find if there's a SMA already created for this series
    for thisSMA in registeredSMAs:
        if thisSMA.name == name:
            sma = thisSMA
            #print( 'found SMA')
            break
    if sma == None:
        sma = sma_c( source, period, df )
        registeredSMAs.append(sma)

    sma.update( df )
    return sma


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
            self.line = chart.create_line( self.name, price_label=False )
            #source['time'] = pd.to_datetime( source['timestamp'], unit='ms' )
            self.line.set( pd.DataFrame({'time': pd.to_datetime( source['timestamp'], unit='ms' ), self.name: source[self.name]}).dropna() )
            #source.drop('time', axis=1, inplace=True)
            self.initialized = True
            return

        # it's initalized so only update the new line
        # if( 1 or len(self.line.data) < len(source) ):
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



def runOpenCandle( chart, df ):

    return


def runCloseCandle( chart, df ):

    ###########################
    # strategy code goes here #
    ###########################

    sma = calcSMA( df, 'close', 30 )
    if( sma != None ):
        plot( sma.name, sma.dataFrame, chart )

    sma = calcSMA( df, 'close', 90 )
    if( sma != None ):
        plot( sma.name, sma.dataFrame, chart )

    plotData = pd.DataFrame({'timestamp': df['timestamp'], 'low': df['low'] - 0.01}).dropna()
    plot( "low", plotData, chart )

    return


def parseCandleUpdate(df, rows, chart = None):
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
                runCloseCandle( chart, df )

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

                #print("\nDataFrame after filling the new row:")
                #print(df.tail())

                runOpenCandle( chart, df )

# get new bars
async def fetchCandleUpdates(df, chart):
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

        parseCandleUpdate(df, response, chart)
        
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

async def runTasks( df, chart ):
    # Start the fetchCandleUpdates function
    task1 = asyncio.create_task( fetchCandleUpdates(df, chart) )
    
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
    parseCandleUpdate( df, last_ohlcv, chart ) 
        
    #########################################################
    # print( df )

    chart.show( block=False )

    asyncio.run( runTasks(df, chart) )

    

