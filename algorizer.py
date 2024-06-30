#lightweight-charts simple test

import pandas as pd
from lightweight_charts import Chart
import asyncio
import ccxt.pro as ccxt
import time
from pprint import pprint

from fetcher import candles_c



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
        
        self.dataFrame = pd.DataFrame({ 'time': df['time'], f'{self.name}': df[source].rolling(window=period).mean() }).dropna()
        self.initialized = True

    def update( self, df ):

        #if non existant try to create new
        if( not self.initialized ):
            if( len(df) >= self.period ):
                self.dataFrame = pd.DataFrame({ 'time': df['time'], f'{self.name}': df[self.source].rolling(window=self.period).mean() }).dropna()
                self.initialized = True
                return True
            return False
        
        #update from the existing one
        sdf = df.tail(self.period)
        if( len(sdf) < 1 ):
            return False 
        
        newval = sdf[self.source].rolling(window=self.period).mean().dropna()
        if( len(newval) < 1 ):
            return False
        newval = newval.iloc[-1]
        timestamp = df.iloc[-1]['timestamp']

        # Assign values to the new row one by one
        new_row_index = len(self.dataFrame)
        self.dataFrame.loc[new_row_index, 'time'] = pd.to_datetime( timestamp, unit='ms' )
        self.dataFrame.loc[new_row_index, 'value'] = newval

        return True
    
registeredSMAs = []
        
def calcSMA( df, source:str, period ):
    name = f'{source} {period}'
    sma = None
    # find if there's a SMA already created for this series
    for thisSMA in registeredSMAs:
        if thisSMA.name == name:
            sma = thisSMA
            break
    if sma == None:
        sma = sma = sma_c( source, period, df )
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
            self.line.set( source )
            self.initialized = True
            return

        # it's initalized so only update the new line
        if( len(self.line.data) < len(source) ):

            # FIXME: I don't think this is solid. It probably can leak.

            # I need to figure out FOR REAL how the series name thing works because it's driving me nuts...

            # newval = source.iloc[-1]['low'] # FIXME AND THIS CAN'T USE 'LOW' it needs to be adaptable
            # self.line.update( pd.Series( {'time': pd.to_datetime( timestamp, unit='ms' ), 'value': newval } ) )
            newval = source.iloc[-1][self.name] # FIXME AND THIS CAN'T USE 'LOW' it needs to be adaptable
            newtime = source.iloc[-1]['time']
            self.line.update( pd.Series( {'time': pd.to_datetime( newtime, unit='ms' ), 'value': newval } ) )


registeredPlots = []
def plot( name, source:pd.DataFrame, chart ):
    plot = None
    for thisPlot in registeredPlots:
        if( name == thisPlot.name ):
            plot = thisPlot
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

# def calculate_sma( df, period: int = 50):
#     return pd.DataFrame({
#         'time': df['time'],
#         f'SMA {period}': df['close'].rolling(window=period).mean()
#     }).dropna()


# def update_sma(sma_line, df, period=50):
#     if len(sma_line.data) < period:
#         # Not enough data points yet to calculate SMA
#         return
    
#     # extract only the candles we need for the SMA update
#     sdf = df.tail(period)
#     timestamp = df.iloc[-1]['timestamp']
    
#     newval = sdf['close'].rolling(window=period).mean().dropna()
#     newval = newval.iloc[-1]
#     data_dict = {'time': pd.to_datetime( timestamp, unit='ms' ), 'value': newval }
#     sma_line.update( pd.Series(data_dict) )


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

    plotData = pd.DataFrame({'time': df['time'], 'low': df['low'] - 0.01}).dropna()
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
                    data_dict = {'time': pd.to_datetime( newrow[0], unit='ms' ), 'open': newrow[1], 'high': newrow[2], 'low': newrow[3], 'close': newrow[4], 'volume': newrow[5]}
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
                    data_dict = {'time': pd.to_datetime( newrow[0], unit='ms' ), 'open': newrow[1], 'high': newrow[2], 'low': newrow[3], 'close': newrow[4], 'volume': newrow[5]}
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
        
        await asyncio.sleep(0.001)

    await exchange.close()


async def otherstuff():
    while True:
        # doing nothing yet
        await asyncio.sleep(1)

async def runTasks( df, chart ):
    # Start the fetchCandleUpdates function
    task1 = asyncio.create_task( fetchCandleUpdates(df, chart) )
    
    # Start the monitorOtherUpdates function
    task2 = asyncio.create_task( otherstuff() )

    # Run both functions concurrently
    await asyncio.gather( task1, task2 )


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

    #pprint( ohlcvs )

    if 0:
        initialohlcvs = ohlcvs[0:1]
        df = pd.DataFrame( initialohlcvs, columns=['time', 'open', 'high', 'low', 'close', 'volume'] )
        df['timestamp'] = df['time'] # create a new timestamp column and copy the timestamps
        df['time'] = pd.to_datetime( df['time'], unit='ms' ) # convert the original timestamps into 'datetime' format
        chunk_size = 10
        for i in range(0, len(ohlcvs), chunk_size):
            chunk = ohlcvs[i:i + chunk_size]
            print('chunk')
            print(chunk)
            parseCandleUpdate(df, chunk, chart = None)

        print(df)
        chart.set(df)
    else:
        df = pd.DataFrame( ohlcvs, columns=['time', 'open', 'high', 'low', 'close', 'volume'] )
        df['timestamp'] = df['time'] # create a new timestamp column and copy the timestamps
        df['time'] = pd.to_datetime( df['time'], unit='ms' ) # convert the original timestamps into 'datetime' format

        # delete the last row in the dataframe and extract the last row in ohlcvs.
        df.drop( df.tail(1).index, inplace=True )
        last_ohlcv = [ohlcvs[-1]]
        #print( last_ohlcv, 'time:', last_ohlcv[0][0], 'todatetime:', df.iloc[-1]['time']  )

        chart.set(df)
        parseCandleUpdate( df, last_ohlcv, chart ) # jump-start the series and plots calculation
        
    #########################################################
    # print( df )


    chart.show( block=False )
    #print( df )

    # for row_index in range(len(df)):
    #     # Create the subset DataFrame from the beginning to the current row
    #     print( row_index )
    #     subset_df = df.iloc[:row_index + 1]
        
    #     # Process the current row with its subset
    #     runCloseCandle( chart, subset_df )

    # get updates from the exchange
    #asyncio.run( fetchCandleUpdates(df, chart) )

    asyncio.run( runTasks(df, chart) )

    

