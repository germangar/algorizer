
import pandas as pd

import pandas_ta as pt
import asyncio
import ccxt.pro as ccxt
import time

from constants import c
import tasks
import tools
from fetcher import ohlcvs_c
from candle import candle_c
import calcseries as calc
from calcseries import generatedSeries_c # just for making lives easier

from window import window_c # I should try to get rid of this import

import active

# ALLOW_NEGATIVE_ILOC = False
SHOW_VOLUME = False
verbose = False


class plot_c:
    def __init__( self, name:str, chart_name = None, color = "#8FA7BBAA", style = 'solid', width = 1 ):
        '''name, color, style, width
        color: str = 'rgba(200, 200, 200, 0.6)',
        style: LINE_STYLE = 'solid', width: int = 2,
        
        LINE_STYLE = Literal['solid', 'dotted', 'dashed', 'large_dashed', 'sparse_dotted']
        '''
        self.name = name
        self.chartName = chart_name
        self.color = color if color.startswith('rgba') else tools.hx2rgba(color)
        self.style = style
        self.width = width
        self.line = None
        self.initialized = False
        self.iat_index = -1

        if name is None:
            raise SystemError( "ERROR: plot_c: Can't create a plot without a name" )

        # Names starting with an underscore are reserved for generated series
        # and must already exist in the dataframe
        if name.startswith('_') :
            if name not in active.timeframe.df.columns:
                raise SystemError( "ERROR: plot_c: Names starting with an underscore are reserved for generated series" )

    def update( self, source, timeframe ):

        if timeframe.stream.initializing and not timeframe.shadowcopy :
            # This is the call made on the last row to force pandas
            # to calculate all the generated series at once. We don't
            # want to do anything here.
            return
        
        if not self.initialized:
            # We want to store a reference to the series that is being plotted
            # to have it ready when the window is opened
            if( timeframe.window is None ):
                # if it's plotting a generated series reference its column in the dataframe
                if( isinstance(source, pd.Series) ):
                    # if self.name not in timeframe.df.columns:
                    #     raise SystemError( "ERROR: plot_c: Names starting with an underscore are reserved for generated series" )
                    # I don't support series which are not in the dataframe yet
                    return

                if not isinstance(source, (int, float, type(None))) :
                    return

                # Create a column in the dataframe for it if there's none, and keep updating it
                if self.iat_index == -1: # self.name not in timeframe.df.columns:
                    timeframe.df[self.name] = None
                    self.iat_index = timeframe.df.columns.get_loc(self.name)
                    
                # timeframe.df.at[timeframe.barindex, self.name] = source
                timeframe.df.iat[timeframe.barindex, self.iat_index] = source
                return
            

            # We've got a window,  INITIALIZE in the chart
            thisSeries = timeframe.df[self.name]
            if( len(thisSeries)<1 ):
                return
            
            chart = timeframe.window.bottomPanel if( self.chartName == 'panel' ) else timeframe.window.chart
            self.line = chart.create_line( self.name, self.color, self.style, self.width, price_line=False, price_label=False )
            self.line.set( pd.DataFrame( {'time': pd.to_datetime( timeframe.df['timestamp'], unit='ms' ), self.name: thisSeries} ) )
            self.iat_index = timeframe.df.columns.get_loc(self.name)
            self.initialized = True
            return


        # No window, but we are initialized. The window has been lost
        if( timeframe.window is None ):
            self.initialized = False
            return

        # if it's not a series update the value into the dataframe
        if( isinstance(source, float) or isinstance(source, int) ):
            if self.name not in timeframe.df.columns: 
                # This would never happen. Maybe I should just raise an error
                print( f'* WARNING plot_c: New plot [{self.name}] was created after initialization' )
                timeframe.df[self.name] = None
            timeframe.df.iat[timeframe.barindex, self.iat_index] = source

            ### fall through ###

        # I don't support series not in the dataframe yet
        if( isinstance(source, pd.Series) ):
            if self.name not in timeframe.df.columns:
                return

        # update the chart
        newval = timeframe.df.iat[timeframe.barindex, self.iat_index]
        self.line.update( pd.Series( {'time': pd.to_datetime( timeframe.timestamp, unit='ms' ), 'value': newval } ) )



def plot( source, name:str = None, chart_name:str = None, color = "#8FA7BBAA", style = 'solid', width = 1 )->plot_c:
    '''
    source: can either be a series or a value. A series can only be plotted when it is in the dataframe. When plotting a value a series will be automatically created in the dataframe.
    chart_name: Leave empty for the main panel. Use 'panel' for plotting in the subpanel.
    color: in a string. Can be hexadecial '#DADADADA' or rgba format 'rgba(255,255,255,1.0)'
    style: LINE_STYLE = Literal['solid', 'dotted', 'dashed', 'large_dashed', 'sparse_dotted']
    width: int
    '''
    return active.timeframe.plot( source, name, chart_name, color, style, width )
    


class marker_c:
    def __init__( self, text:str, timestamp:int, position:str = 'below', shape:str = 'arrow_up', color:str = 'c7c7c7', chart_name:str = None ):
        # MARKER_POSITION = Literal['above', 'below', 'inside']
        # MARKER_SHAPE = Literal['arrow_up', 'arrow_down', 'circle', 'square']
        self.timestamp = timestamp
        self.text = text
        self.position = position
        self.shape = shape
        self.color = color
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
            self.marker = self.chart.marker( time = pd.to_datetime( self.timestamp, unit='ms' ),
                                            position = self.position,
                                            shape = self.shape,
                                            color = self.color,
                                            text = self.text )



class timeframe_c:
    def __init__( self, stream, timeframeStr ):
        self.stream:stream_c = stream
        self.timeframeStr = tools.timeframeString( timeframeStr )
        self.timeframe = tools.timeframeInt(self.timeframeStr) # in minutes
        self.timeframeMsec = tools.timeframeMsec(self.timeframeStr)
        self.callback = None
        self.barindex = -1
        self.timestamp = 0
        self.shadowcopy = False
        self.jumpstart = False
        self.window = None

        self.df:pd.DataFrame = []
        self.initdata:pd.DataFrame = []
        self.generatedSeries: dict[str, generatedSeries_c] = {}
        self.registeredPlots: dict[str, plot_c] = {}

        self.realtimeCandle:candle_c = candle_c()
        self.realtimeCandle.timeframemsec = self.timeframeMsec

        
    def initDataframe( self, ohlcvDF ):
        print( "=================" )
        print( f"Creating dataframe {self.timeframeStr}" )

        active.timeframe = self

        # take out the last row to jumpstart the generatedSeries later
        self.df = ohlcvDF.loc[:len(ohlcvDF) - 2].copy() # self.df = ohlcvDF.iloc[:-1].copy()

        print( f"Calculating generated series {self.timeframeStr}" )

        # do the jump-starting with the last row of the dataframe
        start_time = time.time()

        self.barindex = active.barindex = self.df.iloc[-2].name
        self.timestamp = self.df.iloc[-2]['timestamp']
        self.realtimeCandle.timestamp = int(self.df.iloc[-1]['timestamp'])
        self.realtimeCandle.open = self.df.iloc[-1]['open']
        self.realtimeCandle.high = self.df.iloc[-1]['high']
        self.realtimeCandle.low = self.df.iloc[-1]['low']
        self.realtimeCandle.close = self.df.iloc[-1]['close']
        self.realtimeCandle.volume = self.df.iloc[-1]['volume']

        # set up to process the last row as a new update
        lastRowDF = ohlcvDF.loc[[len(ohlcvDF) - 1]]
        
        self.jumpstart = True
        self.parseCandleUpdate(lastRowDF) # self.parseCandleUpdate(ohlcvDF.iloc[[-1]])
        print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))
        self.jumpstart = False

        print( f"Computing script logic {self.timeframeStr}" )
        # at this point we have the generatedSeries initialized for the whole dataframe
        # move the dataframe to use it as source for running the script logic.
        # Start with a new dataframe with only the first row copied from the precomputed dataframe.
        # The precomputed data will be (shadow)copied into the new dataframe as we progress
        # through the bars.
        ###############################################################################

        # if there is no callback function we don't have anything to compute
        if self.callback is None:
            print( "No call back funtion defined. Skipping. Total time: {:.2f} seconds".format(time.time() - start_time))
            return
        
        if tools.emptyFunction( self.callback ):
            print( "Callback function is empty. Skipping. Total time: {:.2f} seconds".format(time.time() - start_time))
            return
        
    
        # run the script logic accross all the rows
        
        self.shadowcopy = True
        self.barindex = 0
        self.timestamp = 0
        self.realtimeCandle.timestamp = 0 # This forces parseCandleUpdate to reset realtimeCandle
        self.parseCandleUpdate(self.df)
        self.shadowcopy = False
        ###############################################################################

        print( len(self.df), "candles processed. Total time: {:.2f} seconds".format(time.time() - start_time))


    def parseCandleUpdate( self, rows ):

        active.timeframe = self

        for newrow in rows.itertuples( index=False, name=None ):
            newrow_timestamp = newrow[0]
            newrow_open = newrow[1]
            newrow_high = newrow[2]
            newrow_low = newrow[3]
            newrow_close = newrow[4]
            newrow_volume = newrow[5]

            if self.shadowcopy:
                if( self.barindex == 0 and self.timestamp == 0 ): # setup the first row
                    # self.timestamp = int(self.df.iloc[self.barindex]['timestamp'])
                    self.timestamp = int(self.df.iat[self.barindex, 0]) # trying to win performance in every corner

                if newrow_timestamp > self.timestamp :
                    self.barindex = self.df.iloc[self.barindex].name + 1
                    active.barindex = self.barindex
                    # self.timestamp = int(self.df.iloc[self.barindex]['timestamp'])
                    self.timestamp = int(self.df.iat[self.barindex, 0]) # trying to win performance in every corner

                    if( newrow_timestamp != self.timestamp ):
                        print( "** NEWROW TIMESTAMP != SELF TIMESTAMP")
                    
                    # copy the last row into the realtimeCandle row. This would be incorrect in realtime. Only for shadowcopying
                    self.realtimeCandle.timestamp = newrow_timestamp
                    self.realtimeCandle.open = newrow_open
                    self.realtimeCandle.high = newrow_high
                    self.realtimeCandle.low = newrow_low
                    self.realtimeCandle.close = newrow_close
                    self.realtimeCandle.volume = newrow_volume
                    self.realtimeCandle.index = self.barindex + 1
                    if self.timeframeStr == self.stream.timeframeFetch :
                        self.stream.timestampFetch = self.realtimeCandle.timestamp

                    if( self.callback != None ):
                        self.callback( self, self.df['open'], self.df['high'], self.df['low'], self.df['close'], self.df['volume'] )

                    if self.barindex % 5000 == 0:
                        print( self.barindex, "candles processed." )

                    continue

            # NOT SHADOWCOPY: This is realtime
            last_timestamp = int(self.df.iloc[self.barindex]['timestamp']) 
            if( newrow_timestamp < last_timestamp ):
                # print( f"SKIPPING {self.timeframeStr}: {int( newrow.timestamp)}")
                continue

            if( newrow_timestamp == last_timestamp ): # special case. We got an unfinished candle from the server
                if verbose : print( f"VERIFY CANDLE {self.timeframeStr}: {int( newrow_timestamp)}")
                continue

            if( self.realtimeCandle.timestamp == last_timestamp ): # FIX incorrect realtimeCandle
                self.realtimeCandle.timestamp = last_timestamp + self.timeframeMsec
                if verbose : print( f"FIXING {self.timeframeStr} REALTIME CANDLE TIMESTAMP")

            # has it reached a new candle yet?
            if newrow_timestamp < self.realtimeCandle.timestamp + self.timeframeMsec:
                # no. This is still a real time candle update
                if( self.timeframeStr == self.stream.timeframeFetch ):
                    self.realtimeCandle.timestamp = newrow_timestamp
                    self.realtimeCandle.open = newrow_open
                    self.realtimeCandle.high = newrow_high
                    self.realtimeCandle.low = newrow_low
                    self.realtimeCandle.close = newrow_close
                    self.realtimeCandle.volume = newrow_volume
                else:
                    # combine the smaller candles into a bigger one
                    fecthTF = self.stream.timeframes[self.stream.timeframeFetch]
                    fetch_rows = fecthTF.df[fecthTF.df['timestamp'] >= self.realtimeCandle.timestamp]
                    if( fetch_rows.empty ):
                        self.realtimeCandle.open = newrow_open
                        self.realtimeCandle.high = newrow_high
                        self.realtimeCandle.low = newrow_low
                        self.realtimeCandle.close = newrow_close
                        self.realtimeCandle.volume = newrow_volume
                    else:
                        self.realtimeCandle.open = fetch_rows['open'].iloc[0]
                        self.realtimeCandle.high = max(fetch_rows['high'].max(), newrow_high)
                        self.realtimeCandle.low = min(fetch_rows['low'].min(), newrow_low)
                        self.realtimeCandle.close = newrow_close
                        self.realtimeCandle.volume = newrow_volume + fetch_rows['volume'].sum() # add the volume of the smallest candles
                
                self.realtimeCandle.bottom = min( self.realtimeCandle.open, self.realtimeCandle.close )
                self.realtimeCandle.top = max( self.realtimeCandle.open, self.realtimeCandle.close )

                if( not self.stream.initializing and self.window != None ):
                    self.window.updateChart(self)

                continue

            # NEW CANDLE
            self.df.loc[self.barindex+1, 'timestamp'] = self.realtimeCandle.timestamp
            self.df.loc[self.barindex+1, 'open'] = self.realtimeCandle.open
            self.df.loc[self.barindex+1, 'high'] = self.realtimeCandle.high
            self.df.loc[self.barindex+1, 'low'] = self.realtimeCandle.low
            self.df.loc[self.barindex+1, 'close'] = self.realtimeCandle.close
            self.df.loc[self.barindex+1, 'volume'] = self.realtimeCandle.volume
            self.df.loc[self.barindex+1, 'top'] = max( self.realtimeCandle.open, self.realtimeCandle.close )
            self.df.loc[self.barindex+1, 'bottom'] = min( self.realtimeCandle.open, self.realtimeCandle.close )

            # copy newrow into realtimeCandle
            self.realtimeCandle.timestamp = newrow_timestamp
            self.realtimeCandle.open = newrow_open
            self.realtimeCandle.high = newrow_high
            self.realtimeCandle.low = newrow_low
            self.realtimeCandle.close = newrow_close
            self.realtimeCandle.volume = newrow_volume
            self.realtimeCandle.bottom = min( newrow_open, newrow_close )
            self.realtimeCandle.top = max( newrow_open, newrow_close )
            self.realtimeCandle.index = self.barindex + 2
            self.realtimeCandle.updateRemainingTime()

            self.barindex = self.barindex+ 1
            active.barindex = self.barindex
            self.timestamp = int(self.df.iloc[self.barindex]['timestamp'])
            if self.timeframeStr == self.stream.timeframeFetch :
                self.stream.timestampFetch = self.realtimeCandle.timestamp

            if not self.stream.initializing :
                print( f"NEW CANDLE {self.timeframeStr} : {int(self.df.iloc[self.barindex]['timestamp'])} DELTA: {self.realtimeCandle.timestamp - int(self.df.iloc[self.barindex]['timestamp'])}" )
                if( self.window != None ):
                    self.window.updateChart(self)

            if( self.callback != None ):
                self.callback( self, self.df['open'], self.df['high'], self.df['low'], self.df['close'], self.df['volume'] )


    def calcGeneratedSeries( self, type:str, source:pd.Series, period:int, func, param=None, always_reset:bool = False )->generatedSeries_c:
        name = tools.generatedSeriesNameFormat( type, source, period )

        gse = self.generatedSeries.get( name )
        if( gse == None ):
            gse = generatedSeries_c( type, source, period, func, param, always_reset, self )
            self.generatedSeries[name] = gse

        gse.update( source )
        return gse


    def plot( self, source, name:str = None, chart_name:str = None, color = "#8FA7BBAA", style = 'solid', width = 1 )->plot_c:
        '''
        source: can either be a series or a value. A series can only be plotted when it is in the dataframe. When plotting a value a series will be automatically created in the dataframe.
        chart_name: Leave empty for the main panel. Use 'panel' for plotting in the subpanel.
        color: in a string. Can be hexadecial '#DADADADA' or rgba format 'rgba(255,255,255,1.0)'
        style: LINE_STYLE = Literal['solid', 'dotted', 'dashed', 'large_dashed', 'sparse_dotted']
        width: int
        '''
        if self.stream.noplots : return
        plot = self.registeredPlots.get( name )

        if( plot == None ):
            plot = plot_c( name, chart_name, color, style, width )
            self.registeredPlots[name] = plot
        
        plot.update( source, self )
        return plot
    

    def jumpstartPlots( self ):
        for plot in self.registeredPlots.values():
            if not plot.initialized:
                plot.update( None, self )


    def indexForTimestamp( self, timestamp:int )->int:
        # Estimate the index by dividing the offset by the time difference between rows
        baseTimestamp = int(self.df['timestamp'].iloc[0])
        index = (timestamp - baseTimestamp) // self.timeframeMsec
        return max(-1, index - 1) # Return the previous index or -1 if not found


    def valueAtTimestamp( self, column_name, timestamp:int ):
        index = self.indexForTimestamp(timestamp)
        if index == -1:
            return None
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")
        return self.df[column_name].iloc[index]

    def valueByColumnIdx( self, column_index:int, index:int = None ):
        if index == None: index = self.barindex
        return self.df.iat[index, column_index]

    def candle( self, index = None )->candle_c:
        if( index is None ):
            index = self.barindex
        if( index > self.barindex ):
            raise SystemError( f"timeframe_c::candle index out of bounds {index}" )
        if( index < 0 ):
            index = self.barindex + ( index + 1 ) # iloc style
        
        candle = candle_c()
        candle.index = index
        candle.timeframemsec = self.timeframeMsec
        row = self.df.iloc[index].values  # Get as a NumPy array
        candle.timestamp, candle.open, candle.high, candle.low, candle.close, candle.volume, candle.top, candle.bottom = ( int(row[0]), row[1], row[2], row[3], row[4], row[5], row[6], row[7] )
        return candle




class stream_c:
    def __init__( self, symbol, exchangeID:str, timeframeList, callbackList, max_amount = 5000, noplots:bool = False ):
        self.symbol = symbol # FIXME: add verification
        self.initializing = True
        self.isRunning = False
        self.noplots = noplots
        self.timeframeFetch = None
        self.timestampFetch = -1
        self.timeframes: dict[str, timeframe_c] = {}
        self.precision = 0.0
        self.mintick = 0.0

        self.markers:marker_c = []

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

        fetcher = ohlcvs_c( exchangeID, self.symbol )

        self.markets = fetcher.getMarkets()
        self.precision = fetcher.getPrecision()
        self.mintick = fetcher.getMintick()
            
        # fetch OHLCVs
        ohlcvs = fetcher.loadCacheAndFetchUpdate( self.symbol, self.timeframeFetch, max_amount * scale )
        if( len(ohlcvs) == 0 ):
            raise SystemExit( f'No candles available in {exchangeID}. Aborting')
        ohlcvDF = pd.DataFrame( ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'] )
        ohlcvs = []


        #################################################
        # Create the timeframe sets with their dataframes
        #################################################

        for i, t in enumerate(timeframeList):
            if t == self.timeframeFetch:
                candles = ohlcvDF
            else:
                candles = tools.resample_ohlcv( ohlcvDF, t )

            # create top and bottom columns
            candles['top'] = candles[['open', 'close']].max(axis=1)
            candles['bottom'] = candles[['open', 'close']].min(axis=1)

            timeframe = timeframe_c( self, t )

            if i < len(callbackList):
                timeframe.callback = callbackList[i]
            else:
                func_name = f'runCloseCandle_{t}'
                timeframe.callback = globals().get(func_name)

            self.timeframes[t] = timeframe
            timeframe.initDataframe( candles )
            
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
        # tasks.registerTask( 'cli', cli_task(self) )
        # tasks.registerTask( 'fetch', self.fetchCandleUpdates() )
        # tasks.registerTask( 'clocks', update_clocks(self) )

    def run(self):
        # We're done. Start fetching
        self.isRunning = True
        tasks.registerTask( 'cli', cli_task(self) )
        tasks.registerTask( 'fetch', self.fetchCandleUpdates() )
        tasks.registerTask( 'clocks', update_clocks(self) )
        asyncio.run( tasks.runTasks() )

    def parseCandleUpdateMulti( self, rows ):
        for timeframe in self.timeframes.values():
            timeframe.parseCandleUpdate(rows)

    async def fetchCandleUpdates( self ):
        maxRows = 20
        while True:
            try:
                response = await self.exchange.watch_ohlcv( self.symbol, self.timeframeFetch, limit = maxRows )
                #print(response)

            except Exception as e:
                print( 'Exception raised at fetchCandleupdates: Reconnecting', e, type(e) )
                await self.exchange.close()
                await asyncio.sleep(1.0)
                continue
                
            # extract the data

            if( len(response) > maxRows ):
                response = response[len(response)-maxRows:]

            #pprint( response )
            if( len(response) ):
                self.parseCandleUpdateMulti( pd.DataFrame( response, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'] ) )
            
            await asyncio.sleep(0.01)

        await exchange.close()


    def createMarker( self, text:str = '', location:str = 'below', shape:str = 'circle', color:str = "#DEDEDE", timestamp:int = None, chart_name:str = None )->marker_c:
        '''MARKER_POSITION = Literal['above', 'below', 'inside']
        MARKER_SHAPE = Literal['arrow_up', 'arrow_down', 'circle', 'square']'''
        if timestamp == None:
            timestamp = self.timeframes[self.timeframeFetch].timestamp
        marker = marker_c( text, timestamp, location, shape, color, chart_name )
        self.markers.append( marker )
        return marker


    def createWindow( self, timeframeStr ):
        # FIXME: Add proper checks
        timeframe = self.timeframes[tools.timeframeString( timeframeStr )]
        timeframe.window = window_c( timeframe )
        timeframe.jumpstartPlots()


def getRealtimeCandle()->candle_c:
    return active.timeframe.realtimeCandle

def getCandle( index = None )->candle_c:
    return active.timeframe.candle(index)

def requestValue( column_name:str, timeframeName:str = None, timestamp:int = None ):
    '''Request a value from the dataframe in any timeframe at given timestamp. If timestamp is not provided it will return the latest value'''
    if not timestamp : 
        timestamp = active.timeframe.timestamp

    targetTimeframe = active.timeframe.stream.timeframes[timeframeName] if timeframeName is not None else active.timeframe
    return targetTimeframe.valueAtTimestamp( column_name, timestamp )
    

def createMarker( text, location:str = 'below', shape:str = 'circle', color:str = "#DEDEDE", timestamp:int = None, chart_name = None )->marker_c:
    '''MARKER_POSITION = Literal['above', 'below', 'inside']
        MARKER_SHAPE = Literal['arrow_up', 'arrow_down', 'circle', 'square']'''
    if( active.timeframe == None ):
        return None
    return active.timeframe.stream.createMarker( text, location, shape, color, timestamp, chart_name )

def isInitializing():
    return active.timeframe.stream.initializing


async def update_clocks( stream:stream_c ):
    from datetime import datetime

    while True:
        await asyncio.sleep(1-(datetime.now().microsecond/1_000_000))

        for timeframe in stream.timeframes.values():
            timeframe.realtimeCandle.updateRemainingTime()
            if timeframe.window != None:
                timeframe.window.updateClock()



import aioconsole
async def cli_task(stream):
    while True:
        command = await aioconsole.ainput()  # <-- Use aioconsole for non-blocking input

        if command.lower() == 'chart':
            print( 'opening chart' )
            stream.timeframes[stream.timeframeFetch].window = stream.createWindow( stream.timeframeFetch )

        if command.lower() == 'close':
            print( 'closing chart' )
            if stream.timeframes[stream.timeframeFetch].window and stream.timeframes[stream.timeframeFetch].window.chart:
                stream.timeframes[stream.timeframeFetch].window.chart.exit()
                stream.timeframes[stream.timeframeFetch].window = None
                
        
        await asyncio.sleep(0.05)


if __name__ == '__main__':

    def runCloseCandle_1d( timeframe:timeframe_c, open:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series, volume:pd.Series ):
        pass
    def runCloseCandle_5m( timeframe:timeframe_c, open:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series, volume:pd.Series ):
        sma = calc.SMA( close, 350 )
        sma.plot()
        rsi = calc.RSI( close, 14 )
        rsiplot = plot( rsi.series(), rsi.name, 'panel' )
        return

    def runCloseCandle_15m( timeframe:timeframe_c, open:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series, volume:pd.Series ):
        return

    def runCloseCandle_1m( timeframe:timeframe_c, open:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series, volume:pd.Series ):

        ###########################
        # strategy code goes here #
        ###########################
        sma = calc.SMA( close, 75 )
        sma.plot()

        # ema = calcEMA( close, 4 )
        # ema.plot()

        lr = calc.LINREG( close, 300 )
        lr.plot()

        # plot( 0, "lazyline", 'panel' )

        # rsi = calcRSI( close, 14 )
        # rsiplot = plot( rsi.series(), rsi.name, 'panel' )
        
        # FIXME: It crashes when calling to plot the same series
        atr = calc.ATR( 14 )
        plot( atr.series(), atr.name, 'panel' )

        # calcTR(14).plot('panel')

        # cfo = calcCFO( close, 20 )
        # cfo.plot('panel')

        # dev = calcDEV( close, 30 )
        # plot( dev.series(), dev.name, 'panel' )

        # rma = calcRMA( close, 90 )
        # rma.plot()

        # stdev = calcSTDEV( close, 350 )

        # willr = calcWPR( close, 32 ).plot('panel')
        # calcBIAS( close, 32 ).plot('panel')

        # hma = calcHMA( close, 150 )
        # hma.plot()
        # r = rising( hma.series(), 10 )
        # f = falling( hma.series(), 10 )

        # calcBarsSince( timeframe.barindex, r )
        # wt = calcBarsWhileTrue( timeframe.barindex, hma.series() > 1.7 )

        # calcCCI( 100 ).plot('panel')

        # slope1000 = calcSMA( calcSLOPE( close, 200 ).series() * 500000, 14 )
        # plot( slope1000.series(), slope1000.name, 'panel' )

        # hma_rising = rising( hma.series(), 30 )
        # if( hma_rising.value() and not hma_rising.value(1) ):
        #     timeframe.stream.createMarker( '🔼' )

        # hma_falling = falling( hma.series(), 30 )
        # if( hma_falling.value() and not hma_falling.value(1) ):
        #     timeframe.stream.createMarker( '🔽' )


    stream = stream_c( 'LDO/USDT:USDT', 'bitget', ['1m'], [runCloseCandle_1m], 5000 )

    # strategy.print_strategy_stats()

    stream.createWindow( '1m' )

    stream.run()


    



