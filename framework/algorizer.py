import pandas as pd
import numpy as np
import asyncio
import ccxt.pro as ccxt
import time

from .constants import c
from . import tasks
from . import tools
from .fetcher import ohlcvs_c
from .candle import candle_c
from . import calcseries as calc
from .calcseries import generatedSeries_c # just for making lives easier
from .server import start_window_server, push_row_update, push_tick_update, push_marker_update, push_remove_marker_update
from . import active


verbose = False


class plot_c:
    def __init__( self, source, name:str = None, chart_name:str = None, color = "#8FA7BBAA", style = 'solid', width = 1, type = c.PLOT_LINE, hist_margin_top = 0.0, hist_margin_bottom = 0.0, screen_name:str= None ):
        '''name, color, style, width
        color: str = 'rgba(200, 200, 200, 0.6)',
        style: LINE_STYLE = 'solid', width: int = 2,
        
        LINE_STYLE = Literal['solid', 'dotted', 'dashed', 'large_dashed', 'sparse_dotted']
        '''
        self.name = name
        self.type = type
        self.chart_name = chart_name
        self.color = color if color.startswith('rgba') else tools.hx2rgba(color)
        self.style = style
        self.width = width
        self.hist_margin_top = hist_margin_top
        self.hist_margin_bottom = hist_margin_bottom
        self.screen_name = screen_name
        self._temp_values = [] 

        timeframe = active.timeframe # FIXME - ensure active.timeframe is set before plot_c init

        # These types will create a new column in the dataframe where to store values.
        # During __init__, if the source is a direct value (float/int), we prepare the column name
        # but defer filling it until batch processing or real-time updates.
        if source is None or isinstance( source, (float, int) ):
            if name:
                if name in timeframe.df.columns: 
                    raise ValueError( f"plot_c:name [{name}] is already in use" )
                # Names starting with an underscore are reserved for generated series
                # and must already exist in the dataframe
                if name.startswith('_') :
                    raise ValueError( f"plot_c:names starting with an underscore are reserved for generatedSeries_c objects" )
                self.name = name
                if not self.screen_name:
                    self.screen_name = self.name
                
                # We need to ensure the column exists when plot_c is initialized, 
                # especially since `parseCandleUpdate` expects it to be there for `iat_index`.
                # Initialize it with NaNs. The values will be filled later.
                if self.name not in timeframe.df.columns:
                    timeframe.df[self.name] = pd.Series(np.nan, index=timeframe.df.index, dtype=np.float64)


        elif isinstance( source, (pd.Series, generatedSeries_c) ):
            self.name = source.name
            # For Series or generatedSeries, the data is already managed by Pandas
            # or the generatedSeries_c object, so no need for _temp_values.

        if not self.name or self.name not in timeframe.df.columns:
            # This check ensures that either a new column was successfully prepared for a value source,
            # or an existing Series/generatedSeries was successfully referenced.
            raise ValueError( f"plot_c:Couldn't assign a name to the plot [{name}]" )


    def update( self, source, timeframe ):
        """
        Updates the plot's data. 
        - If `timeframe.jumpstart` is True, it does nothing (as plots shouldn't collect during this phase).
        - If `timeframe.backtesting` is True (historical backtesting), it appends values to a temporary list for bulk processing later.
        - Otherwise (real-time updates), it directly updates the DataFrame using .iat.
        """
        # If the source is already a Pandas Series or a generatedSeries_c,
        # its data is managed elsewhere (already in the DataFrame or by calcseries).
        # We don't need to do anything here for these types.
        if isinstance(source, (pd.Series, generatedSeries_c)):
            return 
        
        # If the source is a direct value (float, int, or None), we handle it.
        if isinstance(source, (int, float, type(None))) :
            # During the historical backtesting run (`backtesting=True`),
            # we append the value to a temporary list.
            if timeframe.backtesting:
                self._temp_values.append(source)
            else:
                # For real-time updates, directly assign to the DataFrame using .iat.
                # This is efficient enough for single row updates.
                timeframe.df[self.name].iat[timeframe.barindex] = source
            return

        # If we reach here, the source type is unexpected.
        raise ValueError( f"Unvalid plot type {self.name}: {type(source)}" )

    def _apply_batch_updates(self, timeframe_df):
        """
        Applies all collected temporary values to the DataFrame column in a single,
        efficient vectorized operation. Called after the initial historical data processing.
        """
        if not self._temp_values:
            return # Nothing to apply

        # Ensure the column exists before attempting to assign.
        # This redundant check is for robustness, as it should already be created in __init__.
        if self.name not in timeframe_df.columns:
            timeframe_df[self.name] = pd.Series(np.nan, index=timeframe_df.index, dtype=np.float64)

        # Assign the collected values.
        num_values = len(self._temp_values)
        
        # --- FIX: Explicitly convert to numpy array with float64 dtype ---
        values_to_assign = np.array(self._temp_values, dtype=np.float64)

        if num_values > len(timeframe_df):
            print(f"Warning: Plot '{self.name}' has more collected values ({num_values}) than DataFrame rows ({len(timeframe_df)}). Truncating values_to_assign.")
            # Truncate the values_to_assign if it's longer than the DataFrame slice
            values_to_assign = values_to_assign[:len(timeframe_df)]
            num_values = len(timeframe_df)
            
        timeframe_df.loc[timeframe_df.index[:num_values], self.name] = values_to_assign

        self._temp_values = []



class marker_c:
    def __init__( self, text:str, timestamp:int, position:str = 'below', shape:str = 'arrow_up', color:str = 'c7c7c7', chart_name:str = None ):
        # MARKER_POSITION = Literal['above', 'below', 'inside']
        # MARKER_SHAPE = Literal['arrow_up', 'arrow_down', 'circle', 'square']
        self.id = pd.Timestamp.now().value
        self.timestamp = timestamp
        self.text = text
        self.position = position
        self.shape = shape
        self.color = color
        self.panel = chart_name
        self.chart = None
        self.marker = None

    def descriptor(self):
        return {
                'id':self.id,
                'timestamp':self.timestamp,
                'position':self.position,
                'shape':self.shape,
                'color':self.color,
                'panel':self.panel,
                'text':self.text
            }


class timeframe_c:
    def __init__( self, stream, timeframeStr ):
        self.stream:stream_c = stream
        self.timeframeStr = tools.timeframeString( timeframeStr )
        self.timeframe = tools.timeframeInt(self.timeframeStr) # in minutes
        self.timeframeMsec = tools.timeframeMsec(self.timeframeStr)
        self.callback = None
        self.barindex = -1
        self.timestamp = 0
        self.backtesting = False
        self.jumpstart = False

        self.df:pd.DataFrame = []
        self.generatedSeries: dict[str, generatedSeries_c] = {}
        self.registeredPlots: dict[str, plot_c] = {}

        self.realtimeCandle:candle_c = candle_c()
        self.realtimeCandle.timeframemsec = self.timeframeMsec

 

    def initDataframe( self, ohlcvDF ):
        print( "=================" )
        print( f"Creating dataframe {self.timeframeStr}" )

        active.timeframe = self

        # --- Phase 1: Create a copy of the dataframe skipping the last row as the candle is not closed ---
        start_time = time.time()
        self.df = ohlcvDF.iloc[:-1].copy()

        # --- Phase 2: backtesting (row-by-row backtest simulation) ---
        if self.callback and not tools.emptyFunction( self.callback ):
            print( f"Computing script logic {self.timeframeStr}." )

            self.barindex = -1
            self.timestamp = self.df['timestamp'].iat[self.barindex]
            self.realtimeCandle.timestamp = self.timestamp
            self.realtimeCandle.open = self.df.iloc[0]['open']
            self.realtimeCandle.high = self.df.iloc[0]['high']
            self.realtimeCandle.low = self.df.iloc[0]['low']
            self.realtimeCandle.close = self.df.iloc[0]['close']
            self.realtimeCandle.volume = self.df.iloc[0]['volume']

            self.backtesting = True
            self.jumpstart = True
            self.parseCandleUpdate(self.df)
            self.backtesting = False
        else:
            print( f"No callback function defined or is empty. Skipping script logic {self.timeframeStr}." )

        # set the realtime candle to the row we skipped because it isn't yet closed
        row = ohlcvDF.iloc[-1]
        self.realtimeCandle.timestamp = row['timestamp']
        self.realtimeCandle.open = row['open']
        self.realtimeCandle.high = row['high']
        self.realtimeCandle.low = row['low']
        self.realtimeCandle.close = row['close']
        self.realtimeCandle.volume = row['volume']
        self.stream.timestampFetch = self.realtimeCandle.timestamp

        print( len(self.df), "candles processed." )

        # --- Phase 3: Apply batch updates for plots ---
        # This MUST happen after the backtesting loop is finished,
        # as all plot values for historical data would have been collected in _temp_values by now.
        if len(self.registeredPlots) :
            for plot_obj in self.registeredPlots.values():
                plot_obj._apply_batch_updates(self.df)
            print(f"Batch updates applied to plots {self.timeframeStr}.")
        ###############################################################################

        print( "Total time: {:.2f} seconds".format(time.time() - start_time))

        

    def parseCandleUpdate( self, rows ):

        active.timeframe = self

        for newrow in rows.itertuples( index=False, name=None ):
            newrow_timestamp = newrow[0]
            newrow_open = newrow[1]
            newrow_high = newrow[2]
            newrow_low = newrow[3]
            newrow_close = newrow[4]
            newrow_volume = newrow[5]

            # PROCESSING HISTORICAL DATA (either jumpstart or backtesting)
            if self.backtesting:
                # Increment barindex and timestamp for each historical row
                self.barindex += 1 
                active.barindex = self.barindex
                self.timestamp = int(newrow_timestamp)

                # Update realtimeCandle for the current historical bar being processed
                self.realtimeCandle.timestamp = newrow_timestamp
                self.realtimeCandle.open = newrow_open
                self.realtimeCandle.high = newrow_high
                self.realtimeCandle.low = newrow_low
                self.realtimeCandle.close = newrow_close
                self.realtimeCandle.volume = newrow_volume
                self.realtimeCandle.index = self.barindex + 1

                # Update stream's fetch timestamp if this is the smallest timeframe
                if self.timeframeStr == self.stream.timeframeFetch :
                    self.stream.timestampFetch = self.realtimeCandle.timestamp

                # Execute the user-defined callback for each historical candle.
                if( self.callback != None ):
                    self.callback( self, self.df['open'], self.df['high'], self.df['low'], self.df['close'], self.df['volume'] )

                # Print progress only during the main historical processing loop
                if self.barindex % 5000 == 0 and not self.jumpstart: 
                    print( self.barindex, "candles processed." )

                self.jumpstart = False
                continue # Move to the next row in the `rows` input
            

            # NOT BACKTESTING: This is realtime
            last_timestamp = int(self.df['timestamp'].iat[self.barindex]) 
            if( newrow_timestamp <= last_timestamp ):
                if verbose : print( f"SKIPPING {self.timeframeStr}: {int( newrow.timestamp)}")
                continue

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
                self.realtimeCandle.updateRemainingTime()

                if( self.timeframeStr == self.stream.timeframeFetch and self.stream.tick_callback != None ):
                    self.stream.tick_callback( self.realtimeCandle )

                if not self.stream.initializing:
                    push_tick_update( self )

                continue

            # NEW CANDLE - REAL-TIME
            # Append a new row to the DataFrame for the closed candle data
            # Use .loc with a new index (barindex + 1) to add the new row
            new_idx = self.barindex + 1
            self.df.loc[new_idx, 'timestamp'] = self.realtimeCandle.timestamp
            self.df.loc[new_idx, 'open'] = self.realtimeCandle.open
            self.df.loc[new_idx, 'high'] = self.realtimeCandle.high
            self.df.loc[new_idx, 'low'] = self.realtimeCandle.low
            self.df.loc[new_idx, 'close'] = self.realtimeCandle.close
            self.df.loc[new_idx, 'volume'] = self.realtimeCandle.volume
            self.df.loc[new_idx, 'top'] = max( self.realtimeCandle.open, self.realtimeCandle.close )
            self.df.loc[new_idx, 'bottom'] = min( self.realtimeCandle.open, self.realtimeCandle.close )

            # copy newrow into realtimeCandle for the NEXT incoming tick
            self.realtimeCandle.timestamp = newrow_timestamp
            self.realtimeCandle.open = newrow_open
            self.realtimeCandle.high = newrow_high
            self.realtimeCandle.low = newrow_low
            self.realtimeCandle.close = newrow_close
            self.realtimeCandle.volume = newrow_volume
            self.realtimeCandle.bottom = min( newrow_open, newrow_close )
            self.realtimeCandle.top = max( newrow_open, newrow_close )
            self.realtimeCandle.index = new_idx + 1
            self.realtimeCandle.updateRemainingTime()

            self.barindex = new_idx
            active.barindex = self.barindex
            self.timestamp = int(self.df['timestamp'].iat[self.barindex])
            if self.timeframeStr == self.stream.timeframeFetch :
                self.stream.timestampFetch = self.realtimeCandle.timestamp

            print( f"NEW CANDLE {self.timeframeStr} : {newrow}" )

            if( self.timeframeStr == self.stream.timeframeFetch and self.stream.tick_callback != None ):
                    self.stream.tick_callback( self.realtimeCandle )

            if( self.callback != None ):
                self.callback( self, self.df['open'], self.df['high'], self.df['low'], self.df['close'], self.df['volume'] )

            if not self.stream.initializing:
                push_row_update( self )



    def calcGeneratedSeries( self, type:str, source:pd.Series, period:int, func, param=None, always_reset:bool = False )->generatedSeries_c:
        name = tools.generatedSeriesNameFormat( type, source, period )

        gse = self.generatedSeries.get( name )
        if( gse == None ):
            gse = generatedSeries_c( type, source, period, func, param, always_reset, self )
            self.generatedSeries[name] = gse

        # If we are in the jumpstart phase, initialize and update the series immediately.
        # Otherwise, the update will be handled by generatedSeries_c.update
        # This part should remain as is for generatedSeries which are pre-calculated for speed.
        if self.jumpstart:
            gse.initialize( source ) # Full series calculation during jumpstart
        else:
            gse.update( source ) # Incremental update during live/backtesting
        return gse


    def register_plot( self, source, name:str = None, chart_name:str = None, color = "#8FA7BBAA", style = 'solid', width = 1, type = c.PLOT_LINE, hist_margin_top = 0.0, hist_margin_bottom = 0.0 )->plot_c:
        '''
        source: can either be a series, a generatedSeries or a value. A series can only be plotted when it is in the dataframe. When plotting a value a series will be automatically created in the dataframe.
        chart_name: Leave empty or use 'main' for the main panel. Use the name of a registered panel for plotting in the subpanel.
        color: in a string. Can be hexadecial '#DADADADA' or rgba format 'rgba(255,255,255,1.0)'
        style: plots LINE_STYLE = Literal['solid', 'dotted', 'dashed', 'large_dashed', 'sparse_dotted']
        width: with of the line. For plots.
        type: Is it a plot or a histogram
        hist_margin_top: for histograms only. Scalar margin above the histogram.
        hist_margin_bottom: for histograms only. Scalar margin below the histogram.
        '''
        plot = self.registeredPlots.get( name )

        if( plot == None ):
            plot = plot_c( source, name, chart_name, color, style, width, type, hist_margin_top, hist_margin_bottom )
            self.registeredPlots[name] = plot
        
        plot.update( source, self )
        return plot

    
    def plot( self, source, name:str = None, chart_name:str = None, color = "#8FA7BBAA", style = 'solid', width = 1 )->plot_c:
        '''
        source: can either be a series or a value. A series can only be plotted when it is in the dataframe. When plotting a value a series will be automatically created in the dataframe.
        chart_name: Leave empty for the main panel. Use 'panel' for plotting in the subpanel.
        color: in a string. Can be hexadecial '#DADADADA' or rgba format 'rgba(255,255,255,1.0)'
        style: LINE_STYLE = Literal['solid', 'dotted', 'dashed', 'large_dashed', 'sparse_dotted']
        width: int
        '''
        return self.register_plot( source, name, chart_name, color, style, width )
    
    def histogram( self, source, name:str = None, chart_name:str = None, color = "#8FA7BBAA", margin_top = 0.0, margin_bottom = 0.0 )->plot_c:
        '''
        source: can either be a series or a value. A series can only be plotted when it is in the dataframe. When plotting a value a series will be automatically created in the dataframe.
        chart_name: Leave empty for the main panel. Use 'panel' for plotting in the subpanel.
        color: in a string. Can be hexadecial '#DADADADA' or rgba format 'rgba(255,255,255,1.0)'
        hist_margin_top: for histograms only. Scalar margin above the histogram.
        hist_margin_bottom: for histograms only. Scalar margin below the histogram.
        '''
        return self.register_plot( source, name, chart_name, color, type = c.PLOT_HIST, hist_margin_top= margin_top, hist_margin_bottom= margin_bottom )
    
    def plotsList( self )->dict:
        di = {}
        for p in self.registeredPlots.values():
            plot = {
                'panel': p.chart_name,
                'type': p.type,
                'color': p.color,
                'style': p.style,
                'width': p.width,
                'margin_top': p.hist_margin_top,
                'margin_bottom': p.hist_margin_bottom
            }
            di[p.name] = plot
        return di

    def indexForTimestamp( self, timestamp:int )->int:
        # Estimate the index by dividing the offset by the time difference between rows
        baseTimestamp = int(self.df['timestamp'].iat[0])
        index = (timestamp - baseTimestamp) // self.timeframeMsec
        return max(-1, index - 1) # Return the previous index or -1 if not found


    def valueAtTimestamp( self, column_name, timestamp:int ):
        index = self.indexForTimestamp(timestamp)
        if index == -1:
            return None
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")
        return self.df[column_name].iat[index]

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
    def __init__( self, symbol, exchangeID:str, timeframeList, callbacks, event_callback = None, tick_callback = None, max_amount = 5000, cache_only = False ):
        self.symbol = symbol # FIXME: add verification
        self.initializing = True
        self.isRunning = False
        self.timeframeFetch = None
        self.timestampFetch = -1
        self.timeframes: dict[str, timeframe_c] = {}
        self.precision = 0.0
        self.mintick = 0.0
        self.cache_only = cache_only
        self.tick_callback = tick_callback
        self.event_callback = event_callback
        if event_callback == None:
            self.event_callback = globals().get('event')

        self.markers:list[marker_c] = []
        self.registeredPanels:dict = {}

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
        if self.cache_only:
            ohlcvs = fetcher.loadCache( self.symbol, self.timeframeFetch, max_amount * scale )
            print( "LOADING FROM CACHE")
        else:
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

            if i < len(callbacks):
                timeframe.callback = callbacks[i]
            else:
                func_name = f'runCloseCandle_{t}'
                timeframe.callback = globals().get(func_name)

            self.timeframes[t] = timeframe
            timeframe.initDataframe( candles )
            
            candles = []

        # we skipped the last row at initializing each timeframe
        # dataframe to parse as an update now becasse the last candle is not closed
        self.parseCandleUpdateMulti( ohlcvDF.iloc[-1:] )

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
 

    def run(self, backtest_only = False ):
        # We're done. Start fetching in real time
        self.isRunning = True
        tasks.registerTask( 'cli', cli_task, self )
        if not backtest_only and not self.cache_only : tasks.registerTask( 'fetch', self.fetchCandleUpdates )
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

        await self.exchange.close()


    def broker_event( self, order_type, quantity, quantity_dollars, position_type, position_size_base, position_size_dollars, position_collateral_dollars, leverage ):
        '''
        order_type (Buy/Sell Event): represented as the constants c.LONG (1) and c.SHORT (-1)
        quantity (Order Quantity in Base Currency): The exact amount of the base asset (e.g., 0.001 BTC).
        quantity_dollars (Order Quantity in Dollars): The notional value of the current order in USD (e.g., if you buy 0.001 BTC at $60,000, this would be $60).
        position_type (New Position Type: Long/Short/Flat)
        position_size_base (New Position Size in Base Currency): The total quantity of the base asset currently held (signed for long/short).
        position_size_dollars (New Position Size in Dollars, Leveraged): This represents the total notional exposure of the position, including the effect of leverage.
        leverage (Leverage of the Order)
        position_collateral_dollars (Un-leveraged Capital in Position)
        '''
        self.event_callback( self, "broker_event", (order_type, quantity, quantity_dollars, position_type, position_size_base, position_size_dollars, position_collateral_dollars, leverage), 8 )

    
    def registerPanel( self, name:str, width:float, height:float, fontsize = 14, show_candles:bool = False, show_timescale = True, show_volume = False, show_labels = False, show_priceline = False, show_plotnames = False ):
        """width and height are percentages of the window in 0/1 scale"""
        # to do: ensure the name isn't in use
        if name == None or not isinstance(name, str):
            raise ValueError( f"panels_c:registerPanel - name is not a valid string [{name}]" )
        
        for n in self.registeredPanels.keys():
            if n == name:
                raise ValueError( f"panels_c:registerPanel - name [{name}] is already registered " )
        panel = {
            "position": "bottom",
            "width": min(1.0, max(0.0, width)),
            "height": min(1.0, max(0.0, height)),
            "fontsize": fontsize,
            "show_candles": show_candles,
            "show_timescale": show_timescale,
            "show_labels": show_labels,
            "show_priceline": show_priceline,
            "show_plotnames": show_plotnames,
            "show_volume": show_volume
        }
        self.registeredPanels[name] = panel


    def createMarker( self, text:str = '', location:str = 'below', shape:str = 'circle', color:str = "#DEDEDE", timestamp:int = None, chart_name:str = None )->marker_c:
        '''MARKER_POSITION = Literal['above', 'below', 'inside']
        MARKER_SHAPE = Literal['arrow_up', 'arrow_down', 'circle', 'square']'''
        import bisect

        if timestamp == None:
            timestamp = self.timeframes[self.timeframeFetch].timestamp
        marker = marker_c( text, timestamp, location, shape, color, chart_name )

        if len(self.markers) and marker.timestamp < self.markers[-1].timestamp: # we need to insert back in time
            insertion_index = bisect.bisect_left( [m.timestamp for m in self.markers], marker.timestamp )
            self.markers.insert(insertion_index, marker)
        else:
            self.markers.append( marker )

        if not self.initializing:
            push_marker_update( marker )
        return marker
    
    def removeMarker( self, marker:marker_c ):
        if marker != None and isinstance(marker, marker_c):
            if not self.initializing:
                push_remove_marker_update( marker )
            self.markers.remove( marker )
    
    def getMarkersList( self )->list:
        di = []
        for m in self.markers:
            di.append( m.descriptor() )
        return di
    
    def createWindow(self, timeframeStr):
        """Create and show a window for the given timeframe"""
        # TODO: add validation of the timeframe
        start_window_server( timeframeStr )



def plot( source, name:str = None, chart_name:str = None, color = "#8FA7BBAA", style = 'solid', width = 1 )->plot_c:
    '''
    source: can either be a series or a value. A series can only be plotted when it is in the dataframe. When plotting a value a series will be automatically created in the dataframe.
    chart_name: Leave empty for the main panel. Use 'panel' for plotting in the subpanel.
    color: in a string. Can be hexadecial '#DADADADA' or rgba format 'rgba(255,255,255,1.0)'
    style: LINE_STYLE = Literal['solid', 'dotted', 'dashed', 'large_dashed', 'sparse_dotted']
    width: int
    '''
    return active.timeframe.plot( source, name, chart_name, color, style, width )


def histogram( source, name:str = None, chart_name:str = None, color = "#4A545D", margin_top = 0.0, margin_bottom = 0.0 )->plot_c:
        return active.timeframe.histogram( source, name, chart_name, color, margin_top, margin_bottom )

def createMarker( text, location:str = 'below', shape:str = 'circle', color:str = "#DEDEDE", timestamp:int = None, chart_name = None )->marker_c:
    '''MARKER_POSITION = Literal['above', 'below', 'inside']
        MARKER_SHAPE = Literal['arrow_up', 'arrow_down', 'circle', 'square']'''
    return active.timeframe.stream.createMarker( text, location, shape, color, timestamp, chart_name ) or None

def getRealtimeCandle()->candle_c:
    return active.timeframe.realtimeCandle

def getCandle( index = None )->candle_c:
    return active.timeframe.candle(index)

def getMintick()->float:
    return active.timeframe.stream.mintick

def getPrecision()->float:
    return active.timeframe.stream.precision

def getDataframe()->pd.DataFrame:
    # this is temporary for testing. ToDo: Add selection of the requested dataframe
    stream:stream_c = active.timeframe.stream
    return stream.timeframes[stream.timeframeFetch]

def requestValue( column_name:str, timeframeName:str = None, timestamp:int = None ):
    '''Request a value from the dataframe in any timeframe at given timestamp. If timestamp is not provided it will return the latest value'''
    if not timestamp : 
        timestamp = active.timeframe.timestamp
    if timeframeName not in active.timeframe.stream.timeframes.keys():
        raise ValueError(f"Timeframe {timeframeName} is not in the stream")
    targetTimeframe = active.timeframe.stream.timeframes[timeframeName] if timeframeName is not None else active.timeframe
    return targetTimeframe.valueAtTimestamp( column_name, timestamp )
    
def isInitializing():
    return active.timeframe.stream.initializing


import aioconsole
async def cli_task(stream: 'stream_c'): # Added type hint for clarity
    while True:
        message = await aioconsole.ainput()  # Non-blocking input

        # Split the message into command and arguments
        parts = message.split(' ', 1) # Split only on the first space
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else '' # Get args if they exist

        if command == 'chart':
            print('opening chart')
            timeframeName =  stream.timeframeFetch
            if( args and tools.validateTimeframeName( args ) ):
                timeframeName = args
            stream.createWindow(timeframeName)

        elif command == 'close':
            # TODO: Function to send a command to the client to shutdown
            print('closing chart')

        else:
            stream.event_callback(stream, "cli_command", (command, args), 2)

        await asyncio.sleep(0.05)




    



