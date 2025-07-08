import numpy as np
from typing import Optional
import asyncio
import ccxt.pro as ccxt
import time

from .constants import c
from . import tasks
from . import tools
from .fetcher import ohlcvs_c
from .candle import candle_c
from .nameseries import series_c
from .calcseries import generatedSeries_c # just for making lives easier
from .server import start_window_server, push_row_update, push_tick_update
from . import active


verbose = False


class plot_c:
    def __init__( self, source:float|int|series_c|generatedSeries_c, name:str = None, chart_name:str = None, color = "#8FA7BBAA", style = 'solid', width = 1, type = c.PLOT_LINE, hist_margin_top = 0.0, hist_margin_bottom = 0.0, screen_name:str= None ):
        '''name, color, style, width
        color: str = 'rgba(200, 200, 200, 0.6)',
        style: LINE_STYLE = 'solid', width: int = 2,
        
        LINE_STYLE = Literal['solid', 'dotted', 'dashed', 'large_dashed', 'sparse_dotted']
        '''
        self.name = name
        self.type = type
        self.chart_name = chart_name
        self.column_index = -1
        self.color = color if color.startswith('rgba') else tools.hx2rgba(color)
        self.style = style
        self.width = width
        self.hist_margin_top = hist_margin_top
        self.hist_margin_bottom = hist_margin_bottom
        self.screen_name = screen_name

        timeframe = active.timeframe

        if source is None or isinstance(source, (float, int)):
            if name:
                if name in timeframe.registeredSeries.keys():
                    raise ValueError(f"plot_c:name [{name}] is already in use")
                if name.startswith('_'):
                    raise ValueError(f"plot_c:names starting with an underscore are reserved for generatedSeries_c objects")
                self.name = name
                if not self.screen_name:
                    self.screen_name = self.name

                series = timeframe.createColumnSeries( self.name, True ) # it creates a series_c with its name and stores it in timeframe.registeredSeries
                self.column_index = series.index

        elif isinstance( source, (generatedSeries_c, series_c) ):
            self.name = source.name

        if not self.name or self.name not in timeframe.registeredSeries.keys():
            raise ValueError(f"plot_c:Couldn't assign a name to the plot [{name}]")

    def update(self, source, timeframe):
        if source is None or isinstance(source, (float, int)):
            timeframe.dataset[timeframe.barindex, self.column_index] = np.nan if source is None else float(source)


class marker_c:
    def __init__( self, text:str, timestamp:int, position:str = 'below', shape:str = 'arrow_up', color:str = '#c7c7c7', chart_name:str = None ):
        # MARKER_POSITION = Literal['above', 'below', 'inside']
        # MARKER_SHAPE = Literal['arrow_up', 'arrow_down', 'circle', 'square']
        self.id = id(self)
        self.timestamp = timestamp
        self.text = text
        self.position = position
        self.shape = shape
        self.color = color
        self.panel = chart_name

    def remove(self):
        active.timeframe.stream.removeMarker(self)

    def descriptor(self):
        return vars(self)
    
class line_c:
    def __init__( self, x1, y1, x2, y2, color:str = '#c7c7c7', width = 1, style = 'solid', chart_name:str = 'main' ):
        if isinstance( x1, (generatedSeries_c, series_c, np.ndarray ) ):
            x1 = x1[active.timeframe.barindex]
        if isinstance( y1, (generatedSeries_c, series_c, np.ndarray ) ):
            y1 = y1[active.timeframe.barindex]
        if isinstance( x2, (generatedSeries_c, series_c, np.ndarray ) ):
            x2 = x2[active.timeframe.barindex]
        if isinstance( y2, (generatedSeries_c, series_c, np.ndarray ) ):
            y2 = y2[active.timeframe.barindex]
        self.id = id(self)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.color = color
        self.width = width
        self.style = style
        self.panel = chart_name
        self.timeframe:timeframe_c = active.timeframe

    def timestamp(self, index ):
        timeframe = self.timeframe
        '''Line coordinates are a special case where we allow the indexes to be bigger than the dataset'''
        if( index > timeframe.barindex ):
            return timeframe.timestamp + (( index - timeframe.barindex ) * timeframe.timeframeMsec)
        elif index < 0:
            return timeframe.dataset[0, c.DF_TIMESTAMP] + (index * timeframe.timeframeMsec)

        return timeframe.dataset[index, c.DF_TIMESTAMP]

    def remove(self):
        active.timeframe.stream.removeLine(self)

    def descriptor(self):
        descriptor = {
            'id':self.id,
            'x1':self.timestamp(self.x1),
            'y1':self.y1,
            'x2':self.timestamp(self.x2),
            'y2':self.y2,
            'color':self.color,
            'width':self.width,
            'style':self.style,
            'panel':self.panel,
        }
        return descriptor
        

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

        self.dataset:Optional[np.NDArray[np.float64]] = None
        self.generatedSeries: dict[str, generatedSeries_c] = {}
        self.registeredPlots: dict[str, plot_c] = {}
        self.registeredSeries: dict[str, series_c] = {}

        self.realtimeCandle:candle_c = candle_c()
        self.realtimeCandle.timeframemsec = self.timeframeMsec

 
    def initDataframe( self, ohlcvNP ):
        print( "=================" )
        print( f"Creating dataframe {self.timeframeStr}" )

        active.timeframe = self

        # --- Phase 1: Create a copy of the dataframe skipping the last row as the candle is not closed ---
        start_time = time.time()
        self.dataset = ohlcvNP[:-1, :].copy()

        # create series_c objects representing the columns
        self.registeredSeries['timestamp'] = series_c( self.dataset[:, c.DF_TIMESTAMP], 'timestamp', False )
        self.registeredSeries['open'] = series_c( self.dataset[:, c.DF_OPEN], 'open', False )
        self.registeredSeries['high'] = series_c( self.dataset[:, c.DF_HIGH], 'high', False )
        self.registeredSeries['low'] = series_c( self.dataset[:, c.DF_LOW], 'low', False )
        self.registeredSeries['close'] = series_c( self.dataset[:, c.DF_CLOSE], 'close', False )
        self.registeredSeries['volume'] = series_c( self.dataset[:, c.DF_VOLUME], 'volume', False )
        self.registeredSeries['top'] = series_c( self.dataset[:, c.DF_TOP], 'top', False )
        self.registeredSeries['bottom'] = series_c( self.dataset[:, c.DF_BOTTOM], 'bottom', False )
        

        # --- Phase 2: backtesting (row-by-row backtest simulation) ---
        if self.callback and not tools.emptyFunction( self.callback ):
            print( f"Computing script logic {self.timeframeStr}." )

            self.barindex = -1
            self.timestamp = int(self.dataset[0, c.DF_TIMESTAMP]) - self.timeframeMsec
            self.realtimeCandle.timestamp = self.timestamp
            self.realtimeCandle.open = self.dataset[0, c.DF_OPEN]
            self.realtimeCandle.high = self.dataset[0, c.DF_HIGH]
            self.realtimeCandle.low = self.dataset[0, c.DF_LOW]
            self.realtimeCandle.close = self.dataset[0, c.DF_CLOSE]
            self.realtimeCandle.volume = self.dataset[0, c.DF_VOLUME]
            self.realtimeCandle.top = self.dataset[0, c.DF_TOP]
            self.realtimeCandle.bottom = self.dataset[0, c.DF_BOTTOM]

            self.backtesting = True
            self.jumpstart = True
            self.parseCandleUpdate(self.dataset)
            self.backtesting = False
        else:
            print( f"No callback function defined or is empty. Skipping script logic {self.timeframeStr}." )

        # set the realtime candle to the row we skipped because it isn't yet closed
        row = ohlcvNP[-1]
        self.realtimeCandle.timestamp = row[c.DF_TIMESTAMP]
        self.realtimeCandle.open = row[c.DF_OPEN]
        self.realtimeCandle.high = row[c.DF_HIGH]
        self.realtimeCandle.low = row[c.DF_LOW]
        self.realtimeCandle.close = row[c.DF_CLOSE]
        self.realtimeCandle.volume = row[c.DF_VOLUME]
        self.realtimeCandle.top = max(self.realtimeCandle.open, self.realtimeCandle.close)
        self.realtimeCandle.bottom = min(self.realtimeCandle.open, self.realtimeCandle.close)
        self.stream.timestampFetch = self.realtimeCandle.timestamp

        print( len(self.dataset), "candles processed. Total time: {:.2f} seconds".format(time.time() - start_time))


    def parseCandleUpdate( self, rows ): # rows is a 2D numpy array now

        active.timeframe = self
        is_fetch = self.timeframeStr == self.stream.timeframeFetch

        for newrow in rows:
            newrow_timestamp = int( newrow[0] )
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
                self.timestamp = newrow_timestamp

                # Update realtimeCandle for the current historical bar being processed
                self.realtimeCandle.timestamp = newrow_timestamp
                self.realtimeCandle.open = newrow_open
                self.realtimeCandle.high = newrow_high
                self.realtimeCandle.low = newrow_low
                self.realtimeCandle.close = newrow_close
                self.realtimeCandle.volume = newrow_volume
                self.realtimeCandle.index = self.barindex + 1

                # Update stream's fetch timestamp if this is the smallest timeframe
                if is_fetch:
                    self.stream.timestampFetch = self.realtimeCandle.timestamp

                # Execute the user-defined callback for each historical candle.
                if( self.callback != None ):
                    self.callback( self, self.registeredSeries['open'], self.registeredSeries['high'], self.registeredSeries['low'], self.registeredSeries['close'], self.registeredSeries['volume'], self.registeredSeries['top'], self.registeredSeries['bottom'] )

                # Print progress only during the main historical processing loop
                if self.barindex % 10000 == 0 and not self.jumpstart: 
                    print( self.barindex, "candles processed." )

                self.stream.tickEvent( self.realtimeCandle, False )

                self.jumpstart = False
                continue # Move to the next row in the `rows` input
            

            # NOT BACKTESTING: This is realtime
            last_timestamp = int( self.dataset[self.barindex, c.DF_TIMESTAMP] )
            if( newrow_timestamp <= last_timestamp ):
                if verbose : print( f"SKIPPING {self.timeframeStr}: {int( newrow.timestamp)}")
                continue

            # has it reached a new candle yet?
            if newrow_timestamp < self.realtimeCandle.timestamp + self.timeframeMsec:
                # no. This is still a real time candle update
                if is_fetch:
                    self.realtimeCandle.timestamp = newrow_timestamp
                    self.realtimeCandle.open = newrow_open
                    self.realtimeCandle.high = newrow_high
                    self.realtimeCandle.low = newrow_low
                    self.realtimeCandle.close = newrow_close
                    self.realtimeCandle.volume = newrow_volume
                else:
                    # combine the smaller candles into a bigger one
                    fecthTF = self.stream.timeframes[self.stream.timeframeFetch]
                    mask = fecthTF.dataset[:, c.DF_TIMESTAMP] >= self.realtimeCandle.timestamp
                    fetch_rows = fecthTF.dataset[mask]
                    if fetch_rows.shape[0] == 0:
                        self.realtimeCandle.open = newrow_open
                        self.realtimeCandle.high = newrow_high
                        self.realtimeCandle.low = newrow_low
                        self.realtimeCandle.close = newrow_close
                        self.realtimeCandle.volume = newrow_volume
                    else:
                        self.realtimeCandle.open = fetch_rows[0, c.DF_OPEN]
                        self.realtimeCandle.high = max(fetch_rows[:, c.DF_HIGH].max(), newrow_high)
                        self.realtimeCandle.low = min(fetch_rows[:, c.DF_LOW].min(), newrow_low)
                        self.realtimeCandle.close = newrow_close
                        self.realtimeCandle.volume = newrow_volume + fetch_rows[:, c.DF_VOLUME].sum()
                
                self.realtimeCandle.bottom = min( self.realtimeCandle.open, self.realtimeCandle.close )
                self.realtimeCandle.top = max( self.realtimeCandle.open, self.realtimeCandle.close )
                self.realtimeCandle.updateRemainingTime()

                if is_fetch :
                    self.stream.tickEvent( self.realtimeCandle, True )
                    if not self.stream.initializing:
                        push_tick_update( self )

                continue

            # NEW CANDLE - REAL-TIME
            # Append a new row to the Dataset for the closed candle data
            new_idx = self.barindex + 1

            # Create a new row with values from self.realtimeCandle
            new_row = np.full(self.dataset.shape[1], np.nan)
            new_row[c.DF_TIMESTAMP] = self.realtimeCandle.timestamp
            new_row[c.DF_OPEN]      = self.realtimeCandle.open
            new_row[c.DF_HIGH]      = self.realtimeCandle.high
            new_row[c.DF_LOW]       = self.realtimeCandle.low
            new_row[c.DF_CLOSE]     = self.realtimeCandle.close
            new_row[c.DF_VOLUME]    = self.realtimeCandle.volume
            new_row[c.DF_TOP]    = self.realtimeCandle.top
            new_row[c.DF_BOTTOM]    = self.realtimeCandle.bottom
            self.dataset = np.vstack([self.dataset, new_row]) # Append the new row to the dataset
            
            # reallocate the named series views because a new row was created
            for n in self.registeredSeries.keys():
                gs = self.registeredSeries[n]
                index = gs.index
                name = gs.name
                assignable = gs.assignable
                self.registeredSeries[n] = series_c( self.dataset[:, index], name, assignable= assignable, index= index )

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
            self.timestamp = int( self.dataset[ self.barindex, c.DF_TIMESTAMP ] )
            if is_fetch :
                self.stream.timestampFetch = self.realtimeCandle.timestamp

            print( f"NEW CANDLE {self.timeframeStr} : {newrow.tolist()}" )

            if( self.callback != None ):
                self.callback( self, self.registeredSeries['open'], self.registeredSeries['high'], self.registeredSeries['low'], self.registeredSeries['close'], self.registeredSeries['volume'], self.registeredSeries['top'], self.registeredSeries['bottom'] )

            # make sure no generated series is left unupdated
            for n in self.generatedSeries.keys():
                gs = self.generatedSeries[n]
                if gs.lastUpdatedTimestamp < self.barindex:
                    gs.update(self.registeredSeries[gs.source_name])

            self.stream.tickEvent( self.realtimeCandle, True )

            if not self.stream.initializing:
                push_row_update( self )
        

    def createColumn( self )->int:
        # Add the new column if necessary
        n_rows = self.dataset.shape[0]
        new_col = np.full((n_rows, 1), np.nan, dtype=np.float64)
        self.dataset = np.hstack([self.dataset, new_col])
        index = self.dataset.shape[1] - 1
        return index

    def createColumnSeries( self, name, assignable = True )->series_c:
        if name in self.registeredSeries.keys():
            raise ValueError( f"column [{name}] already exists" )

        index = self.createColumn()
        self.registeredSeries[name] = series_c(self.dataset[:,index], name, assignable, index)
        return self.registeredSeries[name]


    def calcGeneratedSeries( self, type:str, source: np.ndarray|generatedSeries_c, period:int, func, param=None, always_reset:bool = False )->generatedSeries_c:
        name = tools.generatedSeriesNameFormat( type, source, period )

        gse = self.generatedSeries.get( name )
        if( gse == None ):
            gse = generatedSeries_c( type, source, period, func, param, always_reset )
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
    
    def columnsList( self )->list:
        return list( self.registeredSeries.keys() )

    def seriesFromMultiObject( self, source: str|generatedSeries_c|np.ndarray )->series_c:
        if isinstance( source, series_c ):
            return source
        elif isinstance( source, generatedSeries_c ):
            return source.series()
        elif isinstance( source, str ):
            if source in self.registeredSeries.keys():
                return self.registeredSeries[source]
        elif isinstance( source, np.ndarray ):
            # try to guess its index but we won't allow it anyway
            index = tools.get_column_index_from_array( self.dataset, source )
            if index:
                name = list(self.registeredSeries.keys())[index]
                raise ValueError( f"seriesFromMultiObject: Numpy np.ndarray is not a valid object, but array index found [{index}]. Name: [{name}]" )
            raise ValueError( "seriesFromMultiObject: Numpy np.ndarray is not a valid object" )
        else:
            raise ValueError( "seriesFromMultiObject: Not a recognized series object" )

    def indexForTimestamp( self, timestamp:int )->int:
        # Estimate the index by dividing the offset by the time difference between rows
        baseTimestamp = self.dataset[0, c.DF_TIMESTAMP]
        index = int((timestamp - baseTimestamp) // self.timeframeMsec)
        return max(-1, index - 1) # Return the previous index or -1 if not found
    
    def timestampAtIndex( self, index:int )->int:
        return int( self.dataset[index, c.DF_TIMESTAMP] )

    def _columnIndex( self, column_name ):
        return list(self.registeredSeries.keys()).index[column_name]

    def valueAtTimestamp( self, column_name, timestamp:int ):
        index = self.indexForTimestamp(timestamp)
        if index == -1:
            return None
        if column_name not in self.registeredSeries.keys():
            raise ValueError(f"Column '{column_name}' not found in dataset.")
        return self.registeredSeries[column_name][index]

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
        
        row = self.dataset[index, :]
        candle.timestamp, candle.open, candle.high, candle.low, candle.close, candle.volume, candle.top, candle.bottom = ( int(row[0]), row[1], row[2], row[3], row[4], row[5], row[6], row[7] )
        return candle




class stream_c:
    def __init__( self, symbol, exchangeID:str, timeframeList, callbacks, event_callback = None, max_amount = 5000, cache_only = False ):
        self.symbol = symbol # FIXME: add verification
        self.initializing = True
        self.isRunning = False
        self.timeframeFetch = None
        self.timestampFetch = -1
        self.timeframes: dict[str, timeframe_c] = {}
        self.precision = 0.0
        self.mintick = 0.0
        self.cache_only = cache_only
        self.event_callback = event_callback

        self.markers:list[marker_c] = []
        self.lines:list[line_c] = []
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
        self.fee_maker = self.markets[self.symbol].get('maker') if self.markets[self.symbol].get('maker') is not None else 0.0
        self.fee_taker = self.markets[self.symbol].get('taker') if self.markets[self.symbol].get('taker') is not None else 0.0

        # hack thefor kucoin. It expects contracts and their contracts are units. That doesn't work for us
        if exchangeID == "kucoinfutures":
            self.mintick = 0.0
            self.precision = 0.0
            
        # fetch OHLCVs
        if self.cache_only:
            ohlcvs = fetcher.loadCache( self.symbol, self.timeframeFetch, max_amount * scale )
            print( "LOADING FROM CACHE")
        else:
            ohlcvs = fetcher.loadCacheAndFetchUpdate( self.symbol, self.timeframeFetch, max_amount * scale )
        if( len(ohlcvs) == 0 ):
            raise SystemExit( f'No candles available in {exchangeID}. Aborting')
 
        ohlcvNP = np.array( ohlcvs, dtype=np.float64 )
        del ohlcvs


        #################################################
        # Create the timeframe sets with their dataframes
        #################################################

        for i, t in enumerate(timeframeList):
            if t == self.timeframeFetch:
                candles = ohlcvNP
            else:
                candles = tools.resample_ohlcv_np( ohlcvNP, t )
                print( f"Resampled {t} : {len(candles)} rows" )

            # Add the new 'top' and 'bottom' columns to the candles NumPy array
            top_values = np.maximum(candles[:, c.DF_OPEN], candles[:, c.DF_CLOSE]).reshape(-1, 1)
            bottom_values = np.minimum(candles[:, c.DF_OPEN], candles[:, c.DF_CLOSE]).reshape(-1, 1)
            candles = np.hstack((candles, top_values, bottom_values))

            timeframe = timeframe_c( self, t )

            if i < len(callbacks):
                timeframe.callback = callbacks[i]

            self.timeframes[t] = timeframe
            timeframe.initDataframe( candles )
            
            candles = []

        # we skipped the last row at initializing each timeframe
        # so we can parse it now as an update because the last candle
        # we received is almost certainly not closed yet
        self.parseCandleUpdateMulti( ohlcvNP[-1:, :] )
        del ohlcvNP

        #################################################

        self.initializing = False

        #################################################

        # connect to ccxt.pro (FIXME? This should probably reside in the fetcher)
        try:
            self.exchange = getattr(ccxt, exchangeID)({
                    "options": {'defaultType': 'swap', 'adjustForTimeDifference' : True},
                    "enableRateLimit": False
                    }) 
        except Exception as e:
            raise SystemExit( "Couldn't initialize exchange:", exchangeID )
 

    def run(self, backtest_only = False ):
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

            if( len(response) ):
                self.parseCandleUpdateMulti( np.array( response, dtype=np.float64 ) )
            
            await asyncio.sleep(0.01)

        await self.exchange.close()

    def tickEvent( self, candle:candle_c, realtime:bool ):
        if active.timeframe.timeframeStr != self.timeframeFetch :
           return

        if self.event_callback:
            self.event_callback( self, "tick", (candle, realtime), 2 )

        from .trade import newTick
        newTick(candle, realtime)

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

    
    def registerPanel( self, name:str, width:float, height:float, fontsize = 14, show_candles:bool = False, show_timescale = True, show_volume = False, show_labels = False, show_priceline = False, show_plotnames = False, background_color= None, text_color= None ):
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
            "show_volume": show_volume,
            "background_color": background_color,
            "text_color": text_color
        }
        self.registeredPanels[name] = panel


    def createMarker( self, text:str = '', location:str = 'below', shape:str = 'circle', color:str = "#DEDEDE", timestamp:int = None, chart_name:str = None )->marker_c:
        '''MARKER_POSITION = Literal['above', 'below', 'inside']
        MARKER_SHAPE = Literal['arrow_up', 'arrow_down', 'circle', 'square']'''
        import bisect

        if timestamp == None:
            timestamp = self.timeframes[self.timeframeFetch].timestamp
        marker = marker_c( text, int(timestamp), location, shape, color, chart_name )

        if len(self.markers) and marker.timestamp < self.markers[-1].timestamp: # we need to insert back in time
            insertion_index = bisect.bisect_left( [m.timestamp for m in self.markers], marker.timestamp )
            self.markers.insert(insertion_index, marker)
        else:
            self.markers.append( marker )

        return marker
    
    def removeMarker( self, marker:marker_c ):
        if marker != None and isinstance(marker, marker_c):
            self.markers.remove( marker )
    
    def createLine( self, x1, y1, x2, y2, color:str = '#c7c7c7', width = 1, style = 'solid', chart_name:str = 'main' )->line_c:
        '''LINE_STYLE = Literal['solid', 'dotted', 'dashed', 'large_dashed', 'sparse_dotted']'''
        line = line_c( x1, y1, x2, y2, color, width, style, chart_name )
        self.lines.append( line )
        return line
    
    def removeLine( self, line:line_c ):
        if line:
            self.lines.remove(line)

    def createWindow( self, timeframeStr ):
        """Create and show a window for the given timeframe"""
        if not tools.validateTimeframeName( timeframeStr):
            print( f"{timeframeStr} is not a valid timeframe name" )
            return
        if timeframeStr not in self.timeframes.keys():
            print( f"Available timeframes: {list(self.timeframes.keys())}" )
            return
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

def removeMarker( marker:marker_c ):
    active.timeframe.stream.removeMarker(marker)

def createLine( x1, y1, x2, y2, color:str = '#c7c7c7', width = 1, style = 'solid', chart_name:str = 'main' )->line_c:
    '''LINE_STYLE = Literal['solid', 'dotted', 'dashed', 'large_dashed', 'sparse_dotted']'''
    return active.timeframe.stream.createLine( x1, y1, x2, y2, color, width, style, chart_name )

def removeLine( marker:marker_c ):
    active.timeframe.stream.removeLine(marker)

def getRealtimeCandle()->candle_c:
    return active.timeframe.realtimeCandle

def getCandle( index = None )->candle_c:
    return active.timeframe.candle(index)

def getMintick()->float:
    return active.timeframe.stream.mintick

def getPrecision()->float:
    return active.timeframe.stream.precision

def getFees()->tuple[float,float]:
    return active.timeframe.stream.fee_maker, active.timeframe.stream.fee_taker

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
            stream.createWindow( args )

        elif command == 'close':
            # TODO: Function to send a command to the client to shutdown
            print('closing chart')

        else:
            stream.event_callback(stream, "cli_command", (command, args), 2)

        await asyncio.sleep(0.05)




    



