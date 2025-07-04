import zmq
import zmq.asyncio
import asyncio
import sys
import json
import pandas as pd
import numpy as np
import bisect
from typing import Optional, Any

# REPLACE ALL IMPORTS WITH THESE:
# import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Add project root to path

# Now use absolute imports
from framework import tasks
from framework.constants import c
from framework.candle import candle_c

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add port parameter handling
import argparse

# Default ports
DEFAULT_CMD_PORT = 5555
DEFAULT_PUB_PORT = 5556

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, help='Command port number (pub port will be port+1)')
args = parser.parse_args()

# Set ports
cmd_port = args.port if args.port is not None else DEFAULT_CMD_PORT
pub_port = cmd_port + 1 if args.port is not None else DEFAULT_PUB_PORT

debug = False




############################ CHART WINDOW ################################
from lightweight_charts import Chart
# from lightweight_charts_esistjosh import Chart

from dataclasses import dataclass
@dataclass
class plot_c:
    name: str
    panel:str
    type: int
    color:str
    style:str
    width:int
    margin_top:float
    margin_bottom:float
    instance:object

@dataclass
class marker_c:
    id:int
    timestamp:int
    position:str
    shape:str
    color:str
    panel:str
    text:str
    instance:object

@dataclass
class line_c:
    id:int
    x1:int
    y1:int
    x2:int
    y2:int
    color:str
    width:int
    style:str
    panel:str
    clipped:bool
    instance:object
    
class window_c:
    def __init__(self, config):
        self.config = config
        self.descriptor: Optional[dict[str, Any]] = None
        self.columns:list = []
        self.plots:list[plot_c] = []
        self.markers:list = []
        self.lines:list = []
        self.lines_clipped:list = []
        self.numpanels = 0
        self.lastCandle:candle_c = None
        self.barindex = -1
        self.timestamp = 0

        # config (client)
        self.timerOnPriceLabel = False
        self.showRealTimeCandle = True
        self.priceScaleMinimumWidth = 90

        # calculate the panels sizes
        self.panels = config['panels']
        # add the main panel
        if self.panels.get('main') == None:
            # self.panels['main'] = { "type": c.PANEL_HORIZONTAL, "position": "above", "width": 1.0, "height": 1.0 }
            self.panels['main'] = {
                "position": "above",
                "width": 1.0,
                "height": 1.0,
                "fontsize": 12,
                "show_candles": True,
                "show_timescale": True,
                "show_labels": True,
                "show_priceline": True,
                "show_plotnames": True,
                "show_volume": False
            }

        # first figure out how much space are going to 
        # take the other panels which are not the main panel
        fullheightbottom = 0.0
        fullheighttop = 0.0
        for n in self.panels.keys():
            if n == 'main': continue
            self.numpanels += 1
            panel = self.panels[n]
            h = panel.get('height')
            if h == None:
                panel['height'] = h = 0.1
            position = panel.get('position')
            if position is None:
                panel['position'] = position = 'bottom'

            if panel['position'] == 'bottom':
                fullheightbottom += h
            elif panel['position'] == 'top':
                fullheighttop += h
            # is it possible to have panels both above and below the main panel? I have to check that.

        if fullheightbottom + fullheighttop > 1.0:
            # This is way out of bounds. We should rescale but by now let's just crash
            raise ValueError( f"Panels exceed the maximum heigh. Check the accumulated panel heights don't exceed 1.0" )
        
        self.panels['main']['height'] = 1.0 - (fullheightbottom + fullheighttop)

        # to do: figure out how to do the same with widths


        self.read_config()

    def writeConfig(self):
        with open('config.json', 'w') as f:
            configString = '[\n\t{\n'
            configString += f'\t\t"timer_on_price_label":{str(self.timerOnPriceLabel).lower()},\n'
            configString += f'\t\t"show_realtime_candle":{str(self.showRealTimeCandle).lower()},\n'
            configString += f'\t\t"price_scale_minimum_width":{str(self.priceScaleMinimumWidth).lower()}\n'
            configString += '\t}\n]'
            
            f.write( configString )
            f.close()

    def read_config(self):
        try:
            with open('config.json', 'r') as config_file:
                config = json.load(config_file)
                config = config[0]
                config_file.close()
        except FileNotFoundError:
            self.writeConfig()
            print( "Config file created.\n----------------------------")
            return

        # parse the config file
        if( config.get('timer_on_price_label') != None ):
            self.timerOnPriceLabel = bool(config.get('timer_on_price_label'))
        if( config.get('show_realtime_candle') != None ):
            self.showRealTimeCandle = bool(config.get('show_realtime_candle'))
        if( config.get('price_scale_minimum_width') != None ):
            self.priceScaleMinimumWidth = int(config.get('price_scale_minimum_width'))
 
        #rewrite the config file
        self.writeConfig()




    def loadChartData(self, descriptor, df):
        if debug : print( "Initializing window" )

        self.descriptor = descriptor
        self.barindex = len(df) - 1
        self.timestamp = df['timestamp'].iat[self.barindex]

        window_width = 1024
        window_height = 768
        if 1:
            screen_width, screen_height = self.get_screen_resolution()
            window_width = int(screen_width * 0.65)
            window_height = int(screen_height * 0.65)
        self.panels["main"]["chart"] = chart = Chart( window_width, window_height, inner_height=self.panels["main"]["height"], inner_width=self.panels["main"]["width"] )
        chart.layout( font_size=self.panels['main']['fontsize'] )
        if self.numpanels > 0 : 
            chart.time_scale( visible=False, time_visible=False )


        chart.price_scale(minimum_width=self.priceScaleMinimumWidth) # FIXME: try to autoscale it

        self.legend = f"{self.config['symbol']}"
        chart.legend( visible=False, ohlc=False, percent=False, font_size=self.panels['main']['fontsize']+2, text=self.legend )

        volume_alpha = 0.8 if self.panels["main"]["show_volume"] else 0.0
        chart.volume_config(
            scale_margin_top = 0.8, 
            scale_margin_bottom = 0.0, 
            up_color=f'rgba(83,141,131,{volume_alpha})', 
            down_color=f'rgba(200,127,130,{volume_alpha})')
        
        # buttons
        self.initTopbar( chart )

        try:
            time_df = pd.DataFrame( { 'time':pd.to_datetime( df['timestamp'], unit='ms' ), 'open':df['open'], 'high':df['high'], 'low':df['low'], 'close':df['close'], 'volume':df['volume']} )
        except Exception as e:
            print(f"Error converting timestamp to datetime: {e}")

        try:
            chart.set( time_df )
        except Exception as e:
            print(f"Error setting chart dataframe: {e}")

        # create subpanels if there are any
        for n in self.panels.keys():
            if n == 'main': continue
            panel = self.panels[n]
            panel["chart"] = subchart = chart.create_subchart( panel["position"], width = panel["width"], height = panel["height"], sync=chart.id )
            subchart.layout( font_size=panel["fontsize"] )
            allow_line_names = panel["show_plotnames"]
            subchart.legend( visible=False, ohlc=False, percent=False, lines = allow_line_names, font_size=14, text=n ) # lines info crash the script when enabled
            subchart.crosshair( horz_visible=False )
            subchart.time_scale( visible=panel["show_timescale"], time_visible=panel["show_timescale"] )
            subchart.price_line( label_visible=panel["show_labels"], line_visible=panel["show_priceline"] )
            # subchart.precision( self.bottompanel_precision )
            # subchart.price_scale(minimum_width=price_column_width)
            subchart.price_scale(minimum_width=self.priceScaleMinimumWidth) # FIXME: try to autoscale it
            subchart.set(time_df)
            if not panel["show_candles"]:
                subchart.hide_data()

        self.descriptor = descriptor
        self.columns = df.columns

        self.lastCandle = candle_c( df.iloc[-1].tolist() ) # create a candle object for the clock
        self.lastCandle.timeframemsec = descriptor["timeframemsec"]
        self.lastCandle.index = df.index[-1]
        self.lastCandle.updateRemainingTime()
        tasks.registerTask('clocks', self.update_clocks)

        self.createPlots(df)
        self.addMarkers( descriptor['markers'].get("added") )
        self.addLines( descriptor['lines'].get("added") )


        tasks.registerTask('window', chart.show_async)


    def createPlots(self, df:pd.DataFrame):
        plotsList = self.descriptor['plots']
        for name in plotsList.keys():
            info = plotsList[name]
            plot = plot_c( 
                name,
                info.get('panel'),
                int(info.get('type')),
                info.get('color'),
                info.get('style'),
                int(info.get('width')),
                float(info.get('margin_top')),
                float(info.get('margin_bottom')),
                None
            )
            
            if plot.panel:
                panel = self.panels.get(plot.panel)
                if panel == None:
                    print( f"WARNING: Couldn't find panel [{plot.panel}] for plot [{plot.name}]" )
                    continue
                chart = panel['chart']
            else:
                chart = self.panels['main']['chart']
            
            if plot.type == c.PLOT_LINE :
                plot.instance = chart.create_line( plot.name, plot.color, plot.style, plot.width, price_line=False, price_label=False )
                plot.instance.set( pd.DataFrame( {'time': pd.to_datetime( df['timestamp'], unit='ms' ), plot.name: df[plot.name]} ) )
            elif plot.type == c.PLOT_HIST :
                plot.instance = chart.create_histogram( plot.name, plot.color, price_line = False, price_label = False, scale_margin_top = plot.margin_top, scale_margin_bottom = plot.margin_bottom )
                plot.instance.set( pd.DataFrame( {'time': pd.to_datetime( df['timestamp'], unit='ms' ), plot.name: df[plot.name]} ) )

            self.plots.append( plot )


    def createMarker( self, m ):
        marker = marker_c(
                id = int(m.get('id')),
                timestamp = int(m.get('timestamp')),
                position = m.get('position'),
                shape = m.get('shape'),
                color = m.get('color'),
                panel = m.get('panel') if m.get('panel') != None else 'main',
                text = m.get('text'),
                instance = None
            )

        chart = self.panels['main']['chart']

        panel = self.panels.get(marker.panel)
        if panel == None:
            print( f"WARNING: Couldn't find panel [{marker.panel}] for marker. Using main]" )
            marker.panel = 'main'
            chart = panel['main']
        else:
            chart = panel['chart']
            

        marker.instance = chart.marker( time = pd.to_datetime( marker.timestamp, unit='ms' ),
                    position = marker.position,
                    shape = marker.shape,
                    color = marker.color,
                    text = marker.text )
        
        self.markers.append( marker )
    

    def removeMarker( self, msg ):
        id = int(msg.get('id'))
        timestamp = int(msg.get('timestamp'))
        for marker in reversed(self.markers):
            if marker.timestamp < timestamp:
                break
            if marker.id != id:
                continue
            if marker.instance != None:
                assert(marker.panel != None)
                chart = self.panels[marker.panel]['chart']
                assert(chart.remove_marker(marker.instance) == None)
                marker.instance = None
            self.markers.remove(marker)
            break
            

    def addMarker( self, msg ):
        
        if len(self.markers) == 0:
            self.createMarker(msg)
            return
        
        # verify the timestamp is not older than the last marker
        lastMarker = self.markers[-1]
        marker_timestamp = int(msg.get('timestamp'))
        assert(marker_timestamp != None)
        if lastMarker.timestamp <= marker_timestamp :
            self.createMarker( msg )
            return
        
        # we're screwed. We need to remove the most recent markers and put them back.
        try:
            marker = marker_c(
                id = int(msg.get('id')),
                timestamp = marker_timestamp,
                position = msg.get('position'),
                shape = msg.get('shape'),
                color = msg.get('color'),
                panel = msg.get('panel') if msg.get('panel') != None else 'main',
                text = msg.get('text'),
                instance = None
            )

            insertion_index = bisect.bisect_left( [m.timestamp for m in self.markers], marker.timestamp )

            # now we need to remove all the ones above the index from the chart and add them again
            # we are only removing them from the chart, but we keep them in the list to restore them
            for index in range(len(self.markers) - 1, insertion_index - 1, -1):
                cm = self.markers[index]
                if cm.instance == None: 
                    continue

                assert(cm.panel != None)
                chart = self.panels[cm.panel]['chart']
                assert(chart.remove_marker(cm.instance) == None)
                cm.instance = None
                
            # add the new one
            marker.instance = chart.marker( time = pd.to_datetime( marker.timestamp, unit='ms' ),
                position = marker.position,
                shape = marker.shape,
                color = marker.color,
                text = marker.text )
            self.markers.insert(insertion_index, marker)

            # now add them all back
            for index in range(insertion_index, len(self.markers)):
                cm = self.markers[index]
                if cm.instance != None: 
                    continue
                cm.instance = chart.marker( time = pd.to_datetime( cm.timestamp, unit='ms' ),
                    position = cm.position,
                    shape = cm.shape,
                    color = cm.color,
                    text = cm.text )

        except Exception as e:
            print( "Deleting markers failed with:", e )


    def addMarkers( self, addlist ):
        if addlist is None or len(addlist) == 0:
            return
        
        if len(self.markers) == 0: # we don't need to worry about sorting
            if len(addlist) > 500:
                print( f"* Warning: Very high marker count {len(addlist)}. The chart will take a long time to open")
            elif len(addlist) > 250:
                print( f"Hold on. High marker count [{len(addlist)}]")
            for m in addlist:
                self.createMarker(m)
            return
        
        # in here we have a delta so we have to add them one by one so they fix the sorting problems
        for m in addlist:
            self.addMarker(m)


    def removeMarkers( self, removelist ):
        if removelist is None or len(removelist) == 0:
            return
        for m in removelist:
            self.removeMarker(m)


    @staticmethod
    def clip_line_left(x1, y1, x2, y2, x_boundary):
        if x1 < x_boundary:
            if x1 != x2:  # Handle vertical lines
                slope = (y2 - y1) / float(x2 - x1)
                new_y1 = y1 + slope * (x_boundary - x1)
                return (x_boundary, new_y1, True)  # Clipped
        return (x1, y1, False)
    
    @staticmethod
    def clip_line_right(x1, y1, x2, y2, x_boundary):
        if x2 > x_boundary:
            if x1 != x2:  # Handle vertical lines
                slope = (y2 - y1) / float(x2 - x1)
                new_y2 = y1 + slope * (x_boundary - x1)
                return (x_boundary, new_y2, True)
        return (x2, y2, False)  # No clipping needed


    def reclipLine( self, line ):
        if not line.clipped:
            # this should be a ValueError
            return

        if line.x1 >= self.barindex: # all of it is out of range, but will show up eventually
            if not( line.x1 == line.x2 and line.x1 == self.barindex ): # if it's pure vertical it can show up
                self.lines_clipped.append(line)
                return

        line.clipped = False
        try:
            if line.instance != None:
                line.instance.delete()
        except Exception as e:
            print("LINE DIDN'T DELETE", e)
        
        line.instance = None

        x1 = line.x1
        y1 = line.y1
        x2 = line.x2
        y2 = line.y2

        # clip the lines when they go out of bounds
        x1, y1, c = self.clip_line_left(x1, y1, x2, y2, 0)
        x2, y2, line.clipped = self.clip_line_right(x1, y1, x2, y2, self.barindex)
            
        chart:Chart = self.panels[line.panel]['chart']

        # convert the indexes to timestamps
        timeframeMsec = int( self.descriptor['timeframemsec'] )
        time1 = self.timestamp - (( self.barindex - x1 ) * timeframeMsec)
        time2 = self.timestamp - (( self.barindex - x2 ) * timeframeMsec)
        line.instance = chart.trend_line( 
            pd.to_datetime( time1, unit='ms' ), 
            y1, 
            pd.to_datetime( time2, unit='ms' ), 
            y2, 
            round = False, 
            line_color=line.color, 
            width=line.width,
            style=line.style )
        
        if line.clipped == True:
            self.lines_clipped.append(line)


    def reclipLines( self ):
        if len(self.lines_clipped) == 0:
            return
        clippedList = list(self.lines_clipped)
        self.lines_clipped = []
        for line in clippedList:
            self.reclipLine(line)


    def removeLine( self, id ):
        for line in reversed(self.lines):
            if line.id != id:
                continue
            if line.instance != None:
                line.instance.delete()
            line.instance = None
            self.lines.remove(line)
            break


    def removeLines( self, removelist ):
        if removelist is None or len(removelist) == 0:
            return
        for line in removelist:
            self.removeLine(line.id)


    def addLine( self, msg ):
        try:
            line = line_c(
                id = int( msg.get('id') ),
                x1 = int( msg.get('x1') ),
                y1 = float( msg.get('y1') ),
                x2 = int( msg.get('x2') ),
                y2 = float( msg.get('y2') ),
                color = msg.get('color'),
                width = int( msg.get('width') ),
                style = msg.get('style'),
                panel = msg.get('panel'),
                clipped = False,
                instance = None
            )

            if line.x1 > line.x2: # keep x1 and x2 always chronologically aligned
                x1 = line.x1
                y1 = line.y1
                line.x1 = line.x2
                line.y1 = line.y2
                line.x2 = x1
                line.y2 = y1

            x1 = line.x1
            y1 = line.y1
            x2 = line.x2
            y2 = line.y2

            if x1 >= self.barindex: # all of it is out of range, but will show up eventually
                if not( x1 == x2 and x1 == self.barindex ):
                    line.clipped = True
                    self.lines.append(line)
                    return
            
            if x2 < 0: # This like will never be visible
                return

            x1, y1, c = self.clip_line_left(x1, y1, x2, y2, 0)
            x2, y2, line.clipped = self.clip_line_right(x1, y1, x2, y2, self.barindex)

            if line.panel not in self.panels.keys():
                line.panel = 'main'

            chart:Chart = self.panels[line.panel]['chart']

            # convert the indexes to timestamps
            timeframeMsec = int( self.descriptor['timeframemsec'] )
            time1 = self.timestamp - (( self.barindex - x1 ) * timeframeMsec)
            time2 = self.timestamp - (( self.barindex - x2 ) * timeframeMsec)
            line.instance = chart.trend_line( 
                pd.to_datetime( time1, unit='ms' ), 
                y1, 
                pd.to_datetime( time2, unit='ms' ), 
                y2, 
                round = False, 
                line_color=line.color, 
                width=line.width,
                style= line.style )

            if line.instance == None:
                print( "FAILED TO ADD LINE" )
            self.lines.append(line)
            if line.clipped:
                self.lines_clipped.append(line)
        except Exception as e:
            print( "Exception", e )



    def addLines( self, addlist ):
        if addlist is None or len(addlist) == 0:
            return
        
        for line in addlist:
            self.addLine( line )


    def newTick(self, msg):
        
        row = msg.get('data')
        if not row:
            return
        assert(row[c.DF_TIMESTAMP] is not None)
        row[c.DF_TIMESTAMP] = int(row[c.DF_TIMESTAMP])
        
        self.lastCandle.updateFromSource(row) # for the clock
        self.lastCandle.updateRemainingTime()
        
        if not self.showRealTimeCandle :
            return
        
        # OHLCV update to the chart
        data_dict = {
            'time': pd.to_datetime(row[c.DF_TIMESTAMP], unit='ms'),
            'open': row[c.DF_OPEN],
            'high': row[c.DF_HIGH],
            'low': row[c.DF_LOW],
            'close': row[c.DF_CLOSE],
            'volume': row[c.DF_VOLUME]
        }
        
        series = pd.Series(data_dict)
        for n in self.panels.keys():
            chart = self.panels[n]['chart']
            chart.update( series )


    def newRow(self, msg):
        row = msg.get('data')
        row[c.DF_TIMESTAMP] = int(row[c.DF_TIMESTAMP]) # fix type
        assert(row[c.DF_TIMESTAMP] is not None)
        columns = msg.get('columns')
        self.barindex = int( msg.get('barindex') )
        self.timestamp = row[c.DF_TIMESTAMP]

        # OHLCV update
        data_dict = {
            'time': pd.to_datetime(row[c.DF_TIMESTAMP], unit='ms'),
            'open': row[c.DF_OPEN],
            'high': row[c.DF_HIGH],
            'low': row[c.DF_LOW],
            'close': row[c.DF_CLOSE],
            'volume': row[c.DF_VOLUME]
        }
        
        try:
            series = pd.Series(data_dict)
            for n in self.panels.keys():
                chart = self.panels[n]['chart']
                chart.update( series )
        except Exception as e:
            raise ValueError( f"Failed  to locate the charts to update OHLCVs. {e}")

        # Second part - full data update
        if columns is None:
            return

        if len(columns) != len(self.columns):
            # ToDo: The dataframe has changed. We need to reload it
            raise ValueError( "Dataframe columns have changed" )
        
        
        try:
        # run through the list of plots and issue the updates
            for plot in self.plots:
                index = columns.index(plot.name)
                value = row[index]
                if value is None or pd.isna(value):
                    continue
                    # raise ValueError(f"Plot value for [{plot.name}] is None or NaN (row={row})")

                if plot.type == c.PLOT_LINE or plot.type == c.PLOT_HIST:
                    plot.instance.update( pd.Series( {'time': data_dict['time'], 'value': value } ) )
        except Exception as e:
            raise ValueError( f"Failed to update a plot because: {e}")
        
        # markers delta update
        self.removeMarkers( msg['markers'].get("removed") ) 
        self.addMarkers( msg['markers'].get("added") ) 

        self.removeLines( msg['lines'].get("removed") )
        self.reclipLines()
        self.addLines( msg['lines'].get("added") )

        # finally add the opening of the realtime candle
        self.newTick( msg.get('tick') )
        

    def isAlive(self)->bool:
        chart:Chart = self.panels['main']['chart']
        return chart.is_alive


    # There is no reason for this to be a method other than grouping all the window stuff together
    def get_screen_resolution(self):
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()  # Destroy the window after getting the resolution
        return screen_width, screen_height
    
        
    async def update_clocks( self ):
        from datetime import datetime

        while True:
            await asyncio.sleep(1-(datetime.now().microsecond/1_000_000))
            self.lastCandle.updateRemainingTime()
            chart:Chart = self.panels['main']['chart']
            if self.timerOnPriceLabel:
                chart.price_line( True, True, self.lastCandle.remainingTimeStr() )
            else:
                chart.price_line( True, True, '' )

            chart.topbar['timer'].set( f' { self.lastCandle.remainingTimeStr()}' )
    
    def initTopbar(self, chart:Chart):
        try:
            chart.topbar.textbox("header", f'{ self.config["symbol"] } - { self.descriptor["timeframe"] }', align= 'left')
            chart.topbar.button('legendswtich', '∇', func=self.button_legend_press, align= 'left')
            chart.topbar.menu( "hal", ("hol", "hil", "hal"), "hil") 
            #chart.topbar.switcher( 'thisthat', ("this", "that"), "that" )
            # ^—–▽▼▭∆∇∨∧⋀⋁⋎⋏⩔⩡Λ


            chart.topbar.button('timerswtich', 'Timer on price ▢', func=self.button_timerlabel_press, align= 'right')
            chart.topbar.textbox("timer", "--:--", align= 'right')
            
        except Exception as e:
            print( f'{e}')
    
    async def button_timerlabel_press(self, chart):
        try:
            # 'Timer on price ▢■▭▬▮▯▩▧▦▣■□▢▥□▣'
            timeron = 'Timer on price ▣'
            timeroff = 'Timer on price ▢'
            if chart.topbar['timerswtich'].value == timeron:
                chart.topbar['timerswtich'].set(timeroff)
                self.timerOnPriceLabel = False
            elif chart.topbar['timerswtich'].value == timeroff:
                chart.topbar['timerswtich'].set(timeron)
                self.timerOnPriceLabel = True
            
        except Exception as e:
            print( f'Exception {e}')

    async def button_legend_press(self, chart:Chart):
        try:
            # # ^—–▽▼▭∆∇∨∧⋀⋁⋎⋏⩔⩡Λ
            legendon = 'Λ'
            legendoff = '∇'
            if chart.topbar['legendswtich'].value == legendon:
                chart.topbar['legendswtich'].set(legendoff)
                # chart.legend( visible=False, ohlc=False, percent=False, lines=True, font_size=self.panels['main']['fontsize']+2, text=self.legend )
                for n in self.panels.keys():
                    panel = self.panels[n]
                    fontsize = panel['fontsize'] if n != 'main' else panel['fontsize']+2
                    panel['chart'].legend( visible=False, ohlc=False, percent=False, lines=True, font_size=fontsize, text=n )


            elif chart.topbar['legendswtich'].value == legendoff:
                chart.topbar['legendswtich'].set(legendon)
                # chart.legend( visible=True, ohlc=False, percent=False, lines=True, font_size=self.panels['main']['fontsize']+2, text=self.legend )
                for n in self.panels.keys():
                    panel = self.panels[n]
                    fontsize = panel['fontsize'] if n != 'main' else panel['fontsize']+2
                    panel['chart'].legend( visible=True, ohlc=False, percent=False, lines=True, font_size=fontsize, text=n )
            
        except Exception as e:
            print( f'Exception {e}')

window:window_c = None




###########################################################################
###########################################################################
############################### CLIENT ####################################
###########################################################################
###########################################################################

CLIENT_DISCONNECTED = 0
CLIENT_CONNECTED = 1
CLIENT_LOADING = 2  # receiving the data to open the window
CLIENT_READY = 3
CLIENT_LISTENING = 4  # the window has already opened the window and is ready to receive updates.

status = CLIENT_DISCONNECTED


# In client.py, modify the send_command function:

async def send_command(socket, command: str, params: str = ""):
    global status, window

    """Send a command to the server"""
    message = f"{command} {params}".strip()
    await socket.send_string(message)
    if debug : print(f"Sent command: {message}")
    
    # Get the reply
    reply = await socket.recv_string()

    try:
        # Try to parse as JSON
        data = json.loads(reply)
        if isinstance(data, dict) and 'type' in data:
            if data['type'] == 'config':
                if debug:print(f"Received config message for symbol: {data['symbol']}")
                # create the window container
                window = window_c(data)
                status = CLIENT_CONNECTED
                return data
                
            elif data['type'] == 'data_descriptor':
                status = CLIENT_LOADING
                descriptor = data
                
                # Send acknowledgment that we're ready for the data
                await socket.send_string("ready")
                
                try:
                    # Get the raw data
                    raw_data = await socket.recv()
                    
                    # Convert raw bytes back to numpy array
                    array_data = np.frombuffer(raw_data, dtype=np.float64)
                    
                    # Reshape with explicit size calculation
                    rows = data['rows']
                    cols = len(data['columns'])
                    expected_size = rows * cols
                    
                    if array_data.size != expected_size:
                        raise ValueError(f"Data size mismatch. Expected {expected_size}, got {array_data.size}")
                    
                    array_data = array_data.reshape(rows, cols)
                    
                    # Create DataFrame with explicit dtypes
                    df = pd.DataFrame(array_data, columns=data['columns'])
                    
                    # Handle timestamp column separately
                    if 'timestamp' in df.columns:
                        df['timestamp'] = df['timestamp'].astype(np.int64)
                    
                    print("Opening chart")
                    if debug : print(f"DataFrame shape: {df.shape}")
                    ### fall through ###
                except Exception as e:
                    print(f"Error reconstructing DataFrame: {e}")
                    status = CLIENT_CONNECTED
                    return None
                
                # initialize the window with the dataframe and open it
                window.loadChartData( descriptor, df )
                status = CLIENT_READY
                return data

    except json.JSONDecodeError:
        # Not JSON, treat as regular message
        if reply == "connected":
            status = CLIENT_CONNECTED
            print("Connected")
        else:
            if debug : print(f"Received reply: {reply}")
        return reply

async def listen_for_updates(context):
    """Listen for updates from server"""
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://127.0.0.1:{pub_port}")  # Modified line
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    if debug : print("listening...")
    
    try:
        while True:
            try:
                if debug : print("Waiting for message...")
                message = await socket.recv_string()
                if debug: print("Received updates message")
                try:
                    data = json.loads(message)
                    if data['type'] == 'row':
                        window.newRow(data)
                    elif data['type'] == 'tick':
                        window.newTick(data)
                    elif data['type'] == 'marker':
                        if data['action'] == 'add':
                            window.addMarker( data['data'] )
                        elif data['action'] == 'remove':
                            pass
                    
                except json.JSONDecodeError:
                    print(f"Error: Received invalid JSON update")
                
            except Exception as e:
                print(f"Error in listen_for_updates loop: {e}")
            
            # Remove or reduce this sleep - it might be causing us to miss messages
            await asyncio.sleep(0.01)  # Reduced from 0.1 to 0.01
    
    except asyncio.CancelledError:
        print("Update listener cancelled")
    finally:
        socket.close()

async def run_client():
    global status
    # ZeroMQ Context
    context = zmq.asyncio.Context()

    # Define the socket for commands using REQ/REP pattern
    cmd_socket = context.socket(zmq.REQ)
    cmd_socket.connect(f"tcp://127.0.0.1:{cmd_port}")  # Modified line

    if debug : print("Client is running and connected to server...")

    try:
        # Start the update listener
        tasks.registerTask("zmq_updates", listen_for_updates(context))

        while True:
            if status == CLIENT_DISCONNECTED:
                await send_command(cmd_socket, "connect", "")
                await asyncio.sleep(0.25)
                continue

            if status == CLIENT_CONNECTED:
                await send_command(cmd_socket, "dataframe", "")
                await asyncio.sleep(0.25)
                continue

            if status == CLIENT_LOADING:
                await asyncio.sleep(0.5)

            if status == CLIENT_READY:
                status = CLIENT_LISTENING
                await send_command(cmd_socket, "listening", "")
                await asyncio.sleep(0.25)

            if status == CLIENT_LISTENING:
                await send_command(cmd_socket, "ack", "") # keepalive
                await asyncio.sleep(10)

    except asyncio.CancelledError:
        print("Client task cancelled")
    finally:
        cmd_socket.close()
        context.term()

# Register the client as a task
tasks.registerTask("zmq_client", run_client())

if __name__ == "__main__":
    try:
        # Use the tasks system to run the client
        asyncio.run(tasks.runTasks())
    except KeyboardInterrupt:
        print("\nClient stopped by user")