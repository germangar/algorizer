import zmq
import zmq.asyncio
import msgpack
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
parser.add_argument('--timeframe', type=str, help='Timeframe to request (optional)')
args = parser.parse_args()

# Set ports
cmd_port = args.port if args.port is not None else DEFAULT_CMD_PORT
pub_port = cmd_port + 1 if args.port is not None else DEFAULT_PUB_PORT
timeframe_request = args.timeframe  # None if not provided

debug = False




############################ CHART WINDOW ################################
from lightweight_charts import Chart
# from lightweight_charts_esistjosh import Chart

theme_black = {
    'background_color':"#000000",
    'text_color':"#dddddd",
    'price_scale_color':"#dddddd",
    'price_scale_border_color':"#5a5a5aff",
    'grid_color': "#30303386",
    'bull_color':'#279d82',
    'bear_color':'#c86164',
    'candle_bull_color':'#279d82',
    'candle_bear_color':'#c86164',
    'wick_bull_color':'#279d82',
    'wick_bear_color':'#c86164',
    'candle_border_bull_color':'#279d82',
    'candle_border_bear_color':'#c86164'
}

theme_dark = {
    'background_color':"#121414",
    'text_color':"#dddddd",
    'price_scale_color':"#dddddd",
    'price_scale_border_color':"#5a5a5aff",
    'grid_color': "#30303386",
    'bull_color':"#088755",
    'bear_color':"#AE3853",
    'candle_bull_color':"#088755",
    'candle_bear_color':"#AE3853",
    'wick_bull_color':"#088755",
    'wick_bear_color':"#AE3853",
    'candle_border_bull_color':"#088755",
    'candle_border_bear_color':"#AE3853"
}

theme_light = {
    'background_color':"#dddddfff",
    'text_color':"#2D2D2D",
    'price_scale_color':"#2F3031",
    'price_scale_border_color':"#686869ff",
    'grid_color': "#9C9DA08C",
    'bull_color':"#039974",
    'bear_color':"#D44667",
    'candle_bull_color':"#039974",
    'candle_bear_color':"#D44667",
    'wick_bull_color':"#039974",
    'wick_bear_color':"#D44667",
    'candle_border_bull_color':"#039974",
    'candle_border_bear_color':"#D44667"
}

        # self.backgroundColor = "#dddddfff" # "#d5d7daff"
        # self.textColor = "#2D2D2D"
        # self.priceScaleColor = "#2F3031"
        # self.priceScaleBorderColor = "#686869ff"
        # self.gridColor = "#9C9DA08C"
        # self.bullColor = "#039974"
        # self.bearColor = "#D44667"
        # self.candleBullColor = self.bullColor
        # self.candleBearColor = self.bearColor
        # self.wickBullColor = self.bullColor
        # self.wickBearColor = self.bearColor
        # self.candleBorderBullColor = self.bullColor
        # self.candleBorderBearColor = self.bearColor

def hx2rgba(hex_color):
    """Converts a hex color code (with or without alpha) to an RGBA string for CSS."""
    hex_color = hex_color.lstrip('#')
    hex_length = len(hex_color)
    if hex_length not in (6, 8):
        return None  # Invalid hex code length

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    if hex_length == 6:
        a = 1.0
    else:
        a = round(int(hex_color[6:8], 16) / 255, 3)

    return r, g, b, a

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
        self.basetimestamp = 0

        # config (client)
        self.timerOnPriceLabel = False
        self.showRealTimeCandle = True
        self.priceScaleMinimumWidth = 90
        self.showVolume = False
        self.showToolbox = False
        self.fontSize = 14
        self.fontFamily = 'Monaco'
        self.loadTheme( theme_dark )

        # calculate the panels sizes
        self.panels = config['panels']

        # add the main panel
        if self.panels.get('main') == None:
            self.panels['main'] = {
                "position": "above",
                "width": 1.0,
                "height": 1.0,
                "fontsize": self.fontSize,
                "show_candles": True,
                "show_timescale": True,
                "show_labels": True,
                "show_priceline": True,
                "show_plotnames": True,
                "show_volume": self.showVolume,
                "background_color": self.backgroundColor,
                "text_color": self.textColor
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

        # self.backgroundColor = "#dddddfff" # "#d5d7daff"
        # self.textColor = "#2D2D2D"
        # self.priceScaleColor = "#2F3031"
        # self.priceScaleBorderColor = "#686869ff"
        # self.gridColor = "#9C9DA08C"
        # self.bullColor = "#039974"
        # self.bearColor = "#D44667"
        # self.candleBullColor = self.bullColor
        # self.candleBearColor = self.bearColor
        # self.wickBullColor = self.bullColor
        # self.wickBearColor = self.bearColor
        # self.candleBorderBullColor = self.bullColor
        # self.candleBorderBearColor = self.bearColor
        

    def writeConfig(self):
        theme = {
            'background_color':self.backgroundColor,
            'text_color':self.textColor,
            'price_scale_color':self.priceScaleColor,
            'price_scale_border_color':self.priceScaleBorderColor,
            'grid_color': self.gridColor,
            'bull_color':self.bullColor,
            'bear_color':self.bearColor,
            'candle_bull_color':self.candleBullColor,
            'candle_bear_color':self.candleBearColor,
            'wick_bull_color':self.wickBullColor,
            'wick_bear_color':self.wickBearColor,
            'candle_border_bull_color':self.candleBorderBullColor,
            'candle_border_bear_color':self.candleBorderBearColor
        }

        with open('config.json', 'w') as f:
            configDic = {
                "timer_on_price_label":self.timerOnPriceLabel,
                "show_realtime_candle":self.showRealTimeCandle,
                "price_scale_minimum_width":self.priceScaleMinimumWidth,
                "theme":theme
            }
            configString = json.dumps(configDic, indent='\t')
            f.write( configString )
            f.close()

    def read_config(self):
        try:
            with open("config.json", "r") as config_file:
                config = json.load(config_file)
                config_file.close()

            # parse the config file
            if( config.get('timer_on_price_label') != None ):
                self.timerOnPriceLabel = bool(config.get('timer_on_price_label'))
            if( config.get('show_realtime_candle') != None ):
                self.showRealTimeCandle = bool(config.get('show_realtime_candle'))
            if( config.get('price_scale_minimum_width') != None ):
                self.priceScaleMinimumWidth = int(config.get('price_scale_minimum_width'))

            if( config.get('theme') != None ):
                self.loadTheme( config['theme'] )
        except json.JSONDecodeError as e:
            print("Invalid JSON:", e)
        except FileNotFoundError:
            print("Config file not found. Creating new one")
 
        #rewrite the config file
        self.writeConfig()

    def loadTheme( self, theme ):
        self.backgroundColor = theme.get('background_color')
        self.textColor = theme.get('text_color')
        self.priceScaleColor = theme.get('price_scale_color')
        self.priceScaleBorderColor = theme.get('price_scale_border_color')
        self.gridColor = theme.get('grid_color')
        self.bullColor = theme.get('bull_color')
        self.bearColor = theme.get('bear_color')
        self.candleBullColor = theme.get('candle_bull_color')
        self.candleBearColor = theme.get('candle_bear_color')
        self.wickBullColor = theme.get('wick_bull_color')
        self.wickBearColor = theme.get('wick_bear_color')
        self.candleBorderBullColor = theme.get('candle_border_bull_color')
        self.candleBorderBearColor = theme.get('candle_border_bear_color')

    def loadChartData(self, descriptor, df):
        if debug : print( "Initializing window" )

        self.descriptor = descriptor
        self.barindex = len(df) - 1
        self.timestamp = int(df['timestamp'].iloc[-1])
        self.basetimestamp = int(df['timestamp'].iloc[0])

        window_width = 1024
        window_height = 768
        if 1:
            screen_width, screen_height = self.get_screen_resolution()
            window_width = int(screen_width * 0.65)
            window_height = int(screen_height * 0.65)
        self.panels["main"]["chart"] = chart = Chart( window_width, window_height, inner_height=self.panels["main"]["height"], inner_width=self.panels["main"]["width"], toolbox=self.showToolbox )
        chart.layout( background_color=self.backgroundColor,
                    text_color=self.textColor,
                    font_size=self.panels['main']['fontsize'],
                    font_family=self.fontFamily )
        if self.numpanels > 0 : 
            chart.time_scale( visible=False, time_visible=False )

        chart.candle_style( up_color=self.candleBullColor,
                           down_color=self.candleBearColor,
                           wick_up_color=self.wickBullColor,
                           wick_down_color=self.wickBearColor,
                           border_up_color=self.candleBorderBullColor,
                           border_down_color=self.candleBorderBearColor )
        
        # grid(vert_enabled: bool, horz_enabled: bool, color: COLOR, style: LINE_STYLE
        if self.gridColor is not None:
            chart.grid( color = self.gridColor )

        chart.price_scale(minimum_width=self.priceScaleMinimumWidth,   # FIXME: try to autoscale it
                          border_visible=True, 
                          border_color=self.priceScaleBorderColor, 
                          text_color=self.priceScaleColor, 
                          entire_text_only=False, 
                          ticks_visible=True,
                          visible=True )

        self.legend = f"{self.config['symbol']}"
        chart.legend( visible=False, ohlc=False, percent=False, font_size=self.panels['main']['fontsize']+2, text=self.legend, color=self.textColor )

        volume_alpha = 0.8 if self.panels["main"]["show_volume"] else 0.0
        r,g,b,a = hx2rgba( self.bullColor )
        upcolor = f'rgba({r},{g},{b},{volume_alpha})'
        r,g,b,a = hx2rgba( self.bearColor )
        downcolor = f'rgba({r},{g},{b},{volume_alpha})'
        chart.volume_config(
            scale_margin_top = 0.8, 
            scale_margin_bottom = 0.0,
            up_color=upcolor, 
            down_color=downcolor)
        
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
            backgroundColor = self.backgroundColor
            textColor = self.textColor
            if panel.get('background_color'):
                backgroundColor = panel.get('background_color')
            if panel.get('text_color'):
                textColor = panel.get('text_color')
            subchart.layout( background_color=backgroundColor, text_color=textColor, font_size=panel["fontsize"], font_family=self.fontFamily )
            allow_line_names = panel["show_plotnames"]
            subchart.legend( visible=False, ohlc=False, percent=False, lines = allow_line_names, font_size=14, color=textColor, text=n ) # lines info crash the script when enabled
            subchart.crosshair( horz_visible=False )
            subchart.time_scale( visible=panel["show_timescale"], time_visible=panel["show_timescale"] )
            subchart.price_line( label_visible=panel["show_labels"], line_visible=panel["show_priceline"] )
            # subchart.precision( self.bottompanel_precision )
            # subchart.price_scale(minimum_width=price_column_width)
            subchart.price_scale(minimum_width= self.priceScaleMinimumWidth,
                                 border_visible= True,
                                 border_color= self.priceScaleBorderColor,
                                 text_color=self.priceScaleColor
                                 ) # FIXME: try to autoscale it
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


    def modifyMarkers( self, modifiedList ):
        if modifiedList is None or len(modifiedList) == 0:
            return
        # we can't be surgical. We have to go brute force to avoid the sorting problems
        for m in modifiedList:
            self.removeMarker(m)
            self.addMarker(m)


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


    def reclipLine( self, line:line_c ):
        if not line.clipped:
            # this should be a ValueError
            return

        if line.x1 >= self.timestamp: # all of it is out of range, but will show up eventually
            if not( line.x1 == line.x2 and line.x1 == self.timestamp ): # if it's pure vertical it can show up
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
        x1, y1, c = self.clip_line_left(x1, y1, x2, y2, self.basetimestamp)
        x2, y2, line.clipped = self.clip_line_right(x1, y1, x2, y2, self.timestamp)
            
        chart:Chart = self.panels[line.panel]['chart']

        line.instance = chart.trend_line( 
            pd.to_datetime( int(x1), unit='ms' ), 
            y1, 
            pd.to_datetime( int(x2), unit='ms' ), 
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

    def updateLines( self, updatelist ):
        if updatelist is None or len(updatelist) == 0:
            return
        
        for linedesc in updatelist:
            id = int( linedesc.get('id') )
            for line in reversed(self.lines):
                if line.id == id:
                    line.id = int( linedesc.get('id') )
                    line.x1 = int(linedesc.get('x1'))
                    line.y1 = float( linedesc.get('y1') )
                    line.x2 = int(linedesc.get('x2'))
                    line.y2 = float( linedesc.get('y2') )
                    line.color = linedesc.get('color')
                    line.width = int( linedesc.get('width') )
                    line.style = linedesc.get('style')

                    # mark for clipping so the reclipping code updates it into the chart
                    line.clipped = True
                    if line not in self.lines_clipped:
                        self.lines_clipped.append(line)
                    break


    def removeLine( self, id ):
        for line in reversed(self.lines):
            if line.id != id:
                continue
            if line.instance != None:
                line.instance.delete()
            line.instance = None
            if line.clipped:
                self.lines_clipped.remove(line)
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
                x1 = int(msg.get('x1')),
                y1 = float( msg.get('y1') ),
                x2 = int(msg.get('x2')),
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

            if x1 >= self.timestamp: # all of it is out of range, but will show up eventually
                if not( x1 == x2 and x1 == self.timestamp ):
                    line.clipped = True
                    self.lines.append(line)
                    return
            
            if x2 < 0: # This like will never be visible
                return

            x1, y1, c = self.clip_line_left(x1, y1, x2, y2, self.basetimestamp)
            x2, y2, line.clipped = self.clip_line_right(x1, y1, x2, y2, self.timestamp)

            if line.panel not in self.panels.keys():
                line.panel = 'main'

            chart:Chart = self.panels[line.panel]['chart']

            line.instance = chart.trend_line( 
                pd.to_datetime( int(x1), unit='ms' ), 
                y1, 
                pd.to_datetime( int(x2), unit='ms' ), 
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
            print( "Exception addLine", e )



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
        self.modifyMarkers( msg['markers'].get("modified") )
        self.addMarkers( msg['markers'].get("added") ) 

        self.removeLines( msg['lines'].get("removed") )
        self.updateLines( msg['lines'].get("modified") )
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
            chart.topbar.button('legendswtich', '∇', align= 'left', func=self.button_legend_press)
            chart.topbar.menu( "Theme", ("Theme", "Black", "Dark", 'Light'), "Theme", func= self.menu_theme) 
            #chart.topbar.switcher( 'thisthat', ("this", "that"), "that" )
            # ^—–▽▼▭∆∇∨∧⋀⋁⋎⋏⩔⩡Λ


            chart.topbar.button('timerswtich', 'Timer on price', align= 'right', toggle=True, func=self.button_timerlabel_press)
            chart.topbar.textbox("timer", "--:--", align= 'right')
            
        except Exception as e:
            print( f'{e}')

    def menu_theme( self, chart:Chart ):
        
        if chart.topbar['Theme'].value == 'Black':
            self.loadTheme( theme_black )
            self.writeConfig()
            
        elif chart.topbar['Theme'].value == 'Dark':
            self.loadTheme( theme_dark )
            self.writeConfig()

        elif chart.topbar['Theme'].value == 'Light':
            self.loadTheme( theme_light )
            self.writeConfig()
            
        print ( f"{chart.topbar['Theme'].value} theme written to config. It will show next time the chart is opened" )
        chart.topbar['Theme'].value = 'Theme'
    
    async def button_timerlabel_press(self, chart):
        try:
            # 'Timer on price ▢■▭▬▮▯▩▧▦▣■□▢▥□▣'
            # timeron = 'Timer on price ▣'
            # timeroff = 'Timer on price ▢'
            # if chart.topbar['timerswtich'].value == timeron:
            #     chart.topbar['timerswtich'].set(timeroff)
            #     self.timerOnPriceLabel = False
            # elif chart.topbar['timerswtich'].value == timeroff:
            #     chart.topbar['timerswtich'].set(timeron)
            #     self.timerOnPriceLabel = True
            self.timerOnPriceLabel = not self.timerOnPriceLabel
            
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

    # Send a command to the server
    message = f"{command} {params}".strip()
    await socket.send_string(message)
    if debug : print(f"Sent command: {message}")

    # Get the reply
    reply = await socket.recv()

    # Process the reply
    try:
        try:
            data = msgpack.unpackb(reply, raw=False)
            if debug : print( "RECEIVED:", data )
        except Exception as e:
            print( f"MSGPACK ERROR: {e}" )
            print( f"DATA: {data}" )
            raise ValueError( f"MSGPACK ERROR: {e}" )
    
        if isinstance(data, dict):
            if not 'type' in data:
                raise ValueError( "ERROR [send_command]: received dictionary without a message 'type'." )
            
            if data['type'] == 'config':
                if debug : print( f"Received config message for symbol: {data['symbol']}" )
                # create the window container
                window = window_c(data)
                status = CLIENT_CONNECTED
                return
                
            elif data['type'] == 'data_descriptor':

                status = CLIENT_LOADING
                print("Loading chart", data['timeframe'] )
                
                # unpack the dataframe
                if 'data' in data and isinstance(data['data'], bytes):
                    raw_data = data['data']
                else:
                    raise ValueError( "dataframe message doesn't contain raw bytes data." )
                    
                try:   
                    # Convert raw bytes back to numpy array and reshape it to the original structure
                    array_data = np.frombuffer(raw_data, dtype=np.float64)
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

                    # clear the raw data from memory. We won't use it again
                    # FIXME? Maybe I shouldn't convert it to a dictionary here?
                    del data['data']
                    
                    if debug : print(f"DataFrame shape: {df.shape}")

                    ### fall through ###

                except Exception as e:
                    print(f"Error reconstructing DataFrame: {e}")
                    status = CLIENT_CONNECTED
                    return None
                
                # initialize the window with the dataframe and open it
                window.loadChartData( data, df )
                status = CLIENT_READY
                return
        
        if isinstance(data, str):
            if debug : print(f"Received unhandled response: {data}")
            return

    except Exception as e:
        raise ValueError( e )
    
    raise ValueError( "Unhandled reply type in 'send_command'. Type:", type(data) )


async def listen_for_updates(context):
    """Listen for updates from server"""
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://127.0.0.1:{pub_port}")  # Modified line
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    if debug : print("listening...")
    
    try:
        while True:
            try:
                message = await socket.recv()
                
                try:
                    data = msgpack.unpackb(message, raw=False)
                    if debug : print( data )
 
                    if data['type'] == 'row':
                        # Check if the message contains raw data
                        if 'data' in data and isinstance(data['data'], bytes):
                            row_bytes = data['data']
                            dtype = np.float64
                            length = len(row_bytes) // dtype().itemsize
                            row_bytes = np.frombuffer(row_bytes, dtype=dtype, count=length)
                            data['data'] = row_bytes.copy() # Replace the raw bytes with the reconstructed NumPy array
                            del row_bytes
                        
                        window.newRow(data)

                    elif data['type'] == 'tick':
                        window.newTick(data)

                except Exception as e:
                    print( f"ERROR listen_for_updates: {e}")
                
            except Exception as e:
                print(f"Error in listen_for_updates loop: {e}")
            
            await asyncio.sleep(0.01)
    
    except asyncio.CancelledError:
        print("Update listener cancelled")
    finally:
        socket.close()

async def run_client():
    global status, timeframe_request
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
                await send_command( cmd_socket, "connect", "" )
                await asyncio.sleep(0.25)
                continue

            if status == CLIENT_CONNECTED:
                if debug : print( "sending connect request for timeframe", timeframe_request )
                await send_command(cmd_socket, "dataframe", timeframe_request if timeframe_request is not None else "" )
                timeframe_request = None
                await asyncio.sleep(0.25)
                continue

            if status == CLIENT_LOADING:
                await asyncio.sleep(0.05)

            if status == CLIENT_READY:
                status = CLIENT_LISTENING
                await send_command(cmd_socket, "listening", "")
                await asyncio.sleep(0.1)

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