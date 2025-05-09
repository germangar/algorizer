
import pandas as pd
import tasks
import asyncio
from lightweight_charts import Chart
# from lightweight_charts_esistjosh import Chart

from datetime import datetime
from datetime import timedelta


SHOW_VOLUME = False


class window_c:
    def __init__( self, timeframe, precision = 4, bottompanel_precision = 2 ):
        self.timeframe = timeframe
        self.chart = None
        self.bottomPanel = None
        self.precision = precision
        self.bottompanel_precision = bottompanel_precision
        self.legendStr = ''

        if( self.timeframe == None ): raise SystemError( "Attempted to create a window without a stream" )
        
        
        self.chart = chart = Chart( 1024, 768, inner_height=0.8, toolbox = False )
        if( self.chart == None ): raise SystemError( "Failed to create chart" )
        self.legendStr = f"{timeframe.stream.symbol} - {timeframe.timeframeStr} - {timeframe.stream.exchange.id} - candles:{len(timeframe.df)}"
        chart.legend( visible=True, ohlc=False, percent=False, font_size=18, text=self.legendStr )
        chart.time_scale( visible=False )
        chart.layout( font_size=14 )
        chart.precision( self.precision )
        # chart.topbar.button('my_button', 'Off', func=on_button_press)
        
        volume_alpha = 0.8 if SHOW_VOLUME else 0.0
        chart.volume_config(
            scale_margin_top = 0.8, 
            scale_margin_bottom = 0.0, 
            up_color=f'rgba(83,141,131,{volume_alpha})', 
            down_color=f'rgba(200,127,130,{volume_alpha})')

        tmpdf = pd.DataFrame( { 'time':pd.to_datetime( timeframe.df['timestamp'], unit='ms' ), 'open':timeframe.df['open'], 'high':timeframe.df['high'], 'low':timeframe.df['low'], 'close':timeframe.df['close'], 'volume': timeframe.df['volume']} )

        chart.set( tmpdf )


        # bottom panel
        self.bottomPanel = bottomPanel = chart.create_subchart( position='bottom', width=1.0, height=0.2, sync=chart.id )
        if( self.bottomPanel == None ): raise SystemError( "Failed to create bottomPanel" )
        bottomPanel.legend( visible=True, ohlc=False, percent=False, lines = False, font_size=18 ) # lines info crash the script when enabled
        bottomPanel.time_scale( visible=True, time_visible=True )
        bottomPanel.crosshair( horz_visible=False )
        bottomPanel.price_line( label_visible=False, line_visible=False )
        bottomPanel.layout( font_size=14 )
        bottomPanel.precision( self.bottompanel_precision )
        bottomPanel.set(tmpdf)
        bottomPanel.hide_data()

        tempdf = []

        for marker in timeframe.stream.markers:
            if( marker.chart == None ):
                marker.chart = self.bottomPanel if marker.chartName == 'panel' else self.chart
            marker.refreshInChart()

        self.task = self.chart.show_async()
        tasks.registerTask( 'window', self.task )
        
    def destroyWindow( self ):
        self.chart = None
        self.bottomPanel = None

    def updateChart( self, timeframe ):
        if( self.chart == None ): 
            return
        if( self.timeframe != timeframe ):
            return
        if not self.chart.is_alive : # this is false when there's no window. Clean up
            if tasks.findTask( 'window' ) :
                self.destroyWindow()
                return
            timeframe.window = None
            return
        
        #update the chart
        data_dict = {'time': pd.to_datetime( timeframe.realtimeCandle.timestamp, unit='ms' ), 
                     'open': timeframe.realtimeCandle.open, 
                     'high': timeframe.realtimeCandle.high, 
                     'low': timeframe.realtimeCandle.low, 
                     'close': timeframe.realtimeCandle.close, 
                     'volume': timeframe.realtimeCandle.volume }
        
        series = pd.Series(data_dict)
        
        self.chart.update( series )
        if( self.bottomPanel != None ):
            # self.bottomPanel.legend( lines = False ) # It crashes if lines are enabled
            self.bottomPanel.update( series )

    def updateClock( self ):
        timeframe = self.timeframe
        string1 = self.legendStr + " - " + timeframe.realtimeCandle.remainingTimeStr()
        self.chart.legend( visible=True, ohlc=False, percent=False, font_size=18, text=string1 )


async def on_button_press(chart):
    new_button_value = 'On' if chart.topbar['my_button'].value == 'Off' else 'Off'
    chart.topbar['my_button'].set(new_button_value)
    print(f'Turned something {new_button_value.lower()}.')


