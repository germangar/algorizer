
import pandas as pd
import tasks
from lightweight_charts import Chart
# from lightweight_charts_esistjosh import Chart

SHOW_VOLUME = False


class window_c:
    def __init__( self, timeframe, precision = 4, bottompanel_precision = 2 ):
        self.timeframe = timeframe
        self.chart = None
        self.bottomPanel = None
        self.precision = precision
        self.bottompanel_precision = bottompanel_precision

        if( self.timeframe == None ): raise SystemError( "Attempted to create a window without a stream" )
        
        
        self.chart = chart = Chart( inner_height=0.8, toolbox = False )
        if( self.chart == None ): raise SystemError( "Failed to create chart" )
        chart.legend( visible=True, ohlc=False, percent=False, font_size=18, text=timeframe.stream.symbol + ' - ' + timeframe.timeframeStr + ' - ' + timeframe.stream.exchange.id + ' - ' + f'candles:{len(timeframe.df)}' )
        chart.time_scale( visible=False )
        chart.layout( font_size=14 )
        chart.precision( self.precision )
        chart.topbar.button('my_button', 'Off', func=on_button_press)
        
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
        df = timeframe.df
        data_dict = {'time': pd.to_datetime( df['timestamp'].iloc[-1], unit='ms' ), 'open': df['open'].iloc[-1], 'high': df['high'].iloc[-1], 'low': df['low'].iloc[-1], 'close': df['close'].iloc[-1], 'volume': df['volume'].iloc[-1] }

        print(data_dict)
        self.chart.update( pd.Series(data_dict) )
        if( self.bottomPanel != None ):
            self.bottomPanel.legend( lines = False ) # It crashes if lines are enabled
            self.bottomPanel.update( pd.Series(data_dict) )


async def on_button_press(chart):
    new_button_value = 'On' if chart.topbar['my_button'].value == 'Off' else 'Off'
    chart.topbar['my_button'].set(new_button_value)
    print(f'Turned something {new_button_value.lower()}.')


