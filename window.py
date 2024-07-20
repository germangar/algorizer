
import pandas as pd
from lightweight_charts import Chart
# from algorizer import registerTask
import tasks

SHOW_VOLUME = False


class window_c:
    def __init__( self, stream, precision = 4, bottompanel_precision = 2 ):
        self.stream = None
        self.chart = None
        self.bottomPanel = None
        self.precision = precision
        self.bottompanel_precision = bottompanel_precision

        if( stream == None ): raise SystemError( "Attempted to create a window without a stream" )
        self.stream = stream
        
        self.chart = chart = Chart( inner_height=0.8, toolbox = False )
        if( self.chart == None ): raise SystemError( "Failed to create chart" )
        chart.legend( visible=True, ohlc=False, percent=False, font_size=18, text=stream.symbol + ' - ' + stream.timeframeStr + ' - ' + stream.exchange.id + ' - ' + f'candles:{len(stream.df)}' )
        chart.time_scale( visible=False )
        chart.layout( font_size=14 )
        chart.precision( self.precision )
        chart.topbar.button('my_button', 'Off', func=on_button_press)

        tmpdf = pd.DataFrame( { 'time':pd.to_datetime( stream.df['timestamp'], unit='ms' ), 'open':stream.df['open'], 'high':stream.df['high'], 'low':stream.df['low'], 'close':stream.df['close']} )
        if( SHOW_VOLUME ):
            tmpdf['volume'] = stream.df['volume']

        chart.set( tmpdf )

        # bottom panel
        self.bottomPanel = bottomPanel = chart.create_subchart( position='bottom', width=1.0, height=0.2, sync=chart.id )
        if( self.bottomPanel == None ): raise SystemError( "Failed to create bottomPanel" )
        bottomPanel.legend( visible=True, ohlc=False, percent=False, font_size=18 )
        bottomPanel.time_scale( visible=True, time_visible=True )
        bottomPanel.crosshair( horz_visible=False )
        bottomPanel.price_line( label_visible=False, line_visible=False )
        bottomPanel.layout( font_size=14 )
        bottomPanel.precision( self.bottompanel_precision )
        bottomPanel.set(tmpdf)
        bottomPanel.hide_data()

        for marker in stream.markers:
            if( marker.chart == None ):
                marker.chart = self.bottomPanel if marker.chartName == 'panel' else self.chart
            marker.refreshInChart()

        tasks.registerTask( self.chart.show_async() )
        

    def updateChart( self ):
        if( self.stream == None or self.chart == None ): 
            return
        #update the chart
        df = self.stream.df
        data_dict = {'time': pd.to_datetime( df['timestamp'].iloc[-1], unit='ms' ), 'open': df['open'].iloc[-1], 'high': df['high'].iloc[-1], 'low': df['low'].iloc[-1], 'close': df['close'].iloc[-1] }
        if SHOW_VOLUME:
            data_dict['volume'] = df['volume'].iloc[-1]

        self.chart.update( pd.Series(data_dict) )
        if( self.bottomPanel != None ):
            self.bottomPanel.update( pd.Series(data_dict) )


def createWindow( stream )->window_c:
    return window_c( stream )


async def on_button_press(chart):
    new_button_value = 'On' if chart.topbar['my_button'].value == 'Off' else 'Off'
    chart.topbar['my_button'].set(new_button_value)
    print(f'Turned something {new_button_value.lower()}.')