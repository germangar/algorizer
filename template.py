import pandas as pd
from algorizer import stream_c, timeframe_c
from algorizer import plot
import calcseries as calc
from calcseries import generatedSeries_c # for autocompletion and in case the user wants to create a new generated series
from candle import candle_c
import active
import strategy


# from window import window_c # Importing window_c is only required if you want direct access to lightweight charts



def runCloseCandle_5m( timeframe:timeframe_c, open:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series ):
     pass

def runCloseCandle_1m( timeframe:timeframe_c, open:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series ):

    sma = calc.SMA( close, 75 )
    plot( sma.name, sma.series() )

    lr = calc.LINREG( close, 300 )
    lr.plot()

    calc.RSI( close, 14 ).plot('panel')

    size = 20
    if( sma.crossingUp(lr) ):
        shortpos = strategy.getActivePosition(strategy.SHORT)
        if(shortpos and len(shortpos.order_history) > 2 ):
            strategy.close(strategy.SHORT)

        strategy.order( 'buy', strategy.LONG, timeframe.realtimeCandle.close, size )

    if calc.crossingDown( sma.series(), lr ):

        longpos = strategy.getActivePosition(strategy.LONG)
        if( longpos and len(longpos.order_history) > 2 ):
            strategy.close(strategy.LONG)

        strategy.order( 'sell', strategy.SHORT, timeframe.realtimeCandle.close, size )



if __name__ == '__main__':

    # start the candles stream:
    # - The symbol is in CCXT format. If the final :USDT is not included it opens a spot stream, if included it opens a perpetual contracts stream
    # - timeframes:
    #   It's a list of timeframes you want to run. The order in the list will determine the order of execution of
    #   their 'runCloseCandle' function callbacks. You should create a runCloseCandle_{timeframe} function for each 
    #   timeframe you want to use in your strategy. The smallest timeframe will be used for fetching the price updates.
    #   So, if you want to check, let's say, the 4h rsi in a 30m strategy you should add 4h first in the list and then 30m.

    stream = stream_c( 'LDO/USDT:USDT', 'bitget', ['1m'], [runCloseCandle_1m], 5000 )

    # Call only if you want to open the chart window. It's not needed to run the algo
    stream.createWindow( '1m' )

    stream.run()