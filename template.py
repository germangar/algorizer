import pandas as pd
from algorizer import stream_c, timeframe_c, plot, requestValue
import calcseries as calc
from calcseries import generatedSeries_c
from candle import candle_c
import trade

# from window import window_c # Importing window_c is only required if you want direct access to lightweight charts


# 
#   RUNNING THE ALGO
# 

rsi30m = None
def runCloseCandle_30m( timeframe:timeframe_c, open:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series ):
    global rsi30m
    rsi30m = calc.RSI(close, 14)
    rsi30m.plot('panel')


def runCloseCandle_1m( timeframe:timeframe_c, open:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series ):

    sma = calc.SMA( close, 75 )
    plot( sma.series(), sma.name )

    lr = calc.LINREG( close, 300 )
    lr.plot()

    rsi30min = requestValue( rsi30m.name, '30m' )
    plot( rsi30min, 'rsi30m', 'panel' )

    shortpos = trade.getActivePosition(trade.SHORT)
    if shortpos :
        pnl = shortpos.get_unrealized_pnl_percentage()
        if pnl < -0.5 or pnl > 10.0:
            trade.close(trade.SHORT)

    if( sma.crossingUp(lr) ):
        if(shortpos and len(shortpos.order_history) > 1 ):
            trade.close(trade.SHORT)
        if rsi30min < 45:
            trade.order( 'buy', trade.LONG )

    longpos = trade.getActivePosition(trade.LONG)
    if longpos and longpos.get_unrealized_pnl_percentage()  < -0.5:
        trade.close(trade.LONG)

    if calc.crossingDown( sma.series(), lr ):
        if( longpos and len(longpos.order_history) > 1 ):
            trade.close(trade.LONG)

        if rsi30min > 55 :
            trade.order( 'sell', trade.SHORT )



# 
#   SETTING UP THE CANDLES FEED
# 


if __name__ == '__main__':

    # configure the strategy before creating the stream
    trade.strategy.hedged = False
    trade.strategy.order_size = 100 # set to smaller than max_position_size for pyramiding
    trade.strategy.max_position_size = 100

    # Start the candles stream:
    #
    # - The symbol in CCXT format. ('BTC/USDT' means spot, 'BTC/USDT:USDT' means futures PERP)
    #
    # - The exchange to connect to. It must be a exchange supported by the CCXT library https://github.com/ccxt/ccxt?tab=readme-ov-file#certified-cryptocurrency-exchanges
    #   Not all exchanges provide historic data to fetch.
    #
    # - timeframes list:
    #   It's a list of timeframes you want to run. The order in the list will determine the order of execution of
    #   their 'runCloseCandle' function callbacks. The smallest timeframe will be used for fetching the price updates.
    #   So, if you want to check, let's say, the 4h rsi in a 30m strategy you should add 4h first in the list and then 30m.
    #
    # - Callbacks list:
    #   The functions that will be called when each timeframe closes a candle. That's where the heart of your algo resides.
    #
    # - Amount of history candles *from the last timeframe in the list* to calculate. The other timeframes will adjust to it.

    stream = stream_c( 'LDO/USDT:USDT', 'bitget', ['30m', '1m'], [runCloseCandle_30m, runCloseCandle_1m], 10000 )

    print( stream.timeframes[stream.timeframeFetch].df )
    print( stream.timeframes['30m'].df )

    # trade.print_strategy_stats()
    trade.print_summary_stats()

    # Call only if you want to open the chart window. It's not needed to run the algo
    stream.createWindow( '1m' )

    stream.run()