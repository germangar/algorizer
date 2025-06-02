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

    # bollinger bands
    BBlen = 350
    BBmult = 2.0
    BBbasis = calc.SMA(close, BBlen)
    BBdev = BBmult * calc.STDEV(close, BBlen)
    BBupper = BBbasis + BBdev
    BBlower = BBbasis - BBdev
    plot( BBupper, "BBupper" )
    plot( BBlower, "BBlower" )
    BBbasis.plot()

    rsi14 = calc.RSI(close, 14)
    rsi30min = requestValue( rsi30m.name, '30m' )
    plot( rsi30min, 'rsi30m', 'panel' )

    buySignal = rsi14 > 50.0 and calc.crossingUp( close, BBlower ) and rsi30min < 35
    sellSignal = rsi14 < 50.0 and calc.crossingDown( close, BBupper ) and rsi30min > 60

    shortpos = trade.getActivePosition(trade.SHORT)
    longpos = trade.getActivePosition(trade.LONG)

    if buySignal:
        if shortpos is not None:
            trade.close(trade.SHORT)

        trade.order( 'buy', trade.LONG )

    if sellSignal:
        if longpos is not None:
            trade.close(trade.LONG)

        trade.order( 'sell', trade.SHORT )
        



# 
#   SETTING UP THE CANDLES FEED
# 


if __name__ == '__main__':

    # configure the strategy before creating the stream
    trade.strategy.hedged = False
    trade.strategy.currency_mode = 'USD'
    trade.strategy.order_size = 1000 # should allow only pyramiding of 5 orders
    trade.strategy.max_position_size = 5000
    

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
    #
    # - noplots: Disables the plots so processing the script is much faster. For when backtesting large dataframes and only interested in the results.

    stream = stream_c( 'BTC/USDT:USDT', 'bybit', ['30m', '1m'], [runCloseCandle_30m, runCloseCandle_1m], 100000, False )

    # trade.print_strategy_stats()
    trade.print_summary_stats()
    trade.print_pnl_by_period_summary()

    # Call only if you want to open the chart window. It's not needed to run the algo
    stream.createWindow( '1m' )

    stream.run()