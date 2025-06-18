import pandas as pd
from constants import c
from algorizer import stream_c, timeframe_c, plot, histogram, requestValue, createMarker
import calcseries as calc
from calcseries import generatedSeries_c
from candle import candle_c
import trade
import broker_alerts as broker


# the strategy code has issued an order. When running in realtime it will call this
# function so a real order can be sent to the broker. The type of broker is up to
# the user to import and configure and issue the order here for it. In this case we
# send an alert with broker_alerts
def broker_event( stream:stream_c, type:int, quantity, quantity_dollars, position_type, position_size_base, position_size_dollars, position_collateral_dollars, leverage ):
        '''
        type (Buy/Sell Event): represented as the constants c.LONG (1) and c.SHORT (-1)
        quantity (Order Quantity in Base Currency): The exact amount of the base asset (e.g., 0.001 BTC).
        quantity_dollars (Order Quantity in Dollars): The notional value of the current order in USD (e.g., if you buy 0.001 BTC at $60,000, this would be $60).
        position_type (New Position Type: Long/Short/Flat)
        position_size_base (New Position Size in Base Currency): The total quantity of the base asset currently held (signed for long/short).
        position_size_dollars (New Position Size in Dollars, Leveraged): This represents the total notional exposure of the position, including the effect of leverage.
        leverage (Leverage of the Order)
        position_collateral_dollars (Un-leveraged Capital in Position)
        '''
        exchange = "blabla"
        message = f"{exchange} {stream.symbol} pos {position_collateral_dollars:.4f}$ {leverage}x"
        if( broker.alert( message, 'https://webhook.site/ae09b310-eab0-4086-a0d1-2da80ab722d1' ) ):
            print( 'Alert sent:', message )



# 
#   RUNNING THE ALGO
# 

rsi30m = None
def runCloseCandle_slow( timeframe:timeframe_c, open:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series, volume:pd.Series ):
    global rsi30m
    rsi30m = calc.IFTrsi(close, 14)
    rsi30m.plot('rsi')

    


def runCloseCandle_fast( timeframe:timeframe_c, open:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series, volume:pd.Series ):

    # bollinger bands
    BBbasis, BBupper, BBlower = calc.BollingerBands( close, 250 )
    BBbasis.plot( color = "#769EB4AC", width=2 )
    BBupper.plot( style='dotted' )
    BBlower.plot( style='dotted' )

    rsi14 = calc.RSI(close, 14)
    rsiSlow = requestValue( rsi30m.name, '30m' )
    plot( rsiSlow, 'rsiSlow', 'rsi' ) # plotting a float takes a huge dent on performance


    buySignal = rsi14 > 50.0 and calc.crossingUp( close, BBlower ) and rsiSlow < -0.7
    sellSignal = rsi14 < 50.0 and calc.crossingDown( close, BBupper ) and rsiSlow > 0.65
    # buySignal = rsi14 > 50.0 and BBlower.crossingDown(close) and rsiSlow < -0.7
    # sellSignal = rsi14 < 50.0 and BBupper.crossingUp(close) and rsiSlow > 0.65

    shortpos = trade.getActivePosition(c.SHORT)
    longpos = trade.getActivePosition(c.LONG)

    if buySignal:
        if shortpos is not None:
            trade.close(c.SHORT)
        offset = 50
        if longpos:
            lastorderindex = longpos.get_order_by_direction(c.LONG)['barindex']
            offset = timeframe.barindex - lastorderindex
        if offset > 40:
            trade.order( 'buy', c.LONG )

    if sellSignal:
        if longpos is not None:
            trade.close(c.LONG)
        offset = 50
        if shortpos:
            lastorderindex = shortpos.get_order_by_direction(c.SHORT)['barindex']
            offset = timeframe.barindex - lastorderindex
        if offset > 40:
            trade.order( 'sell', c.SHORT )

    pivots = calc.pivots( timeframe.df['top'], timeframe.df['bottom'], 4 )
    if pivots.isNewPivot:
        thisPivot = pivots.getLast()
        if thisPivot.type == c.PIVOT_HIGH:
            createMarker('▽', 'above', color = "#BDBDBD", timestamp=thisPivot.timestamp)
        else:
            createMarker('△', 'below', color = "#BDBDBD", timestamp=thisPivot.timestamp)

    macd_line, signal_line, histo = calc.MACD(close)
    histo.histogram( 'macd', "#4A545D" )
    macd_line.plot( 'macd', color = "#AB1212", width=2 )
    signal_line.plot( 'macd', color = "#1BC573" )
    



# 
#   SETTING UP THE CANDLES FEED
# 


if __name__ == '__main__':

    # configure the strategy before creating the stream
    trade.strategy.verbose = False
    trade.strategy.hedged = False
    trade.strategy.currency_mode = 'USD'
    trade.strategy.order_size = 1000 # should allow only pyramiding of 5 orders
    trade.strategy.max_position_size = 1000
    trade.strategy.leverage_long = 1
    trade.strategy.leverage_short = 1
    
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

    stream = stream_c( 'LDO/USDT:USDT', 'bybit', ['30m', '1m'], [runCloseCandle_slow, runCloseCandle_fast], broker_event, 25000, plots = True )

    # trade.print_strategy_stats()
    trade.print_summary_stats()
    trade.print_pnl_by_period_summary()

    # print(stream.timeframes[stream.timeframeFetch].df.columns)

    stream.registerPanel('macd', 1.0, 0.15, show_timescale=True )
    stream.registerPanel('rsi', 1.0, 0.1 )

    # stream.createWindow( '1m' )

    stream.run(True)