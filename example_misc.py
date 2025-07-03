from framework import calc, trade
from framework import stream_c, timeframe_c, generatedSeries_c, candle_c, pivots_c, pivot_c, c
from framework import plot, histogram, requestValue, createMarker, createLine


# The 'event' function is called by the engine when something happens 
# the user may want to interact with. It's main purpose is to handle 'broker_event',
# but the users can create their own console commands.

def event( stream:stream_c, event:str, param, numparams ):
    if event == "cli_command":
        assert( isinstance(param, tuple) and len(param) == numparams)
        cmd, args = param
        if cmd == 'echo': # command will always be lower case
            print( 'Echo ', args )

    elif event == "tick":
        assert( isinstance(param, tuple) and len(param) == numparams)
        '''
        candle : a cancle_c containing the OHLCV values of the latest price.
        realtime : boolean. True for realtime, false for backtesting.
        '''
        candle, realtime = param
        if not realtime:
            return

    # This event will be called when the strategy executes an order in real time. Not when backtesting.
    elif event == "broker_event":
        assert( isinstance(param, tuple) and len(param) == numparams)
        '''
        order_type (Buy/Sell): represented as the constants c.LONG (1) and c.SHORT (-1)
        quantity (Order Quantity in Base Currency): The exact amount of the base asset (e.g., 0.01 BTC).
        quantity_dollars (Order Quantity in Dollars): The notional value of the current order in USD (e.g., if you buy 0.001 BTC at $60,000, this would be $60).
        position_type (New Position Type: Long/Short/Flat)
        position_size_base (New Position Size in Base Currency): The total quantity of the base asset currently held (signed for long/short).
        position_size_dollars (New Position Size in Dollars, Leveraged): This represents the total notional exposure of the position, including the effect of leverage.
        leverage (Leverage of the Order)
        position_collateral_dollars (Un-leveraged Capital in Position)
        '''
        import requests
        
        order_type, quantity, quantity_dollars, position_type, position_size_base, position_size_dollars, position_collateral_dollars, leverage = param

        # this is an example of an alert for my webhook 'whook': https://github.com/germangar/whook
        account = "blabla"
        url = 'https://webhook.site/ae09b310-eab0-4086-a0d1-2da80ab722d1'
        if position_size_base == 0:
            message = f"{account} {stream.symbol} close"
        else:
            order = 'buy' if order_type == c.LONG else 'sell'
            message = f"{account} {stream.symbol} {order} {quantity_dollars:.4f}$ {leverage}x"
        if url:
            req = requests.post( url, data=message.encode('utf-8'), headers={'Content-Type': 'text/plain; charset=utf-8'} )

        

# 
# 
#   RUNNING THE ALGO
# 
# 


# The tick callback funtion will be called for every price update. The candle will not be closed.
def tick( realtimeCandle:candle_c ):
    pass


# User defined closeCandle callbacks. They are called when a candle of the given frametime has closed.
# You can define one for each timeframe, or not. They can be set to None. It's up to you.


rsiSlow = None
def runCloseCandle_slow( timeframe:timeframe_c, open, high, low, close, volume, top, bottom ):
    global rsiSlow

    # calc is where the generatedSeries operations reside. 
    # You can rename it to 'ta' if you like. It's the equivalent. I just like 'calc' better.
    rsiSlow = calc.IFTrsi(close, 8)

    # This plot will only show when charting this timeframe.
    # the main logic resides in the 'fast' timeframe, so you most likely will never see it unless
    # you want to check this timeframe.
    rsiSlow.plot('rsi')


def runCloseCandle_fast( timeframe:timeframe_c, open, high, low, close, volume, top, bottom ):
    barindex = timeframe.barindex # for simplicity

    # bollinger bands returns 3 generatedSeries
    BBbasis, BBupper, BBlower = calc.BollingerBands( close, 350 )
    # You can plot generatedSeries directly with their plot method.
    BBbasis.plot( color = "#769EB4AC", width=2 )
    BBupper.plot( style='dotted' )
    BBlower.plot( style='dotted' )

    # I didn't add horizontal lines yet
    plot( 80, 'overbought', 'rsi', color="#CECECE8B", style='dotted', width=1 )
    plot( 20, 'oversold', 'rsi', color="#CECECE8B", style='dotted', width=1 )

    # can request values from a different timeframe with this function
    # invRSI = requestValue( rsiSlow.name, '1d' ) 

    # but in this case it's easier to use the generatedSeries object to retrieve it.
    # You can read its value here, but not operate with the object nor use the plot method
    # since this object belongs to a different timeframe.
    
    # These all do the same, but negative indexing should only be used in the iloc() method:
    invRSI = rsiSlow[barindex]
    # invRSI = rsiSlow.current()
    # invRSI = rsiSlow.iloc(barindex)
    # invRSI = rsiSlow.iloc(-1)
    #invRSI = rsiSlow.series()[barindex]

    # We convert the -1/+1 value to the scale of standard rsi so they can share the same panel.
    if invRSI is not None:
        invRSI = (invRSI * 50) + 50

    # we can plot a float (rsiSlow.current() is a float) using the plot function 
    # we can't directly plot the rsiSlow object/series-method as it belongs to a different timeframe and the lenght of the dataframes don't match
    plot( invRSI, 'rsiSlow', 'rsi', color="#ef38cd44", width=10 ) # The rsi panel was created by us. See below.

    # standard rsi
    rsi14 = calc.RSI(close, 14).plot( 'rsi' )

    # There's a built-in pivot indicator which is performance savy.
    # 'top' and 'bottom' are columns in the dataframe containing
    # the top and bottom prices of the candles bodies. Wicks excluded.
    # You can use high and low instead, or whatever you prefer.
    # 'Amplitude' is the main value you want to tweak for each symbol/timeframe
    pivots = calc.pivots( top, bottom, 11 )
    if pivots.isNewPivot:
        thisPivot = pivots.getLast() # only confirmed pivots. You can check the WIP pivot values in pivots.temp_pivot
        if thisPivot.type == c.PIVOT_HIGH:
            createMarker('▽', 'above', color = "#BDBDBD", timestamp=thisPivot.timestamp)
        else:
            createMarker('△', 'below', color = "#BDBDBD", timestamp=thisPivot.timestamp)



    # MACD in one go
    macd_line, signal_line, histo = calc.MACD(close)
    histo.histogram( 'macd', "#4A545D" )
    macd_line.plot( 'macd', color = "#AB1212", width=2 ) # The macd panel was created by us. See below
    signal_line.plot( 'macd', color = "#1BC573" )
    # print( signal_line.current() )

    
    # trading logic 
    # buySignal = calc.crossingUp( close, BBlower ) and invRSI < 30
    # sellSignal = calc.crossingDown( close, BBupper ) and invRSI > 70

    
    # same thing using methods
    buySignal = rsi14[barindex] > 50.0 and BBlower.crossingDown(close) and invRSI < 20
    sellSignal = rsi14[barindex] < 50.0 and BBupper.crossingUp(close) and invRSI > 80

    # accesing positions and making orders
    shortpos = trade.getActivePosition(c.SHORT)
    longpos = trade.getActivePosition(c.LONG)

    if buySignal:
        if shortpos is not None:
            trade.close(c.SHORT)
        offset = 50
        if longpos:
            lastorderindex = longpos.get_order_by_direction(c.LONG)['barindex']
            offset = barindex - lastorderindex
        if offset > 40:
            trade.order( 'buy', c.LONG )

    if longpos:
        if longpos.get_unrealized_pnl_percentage() > 75 and longpos.collateral >= trade.strategy.order_size * 1.9:
            trade.order( 'sell', c.LONG )

    if sellSignal:
        if longpos is not None:
            trade.close(c.LONG)
        offset = 50
        if shortpos:
            lastorderindex = shortpos.get_order_by_direction(c.SHORT)['barindex']
            offset = barindex - lastorderindex
        if offset > 40:
            trade.order( 'sell', c.SHORT )

    if shortpos:
        if shortpos.get_unrealized_pnl_percentage() > 75 and shortpos.collateral >= trade.strategy.order_size * 1.9:
            trade.order( 'buy', c.SHORT )

    

    


# 
#   SETTING UP THE CANDLES FEED
# 


if __name__ == '__main__':

    # strategy configuration.
    # order_size will be used when a buy/sell order is provided without a quantity.
    # The strategy will execute the orders until the position reaches max_position_size.
    # max_position_size is not the total liquidity. That is initial_liquidity.
    # max_position_size is the max ammount you want to expose in a position.
    # The stats present two pnls. One is calculated against max_position_size and the other against initial_liquidity.
    # currency_mode 'USD' or 'BASE' will change the calculation of your orders to use USD or the base currency.

    # configure the strategy before creating the stream
    trade.strategy.verbose = False
    trade.strategy.hedged = False
    trade.strategy.currency_mode = 'USD'
    trade.strategy.order_size = 1000
    trade.strategy.max_position_size = 3000 # allow pyramiding of 3 orders
    trade.strategy.leverage_long = 5
    trade.strategy.leverage_short = 5
    
    #   Create the candles stream:
    #
    # - symbol: 
    #   The symbol in CCXT format. ('BTC/USDT' means spot, 'BTC/USDT:USDT' means perpetual USDT contracts)
    #
    # - exchange:
    #   It must be a exchange supported by the CCXT library https://github.com/ccxt/ccxt?tab=readme-ov-file#certified-cryptocurrency-exchanges
    #   Not all exchanges provide historic data to fetch. These are some good data providers (tested with Bitcoin only):
    #   PERP: Bybit, kucoin, okx, binance, htx, poloniexfutures
    #   SPOT: gate, kucoin, okx, binance, probit, upbit
    #
    # - timeframes list: <--- IMPORTANT
    #   It's a list of timeframes you want to run. The order in the list will determine the order of execution of
    #   their 'closeCandle' function callbacks. If you want to read data from a bigger timeframe you should 
    #   add the bigger one before in the list.
    #   The smallest timeframe will be used for fetching the price updates from the exchange.
    #   
    # - Callbacks list: <--- IMPORTANT
    #   The 'closeCandle' functions that will be called when each timeframe closes a candle.
    #   These are where the heart of your algo resides.
    #
    # - event_callback: 
    #   Funtion to be called when an event happens that the user could interpret.
    #   The tick event is called when a price update happens (realtime candle)
    #
    # - max_amount:
    #   Amount of history candles to fetch and backtest. These candles refer to the last
    #   timeframe in the list of timeframes. The other timeframes will adjust to it.
    #
    # - cache_only:
    #   Use the candle datas in cache without trying to fetch new candles to update it


    stream = stream_c( 'BTC/USDT:USDT', 'bitget', ['4h', '1h'], [runCloseCandle_slow, runCloseCandle_fast], event, 35000 )

    # Create subpanels to plot the oscilators.
    # with and height are values between 0 amd 1, representing the percentage of the
    # screen the subpanel will take. The order of creation will determine which one
    # is above or below.
    stream.registerPanel('macd', 1.0, 0.1, show_timescale=False ) # usually you only want the one at the bottom to show the timescale
    stream.registerPanel('rsi', 1.0, 0.2 )

    # Some options to print the results. The first one prints all orders so it's disabled for being spammy
    # trade.print_strategy_stats()
    trade.print_summary_stats()
    trade.print_pnl_by_period_summary()


    # Execute this call only if you want to check the chart. It's not neccesary to run the strategy.
    # You can also open the chart using the command 'chart' in the console.
    stream.createWindow( '1h' )

    # Execute this call only if you want the strategy to keep running in realtime.
    # It's not neccesary if you only want a backtest.
    stream.run()