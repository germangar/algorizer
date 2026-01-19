from algorizer import ta
from algorizer import trade
from algorizer import constants as c
from algorizer import stream_c, timeframe_c, generatedSeries_c, marker_c, line_c, candle_c
# from algorizer import pivots_c, pivot_c
from algorizer import plot, histogram, createMarker, removeMarker, createLine, removeLine


# The 'event' function is called by the engine when something happens 
# the user may want to interact with. It's main purpose is to handle 'broker_event',
# but the users can create their own console commands.

def event( stream:stream_c, event:str, param, numparams ):
    if event == "cli_command":
        assert( isinstance(param, tuple) and len(param) == numparams)
        cmd, args = param
        # command "chart [timeframe] opens the chart window"
        if cmd == 'chart' or cmd == 'c':
            stream.createWindow( args )
            
    elif event == "tick":
        '''
        candle : a cancle_c containing the OHLCV values of the latest price.
        '''
        if not stream.running: # it's backtesting
            return
        
        candle = param

        ## Show remaining candle time and open position info on the console status line.
        candle.updateRemainingTime()
        message = f"{stream.symbol.split(':')[0]} [{candle.remainingTimeStr()}]"
        longpos = trade.getActivePosition(c.LONG)
        if longpos:
            moreMsg = f" long:{longpos.collateral:.1f}({longpos.get_unrealized_pnl_percentage():.1f}%)"
            message += moreMsg
        shortpos = trade.getActivePosition(c.SHORT)
        if shortpos:
            moreMsg = f" short:{shortpos.collateral:.1f}({shortpos.get_unrealized_pnl_percentage():.1f}%)"
            message += moreMsg
        stream.setStatusLineMsg( message )
        return

    # This event will be called when the strategy executes an order in real time. Not when backtesting.
    elif event == "broker_event":
        '''
        info = {
                "order_type": order_type,                           # c.BUY / c.SELL (1/-1)
                "order_quantity": quantity,                         # Notional. Already scaled by leverage
                "order_quantity_dollars": quantity_dollars,         # Notional. Already scaled by leverage
                "position_type": self.type,                         # c.LONG / c.SHORT (1/-1)
                "position_size": position_size_base,                # Notional. Already scaled by leverage
                "position_size_dollars": position_size_dollars,     # Notional. Already scaled by leverage
                "leverage": leverage,
                "price": price,
                "source": [ 'order', 'liquidation_trigger', 'takeprofit_trigger', 'stoploss_trigger', 'takeprofit_create', 'stoploss_create' ]
            }
        '''
        if not stream.running: # is backtesting
            return
        
        source = param['source']
        if source == 'takeprofit_create' or source == 'stoploss_create':
            # These are meant to create a stoploss or takeprofit in the exchange. Not to execute a market order.
            # When the takeprofit, stoploss and liquidation are triggered there will also be an event ('*_trigger')
            # and in that case the orders will be sent unless we choose to filter them out.
            return
        
        order_type = param['order_type']
        leverage = param['leverage']
        position_size = param['position_size'] * param['position_type'] # added directionality for no reason
        order_quantity_dollars = param['order_quantity_dollars'] / leverage # whook by default expects unleveraged collateral in the order


        # this is an example of an alert for my webhook 'whook': https://github.com/germangar/whook
        account = "blabla"
        url = 'https://webhook.site/ae09b310-eab0-4086-a0d1-2da80ab722d1'
        if position_size == 0:
            message = f"{account} {stream.symbol} close"
        else:
            order = 'buy' if order_type == c.LONG else 'sell'
            message = f"{account} {stream.symbol} {order} {order_quantity_dollars:.4f}$ {leverage}x"
        if url:
            import requests
            req = requests.post( url, data=message.encode('utf-8'), headers={'Content-Type': 'text/plain; charset=utf-8'} )

        

# 
# 
#   RUNNING THE ALGO
# 
# 

rsiSlow = None

# User defined closeCandle callbacks. They are called when a candle of the given frametime has closed.
# You can define one for each timeframe, or not. They can be set to None. It's up to you.


def runCloseCandle_slow( timeframe:timeframe_c, open, high, low, close, volume, top, bottom ):
    global rsiSlow

    rsiSlow = ta.IFTrsi(close, 8)

    # This plot will only show when charting this timeframe.
    # the main logic resides in the 'fast' timeframe, so you most likely will never see it unless
    # you want to check this timeframe.
    rsiSlow.plot('subpanel')


def runCloseCandle_fast( timeframe:timeframe_c, open, high, low, close, volume, top, bottom ):

    barindex = timeframe.barindex # for simplicity

    # There's a ta.BollingerBands function, but I'm doin'g it this
    # way to demonstrate you can freely operate with series.
    mult = 2.0
    BBbasis = ta.SMA(close, 350)
    stdev = ta.STDEV(close, 350, 1.0)     # You can feed the multiplier in the STDEV function directly
    BBupper = BBbasis + (stdev * mult)      # but again, for demosntrative purposes I calculate it here
    BBlower = BBbasis - (stdev * mult)

    BBbasis.plot( color = "#769EB4AC", width=2 ) # You can plot series directly with their plot method.
    BBupper.plot( style='dotted' )
    BBlower.plot( style='dotted' )


    # I didn't add horizontal lines yet so I'm plotting constant floats here
    plot( 80, 'overbought', 'subpanel', color="#CECECE8B", style='dotted', width=1 )
    plot( 20, 'oversold', 'subpanel', color="#CECECE8B", style='dotted', width=1 )


    # You can read the array from a different timeframe, but be careful because
    # the barindexes don't match. You can use relative indexing or use its own
    # barindex accessing its own timeframe from inside the series class
    invRSI = rsiSlow[ rsiSlow.timeframe.barindex ]
    # invRSI = rsiSlow[-1] # also does the job

    # We convert the -1/+1 oscilator to the scale of standard rsi so they can share the same panel.
    if invRSI is not None:
        invRSI = (invRSI * 50) + 50

    # InvRSI is a float result of the operations now.
    # we can't directly plot the rsiSlow object as it belongs to a different timeframe
    # but we can plot the current value we obtain from the operation.
    # When plotting floats the plot class will create a series that you will be 
    # filling every time you call it.
    plot( invRSI, 'rsiSlow', 'subpanel', color="#ef38cd44", width=10 ) # The subpanel panel was created by us. See below.

    # standard rsi
    rsi14 = ta.RSI(close, 14).plot( 'subpanel' )


    # There's a built-in pivot indicator which is performance savy.
    # 'top' and 'bottom' are columns in the dataframe containing
    # the top and bottom prices of the candles bodies. Wicks excluded.
    # You can use high and low instead, or whatever you prefer.
    # 'Amplitude' is the main value you want to tweak for each symbol/timeframe
    pivots = ta.pivots( top, bottom, 11 )
    if pivots.isNewPivot:
        thisPivot = pivots.getLast() # only confirmed pivots. You can check the WIP pivot values in pivots.temp_pivot
        if thisPivot.type == c.PIVOT_HIGH:
            createMarker('▽', 'above', color = "#BDBDBD", timestamp=thisPivot.timestamp)
        else:
            createMarker('△', 'below', color = "#BDBDBD", timestamp=thisPivot.timestamp)

    # MACD in one go
    macd_line, signal_line, histo = ta.MACD(close)
    histo.histogram( 'macd', "#4A545D" )
    macd_line.plot( 'macd', color = "#AB1212", width=2 ) # The macd panel was created by us. See below
    signal_line.plot( 'macd', color = "#1BC573" )

    # same thing using methods
    buySignal = rsi14[barindex] > 50.0 and BBlower.crossingDown(close) and invRSI and invRSI < 20
    sellSignal = rsi14[barindex] < 50.0 and BBupper.crossingUp(close) and invRSI and invRSI > 80

    # accesing positions and making orders
    shortpos = trade.getActivePosition(c.SHORT)
    longpos = trade.getActivePosition(c.LONG)

    if buySignal:
        if shortpos is not None:
            trade.close( c.SHORT )
        offset = 50
        if longpos:
            lastorderindex = longpos.get_order_by_direction(c.LONG)['barindex']
            offset = barindex - lastorderindex
            if longpos.priceAvg < close[barindex]:
                offset = 0
        if offset > 40:
            trade.order( 'buy', c.LONG )

    if longpos:
        if longpos.get_unrealized_pnl_percentage() > 50 and longpos.collateral >= trade.strategy.order_size * 1.9 and invRSI > 20:
            trade.order( 'sell', c.LONG )
        elif longpos.get_unrealized_pnl_percentage() > 100 and invRSI > 60:
            trade.close( c.LONG )

    if sellSignal:
        if longpos is not None:
            trade.close( c.LONG )
        offset = 50
        if shortpos:
            lastorderindex = shortpos.get_order_by_direction(c.SHORT)['barindex']
            offset = barindex - lastorderindex
            if shortpos.priceAvg > close[barindex]:
                offset = 0
        if offset > 40:
            trade.order( 'sell', c.SHORT )

    if shortpos:
        if shortpos.get_unrealized_pnl_percentage() > 50 and shortpos.collateral >= trade.strategy.order_size * 1.9 and invRSI < 80:
            trade.order( 'buy', c.SHORT )
        elif shortpos.get_unrealized_pnl_percentage() > 100 and invRSI < 40:
            trade.close( c.SHORT )

    
    # draw a line where the liquidation is awaiting
    if shortpos:
        shortpos.drawLiquidation()

    if longpos:
        longpos.drawLiquidation()

    


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


    stream = stream_c( 'BTC/USDT:USDT', 'bitget', ['5m', '1m'], [runCloseCandle_slow, runCloseCandle_fast], event, 35000 )

    # Create subpanels to plot the oscilators.
    # width and height are values between 0 amd 1, representing the percentage of the
    # screen the subpanel will take. The order of creation will determine which one
    # is above or below.
    stream.registerPanel('macd', 1.0, 0.1, show_timescale=False ) # usually you only want the one at the bottom to show the timescale
    stream.registerPanel('subpanel', 1.0, 0.2, background_color="#292a2cac" )

    # Some options to print the results. The first one prints all orders so it's disabled for being spammy
    # trade.print_strategy_stats()
    trade.print_summary_stats()
    trade.print_pnl_by_period_summary()


    # Execute this call only if you want to check the chart. It's not neccesary to run the strategy.
    # You can also open the chart using the command 'chart' in the console, and select the timeframe by adding the timeframe name ('1h') to the command.
    # opening the chart is often way slower than calculating the strategy results. Don't look at me. Blame lightweight-charts.
    stream.createWindow()

    # Execute this call only if you want the strategy to keep running in realtime.
    # It's not neccesary if you only want a backtest.
    stream.run()