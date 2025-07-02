from framework import calc, trade
from framework import stream_c, timeframe_c, generatedSeries_c, candle_c, pivots_c, pivot_c, c
from framework import plot, histogram, requestValue, createMarker, createLine


def event( stream:stream_c, event:str, param, numparams ):
    if event == "tick":
        '''
        candle : a cancle_c containing the OHLCV values of the latest price.
        realtime : boolean. True for realtime, false for backtesting.
        '''
        assert( isinstance(param, tuple) and len(param) == numparams)
        candle, realtime = param
        if not realtime : return

    elif event == "cli_command":
        assert( isinstance(param, tuple) and len(param) == numparams)
        cmd, args = param
        if cmd == 'echo': # command will always be lower case
            print( 'Echo ', args )

    elif event == "broker_event":
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
        assert( isinstance(param, tuple) and len(param) == numparams)
        order_type, quantity, quantity_dollars, position_type, position_size_base, position_size_dollars, position_collateral_dollars, leverage = param

        # Example of an alert for my webhook 'whook': https://github.com/germangar/whook
        account = ""
        url = ''
        if position_size_base == 0:
            message = f"{account} {stream.symbol} close"
        else:
            order = 'buy' if order_type == c.LONG else 'sell'
            message = f"{account} {stream.symbol} {order} {quantity_dollars:.4f}$ {leverage}x"
        if url:
            req = requests.post( url, data=message.encode('utf-8'), headers={'Content-Type': 'text/plain; charset=utf-8'} )


def runCloseCandle( timeframe:timeframe_c, open, high, low, close, volume, top, bottom ):
    calc.SMA(close, 200).plot()
    calc.RSI(close, 14).plot('rsi_subpanel')


if __name__ == '__main__':

    # configure the strategy before creating the stream
    trade.strategy.verbose = False
    trade.strategy.hedged = False
    trade.strategy.currency_mode = 'USD'
    trade.strategy.order_size = 1000
    trade.strategy.max_position_size = 2000
    trade.strategy.leverage_long = 1
    trade.strategy.leverage_short = 1
    
    stream = stream_c( 'BTC/USDT:USDT', 'bitget', ['1h'], [runCloseCandle], event, 25000 )

    stream.registerPanel('rsi_subpanel', 1.0, 0.2 )

    stream.createWindow( '1h' )

    # trade.print_strategy_stats()
    trade.print_summary_stats()
    trade.print_pnl_by_period_summary()

    stream.run()