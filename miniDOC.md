# Algorizer documentation.
## Writing a Basic Strategy Script.

<br><br><br><br><br>

---

# Chapter 1. The Core: stream_c – Managing Data and Execution

The heart of any strategy in Algorizer is the `stream_c` object. It coordinates data fetching, manages multiple timeframes, and orchestrates both backtesting and live trading logic. Understanding `stream_c` lays the foundation for building robust strategies.

---

<br><br>

## What Does `stream_c` Do?

- Fetches and caches OHLCV price data via CCXT (crypto exchange API).
- Creates and maintains multiple `timeframe_c` instances for your chosen timeframes.
- Dispatches events and controls strategy callbacks for each candle close, price tick, and broker event.
- Manages all visualization elements (panels, markers, plots, lines).

---

## Internal Candle Processing Logic

Algorizer processes historical and real-time market data as follows:

- For each symbol and timeframe, OHLCV candles are obtained and stored in cache for efficient backtesting and live trading.
- Each candle is run in strict chronological order. When a candle for a timeframe closes, its matching callback function (the "closeCandle") is invoked, passing all price and volume data for that candle.
- Candles are processed "largest timeframe first" (if using multiple timeframes): this lets smaller timeframes access results from bigger timeframes in the same strategy run.
- During live trading, tick events (price updates) are continuously dispatched, so your script can react to changing conditions in real time. The system ensures real-time and backtest executions are consistent.
- As each event occurs (candle close, price tick, broker event), your registered callbacks are executed in response.

---

## Initializing the Stream: `stream_c.__Init__`

To set up the market data and control the strategy’s execution, initialize a `stream_c`:

```python
stream = stream_c(
    symbol,             # E.g., 'BTC/USDT:USDT' - symbol and quote currency
    exchangeID,         # E.g., 'bitget' - exchange name as understood by CCXT
    timeframeList,      # E.g., ['4h', '1h'] - timeframes of interest (largest first)
    callbacks,          # List of functions called on each respective candle close
    event_callback,     # (Optional) Custom handler for asynchronous events
    max_amount=35000,   # Number of historical candles to fetch for backtest
    cache_only=False    # Only use cached data if True
)
```

**Order matters:** List your timeframes largest-to-smallest to enable cross-timeframe logic.

**Example:**
```python
stream = stream_c(
    'BTC/USDT:USDT',
    'bitget',
    ['4h', '1h'],
    [runCloseCandle_4h, runCloseCandle_1h],
    event,
    35000
)
```
You must declare one callback function for each timeframe.

---

## The closeCandle Callback

For each timeframe you use, you must provide a function to handle the candle close event:

```python
def runCloseCandle_fast(timeframe: timeframe_c, open, high, low, close, volume, top, bottom):
    # Indicator computation, trade logic, order management, plotting, etc
```
- Attach these callback functions in the same order as your timeframes.
- Use them for calculations, trading signals, and chart updates.

---

## Event and Tick Callbacks

Define an **event callback** (commonly named `event`) to handle:

- **tick**: Called for every new price update in live mode.
- **broker_event**: Called when orders are executed in real time (not during backtesting).
- **cli_command**: For custom commands or live script interaction.

Example:
```python
def event(stream: stream_c, event: str, param, numparams):
    if event == "tick":
        if not stream.running:
            return
        candle = param
        candle.updateRemainingTime()
        # Update status, log positions, etc
    elif event == "broker_event":
        # Handle trade execution and external notifications
    elif event == "cli_command":
        cmd, args = param
        if cmd == "echo":
            print('Echo', args)
```
<br><br><br><br><br>

---

# Chapter 2. Strategy Setup and Trade Management

In Algorizer, the `trade` object is your assistant for managing orders and positions. It offers a standardized interface for placing, closing, and inspecting trades, supporting both live and backtesting modes. Configuring your strategy via `trade.strategy` ensures consistent behavior and accuracy during all stages of execution.

---
<br><br>


## Strategy Configuration

**Configure your strategy before initializing `stream_c`** so your settings are used during backtesting and live trading. This control is managed through `trade.strategy`:

```python
trade.strategy.verbose = False              # Enable/disable logging output
trade.strategy.hedged = False               # 'False' for oneway mode, 'True' for hedged mode
trade.strategy.currency_mode = 'USD'        # Quoting currency: 'USD' or 'BASE'
trade.strategy.order_size = 1000            # Default size for new orders
trade.strategy.max_position_size = 3000     # Maximum overall exposure (pyramiding limit)
trade.strategy.leverage_long = 5            # Leverage for long trades
trade.strategy.leverage_short = 5           # Leverage for short trades
```
> **Note:** All of these must be set before your call to `stream_c(...)`.

---

## Placing Orders

Use the `trade.order` function to place buy or sell orders.

```python
trade.order(cmd, target_position_type=None, quantity=None, leverage=None)
```
- `cmd`: Either `"buy"` or `"sell"` (string).
- `target_position_type`: Use `c.LONG` (1) or `c.SHORT` (-1). In hedged mode this must be supplied.
- `quantity`: Size in base currency when trade.strategy.currency_mode is 'BASE' or in quote currency when trade.strategy.currency_mode is 'USD'  (if not given, uses `strategy.order_size`).
- `leverage`: Overrides default leverage (leave `None` to use strategy defaults).

**Behavior:**
- If no `quantity` is specified, uses `order_size` and manages pyramiding: new orders will increase position size up to `max_position_size`. 
- To **disable pyramiding**, set `max_position_size` equal to `order_size`; further orders will not add to position once that limit is reached.
- In **hedged mode**, you must specify the position type to distinguish between long/short sides.

---

## Closing Positions

To close a current position, use:

```python
trade.close(pos_type=None)
```
- If in oneway mode, no argument is needed; closes any active position.
- If in hedged mode, you must specify `pos_type` (`c.LONG` or `c.SHORT`).

Alternatively, you can get the active position and call the `.close()` method:

```python
pos = trade.getActivePosition(pos_type)  # pos_type: c.LONG or c.SHORT
if pos:
    pos.close()
```

---

## Inspecting Positions

To retrieve the current active position:

```python
pos = trade.getActivePosition(pos_type=None)
```
- In oneway mode, `pos_type` is optional (returns the only active position).
- In hedged mode, supply `c.LONG` or `c.SHORT` to specify side.

Once you have a position object, the following methods are available:

```python
pnl = pos.get_unrealized_pnl()               # Current absolute PnL
pnl_pct = pos.get_unrealized_pnl_percentage() # Current PnL as percentage of collateral

order_info = pos.get_order_by_direction(order_direction, older_than_bar_index=None) 
# Returns info for a specific direction/order
```

---

## Take Profit, Stoploss, and Liquidation Orders

Set takeprofit and stoploss orders directly from the position object:

```python
tp = pos.createTakeprofit(price=None, quantity=None, win_pct=None, reduce_pct=None)
sl = pos.createStoploss(price=None, quantity=None, loss_pct=None, reduce_pct=None)
```
- `price`: Exact target price, or
- `win_pct` / `loss_pct`: Desired profit/loss % triggers
- `quantity` or `reduce_pct`: How much of the position to close (in base currency or by %), defaults to current position size

**Visualizing Orders:**  
You can plot TP, SL, and liquidation levels on your strategy chart:

```python
pos.drawTakeprofit(color="#17c200", style="dotted", width=2)
pos.drawStoploss(color="#e38100", style="dotted", width=2)
pos.drawLiquidation(color="#a00000", style="dotted", width=2)
```

> **Note:** Take profit, stoploss, and liquidation conditions are checked and triggered on every price update (tick), not just on candle close. This allows your risk management to be executed in real time, ensuring your stops and targets respond instantly to market moves.

---

## Order Sizing and Pyramiding

- If you do **not** pass `quantity` to `trade.order(...)`, the strategy uses `trade.strategy.order_size`.
- Orders are **pyramided** by default—adding to your position up to `trade.strategy.max_position_size`.
- To **disable pyramiding** (maintain single-size positions), set `max_position_size` equal to `order_size`.

---

## Summary

- The `trade` object coordinates configuration, order placement, position closing, and position inspection.
- Careful pre-stream setup of strategy parameters ensures correct backtest and live behavior.
- All order and risk management is handled with the core public API: `trade.order`, `trade.close`, and position methods.
- Pyramiding is managed automatically, but can be disabled by matching `order_size` and `max_position_size`.
- Take profit, stoploss, and liquidation are triggered at every tick, not only at candle close.

---
