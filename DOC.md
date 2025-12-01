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
# order_info is stored in the position for every executed order and it looks like this:

order_info = {
            'type': order_type,
            'price': price,
            'quantity': quantity,
            'collateral_change': collateral_change,
            'leverage': leverage,
            'barindex': active.barindex,
            'timestamp': active.timeframe.timestamp,
            'fees_cost': fee,
            'pnl': pnl_q,
            'pnl_percentage': pnl_pct,
        }
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
You can use this helpers to draw TP, SL, and liquidation levels on your strategy chart:

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
<br><br><br><br><br>

---

# Chapter 3. The timeframe_c Class – Candle Data, Updates, and Timeframe Utilities

The `timeframe_c` class represents and manages a single timeframe within your strategy. Each instance is responsible for hosting the OHLCV candle dataset, for driving candle updates (both in backtest and real time), and for orchestrating your close candle callback logic.

---
<br><br>

## Responsibilities and Core Logic

- **Candle Hosting:** Each `timeframe_c` manages a dataset, storing all OHLCV rows loaded for its specific timeframe. This includes the columns: timestamp, open, high, low, close, volume, top, bottom.
- **Update Engine:** On historical data load, the class copies all fully closed candles and prepares internal series. In both backtest mode and live trading, it advances bar-by-bar, updating its internal state and calling user closeCandle callbacks as each candle closes.
- **Callback Invocation:** When a candle closes for this timeframe, it invokes the matching callback, passing all relevant candle fields directly to your function for strategy logic and indicator calculation.

---

## Structure of the Dataset

- The dataframe (numpy NDArray) stores one row per closed candle.
- Columns include at least: timestamp, open, high, low, close, volume, top, bottom.
- Each column is associated with a corresponding `generatedSeries_c` object, which enables you to create calculations, plots, and run technical analysis logic over any series.
- For full details of `generatedSeries_c`, see the next chapter.

---

## Timeframe Indexing and Utilities

- **`barindex`**: Indicates the current index within the timeframe dataset, corresponding to the most recent closed candle. Use this for relative indexing and to synchronize with other data series.
- **`timestamp`**: Time (in ms) of the currently focused bar or candle. This allows for precise synchronization between multi-timeframe logic.

- **`ready`**: True when the timeframe is fully initialized and ready for execution. The backtest is run for each timeframe at once, and they get marked as ready one by one. A timeframe may be "ready" while the backtesting flag is still on as it runs the backtest on the other timeframes.
- **`backtesting`**: True if running in historical simulation mode (False in live trading).

You can use the following utility methods to retrieve or work with data:

- **`indexForTimestamp(timestamp)`**: Returns the bar index corresponding to a given timestamp. Useful for aligning signals or retrieving data at a specific point in time.
- **`ValueAtTimestamp(name, timestamp)`**: Retrieves the value of a named column or generatedSeries at a particular timestamp.

Other noteworthy properties:
- **`realtimeCandle`**: For live mode, represents the currently updating (not closed) OHLCV candle.

---

## Visual Elements: Plots, Histograms, Markers, and Lines

Plots and histograms are always **associated to a specific timeframe**. When you use `.plot()` or create a histogram in your closeCandle logic, these elements are tied directly to the timeframe, appearing only when the chart is displaying it.

Markers and lines, in contrast, are **independent of timeframe**; they can be placed universally on the chart and aren't bound to one dataset.

Example of plotting series in your callback:
```python
calc.SMA(close, 200).plot()           # Associated to the timeframe
histogram(rsi, "rsiPanel")            # Associated to the timeframe
# Markers/lines (createMarker, createLine) – not tied to timeframe
```

---

## Typical Usage in a Strategy

In practice, you use `timeframe_c` objects via your closeCandle callbacks. You access raw price arrays, plot indicators, and retrieve historical values with utility methods:
```python
def runCloseCandle(timeframe: timeframe_c, open, high, low, close, volume, top, bottom):
    barindex = timeframe.barindex
    # Compute indicators, plot series, reference previous price points
    value = timeframe.ValueAtTimestamp("close", some_timestamp)
    idx = timeframe.indexForTimestamp(some_timestamp)
```
You can also safely access and operate on the `dataset` and its generatedSeries objects for indicator calculations.

---

## Summary

- `timeframe_c` encapsulates candle data, state, update logic, and closeCandle callback execution for a single timeframe.
- It organizes all base columns and generated series for price, volume, and derived indicators.
- Provides bar and timestamp indexing utilities for robust multi-timeframe logic.
- Plots and histograms are bound to a timeframe; markers and lines are chart-wide.
- Utility methods like `ValueAtTimestamp` and `indexForTimestamp` empower precise data access within your strategy.
- Properties `ready` and `backtesting` help you distinguish mode and execution state.

---
