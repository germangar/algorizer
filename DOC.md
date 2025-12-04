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
<br><br><br><br><br>

---

# Chapter 4. The generatedSeries_c Class – Representing Series in the Dataset

generatedSeries_c objects represent series (columns) inside a timeframe's numpy dataset. They are the building blocks you use to compute indicators, publish plotting values, and store any per-bar derived or user-managed data.

---
<br><br>

## What generatedSeries_c Is and What It Does

- A generatedSeries_c is a lightweight descriptor that maps a logical series name to a physical column in the timeframe's numpy dataset.
- It exposes a Python-friendly view over that column so scripts can read values using normal Python indexing (e.g., mySeries[-1], mySeries[barindex]).
- Some generatedSeries_c instances are "computed" by the engine (indicators such as SMA, RSI, etc.). These computed series automatically produce values as new candles are processed.
- Other generatedSeries_c instances are "manual" (user-declared): you create them or they are created implicitly by plotting floats/ints, and the script is responsible for updating their last value when appropriate.

Terminology used in this chapter:
- "Computed series" — automatically calculated indicator series (engine-managed).
- "Manual series" — user-declared or plot-created series that the script updates explicitly.

---

## Key Properties and Shape

Typical properties accessible on a generatedSeries_c object:
- name — logical name of the series (string).
- column_index — integer index of the corresponding column in the timeframe.dataset.
- timeframe — implicit association to the timeframe that owns the dataset (so series from different timeframes are not directly interoperable).

The underlying storage is a numpy array column inside timeframe.dataset, so reads (indexing) are fast and memory-efficient.

---

## Reading Values (Indexing)

- Use Python-style indexing to read values:
  - mySeries[-1] → last closed value
  - mySeries[timeframe.barindex] → value at current barindex (same as -1)
  - mySeries[i] → arbitrary historical bar
- The indexing behavior matches normal Python semantics (negative indexes allowed) mapped to the numpy column.

Note: Because generatedSeries_c maps directly to a dataframe column, it is efficient to read entire slices or specific indices for indicator logic.

---

## Computed vs Manual Series

- Computed series (indicators):
  - Created by the calculation utilities (for example, via functions in the series/calc module).
  - The engine updates their values automatically at once in backtests and one by one as it advances candles in live runs.
  - They are the primary mechanism for indicators like SMA, RSI, STDEV, and similar.

- Manual series:
  - Created implicitly when you plot a scalar with a name, or explicitly by the user.
  - Your script must assign/update the most recent value on each bar if you want it to persist.
  - Useful for storing custom signals, state, or any derived number that is not produced by the built-in calculators.

---

## Plotting and Histograms

- generatedSeries_c instances are intended to be used directly as sources when plotting:
  - series.plot(...) will register a plot tied to the timeframe and use the series column as source.
- When you call top-level helpers like plot(42.0, 'name', ...), the framework may create a generatedSeries_c behind the scenes and write the float into its last column for display.
- Plots and histograms created from a generatedSeries_c are bound to the timeframe that owns the series (they appear only when that timeframe is displayed).

---

## Interoperability and Operations

- Series arithmetic and higher-level indicator builders in the project are designed to work with generatedSeries_c objects belonging to the same timeframe (same dataset).
- You can pass a generatedSeries_c to indicator functions or to other generatedSeries_c-based operations; mixing series from different timeframes is not supported directly and requires alignment through timestamps/index lookups.

---

## Methods of Immediate Interest

- Indexing / value access:
  - mySeries[i] — read value at index i (supports negative indices and Python-style slicing)
- Plotting:
  - mySeries.plot(panel_name=None, color=None, style=None, width=None) — convenience to register a plotted line for this series on the timeframe (plot parameters follow the plotting API used elsewhere in the framework).
- Cross detection helpers:
  - mySeries.crossing(otherSeries) — returns boolean/series indicating cross occurrences with another series
  - mySeries.crossingUp(otherSeries) — detects upward crossings (this crossing from below the other)
  - mySeries.crossingDown(otherSeries) — detects downward crossings (this crossing from above the other)
- Introspection:
  - mySeries.name — readable name
  - mySeries.column_index — integer column position in the timeframe.dataset

Because the class is a thin wrapper over a dataset column, many interactions are just reading/writing into the dataset; high-level helpers (like .plot() and the crossing methods) exist to simplify common tasks.

---

## Typical Usage Patterns

- Use computed series returned by the calculation utilities:
  - rsi = calc.RSI(close, 14)
  - rsi.plot('subpanel')
- Create or use manual series for ad-hoc values:
  - mySeries = generatedSeries_c('signal', None)  # 'source = None': manual series have no source; other args are optional for manual series
  - mySeries[timeframe.barindex] = some_value
- Operate series within the same timeframe:
  - bb_mid = calc.SMA(close, 21)
  - bb_upper = bb_mid + (calc.STDEV(close, 21) * 2)
- Detect crosses:
  - cross = fastEMA.crossing(slowEMA)
  - up = fastEMA.crossingUp(slowEMA)
  - down = fastEMA.crossingDown(slowEMA)

---

## Practical Notes and Caveats

- Series belong to a single timeframe/dataset. If you need cross-timeframe values, query the other timeframe via its dataset / utility functions (indexForTimestamp / ValueAtTimestamp) and then reference the appropriate series column there.
- A generatedSeries_c can be retrieved by name from its timeframe using:
  - timeframe.generatedSeries['name']
  This is a convenient way to access series created earlier in the script or by the engine.
- The arguments passed to your closeCandle callback — open, high, low, close, top, bottom, volume — are provided as built-in manual generatedSeries_c objects created and maintained by the timeframe. They map directly to the underlying dataset columns and are the canonical way to reference the primary price/volume series inside your callbacks.
- For manual series, remember to update the series on the new bar if persistence is desired — otherwise only the current run-time value remains visible.
- Because generatedSeries_c maps to numpy columns, vectorized operations and slice reads are efficient; prefer bulk reads when computing multi-bar indicators if you are implementing your own custom calculations.

---

## Summary

- generatedSeries_c is the framework's representation of a column (series) in a timeframe's dataset.
- There are computed series (indicators) updated automatically by the engine and manual series that you update explicitly.
- Values are accessed with plain Python indexing and are efficient because they map directly to numpy columns.
- Use .plot() on series to quickly visualize them on the timeframe's chart; remember series are bound to the timeframe they belong to and interoperate only with series from the same timeframe.
- Useful helpers include crossing, crossingUp and crossingDown for detecting interactions between two series.

---
<br><br>

Extension — Creating Custom Computed Series
---
<br><br>
You can create your own computed (auto-calculated) generatedSeries_c instances from a user script without modifying the framework. The pattern consists of two pieces:

- A calculation function that receives raw arrays and computes the resulting numpy array for the full series.
- A factory function that registers that calculation with the timeframe and returns a generatedSeries_c you can use like any engine-provided indicator.

High-level notes:
- The calculation function should accept the raw source array and the other arguments the engine expects and return a numpy array of the same length as the provided source array.
- The factory function typically calls timeframe.calcGeneratedSeries(...) to create or retrieve the generatedSeries_c and register the calculation function.

Minimal example signatures:

```python
# Calculation function signature (example)
def my_calc_function(source_array: np.ndarray, period: int, dataset: np.ndarray, cindex: int, param=None) -> np.ndarray:
    # compute and return a numpy array of values (same length as source_array)
    ...

# Factory function signature (example)
def MyIndicator(source: generatedSeries_c, period: int) -> generatedSeries_c:
    return source.timeframe.calcGeneratedSeries(
        'my_indicator_name',
        source,
        period,
        my_calc_function,
        always_reset = False  # or True when full recalculation is required every update
    )
```

Important details:
- always_reset: If True, the engine will recalculate the full array each time new data arrives (safer for algorithms that need the full history recomputed). If False, the engine may attempt incremental updates. When in doubt, test with False and switch to True if realtime outputs appear incorrect.
- The created generatedSeries_c will belong to the same timeframe as the `source` and can be used immediately in your closeCandle logic:
  ```python
  cg = MyIndicator(close, 14)
  cg.plot('subpanel')
  ```
- Because the generated series is produced and managed by the timeframe, it will be updated automatically during backtest initialization (full calculation) and during live ticks (incrementally or fully depending on always_reset).

Full example:
- See the complete example file for a working example of this pattern:
  https://github.com/germangar/algorizer/blob/main/example_custom_calculated_series.py

Practical tip:
- To craft the calculation function correctly, inspect the implementation of generatedSeries_c.calculate_full and generatedSeries_c.update in algorizer/series.py (formerly calcseries.py) and other existing calculation functions (in the same module) to match expectations about array shapes, NaN handling, and performance patterns.

---
