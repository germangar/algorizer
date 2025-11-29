# Algorizer documentation.
## Writing a Basic Strategy Script.

---

# Chapter 1. The Core: stream_c – Managing Data and Execution

The heart of any strategy in Algorizer is the `stream_c` object. It coordinates data fetching, manages multiple timeframes, and orchestrates both backtesting and live trading logic. Understanding `stream_c` lays the foundation for building robust strategies.

---

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
You can expand on this pattern to handle any custom event.
