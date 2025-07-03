Note: I asked copilot to create a very basic documentation file. I skimmed over it and I don't find anything vastly wrong, but don't expect much enlightening from it. It should help a little, tho.
Right now I'm busy writing the actual thing. I'll improve the documentation later on.

# Algorizer Framework Documentation

This document covers the main classes, constructs, and API of the Algorizer trading framework.
If you are new to the project, you are encouraged to review `example_misc.py` for a practical, commented example.

---

## Table of Contents

- [User Script Structure](#user-script-structure)
  - [Strategy Configuration](#strategy-configuration)
  - [Setting up a Stream](#setting-up-a-stream)
  - [The closeCandle Callback](#the-closecandle-callback)
  - [Events Callback](#events-callback)
- [Core Framework Classes](#core-framework-classes)
  - [stream_c](#stream_c)
  - [timeframe_c](#timeframe_c)
  - [generatedSeries_c](#generatedseries_c)
  - [plot_c, marker_c, line_c](#plot_c-marker_c-line_c)
- [Drawing API Overview](#drawing-api-overview)
  - [Plotting Series and Values](#plotting-series-and-values)
  - [Histograms](#histograms)
  - [Markers](#markers)
  - [Lines](#lines)
- [Panels and Visualization](#panels-and-visualization)
- [Additional Notes and Limitations](#additional-notes-and-limitations)
- [Indicators and helper functions](#Indicator-and-Helper-Functions)

---

## User Script Structure

A user script in Algorizer typically follows this structure:

### Strategy Configuration

Set up your strategy parameters before creating the stream (see `example_misc.py` for full details):

```python
trade.strategy.verbose = False
trade.strategy.hedged = False
trade.strategy.currency_mode = 'USD'
trade.strategy.order_size = 1000
trade.strategy.max_position_size = 3000
trade.strategy.leverage_long = 5
trade.strategy.leverage_short = 5
```

### Setting up a Stream

The key entry point is the creation of a `stream_c` object. The constructor's signature is:

```python
stream = stream_c(
    symbol,             # e.g. 'BTC/USDT:USDT'
    exchangeID,         # e.g. 'bitget'
    timeframeList,      # e.g. ['4h', '1h']
    callbacks,          # e.g. [closeCanle_4h, closeCandle_1h] List of closeCandle callback functions, one per timeframe
    event_callback,     # Optional: event handler function
    max_amount=35000,   # How many historical candles to fetch
    cache_only=False    # Use only cached data
)
```

**Important:**
- `timeframeList`: List of timeframes you want to run, e.g. `['4h', '1h']`. The order matters: larger timeframes first if you want to access their results from smaller ones.
- `callbacks`: List of functions to be called when each respective timeframe closes a candle (the closeCandle callback).
- `event_callback`: Optional function for handling events, such as broker events or CLI commands.

**Example:**
```python
stream = stream_c(
    'BTC/USDT:USDT',
    'bitget',
    ['4h', '1h'],
    [runCloseCandle_slow, runCloseCandle_fast],
    event,
    35000
)
```
*For a full example including subpanel registration and chart launching, see `example_misc.py`.*

### The closeCandle Callback

This function is the core of your strategy logic. It is called every time a candle closes for its assigned timeframe.

**Signature:**
```python
def runCloseCandle_fast(timeframe: timeframe_c, open, high, low, close, volume, top, bottom):
    # Your trading logic here
```
- You can attach one callback per timeframe, in the same order as in `timeframeList`.
- Use this function to calculate indicators, plot data, make orders, and manage trades.

### Events Callback

Optionally, you can define an `event_callback` function to handle asynchronous events (CLI commands, broker events, real-time ticks, etc). This is less important for most users, but allows for flexible integration (see `example_misc.py`).

---

## Core Framework Classes

### stream_c

Represents a symbol's data stream across one or more timeframes. Key responsibilities:
- Fetches and caches OHLCV data from exchanges via CCXT.
- Initializes and manages all `timeframe_c` objects.
- Handles backtesting and real-time updates.
- Manages visualization elements (panels, markers, lines).

**Key Methods:**
- `createMarker()`: Add a marker to the chart.
- `createLine()`: Add a line (point-to-point) to the chart.
- `registerPanel()`: Define a custom visualization panel.
- `run()`: Start the main loop (real-time or backtest).

### timeframe_c

Encapsulates all logic and data for a single timeframe:
- Holds the candle dataframe and all registered series (OHLCV, custom columns, calculated indicators).
- Registers and updates plots (lines, histograms).
- Handles callback logic for each candle close.
- Manages generated series.

**Key Methods:**
- `plot()`: Plot a line for a series or value.
- `histogram()`: Draw a histogram for a series or value.
- `createColumnSeries()`: Add a new custom data column.
- `calcGeneratedSeries()`: Register/calculate a generated indicator series.

### generatedSeries_c

Defined in `framework/calcseries.py`.
- Represents a calculated indicator series (e.g., MA, RSI).
- Provides `.plot()` and `.histogram()` methods for direct visualization.
- Always associated with a `timeframe_c`.

### plot_c, marker_c, line_c

- **plot_c**: Represents a visual plot (line or histogram) on a chart, with style, color, and panel assignment.
- **marker_c**: Represents a chart marker (label/arrow/circle).
- **line_c**: Represents a line drawn between two points on a chart (supports series or values).

---

## Drawing API Overview

You can use the following functions/classes for charting. These are available both as top-level functions and as methods on their respective objects.

### Plotting Series and Values

```python
plot(series_or_value, name=None, chart_name=None, color="#8FA7BBAA", style='solid', width=1)
```
- Plots a series or value on the chart.
- `chart_name` lets you target a subpanel.
- `style` supports: 'solid', 'dotted', 'dashed', 'large_dashed', 'sparse_dotted'.

### Histograms

```python
histogram(series_or_value, name=None, chart_name=None, color="#4A545D", margin_top=0.0, margin_bottom=0.0)
```
- Draws a histogram (bar plot).
- Can be attached to subpanels.

### Markers

```python
createMarker(text, location='below', shape='circle', color="#DEDEDE", timestamp=None, chart_name=None)
```
- Places a marker at a specific timestamp.
- `location`: 'above', 'below', 'inside'
- `shape`: 'arrow_up', 'arrow_down', 'circle', 'square'

### Lines

```python
createLine(x1, y1, x2, y2, color="#c7c7c7", width=1, chart_name='main')
```
- Draws a line between two points (x1, y1) and (x2, y2).
- x/y can be values or series.

---

## Panels and Visualization

You can add subpanels for plotting indicators separately:

```python
stream.registerPanel('macd', 1.0, 0.1, show_timescale=False)
stream.registerPanel('rsi', 1.0, 0.2)
```
- `width` and `height` are fractions of the chart window.
- The panel name is used in the `chart_name` argument of plot/histogram functions.

---

## Additional Notes and Limitations

- **Multiple timeframes**: The framework natively supports multi-timeframe strategies. For accessing another timeframe's data, ensure it is registered *before* the faster timeframe in the list.
- **Backtesting caveats**: Always use absolute indexing with data series; avoid negative indices except with the `.iloc()` method on generatedSeries.
- **Drawing limitations**: The drawing API currently supports lines, histograms, markers, and subpanels. Horizontal lines, boxes, and tables are not yet implemented.
- **More**: For practical usage patterns, see `example_misc.py`. For indicator functions and generatedSeries, see `framework/calcseries.py`.

---

# Indicator and Helper Functions`

This document lists the indicator, analytic, and helper factory functions available for user scripts in the Algorizer framework. Each function creates a time series (or related value) for use in trading algorithms.

---

## Indicators

**SMA(source, period)**
- *Simple Moving Average*: Arithmetic mean of the source over the period.

**EMA(source, period)**
- *Exponential Moving Average*: Moving average that gives more weight to recent values.

**DEMA(source, period)**
- *Double Exponential Moving Average*: Reduces lag compared to standard EMA.

**RMA(source, period)**
- *Relative/Smoothed Moving Average*: EMA variant with different smoothing.

**WMA(source, period)**
- *Weighted Moving Average*: Moving average with linearly increasing weights.

**HMA(source, period)**
- *Hull Moving Average*: Fast, smooth moving average with reduced lag.

**STDEV(source, period)**
- *Standard Deviation*: Measures volatility of the source over the period.

**DEV(source, period)**
- *Mean Absolute Deviation*: Average absolute distance from the mean.

**WILLR(period)**
- *Williams %R*: Momentum oscillator for overbought/oversold detection.

**TR(period)**
- *True Range*: Measures price range volatility.

**ATR(period)**
- *Average True Range*: Moving average of True Range, for volatility.

**SLOPE(source, period)**
- *Linear Regression Slope*: Slope of the regression line over the period.

**VHMA(source, period)**
- *Vertical Horizontal Filter Moving Average*: Adaptive average using price volatility.

**BIAS(source, period)**
- *Bias*: Percentage difference between source and its SMA.

**LINREG(source, period)**
- *Linear Regression*: Value of the regression line at the current bar.

**CCI(period=20)**
- *Commodity Channel Index*: Measures price deviation from statistical mean.

**CFO(source, period)**
- *Cumulative Forecast Oscillator*: Deviation from regression forecast.

**CMO(source, period=9)**
- *Chande Momentum Oscillator*: Measures momentum as difference between gains and losses.

**FWMA(source, period)**
- *Fibonacci Weighted Moving Average*: Weighted by Fibonacci sequence.

**RSI(source, period)**
- *Relative Strength Index*: Momentum oscillator from 0 to 100.

**IFTrsi(source, period)**
- *Inverse Fisher Transform RSI*: RSI passed through inverse Fisher transform for sharper signals.

**Fisher(period, signal=None)**
- *Fisher Transform*: Converts price into Gaussian normal distribution. Returns (main line, signal line).

**AO(fast=5, slow=34)**
- *Awesome Oscillator*: Difference of two SMAs of median price.

**BR(period)**
- *Buying Pressure Ratio*: Ratio of buying to selling pressure.

**AR(period)**
- *Active Ratio*: Ratio of price movement above and below open.

**BRAR(period)**
- *Buying Pressure and Active Ratio*: Returns both BR and AR as a tuple.

**CG(source, period)**
- *Center of Gravity Oscillator*: Weighted average emphasizing recent prices.

**STOCHk(source, period)**
- *Stochastic Oscillator %K*: Position of close relative to high/low over period.

**OBV()**
- *On-Balance Volume*: Cumulative indicator using volume for buying/selling pressure.

**LAGUERRE(source, gamma=0.7)**
- *Laguerre Oscillator*: Adaptive oscillator with user-defined smoothing gamma.

**Stochastic(source, k_period=14, d_period=3)**
- *Stochastic Oscillator*: Returns both %K and %D lines as a tuple.

**BollingerBands(source, period, mult=2.0)**
- *Bollinger Bands*: Returns (basis SMA, upper band, lower band).

**MACD(source, fast=12, slow=26, signal=9)**
- *Moving Average Convergence Divergence*: Returns (MACD line, signal line, histogram).

---

## Analytic & Helper Tools

**highest(source, period)**
- Returns the highest value of `source` over the period.

**lowest(source, period)**
- Returns the lowest value of `source` over the period.

**highestbars(source, period)**
- Number of bars since the highest value in the window.

**lowestbars(source, period)**
- Number of bars since the lowest value in the window.

**falling(source, period)**
- Boolean series: True where `source` has been strictly falling for the full period.

**rising(source, period)**
- Boolean series: True where `source` has been strictly rising for the full period.

**barsSinceSeries(source, period)**
- Number of bars since the most recent True value in `source` (up to a max of `period`). Returns NaN if none found.

**indexWhenTrueSeries(source, period=None)**
- Index of the last True value in `source` (NaN if none).

**indexWhenFalseSeries(source, period)**
- Index of the last False value in `source` (NaN if none).

**barsWhileTrueSeries(source, period=None)**
- Number of consecutive bars with True values up to the present (optionally limited to `period`).

**barsWhileFalseSeries(source, period=None)**
- Number of consecutive bars with False values up to the present (optionally limited to `period`).

---

## Non-Series Helpers (Return Single Values)

**indexWhenTrue(source)**
- Returns the index of the last True value in `source`, or None if not found.

**indexWhenFalse(source)**
- Returns the index of the last False value in `source`, or None if not found.

**barsSince(source)**
- Returns the number of bars since the last True value in `source`, or None if none.

**barsWhileTrue(source)**
- Returns the number of consecutive bars with True values up to now, or None if none.

---

## Special: Pivots

**pivots(high, low, amplitude=1.0, reversal_percent=32.0)**
- Returns a `pivots_c` object for pivot detection (see pivots_c documentation for usage). (Human comment: There is no such documentation ^_^')

---

**Notes:**
- All `source` arguments can be a registered series, a generated series, or a numpy array.
- Most outputs are `generatedSeries_c` objects (except helpers as noted).
- For usage examples, see `example_misc.py`.


*For further questions, consult the README and the examples in the repository.*
