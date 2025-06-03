
import pandas as pd
import pandas_ta as pt
import numpy as np
import time
import active
import tools


# Dynamically set __all__ to include all names that don't start with '_' and are not in _exclude
_exclude = ['active']
__all__ = [name for name in globals() if not (name.startswith('_') or name in _exclude)]




# #
# # GENERATED SERIES : These are series of values that are calculated always using the same formula
# #

def _generatedseries_calculate_highest(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    return series.rolling(window=period, min_periods=period).max()

def _generatedseries_calculate_lowest(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    return series.rolling(window=period, min_periods=period).min()

def _generatedseries_calculate_highestbars(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    def offset_to_max(x):
        return (period - 1) - np.argmax(x)
    return series.rolling(window=period, min_periods=period).apply(offset_to_max, raw=True)

def _generatedseries_calculate_lowestbars(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    def offset_to_min(x):
        return (period - 1) - np.argmin(x)
    return series.rolling(window=period, min_periods=period).apply(offset_to_min, raw=True)

def _generatedseries_calculate_sma(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.sma( series, period )

def _generatedseries_calculate_ema(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.ema( series, period )

def _generatedseries_calculate_dema(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.dema( series, period )

def _generatedseries_calculate_linreg(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.linreg( series, period )

def _generatedseries_calculate_rma(series: pd.Series, length: int, df:pd.DataFrame, param=None) -> pd.Series:
    return series.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

def _generatedseries_calculate_stdev(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.stdev( series, period )

def _generatedseries_calculate_bias(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.bias( series, period )

def _generatedseries_calculate_cfo(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.cfo( series, period )

def _generatedseries_calculate_fwma(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.fwma( series, period )

def _generatedseries_calculate_dev(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    if 1:
        return pt.mad( series, period )
    else:
        # Calculate the average deviation over a given rolling window in a pandas Series.
        # Initialize a list to hold the deviation values
        deviations = [pd.NA] * (period - 1)  # Start with NA values for the initial periods
        # Iterate over each rolling window
        for i in range(period - 1, len(series)):
            rolwindow = series[i - period + 1:i + 1]
            mean = rolwindow.mean()
            deviation = (rolwindow - mean).abs().sum() / period
            deviations.append(deviation)
        return pd.Series(deviations, index=series.index).dropna()

def _generatedseries_calculate_williams_r(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    """
        Calculate Williams %R for a given series using OHLC data from df over a period.

        Args:
        - series: pd.Series, typically a placeholder, but required for compatibility with generatedSeries_c.
        - period: int, the period/window for the Williams %R calculation.

        Returns:
        - pd.Series, the calculated Williams %R values.
        """
    if 1:
        return pt.willr( df['high'], df['low'], df['close'], length=period )
    else:
        # Ensure the DataFrame has the required columns
        # if not all(col in df.columns for col in ['high', 'low', 'close']):
        #     raise ValueError("The global DataFrame must contain 'high', 'low', and 'close' columns")

        if len(df) < period:
            return pd.Series([pd.NA] * len(df), index=df.index)  # Not enough data to calculate Williams %R

        # Initialize a list to hold the Williams %R values
        williams_r_values = [pd.NA] * (period - 1)  # NA for the initial period

        # Calculate Williams %R for each rolling window
        for i in range(period - 1, len(df)):
            highest_high = df['high'].iloc[i - period + 1:i + 1].max()
            lowest_low = df['low'].iloc[i - period + 1:i + 1].min()
            current_close = df['close'].iloc[i]

            if highest_high == lowest_low:  # Prevent division by zero
                williams_r_values.append(pd.NA)
            else:
                williams_r = (highest_high - current_close) / (highest_high - lowest_low) * -100
                williams_r_values.append(williams_r)

        return pd.Series(williams_r_values, index=df.index)

def _generatedseries_calculate_rsi(series, period, df:pd.DataFrame, param=None) -> pd.Series:
    deltas = series.diff()
    gain = deltas.where(deltas > 0, 0).rolling(window=period).mean()
    loss = -deltas.where(deltas < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _generatedseries_calculate_tr(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    """
    Calculate the True Range (TR) for a given series.

    Args:
    - series: pd.Series, the input series (only used to align with generatedSeries_c interface).
    - period: int, the period for the True Range calculation.

    Returns:
    - pd.Series, the calculated True Range series.
    """
    if 1:
        if len(series) < period:
            return pd.Series( [pd.NA] * len(series), index=series.index )  # Not enough data to calculate the slope
        return pt.true_range( df['high'], df['low'], df['close'], length=period )
    else:
        
        high = df['high']
        low = df['low']
        close = df['close']

        high_low = high - low
        high_close_prev = (high - close.shift()).abs()
        low_close_prev = (low - close.shift()).abs()

        tr = high_low.combine(high_close_prev, max).combine(low_close_prev, max)
        return tr

def _generatedseries_calculate_atr(series, period, df:pd.DataFrame, param=None) -> pd.Series:
    if len(series) < period:
        return pd.Series( [pd.NA] * len(series), index=series.index )  # Not enough data to calculate the slope
    return pt.atr( df['high'], df['low'], df['close'], length=period )
    

def _generatedseries_calculate_rising(series: pd.Series, length: int, df:pd.DataFrame, param=None) -> pd.Series:
    """
    Check if the series has been rising for the given length of time.

    Args:
    - series: pd.Series, the input series.
    - length: int, the number of periods to check.

    Returns:
    - pd.Series, a boolean series indicating where the series is rising.
    """
    if len(series) < length:
        return pd.Series([pd.NA] * len(series), index=series.index)  # Not enough data to perform the check

    # Calculate the difference between consecutive elements
    diff_series = series.diff().dropna()
    
    # Create a boolean series indicating whether each rolling window is rising
    is_rising = diff_series.rolling(window=length-1).apply(lambda x: (x > 0).all(), raw=True).astype(bool)
    is_rising = pd.concat([pd.Series([pd.NA] * (length-1), index=series.index[:length-1]), is_rising])

    return is_rising

def _generatedseries_calculate_falling(series: pd.Series, length: int, df:pd.DataFrame, param=None) -> pd.Series:
    """
    Check if the series has been falling for the given length of time.

    Args:
    - series: pd.Series, the input series.
    - length: int, the number of periods to check.

    Returns:
    - pd.Series, a boolean series indicating where the series is falling.
    """
    if len(series) < length:
        return pd.Series([pd.NA] * len(series), index=series.index)  # Not enough data to perform the check
    
    # Calculate the difference between consecutive elements
    diff_series = series.diff().dropna()

    # Create a boolean series indicating whether each rolling window is falling
    is_falling = diff_series.rolling(window=length-1).apply(lambda x: (x < 0).all(), raw=True).astype(bool)
    is_falling = pd.concat([pd.Series([pd.NA] * (length-1), index=series.index[:length-1]), is_falling])
    return is_falling

def _generatedseries_calculate_wma(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    """
    Calculate the Weighted Moving Average (WMA) for a given series and length.
    
    Args:
    - series: pd.Series, the input series.
    - length: int, the period/window for the WMA calculation.
    
    Returns:
    - pd.Series, the calculated WMA series.
    """
    if 1:
        if len(series) < period:
            return pd.Series([pd.NA] * len(series), index=series.index)  # Not enough data to calculate the slope
        return pt.wma( series, period )
    else:
        
        weights = pd.Series(range(1, period + 1))
        wma = series.rolling(period).apply(lambda prices: (prices * weights).sum() / weights.sum(), raw=True)
        return wma

def _generatedseries_calculate_hma(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    """
    Calculate the Hull Moving Average (HMA) for a given series and length.
    
    Args:
    - series: pd.Series, the input series.
    - length: int, the period/window for the HMA calculation.
    
    Returns:
    - pd.Series, the calculated HMA series.
    """
    if 1:
        if len(series) < period:
            return pd.Series([pd.NA] * len(series), index=series.index)  # Not enough data to calculate the slope
        return pt.hma( series, period )
    else:
        if len(series) < period:
            return pd.Series([pd.NA] * len(series), index=series.index)  # Not enough data to calculate the HMA
        
        half_length = int(period / 2)
        sqrt_length = int(period ** 0.5)
        
        wma_half_length = pt.wma(series, half_length)
        wma_full_length = pt.wma(series, period)
        
        diff_wma = 2 * wma_half_length - wma_full_length
        
        hma = pt.wma(diff_wma, sqrt_length)
        
        return hma

def _generatedseries_calculate_slope(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    """
    Calculate the slope of a rolling window for a given length in a pandas Series without using numpy.

    Args:
    - series: pd.Series, the input series.
    - length: int, the period/window for the slope calculation.

    Returns:
    - pd.Series, the calculated slope series.
    """
    if len(series) < period:
        return pd.Series([pd.NA] * len(series), index=series.index)  # Not enough data to calculate the slope
    
    if 1: 
        return pt.slope( series, period )
    else:
        # this one doesn't fail on single candle updates but it's slower than recalculating it all using pandas_ta
        def slope_calc(y):
            x = range(len(y))
            n = len(y)
            x_mean = sum(x) / n
            y_mean = sum(y) / n

            num = sum((x_i - x_mean) * (y_i - y_mean) for x_i, y_i in zip(x, y))
            den = sum((x_i - x_mean) ** 2 for x_i in x)

            if den == 0:
                return 0  # Prevent division by zero

            slope = num / den
            return slope

        # Apply the slope calculation to each rolling window
        slope_series = series.rolling(window=period).apply(slope_calc, raw=False)

        return slope_series
    
def _generatedseries_calculate_cci(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.cci( df['high'], df['low'], df['close'], period )

def _generatedseries_calculate_bbupper(series, period, df: pd.DataFrame, param=None) -> pd.Series:
    BBmult = param
    sma_name = tools.generatedSeriesNameFormat('sma', series, period)
    stdev_name = tools.generatedSeriesNameFormat('stdev', series, period)

    if len(series) == period:
        # Real-time update case — return just last value
        upper = df[sma_name].iloc[-1] + BBmult * df[stdev_name].iloc[-1]
        return pd.Series([upper], index=[df.index[-1]])

    # Initialization case — full series calculation
    return df[sma_name] + (BBmult * df[stdev_name])

def _generatedseries_calculate_bblower(series, period, df: pd.DataFrame, param=None) -> pd.Series:
    BBmult = param
    sma_name = tools.generatedSeriesNameFormat('sma', series, period)
    stdev_name = tools.generatedSeriesNameFormat('stdev', series, period)

    if len(series) == period:
        # Real-time update case — compute only the last value
        lower = df[sma_name].iloc[-1] - BBmult * df[stdev_name].iloc[-1]
        return pd.Series([lower], index=[df.index[-1]])

    # Initialization case — full series
    return df[sma_name] - (BBmult * df[stdev_name])

def _generatedseries_calculate_inverse_fisher_rsi(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    rsi = df[tools.generatedSeriesNameFormat("rsi", series, period)]
    v1 = 0.1 * (rsi - 50)

    weights = np.arange(1, period + 1)
    wma = v1.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    # Clip v2 before exponentiation to prevent overflow
    v2_clipped = wma.clip(lower=-10, upper=10)
    exp_val = np.exp(2 * v2_clipped)
    iftrsi = (exp_val - 1) / (exp_val + 1)
    return iftrsi

def _generatedseries_calculate_fisher(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    high = df["high"]
    low = df["low"]
    fisher_df = pt.fisher(high, low, length=period, signal=1)  # Minimal signal to isolate Fisher line
    return fisher_df.iloc[:, 0]  # Main Fisher line

def _generatedseries_calculate_fisher_signal(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    high = df["high"]
    low = df["low"]
    fisher_df = pt.fisher(high, low, length=period, signal=param if param else 9)
    return fisher_df.iloc[:, 1]  # Signal line

def _generatedseries_calculate_ao(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    """
    Calculate Awesome Oscillator: SMA(median_price, 5) - SMA(median_price, 34)
    `period` is ignored; `param` can optionally override the two SMA lengths as a tuple: (fast, slow)
    """
    if 'high' not in df.columns or 'low' not in df.columns:
        raise ValueError("DataFrame must contain 'high' and 'low' columns for AO calculation.")
    
    fast, slow = (5, 34)
    if isinstance(param, tuple) and len(param) == 2:
        fast, slow = param

    median_price = (df['high'] + df['low']) / 2
    sma_fast = median_price.rolling(window=fast).mean()
    sma_slow = median_price.rolling(window=slow).mean()

    ao = sma_fast - sma_slow
    return ao

def _generatedseries_calculate_br(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    brar_df = pt.brar(high=df['high'], low=df['low'], close=df['close'], length=period)
    return brar_df['BR']

def _generatedseries_calculate_ar(series: pd.Series, period: int, df: pd.DataFrame, param=None) -> pd.Series:
    brar_df = pt.brar(high=df['high'], low=df['low'], close=df['close'], length=period)
    return brar_df['AR']

def _generatedseries_calculate_cg(series: pd.Series, period: int, df:pd.DataFrame, param=None) -> pd.Series:
    return pt.cg( series, period )


class generatedSeries_c:
    def __init__( self, type:str, source:pd.Series, period:int, func = None, param=None, always_reset:bool = False, timeframe = None ):
        self.name = tools.generatedSeriesNameFormat( type, source, period )
        self.sourceName = source.name
        self.period = period
        self.param = param
        self.func = func
        self.timeframe = timeframe
        self.timestamp = 0
        self.alwaysReset = always_reset

        if( self.timeframe == None ):
            raise SystemError( f"Generated Series has no assigned timeframe [{self.name}]")
        
        if( self.timeframe.shadowcopy ):
            raise SystemError( f'Tried to create series [{self.name}] while shadowcopying.' )

        if( self.func == None ):
            raise SystemError( f"Generated Series without a func [{self.name}]")

        if( self.period < 1 ):
            raise SystemError( f"Generated Series  with invalid period [{period}]")
        

    def initialize( self, source:pd.Series ):
        if( len(source) >= self.period and ( not self.name in self.timeframe.df.columns or self.alwaysReset ) ):
            if( self.timeframe.shadowcopy ):
                raise SystemError( f"[{self.name}] tried to initialize as shadowcopy" )
            start_time = time.time()
            self.timeframe.df[self.name] = self.func(source, self.period, self.timeframe.df, self.param).dropna()
            self.timestamp = self.timeframe.df['timestamp'].iloc[self.timeframe.barindex]
            if( self.timeframe.stream.initializing ):
                print( f"Initialized {self.name}." + " Elapsed time: {:.2f} seconds".format(time.time() - start_time))


    def update( self, source:pd.Series ):
        if( self.timeframe.shadowcopy ):
            return

        # has this row already been updated?
        if( self.timestamp >= self.timeframe.df.iat[self.timeframe.barindex, 0] ): # same as self.timeframe.df['timestamp'].iloc[self.timeframe.barindex]
            return

        # if non existant try to create new. A few need to be made new every time
        if( self.timestamp == 0 or self.alwaysReset ):
            self.initialize( source )
            return
        
        
        if( len(self.timeframe.df) < self.period ):
            return
        
        # realtime updates

        # slice the required block of candles to calculate the current value of the generated series
        newval = self.func(source[-self.period:], self.period, self.timeframe.df, self.param).loc[self.timeframe.barindex]
        self.timeframe.df.loc[self.timeframe.df.index[-1], self.name] = newval
        self.timestamp = self.timeframe.timestamp

    def value( self, backindex = 0 ):
        if( backindex >= len(self.timeframe.df) ):
            raise SystemError( "generatedseries_c.value() : backindex out of bounds")

        return self.timeframe.df[self.name].iloc[self.timeframe.barindex - backindex]

    def series( self ):
        return self.timeframe.df[self.name]
    
    # direct operations will always operate the last value.
    # If you want to operate the whole series do it with the .series() method.
    def __add__(self, other):
        if isinstance(other, generatedSeries_c):
            return self.value() + other.value()
        if isinstance(other, pd.Series):
            return self.value() + other.iloc[active.barindex]
        return self.value() + other
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, generatedSeries_c):
            return self.value() - other.value()
        if isinstance(other, pd.Series):
            return self.value() - other.iloc[active.barindex]
        return self.value() - other
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __mul__(self, other):
        if isinstance(other, generatedSeries_c):
            return self.value() * other.value()
        elif isinstance(other, pd.Series):
            return self.value() * other.iloc[active.barindex]
        else:
            return self.value() * other
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, generatedSeries_c):
            return self.value() / other.value()
        if isinstance(other, pd.Series):
            return self.value() / other.iloc[active.barindex]
        return self.value() / other
    
    def __rtruediv__(self, other):
        return self.__truediv__(other)
    
    def __lt__(self, other):
        if isinstance(other, generatedSeries_c):
            return self.value() < other.value()
        if isinstance(other, pd.Series):
            return self.value() < other.iloc[active.barindex]
        return self.value() < other
    
    def __rlt__(self, other):
        return self.__lt__(other)
    
    def __gt__(self, other):
        if isinstance(other, generatedSeries_c):
            return self.value() > other.value()
        if isinstance(other, pd.Series):
            return self.value() > other.iloc[active.barindex]
        return self.value() > other
    
    def __rgt__(self, other):
        return self.__gt__(other)
    
    def __eq__(self, other):
        if other is None:
            return False
        if isinstance(other, generatedSeries_c):
            return self.value() == other.value()
        if isinstance(other, pd.Series):
            return self.value() == other.iloc[active.barindex]
        return self.value() == other
    
    # def __req__(self, other):
    #     return self.__eq__(other)

    def plot( self, chart = None ):
        '''it returns the generatedSeries even when calling plot from the timeframe and the function returns the plot_c'''
        if( self.timestamp > 0 ):
            self.timeframe.plot( self.series(), self.name, chart )
            return self
    
    def crossingUp( self, other ):
        df = self.timeframe.df
        if( self.timestamp == 0 or len(df)<2 or self.value() == None or self.value(1) == None ):
            return False
        if isinstance( other, generatedSeries_c ):
            if( other.timestamp == 0  or other.value() == None or other.value(1) == None ):
                return False
            return ( self.value(1) <= other.value(1) and self.value() >= other.value() and self.value() != self.value(1) )
        if isinstance( other, pd.Series ):
            if( len(other) < 2 ):
                return False
            if pd.isna(other.iloc[self.timeframe.barindex-1]) or pd.isna(other.iloc[self.timeframe.barindex]) :
                return False
            return ( self.value(1) <= other.iloc[self.timeframe.barindex-1] and self.value() >= other.iloc[self.timeframe.barindex] and self.value() != self.value(1) )
        try:
            float(other)
        except ValueError:
            return False
        else:
            return ( self.value(1) <= float(other) and self.value() >= float(other) and self.value() != self.value(1) )
    
    def crossingDown( self, other ):
        df = self.timeframe.df
        if( self.timestamp == 0 or len(df)<2 or self.value() == None or self.value(1) == None ):
            return False
        if isinstance( other, generatedSeries_c ):
            if( other.timestamp == 0  or other.value() == None or other.value(1) == None ):
                return False
            return ( self.value(1) >= other.value(1) and self.value() <= other.value() and self.value() != self.value(1) )
        if isinstance( other, pd.Series ):
            if( len(other) < 2 ):
                return False
            if pd.isna(other.iloc[self.timeframe.barindex-1]) or pd.isna(other.iloc[self.timeframe.barindex]) :
                return False
            return ( self.value(1) >= other.iloc[self.timeframe.barindex-1] and self.value() <= other.iloc[self.timeframe.barindex] and self.value() != self.value(1) )
        try:
            float(other)
        except ValueError:
            return False
        else:
            return ( self.value(1) >= float(other) and self.value() <= float(other) and self.value() != self.value(1) )
    
    def crossing( self, other ):
        return self.crossingUp(other) or self.crossingDown(other)
    
    
    


def highest(source: pd.Series, period: int, timeframe=None) -> generatedSeries_c:
    if timeframe is None: timeframe = active.timeframe
    return timeframe.calcGeneratedSeries('highest', source, period, _generatedseries_calculate_highest)

def lowest(source: pd.Series, period: int, timeframe=None) -> generatedSeries_c:
    if timeframe is None: timeframe = active.timeframe
    return timeframe.calcGeneratedSeries('lowest', source, period, _generatedseries_calculate_lowest)

def highestbars(source: pd.Series, period: int, timeframe=None) -> generatedSeries_c:
    if timeframe is None: timeframe = active.timeframe
    return timeframe.calcGeneratedSeries('highestbars', source, period, _generatedseries_calculate_highestbars)

def lowestbars(source: pd.Series, period: int, timeframe=None) -> generatedSeries_c:
    if timeframe is None: timeframe = active.timeframe
    return timeframe.calcGeneratedSeries('lowestbars', source, period, _generatedseries_calculate_lowestbars)

def falling( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'falling', source, period, _generatedseries_calculate_falling )

def rising( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'rising', source, period, _generatedseries_calculate_rising )

def SMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'sma', source, period, _generatedseries_calculate_sma )

def EMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( "ema", source, period, _generatedseries_calculate_ema )

def DEMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( "dema", source, period, _generatedseries_calculate_dema, always_reset=True )

def WMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( "wma", source, period, _generatedseries_calculate_wma )

def HMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( "hma", source, period, _generatedseries_calculate_hma, always_reset=True )

# def JMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
#     if timeframe == None : timeframe = active.timeframe
#     return timeframe.calcGeneratedSeries( "jma", source, period, pt.jma )

# def KAMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
#     if timeframe == None : timeframe = active.timeframe
#     return timeframe.calcGeneratedSeries( "kama", source, period, pt.kama )

def LINREG( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( "linreg", source, period, _generatedseries_calculate_linreg )

def RSI( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'rsi', source, period, _generatedseries_calculate_rsi )

def DEV( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'dev', source, period, _generatedseries_calculate_dev )

def STDEV( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'stdev', source, period, _generatedseries_calculate_stdev )

def RMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'rma', source, period, _generatedseries_calculate_rma, always_reset=True )

def WPR( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'wpr', source, period, _generatedseries_calculate_williams_r )

def TR( period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'tr', pd.Series([pd.NA] * period, name = 'tr'), period, _generatedseries_calculate_tr )

def ATR2( period:int, timeframe = None )->generatedSeries_c: # The other one using pt is much faster
    if timeframe == None : timeframe = active.timeframe
    tr = timeframe.calcGeneratedSeries( 'tr', timeframe.df['close'], period, _generatedseries_calculate_tr )
    return timeframe.calcGeneratedSeries( 'atr', tr.series(), period, _generatedseries_calculate_rma )

def ATR( period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'atr', timeframe.df['close'], period, _generatedseries_calculate_atr )

def SLOPE( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'slope', source, period, _generatedseries_calculate_slope, always_reset=True )

def BIAS( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'bias', source, period, _generatedseries_calculate_bias )

def CCI( period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'cci', timeframe.df['close'], period, _generatedseries_calculate_cci )

def CFO( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'cfo', source, period, _generatedseries_calculate_cfo )

def FWMA( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'fwma', source, period, _generatedseries_calculate_fwma )

def BBu( source:pd.Series, period:int, mult:float, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'bbu', source, period, _generatedseries_calculate_bbupper, mult )

def BBl( source:pd.Series, period:int, mult:float, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'bbl', source, period, _generatedseries_calculate_bblower, mult )

def IFTrsi( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    rsi = timeframe.calcGeneratedSeries( 'rsi', source, period, _generatedseries_calculate_rsi )
    return timeframe.calcGeneratedSeries( 'iftrsi', timeframe.df['close'], period, _generatedseries_calculate_inverse_fisher_rsi )

def Fisher( period:int, signal:float=None, timeframe = None )->tuple[generatedSeries_c, generatedSeries_c]:
    if timeframe == None : timeframe = active.timeframe
    fish = timeframe.calcGeneratedSeries( 'fisher', timeframe.df['close'], period, _generatedseries_calculate_fisher )
    sig = timeframe.calcGeneratedSeries( 'fishersig', timeframe.df['close'], period, _generatedseries_calculate_fisher_signal, signal )
    return fish, sig
    
def AO( period: int = 0, fast: int = 5, slow: int = 34, timeframe = None ) -> generatedSeries_c:
    if timeframe is None: timeframe = active.timeframe
    param = (fast, slow)
    return timeframe.calcGeneratedSeries('ao', timeframe.df['close'], period, _generatedseries_calculate_ao, param)

def BRAR( period:int, signal:float=None, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    br = timeframe.calcGeneratedSeries( 'br', timeframe.df['close'], period, _generatedseries_calculate_br )
    ar = timeframe.calcGeneratedSeries( 'ar', timeframe.df['close'], period, _generatedseries_calculate_ar )
    return br, ar

def CG( source:pd.Series, period:int, timeframe = None )->generatedSeries_c:
    if timeframe == None : timeframe = active.timeframe
    return timeframe.calcGeneratedSeries( 'cg', source, period, _generatedseries_calculate_cg )





# #
# # OTHER NOT GENERATED SERIES
# #

def BollingerBands( source:pd.Series, period:int, mult:float = 2.0 )->tuple[generatedSeries_c, generatedSeries_c, generatedSeries_c]:
    """
    Returns the Bollinger Bands (basis, upper, lower) for the given source series and period.

    Args:
        source (pd.Series): The input pandas Series to calculate the Bollinger Bands on.
        period (int): The period/window for the Bollinger Bands calculation.

    Returns:
        Tuple[generatedSeries_c, generatedSeries_c, generatedSeries_c]: The basis (SMA), upper band, and lower band as generatedSeries_c objects.
    """
    BBbasis = SMA(source, period)
    STDEV(source, period)
    BBupper = active.timeframe.calcGeneratedSeries( 'bbu', source, period, _generatedseries_calculate_bbupper, mult )
    BBlower = active.timeframe.calcGeneratedSeries( 'bbl', source, period, _generatedseries_calculate_bblower, mult )
    return BBbasis, BBupper, BBlower


def indexWhenTrue( source ):
    """
    Returns the index of the most recent (from the end) True value in the given series.

    Args:
        source: The input series or generatedSeries_c to inspect.

    Returns:
        int or None: The index position of the most recent True value, or None if not found.
    """
    if( not isinstance(source, pd.Series ) ):
        if( isinstance( source, generatedSeries_c) ):
            source = source.series()
        else:
            raise ValueError( "calcIndexWhenTrue must be called with a series" )
    boolean_source = source.astype(bool) if source.dtype != bool else source
    return boolean_source[::-1].idxmax() if source.any() else None


def barsSince( source ):
    """
    Returns the number of bars since the last True value in the given series.

    Args:
        source: The input series or generatedSeries_c to inspect.

    Returns:
        int or None: The number of bars since the last True value, or None if not found.
    """
    index_when_true = indexWhenTrue( source )
    if( index_when_true == None ):
        return None
    return active.barindex - index_when_true


def indexWhenFalse( source ):
    """
    Returns the index of the most recent (from the end) False value in the given series.

    Args:
        source: The input series or generatedSeries_c to inspect.

    Returns:
        int: The index position of the most recent False value, or 0 if not found or series is empty.
    """
    if( not isinstance(source, pd.Series ) ):
        if( isinstance( source, generatedSeries_c) ):
            source = source.series()
        else:
            raise ValueError( "calcIndexWhenFalse must be called with a series" )
    if not source.empty and source.any():
        # Ensure the source is boolean
        boolean_source = source.astype(bool) if source.dtype != bool else source
        
        # Reverse the boolean source and Calculate the cumulative sum of the negated values
        cumsum_negated = (~boolean_source[::-1]).cumsum()
        first_false_index = cumsum_negated[cumsum_negated == 1].index.min() # Find the first 1 in the cumulative sum (which means the first False in the original series from the end)

        return first_false_index
    else:
        return 0  # Return 0 if the series is empty or has no True values''


def barsWhileTrue( barindex, source ):
    """
    Returns the number of bars for which the given series has continuously been True, up to the current barindex.

    Args:
        barindex: The current bar index.
        source: The input series or generatedSeries_c to inspect.

    Returns:
        int or None: The number of bars while True, or None if not found.
    """
    index_when_false = indexWhenFalse( source )
    if( index_when_false == None ):
        return None
    return active.barindex - index_when_false


def crossingUp( self, other ):
    """
    Determines if 'self' crosses up over 'other' between the previous and current bar.

    Args:
        self: The first value/series.
        other: The second value/series to compare against.

    Returns:
        bool: True if a crossing up occurred, False otherwise.
    """
    if isinstance( self, generatedSeries_c ):
        return self.crossingUp( other )
    
    if isinstance( self, int ):
        self = float(self)

    if isinstance( other, int ):
        other = float(other)

    if( isinstance(self, float) and isinstance(other, float) ):
        print( "* WARNING: crossinUp: Two static values can never cross" )
        return False
    
    self_old = 0
    self_new = 0
    other_old = 0
    other_new = 0
    if isinstance( self, pd.Series ):
        if( len(self) < 2 or active.barindex < 1 ):
            return False
        self_old = self.iloc[active.barindex-1]
        self_new = self.iloc[active.barindex]
        if isinstance( other, pd.Series ):
            if( len(other) < 2 ):
                return False
            other_old = other.iloc[active.barindex-1]
            other_new = other.iloc[active.barindex]
        elif isinstance( other, generatedSeries_c ):
            if( other.timestamp == 0 or len(other.series()) < 2 or active.barindex < 1 ):
                return False
            other_old = other.value(1)
            other_new = other.value()
        else:
            try:
                float(other)
            except ValueError:
                return False
            else:
                other_old = float(other)
                other_new = float(other)
    else:
        try:
            float(self)
        except ValueError:
            print( "crossinUp: Unsupported type", type(self) )
            return False
        else:
            return crossingDown( other, self )

    return ( self_old <= other_old and self_new >= other_new and self_old != self_new )


def crossingDown( self, other ):
    """
    Determines if 'self' crosses down below 'other' between the previous and current bar.

    Args:
        self: The first value/series.
        other: The second value/series to compare against.

    Returns:
        bool: True if a crossing down occurred, False otherwise.
    """
    if isinstance( self, generatedSeries_c ):
        return self.crossingDown( other )
    
    if isinstance( self, int ):
        self = float(self)

    if isinstance( other, int ):
        other = float(other)

    if( isinstance(self, float) and isinstance(other, float) ):
        print( "* WARNING: crossinDown: Two static values can never cross" )
        return False
    
    self_old = 0
    self_new = 0
    other_old = 0
    other_new = 0
    if isinstance( self, pd.Series ):
        if( len(self) < 2 or active.barindex < 1 ):
            return False
        self_old = self.iloc[active.barindex-1]
        self_new = self.iloc[active.barindex]
        if isinstance( other, pd.Series ):
            if( len(other) < 2 ):
                return False
            other_old = other.iloc[active.barindex-1]
            other_new = other.iloc[active.barindex]
        elif isinstance( other, generatedSeries_c ):
            if( other.timestamp == 0 or len(other.series()) < 2 or active.barindex < 1 ):
                return False
            other_old = other.value(1)
            other_new = other.value()
        else:
            try:
                float(other)
            except ValueError:
                return False
            else:
                other_old = float(other)
                other_new = float(other)
    else:
        try:
            float(self)
        except ValueError:
            print( "crossinDown: Unsupported type", type(self) )
            return False
        else:
            return crossingUp( other, self )

    return ( self_old >= other_old and self_new <= other_new and self_old != self_new )

def crossing( self, other ):
    """
    Determines if 'self' crosses either up or down with respect to 'other' between the previous and current bar.

    Args:
        self: The first value/series.
        other: The second value/series to compare against.

    Returns:
        bool: True if a crossing (up or down) occurred, False otherwise.
    """
    return crossingUp( other, self ) or crossingDown( other, self )
