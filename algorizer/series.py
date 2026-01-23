from typing import Union, TYPE_CHECKING


import numpy as np
import time

from .constants import constants as c
from . import active
from . import tools


if TYPE_CHECKING:
    from .stream import timeframe_c

NumericScalar = Union[float, int]
OperandType = Union['generatedSeries_c', NumericScalar]

# #
# # GENERATED SERIES : These are series of values that are calculated always using the same formula
# #

class generatedSeries_c:
    
    def __init__(self, name: str, source: np.ndarray, period:int= 1, func=None, param=None, always_reset:bool= False):

        self.timeframe: 'timeframe_c' = active.timeframe
        timeframe = self.timeframe
        
        # These are generated series created by the user, used by plots, or the built in (ohlcv) columns
        if func is None:
            self.name = name
            self.source_name = name
            self.column_index = -1
            self.period = 1 # ignore the period

            self.param = None
            self.func = None
            self.lastUpdatedTimestamp = timeframe.timestamp
            self.alwaysReset = False
            self._is_generated_series = False # do not touch. 
            
            # ohlcv (and top/bottom) are special. They don't need a column created.
            if self.name == 'timestamp' : self.column_index = c.DF_TIMESTAMP
            if self.name == 'open' : self.column_index = c.DF_OPEN
            if self.name == 'high' : self.column_index = c.DF_HIGH
            if self.name == 'low' : self.column_index = c.DF_LOW
            if self.name == 'close' : self.column_index = c.DF_CLOSE
            if self.name == 'volume' : self.column_index = c.DF_VOLUME
            if self.name == 'top' : self.column_index = c.DF_TOP
            if self.name == 'bottom' : self.column_index = c.DF_BOTTOM

            # create a column
            if self.column_index == -1:
                assert name not in timeframe.generatedSeries.keys(), f"A generatedSeries_c with the name '{name}' already exists"
                self.column_index = timeframe.dataset_createColumn()

            # register the column
            timeframe.generatedSeries[self.name] = self
            return
        
        if not isinstance( source, generatedSeries_c ):
            raise ValueError( f"Source must be 'generatedSeries_c' type [{name}] for series with a func" )

        testname = tools.generatedSeriesNameFormat(name, source, period)
        if testname in timeframe.generatedSeries.keys():
            raise ValueError( f"A generatedSeries_c with the name '{testname}' already exists" )
        self.name = testname
        self.column_index = -1
        self.source_name = source.name
        self.period = max(period, 1) if period is not None else len(source)
        self.param = param
        self.func = func
        self.lastUpdatedTimestamp = 0
        self.alwaysReset = always_reset
        self._is_generated_series = True # do not touch. 
        
        # create a column and register it
        self.column_index = timeframe.dataset_createColumn()
        timeframe.generatedSeries[self.name] = self
        
        
    def calculate_full( self, source ):
        if not self.func:
            raise ValueError( f"Tried to initialize {self.name} without a func" )
        
        # FIXME: All of these are overdone. They're temporary. Remove them
        assert isinstance(source, generatedSeries_c), f"Source {source.name} must be generatedSeries_c type"  # temporary while make sure everything is
        assert self.timeframe == active.timeframe
        assert self.column_index != -1
        assert self.name in self.timeframe.generatedSeries.keys()
        assert self.source_name in self.timeframe.generatedSeries.keys()
        
        if len(source) < self.period:
            return
        
        timeframe = self.timeframe

        if self.lastUpdatedTimestamp >= timeframe.timestamp:
            return

        start_time = time.time()

        # Call the func, which must now accept a 1D numpy array as the source and the 2D array as "dataset"
        # Expect func to return a 1D numpy array of values, aligned with the full dataset length
        array = timeframe.dataset[:,source.column_index]
        values = self.func(array, self.period, timeframe.dataset, self.column_index, self.param)
        if isinstance(values, (list, tuple)):
            values = np.array(values, dtype=np.float64)

        timeframe.dataset[:, self.column_index] = values

        # Update the timestamp from the last row
        barindex = len(timeframe.dataset) - 1
        self.lastUpdatedTimestamp = int(timeframe.dataset[barindex, c.DF_TIMESTAMP])

        if timeframe.stream.initializing:
            print(f"Initialized {self.name} ({self.column_index}). Elapsed time: {time.time() - start_time:.2f} seconds")

    def update( self, source ):
        if not self.func:
            self.lastUpdatedTimestamp = self.timeframe.timestamp
            return

        assert isinstance(source, generatedSeries_c), "Source must be generatedSeries_c type"  # temporary while make sure everything is
        assert self.timeframe == active.timeframe
        
        timeframe = self.timeframe

        # has this row already been updated?
        if self.lastUpdatedTimestamp >= timeframe.timestamp:
            return

        # if non existent or needs reset, initialize
        if self.alwaysReset or self.lastUpdatedTimestamp == 0:
            self.calculate_full(source)
            return

        # slice the required block for current calculation
        period_slice = timeframe.dataset[-self.period:, source.column_index]

        # func should return a 1D array or scalar; we want the most recent value
        newval = self.func(period_slice, self.period, timeframe.dataset, self.column_index, self.param)
        
        if isinstance(newval, (np.ndarray, list, tuple)):
            newval = newval[-1]
            
        timeframe.dataset[timeframe.barindex, self.column_index] = newval
        self.lastUpdatedTimestamp = timeframe.timestamp


    def iloc( self, index = -1 ):
        barindex = self.timeframe.barindex

        if self.timeframe != active.timeframe :
            timestamp = active.timeframe.timestamp + ( (index+1) * self.timeframe.timeframeMsec )
            return self.timeframe.valueAtTimestamp( self.name, timestamp )

        # Original iloc logic for other indices
        if index < 0:
            index = barindex + 1 + index
        
        # Ensure the index is within valid bounds after translation/clamping
        if index < 0 or index >= len(self.timeframe.dataset):
            return np.nan # Return NaN for out-of-bounds access
            
        return self.timeframe.dataset[index, self.column_index]
    iat = iloc # alias for the same method
    
    def current( self ):
        '''returns the last value in the series'''
        return self.__getitem__(self.timeframe.barindex)
    
    def series(self)->np.ndarray:
        try:
            return self.timeframe.dataset[:,self.column_index]
        except Exception as e:
            raise ValueError( "series() method couldn't produce an array" )
        
    def tolist(self)->list:
        return self.timeframe.dataset[:,self.column_index].tolist()
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            key = slice(start, stop, step)
        else:
            if key < 0:
                key = self.timeframe.barindex + 1 + key
            if key < 0 or key >= len(self.timeframe.dataset):
                return np.nan # Return NaN for out-of-bounds access
        
        return self.timeframe.dataset[key, self.column_index]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            # Handle slice keys like [0:5]
            start, stop, step = key.indices(len(self))
            key = slice(start, stop, step)
        elif key < 0:
            key = self.timeframe.barindex + 1 + key
        
        # Allow both scalar and array assignment
        self.timeframe.dataset[key, self.column_index] = value

    def __len__(self):
        return self.timeframe.dataset.shape[0]

    def __bool__(self):
        raise ValueError(
            "The truth value of a generatedSeries_c with more than one element is ambiguous. "
            "To combine series logically, use the bitwise operators '&' (and), '|' (or). "
            "To check a specific value, access it directly (e.g., `if my_series[-1]:`)."
        )
    
    def _lenError(self, other):
        if isinstance(other, generatedSeries_c):
            if self.timeframe != other.timeframe:
                raise ValueError( f"Can't operate on series from different timeframes. {self.name} [{self.timeframe.timeframeStr}] - {other.name} [{other.timeframe.timeframeStr}]" )
        raise ValueError( f"Can't operate on series of different lengths. {self.name} != {other.name}" )
    
    def __add__(self, other):
        if isinstance(other, generatedSeries_c):
            if len(self) != len(other):
                self._lenError(other)
        elif not np.isscalar(other):
            raise TypeError( f"Can't add type {type(other)} to generatedSeries_c" )
        return addSeries(self, other)

    def __radd__(self, other):
        if np.isscalar(other):
            return addScalar(other, self)
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, generatedSeries_c):
            if len(self) != len(other):
                self._lenError(other)
        elif not np.isscalar(other):
            raise TypeError( f"Can't subtract type {type(other)} to generatedSeries_c" )
        return subtractSeries(self, other)

    def __rsub__(self, other):
        if np.isscalar(other):
            return subtractScalar(other, self)
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, generatedSeries_c):
            if len(self) != len(other):
                self._lenError(other)
        elif not np.isscalar(other):
            raise TypeError( f"Can't multiply type {type(other)} to generatedSeries_c" )
        return multiplySeries(self, other)

    def __rmul__(self, other):
        if np.isscalar(other):
            return multiplyScalar(other, self)
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, generatedSeries_c):
            if len(self) != len(other):
                self._lenError(other)
        elif not np.isscalar(other):
            raise TypeError( f"Can't divide generatedSeries_c by type {type(other)}" )
        return divideSeries(self, other)

    def __rtruediv__(self, other):
        if np.isscalar(other):
            return divideScalar(other, self)
        raise TypeError("rtruediv only defined for const / series")
    
    def __neg__(self):
        return multiplySeries(self, -1)

    def __lt__(self, other):
        if isinstance(other, generatedSeries_c):
            if len(self) != len(other):
                self._lenError(other)
        elif not np.isscalar(other):
            raise TypeError("Unsupported operand type for <")
        return lessSeries(self, other)
    
    def __rlt__(self, other):
        if np.isscalar(other):
            return lessScalar(other, self)
        raise TypeError("Unsupported reversed operand for <")

    def __le__(self, other):
        if isinstance(other, generatedSeries_c):
            if len(self) != len(other):
                self._lenError(other)
        elif not np.isscalar(other):
            raise TypeError("Unsupported operand type for <=")
        return lessOrEqualSeries(self, other)
    
    def __rle__(self, other):
        if np.isscalar(other):
            return lessOrEqualScalar(other, self)
        raise TypeError("Unsupported reversed operand for <=")

    def __gt__(self, other):
        if isinstance(other, generatedSeries_c):
            if len(self) != len(other):
                self._lenError(other)
        elif not np.isscalar(other):
            raise TypeError("Unsupported operand type for >")
        return greaterSeries(self, other)
    
    def __rgt__(self, other):
        if np.isscalar(other):
            return greaterScalar(self, other)
        raise TypeError("Unsupported reversed operand for >")

    def __ge__(self, other):
        if isinstance(other, generatedSeries_c):
            if len(self) != len(other):
                self._lenError(other)
        elif not np.isscalar(other):
            raise TypeError("Unsupported operand type for >=")
        return greaterOrEqualSeries(self, other)
    
    def __rge__(self, other):
        if np.isscalar(other):
            return greaterOrEqualScalar(self, other)
        raise TypeError("Unsupported reversed operand for >=")

    def __eq__(self, other):
        if isinstance(other, generatedSeries_c):
            if len(self) != len(other):
                self._lenError(other)
        elif not np.isscalar(other):
            return NotImplemented
        return equalSeries(self, other)
    
    def __req__(self, other):
        if np.isscalar(other):
            return equalScalar(self, other)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, generatedSeries_c):
            if len(self) != len(other):
                self._lenError(other)
        elif not np.isscalar(other):
            return NotImplemented
        return notequalSeries(self, other)

    def __rne__(self, other):
        if np.isscalar(other):
            return notEqualScalar(self, other)
        return NotImplemented
    
    def __and__(self, other):
        if isinstance(other, generatedSeries_c):
            if len(self) != len(other):
                self._lenError(other)
        elif not np.isscalar(other):
            return NotImplemented
        return bitwiseAndSeries(self, other)

    def __rand__(self, other):
        if np.isscalar(other):
            return bitwiseAndScalar(other, self)
        return NotImplemented
    
    def __or__(self, other):
        if isinstance(other, generatedSeries_c):
            if len(self) != len(other):
                self._lenError(other)
        elif not np.isscalar(other):
            return NotImplemented
        return bitwiseOrSeries(self, other)

    def __ror__(self, other):
        if np.isscalar(other):
            return bitwiseOrScalar(other, self)
        return NotImplemented
    
    def __xor__(self, other):
        if isinstance(other, generatedSeries_c):
            if len(self) != len(other):
                self._lenError(other)
        elif not np.isscalar(other):
            return NotImplemented 
        return bitwiseXorSeries(self, other)

    def __rxor__(self, other):
        if np.isscalar(other):
            return bitwiseXorScalar(other, self)
        return NotImplemented

    def __invert__(self):
        return bitwiseNotSeries(self)
    
    def __abs__(self):
        return ABS(self)


    def plot( self, chart_name = None, color = "#8FA7BBAA", style = 'solid', width = 1  ):
        '''* it returns the generatedSeries. Calling plot from the timeframe and the function returns the plot_c but not here*

        source: can either be a series or a value. A series can only be plotted when it is in the dataframe. When plotting a value a series will be automatically created in the dataframe.
        chart_name: Leave empty for the main panel. Use 'panel' for plotting in the subpanel.
        color: in a string. Can be hexadecial '#DADADADA' or rgba format 'rgba(255,255,255,1.0)'
        style: LINE_STYLE = Literal['solid', 'dotted', 'dashed', 'large_dashed', 'sparse_dotted']
        width: int
        '''
        if( self.lastUpdatedTimestamp > 0 ):
            self.timeframe.plot( self.series(), self.name, chart_name, color, style, width )
            return self
    def histogram( self, chart_name = None, color = "#4A545D", margin_top = 0.0, margin_bottom = 0.0 ):
        '''* it returns the generatedSeries. Calling plot from the timeframe and the function returns the plot_c but not here*

        source: can either be a series or a value. A series can only be plotted when it is in the dataframe. When plotting a value a series will be automatically created in the dataframe.
        chart_name: Leave empty for the main panel. Use 'panel' for plotting in the subpanel.
        color: in a string. Can be hexadecial '#DADADADA' or rgba format 'rgba(255,255,255,1.0)'
        style: LINE_STYLE = Literal['solid', 'dotted', 'dashed', 'large_dashed', 'sparse_dotted']
        width: int
        '''
        if( self.lastUpdatedTimestamp > 0 ):
            self.timeframe.histogram( self.series(), self.name, chart_name, color, margin_top, margin_bottom )
            return self
    
    def crossingUp( self, other )->bool:
        '''returns a single boolean indicating if self is crossing up a value or a series'''
        current_self_val = self[-1]
        previous_self_val = self[-2]

        if np.isnan(current_self_val) or np.isnan(previous_self_val):
            return False

        if isinstance( other, generatedSeries_c ):
            current_other_val = other[-1]
            previous_other_val = other[-2]
            if np.isnan(current_other_val) or np.isnan(previous_other_val):
                return False
            # the old value can match but the new value must always be bigger than the counterpart.
            if current_self_val > current_other_val:
                if previous_self_val <= previous_other_val:
                    return True
            return False

        elif np.isscalar( other ):
            try:
                float_other = float(other)
            except ValueError:
                raise ValueError( f"generatedSeries_c.crossingUp: Error casting scalar to float [{type(other)}]" )
                return False
            # the old value can match but the new value must always be bigger than the counterpart.
            if current_self_val > float_other:
                if previous_self_val <= float_other:
                    return True
            return False
        elif np.isnan(other):
            return False
        
        raise ValueError( f"generatedSeries_c.crossingUp: Invalid type 'other' [{type(other)}]" )
    
    def crossingDown( self, other )->bool:
        '''returns a single boolean indicating if self is crossing down a value or a series'''
        current_self_val = self[-1]
        previous_self_val = self[-2]

        if np.isnan(current_self_val) or np.isnan(previous_self_val):
            return False

        if isinstance( other, generatedSeries_c ):
            current_other_val = other[-1]
            previous_other_val = other[-2]
            if np.isnan(current_other_val) or np.isnan(previous_other_val):
                return False
            # the old value can match but the new value must always be smaller than the counterpart.
            if current_self_val < current_other_val:
                if previous_self_val >= previous_other_val:
                    return True
            return False

        elif np.isscalar( other ):
            try:
                float_other = float(other)
            except ValueError:
                raise ValueError( f"generatedSeries_c.crossingUp: Error casting scalar to float [{type(other)}]" )
                return False
            # the old value can match but the new value must always be smaller than the counterpart.
            if current_self_val < float_other:
                if previous_self_val >= float_other:
                    return True
            return False
        elif np.isnan(other):
            return False
        
        raise ValueError( f"generatedSeries_c.crossingUp: Invalid type 'other' [{type(other)}]" )
    
    def crossing( self, other )->bool:
        '''returns a single boolean indicating if self is crossing a value or a series'''
        return self.crossingUp(other) or self.crossingDown(other)
    


# # ---------------------------------------------------------------
# # CALCULATION FUNCTIONS : calculate the series slice or full
# # ---------------------------------------------------------------


def _prepare_param_for_op(param, src_len: int, dataset) -> np.ndarray:
    """
    Normalize 'param' into a 1-D np.float64 array of length src_len aligned to the tail.

    Rules:
    - Python or NumPy scalar -> full array filled with scalar.
    - or 1-D ndarray -> tail-aligned: last src_len values (left-pad with NaNs if shorter).
    - None or empty -> full NaN array.
    - 0-D np.ndarray (numpy scalar) treated like scalar.
    - 2-D arrays raise ValueError.
    """

    # Scalar (covers Python and NumPy scalars)
    if np.isscalar(param):
        val = float(param)
        return np.full(src_len, val, dtype=np.float64)

    # None or empty
    if param is None:
        return np.full(src_len, np.nan, dtype=np.float64)
    
    if isinstance(param, generatedSeries_c):
        param = param.series()
        if len(param) >= src_len:
            return param[-src_len:].astype(np.float64, copy=False)
        return np.asarray(param, dtype=np.float64)

    # coerce to ndarray
    param_arr = np.asarray(param, dtype=np.float64)

    # If param_arr ended up 0-D (numpy scalar), treat like scalar
    if param_arr.ndim == 0:
        return np.full(src_len, float(param_arr), dtype=np.float64)

    # Reject 2-D or higher arrays — callers should pass a column or 1-D array
    if param_arr.ndim != 1:
        raise ValueError("param must be scalar or 1-D array/series")

    p_len = param_arr.shape[0]
    # If param has at least src_len elements, take last src_len
    if p_len >= src_len:
        return param_arr[-src_len:].astype(np.float64, copy=False)

    # p_len < src_len -> left-pad with NaNs so tails align
    pad = np.full(src_len - p_len, np.nan, dtype=np.float64)
    return np.concatenate((pad, param_arr)).astype(np.float64)


def _generatedseries_calculate_add_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) + _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_subtract_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) - _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_multiply_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) * _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_divide_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) / _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_power_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.power(np.asarray(source, dtype=np.float64), _prepare_param_for_op( param, source.shape[0], dataset ))

def _generatedseries_calculate_equal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) == _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_notequal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) != _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_greater_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) > _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_greaterorequal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) >= _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_less_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) < _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_lessequal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: OperandType) -> np.ndarray:
    return np.asarray(source, dtype=np.float64) <= _prepare_param_for_op( param, source.shape[0], dataset )

def _generatedseries_calculate_bitwise_and_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param= None) -> np.ndarray:
    param_array = _prepare_param_for_op( param, source.shape[0], dataset )
    mask = np.isnan(source) | np.isnan(param_array)
    src_int = np.where(mask, 0, source).astype(np.int64)
    prm_int = np.where(mask, 0, param_array).astype(np.int64)
    int_result = np.bitwise_and(src_int, prm_int)
    res = int_result.astype(np.float64)
    res[mask] = np.nan
    return res

def _generatedseries_calculate_bitwise_or_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param) -> np.ndarray:
    param_array = _prepare_param_for_op( param, source.shape[0], dataset )
    mask = np.isnan(source) | np.isnan(param_array)
    src_int = np.where(mask, 0, source).astype(np.int64)
    prm_int = np.where(mask, 0, param_array).astype(np.int64)
    int_result = np.bitwise_or(src_int, prm_int)
    res = int_result.astype(np.float64)
    res[mask] = np.nan
    return res

def _generatedseries_calculate_bitwise_xor_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param) -> np.ndarray:
    param_array = _prepare_param_for_op( param, source.shape[0], dataset )
    mask = np.isnan(source) | np.isnan(param_array)
    src_int = np.where(mask, 0, source).astype(np.int64)
    prm_int = np.where(mask, 0, param_array).astype(np.int64)
    int_result = np.bitwise_xor(src_int, prm_int)
    res = int_result.astype(np.float64)
    res[mask] = np.nan
    return res

def _generatedseries_calculate_bitwise_not_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param) -> np.ndarray:
    mask = np.isnan(source)
    src_int = np.where(mask, 0, source).astype(np.int64)
    int_result = np.bitwise_not(src_int)
    res = int_result.astype(np.float64)
    res[mask] = np.nan
    return res

def _generatedseries_calculate_abs_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: None) -> np.ndarray:
    return np.abs(source)

##### scalars by series

def _generatedseries_calculate_scalar_add_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param + source # Note: param is the scalar, source is the series

def _generatedseries_calculate_scalar_subtract_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param - source

def _generatedseries_calculate_scalar_multiply_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param * source

def _generatedseries_calculate_scalar_divide_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param / source

def _generatedseries_calculate_scalar_power_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return np.power(param, source) # Note the order: scalar (param) first, then series (source)

def _generatedseries_calculate_scalar_equal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param == source

def _generatedseries_calculate_scalar_notequal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param != source

def _generatedseries_calculate_scalar_greater_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param > source

def _generatedseries_calculate_scalar_greaterorequal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param >= source

def _generatedseries_calculate_scalar_less_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param < source

def _generatedseries_calculate_scalar_lessequal_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    return param <= source

def _generatedseries_calculate_scalar_bitwise_and_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    """
    Performs element-wise bitwise AND operation: scalar & series.
    Casts result to np.float64 for storage compatibility.
    """
    int_result = np.bitwise_and(param, source.astype(np.int64)) # to int
    return int_result.astype(np.float64) # back to float

def _generatedseries_calculate_scalar_bitwise_or_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    """
    Performs element-wise bitwise OR operation: scalar | series.
    Casts operands to np.int64 for the bitwise operation, and the result back to np.float64.
    """
    int_result = np.bitwise_or(param, source.astype(np.int64))
    return int_result.astype(np.float64)

def _generatedseries_calculate_scalar_bitwise_xor_series(source: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param: NumericScalar) -> np.ndarray:
    """
    Performs element-wise bitwise XOR operation: scalar ^ series.
    Casts operands to np.int64 for the bitwise operation, and the result back to np.float64.
    """
    int_result = np.bitwise_xor(param, source.astype(np.int64))
    return int_result.astype(np.float64)


# # ---------------------------------------------------------------
# # FACTORY FUNCTIONS: Set up the generated series to be calculated
# # ---------------------------------------------------------------


def _ensure_object_array( data: generatedSeries_c )-> generatedSeries_c:
    """
    Helper function to ensure the input is a series object.
    This replaces the initial type checking and conversion logic.
    """
    if isinstance(data, generatedSeries_c):
        return data
    elif isinstance(data, np.ndarray):
        raise TypeError( "_ensure_object_array: Numpy ndarray is not a valid object" )
    else:
        raise TypeError(f"Unsupported input type: {type(data)}. Expected generatedSeries_c.")

#
########## SERIES By SERIES or SCALARS
#

def addSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"add_{colA.column_index}_s{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"add_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")

    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_add_series, param= colB )

def subtractSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"sub_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"sub_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries( name, colA, 1, _generatedseries_calculate_subtract_series, param= colB )

def multiplySeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"mul_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"mul_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_multiply_series, param= colB)

def divideSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"div_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"div_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_divide_series, param= colB)

def powerSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"pow_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"pow_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_power_series, param= colB)

def equalSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"eq_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"eq_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_equal_series, param= colB)

def notequalSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"neq_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"neq_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_notequal_series, param= colB)

def greaterSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"gr_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"gr_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_greater_series, param= colB)

def greaterOrEqualSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"gre_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"gre_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_greaterorequal_series, param= colB)

def lessSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"lt_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"lt_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_less_series, param= colB)

def lessOrEqualSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if isinstance( colB, NumericScalar ):
        name = f"le_{colA.column_index}_{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"le_{colA.column_index}_{colB.column_index}" # Using resolved indices/names for consistent naming
        if len(colA) != len(colB): # Ensure arrays have compatible shapes for element-wise operation (usually same length)
            raise ValueError("Operands must have the same shape for element-wise addition.")
        
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_lessequal_series, param= colB)

def bitwiseAndSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if np.isscalar( colB ):
        name = f"and_{colA.column_index}_s{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"and_{colA.column_index}_{colB.column_index}"
        if len(colA) != len(colB):
            raise ValueError("Operands must have the same shape for element-wise bitwise AND.")

    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_bitwise_and_series, param= colB )

def bitwiseOrSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if np.isscalar( colB ):
        name = f"or_{colA.column_index}_s{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"or_{colA.column_index}_{colB.column_index}"
        if len(colA) != len(colB):
            raise ValueError("Operands must have the same shape for element-wise bitwise OR.")

    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_bitwise_or_series, param= colB )


def bitwiseXorSeries(colA: generatedSeries_c, colB: generatedSeries_c | NumericScalar) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    if np.isscalar( colB ):
        name = f"xor_{colA.column_index}_s{colB}"
    else:
        colB = _ensure_object_array( colB )
        name = f"xor_{colA.column_index}_{colB.column_index}"
        if len(colA) != len(colB):
            raise ValueError("Operands must have the same shape for element-wise bitwise XOR.")

    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_bitwise_xor_series, param= colB )

def bitwiseNotSeries(colA: generatedSeries_c) -> generatedSeries_c:
    colA = _ensure_object_array( colA )
    timeframe = colA.timeframe
    # The name only needs the index of the single operand
    name = f"not_{colA.column_index}"
    return timeframe.calcGeneratedSeries(name, colA, 1, _generatedseries_calculate_bitwise_not_series, param= None )

def ABS(source: generatedSeries_c)->generatedSeries_c:
    source = _ensure_object_array( source )
    timeframe = source.timeframe
    return timeframe.calcGeneratedSeries( "abs", source, 1, _generatedseries_calculate_abs_series )

#
########## SCALARS By SERIES
#

def addScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"add_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries( name, series, 1, _generatedseries_calculate_scalar_add_series, param= scalar )

def subtractScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"sub_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_subtract_series, param= scalar)

def multiplyScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"mul_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_multiply_series, param= scalar)

def divideScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"div_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_divide_series, param= scalar)

def powerScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"pow_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_power_series, param= scalar)

def equalScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"eq_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_equal_series, param= scalar)

def notEqualScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"neq_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_notequal_series, param= scalar)

def greaterScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"gt_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_greater_series, param= scalar)

def greaterOrEqualScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"ge_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_greaterorequal_series, param= scalar)

def lessScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"lt_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_less_series, param= scalar)

def lessOrEqualScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"le_{scalar}_{series.column_index}" # Consistent naming for scalar first
    return timeframe.calcGeneratedSeries(name, series, 1, _generatedseries_calculate_scalar_lessequal_series, param= scalar)

def bitwiseAndScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"and_{scalar}_{series.column_index}" 
    return timeframe.calcGeneratedSeries( name, series, 1, _generatedseries_calculate_scalar_bitwise_and_series, param= scalar )

def bitwiseOrScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"or_{scalar}_{series.column_index}" 
    return timeframe.calcGeneratedSeries( name, series, 1, _generatedseries_calculate_scalar_bitwise_or_series, param= scalar )

def bitwiseXorScalar(scalar: NumericScalar, series: generatedSeries_c) -> generatedSeries_c:
    series = _ensure_object_array( series )
    timeframe = series.timeframe
    name = f"xor_{scalar}_{series.column_index}" 
    return timeframe.calcGeneratedSeries( name, series, 1, _generatedseries_calculate_scalar_bitwise_xor_series, param= scalar )

