# the big beautiful enum

from enum import IntEnum

class c(IntEnum):  # Ultra-short class name
    PIVOT_HIGH = 1
    PIVOT_LOW = -1
    LONG = 1
    SHORT = -1
    

    # Columns in the dataframe by index
    DF_TIMESTAMP = 0
    DF_OPEN = 1
    DF_HIGH = 2
    DF_LOW = 3
    DF_CLOSE = 4
    DF_VOLUME = 5
    DF_TOP = 6
    DF_BOTTOM = 7