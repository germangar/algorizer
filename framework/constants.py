# the big beautiful enum

from enum import IntEnum

class c(IntEnum):  # Ultra-short class name
    LONG = 1
    SHORT = -1
    IDLE = 0

    BUY = 1
    SELL = -1

    PIVOT_HIGH = 1
    PIVOT_LOW = -1
    
    PLOT_LINE = 0
    PLOT_HIST = 1

    # Columns in the dataframe by index
    DF_TIMESTAMP = 0
    DF_OPEN = 1
    DF_HIGH = 2
    DF_LOW = 3
    DF_CLOSE = 4
    DF_VOLUME = 5
    DF_TOP = 6
    DF_BOTTOM = 7

    # window panel types (I don't even know if vertical is possible right now)
    PANEL_HORIZONTAL = 0
    PANEL_VERTICAL = 0