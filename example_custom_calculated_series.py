from algorizer import stream_c, timeframe_c, generatedSeries_c

#
# This is an example on how you can create your custom auto-calculated generatedSeries_c
# The CG oscilator is already available in calc
#

import numpy as np
def _generatedseries_calculate_centerofgravity(series: np.ndarray, period: int, dataset: np.ndarray, cindex:int, param=None) -> np.ndarray:
    """
    Calculation function for Center of Gravity (CG) oscillator.
    """
    arr = series.astype(np.float64)
    length = len(arr)
    cg = np.full(length, np.nan, dtype=np.float64)
    for i in range(period - 1, length):
        window = arr[i - period + 1:i + 1]
        if np.all(np.isnan(window)):
            continue
        weights = np.arange(1, period + 1)[::-1]
        denominator = np.nansum(window)
        if denominator == 0:
            cg[i] = np.nan
        else:
            cg[i] = np.nansum(window * weights) / denominator
    return cg

def CG( source:generatedSeries_c, period:int )->generatedSeries_c:
    """
    Factory function for Center of Gravity (CG) oscillator.
    """
    # Notice: Some calculations require the array to be recalculated in full with every update. In that case you
    # set 'always_reset' to True. In case of doubt always start at false and watch it run realtime for a couple of
    # candles. If it missbehaves in realtime it most likely needs the reset, otherwise it doesn't.
    return source.timeframe.calcGeneratedSeries( 'centergravity', source, period, _generatedseries_calculate_centerofgravity, always_reset = False )

# Tip: AIs are pretty good at creating the calculation functions. Provide their context with the series.py and stream.py files.
# Tell them to analize the code paying special attention to the generatedSeries_c class, specially to the 
# generatedSeries_c.calculate_full and generatedSeries_c.update methods and to look at the other calculation
# functions and tell them what calculation function you want them to write. In most cases they'll succeed.

def runCloseCandle( timeframe:timeframe_c, open, high, low, close, volume, top, bottom ):
    cg = CG(close, 14)
    cg.plot('subpanel')

def event( stream:stream_c, event:str, param, numparams ):
    return

if __name__ == '__main__':
    stream = stream_c( 'LDO/USDT:USDT', 'bitget', ['1m'], [runCloseCandle], event, 25000 )

    stream.registerPanel('subpanel', 1.0, 0.2 )
    stream.createWindow( '1m' )

    stream.run()
