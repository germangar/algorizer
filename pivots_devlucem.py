
'''

This pivots class is based of devLucem's ZigZag library, but it's too slow
when ran in python, and I'm not even sure I got to reproduce it 100% right.
I'm keeping the code here just in case I ever want to return to it, but
it's not imported anywhere in the algorizer code.

'''




class pivotsLucem_c:
    def __init__(self, high: pd.Series, low: pd.Series, depth: int = 64, deviation: float = 2, backstep: int = 32):
        self.depth = depth
        self.deviation = deviation
        self.backstep = backstep
        self.mintick = active.timeframe.stream.mintick
        
        # State variables
        self.direction = 0
        self.heightDiffTopEnough = False
        self.heightDiffBottomEnough = False

        # Initialize tracking columns and get their indices
        self.initialized = False

        # State variables
        self.zindex = active.barindex
        self.z1index = active.barindex
        self.z2index = active.barindex

        # confirmed pivots
        self.new = 0
        self.last_pivot_index = 0
        self.last_pivot_price = 0.0
        self.last_pivot_timestamp = 0
        self.last_direction = 0

    def makeName(self):
        return f'OHDTE{active.timeframe.timeframeStr}{self.depth}{self.deviation}{self.backstep}'

    def update2(self, high: pd.Series, low: pd.Series):
        self.new = 0
        barindex = active.barindex
        if barindex < self.depth:
            return (self.direction, self.z1index, self.z2index)
        
        hbGenSeries = highestbars(high, self.depth)
        lbGenSeries = lowestbars(low, self.depth)
        hb_offset = hbGenSeries.value()
        lb_offset = lbGenSeries.value()

        # store in dataframe columns the old results from heighDiffTopEnough and heightDiffBottomEnough for vectorized comparison
        if not self.initialized:
            self.initializeTrackingColumns()
        else:
            if( not active.timeframe.backtesting ):
                active.timeframe.df.iat[barindex, self.OHDTEcolumnindex] = not self.heightDiffTopEnough
                active.timeframe.df.iat[barindex, self.OHDBEcolumnindex] = not self.heightDiffBottomEnough

        self.heightDiffTopEnough = high[barindex-hb_offset] - high[barindex] > self.deviation * self.mintick
        self.heightDiffBottomEnough = low[barindex] - low[barindex-lb_offset] > self.deviation * self.mintick

        hr_gs = barsSinceSeries(active.timeframe.df.iloc[:, self.OHDTEcolumnindex], self.depth)
        lr_gs = barsSinceSeries(active.timeframe.df.iloc[:, self.OHDBEcolumnindex], self.depth)

        condition = barsSinceSeries(hr_gs.series() <= lr_gs.series(), self.depth + 1).value()
        if not condition:
            condition = 0

        self.last_direction = self.direction

        new_direction = -1 if condition >= self.backstep else 1
        

        if new_direction != self.direction:
            self.z1index = self.z2index
            self.z2index = self.zindex
            self.direction = new_direction

        high_now = high.at[barindex]
        low_now = low.at[barindex]

        if new_direction > 0:
            if  high_now > high.at[self.z2index]:
                self.z2index = barindex
                self.zindex = barindex

            if low_now < low.at[self.zindex]:
                self.zindex = barindex

        if new_direction < 0:
            if low_now < low.at[self.z2index]:
                self.z2index = barindex
                self.zindex = barindex
            if high_now > high.at[self.zindex]:
                self.zindex = barindex

        return (self.direction, self.z1index, self.z2index)
    
    def update(self, high: pd.Series, low: pd.Series):
        direction, confirmed, current = self.update2(high, low)
        if( self.last_direction != direction ):
            self.last_pivot_index = confirmed
            self.last_pivot_price = high[confirmed]
            self.last_pivot_timestamp = int(active.timeframe.df['timestamp'].at[confirmed])
            self.new = direction

    def initializeTrackingColumns(self):
        """Initialize columns for tracking height differences and store their indices"""
        df = active.timeframe.df
        
        # Initialize result arrays
        height_diff_top = pd.Series(False, index=df.index)
        height_diff_bottom = pd.Series(False, index=df.index)
        
        # Calculate only from depth onwards
        for i in range(self.depth, len(df)):
            window = slice(i - self.depth + 1, i + 1)
            window_high = df.high.iloc[window]
            window_low = df.low.iloc[window]
            
            # Find highest and lowest in window
            highest_idx = i - self.depth + 1 + window_high.argmax()
            lowest_idx = i - self.depth + 1 + window_low.argmin()
            
            # Calculate differences
            height_diff_top.iloc[i] = df.high.iloc[highest_idx] - df.high.iloc[i] > self.deviation * self.mintick
            height_diff_bottom.iloc[i] = df.low.iloc[i] - df.low.iloc[lowest_idx] > self.deviation * self.mintick

        # Create and store columns
        active.timeframe.df[self.makeName()+'T'] = ~height_diff_top
        active.timeframe.df[self.makeName()+'B'] = ~height_diff_bottom
        self.OHDTEcolumnindex = active.timeframe.df.columns.get_loc(self.makeName()+'T')
        self.OHDBEcolumnindex = active.timeframe.df.columns.get_loc(self.makeName()+'B')

        self.initialized = True


pivotsDNow:pivotsLucem_c = None
def pivotsDevlucem( high:pd.Series, low:pd.Series, depth: int = 64, deviation: float = 2, backstep: int = 32)->pivotsLucem_c:
    global pivotsDNow
    if pivotsNow == None:
        pivotsDNow = pivotsLucem_c(high, low, depth, deviation, backstep)

    pivotsDNow.update(high, low)
    return pivotsDNow

