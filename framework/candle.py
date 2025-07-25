

class candle_c:
    def __init__( self, source = None ):
        self.timestamp = 0
        self.open = 0.0
        self.high = 0.0
        self.low = 0.0
        self.close = 0.0
        self.volume = 0.0
        self.bottom = 0.0
        self.top = 0.0

        self.index = 0
        self.timeframemsec = 0

        # This is only really used for the realtime candle. Closed candles ramining time is always zero
        self.remainingmsec = 0
        self.remainingseconds = 0
        self.remainingminutes = 0
        self.remaininghours = 0
        self.remainingdays = 0

        if source:
            self.updateFromSource(source)
        

    def updateFromSource(self, source):
        if source and isinstance(source, list):
            self.timestamp = int(source[0])
            self.open = source[1]
            self.high = source[2]
            self.low = source[3]
            self.close = source[4]
            self.volume = source[5]
            self.top = max(self.open, self.close)
            self.bottom = min(self.open, self.close)
                
    def str( self ):
        return f'timestamp:{self.timestamp} open:{self.open} high:{self.high} low:{self.low} close:{self.close} volume:{self.volume}'
    
    def updateRemainingTime( self ):
        from datetime import datetime
        if( self.timestamp <= 0 ):
            return
        
        endTime = self.timestamp + self.timeframemsec
        currentTime = datetime.now().timestamp() * 1000
        if( currentTime >= endTime ):
            self.remainingmsec = self.remainingdays = self.remaininghours = self.remainingminutes = self.remainingseconds = 0
            return
        
        self.remainingmsec = endTime - currentTime
        sec = self.remainingmsec // 1000

        # Calculate days, hours, minutes, and seconds
        self.remainingdays = sec // 86400  # 86400 seconds in a day
        self.remaininghours = (sec % 86400) // 3600  # Remaining seconds divided by seconds in an hour
        self.remainingminutes = (sec % 3600) // 60  # Remaining seconds divided by seconds in a minute
        self.remainingseconds = sec % 60  # Remaining seconds
    
    def remainingTimeStr( self ):
        rtstring = ''
        if self.remainingdays > 0:
            rtstring = f"{int(self.remainingdays)}:"  # Days do not need two digits
        if self.remaininghours > 0 or self.remainingdays > 0:
            rtstring += f"{int(self.remaininghours):02}:"  # Ensure two digits for hours if there are days
        rtstring += f"{int(self.remainingminutes):02}:{int(self.remainingseconds):02}"  # Ensure two digits for minutes and seconds
        return rtstring
    
    def tickData( self ):
        return [ self.timestamp, self.open, self.high, self.low, self.close, self.volume ]

    def print( self ):
        print( self.str() )