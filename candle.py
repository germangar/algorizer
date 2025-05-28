

class candle_c:
    def __init__( self ):
        self.timestamp = 0
        self.open = 0.0
        self.high = 0.0
        self.low = 0.0
        self.close = 0.0
        self.volume = 0.0
        self.bottom = 0.0
        self.top = 0.0

        self.timeframemsec = 0

        self.remainingmsec = 0
        self.remainingseconds = 0
        self.remainingminutes = 0
        self.remaininghours = 0
        self.remainingdays = 0

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

    def print( self ):
        print( self.str() )