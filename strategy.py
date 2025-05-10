
import pandas as pd
from candle import candle_c

from algorizer import getRealtimeCandle
from algorizer import createMarker

SHORT = -1
LONG = 1

class strategy_c:
    def __init__( self ):
        return
    
class position_c:
    def __init__( self ):
        self.active = False
        self.type = 0
        self.size = 0.0
        self.priceAvg = 0.0
        self.leverage = 1

    def update( self, optype:int, price, quantity, leverage ):
        if( self.type == LONG and optype == SHORT ):

            self.size -= quantity  # FIXME
            if( self.size > 0.0 ):
                return
            
            if( self.size == 0.0 ):
                self.active = False
                return
            
            #open a new reversed position
            quantity = -self.size
            openPosition( optype, price, quantity, leverage )

    def close(self, price):
        if( not self.active ):
            return
        
        if self.type == SHORT:
            self.update( LONG, price, self.size, self.leverage )
            createMarker( "❌" )
            print( f'closed short position {len(positions)}' )

        elif self.type == LONG:
            self.update( SHORT, price, self.size, self.leverage )
            createMarker( "❌" )
            print( f'closed long position {len(positions)}' )


    
positions:position_c = []

def openPosition( posType:int, price, quantity, leverage )->position_c:
    pos = position_c()
    pos.active = True
    pos.type = posType
    pos.size = quantity
    pos.leverage = leverage
    pos.priceAvg = price

    positions.append( pos )
    icon = '🔴' if pos.type == SHORT else '🟢'
    createMarker( icon )
    print( f'opened position: {len(positions)}' )
    return pos

def getActivePosition()->position_c:
    if len(positions) == 0:
        return None
    pos = positions[-1]
    if( pos.active != True ):
        return None
    
    return pos

def direction():
    pos = getActivePosition()
    if( pos == None or not pos.active ):
        return 0
    return pos.type
    

def order( cmd:str, price:float, quantity:float, leverage:int=1 ):
    if cmd is None: 
        return # FIXME
    
    cmd = cmd.lower()

    if cmd == 'buy':
        pos = getActivePosition()
        if( pos == None or not pos.active ):
            print('buy')
            pos = openPosition( LONG, price, quantity, leverage )
            return
        
        pos.update( LONG, price, quantity, leverage )

    elif cmd == 'sell':
        pos = getActivePosition()
        if( pos == None or not pos.active ):
            pos = openPosition( SHORT, price, quantity, leverage )
            return
        
        pos.update( SHORT, price, quantity, leverage )
    elif cmd == 'close':
        pos = getActivePosition()
        if( pos == None or not pos.active ):
            return
        pos.close( price )

def close():
    pos = getActivePosition()
    if( pos == None or not pos.active ):
        return
    realtimeCandle = getRealtimeCandle()
    if realtimeCandle != None :
        pos.close( realtimeCandle.close )
        
