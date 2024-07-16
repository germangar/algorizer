


class s_entry_c:
    def __init__( self, type:str ):
        self.type = type.lower()
        return
    
class s_position_c:
    def __init__( self, type:str ):
        self.type = type.lower()
        self.active = True
        self.ave_entry = 0.0
        self.quantity = 0.0

        self.entries:s_entry_c = []
        return
    
    def close(sefl):
        self.active = False
        return

positions:s_position_c = []


def findActivePosition() -> s_position_c:
    for pos in positions: # This should always be incremental, but let's do it like this by now
        if( pos.active == True ):
            return pos
    return None

def entry( type:str, quantity ):
    type = type.lower()
    position = findActivePosition()
    if( position != None and position.type != type ):
        position.quantity -= quantity
        if( position.quantity < 0 ):
            quantity = -position.quantity
        if position.quantity <= 0 :
            position.close()


    if( position == None ):
        position = s_position_c( type )
    return