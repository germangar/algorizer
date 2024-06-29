
import csv
import ccxt

from pprint import pprint

class candles_c:
    def __init__(self, exchangeID = 'binance', symbol = 'BTC/USDT:USDT', type = 'swap') -> None:
        self.symbol = symbol
        self.maxRetries = 3
        self.exchange = getattr(ccxt, exchangeID)() 
        self.exchange.defaultType = type
        print( 'Connecting to exchange:', self.exchange.id )

    def loadOHLCVfile( self, symbol, timeframe, exchange ):
        symbolName = symbol[:-len(':USDT')]
        symbolName = symbolName.replace( '/', '-' )
        filename = symbolName+'-'+timeframe
        filename = 'stuff/'+ exchange+'-'+ filename +'.csv'
        ohlcv_cache = []
        print('reading from file. ', end = '')

        try:
            with open(filename, 'r') as file:
                reader = csv.reader(file)

                # Iterate through the rows in the CSV file
                i = 0
                for row in reader:
                    if( row[0].isnumeric() ):
                        ohlcv_cache.insert( i, [ int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]) ] )
                        i += 1
        except Exception:
            with open(filename, 'x') as file:
                writer = csv.writer(file)
                file.close()

        print( len(ohlcv_cache), 'candles loaded')
        return ohlcv_cache
    
    def writeOHLCVfile( self, symbol, timeframe, exchange, ohlcv_data ):
        symbolName = symbol[:-len(':USDT')]
        symbolName = symbolName.replace( '/', '-' )
        filename = symbolName+'-'+timeframe
        filename = 'stuff/'+ exchange+'-'+ filename +'.csv'
        print('Writing')
        print( filename )

        with open( filename, 'w', newline='' ) as file:
            writer = csv.writer(file)
            #writer.writerow( [ 'time', 'open', 'high', 'low', 'close', 'volume' ] )
            file.close()

        with open( filename, 'a', newline='' ) as file:
            writer = csv.writer(file)

            # Write the data to the CSV file
            for row in ohlcv_data:
                writer.writerow(row)
            file.close()
        print('Done.')

    def fetchBlock(self, symbol, timeframe, earliest_timestamp, size):
        timeframe_duration_in_ms = self.exchange.parse_timeframe(timeframe) * 1000
        timedelta = size * timeframe_duration_in_ms
        fetch_since = earliest_timestamp - timedelta
        
        num_retries = 0
        ohlcv = None
        while num_retries < self.maxRetries:
            try:
                block = self.exchange.fetch_ohlcv(symbol, timeframe, fetch_since, size)
                ohlcv = []
                for o in block:
                    if( o[0] < earliest_timestamp ):
                        ohlcv.append(o)
                break
            except Exception:
                num_retries += 1
                continue

        return ohlcv
    
    def fetchRange(self, symbol, timeframe, oldestTimestamp = None, newestTimestamp = None):
        ohlcv_dictionary = {}
        ohlcv_list = []
        start_timestamp = newestTimestamp if( newestTimestamp != None ) else self.exchange.milliseconds()
        blockSize = 5000

        # check for the blocksize the exchange returns
        block = self.fetchBlock( symbol, timeframe, self.exchange.milliseconds(), blockSize )
        if( block == None or len(block) == 0 ):
            print( 'Nothing to retrieve' )
            return
        blockSize = min( blockSize, len(block) )

        # Do the real thing
        timestamp = start_timestamp
        while True:
            block = self.fetchBlock( symbol, timeframe, timestamp, blockSize )
            #print( 'block', len(block))
            if( block == None or len(block) == 0 ):
                break

            if( len(block) == 0 or (oldestTimestamp != None and block[0][0] < oldestTimestamp ) ): # we are done
                if len(block) < blockSize : print( "We are done len(block) < blockSize" )
                elif block[0][0] < oldestTimestamp : print( "We are done block[0][0] < oldestTimestamp" )
                ohlcv_dictionary = self.exchange.extend(ohlcv_dictionary, self.exchange.indexBy(block, 0))
                ohlcv_list = self.exchange.sort_by(ohlcv_dictionary.values(), 0)
                print('Fetched', len(ohlcv_list), self.exchange.id, symbol, timeframe, 'candles from', self.exchange.iso8601(ohlcv_list[0][0]), 'to', self.exchange.iso8601(ohlcv_list[-1][0]))
                break
            
            ohlcv_dictionary = self.exchange.extend(ohlcv_dictionary, self.exchange.indexBy(block, 0))
            ohlcv_list = self.exchange.sort_by(ohlcv_dictionary.values(), 0)

            timestamp = ohlcv_list[0][0] - ( self.exchange.parse_timeframe(timeframe) * 1000 )

            if len(ohlcv_list):
                print('Fetched', len(ohlcv_list), self.exchange.id, symbol, timeframe, 'candles from', self.exchange.iso8601(ohlcv_list[0][0]), 'to', self.exchange.iso8601(ohlcv_list[-1][0]))
            
        #remove unwanted bars
        if( oldestTimestamp != None ):
            for o in ohlcv_list:
                if( o[0] < oldestTimestamp ):
                    ohlcv_list.remove(o)

        return ohlcv_list
    
    def fetchAll(self, symbol, timeframe):
        return self.fetchRange( symbol, timeframe )

    def fetchCandles( self, symbol, timeframe, ohlcv_cache = None ):
        if( len(ohlcv_cache) > 0 ):
            oldestTimestamp = ohlcv_cache[0][0]
            newestTimestamp = ohlcv_cache[-1][0]

        if( len(ohlcv_cache) == 0 ):
            print( 'fetching all candles' )
            ohlcv_cache = thisMarket.fetchAll( symbol, timeframe )
            print( len(ohlcv_cache), 'candles fetched')
            return ohlcv_cache
        
        print( "fetching newer candles")
        newerBlock = thisMarket.fetchRange( symbol, timeframe, newestTimestamp, None )

        if( len(newerBlock) ):
            cleanRows = []
            minTime = ohlcv_cache[-1][0]
            for o in newerBlock:
                if( o[0] > minTime ):
                    cleanRows.append(o)
            newerBlock = cleanRows
            print( "Added", len(cleanRows), "newer candles" )
        
        print( "fetching older candles")
        olderBlock = thisMarket.fetchRange( symbol, timeframe, None, oldestTimestamp )

        if( len(olderBlock) ):
            cleanRows = []
            maxTime = ohlcv_cache[0][0]
            for o in olderBlock:
                if( o[0] < maxTime ):
                    cleanRows.append(o)
            olderBlock = cleanRows
            print( "Added", len(cleanRows), "older candles" )

        return thisMarket.exchange.sort_by( olderBlock + ohlcv_cache + newerBlock, 0 )
    
    def cacheCandles(self, symbol, timeframe): # feches brand new / updates existing cache

        ohlcv_cache = self.loadOHLCVfile( symbol, timeframe, exchange )
        ohlcvs = self.fetchCandles( symbol, timeframe, ohlcv_cache )
        self.writeOHLCVfile( symbol, timeframe, exchange, ohlcvs )
        return ohlcvs

    
    def fetchSampleTest(self, symbol, timeframe):
        bar5k = self.exchange.milliseconds() - (self.exchange.parse_timeframe(timeframe) * 1000 * 5000)
        bar2k = self.exchange.milliseconds() - (self.exchange.parse_timeframe(timeframe) * 1000 * 2345)
        return self.fetchRange( symbol, timeframe, bar5k, bar2k )
    
    def fetchAmount(self, symbol, timeframe, amount ):
        bars = self.exchange.milliseconds() - (self.exchange.parse_timeframe(timeframe) * amount * 1000)
        ohlcvs = self.fetchRange( symbol, timeframe, bars )
        if( len(ohlcvs) > amount ):
            ohlcvs = ohlcvs[-amount:]
        return ohlcvs




if __name__ == '__main__':

    symbol = 'ETH/USDT:USDT'
    timeframe = '1m'
    exchange = 'bitget'
    thisMarket = candles_c( exchange, symbol )

    ohlcv_cache = thisMarket.loadOHLCVfile( symbol, timeframe, exchange )

    ohlcvs = thisMarket.fetchCandles( symbol, timeframe, ohlcv_cache )

    thisMarket.writeOHLCVfile( symbol, timeframe, exchange, ohlcvs )





