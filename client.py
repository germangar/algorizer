import zmq
import zmq.asyncio
import asyncio
import sys
import tasks
import json
import pandas as pd
import numpy as np

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())






############################ CHART WINDOW ################################
from constants import c
from lightweight_charts import Chart
# from lightweight_charts_esistjosh import Chart
from typing import Optional, Any

from dataclasses import dataclass
@dataclass
class plot_c:
    name: str
    panel:str
    type: int
    color:str
    style:str
    width:int
    margin_top:float
    margin_bottom:float
    instance:object

@dataclass
class marker_c:
    id:str
    timestamp:int
    position:str
    shape:str
    color:str
    panel:str
    text:str
    instance:object
    
class window_c:
    def __init__(self, config):
        self.config = config
        self.descriptor: Optional[dict[str, Any]] = None
        self.df: Optional[pd.DataFrame] = None
        self.chart = None
        self.plots:list = []
        self.markers:list = []

    def openWindow(self, descriptor, df):
        try:
            tmpdf = pd.DataFrame( { 'time':pd.to_datetime( df['timestamp'], unit='ms' ), 'open':df['open'], 'high':df['high'], 'low':df['low'], 'close':df['close'], 'volume':df['volume']} )
        except Exception as e:
            print(f"Error converting timestamp to datetime: {e}")

        print( "Initializing window" )

        window_width = 800
        window_height = 600
        if 1:
            screen_width, screen_height = self.get_screen_resolution()
            window_width = int(screen_width * 0.65)
            window_height = int(screen_height * 0.65)
        chart = Chart( window_width, window_height )

        legend = f"{self.config['symbol']} - {descriptor['timeframe']} - candles:{len(df)}"
        chart.legend( visible=True, ohlc=False, percent=False, font_size=18, text=legend )

        try:
            chart.set( tmpdf )
        except Exception as e:
            print(f"Error setting chart dataframe: {e}")
            
        self.chart = chart
        self.descriptor = descriptor
        self.df = df

        self.createPlots()
        self.createMarkers()

        task = chart.show_async()
        tasks.registerTask( 'window', task )

    def createPlots(self):
        plotsList = self.descriptor['plots']
        for name in plotsList.keys():
            info = plotsList[name]
            plot = plot_c( 
                name,
                info.get('panel'),
                int(info.get('type')),
                info.get('color'),
                info.get('style'),
                int(info.get('width')),
                float(info.get('margin_top')),
                float(info.get('margin_bottom')),
                None
            )
            
            if plot.panel:
                continue

            chart = self.chart
            
            if plot.type == c.PLOT_LINE :
                plot.instance = chart.create_line( plot.name, plot.color, plot.style, plot.width, price_line=False, price_label=False )
                plot.instance.set( pd.DataFrame( {'time': pd.to_datetime( self.df['timestamp'], unit='ms' ), plot.name: self.df[plot.name]} ) )
            elif plot.type == c.PLOT_HIST :
                plot.instance = chart.create_histogram( plot.name, plot.color, price_line = False, price_label = False, scale_margin_top = plot.margin_top, scale_margin_bottom = plot.margin_bottom )
                plot.instance.set( pd.DataFrame( {'time': pd.to_datetime( self.df['timestamp'], unit='ms' ), plot.name: self.df[plot.name]} ) )

            self.plots.append( plot )
    
    def createMarkers(self):
        markersList = self.descriptor['markers']

        for m in markersList:
            marker = marker_c(
                id = m.get('id'),
                timestamp = int(m.get('timestamp')),
                position = m.get('position'),
                shape = m.get('shape'),
                color = m.get('color'),
                panel = m.get('panel'),
                text = m.get('text'),
                instance = None
            )

            if marker.panel:
                continue

            chart = self.chart
            marker.instance = chart.marker( time = pd.to_datetime( marker.timestamp, unit='ms' ),
                        position = marker.position,
                        shape = marker.shape,
                        color = marker.color,
                        text = marker.text )
            
            self.markers.append( marker )

    def newRow(self, msg):
        row = msg.get('data')
        columns = msg.get('columns')
        if row is None:
            return

        # First part - OHLCV update (working fine)
        data_dict = {
            'time': pd.to_datetime(row[c.DF_TIMESTAMP], unit='ms'),
            'open': row[c.DF_OPEN],
            'high': row[c.DF_HIGH],
            'low': row[c.DF_LOW],
            'close': row[c.DF_CLOSE],
            'volume': row[c.DF_VOLUME]
        }
        
        print(data_dict)
        
        series = pd.Series(data_dict)
        self.chart.update(series)

        # Second part - full data update
        if columns is None:
            return

        # Create DataFrame from single row - wrap row in a list
        df = pd.DataFrame([row], columns=columns)
        if len(columns) != len(self.df.columns):
            # we need to create a column
            pass




            

    # There is no reason for this to be a method other than grouping all the window stuff together
    def get_screen_resolution(self):
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()  # Destroy the window after getting the resolution
        return screen_width, screen_height

window:window_c = None

###########################################################################
###########################################################################
############################### CLIENT ####################################
###########################################################################
###########################################################################

CLIENT_DISCONNECTED = 0
CLIENT_CONNECTED = 1
CLIENT_LOADING = 2  # receiving the data to open the window
CLIENT_READY = 3
CLIENT_LISTENING = 4  # the window has already opened the window and is ready to receive updates.

status = CLIENT_DISCONNECTED


# In client.py, modify the send_command function:

async def send_command(socket, command: str, params: str = ""):
    global status, window

    """Send a command to the server"""
    message = f"{command} {params}".strip()
    await socket.send_string(message)
    print(f"Sent command: {message}")
    
    # Get the reply
    reply = await socket.recv_string()

    try:
        # Try to parse as JSON
        data = json.loads(reply)
        if isinstance(data, dict) and 'type' in data:
            if data['type'] == 'config':
                print(f"Received config message for symbol: {data['symbol']}")
                # create the window container
                window = window_c(data)
                status = CLIENT_CONNECTED
                return data
                
            elif data['type'] == 'data_descriptor':
                print(f"Receiving DataFrame data...")
                status = CLIENT_LOADING

                descriptor = data
                print( descriptor )
                
                # Send acknowledgment that we're ready for the data
                await socket.send_string("ready")
                
                try:
                    # Get the raw data
                    raw_data = await socket.recv()
                    
                    # Convert raw bytes back to numpy array
                    array_data = np.frombuffer(raw_data, dtype=np.float64)
                    
                    # Reshape with explicit size calculation
                    rows = data['rows']
                    cols = len(data['columns'])
                    expected_size = rows * cols
                    
                    if array_data.size != expected_size:
                        raise ValueError(f"Data size mismatch. Expected {expected_size}, got {array_data.size}")
                    
                    array_data = array_data.reshape(rows, cols)
                    
                    # Create DataFrame with explicit dtypes
                    df = pd.DataFrame(array_data, columns=data['columns'])
                    
                    # Handle timestamp column separately
                    if 'timestamp' in df.columns:
                        df['timestamp'] = df['timestamp'].astype(np.int64)
                    
                    print("DataFrame received and reconstructed")
                    print(f"DataFrame shape: {df.shape}")
                    ### fall through ###
                except Exception as e:
                    print(f"Error reconstructing DataFrame: {e}")
                    status = CLIENT_CONNECTED
                    return None
                
                # initialize the window with the dataframe and open it
                window.openWindow( descriptor, df )
                status = CLIENT_READY
                return data

    except json.JSONDecodeError:
        # Not JSON, treat as regular message
        if reply == "connected":
            status = CLIENT_CONNECTED
            print("Connected")
        else:
            print(f"Received reply: {reply}")
        return reply

async def listen_for_updates(context):
    """Listen for updates from server"""
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:5556")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    print("Data update listener started...")
    
    try:
        while True:
            try:
                print("Waiting for message...")  # Debug
                message = await socket.recv_string()
                print("Received message")  # Debug
                try:
                    data = json.loads(message)
                    if data['type'] == 'row':
                        print(f"Received row update for timeframe {data['timeframe']}")
                        window.newRow(data)
                except json.JSONDecodeError:
                    print(f"Error: Received invalid JSON update")
                
            except Exception as e:
                print(f"Error in listen_for_updates loop: {e}")
            
            # Remove or reduce this sleep - it might be causing us to miss messages
            await asyncio.sleep(0.01)  # Reduced from 0.1 to 0.01
    
    except asyncio.CancelledError:
        print("Update listener cancelled")
    finally:
        socket.close()

async def run_client():
    global status
    # ZeroMQ Context
    context = zmq.asyncio.Context()

    # Define the socket for commands using REQ/REP pattern
    cmd_socket = context.socket(zmq.REQ)
    cmd_socket.connect("tcp://127.0.0.1:5555")

    print("Client is running and connected to server...")

    try:
        # Start the update listener
        tasks.registerTask("zmq_updates", listen_for_updates(context))

        while True:
            if status == CLIENT_DISCONNECTED:
                await send_command(cmd_socket, "connect", "")
                await asyncio.sleep(0.25)
                continue

            if status == CLIENT_CONNECTED:
                await send_command(cmd_socket, "dataframe", "")
                await asyncio.sleep(0.25)
                continue

            if status == CLIENT_LOADING:
                await asyncio.sleep(0.5)

            if status == CLIENT_READY:
                status = CLIENT_LISTENING
                await send_command(cmd_socket, "listening", "")
                await asyncio.sleep(0.25)

            if status == CLIENT_LISTENING:
                await send_command(cmd_socket, "ack", "")
                await asyncio.sleep(10)

            # await send_command(cmd_socket, "print", "ack")
            # await asyncio.sleep(2)

    except asyncio.CancelledError:
        print("Client task cancelled")
    finally:
        cmd_socket.close()
        context.term()

# Register the client as a task
tasks.registerTask("zmq_client", run_client())

if __name__ == "__main__":
    try:
        # Use the tasks system to run the client
        asyncio.run(tasks.runTasks())
    except KeyboardInterrupt:
        print("\nClient stopped by user")