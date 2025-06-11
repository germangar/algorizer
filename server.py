import zmq
import zmq.asyncio
import asyncio
import tasks
import sys
import json
# from algorizer import getDataframe
import active

# Fix for Windows proactor event loop
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def getDataframe():
    timeframe = active.timeframe.stream.timeframes[active.timeframe.stream.timeframeFetch]
    return timeframe.df

CLIENT_DISCONNECTED = 0
CLIENT_CONNECTED = 1
CLIENT_LOADING = 2 # receiving the data to open the window
CLIENT_LISTENING = 3 # the window has already opened the window and is ready to receive updates.

class client_t:
    def __init__(self):
        self.status = CLIENT_DISCONNECTED

client = client_t()

def create_data_message(datatype: str, data: any) -> str:
    """Create a JSON message for data transmission"""
    message = {
        "type": "data",
        "datatype": datatype,
        "payload": data
    }
    return json.dumps(message)

def create_bars_update(rows: list) -> str:
    """Create a JSON message for bar data updates"""
    message = {
        "type": "bars",
        "len": len(rows),
        "data": rows
    }
    return json.dumps(message)

def create_config_message() -> str:
    """Create a JSON message for data transmission"""
    message = {
        "type": "config",
        "symbol": active.timeframe.stream.symbol,
        "timeframes": list( active.timeframe.stream.timeframes.keys() ),
        "panels": 2, # To do. By now this is a placeholder
        "payload": ""
    }
    return json.dumps(message)

def create_command_response(message: str) -> str:
    """Create a simple response for command acknowledgment"""
    return 'ok'
    # return f"Command processed: {message}"

def server_cmd_dataframe(msg):
    df = getDataframe()
    # Convert dataframe to dict for JSON serialization
    data = df.to_dict('records') if df is not None else []
    return create_data_message("dataframe", data)

def proccess_message(msg: str):
    msg = msg.lstrip()           # Remove leading whitespace
    parts = msg.split(maxsplit=1)    # Split extracting the first word
    command = parts[0].lower() if parts else ""
    msg = parts[1] if len(parts) > 1 else ""

    response = None

    if len(command):
        if command == 'connect':
            print( 'client connected' )
            client.status = CLIENT_CONNECTED
            response = create_config_message()
        elif command == 'print':
            print(msg)
            response = create_command_response(msg)

        # if command == 'dataframe':
        #     response = server_cmd_dataframe(msg)

    return response if response else create_command_response("unknown command")

async def publish_updates(pub_socket):
    """Task to publish bar updates to clients"""
    last_rows = []  # Keep track of last sent rows
    
    while True:
        try:
            if( client.status == CLIENT_LISTENING ):
                df = getDataframe()
                if df is not None:
                    # Convert dataframe to list of lists (more efficient than dict)
                    new_rows = df.values.tolist()
                    if new_rows != last_rows:  # Only send if there are changes
                        update = create_bars_update(new_rows)
                        await pub_socket.send_string(update)
                        last_rows = new_rows
            
            await asyncio.sleep(0.1)  # Adjust the interval as needed
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in publish_updates: {e}")
            await asyncio.sleep(1)

async def run_server():
    # ZeroMQ Context
    context = zmq.asyncio.Context()

    # Socket for handling commands (REP)
    cmd_socket = context.socket(zmq.REP)
    cmd_socket.bind("tcp://127.0.0.1:5555")

    # Socket for publishing updates (PUB)
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind("tcp://127.0.0.1:5556")

    print("ZMQ Server is running (localhost TCP mode)...")

    # Start the update publisher task
    tasks.registerTask("zmq_publisher", publish_updates(pub_socket))

    try:
        while True:
            # if client.status != CLIENT_DISCONNECTED:
            # Wait for client request (non-blocking)
            message = await cmd_socket.recv_string()
            response = proccess_message(message)
            
            # Send reply (non-blocking)
            await cmd_socket.send_string(response)
            
            # Small delay to prevent CPU overuse
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        print("Server task cancelled")
    finally:
        cmd_socket.close()
        pub_socket.close()
        context.term()

# Register the server task
tasks.registerTask("zmq_server", run_server())