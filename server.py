import zmq
import zmq.asyncio
import asyncio
import tasks
import sys
import json
import active
import pandas as pd

# Fix for Windows proactor event loop
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def getDataframe():
    timeframe = active.timeframe.stream.timeframes[active.timeframe.stream.timeframeFetch]
    return timeframe.df

CLIENT_DISCONNECTED = 0
CLIENT_CONNECTED = 1
CLIENT_LOADING = 2  # receiving the data to open the window
CLIENT_LISTENING = 3  # the window has already opened the window and is ready to receive updates.

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

def create_data_descriptor(df, timeframeStr: str) -> str:
    """Create a descriptor message for the DataFrame that will be sent"""
    message = {
        "type": "data_descriptor",
        "datatype": "dataframe",
        "timeframe": timeframeStr,
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns}
    }
    return json.dumps(message)

def create_dataframe_message(data: any, timeframeStr: str) -> str:
    columns = list(data.columns)
    data = data.values.tolist()  # More efficient than to_dict
    message = {
        "type": "data",
        "datatype": "dataframe",
        "timeframe": timeframeStr,
        "columns": columns,
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
        "timeframes": list(active.timeframe.stream.timeframes.keys()),
        "panels": 2,  # To do. By now this is a placeholder
        "payload": ""
    }
    return json.dumps(message)

def create_command_response(message: str) -> str:
    """Create a simple response for command acknowledgment"""
    return 'ok'

def server_cmd_dataframe(msg):
    df = getDataframe()
    # Convert dataframe to dict for JSON serialization
    data = df.to_dict('records') if df is not None else []
    return create_data_message("dataframe", data)

async def proccess_message(msg: str, cmd_socket):
    msg = msg.lstrip()
    parts = msg.split(maxsplit=1)
    command = parts[0].lower() if parts else ""
    msg = parts[1] if len(parts) > 1 else ""

    response = None

    if len(command):
        if command == 'connect':
            print('client connected')
            client.status = CLIENT_CONNECTED
            response = create_config_message()

        elif command == 'dataframe':
            client.status = CLIENT_LOADING
            df = active.timeframe.stream.timeframes[active.timeframe.stream.timeframeFetch].df
            timeframeStr = active.timeframe.stream.timeframeFetch
            
            # Convert object columns to string type for consistent handling
            for col in df.select_dtypes(include=['object']):
                df[col] = df[col].astype(str)
            
            # Convert all data to float64 for consistent binary transmission
            df_float = df.astype('float64')
            
            # First send the descriptor
            descriptor = create_data_descriptor(df, timeframeStr)
            await cmd_socket.send_string(descriptor)

            # Wait for acknowledgment
            ack = await cmd_socket.recv_string()
            if ack == "ready":
                # Send the raw data
                raw_data = df_float.values.tobytes()
                await cmd_socket.send(raw_data)
            else:
                print(f"Unexpected acknowledgment: {ack}")
            
            return None  # We've already handled the complete exchange

        elif command == 'print':
            print(msg)
            response = create_command_response(msg)

    return response if response else create_command_response("unknown command")

async def publish_updates(pub_socket):
    """Task to publish bar updates to clients"""
    last_rows = []  # Keep track of last sent rows
    
    while True:
        try:
            if(client.status == CLIENT_LISTENING):
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

    # Socket to handle command messages (REQ/REP pattern)
    cmd_socket = context.socket(zmq.REP)
    cmd_socket.bind("tcp://127.0.0.1:5555")

    # Socket to publish bar updates (PUB/SUB pattern)
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind("tcp://127.0.0.1:5556")

    print("Server is running...")

    try:
        # Start the update publisher task
        tasks.registerTask("zmq_updates", publish_updates(pub_socket))

        # Main command handling loop
        while True:
            try:
                message = await cmd_socket.recv_string()
                print(f"Received command: {message}")

                # Process the command
                response = await proccess_message(message, cmd_socket)  # Added await

                # Only send response if it wasn't already sent (for DataFrame case)
                if response is not None:
                    await cmd_socket.send_string(response)

            except Exception as e:
                print(f"Error processing message: {e}")
                await cmd_socket.send_string("error")

    except asyncio.CancelledError:
        print("Server task cancelled")
    finally:
        cmd_socket.close()
        pub_socket.close()
        context.term()

# Register the server as a task
tasks.registerTask("zmq_server", run_server())

if __name__ == "__main__":
    try:
        # Use the tasks system to run the server
        asyncio.run(tasks.runTasks())
    except KeyboardInterrupt:
        print("\nServer stopped by user")