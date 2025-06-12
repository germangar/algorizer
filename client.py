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

CLIENT_DISCONNECTED = 0
CLIENT_CONNECTED = 1
CLIENT_LOADING = 2  # receiving the data to open the window
CLIENT_ONLINE = 3  # the window has already opened the window and is ready to receive updates.

status = CLIENT_DISCONNECTED

symbol = None
timeframesList = []
panels = 0
df = []

# In client.py, modify the send_command function:

async def send_command(socket, command: str, params: str = ""):
    global symbol, timeframesList, panels, status, df

    """Send a command to the server"""
    message = f"{command} {params}".strip()
    await socket.send_string(message)
    print(f"Sent command: {message}")
    
    # Get the reply
    reply = await socket.recv_string()

    try:
        # Try to parse as JSON
        data = json.loads(reply)
        print(data)
        if isinstance(data, dict) and 'type' in data:
            if data['type'] == 'config':
                print(f"Received config message for symbol: {data['symbol']}")
                symbol = data['symbol']
                timeframesList = data['timeframes']
                panels = data['panels']
                status = CLIENT_CONNECTED
                return data
                
            elif data['type'] == 'data_descriptor':
                print(f"Receiving DataFrame data...")
                status = CLIENT_LOADING
                
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
                    return data
                except Exception as e:
                    print(f"Error reconstructing DataFrame: {e}")
                    status = CLIENT_CONNECTED
                    return None

    except json.JSONDecodeError:
        # Not JSON, treat as regular message
        if reply == "connected":
            status = CLIENT_CONNECTED
            print("Connected")
        else:
            print(f"Received reply: {reply}")
        return reply

async def listen_for_updates(context):
    """Listen for bar updates from server"""
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:5556")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    print("Bar data update listener started...")
    
    try:
        while True:
            message = await socket.recv_string()
            try:
                data = json.loads(message)
                if data['type'] == 'bars':
                    print(f"Received {data['len']} bars")
                    bars = data['data']
            except json.JSONDecodeError:
                print(f"Error: Received invalid JSON update")
            
            await asyncio.sleep(0.1)
    
    except asyncio.CancelledError:
        print("Update listener cancelled")
    finally:
        socket.close()

async def run_client():
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
                await asyncio.sleep(0.5)
                continue

            if status == CLIENT_CONNECTED:
                await send_command(cmd_socket, "dataframe", "")
                await asyncio.sleep(0.5)
                continue

            if status == CLIENT_LOADING:
                pass

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