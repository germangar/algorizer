import zmq
import zmq.asyncio
import asyncio
import sys
import tasks
import json
import pandas as pd

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

CLIENT_DISCONNECTED = 0
CLIENT_CONNECTED = 1
CLIENT_LOADING = 2 # receiving the data to open the window
CLIENT_ONLINE = 3 # the window has already opened the window and is ready to receive updates.

status = CLIENT_DISCONNECTED

symbol = None
timeframesList = []
panels = 0
df:pd.DataFrame = []

async def send_command(socket, command: str, params: str = ""):
    global symbol, timeframesList, panels, status, df

    """Send a command to the server"""
    message = f"{command} {params}".strip()
    await socket.send_string(message)
    print(f"Sent command: {message}")
    
    # Get the reply
    reply = await socket.recv_string()

    try:
        # Try to parse as JSON in case it's a data message
        data = json.loads(reply)
        print( data )
        if isinstance(data, dict) and 'type' in data:
            if data['type'] == 'config':
                print(f"Received config message for symbol: {data['symbol']}")
                '''message = {
                    "type": "config",
                    "symbol": active.timeframe.stream.symbol,
                    "timeframes": list( active.timeframe.stream.timeframes.keys() ),
                    "panels": 2, # To do. By now this is a placeholder
                    "payload": ""
                    }'''
                symbol = data['symbol']
                timeframesList = data['timeframes']
                panels = data['panels']

                status = CLIENT_CONNECTED
                return data
                
            elif data['type'] == 'data':
                print(f"Received data message: {data['datatype']}")
                if data['datatype'] == 'dataframe':
                    columns = data.get('columns')  # Get columns if they exist
                    df = pd.DataFrame(data['payload'], columns=columns)

                    print( df )

                # Handle the data based on its type
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
                    # Here you would update your local dataframe with the new bars
                    # The data structure will be a list of lists, where each inner list
                    # represents a row with [timestamp, open, high, low, close, volume]
                    bars = data['data']
                    # Example of processing:
                    # df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
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


            # Example commands

            
            await send_command(cmd_socket, "print", "ack")
            await asyncio.sleep(2)

            # await send_command(cmd_socket, "dataframe", "request_data")
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