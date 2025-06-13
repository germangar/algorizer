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

def getDataframe()->pd.DataFrame:
    timeframe = active.timeframe.stream.timeframes[active.timeframe.stream.timeframeFetch]
    return timeframe.df

def getPlotsList()->dict:
    timeframe = active.timeframe.stream.timeframes[active.timeframe.stream.timeframeFetch]
    return timeframe.plotsList()

def getMarkersList()->list:
    return active.timeframe.stream.getMarkersList()

# Global queue for updates
update_queue = asyncio.Queue(maxsize=1000)  # Set maxsize to match MAX_QUEUE_SIZE

CLIENT_DISCONNECTED = 0
CLIENT_CONNECTED = 1
CLIENT_LOADING = 2  # receiving the data to open the window
CLIENT_LISTENING = 3  # the window has already opened the window and is ready to receive updates.

LISTENING_TIMEOUT = 20.0    # 5 seconds timeout for listening state
LOADING_TIMEOUT = 60.0    # 1 minute timeout for other states
MAX_QUEUE_SIZE = 1000

class client_t:
    def __init__(self):
        self.status = CLIENT_DISCONNECTED
        self.last_successful_send = 0.0
        
    def update_last_send(self):
        """Update the last successful send timestamp"""
        self.last_successful_send = asyncio.get_event_loop().time()
        
    def is_timed_out(self):
        """Check if client has timed out based on its state"""
        if self.status == CLIENT_DISCONNECTED:
            return False
            
        current_time = asyncio.get_event_loop().time()
        elapsed = current_time - self.last_successful_send
        
        if self.status == CLIENT_LISTENING:
            # Strict timeout for listening state
            return elapsed > LISTENING_TIMEOUT
        else:
            # More lenient timeout for connecting/loading states
            return elapsed > LOADING_TIMEOUT


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
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "plots": getPlotsList(),
        "markers":getMarkersList()
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


def create_tick_update(tick_data: list) -> str:
    """Create a JSON message for tick/realtime updates"""
    message = {
        "type": "tick",
        "len": len(tick_data),
        "data": tick_data
    }
    queue_update( json.dumps(message) )

# def create_candle_update(candle_data: list) -> str:
#     """Create a JSON message for closed candle updates with full data"""
#     message = {
#         "type": "candle",
#         "len": len(candle_data),
#         "columns": 22,
#         "data": candle_data  # This will include all dataframe columns
#     }
#     return json.dumps(message)

def push_row_update(timeframe):
    df = timeframe.df
    # Convert the row to native Python types
    row_data = [item.item() if hasattr(item, 'item') else item for item in df.iloc[-1].tolist()]
    
    message = {
        "type": "row",
        "timeframe": timeframe.timeframeStr,
        "columns": list(df.columns),
        "data": row_data
    }
    print(f"Queueing update for timestamp: {row_data[df.columns.get_loc('timestamp')]}")  # Debug
    asyncio.get_event_loop().create_task(queue_update(json.dumps(message)))


async def queue_update(update):
    """Queue an update to be sent to clients"""
    if client.status == CLIENT_LISTENING:
        if update_queue.qsize() < MAX_QUEUE_SIZE:
            await update_queue.put(update)
            print(f"Added to queue. Queue size: {update_queue.qsize()}")  # Debug
        else:
            print("Update queue full - dropping update")


async def publish_updates(pub_socket):
    """Task to publish bar updates to clients"""
    while True:
        try:
            # Check for timeout based on state
            if client.is_timed_out():
                if client.status == CLIENT_LISTENING:
                    print("Client timed out while listening - marking as disconnected")
                else:
                    print(f"Client timed out during {['DISCONNECTED', 'CONNECTED', 'LOADING', 'LISTENING'][client.status]} state - marking as disconnected")
                client.status = CLIENT_DISCONNECTED
                # Clear the queue
                while not update_queue.empty():
                    try:
                        update_queue.get_nowait()
                        update_queue.task_done()
                    except asyncio.QueueEmpty:
                        break

            if client.status == CLIENT_LISTENING:
                try:
                    # Wait for an update with a timeout
                    update = await asyncio.wait_for(update_queue.get(), timeout=1.0)
                    print(f"Got update from queue. Queue size: {update_queue.qsize()}")  # Debug
                    try:
                        await asyncio.wait_for(pub_socket.send_string(update), timeout=1.0)
                        print("Successfully sent update")  # Debug
                        client.update_last_send()  # Mark successful send
                    except (asyncio.TimeoutError, zmq.error.Again):
                        print("Send timed out - requeueing update")
                        # Requeue the update if send failed
                        if update_queue.qsize() < MAX_QUEUE_SIZE:
                            await update_queue.put(update)
                    finally:
                        update_queue.task_done()
                except asyncio.TimeoutError:
                    # No updates in queue - this is normal
                    pass
            else:
                # Client not listening - clear queue periodically
                try:
                    update = await asyncio.wait_for(update_queue.get(), timeout=0.1)
                    update_queue.task_done()
                except asyncio.TimeoutError:
                    pass
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in publish_updates: {e}")
            await asyncio.sleep(1)


async def proccess_message(msg: str, cmd_socket):
    msg = msg.lstrip()
    parts = msg.split(maxsplit=1)
    command = parts[0].lower() if parts else ""
    msg = parts[1] if len(parts) > 1 else ""

    response = None

    if len(command):
        client.update_last_send()
        if command == 'connect':
            print('client connected')
            client.status = CLIENT_CONNECTED
            response = create_config_message()

        elif command == 'dataframe':
            client.status = CLIENT_LOADING
            df = active.timeframe.stream.timeframes[active.timeframe.stream.timeframeFetch].df
            timeframeStr = active.timeframe.stream.timeframeFetch
            
            # Create a copy to avoid modifying the original dataframe
            df_copy = df.copy()
            
            # Handle different data types
            for col in df_copy.columns:
                # Convert object/string columns that might be booleans
                if df_copy[col].dtype == 'object' or df_copy[col].dtype == 'bool':
                    # Convert boolean-like strings to 1.0/0.0
                    if df_copy[col].isin(['True', 'False', True, False]).all():
                        df_copy[col] = df_copy[col].map({'True': 1.0, 'False': 0.0, True: 1.0, False: 0.0})
                    else:
                        # For other string columns, convert to string
                        df_copy[col] = df_copy[col].astype(str)
            
            # Convert all numeric data to float64
            df_float = df_copy.select_dtypes(include=['int', 'float', 'number']).astype('float64')
            
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
        elif command == 'listening':
            client.status = CLIENT_LISTENING
            response = create_command_response(msg)
        elif command == 'ack': # keep alive
            pass

    client.update_last_send()
    return response if response else create_command_response("unknown command")


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