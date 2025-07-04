import zmq
import zmq.asyncio
import asyncio
import sys
import json
import numpy as np

from . import tasks
from .constants import c
from . import tools
from . import active

# Fix for Windows proactor event loop
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# BIG HUGE FIXME: The plots should be grabbing the timeframe requested by the client
def getPlotsList()->dict:
    timeframe = active.timeframe.stream.timeframes[active.timeframe.stream.timeframeFetch]
    return timeframe.plotsList()

debug = False

# Global queue for updates
update_queue = asyncio.Queue(maxsize=1000)  # Set maxsize to match MAX_QUEUE_SIZE
server_cmd_port = None
server_pub_port = None

CLIENT_DISCONNECTED = 0
CLIENT_CONNECTED = 1
CLIENT_LOADING = 2  # receiving the data to open the window
CLIENT_LISTENING = 3  # the window has already opened the window and is ready to receive updates.

LISTENING_TIMEOUT = 20.0    # 5 seconds timeout for listening state
LOADING_TIMEOUT = 60.0    # 1 minute timeout for other states
MAX_QUEUE_SIZE = 1000


class client_state_t:
    def __init__(self):
        self.status = CLIENT_DISCONNECTED
        self.last_successful_send = 0.0
        self.timeframeStr = ""

        self.last_markers_dict:dict = {}
        self.last_lines_dict:dict = {}


    def prepareMarkersUpdate( self, markers ):
        # Convert old and new markers to dictionaries
        old_dict = self.last_markers_dict
        new_dict = {marker.id: marker for marker in markers}

        # Find added and removed marker IDs using set operations
        old_ids = set(old_dict.keys())
        new_ids = set(new_dict.keys())

        added_ids = new_ids - old_ids
        removed_ids = old_ids - new_ids

        # Generate lists of added and removed markers and sort them by timestamp
        # Parenthesize the generator expressions
        added = sorted((new_dict[id] for id in added_ids), key=lambda m: m.timestamp)
        removed = sorted((old_dict[id] for id in removed_ids), key=lambda m: m.timestamp)

        # Build delta with descriptors
        delta = {
            "added": [marker.descriptor() for marker in added],
            "removed": [marker.descriptor() for marker in removed]
        }

        self.last_markers_dict = new_dict
        return delta

    def prepareLinesUpdate( self, lines ):
        # Convert old and new markers to dictionaries
        old_dict = self.last_lines_dict
        new_dict = {marker.id: marker for marker in lines}

            # Find added and removed marker IDs using set operations
        added_ids = new_dict.keys() - old_dict.keys()
        removed_ids = old_dict.keys() - new_dict.keys()
        added = [new_dict[id] for id in added_ids]
        removed = [old_dict[id] for id in removed_ids]

        # Detect MODIFIED objects (efficiently)
        common_ids = new_dict.keys() & old_dict.keys()
        modified = [
            new_dict[id] for id in common_ids
            if new_dict[id].descriptor() != old_dict[id].descriptor()
        ]

        # Build delta
        delta = {
            "added": [m.descriptor() for m in added],
            "removed": [m.descriptor() for m in removed],
            "modified": [m.descriptor() for m in modified]
        }

        self.last_lines_dict = new_dict
        return delta
        ############################################################################


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

client = client_state_t()


def create_command_response(message: str) -> str:
    """Create a simple response for command acknowledgment"""
    return 'ok'


def create_config_message() -> str:
    """Create a JSON message for data transmission"""
    stream = active.timeframe.stream
    message = {
        "type": "config",
        "symbol": stream.symbol,
        "timeframes": list(stream.timeframes.keys()),
        "panels": stream.registeredPanels
    }
    return json.dumps(message)


def prepare_dataframe_for_sending(dataset):
    """
    Prepare a 2D NumPy float64 dataset for sending.
    Returns a float64 NumPy array.
    """
    # Ensure dataset is float64
    return np.asarray(dataset, dtype=np.float64)


def create_data_descriptor(dataset, timeframeStr: str, columns):
    """
    Create a descriptor message for the dataset that will be sent.
    """
    message = {
        "type": "data_descriptor",
        "datatype": "dataframe",
        "timeframe": timeframeStr,
        "timeframemsec": tools.timeframeMsec(timeframeStr),
        "rows": len(dataset),
        "columns": columns,
        "dtypes": {col: "float64" for col in columns},
        "plots": getPlotsList(),
        "markers": client.prepareMarkersUpdate( active.timeframe.stream.markers ), # fixme: Markers aren't timeframe based but this isn't a clean way to grab them
        "lines": client.prepareLinesUpdate( active.timeframe.stream.lines ) # same as above
    }
    return json.dumps(message)
    
  
async def send_dataframe(cmd_socket, timeframe):
    """
    Send the dataset (NumPy 2D array of float64) to client with proper descriptor and data handling.
    """

    dataset = timeframe.dataset
    columns = timeframe.columnsList()
    timeframe_str = timeframe.timeframeStr
    client.timeframeStr = timeframe_str

    try:
        # Prepare the dataset for sending
        arr = prepare_dataframe_for_sending(dataset)

        # Send descriptor
        descriptor = create_data_descriptor(dataset, timeframe_str, columns)
        await cmd_socket.send_string(descriptor)

        # Wait for acknowledgment
        ack = await cmd_socket.recv_string()
        if ack != "ready":
            raise ValueError(f"Unexpected acknowledgment: {ack}")

        # Send the raw data (as bytes)
        await cmd_socket.send(arr.tobytes())

        return True

    except Exception as e:
        print(f"Error sending DataFrame: {e}")
        return False

def push_tick_update(timeframe) -> str:
    """Create a JSON message for tick/realtime updates"""
    message = {
        "type": "tick",
        "data": timeframe.realtimeCandle.tickData()
    }
    asyncio.get_event_loop().create_task( queue_update(json.dumps(message)) )


def push_row_update(timeframe):
    if client.status != CLIENT_LISTENING or client.timeframeStr != active.timeframe.timeframeStr:
        return
    row = timeframe.dataset[-1].tolist()
    row[c.DF_TIMESTAMP] = int(row[c.DF_TIMESTAMP])
    
    message = {
        "type": "row",
        "timeframe": timeframe.timeframeStr,
        "barindex": active.timeframe.barindex,
        "columns": timeframe.columnsList(),
        "data": row,
        "markers": client.prepareMarkersUpdate( active.timeframe.stream.markers ),
        "lines": client.prepareLinesUpdate( active.timeframe.stream.lines ),
        "tick": { "type": "tick", "data": timeframe.realtimeCandle.tickData() }
    }
    asyncio.get_event_loop().create_task( queue_update(json.dumps(message)) )


async def queue_update(update):
    """Queue an update to be sent to clients"""
    if client.status == CLIENT_LISTENING:
        if update_queue.qsize() < MAX_QUEUE_SIZE:
            await update_queue.put(update)
            if debug : print(f"Added to queue. Queue size: {update_queue.qsize()}")  # Debug
        else:
            print("Update queue full - dropping update")


async def publish_updates(pub_socket):
    """Task to publish bar updates to clients"""
    while True:
        try:
            # Check for timeout based on state
            if client.is_timed_out():
                if client.status == CLIENT_LISTENING:
                    print("Chart disconnected")
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
                    if debug : print(f"Got update from queue. Queue size: {update_queue.qsize()}")  # Debug
                    try:
                        await asyncio.wait_for(pub_socket.send_string(update), timeout=1.0)
                        if debug : print("Successfully sent update")  # Debug
                        client.update_last_send()  # Mark successful send
                    except (asyncio.TimeoutError, zmq.error.Again):
                        if debug : print("Send timed out - requeueing update")
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
            if debug:print('client connected')
            client.status = CLIENT_CONNECTED
            response = create_config_message()

        elif command == 'dataframe':
            client.status = CLIENT_LOADING

            # Send DataFrame to client
            success = await send_dataframe(cmd_socket, active.timeframe)
            if not success:
                return create_command_response("error sending dataframe")
            
            return None  # We've already handled the complete exchange
        elif command == 'print':
            print(msg)
            response = create_command_response(msg)
        elif command == 'listening':
            client.status = CLIENT_LISTENING
            response = create_command_response(msg)
        elif command == 'ack': # keep alive
            response = ''

    client.update_last_send()
    return response if response else create_command_response("unknown command")

def launch_client_window(cmd_port):
    """Launch client.py - SIMPLE VERSION THAT JUST WORKS"""
    import sys
    from subprocess import Popen
    from pathlib import Path
    
    # Get path to client.py (now with proper parentheses)
    client_path = str(Path(__file__).parent / "client.py")
    
    # Launch with Python path set properly
    process = Popen([
        sys.executable,
        client_path,
        "--port", str(cmd_port)
    ])
    return process


def find_available_ports(base_cmd_port=5555, base_pub_port=5556, max_attempts=10):
    """Find available ports for both command and publish sockets"""
    for attempt in range(max_attempts):
        cmd_port = base_cmd_port + (attempt * 2)
        pub_port = base_pub_port + (attempt * 2)
        
        try:
            # Test command port (REP)
            context = zmq.Context()
            cmd_socket = context.socket(zmq.REP)
            cmd_socket.bind(f"tcp://127.0.0.1:{cmd_port}")
            
            # Test publish port (PUB)
            pub_socket = context.socket(zmq.PUB)
            pub_socket.bind(f"tcp://127.0.0.1:{pub_port}")
            
            # If we got here, both ports are available
            cmd_socket.close()
            pub_socket.close()
            context.term()
            
            return cmd_port, pub_port
            
        except zmq.error.ZMQError:
            # Port(s) already in use, clean up and try next pair
            try:
                cmd_socket.close()
                pub_socket.close()
                context.term()
            except:
                pass
            continue
            
    raise RuntimeError(f"Could not find available ports after {max_attempts} attempts")


def start_window_server(timeframeName = None):
    """Initialize and start the window server"""
    global server_cmd_port
    
    # If server is running, use its ports
    if server_cmd_port is not None:
        if debug : print(f"Launching client for existing server on port {server_cmd_port}")
        return launch_client_window(server_cmd_port) is not None

    # Server not running yet, start it with new ports
    try:
        cmd_port, pub_port = find_available_ports()
        if debug : print(f"Starting new server using ports: CMD={cmd_port}, PUB={pub_port}")
        server_cmd_port, server_pub_port = cmd_port, pub_port
    except RuntimeError as e:
        print(f"Error finding available port: {e}")
        return False

    # Launch client window
    client_process = launch_client_window(cmd_port)
    if not client_process:
        print("Failed to launch client window")
        return False
        
    return True


async def run_server():
    global server_cmd_port, server_pub_port
    
    # Find available ports if we don't have them yet
    if server_cmd_port is None or server_pub_port is None:
        try:
            server_cmd_port, server_pub_port = find_available_ports()
            if debug:print(f"Server using ports: CMD={server_cmd_port}, PUB={server_pub_port}")
        except RuntimeError as e:
            print(f"Error finding available ports: {e}")
            return
    else:
        if debug:print(f"Server already running on ports: CMD={server_cmd_port}, PUB={server_pub_port}")
    
    # ZeroMQ Context
    context = zmq.asyncio.Context()

    # Socket to handle command messages (REQ/REP pattern)
    cmd_socket = context.socket(zmq.REP)
    cmd_socket.bind(f"tcp://127.0.0.1:{server_cmd_port}")

    # Socket to publish bar updates (PUB/SUB pattern)
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(f"tcp://127.0.0.1:{server_pub_port}")

    if debug:print("Server is running...")

    try:
        # Start the update publisher task
        tasks.registerTask("zmq_updates", publish_updates, pub_socket)

        # Main command handling loop
        while True:
            try:
                message = await cmd_socket.recv_string()
                if debug : print(f"Received command: {message}")

                # Process the command
                response = await proccess_message(message, cmd_socket)

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
tasks.registerTask("zmq_server", run_server)

if __name__ == "__main__":
    try:
        # Use the tasks system to run the server
        asyncio.run(tasks.runTasks())
    except KeyboardInterrupt:
        print("\nServer stopped by user")