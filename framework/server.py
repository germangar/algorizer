import zmq
import zmq.asyncio
import asyncio
import sys
import json
import numpy as np
import msgpack

from . import tasks
from .constants import c
from . import tools
from . import active

# Fix for Windows proactor event loop
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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
        self.streaming_timeframe = None

        self.last_markers_dict:dict = {}
        self.last_lines_dict:dict = {}
        self.last_plots_dict:dict = {}
    
    def prepareMarkersUpdate(self, markers):
        # Convert old and new markers to dictionaries
        old_dict = self.last_markers_dict  # old_dict maps marker.id to a descriptor snapshot (dict)
        new_dict = {marker.id: marker for marker in markers}
        old_ids = set(old_dict.keys())
        new_ids = set(new_dict.keys())

        added_ids = new_ids - old_ids
        removed_ids = old_ids - new_ids

        # Generate lists of added and removed markers and sort them by timestamp
        added = sorted((new_dict[_id] for _id in added_ids), key=lambda m: m.timestamp)
        removed = sorted((old_dict[_id] for _id in removed_ids), key=lambda m: m['timestamp'])

        # Detect modified markers by comparing descriptors
        common_ids = old_ids & new_ids
        modified = [
            new_dict[_id] for _id in common_ids
            if new_dict[_id].descriptor() != old_dict[_id]
        ]

        # Build delta with descriptors
        delta = {
            "added": [marker.descriptor() for marker in added],
            "removed": [marker for marker in removed],  # already descriptors
            "modified": [marker.descriptor() for marker in modified]
        }

        # Store descriptor snapshots for next time
        self.last_markers_dict = {k: v.descriptor().copy() for k, v in new_dict.items()}
        return delta
    
    
    def prepareLinesUpdate(self, lines):
        """
        Calculates the delta between the previous and current state of lines, detecting additions,
        removals, and modifications (even if only a property like 'color' changes).
        This version snapshots descriptors to detect mutations.
        """

        # old_dict maps marker.id to a descriptor snapshot (not to the object itself)
        old_dict = getattr(self, 'last_lines_dict', {})

        # new_dict maps marker.id to the object
        new_dict = {marker.id: marker for marker in lines}

        # Find added and removed marker IDs using set operations
        added_ids = new_dict.keys() - old_dict.keys()
        removed_ids = old_dict.keys() - new_dict.keys()
        added = [new_dict[_id] for _id in added_ids]
        removed = [old_dict[_id] for _id in removed_ids]  # old_dict values are descriptors

        # Detect MODIFIED objects by comparing descriptors (not objects!)
        common_ids = new_dict.keys() & old_dict.keys()
        modified = [
            new_dict[_id] for _id in common_ids
            if new_dict[_id].descriptor() != old_dict[_id]
        ]

        # Build delta
        delta = {
            "added": [m.descriptor() for m in added],
            "removed": [old_dict[_id] for _id in removed_ids],  # Already descriptors
            "modified": [m.descriptor() for m in modified]
        }

        # Store descriptor snapshots for next time
        self.last_lines_dict = {k: v.descriptor().copy() for k, v in new_dict.items()}
        return delta
    

    def preparePlotsUpdate(self, timeframe):
        """
        Calculates the delta between the previous and current state of plot descriptors,
        detecting additions, removals, and modifications.
        """
        # Retrieve the last known plot descriptors dictionary
        old_plot_descriptors = getattr(self, 'last_plots_dict', {})

        # Get the current plot descriptors
        new_plot_descriptors = timeframe.plotsList()

        # Determine added and removed plot names by comparing keys
        old_keys = set(old_plot_descriptors.keys())
        new_keys = set(new_plot_descriptors.keys())

        # Added plots are those whose names are in new but not in old
        added = {name: new_plot_descriptors[name] for name in new_keys - old_keys}

        # Removed plots are those whose names are in old but not in new
        removed = {name: old_plot_descriptors[name] for name in old_keys - new_keys}

        # Determine modified plots by comparing descriptor values for common keys
        common_keys = new_keys & old_keys
        modified = {
            name: new_plot_descriptors[name] for name in common_keys
            if old_plot_descriptors[name] != new_plot_descriptors[name]
        }

        # Create a list of plots that need their values updated
        updated_plots = {
            name: timeframe.dataset[timeframe.barindex, timeframe.registeredPlots[name].column_index] for name in common_keys
            if name not in added and name not in removed
        }

        # print( updated_plots )

        # Update the last known plot descriptors for the next comparison
        self.last_plots_dict = new_plot_descriptors.copy()

        for name in added.keys():
            column_array = timeframe.registeredSeries.get(name)
            if column_array is None: # something went terribly wrong
                raise ValueError( f"ERROR [{name}] is not a registeredSeries in the dataframe" )
            # pack the data in the message
            descriptor = added[name]
            descriptor['array'] = pack_array( timeframe.dataset[:, column_array.index] )

        timestamps = None
        if added:
            timestamps = pack_array( timeframe.dataset[:, c.DF_TIMESTAMP] )

        # Build delta dictionary
        delta = {
            "added": added,
            "removed": removed,
            "modified": modified,
            "updated": updated_plots,
            "timestamp": timestamps # the timestamps array is only included if there are new plots added and it's shared by all of them
        }

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



def create_config_message():
    """Create a JSON message for data transmission"""
    stream = active.timeframe.stream
    message = {
        "type": "config",
        "symbol": stream.symbol,
        "timeframes": list(stream.timeframes.keys()),
        "panels": stream.registeredPanels
    }
    return msgpack.packb(message, use_bin_type=True)


def pack_array(array):
    # Create a dictionary to hold the bytes data and shape information
    data_container = {
        "shape": array.shape,
        "dtype": str(array.dtype),
        "data": array.tobytes()
    }
    return data_container


def create_dataframe_message( timeframe ):
    assert(timeframe != None)
    dataset = timeframe.dataset[:, :6] # send only timestamp and OHLCV columns
    client.streaming_timeframe = timeframe

    client.last_lines_dict = {}  # reset the last_lines_dict for each timeframe
    client.last_markers_dict = {}  # reset the last_markers_dict for each timeframe
    client.last_plots_dict = {}
    message = {
        "type": "dataframe",
        "timeframe": timeframe.timeframeStr,
        "timeframemsec": tools.timeframeMsec(timeframe.timeframeStr),
        "arrays": pack_array(dataset),
    }

    return msgpack.packb(message, use_bin_type=True)

def create_graphs_baseline_message( timeframe ):
    assert(timeframe != None)

    client.last_lines_dict = {}  # reset the last_lines_dict for each timeframe
    client.last_markers_dict = {}  # reset the last_markers_dict for each timeframe
    client.last_plots_dict = {}
    message = {
        "type": "graphs",
        "timeframe": timeframe.timeframeStr,
        "plots": client.preparePlotsUpdate( timeframe ),
        "markers": client.prepareMarkersUpdate( timeframe.stream.markers ), # fixme: Markers aren't timeframe based but this isn't a clean way to grab them
        "lines": client.prepareLinesUpdate( timeframe.stream.lines ) # same as above
    }

    return msgpack.packb(message, use_bin_type=True)

def push_tick_update(timeframe):
    """Create a JSON message for tick/realtime updates"""
    message = {
        "type": "tick",
        "data": timeframe.realtimeCandle.tickData()
    }
    asyncio.get_event_loop().create_task( queue_update( msgpack.packb(message, use_bin_type=True) ) )


def push_row_update(timeframe):
    if client.status != CLIENT_LISTENING or client.streaming_timeframe != active.timeframe:
        return
    # rows = timeframe.dataset[-1]

    rows = timeframe.dataset[-1, :6]
    
    message = {
        "type": "row",
        "timeframe": timeframe.timeframeStr,
        "barindex": timeframe.barindex,
        "columns": timeframe.columnsList(),
        "row_array": pack_array(rows),
        "markers": client.prepareMarkersUpdate( timeframe.stream.markers ),
        "lines": client.prepareLinesUpdate( timeframe.stream.lines ),
        "plots": client.preparePlotsUpdate( timeframe ),
        "tick": { "type": "tick", "data": timeframe.realtimeCandle.tickData() }
    }
    asyncio.get_event_loop().create_task( queue_update( msgpack.packb(message, use_bin_type=True) ) )


async def queue_update(update):
    """Queue an update to be sent to clients"""
    if client.status == CLIENT_LISTENING:
        if update_queue.qsize() < MAX_QUEUE_SIZE:
            await update_queue.put(update)
            if debug : print(f"Added to queue. Queue size: {update_queue.qsize()}")
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
                    if debug : print(f"Got update from queue. Queue size: {update_queue.qsize()}")

                    try:
                        await asyncio.wait_for(pub_socket.send(update), timeout=1.0)
                        if debug : print("Successfully sent update")
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


def launch_client_window( cmd_port, timeframeName:str = None ):
    """Launch client.py - SIMPLE VERSION THAT JUST WORKS"""
    import sys
    from subprocess import Popen
    from pathlib import Path
    
    # Get path to client.py (now with proper parentheses)
    client_path = str(Path(__file__).parent / "client.py")
    
    # Launch with Python path set properly
    cmd = [
        sys.executable,
        client_path,
        "--port", str(cmd_port)
    ]
    if timeframeName is not None:
        cmd.extend( ["--timeframe", str(timeframeName)] )
    process = Popen(cmd)
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
        return launch_client_window(server_cmd_port, timeframeName) is not None

    # Server not running yet, start it with new ports
    try:
        cmd_port, pub_port = find_available_ports()
        if debug : print(f"Starting new server using ports: CMD={cmd_port}, PUB={pub_port}")
        server_cmd_port, server_pub_port = cmd_port, pub_port
    except RuntimeError as e:
        print(f"Error finding available port: {e}")
        return False

    # Launch client window
    client_process = launch_client_window(cmd_port, timeframeName)
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
            message = await cmd_socket.recv_string()
            if debug : print(f"Received command: {message}")

            # Process the command
            msg = message.lstrip()
            parts = msg.split(maxsplit=1)
            command = parts[0].lower() if parts else ""
            msg = parts[1] if len(parts) > 1 else ""
            response = None

            if len(command):
                #-----------------------
                if command == 'connect':
                    if debug:print('client connected')
                    client.status = CLIENT_CONNECTED
                    response = create_config_message()

                #---------------------------
                elif command == 'dataframe':
                    client.status = CLIENT_LOADING
                    timeframe = active.timeframe.stream.timeframes[active.timeframe.stream.timeframeFetch]
                    if tools.validateTimeframeName( msg ) and msg in active.timeframe.stream.timeframes.keys():
                        timeframe = active.timeframe.stream.timeframes[msg]

                    if timeframe != client.streaming_timeframe:
                        pass # FIXME: Execute a reset?

                    response = create_dataframe_message( timeframe )

                #---------------------------
                elif command == 'graphs':
                    response = create_graphs_baseline_message( client.streaming_timeframe )

                #---------------------------
                elif command == 'listening': # the chart is ready. Waiting for realtime updates
                    client.status = CLIENT_LISTENING
                    response = 'ok'

                #---------------------
                elif command == 'ack': # keep alive
                    response = 'ok'

            if response is not None:
                if type(response) == str: # we didn't explicitly pack strings
                    response = msgpack.packb(response, use_bin_type=True)

                client.update_last_send()
                await cmd_socket.send(response)

    except asyncio.CancelledError as e:
        print( f"Server task cancelled [{e}]" )
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