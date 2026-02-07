import zmq
import zmq.asyncio
import asyncio
import sys
import json
import numpy as np
import msgpack

from . import tasks
from .constants import constants as c
from . import tools
from . import active

# Fix for Windows proactor event loop
import platform
if sys.platform == 'win32' and platform.python_implementation() == 'CPython':
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

def svprint(*args, **kwargs):
    from . import console
    if active.timeframe.stream.running:
        console.delete_last_line()
    print(*args, **kwargs)
    if active.timeframe.stream.running:
        console.print_status_line()


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
        if delta["added"] or delta["removed"] or delta["modified"]:
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
            name: timeframe.dataset[timeframe.barindex, timeframe.generatedSeries[name].column_index] for name in common_keys
            if name not in added and name not in removed
        }

        # print( updated_plots )

        # Update the last known plot descriptors for the next comparison
        self.last_plots_dict = new_plot_descriptors.copy()

        for name in added.keys():
            column_array = timeframe.generatedSeries.get(name)
            if column_array is None: # something went terribly wrong
                raise ValueError( f"ERROR [{name}] is not registered in the dataframe" )
            # pack the data in the message
            descriptor = added[name]
            descriptor['array'] = pack_array( timeframe.dataset[:, column_array.column_index] )

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



def create_config_message(stream_instance):
    """Create a JSON message for data transmission"""
    message = {
        "type": "config",
        "symbol": stream_instance.symbol,
        "timeframes": list(stream_instance.timeframes.keys()),
        "mintick": stream_instance.mintick,
        "panels": stream_instance.registeredPanels
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
        "tick": { "type": "tick", "data": timeframe.realtimeCandle.tickData() }
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
    if client.status != CLIENT_LISTENING or client.streaming_timeframe != timeframe:
        return
    message = {
        "type": "tick",
        "data": timeframe.realtimeCandle.tickData()
    }
    asyncio.get_event_loop().create_task( queue_update( msgpack.packb(message, use_bin_type=True) ) )


def push_row_update(timeframe):
    if client.status != CLIENT_LISTENING or client.streaming_timeframe != timeframe:
        return
    # rows = timeframe.dataset[-1]

    rows = timeframe.dataset[-1, :6]
    
    message = {
        "type": "row",
        "timeframe": timeframe.timeframeStr,
        "barindex": timeframe.barindex,
        "columns": list( timeframe.generatedSeries.keys() ),
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
            if debug : svprint(f"Added to queue. Queue size: {update_queue.qsize()}")
        else:
            svprint("Update queue full - dropping update")


async def publish_updates(pub_socket):
    global server_cmd_port, server_pub_port
    """Task to publish bar updates to clients"""
    while True:
        try:
            # Check for timeout based on state
            if client.is_timed_out():
                if client.status == CLIENT_LISTENING:
                    svprint("Chart disconnected (timed out)")
                else:
                    svprint(f"Client timed out during {['DISCONNECTED', 'CONNECTED', 'LOADING', 'LISTENING'][client.status]} state - marking as disconnected")
                client.status = CLIENT_DISCONNECTED
                
                # Instead of cancelling tasks, just reset port state. run_server will close sockets.
                server_cmd_port = None
                server_pub_port = None
                client.last_successful_send = 0.0
                break # Exit this publish_updates task, it will be re-registered by run_server

            if client.status == CLIENT_LISTENING:
                try:
                    update = await asyncio.wait_for(update_queue.get(), timeout=1.0)
                    if debug : svprint(f"Got update from queue. Queue size: {update_queue.qsize()}")
                    
                    if pub_socket.closed: # Check if the socket is still open
                        if debug: svprint("publish_updates: pub_socket is closed, breaking.")
                        break # Exit if socket is closed externally
                    
                    try:
                        await asyncio.wait_for(pub_socket.send(update), timeout=1.0)
                        if debug : svprint("Successfully sent update")
                        client.update_last_send()

                    except (asyncio.TimeoutError, zmq.error.Again, zmq.error.ZMQError) as e:
                        if debug : print(f"Send timed out or socket error ({e}) - requeueing/breaking")
                        if not pub_socket.closed: # Only requeue if socket is still open
                            if update_queue.qsize() < MAX_QUEUE_SIZE:
                                await update_queue.put(update)
                        else:
                            break # Socket closed, break out
                    finally:
                        update_queue.task_done()
                except asyncio.TimeoutError:
                    pass # No updates, just continue
                except zmq.error.ZMQError as e: # Catch socket errors specifically
                    if debug: svprint(f"publish_updates: ZMQError on send/get ({e}), breaking.")
                    break # Socket likely closed, exit
            else:
                # Client not listening - clear queue periodically
                try:
                    update = await asyncio.wait_for(update_queue.get(), timeout=0.1)
                    update_queue.task_done()
                except asyncio.TimeoutError:
                    pass
            
        except asyncio.CancelledError:
            if debug: print("publish_updates: Task cancelled.")
            break
        except Exception as e:
            svprint(f"Error in publish_updates: {e}")
            await asyncio.sleep(1)

        await asyncio.sleep(0) # Yield control
    if debug: svprint("publish_updates: Exiting.")


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


async def start_window_server(stream_instance, timeframeName:str = None):
    """Initialize and start the window server"""
    global server_cmd_port, server_pub_port

    server_task_already_registered = False
    for task in asyncio.all_tasks():
        if task.get_name() == "zmq_server" and not task.done():
            server_task_already_registered = True
            break
    
    if not server_task_already_registered:
        # If the task isn't running, register it. It will wait for ports.
        tasks.registerTask("zmq_server", run_server, stream_instance)
        # Give a moment for the task to start up and enter its waiting loop
        # For simplicity, we assume the event loop is running and it will get scheduled.
        # A small sleep here could ensure it's ready, but it would have to be an async function.
        # For now, we rely on the task being scheduled quickly.

    # Find new ports for this client connection
    try:
        cmd_port, pub_port = find_available_ports()
        if debug : svprint(f"start_window_server: Found new ports: CMD={cmd_port}, PUB={pub_port}")
        server_cmd_port = cmd_port # Set global ports to trigger run_server to bind
        server_pub_port = pub_port # Set global ports to trigger run_server to bind
    except RuntimeError as e:
        svprint(f"Error finding available port: {e}")
        return False
    
    if debug : print(f"start_window_server: Launching client for server on port {server_cmd_port}")
    client_process = launch_client_window(cmd_port, timeframeName)
    if not client_process:
        svprint("Failed to launch client window")
        # If client launch fails, we should clear the global ports so run_server goes back to waiting
        server_cmd_port = None
        server_pub_port = None
        return False
        
    return True


async def run_server(stream_instance):
    global server_cmd_port, server_pub_port

    while True: # Outer loop for persistence
        # Wait for ports to be assigned (new client connection attempt)
        if server_cmd_port is None or server_pub_port is None:
            if debug: svprint("run_server: Waiting for ports...")
            await asyncio.sleep(0.05)
            continue # Loop back and check again

        if debug: svprint(f"run_server: Starting server with CMD={server_cmd_port}, PUB={server_pub_port}")

        context = None
        cmd_socket = None
        pub_socket = None

        try:
            # ZeroMQ Context
            context = zmq.asyncio.Context()

            # Socket to handle command messages (REQ/REP pattern)
            cmd_socket = context.socket(zmq.REP)
            cmd_socket.bind(f"tcp://127.0.0.1:{server_cmd_port}")

            # Socket to publish bar updates (PUB/SUB pattern)
            pub_socket = context.socket(zmq.PUB)
            pub_socket.bind(f"tcp://127.0.0.1:{server_pub_port}")

            if debug:print("Server is running...")
            
            # Cancel any old publish_updates task and register a new one with the new pub_socket
            tasks.cancelTask("zmq_updates")
            tasks.registerTask("zmq_updates", publish_updates, pub_socket)

            # Main command handling loop (inner loop for client interaction)
            while True:
                message = await cmd_socket.recv_string()
                if debug : svprint(f"Received command: {message}")

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
                        response = create_config_message(stream_instance)

                    #---------------------------
                    elif command == 'dataframe':
                        client.status = CLIENT_LOADING
                        timeframe = stream_instance.timeframes[stream_instance.timeframeFetch]
                        if tools.validateTimeframeName( msg ) and msg in stream_instance.timeframes.keys():
                            timeframe = stream_instance.timeframes[msg]

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
                    
                    #---------------------------
                    elif command == 'disconnect':
                        svprint('chart disconnected.')
                        client.status = CLIENT_DISCONNECTED
                        server_cmd_port = None # Reset global ports
                        server_pub_port = None # Reset global ports
                        client.last_successful_send = 0.0
                        tasks.cancelTask("zmq_updates") # Cancel the current publish task
                        response = 'ok'
                        break # Exit the inner while True loop
                
                if response is not None:
                    if type(response) == str: # we didn't explicitly pack strings
                        response = msgpack.packb(response, use_bin_type=True)

                    client.update_last_send()
                    await cmd_socket.send(response)

        except asyncio.CancelledError as e:
            if debug: svprint( f"run_server: Inner loop cancelled [{e}]" )
            pass # The outer loop will catch this or it's a natural exit from inner loop
        except Exception as e:
            svprint(f"Error in run_server inner loop: {e}")
        finally:
            if debug: svprint("run_server: Closing sockets and terminating context.")
            if cmd_socket:
                cmd_socket.close()
            if pub_socket:
                pub_socket.close()
            if context:
                context.term()
            # The outer loop will now continue, waiting for new ports