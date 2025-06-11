import zmq
import zmq.asyncio
import asyncio
import tasks
import sys

# Fix for Windows proactor event loop
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def run_server():
    # ZeroMQ Context
    context = zmq.asyncio.Context()

    # Define the socket using the "Context"
    socket = context.socket(zmq.REP)
    # Using localhost TCP for Windows compatibility
    socket.bind("tcp://127.0.0.1:5555")

    print("ZMQ Server is running (localhost TCP mode)...")

    try:
        while True:
            # Wait for client request (non-blocking)
            message = await socket.recv_string()
            proccess_message( message )
            
            # Send reply (non-blocking)
            await socket.send_string("Hello World")
            print("Sent: Hello World")
            
            # Small delay to prevent CPU overuse
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        print("ZMQ Server task cancelled")
    finally:
        socket.close()
        context.term()

# Register the server task - create the coroutine
tasks.registerTask("zmq_server", run_server())  # Add back the parentheses



def proccess_message( msg:str ): # the client will only send string messages.
    msg = msg.lstrip()           # Remove leading whitespace
    parts = msg.split(maxsplit=1)    # Split extracting the first word
    command = parts[0].lower() if parts else ""
    msg = parts[1] if len(parts) > 1 else ""

    if len(command) : # we will improve this later
        if command == 'print':
            print( msg )

    pass