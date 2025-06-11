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
            print(f"Received request: {message}")
            
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