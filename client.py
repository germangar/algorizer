import zmq
import zmq.asyncio
import asyncio
import sys
import tasks  # importing your task management system

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def run_client():
    # ZeroMQ Context
    context = zmq.asyncio.Context()

    # Define the socket using the "Context"
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5555")

    print("Client is running and connected to server...")

    try:
        while True:
            # Send request
            await socket.send_string("print Hello")
            print("Sent: Hello")

            # Get the reply
            message = await socket.recv_string()
            print(f"Received reply: {message}")

            # Wait a bit before next request
            await asyncio.sleep(2)  # Wait 2 seconds between requests

    except asyncio.CancelledError:
        print("Client task cancelled")
    finally:
        socket.close()
        context.term()

# Register the client as a task
tasks.registerTask("zmq_client", run_client())

if __name__ == "__main__":
    try:
        # Use the tasks system to run the client
        asyncio.run(tasks.runTasks())
    except KeyboardInterrupt:
        print("\nClient stopped by user")