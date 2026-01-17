import asyncio
from typing import Union, Callable, Coroutine, Any

pendingTasks = {}

def handle_task_result(task: asyncio.Task):
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        import traceback
        print(f"Exception raised by task = {task.get_name()}:")
        traceback.print_exc()

def registerTask(name: str, func_or_coro: Union[Callable, Coroutine], *args, **kwargs):
    """Register a task to be run
    Args:
        name: Name of the task
        func_or_coro: Either an async function or a coroutine object
        *args: Positional arguments (only used if func_or_coro is a function)
        **kwargs: Keyword arguments (only used if func_or_coro is a function)
    """
    if asyncio.iscoroutine(func_or_coro):
        # If we got a coroutine object directly, store it
        pendingTasks[name] = (None, (func_or_coro,), {})
    else:
        # If we got a function (sync or async) and its arguments
        pendingTasks[name] = (func_or_coro, args, kwargs)


async def watch_for_new_tasks(tasks):
    while True:
        to_remove = []
        for taskname, (func, args, kwargs) in pendingTasks.items():
            if func is None:
                # This is a pre-created coroutine
                t = asyncio.create_task(args[0])
            else:
                # This is a function to call with args
                t = asyncio.create_task(func(*args, **kwargs))
            t.set_name(taskname)
            t.add_done_callback(handle_task_result)
            tasks.append(t)
            to_remove.append(taskname)

        # Clean up after launching tasks
        for name in to_remove:
            del pendingTasks[name]

        await asyncio.sleep(0.2)


async def runTasks():
    tasks = []

    # Launch all currently registered pending tasks
    for name, (func, args, kwargs) in list(pendingTasks.items()):
        if func is None:
            # This is a pre-created coroutine
            task = asyncio.create_task(args[0])
        else:
            # This is a function to call with args
            task = asyncio.create_task(func(*args, **kwargs))
        task.set_name(name)
        task.add_done_callback(handle_task_result)
        tasks.append(task)
        del pendingTasks[name]  # Clean them out of pendingTasks

    # Start the watcher to launch future ones
    watcher_task = asyncio.create_task(watch_for_new_tasks(tasks))
    watcher_task.set_name("task_watcher")
    watcher_task.add_done_callback(handle_task_result)
    tasks.append(watcher_task)

    # Let tasks run forever instead of waiting for completion
    await asyncio.Event().wait()
    

def findTask(taskname):
    for task in asyncio.all_tasks():
        if task.get_name() == taskname:
            print(f"FOUND TASK: Task Name: {task.get_name()}, Status: {task._state}")
            return
        
    print(f"TASK: {taskname} not found")


def cancelTask(taskname):
    running_tasks = asyncio.all_tasks()
    for task in running_tasks:
        if taskname == task.get_name():
            if(task.cancelled() or task.cancelling()):
                print("Task", taskname, "already cancelled")
            else:
                task.cancel()
                print(taskname, "cancelled")
