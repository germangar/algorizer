import asyncio

pendingTasks = {}


def registerTask( name, task ):
    pendingTasks[name] = task


async def watch_for_new_tasks(tasks):
    while True:
        to_remove = []
        for taskname, task_coro in pendingTasks.items():
            t = asyncio.create_task(task_coro)
            t.set_name(taskname)
            tasks.append(t)
            to_remove.append(taskname)

        # Clean up after launching tasks
        for name in to_remove:
            del pendingTasks[name]

        await asyncio.sleep(0.2)


async def runTasks():
    tasks = []

    # Launch all currently registered pending tasks
    for name, coro in list(pendingTasks.items()):
        task = asyncio.create_task(coro)
        task.set_name(name)
        tasks.append(task)
        del pendingTasks[name]  # Clean them out of pendingTasks

    # Start the watcher to launch future ones
    watcher_task = asyncio.create_task(watch_for_new_tasks(tasks))
    watcher_task.set_name("task_watcher")
    tasks.append(watcher_task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    

def findTask( taskname ):
    for task in asyncio.all_tasks():
        if task.get_name() == taskname:
            print(f"FOUND TASK: Task Name: {task.get_name()}, Status: {task._state}")
            return
        
    print(f"TASK: {taskname} not found")

def cancelTask( taskname ) :
    running_tasks = asyncio.all_tasks()
    for task in running_tasks:
        if taskname == task.get_name():
            if( task.cancelled() or task.cancelling() ):
                print( "Task", taskname, "already cancelled")
            else:
                task.cancel()
                print( taskname, "cancelled")

# async def runTasks(stream):
#     task1 = asyncio.create_task(fetchCandleUpdates(stream))
#     task2 = asyncio.create_task(update_clock(stream))

#     tasks = [task1, task2]

#     # Create a task that waits for window.chart to become available
#     task3 = asyncio.create_task(watch_chart_and_add_task(tasks))
#     tasks.append(task3)

#     await asyncio.gather(*tasks)