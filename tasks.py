import asyncio

pendingTasks = []


def registerTask( task ):
    pendingTasks.append( task )

async def dummy():
    return

async def watch_for_new_tasks(tasks):
    while 1:
        for task in pendingTasks :
            tasks.append( asyncio.create_task(task) )
            pendingTasks.remove( task )
        await asyncio.sleep(0.2)


async def runTasks():
    task1 = dummy()

    tasks = [task1] + pendingTasks
    for task in pendingTasks:
        # tasks.append( task )
        pendingTasks.remove( task )

    last_task = watch_for_new_tasks(tasks)
    tasks.append( last_task )

    await asyncio.gather(*tasks)
    

# async def runTasks(stream):
#     task1 = asyncio.create_task(fetchCandleUpdates(stream))
#     task2 = asyncio.create_task(update_clock(stream))

#     tasks = [task1, task2]

#     # Create a task that waits for window.chart to become available
#     task3 = asyncio.create_task(watch_chart_and_add_task(tasks))
#     tasks.append(task3)

#     await asyncio.gather(*tasks)