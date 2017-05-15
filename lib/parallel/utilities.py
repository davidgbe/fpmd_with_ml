import multiprocessing as mp
from lib.gaussian_process.utilities import create_pool
from functools import partial

def batched_task(bundled_task_args, task):
    return list(map(task, bundled_task_args))

def parallel(task, task_args, max_concurrent=None, batch_size=80):
    cores = mp.cpu_count() - 1
    max_concurrent = cores if max_concurrent is None else max_concurrent
    bundled_task_args = []

    for i in range(0, len(task_args), batch_size):
        bundled_task_args.append(task_args[i:i+batch_size])

    async_results = []
    bound_task = partial(batched_task, task=task)

    for i in range(0, len(bundled_task_args), max_concurrent):
        pool = create_pool(max_concurrent)
        async_results.append(pool.map_async(bound_task, bundled_task_args[i:i+max_concurrent]).get())
        pool.close()
        pool.join()
    return [result for bundle in async_results for result in bundle]
