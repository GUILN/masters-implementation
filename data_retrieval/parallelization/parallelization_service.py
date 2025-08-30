import multiprocessing as mp
from typing import Any, Callable, List

from common_setup import CommonSetup


logger = CommonSetup.get_logger()

class ParallelizationService:
    def __init__(
        self,
        num_workers: int = mp.cpu_count()
    ):
        self._num_workers = num_workers

    def run_in_parallel(
        self,
        func: Callable[[Any], Any],
        tasks: List[Any]
    ) -> List[Any]:
        """
        Run the given function in parallel over the list of tasks.
        Each task is passed as a single argument to func.
        """
        logger.info(f"Running {len(tasks)} tasks in parallel using {self._num_workers} workers.")
        with mp.Pool(processes=self._num_workers) as pool:
            results = pool.map(func, tasks)
        logger.info(f"Parallel execution completed. Processed {len(results)} results.")
        return results
