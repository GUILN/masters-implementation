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
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any
    ) -> List[Any]:
        logger.info(f"Running tasks in parallel using {self._num_workers} workers.")
        with mp.Pool(processes=self._num_workers) as executor:
            results = list(executor.submit(func, *args, **kwargs))
        return results
