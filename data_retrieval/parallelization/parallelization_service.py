import os
import glob
import multiprocessing as mp
from typing import Any, Callable, List

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
    with mp.Pool(processes=self._num_workers) as executor:
        results = list(executor.submit(func, *args, **kwargs))
    return results
