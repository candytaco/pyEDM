"""
Execution strategies for pyEDM methods.

This module provides a clean abstraction for different execution modes,
hiding multiprocessing complexity from users while allowing advanced
customization when needed.
"""

from abc import ABC, abstractmethod
from enum import Enum
from multiprocessing import get_context
from typing import Callable, Iterable, Optional
import os


class ExecutionMode(Enum):
    """
    Enumeration of execution strategies.

    :param SEQUENTIAL: Sequential execution (no parallelism)
    :param MULTIPROCESS: Multiprocessing with default context
    :param SPAWN: Multiprocessing with spawn context (starts fresh Python interpreter)
    :param FORK: Multiprocessing with fork context (copies parent process)
    :param FORKSERVER: Multiprocessing with forkserver context (hybrid approach)
    """
    SEQUENTIAL = "sequential"
    MULTIPROCESS = "multiprocess"
    SPAWN = "spawn"
    FORK = "fork"
    FORKSERVER = "forkserver"


class ExecutionStrategy(ABC):
    """
    Abstract base class for execution strategies.
    """

    @abstractmethod
    def map(self, func: Callable, iterable: Iterable):
        """
        Execute function over iterable.

        :param func: Function to execute
        :param iterable: Iterable of arguments to pass to func
        :return: Results from executing func over iterable
        """
        pass


class SequentialExecution(ExecutionStrategy):
    """
    Sequential execution strategy (no parallelism).

    Useful for debugging or when multiprocessing overhead exceeds benefits.
    """

    def map(self, func, iterable):
        """
        Execute function sequentially over iterable.

        :param func: Function to execute
        :param iterable: Iterable of argument tuples to pass to func
        :return: Results from executing func over iterable
        """
        return [func(*args) for args in iterable]


class MultiprocessExecution(ExecutionStrategy):
    """
    Multiprocessing execution strategy.

    :param numProcess: Number of processes to use. If None, uses os.cpu_count()
    :param mpMethod: Multiprocessing context method ('spawn', 'fork', 'forkserver'). If None, uses platform default
    :param chunksize: Chunk size for pool.starmap
    """

    def __init__(self, numProcess: Optional[int] = None,
                 mpMethod: Optional[str] = None,
                 chunksize: int = 1):
        """
        Initialize multiprocessing execution strategy.

        :param numProcess: Number of processes. Defaults to CPU count
        :param mpMethod: Context method. Defaults to platform default
        :param chunksize: Chunk size for starmap
        """
        self.numProcess = numProcess or os.cpu_count()
        self.mpMethod = mpMethod
        self.chunksize = chunksize

    def map(self, func, iterable):
        """
        Execute function in parallel using multiprocessing.

        :param func: Function to execute
        :param iterable: Iterable of argument tuples to pass to func
        :return: Results from executing func over iterable
        """
        mpContext = get_context(self.mpMethod)
        with mpContext.Pool(processes=self.numProcess) as pool:
            return pool.starmap(func, iterable, chunksize=self.chunksize)


def create_executor(
    mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
    numProcess: Optional[int] = None,
    chunksize: int = 1
) -> ExecutionStrategy:
    """
    Factory function to create execution strategy from enum.

    :param mode: Execution mode to use
    :param numProcess: Number of processes (ignored for SEQUENTIAL mode)
    :param chunksize: Chunk size for parallel execution (ignored for SEQUENTIAL mode)
    :return: Configured execution strategy
    :raises ValueError: If mode is not a valid ExecutionMode
    """
    if mode == ExecutionMode.SEQUENTIAL:
        return SequentialExecution()
    elif mode == ExecutionMode.MULTIPROCESS:
        return MultiprocessExecution(numProcess=numProcess, chunksize=chunksize)
    elif mode in (ExecutionMode.SPAWN, ExecutionMode.FORK, ExecutionMode.FORKSERVER):
        return MultiprocessExecution(
            numProcess=numProcess,
            mpMethod=mode.value,
            chunksize=chunksize
        )
    else:
        raise ValueError(f"Unknown execution mode: {mode}")
