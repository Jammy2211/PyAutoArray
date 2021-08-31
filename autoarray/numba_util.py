from functools import wraps
import numba
import time
from typing import Callable

from autoconf import conf

"""
Depending on if we're using a super computer, we want two different numba decorators:

If on laptop:

@numba.jit(nopython=True, cache=True, parallel=False)

If on super computer:

@numba.jit(nopython=True, cache=False, parallel=True)
"""

try:
    nopython = conf.instance["general"]["numba"]["nopython"]
    cache = conf.instance["general"]["numba"]["cache"]
    parallel = conf.instance["general"]["numba"]["parallel"]
except Exception:
    nopython = True
    cache = True
    parallel = False


def jit(nopython=nopython, cache=cache, parallel=parallel):
    def wrapper(func):
        return numba.jit(func, nopython=nopython, cache=cache, parallel=parallel)

    return wrapper


def profile_func(func: Callable):
    """
    Time every function called in a class and averages over repeated calls for profiling likelihood functions.

    The timings are stored in the variable `_profiling_dict` of the class(s) from which each function is called,
    which are collected at the end of the profiling process via recursion.

    Parameters
    ----------
    func : (obj, grid, *args, **kwargs) -> Object
        A function which is used in the likelihood function..

    Returns
    -------
        A function that times the function being called.
    """

    @wraps(func)
    def wrapper(obj: object, *args, **kwargs):
        """
        Time a function and average over repeated calls for profiling an `Analysis` class's likelihood function. The
        time is stored in a `_profiling_dict` attribute.

        Returns
        -------
            The result of the function being timed.
        """

        if obj.profiling_dict is None:
            return func(obj, *args, **kwargs)

        repeats = (
            obj.profiling_dict["repeats"] if "repeats" in obj.profiling_dict else 1
        )

        start = time.time()
        for i in range(repeats):
            result = func(obj, *args, **kwargs)

        time_calc = (time.time() - start) / repeats

        obj.profiling_dict[func.__name__] = time_calc

        return result

    return wrapper
