from functools import wraps
import logging
import time
from typing import Callable

from autoconf import conf

from autoarray import exc

logger = logging.getLogger(__name__)

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

try:

    import numba

except ModuleNotFoundError:

    logger.warning(
        f"\n******************************************************************************\n"
        f"Numba is not being used, either because it is disabled in `config.general.ini` "
        f"or because it is not installed.\n\n. "
        f"This will lead to slow performance.\n\n. "
        f"Install numba as described at the following webpage for improved performance. \n"
        f"********************************************************************************"
    )



def jit(nopython=nopython, cache=cache, parallel=parallel):
    def wrapper(func):

        try:
            use_numba = conf.instance["general"]["numba"]["use_numba"]

            if not use_numba:
                return func

        except KeyError:

            pass

        try:

            import numba
            return numba.jit(func, nopython=nopython, cache=cache, parallel=parallel)

        except ModuleNotFoundError:

            return func

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
    def wrapper(obj, *args, **kwargs):
        """
        Time a function and average over repeated calls for profiling an `Analysis` class's likelihood function. The
        time is stored in a `profiling_dict` attribute.

        It is possible for multiple functions with the `profile_func` decorator to be called. In this circumstance,
        we risk repeated profiling of the same functionality in these nested functions. Thus, before added
        the time to the profiling_dict, the keys of the dictionary are iterated over in reverse, subtracting off the
        times of nested functions (which will already have been added to the profiling dict).

        Returns
        -------
            The result of the function being timed.
        """
        if obj.profiling_dict is None:
            return func(obj, *args, **kwargs)

        repeats = conf.instance["general"]["profiling"]["repeats"]

        last_key_before_call = (
            list(obj.profiling_dict)[-1] if obj.profiling_dict else None
        )

        start = time.time()
        for i in range(repeats):
            result = func(obj, *args, **kwargs)

        time_func = (time.time() - start) / repeats

        last_key_after_call = (
            list(obj.profiling_dict)[-1] if obj.profiling_dict else None
        )

        profile_call_max = 5

        for i in range(profile_call_max):

            key_func = f"{func.__name__}_{i}"

            if key_func not in obj.profiling_dict:

                if last_key_before_call == last_key_after_call:
                    obj.profiling_dict[key_func] = time_func
                else:
                    for key, value in reversed(list(obj.profiling_dict.items())):

                        if last_key_before_call == key:

                            obj.profiling_dict[key_func] = time_func
                            break

                        time_func -= obj.profiling_dict[key]

                break

            if i == 5:
                raise exc.ProfilingException(
                    f"Attempt to make profiling dict failed, because a function has been"
                    f"called more than {profile_call_max} times, exceed the number of times"
                    f"a profiled function may be called"
                )

        return result

    return wrapper
