import os
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
    if os.environ.get("USE_JAX") == "1":
        1
    else:
        import numba

except ModuleNotFoundError:
    logger.warning(
        f"\n******************************************************************************\n"
        f"Numba is not being used, either because it is disabled in `config/general.yaml` "
        f"or because it is not installed.\n\n. "
        f"This will lead to slow performance.\n\n. "
        f"Install numba as described at the following webpage for improved performance. \n"
        f"https://pyautolens.readthedocs.io/en/latest/installation/numba.html \n"
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
