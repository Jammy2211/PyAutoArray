import logging
import multiprocessing
import os
import sys
import threading
import types

from autonerves import conf


logger = logging.getLogger(__name__)

try:
    nopython = conf.instance["general"]["numba"]["nopython"]
    cache = conf.instance["general"]["numba"]["cache"]
    parallel = conf.instance["general"]["numba"]["parallel"]
except Exception:
    nopython = True
    cache = True
    parallel = False


# Decorated functions are queued here and only handed to numba on the first
# call of any of them, keeping ``import numba`` off the library import path.
# Materialization converts every queued function at once and rebinds the
# defining module's global, because numba's nopython mode must resolve
# cross-calls between decorated functions to real dispatchers at compile time.
_pending = []
_materialize_lock = threading.Lock()

# The pid of the process that first asked to launch a numba ``parallel=True`` kernel; see
# ``parallel_kernel_allowed``.
_parallel_pid = None


def parallel_kernel_allowed() -> bool:
    """
    Whether this process may launch a numba ``parallel=True`` (``prange``) kernel, or must take
    a serial route instead: the Sibson kernel's ``parallel=False`` compilation of the same body
    (identical results), and the interferometer direct_conv curvature's serial kernels.

    False in any Python multiprocessing child (``multiprocessing.parent_process()`` is set, for
    fork, spawn and forkserver alike -- how PyAutoFit / Nautilus search pools run), and in a
    process whose pid differs from the one that first called this (an ``os.fork`` child that
    ``parent_process`` does not see). True otherwise.

    There is never nested numba parallelism inside a search's worker processes, for two
    reasons:

    * Without TBB, numba's threading layer is GNU OpenMP (``omp``), whose pool registers an
      after-fork handler that terminates a forked child launching a parallel region after the
      parent used the pool ("Terminating: fork() called from a process already using GNU
      OpenMP, this is unsafe."). PyAutoFit evaluates the likelihood once in the parent before
      forking its workers, so every worker would die.
    * Each child would start ``NUMBA_NUM_THREADS`` threads (the node's core count by default),
      so N search processes times N threads oversubscribe the node. Parallelism inside a search
      comes from its processes.

    The serial variant never initialises a numba threading layer, so the single-process path
    keeps numba's default layer and parallel performance unchanged. Cheap on repeat calls (it
    runs every likelihood evaluation).
    """
    global _parallel_pid

    if multiprocessing.parent_process() is not None:
        return False

    pid = os.getpid()

    if _parallel_pid is None:
        _parallel_pid = pid

    return pid == _parallel_pid


def function_copy(func, name):
    """
    A copy of the plain Python function ``func`` under the name ``name`` (same code, globals and
    closure), so one body can be compiled by numba twice -- e.g. ``parallel=True`` and
    ``parallel=False`` -- with both cached on disk: numba's cache files are named by qualified
    name and its index key ignores compile flags, so two dispatchers of one function object
    would overwrite each other's cache entries.
    """
    copy = types.FunctionType(
        func.__code__, func.__globals__, name, func.__defaults__, func.__closure__
    )
    copy.__kwdefaults__ = func.__kwdefaults__
    copy.__module__ = func.__module__
    copy.__doc__ = func.__doc__
    prefix, _, _ = func.__qualname__.rpartition(".")
    copy.__qualname__ = f"{prefix}.{name}" if prefix else name
    return copy


def _materialize_all():
    with _materialize_lock:
        if not _pending:
            return

        try:
            import numba
        except ModuleNotFoundError:
            numba = None

        if (
            numba is not None
            and any(options["parallel"] for _, options, _, _ in _pending)
            and not parallel_kernel_allowed()
        ):
            logger.warning(
                "general.yaml -> numba -> parallel is True in a multiprocessing worker: "
                "numba_util.jit kernels compiled with parallel=True are unsafe under "
                "multiprocessing searches on numba's GNU OpenMP threading layer. Set it "
                "False (it is for single-process benchmarking only)."
            )

        while _pending:
            func, options, placeholder, state = _pending.pop()

            if numba is None:
                target = func
            else:
                target = numba.jit(func, **options)

            state["target"] = target

            module = sys.modules.get(func.__module__)
            if module is not None and getattr(module, func.__name__, None) is placeholder:
                setattr(module, func.__name__, target)


def jit(nopython=nopython, cache=cache, parallel=parallel, fastmath=False):
    options = dict(
        nopython=nopython,
        cache=cache,
        parallel=parallel,
        fastmath=fastmath,
    )

    def wrapper(func):
        import functools

        state = {"target": None}

        @functools.wraps(func)
        def lazy(*args, **kwargs):
            if state["target"] is None:
                _materialize_all()
            return state["target"](*args, **kwargs)

        _pending.append((func, options, lazy, state))

        return lazy

    return wrapper
