import numba

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


class CachedAttribute(object):
    """
    Computes attribute value and caches it in instance.

    Example:
        class MyClass(object):
            def myMethod(self):
                # ...
            myMethod = CachedAttribute(myMethod)
    Use "del inst.myMethod" to clear cache.
    """

    def __init__(self, method, name=None):
        self.method = method
        self.name = name or method.__name__

    def __get__(self, inst, cls):
        if inst is None:
            return self
        result = self.method(inst)
        setattr(inst, self.name, result)
        return result


class CachedClassAttribute(object):
    """Computes attribute value and caches it in class.

    Example:
        class MyClass(object):
            def myMethod(cls):
                # ...
            myMethod = CachedClassAttribute(myMethod)
    Use "del MyClass.myMethod" to clear cache."""

    def __init__(self, method, name=None):
        self.method = method
        self.name = name or method.__name__

    def __get__(self, inst, cls):
        result = self.method(cls)
        setattr(cls, self.name, result)
        return result
