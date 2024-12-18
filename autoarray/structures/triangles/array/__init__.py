from .array import ArrayTriangles
try:
    from .jax_array import ArrayTriangles as JAXArrayTriangles
except ImportError:
    pass
