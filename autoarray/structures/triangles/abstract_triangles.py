from abc import ABC, abstractmethod
from functools import cached_property
from typing import List, Tuple

from autoarray import Grid2DIrregular
from autoarray.structures.triangles.triangle import Triangle


class AbstractTriangles(ABC):
    @cached_property
    @abstractmethod
    def triangles(self) -> List[Triangle]:
        """A list of triangles in the image plane."""

    @cached_property
    @abstractmethod
    def grid_2d(self) -> Grid2DIrregular:
        """
        A 2D grid comprising the coordinates of the vertices of the triangles
        """
