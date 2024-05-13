from abc import ABC, abstractmethod
from functools import cached_property
from typing import List, Tuple

from autoarray import Grid2DIrregular
from autoarray.structures.triangles.triangle import Triangle


class AbstractTriangles(ABC):
    @cached_property
    @abstractmethod
    def triangles(self) -> List[Triangle]:
        pass

    def containing(self, point: Tuple[float, float]):
        return [triangle for triangle in self.triangles if triangle.contains(point)]

    @cached_property
    @abstractmethod
    def grid_2d(self) -> Grid2DIrregular:
        pass

    @abstractmethod
    def with_updated_grid(self, grid: Grid2DIrregular):
        pass
