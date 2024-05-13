from typing import List

from autoarray import Grid2DIrregular
from autoarray.structures.triangles.abstract_triangles import AbstractTriangles
from autoarray.structures.triangles.triangle import Triangle
from autoconf import cached_property


class SubsampleTriangles(AbstractTriangles):
    def __init__(self, parent_triangles: List[Triangle]):
        self.parent_triangles = parent_triangles

    @cached_property
    def triangles(self) -> List[Triangle]:
        return [
            triangle
            for parent_triangle in self.parent_triangles
            for triangle in parent_triangle.subdivide()
        ]

    @cached_property
    def grid_2d(self) -> Grid2DIrregular:
        pass

    def with_updated_grid(self, grid: Grid2DIrregular):
        pass
