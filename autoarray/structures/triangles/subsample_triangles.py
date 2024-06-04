from typing import List

from autoarray import Grid2DIrregular
from autoarray.structures.triangles.abstract_triangles import AbstractTriangles
from autoarray.structures.triangles.triangle import Triangle
from autoconf import cached_property


class SubsampleTriangles(AbstractTriangles):
    def __init__(self, parent_triangles: List[Triangle]):
        """
        Represents a grid of equilateral triangles in the image plane. These triangles are subdivided into smaller
        triangles.

        Parameters
        ----------
        parent_triangles
            The triangles to subdivide.
        """
        self.parent_triangles = parent_triangles

    @cached_property
    def triangles(self) -> List[Triangle]:
        """
        A list of triangles in the image plane which have been subdivided.
        """
        return [
            triangle
            for parent_triangle in self.parent_triangles
            for triangle in parent_triangle.subdivide()
        ]

    @cached_property
    def grid_2d(self) -> Grid2DIrregular:
        """
        A 2D grid comprising the coordinates of the vertices of the triangles.
        """
        return Grid2DIrregular(
            [
                point
                for triangle in self.parent_triangles
                for point in (triangle.subdivision_points + triangle.points)
            ]
        )

    def with_updated_grid(self, grid: Grid2DIrregular):
        triangles = []
        points_per_group = 6
        for start in range(0, len(grid), points_per_group):
            finish = start + points_per_group
            points = grid[start:finish]
            mid_1, mid_2, mid_3, *points = points
            triangles.extend(
                (
                    Triangle(points[0], mid_1, mid_3),
                    Triangle(mid_1, points[1], mid_2),
                    Triangle(mid_3, mid_2, points[2]),
                    Triangle(mid_1, mid_2, mid_3),
                )
            )
        return triangles