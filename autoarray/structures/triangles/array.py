from typing import Tuple

from autofit.jax_wrapper import numpy as np


class ArrayTriangles:
    def __init__(
        self,
        indices: np.ndarray,
        vertices: np.ndarray,
    ):
        self.indices = indices
        self.vertices = vertices

    @property
    def triangles(self):
        return self.vertices[self.indices]

    def containing(self, point: Tuple[float, float]):
        y, x = point

        triangles = self.triangles

        y1, x1 = triangles[:, 0, 1], triangles[:, 0, 0]
        y2, x2 = triangles[:, 1, 1], triangles[:, 1, 0]
        y3, x3 = triangles[:, 2, 1], triangles[:, 2, 0]

        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
        c = 1 - a - b

        inside = (0 <= a) & (a <= 1) & (0 <= b) & (b <= 1) & (0 <= c) & (c <= 1)

        containing_triangles = triangles[inside]
        unique_vertices, inverse_indices = np.unique(
            containing_triangles.reshape(-1, 2), axis=0, return_inverse=True
        )
        new_indices = inverse_indices.reshape(-1, 3)

        return ArrayTriangles(
            indices=new_indices,
            vertices=unique_vertices,
        )

    def up_sample(self):
        triangles = self.triangles

        m01 = (triangles[:, 0] + triangles[:, 1]) / 2
        m12 = (triangles[:, 1] + triangles[:, 2]) / 2
        m20 = (triangles[:, 2] + triangles[:, 0]) / 2

        new_triangles = np.concatenate(
            [
                np.stack([triangles[:, 0], m01, m20], axis=1),
                np.stack([triangles[:, 1], m12, m01], axis=1),
                np.stack([triangles[:, 2], m20, m12], axis=1),
                np.stack([m01, m12, m20], axis=1),
            ],
            axis=0,
        )

        # Make vertices unique
        unique_vertices, inverse_indices = np.unique(
            new_triangles.reshape(-1, 2), axis=0, return_inverse=True
        )
        new_indices = inverse_indices.reshape(-1, 3)

        return ArrayTriangles(
            indices=new_indices,
            vertices=unique_vertices,
        )
