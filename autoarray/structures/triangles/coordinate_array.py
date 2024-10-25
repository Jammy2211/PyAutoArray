import numpy as np

from autoarray.structures.triangles.abstract import HEIGHT_FACTOR


class CoordinateArrayTriangles:
    def __init__(
        self,
        coordinates: np.ndarray,
        side_length: float,
    ):
        self.coordinates = coordinates
        self.side_length = side_length

        self.scaling_factors = np.array([side_length, HEIGHT_FACTOR * side_length])

    @property
    def triangles(self) -> np.ndarray:
        centres = self.centres
        return np.concatenate(
            (
                centres
                + np.array(
                    [0.0, 0.5 * self.side_length * HEIGHT_FACTOR],
                ),
                centres
                + np.array(
                    [0.5 * self.side_length, -0.5 * self.side_length * HEIGHT_FACTOR]
                ),
                centres
                + np.array(
                    [-0.5 * self.side_length, -0.5 * self.side_length * HEIGHT_FACTOR]
                ),
            )
        )

    @property
    def centres(self) -> np.ndarray:
        return self.scaling_factors * self.coordinates

    def __iter__(self):
        return iter(self.triangles)
