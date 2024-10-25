import numpy as np

from autoarray.structures.triangles.abstract import HEIGHT_FACTOR
from autoconf import cached_property


class CoordinateArrayTriangles:
    def __init__(
        self,
        coordinates: np.ndarray,
        side_length: float,
    ):
        self.coordinates = coordinates
        self.side_length = side_length

        self.scaling_factors = np.array(
            [0.5 * side_length, HEIGHT_FACTOR * side_length]
        )

    @property
    def triangles(self) -> np.ndarray:
        centres = self.centres
        return np.concatenate(
            (
                centres
                + self.flip_mask
                * np.array(
                    [0.0, 0.5 * self.side_length * HEIGHT_FACTOR],
                ),
                centres
                + self.flip_mask
                * np.array(
                    [0.5 * self.side_length, -0.5 * self.side_length * HEIGHT_FACTOR]
                ),
                centres
                + self.flip_mask
                * np.array(
                    [-0.5 * self.side_length, -0.5 * self.side_length * HEIGHT_FACTOR]
                ),
            )
        )

    @property
    def centres(self) -> np.ndarray:
        return self.scaling_factors * self.coordinates

    @cached_property
    def flip_mask(self):
        array = np.ones(self.coordinates.shape[0])
        mask = (self.coordinates[:, 0] + self.coordinates[:, 1]) % 2 != 0
        array[mask] = -1
        return array

    def __iter__(self):
        return iter(self.triangles)
