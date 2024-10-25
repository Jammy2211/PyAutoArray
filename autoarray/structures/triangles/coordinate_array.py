import numpy as np

from autoarray.structures.triangles.abstract import HEIGHT_FACTOR
from autoconf import cached_property


class CoordinateArrayTriangles:
    def __init__(
        self,
        coordinates: np.ndarray,
        side_length: float,
        flipped: bool = False,
        offset: float = 0.0,
    ):
        self.coordinates = coordinates
        self.side_length = side_length
        self.flipped = flipped

        self.scaling_factors = np.array(
            [0.5 * side_length, HEIGHT_FACTOR * side_length]
        )
        self.offset = offset

    @property
    def triangles(self) -> np.ndarray:
        centres = self.centres
        return np.stack(
            (
                centres
                + self.flip_array
                * np.array(
                    [0.0, 0.5 * self.side_length * HEIGHT_FACTOR],
                ),
                centres
                + self.flip_array
                * np.array(
                    [0.5 * self.side_length, -0.5 * self.side_length * HEIGHT_FACTOR]
                ),
                centres
                + self.flip_array
                * np.array(
                    [-0.5 * self.side_length, -0.5 * self.side_length * HEIGHT_FACTOR]
                ),
            ),
            axis=1,
        )

    @property
    def centres(self) -> np.ndarray:
        return self.scaling_factors * self.coordinates + np.array([0.0, self.offset])

    @cached_property
    def flip_mask(self):
        mask = (self.coordinates[:, 0] + self.coordinates[:, 1]) % 2 != 0
        if self.flipped:
            mask = ~mask
        return mask

    @cached_property
    def flip_array(self):
        array = np.ones(self.coordinates.shape[0])
        array[self.flip_mask] = -1

        return array[:, np.newaxis]

    def __iter__(self):
        return iter(self.triangles)

    def up_sample(self):
        new_coordinates = np.zeros((4 * self.coordinates.shape[0], 2))
        n_normal = 4 * np.sum(~self.flip_mask)

        new_coordinates[:n_normal] = np.vstack(
            (
                2 * self.coordinates[~self.flip_mask],
                2 * self.coordinates[~self.flip_mask] + np.array([1, 0]),
                2 * self.coordinates[~self.flip_mask] + np.array([-1, 0]),
                2 * self.coordinates[~self.flip_mask] + np.array([0, 1]),
            )
        )
        new_coordinates[n_normal:] = np.vstack(
            (
                2 * self.coordinates[self.flip_mask],
                2 * self.coordinates[self.flip_mask] + np.array([1, 1]),
                2 * self.coordinates[self.flip_mask] + np.array([-1, 1]),
                2 * self.coordinates[self.flip_mask] + np.array([0, 1]),
            )
        )

        return CoordinateArrayTriangles(
            coordinates=new_coordinates,
            side_length=self.side_length / 2,
            flipped=True,
            offset=self.offset + -0.25 * HEIGHT_FACTOR * self.side_length,
        )
