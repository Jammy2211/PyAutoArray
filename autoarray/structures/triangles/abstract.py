from abc import abstractmethod, ABC

import numpy as np

from autoarray import Grid2D

HEIGHT_FACTOR = 3**0.5 / 2


class AbstractTriangles(ABC):
    def __len__(self):
        return len(self.triangles)

    @property
    def area(self) -> float:
        """
        The total area covered by the triangles.
        """
        triangles = self.triangles
        return (
            0.5
            * np.abs(
                (triangles[:, 0, 0] * (triangles[:, 1, 1] - triangles[:, 2, 1]))
                + (triangles[:, 1, 0] * (triangles[:, 2, 1] - triangles[:, 0, 1]))
                + (triangles[:, 2, 0] * (triangles[:, 0, 1] - triangles[:, 1, 1]))
            ).sum()
        )

    @property
    def indices(self):
        return self._indices

    @property
    def vertices(self):
        return self._vertices

    def __str__(self):
        return f"{self.__class__.__name__} with {len(self.indices)} triangles"

    def __repr__(self):
        return str(self)

    @property
    @abstractmethod
    def triangles(self):
        pass

    @classmethod
    @abstractmethod
    def for_limits_and_scale(
        cls,
        y_min: float,
        y_max: float,
        x_min: float,
        x_max: float,
        scale: float,
        **kwargs,
    ) -> "AbstractTriangles":
        pass

    @classmethod
    def for_grid(
        cls,
        grid: Grid2D,
        **kwargs,
    ) -> "AbstractTriangles":
        """
        Create a grid of equilateral triangles from a regular grid.

        Parameters
        ----------
        grid
            The regular grid to convert to a grid of triangles.

        Returns
        -------
        The grid of triangles.
        """

        scale = grid.pixel_scale

        y = grid[:, 0]
        x = grid[:, 1]

        y_min = y.min()
        y_max = y.max()
        x_min = x.min()
        x_max = x.max()

        return cls.for_limits_and_scale(
            y_min,
            y_max,
            x_min,
            x_max,
            scale,
            **kwargs,
        )

    @abstractmethod
    def with_vertices(self, vertices: np.ndarray) -> "AbstractTriangles":
        """
        Create a new set of triangles with the vertices replaced.

        Parameters
        ----------
        vertices
            The new vertices to use.

        Returns
        -------
        The new set of triangles with the new vertices.
        """

    @abstractmethod
    def for_indexes(self, indexes: np.ndarray) -> "AbstractTriangles":
        """
        Create a new ArrayTriangles containing indices and vertices corresponding to the given indexes
        but without duplicate vertices.

        Parameters
        ----------
        indexes
            The indexes of the triangles to include in the new ArrayTriangles.

        Returns
        -------
        The new ArrayTriangles instance.
        """

    @abstractmethod
    def neighborhood(self) -> "AbstractTriangles":
        """
        Create a new set of triangles that are the neighborhood of the current triangles.

        Includes the current triangles and the triangles that share an edge with the current triangles.
        """

    @abstractmethod
    def up_sample(self) -> "AbstractTriangles":
        """
        Up-sample the triangles by adding a new vertex at the midpoint of each edge.

        This means each triangle becomes four smaller triangles.
        """

    @property
    @abstractmethod
    def means(self):
        pass
