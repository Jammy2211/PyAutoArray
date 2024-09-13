from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from autoarray.numpy_wrapper import register_pytree_node_class


class Shape(ABC):
    """
    A shape in the source plane for which we identify corresponding image plane
    pixels using up-sampling.
    """

    @abstractmethod
    def mask(self, triangles: np.ndarray) -> np.ndarray:
        """
        Determine which triangles contain the shape.

        Parameters
        ----------
        triangles
            The vertices of the triangles.

        Returns
        -------
        A boolean array indicating which triangles contain the shape.
        """


@register_pytree_node_class
class Point(Shape):
    def __init__(self, x: float, y: float):
        """
        A point in the source plane for which we want to identify pixels in the
        image plane that trace to it.

        Parameters
        ----------
        x
        y
            The coordinates of the point.
        """
        self.x = x
        self.y = y

    def mask(self, triangles: np.ndarray) -> np.ndarray:
        """
        Determine which triangles contain the point.

        Parameters
        ----------
        triangles
            The vertices of the triangles

        Returns
        -------
        A boolean array indicating which triangles contain the point.
        """
        y1, x1 = triangles[:, 0, 1], triangles[:, 0, 0]
        y2, x2 = triangles[:, 1, 1], triangles[:, 1, 0]
        y3, x3 = triangles[:, 2, 1], triangles[:, 2, 0]

        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

        a = ((y2 - y3) * (self.x - x3) + (x3 - x2) * (self.y - y3)) / denominator
        b = ((y3 - y1) * (self.x - x3) + (x1 - x3) * (self.y - y3)) / denominator
        c = 1 - a - b

        return (0 <= a) & (a <= 1) & (0 <= b) & (b <= 1) & (0 <= c) & (c <= 1)

    def tree_flatten(self):
        """
        Flatten this model as a PyTree.
        """
        return (self.x, self.y), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten a PyTree into a model.
        """
        return cls(
            x=children[0],
            y=children[1],
        )


def centroid(triangles: np.ndarray):
    y1, x1 = triangles[:, 0, 1], triangles[:, 0, 0]
    y2, x2 = triangles[:, 1, 1], triangles[:, 1, 0]
    y3, x3 = triangles[:, 2, 1], triangles[:, 2, 0]

    return (x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3


@register_pytree_node_class
class Circle(Point):
    def __init__(
        self,
        x: float,
        y: float,
        radius: float,
    ):
        """
        A circle in the source plane for which we want to identify pixels in the
        image plane that trace to it.

        Parameters
        ----------
        x
        y
            The coordinates of the center of the circle.
        radius
            The radius of the circle.
        """
        super().__init__(x, y)
        self.radius = radius

    def mask(self, triangles: np.ndarray) -> np.ndarray:
        """
        Determine which triangles intersect the circle.

        This is approximated by checking if the centroid of the triangle is within
        the circle or if the triangle contains the centroid of the circle.

        Parameters
        ----------
        triangles
            The vertices of the triangles.

        Returns
        -------
        A boolean array indicating which triangles intersect the circle.
        """
        centroid_x, centroid_y = centroid(triangles)

        a = centroid_x - self.x
        b = centroid_y - self.y

        distance_squared = a * a + b * b

        radius_2 = self.radius * self.radius

        return (distance_squared <= radius_2) | super().mask(triangles)

    def tree_flatten(self):
        """
        Flatten this model as a PyTree.
        """
        return (self.x, self.y, self.radius), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten a PyTree into a model.
        """
        return cls(
            x=children[0],
            y=children[1],
            radius=children[2],
        )


class Triangle(Point):
    def __init__(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        c: Tuple[float, float],
    ):
        """
        A triangle in the source plane for which we want to identify pixels in the
        image plane that trace to it.

        Parameters
        ----------
        a, b, c
            The vertices of the triangle.
        """
        xs, ys = zip(a, b, c)
        super().__init__(
            x=np.mean(xs),
            y=np.mean(ys),
        )
        self.a = a
        self.b = b
        self.c = c

    def tree_flatten(self):
        """
        Flatten this model as a PyTree.
        """
        return (
            self.a,
            self.b,
            self.c,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten a PyTree into a model.
        """
        return cls(
            *children,
        )

    def mask(self, triangles: np.ndarray) -> np.ndarray:
        return self.triangle_contains_mask(triangles) | super().mask(triangles)

    def triangle_contains_mask(self, triangles: np.ndarray) -> np.ndarray:
        y1, x1 = self.a
        y2, x2 = self.b
        y3, x3 = self.c

        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

        centroid_x, centroid_y = centroid(triangles)

        a = (
            (y2 - y3) * (centroid_x - x3) + (x3 - x2) * (centroid_y - y3)
        ) / denominator
        b = (
            (y3 - y1) * (centroid_x - x3) + (x1 - x3) * (centroid_y - y3)
        ) / denominator
        c = 1 - a - b

        return (0 <= a) & (a <= 1) & (0 <= b) & (b <= 1) & (0 <= c) & (c <= 1)


class Polygon(Point):
    def __init__(
        self,
        vertices: List[Tuple[float, float]],
    ):
        """
        A polygon in the source plane for which we want to identify pixels in the
        image plane that trace to it.

        Parameters
        ----------
        vertices
            The vertices of the polygon.
        """
        self.vertices = vertices

        if len(vertices) < 3:
            raise ValueError("A polygon must have at least 3 vertices.")

        x = np.mean([vertex[0] for vertex in vertices])
        y = np.mean([vertex[1] for vertex in vertices])
        super().__init__(x, y)

        first = vertices[0]

        self.triangles = [
            Triangle(
                first,
                second,
                third,
            )
            for second, third in zip(vertices[1:], vertices[2:])
        ]

    def tree_flatten(self):
        """
        Flatten this model as a PyTree.
        """
        return (self.vertices,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten a PyTree into a model.
        """
        return cls(
            vertices=children[0],
        )

    def mask(self, triangles: np.ndarray) -> np.ndarray:
        """
        Determine which triangles intersect the Voronoi cell.

        Parameters
        ----------
        triangles
            The vertices of the triangles

        Returns
        -------
        A boolean array indicating which triangles intersect the Voronoi cell.
        """
        return np.any(
            [triangle.mask(triangles) for triangle in self.triangles],
            axis=0,
        ) | super().mask(triangles)


class Square(Point):
    def __init__(self, top, bottom, left, right):
        """
        A square in the source plane for which we want to identify pixels in the
        image plane that trace to it.

        Parameters
        ----------
        top
        bottom
        left
        right
            The coordinates of the top, bottom, left, and right edges of the square.
        """
        x = (left + right) / 2
        y = (top + bottom) / 2
        super().__init__(x, y)
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def mask(self, triangles: np.ndarray) -> np.ndarray:
        """
        Determine which triangles intersect the square.

        This is approximated by checking if the centroid of the triangle is within
        the square or if the triangle contains the centroid of the square.

        Parameters
        ----------
        triangles
            The vertices of the triangles.

        Returns
        -------
        A boolean array indicating which triangles intersect the square.
        """
        centroid_x, centroid_y = centroid(triangles)

        return (
            (self.left <= centroid_x)
            & (centroid_x <= self.right)
            & (self.bottom <= centroid_y)
            & (centroid_y <= self.top)
        ) | super().mask(triangles)
