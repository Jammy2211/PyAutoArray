from abc import ABC, abstractmethod

import numpy as np

from autoarray.numpy_wrapper import register_pytree_node_class


class Shape(ABC):
    @abstractmethod
    def mask(self, triangles: np.ndarray) -> np.ndarray:
        pass


@register_pytree_node_class
class Point(Shape):
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def mask(self, triangles: np.ndarray) -> np.ndarray:
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


@register_pytree_node_class
class Circle(Point):
    def __init__(
        self,
        x: float,
        y: float,
        radius: float,
    ):
        super().__init__(x, y)
        self.radius = radius

    def mask(self, triangles: np.ndarray) -> np.ndarray:
        # Get the vertices of the triangles
        y1, x1 = triangles[:, 0, 1], triangles[:, 0, 0]
        y2, x2 = triangles[:, 1, 1], triangles[:, 1, 0]
        y3, x3 = triangles[:, 2, 1], triangles[:, 2, 0]

        centroid_x = (x1 + x2 + x3) / 3
        centroid_y = (y1 + y2 + y3) / 3

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
