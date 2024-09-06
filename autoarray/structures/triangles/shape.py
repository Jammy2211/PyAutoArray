from abc import ABC, abstractmethod

import numpy as np


class Shape(ABC):
    @abstractmethod
    def mask(self, triangles: np.ndarray) -> np.ndarray:
        pass


class Point:
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
        y1, x1 = triangles[:, 0, 1], triangles[:, 0, 0]
        y2, x2 = triangles[:, 1, 1], triangles[:, 1, 0]
        y3, x3 = triangles[:, 2, 1], triangles[:, 2, 0]

        a = x1 - self.x
        b = y1 - self.y
        c = x2 - self.x
        d = y2 - self.y
        e = x3 - self.x
        f = y3 - self.y

        aa = a * a + b * b
        bb = c * c + d * d
        cc = e * e + f * f

        radius_2 = self.radius * self.radius

        return (
            (aa <= radius_2)
            | (bb <= radius_2)
            | (cc <= radius_2)
            | super().mask(triangles)
        )
