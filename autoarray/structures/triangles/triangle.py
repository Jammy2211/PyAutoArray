from functools import cached_property
from typing import Tuple


class Triangle:
    def __init__(self, *points: Tuple[float, float]):
        assert len(points) == 3
        self.points = points

    def __str__(self):
        return f"<Triangle({self.points})>"

    def __repr__(self):
        return str(self)

    def contains(self, point: Tuple[float, float]) -> bool:
        y1, x1 = self.points[0]
        y2, x2 = self.points[1]
        y3, x3 = self.points[2]
        y, x = point

        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
        c = 1 - a - b

        return 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1

    @cached_property
    def mid_1(self):
        return self.midpoint(0, 1)

    @cached_property
    def mid_2(self):
        return self.midpoint(1, 2)

    @cached_property
    def mid_3(self):
        return self.midpoint(2, 0)

    def subdivide(self):
        return (
            Triangle(self.points[0], self.mid_1, self.mid_3),
            Triangle(self.mid_1, self.points[1], self.mid_2),
            Triangle(self.mid_3, self.mid_2, self.points[2]),
            Triangle(self.mid_1, self.mid_2, self.mid_3),
        )

    @cached_property
    def subdivision_points(self):
        return self.mid_1, self.mid_2, self.mid_3

    def midpoint(self, i, j):
        y0, x0 = self.points[i]
        y1, x1 = self.points[j]
        return (y0 + y1) / 2, (x0 + x1) / 2
