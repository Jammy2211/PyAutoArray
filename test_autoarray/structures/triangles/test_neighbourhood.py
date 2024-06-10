import pytest

from autoarray.structures.triangles.triangle import Triangle


@pytest.fixture
def neighbourhood(right_triangle):
    return right_triangle.neighbourhood


def test_neighbourhood(
    neighbourhood,
    right_triangle,
):
    assert len(neighbourhood) == 4
    assert right_triangle in neighbourhood


def test_reflection(neighbourhood):
    assert (
        Triangle(
            (1.0, 1.0),
            (1.0, 0.0),
            (0.0, 1.0),
        )
        in neighbourhood
    )
    assert (
        Triangle(
            (0.0, 0.0),
            (-1.0, 1.0),
            (0.0, 1.0),
        )
        in neighbourhood
    )
    assert (
        Triangle(
            (0.0, 0.0),
            (1.0, -1.0),
            (1.0, 0.0),
        )
        in neighbourhood
    )
