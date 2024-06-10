import pytest


@pytest.fixture
def neighbourhood(right_triangle):
    return right_triangle.neighbourhood


def test_neighbourhood(
    neighbourhood,
    right_triangle,
):
    assert len(neighbourhood) == 4
    assert right_triangle in neighbourhood
