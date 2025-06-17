from jax import numpy as np
import jax
import numpy as np

jax.config.update("jax_log_compiles", True)
import pytest

from autoarray.structures.triangles.abstract import HEIGHT_FACTOR
from autoarray.structures.triangles.shape import Point

from autoarray.structures.triangles.coordinate_array import (
    CoordinateArrayTriangles,
)


def test__two(two_triangles):

    assert np.all(two_triangles.centres == np.array([[0, 0], [0.5, 0]]))
    assert two_triangles.triangles == pytest.approx(
        np.array(
            [
                [
                    [0.0, HEIGHT_FACTOR / 2],
                    [0.5, -HEIGHT_FACTOR / 2],
                    [-0.5, -HEIGHT_FACTOR / 2],
                ],
                [
                    [0.5, -HEIGHT_FACTOR / 2],
                    [0.0, HEIGHT_FACTOR / 2],
                    [1.0, HEIGHT_FACTOR / 2],
                ],
            ]
        ),
        1.0e-4,
    )


def test__trivial_triangles(one_triangle):
    assert one_triangle.flip_array == np.array([1])
    assert np.all(one_triangle.centres == np.array([[0, 0]]))
    assert one_triangle.triangles == pytest.approx(
        np.array(
            [
                [
                    [0.0, HEIGHT_FACTOR / 2],
                    [0.5, -HEIGHT_FACTOR / 2],
                    [-0.5, -HEIGHT_FACTOR / 2],
                ],
            ]
        ),
        1.0e-4,
    )


def test__above():
    triangles = CoordinateArrayTriangles(
        coordinates=np.array([[0, 1]]),
        side_length=1.0,
    )
    assert triangles.up_sample().triangles == pytest.approx(
        np.array(
            [
                [
                    [0.0, 0.43301270189221935],
                    [-0.25, 0.8660254037844386],
                    [0.25, 0.8660254037844386],
                ],
                [
                    [0.25, 0.8660254037844388],
                    [0.0, 1.299038105676658],
                    [0.5, 1.299038105676658],
                ],
                [
                    [-0.25, 0.8660254037844388],
                    [-0.5, 1.299038105676658],
                    [0.0, 1.299038105676658],
                ],
                [
                    [0.0, 1.299038105676658],
                    [0.25, 0.8660254037844388],
                    [-0.25, 0.8660254037844388],
                ],
            ]
        ),
        1.0e-4,
    )


@pytest.fixture
def upside_down():
    return CoordinateArrayTriangles(
        coordinates=np.array([[1, 0]]),
        side_length=1.0,
    )


def test_upside_down(upside_down):
    assert np.all(upside_down.centres == np.array([[0.5, 0]]))
    assert upside_down.triangles == pytest.approx(
        np.array(
            [
                [
                    [0.5, -HEIGHT_FACTOR / 2],
                    [0.0, HEIGHT_FACTOR / 2],
                    [1.0, HEIGHT_FACTOR / 2],
                ],
            ]
        ),
        1.0e-4,
    )


def test_up_sample(one_triangle):
    up_sampled = one_triangle.up_sample()
    assert up_sampled.side_length == 0.5
    assert up_sampled.triangles == pytest.approx(
        np.array(
            [
                [[0.0, -0.4330127018922193], [-0.25, 0.0], [0.25, 0.0]],
                [[0.25, 0.0], [0.5, -0.4330127018922193], [0.0, -0.4330127018922193]],
                [[-0.25, 0.0], [0.0, -0.4330127018922193], [-0.5, -0.4330127018922193]],
                [[0.0, 0.4330127018922193], [0.25, 0.0], [-0.25, 0.0]],
            ]
        ),
        1.0e-4,
    )


def test_up_sample_upside_down(upside_down):
    up_sampled = upside_down.up_sample()
    assert up_sampled.side_length == 0.5
    assert up_sampled.triangles == pytest.approx(
        np.array(
            [
                [[0.5, -0.4330127018922193], [0.25, 0.0], [0.75, 0.0]],
                [[0.75, 0.0], [0.5, 0.4330127018922193], [1.0, 0.4330127018922193]],
                [[0.25, 0.0], [0.0, 0.4330127018922193], [0.5, 0.4330127018922193]],
                [[0.5, 0.4330127018922193], [0.75, 0.0], [0.25, 0.0]],
            ]
        ),
        1.0e-4,
    )


def _test_up_sample_twice(one_triangle, plot):
    plot(one_triangle)
    one = one_triangle.up_sample()
    two = one.up_sample()
    three = two.up_sample()
    plot(three, color="blue")
    plot(two, color="green")
    plot(one, color="red")


def test_neighborhood(one_triangle):
    assert one_triangle.neighborhood().triangles == pytest.approx(
        np.array(
            [
                [
                    [-0.5, -0.4330127018922193],
                    [-1.0, 0.4330127018922193],
                    [0.0, 0.4330127018922193],
                ],
                [
                    [0.0, -1.299038105676658],
                    [-0.5, -0.4330127018922193],
                    [0.5, -0.4330127018922193],
                ],
                [
                    [0.0, 0.4330127018922193],
                    [0.5, -0.4330127018922193],
                    [-0.5, -0.4330127018922193],
                ],
                [
                    [0.5, -0.4330127018922193],
                    [0.0, 0.4330127018922193],
                    [1.0, 0.4330127018922193],
                ],
            ]
        ),
        1.0e-4,
    )


def test_upside_down_neighborhood(upside_down):
    assert upside_down.neighborhood().triangles == pytest.approx(
        np.array(
            [
                [
                    [0.0, 0.4330127018922193],
                    [0.5, -0.4330127018922193],
                    [-0.5, -0.4330127018922193],
                ],
                [
                    [0.5, -0.4330127018922193],
                    [0.0, 0.4330127018922193],
                    [1.0, 0.4330127018922193],
                ],
                [
                    [0.5, 1.299038105676658],
                    [1.0, 0.4330127018922193],
                    [0.0, 0.4330127018922193],
                ],
                [
                    [1.0, 0.4330127018922193],
                    [1.5, -0.4330127018922193],
                    [0.5, -0.4330127018922193],
                ],
            ]
        ),
        1.0e-4,
    )


def _test_complicated(plot, one_triangle):
    triangles = one_triangle.neighborhood().neighborhood()
    up_sampled = triangles.up_sample()


def test_vertices(one_triangle):
    assert one_triangle.vertices == pytest.approx(
        np.array(
            [
                [-0.5, -0.4330127018922193],
                [0.0, 0.4330127018922193],
                [0.5, -0.4330127018922193],
            ]
        ),
        1.0e-4,
    )


def test_up_sampled_vertices(one_triangle):
    assert one_triangle.up_sample().vertices[0:6, :] == pytest.approx(
        np.array(
            [
                [-0.5, -0.4330127018922193],
                [-0.25, 0.0],
                [0.0, -0.4330127018922193],
                [0.0, 0.4330127018922193],
                [0.25, 0.0],
                [0.5, -0.4330127018922193],
            ]
        ),
        1.0e-4,
    )


def test_with_vertices(one_triangle):
    triangle = one_triangle.with_vertices(np.array([[0, 0], [1, 0], [0.5, 1]]))
    assert triangle.triangles == pytest.approx(
        np.array([[[1.0, 0.0], [0.5, 1.0], [0.0, 0.0]]]), 1.0e-4
    )


def _test_multiple_with_vertices(one_triangle, plot):
    up_sampled = one_triangle.up_sample()
    plot(up_sampled.with_vertices(2 * up_sampled.vertices).triangles.tolist())


def test_for_indexes(two_triangles):
    assert two_triangles.for_indexes(np.array([0])).triangles == pytest.approx(
        np.array(
            [
                [
                    [0.0, 0.4330127018922193],
                    [0.5, -0.4330127018922193],
                    [-0.5, -0.4330127018922193],
                ]
            ]
        ),
        1.0e-4,
    )


def test_means(one_triangle):
    assert one_triangle.means == pytest.approx(
        np.array([[0.0, -0.14433756729740643]]), 1.0e-4
    )


def test_triangles_touch():
    triangles = CoordinateArrayTriangles(
        np.array([[0, 0], [2, 0]]),
    )

    assert max(triangles.triangles[0][:, 0]) == min(triangles.triangles[1][:, 0])

    triangles = CoordinateArrayTriangles(
        np.array([[0, 0], [0, 1]]),
    )
    assert max(triangles.triangles[0][:, 1]) == min(triangles.triangles[1][:, 1])


def test_from_grid_regression():
    triangles = CoordinateArrayTriangles.for_limits_and_scale(
        x_min=-4.75,
        x_max=4.75,
        y_min=-4.75,
        y_max=4.75,
        scale=0.5,
    )

    x = triangles.vertices[:, 0]
    assert min(x) <= -4.75
    assert max(x) >= 4.75

    y = triangles.vertices[:, 1]
    assert min(y) <= -4.75
    assert max(y) >= 4.75


@pytest.fixture
def one_triangle():
    return CoordinateArrayTriangles(
        coordinates=np.array([[0, 0]]),
        side_length=1.0,
    )


@jax.jit
def full_routine(triangles):
    neighborhood = triangles.neighborhood()
    up_sampled = neighborhood.up_sample()
    with_vertices = up_sampled.with_vertices(up_sampled.vertices)
    indexes = with_vertices.containing_indices(Point(0.1, 0.1))
    return up_sampled.for_indexes(indexes)


# def test_full_routine(one_triangle, compare_with_nans):
#     result = full_routine(one_triangle)
#
#     assert compare_with_nans(
#         result.triangles,
#         np.array(
#             [
#                 [
#                     [0.0, 0.4330126941204071],
#                     [0.25, 0.0],
#                     [-0.25, 0.0],
#                 ]
#             ]
#         ),
#     )


def test_neighborhood(one_triangle):
    assert np.allclose(
        np.array(jax.jit(one_triangle.neighborhood)().triangles),
        np.array(
            [
                [
                    [-0.5, -0.4330126941204071],
                    [-1.0, 0.4330126941204071],
                    [0.0, 0.4330126941204071],
                ],
                [
                    [0.0, -1.299038052558899],
                    [-0.5, -0.4330126941204071],
                    [0.5, -0.4330126941204071],
                ],
                [
                    [0.0, 0.4330126941204071],
                    [0.5, -0.4330126941204071],
                    [-0.5, -0.4330126941204071],
                ],
                [
                    [0.5, -0.4330126941204071],
                    [0.0, 0.4330126941204071],
                    [1.0, 0.4330126941204071],
                ],
            ]
        ),
    )


def test_up_sample(one_triangle):
    up_sampled = jax.jit(one_triangle.up_sample)()
    assert np.allclose(
        np.array(up_sampled.triangles),
        np.array(
            [
                [
                    [[0.0, -0.4330126941204071], [-0.25, 0.0], [0.25, 0.0]],
                    [
                        [0.25, 0.0],
                        [0.5, -0.4330126941204071],
                        [0.0, -0.4330126941204071],
                    ],
                    [
                        [-0.25, 0.0],
                        [0.0, -0.4330126941204071],
                        [-0.5, -0.4330126941204071],
                    ],
                    [[0.0, 0.4330126941204071], [0.25, 0.0], [-0.25, 0.0]],
                ]
            ]
        ),
    )


def test_means(one_triangle):
    assert len(one_triangle.means) == 1

    up_sampled = one_triangle.up_sample()
    neighborhood = up_sampled.neighborhood()
    assert np.count_nonzero(~np.isnan(neighborhood.means).any(axis=1)) == 10


ONE_TRIANGLE_AREA = HEIGHT_FACTOR * 0.5


def test_area(one_triangle):
    assert one_triangle.area == ONE_TRIANGLE_AREA
    assert one_triangle.up_sample().area == ONE_TRIANGLE_AREA

    neighborhood = one_triangle.neighborhood()
    assert neighborhood.area == 4 * ONE_TRIANGLE_AREA
    assert neighborhood.up_sample().area == 4 * ONE_TRIANGLE_AREA
    assert neighborhood.neighborhood().area == 10 * ONE_TRIANGLE_AREA
