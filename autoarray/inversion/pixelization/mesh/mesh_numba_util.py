import numpy as np

from typing import List, Tuple, Union

from autoarray import numba_util


@numba_util.jit()
def delaunay_triangle_area_from(
    corner_0: Tuple[float, float],
    corner_1: Tuple[float, float],
    corner_2: Tuple[float, float],
) -> float:
    """
    Returns the area within a Delaunay triangle where the three corners are located at the (x,y) coordinates given by
    the inputs `corner_a` `corner_b` and `corner_c`.

    This function actually returns the area of any triangle, but the term `delaunay` is included in the title to
    separate it from the `rectangular` and `voronoi` methods in `mesh_util.py`.

    Parameters
    ----------
    corner_0
        The (x,y) coordinates of the triangle's first corner.
    corner_1
        The (x,y) coordinates of the triangle's second corner.
    corner_2
        The (x,y) coordinates of the triangle's third corner.

    Returns
    -------
    The area of the triangle given the input (x,y) corners.
    """

    x1 = corner_0[0]
    y1 = corner_0[1]
    x2 = corner_1[0]
    y2 = corner_1[1]
    x3 = corner_2[0]
    y3 = corner_2[1]

    return 0.5 * np.abs(x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x3 * y2 - x1 * y3)


def delaunay_interpolated_array_from(
    shape_native: Tuple[int, int],
    interpolation_grid_slim: np.ndarray,
    pixel_values: np.ndarray,
    delaunay: "scipy.spatial.Delaunay",
) -> np.ndarray:
    """
    Given a Delaunay triangulation and 1D values at the node of each Delaunay pixel (e.g. the connecting points where
    triangles meet), interpolate these values to a uniform 2D (y,x) grid.

    By mapping the delaunay's value to a regular grid this enables a source reconstruction of an inversion to be
    output to a .fits file.

    The `grid_interpolate_slim`, which gives the (y,x) coordinates the values are evaluated at for interpolation,
    need not be regular and can have undergone coordinate transforms (e.g. it can be the `source_plane_mesh_grid`)
    of a `Mapper`.

    The shape of `grid_interpolate_slim` therefore must be equal to `shape_native[0] * shape_native[1]`, but the (y,x)
    coordinates themselves do not need to be uniform.

    Parameters
    ----------
    shape_native
        The 2D (y,x) shape of the uniform grid the values are interpolated on too.
    interpolation_grid_slim
        A 1D grid of (y,x) coordinates where each interpolation is evaluated. The shape of this grid must be equal to
        shape_native[0] * shape_native[1], but it does not need to be uniform itself.
    pixel_values
        The values of the Delaunay nodes (e.g. the connecting points where triangles meet) which are interpolated
        to compute the value in each pixel on the `interpolated_grid`.
    delaunay
        A `scipy.spatial.Delaunay` object which contains all functionality describing the Delaunay triangulation.

    Returns
    -------
    The input values interpolated to the `grid_interpolate_slim` (y,x) coordintes given the Delaunay triangulation.

    """
    simplex_index_for_interpolate_index = delaunay.find_simplex(interpolation_grid_slim)

    simplices = delaunay.simplices
    pixel_points = delaunay.points

    interpolated_array = np.zeros(len(interpolation_grid_slim))

    for slim_index in range(len(interpolation_grid_slim)):
        simplex_index = simplex_index_for_interpolate_index[slim_index]
        interpolating_point = tuple(interpolation_grid_slim[slim_index])

        if simplex_index == -1:
            cloest_pixel_index = np.argmin(
                np.sum((pixel_points - interpolating_point) ** 2.0, axis=1)
            )
            interpolated_array[slim_index] = pixel_values[cloest_pixel_index]
        else:
            triangle_points = pixel_points[simplices[simplex_index]]
            triangle_values = pixel_values[simplices[simplex_index]]

            area_0 = delaunay_triangle_area_from(
                corner_0=triangle_points[1],
                corner_1=triangle_points[2],
                corner_2=interpolating_point,
            )
            area_1 = delaunay_triangle_area_from(
                corner_0=triangle_points[0],
                corner_1=triangle_points[2],
                corner_2=interpolating_point,
            )
            area_2 = delaunay_triangle_area_from(
                corner_0=triangle_points[0],
                corner_1=triangle_points[1],
                corner_2=interpolating_point,
            )
            norm = area_0 + area_1 + area_2

            weight_abc = np.array([area_0, area_1, area_2]) / norm

            interpolated_array[slim_index] = np.sum(weight_abc * triangle_values)

    return interpolated_array.reshape(shape_native)
