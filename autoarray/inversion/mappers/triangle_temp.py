import numpy as np
from scipy.spatial import Delaunay


def triangle_area(pa, pb, pc):

    x1 = pa[0]
    y1 = pa[1]
    x2 = pb[0]
    y2 = pb[1]
    x3 = pc[0]
    y3 = pc[1]

    return 0.5 * np.abs(x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x3 * y2 - x1 * y3)


def get_triangle_points_neighbours(relocated_grid, relocated_pixelization_grid):

    tri = Delaunay(relocated_pixelization_grid)
    triangle_ids_of_points = tri.find_simplex(relocated_grid)
    tri_simplices = tri.simplices

    return triangle_ids_of_points, tri_simplices


def mapping_matrix_Delaunay_bilinear_interpolation_from(
    relocated_grid,
    relocated_pixelization_grid,
    triangle_ids_of_points,
    tri_simplices,
    pixels: int,
    total_mask_pixels: int,
    slim_index_for_sub_slim_index: np.ndarray,
    sub_fraction: float,
) -> np.ndarray:

    mapping_matrix = np.zeros((total_mask_pixels, pixels))

    for sub_slim_index in range(slim_index_for_sub_slim_index.shape[0]):

        neighbor_triangle_id = triangle_ids_of_points[sub_slim_index]

        if neighbor_triangle_id != -1:

            sub_gird_coordinate_on_source_place = relocated_grid[sub_slim_index]

            neighbor_triangle_veritces_ids = tri_simplices[neighbor_triangle_id]
            neighbor_triangle_veritces_coordinates = relocated_pixelization_grid[
                neighbor_triangle_veritces_ids
            ]

            term0 = triangle_area(
                pa=neighbor_triangle_veritces_coordinates[1],
                pb=neighbor_triangle_veritces_coordinates[2],
                pc=sub_gird_coordinate_on_source_place,
            )
            term1 = triangle_area(
                pa=neighbor_triangle_veritces_coordinates[0],
                pb=neighbor_triangle_veritces_coordinates[2],
                pc=sub_gird_coordinate_on_source_place,
            )
            term2 = triangle_area(
                pa=neighbor_triangle_veritces_coordinates[0],
                pb=neighbor_triangle_veritces_coordinates[1],
                pc=sub_gird_coordinate_on_source_place,
            )

            norm = term0 + term1 + term2

            weight_abc = np.array([term0, term1, term2]) / norm

            mapping_matrix[slim_index_for_sub_slim_index[sub_slim_index]][
                neighbor_triangle_veritces_ids
            ] += (sub_fraction * weight_abc)

    return mapping_matrix
