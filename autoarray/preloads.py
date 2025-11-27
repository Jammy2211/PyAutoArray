import logging

import numpy as np

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


def mapper_indices_from(total_linear_light_profiles, total_mapper_pixels):

    return np.arange(
        total_linear_light_profiles,
        total_linear_light_profiles + total_mapper_pixels,
        dtype=int,
    )


class Preloads:

    def __init__(
        self,
        mapper_indices: np.ndarray = None,
        source_pixel_zeroed_indices: np.ndarray = None,
        image_plane_mesh_grid_list: np.ndarray = None,
        linear_light_profile_blurred_mapping_matrix=None,
    ):
        """
        Stores preloaded arrays and matrices used during pixelized linear inversions, improving both performance
        and compatibility with JAX.

        Some arrays (e.g. `mapper_indices`) are required to be defined before sampling begins, because JAX demands
        that input shapes remain static. These are used during each inversion to ensure consistent matrix shapes
        for all likelihood evaluations.

        Other arrays (e.g. parts of the curvature matrix) are preloaded purely to improve performance. In cases where
        the source model is fixed (e.g. when fitting only the lens light), sections of the curvature matrix do not
        change and can be reused, avoiding redundant computation.

        Returns a list of image-plane mesh-grids, which are image-plane grids defining the centres of the pixels of
        the pixelization's mesh (e.g. the centres of Voronoi pixels).

        The `image_mesh` attribute of the pixelization object defines whether the centre of each mesh pixel are
        determined in the image-plane. When this is the case, the pixelization therefore has an image-plane mesh-grid,
        which needs to be computed before the inversion is performed.

        This function iterates over all galaxies with pixelizations, determines which pixelizations have an
        `image_mesh` and for these pixelizations computes the image-plane mesh-grid.

        It returns a list of all image-plane mesh-grids, which in the functions `mapper_from` and `mapper_galaxy_dict`
        are grouped into a `Mapper` object with other information required to perform the inversion using the
        pixelization.

        The order of this list is not important, because the `linear_obj_galaxy_dict` function associates each
        mapper object (and therefore image-plane mesh-grid) with the galaxy it belongs to and is therefore used
        elsewhere in the code (e.g. the fit module) to match inversion results to galaxies.

        Certain image meshes adapt their pixels to the dataset, for example congregating the pixels to the brightest
        regions of the image. This requires that `adapt_images` are used when setting up the image-plane mesh-grid.
        This function uses the `adapt_images` attribute of the `GalaxiesToInversion` object pass these images and
        raise an error if they are not present.

        Parameters
        ----------
        mapper_indices
            The integer indices of mapper pixels in the inversion. Used to extract reduced matrices (e.g.
            `curvature_matrix_reduced`) that compute the pixelized inversion's log evidence term, where the indicies
            are requirred to separate the rows and columns of matrices from linear light profiles.
        source_pixel_zeroed_indices
            Indices of source pixels that should be set to zero in the reconstruction. These typically correspond to
            outer-edge source-plane regions with no image-plane mapping (e.g. outside a circular mask), helping
            separate the lens light from the pixelized source model.
        image_plane_mesh_grid
            The (y,x) coordinates of the image-plane mesh grid used by pixelizations that start from pixels
            being defined in the image-plane (e.g. overlaying a uniform grid of pixels on the image-plane, which
            make up Delaunay triangles in the source-plane).
        linear_light_profile_blurred_mapping_matrix
            The evaluated images of the linear light profiles that make up the blurred mapping matrix component of the
            inversion, with the other component being the pixelization's pixels. These are fixed when the lens light
            is fixed to the maximum likelihood solution, allowing the blurred mapping matrix to be preloaded, but
            the intensity values will still be solved for during the inversion.
        """
        self.mapper_indices = None
        self.source_pixel_zeroed_indices = None
        self.source_pixel_zeroed_indices_to_keep = None
        self.linear_light_profile_blurred_mapping_matrix = None

        if mapper_indices is not None:

            self.mapper_indices = np.array(mapper_indices)

        if source_pixel_zeroed_indices is not None:

            self.source_pixel_zeroed_indices = np.array(source_pixel_zeroed_indices)

            ids_zeros = np.array(source_pixel_zeroed_indices, dtype=int)

            values_to_solve = np.ones(np.max(mapper_indices) + 1, dtype=bool)
            values_to_solve[ids_zeros] = False

            self.source_pixel_zeroed_indices_to_keep = np.where(values_to_solve)[0]

        if image_plane_mesh_grid_list is not None:

            self.image_plane_mesh_grid_list = []

            for image_plane_mesh_grid in image_plane_mesh_grid_list:

                if image_plane_mesh_grid is not None:
                    self.image_plane_mesh_grid_list.append(np.array(image_plane_mesh_grid))
                else:
                    self.image_plane_mesh_grid_list.append(None)

        if linear_light_profile_blurred_mapping_matrix is not None:

            self.linear_light_profile_blurred_mapping_matrix = np.array(
                linear_light_profile_blurred_mapping_matrix
            )
