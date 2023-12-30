from __future__ import annotations
import numpy as np
from typing import Optional

from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.inversion.pixelization.image_mesh.abstract_weighted import (
    AbstractImageMeshWeighted,
)
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray.inversion.pixelization.image_mesh.hilbert import image_and_grid_from
from autoarray.inversion.pixelization.image_mesh.hilbert import (
    inverse_transform_sampling_interpolated,
)

from autoarray import exc


class HilbertBalanced(AbstractImageMeshWeighted):
    def __init__(
        self,
        pixels=10.0,
        weight_floor=0.0,
        weight_power=0.0,
        ratio=0.8,
    ):
        """
        Computes a balanced image-mesh by computing the Hilbert curve of the adapt data and drawing points from it.

        The standard `Hilbert` image-mesh suffers a systematic where the vast majority of points are drawn from
        the high weighted reigons. This often leaves few points to reconstruct the lower weight regions, leading to
        discontinuities in the reconstruction.

        This image-mesh addresses this by drawing half the points from the weight map and the other half from
        (1 - weight map). This ensures both high and low weighted regions are sampled equally, but still has sufficient
        flexibility to dedicate many points to the highest weighted regions.

        This requires an adapt-image, which is the image that the Hilbert curve algorithm adapts to in order to compute
        the image mesh. This could simply be the image itself, or a model fit to the image which removes certain
        features or noise.

        For example, using the adapt image, the image mesh is computed as follows:

        1) Convert the adapt image to a weight map, which is a 2D array of weight values.

        2) Run the Hilbert algorithm on the weight map, such that the image mesh pixels cluster around the weight map
        values with higher values.

        Parameters
        ----------
        pixels
            The total number of pixels in the image mesh and drawn from the Hilbert curve.
        weight_floor
            The minimum weight value in the weight map, which allows more pixels to be drawn from the lower weight
            regions of the adapt image.
        weight_power
            The power the weight values are raised too, which allows more pixels to be drawn from the higher weight
            regions of the adapt image.
        ratio
            The ratio between the number of pixdels in the image mesh drawn from the weight map and (1 - weight map).
            For example, if there are 1000 pixels and a ratio of 0.8, 800 pixels are drawn from the weight map
            and therefore make up the high weighted region of the image mesh.
        """

        super().__init__(
            pixels=pixels,
            weight_floor=weight_floor,
            weight_power=weight_power,
        )

        self.ratio = ratio

    def image_plane_mesh_grid_from(
        self, grid: Grid2D, adapt_data: Optional[np.ndarray]
    ) -> Grid2DIrregular:
        """
        Returns an image mesh by running the balanced Hilbert curve on the weight map.

        See the `__init__` docstring for a full description of how this is performed.

        Parameters
        ----------
        grid
            The grid of (y,x) coordinates of the image data the pixelization fits, which the Hilbert curve adapts to.
        adapt_data
            The weights defining the regions of the image the Hilbert curve adapts to.

        Returns
        -------

        """

        if not grid.mask.is_circular:
            raise exc.PixelizationException(
                """
                Hilbert image-mesh has been called but the input grid does not use a circular mask.

                Ensure that analysis is using a circular mask via the Mask2D.circular classmethod.
                """
            )

        adapt_data_hb, grid_hb = image_and_grid_from(
            image=adapt_data,
            mask=grid.mask,
            mask_radius=grid.mask.circular_radius,
            pixel_scales=grid.mask.pixel_scales,
            hilbert_length=193,
        )

        weight_map = self.weight_map_from(adapt_data=adapt_data_hb)

        weight_map_background = 1.0 - weight_map

        weight_map /= np.sum(weight_map)
        weight_map_background /= np.sum(weight_map_background)

        pixels = int(self.pixels * self.ratio)

        (
            drawn_id,
            drawn_x,
            drawn_y,
        ) = inverse_transform_sampling_interpolated(
            probabilities=weight_map,
            n_samples=pixels,
            gridx=grid_hb[:, 1],
            gridy=grid_hb[:, 0],
        )

        grid = np.stack((drawn_y, drawn_x), axis=-1)

        (
            drawn_id,
            drawn_x,
            drawn_y,
        ) = inverse_transform_sampling_interpolated(
            probabilities=weight_map_background,
            n_samples=(self.pixels  - pixels) + 1,
            gridx=grid_hb[:, 1],
            gridy=grid_hb[:, 0],
        )

        grid_background = np.stack((drawn_y, drawn_x), axis=-1)

        return Grid2DIrregular(
            values=np.concatenate((grid, grid_background[1:, :]), axis=0)
        )
