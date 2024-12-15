from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d, griddata
from typing import Optional

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.mask.mask_2d import Mask2D
from autoarray.inversion.pixelization.image_mesh.abstract_weighted import (
    AbstractImageMeshWeighted,
)
from autoarray.operators.over_sampling.over_sampler import OverSampler
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray import exc


def gilbert2d(width, height):
    """
    Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
    2D rectangular grids. Generates discrete 2D coordinates to fill a rectangle
    of size (width x height).
    """

    if width >= height:
        yield from generate2d(0, 0, width, 0, 0, height)
    else:
        yield from generate2d(0, 0, 0, height, width, 0)


def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def generate2d(x, y, ax, ay, bx, by):
    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (sgn(ax), sgn(ay))  # unit major direction
    (dbx, dby) = (sgn(bx), sgn(by))  # unit orthogonal direction

    if h == 1:
        # trivial row fill
        for i in range(0, w):
            yield (x, y)
            (x, y) = (x + dax, y + day)
        return

    if w == 1:
        # trivial column fill
        for i in range(0, h):
            yield (x, y)
            (x, y) = (x + dbx, y + dby)
        return

    (ax2, ay2) = (ax // 2, ay // 2)
    (bx2, by2) = (bx // 2, by // 2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2 * w > 3 * h:
        if (w2 % 2) and (w > 2):
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)

        # long case: split in two parts only
        yield from generate2d(x, y, ax2, ay2, bx, by)
        yield from generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)

    else:
        if (h2 % 2) and (h > 2):
            # prefer even steps
            (bx2, by2) = (bx2 + dbx, by2 + dby)

        # standard case: one step up, one long horizontal, one step down
        yield from generate2d(x, y, bx2, by2, ax2, ay2)
        yield from generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
        yield from generate2d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            -bx2,
            -by2,
            -(ax - ax2),
            -(ay - ay2),
        )


def super_resolution_grid_from(img_2d, mask, mask_radius, pixel_scales, sub_scale=11):
    """
    This function will create a higher resolution grid for the img_2d. The new grid and its
    interpolated values will be used to generate a sparse image grid.

    img_2d: the hyper image in 2d (e.g. hyper_source_model_image.native)
    mask: the mask used for the fitting.
    mask_radius: the circular mask radius. Currently, the code only works with a circular mask.
    sub_scale: oversampling scale for each image pixel.
    """

    shape_nnn = np.shape(mask)[0]

    grid = Grid2D.uniform(
        shape_native=(shape_nnn, shape_nnn),
        pixel_scales=pixel_scales,
    )

    new_mask = Mask2D.circular(
        shape_native=(shape_nnn, shape_nnn),
        pixel_scales=pixel_scales,
        centre=mask.origin,
        radius=mask_radius,
    )

    over_sampler = OverSampler(mask=new_mask, sub_size=sub_scale)

    new_grid = over_sampler.over_sampled_grid

    new_img = griddata(
        points=grid, values=img_2d.ravel(), xi=new_grid, fill_value=0.0, method="linear"
    )

    return new_img, new_grid


def grid_hilbert_order_from(length, mask_radius):
    """
    This function will create a grid in the Hilbert space-filling curve order.

    length: the size of the square grid.
    mask_radius: the circular mask radius. This code only works with a circular mask.
    """

    xy_generator = gilbert2d(length, length)

    x1d_hb = np.zeros(length * length)
    y1d_hb = np.zeros(length * length)

    count = 0

    for x, y in xy_generator:
        x1d_hb[count] = x
        y1d_hb[count] = y
        count += 1

    x1d_hb /= length
    y1d_hb /= length

    x1d_hb -= 0.5
    y1d_hb -= 0.5

    x1d_hb *= 2.0 * mask_radius
    y1d_hb *= 2.0 * mask_radius

    return x1d_hb, y1d_hb


def image_and_grid_from(image, mask, mask_radius, pixel_scales, hilbert_length):
    """
    This code will create a grid in Hilbert space-filling curve order and an interpolated hyper
    image associated to that grid.
    """

    # For multi wavelength fits the input image may be a different resolution than the mask.

    try:
        shape_nnn = np.shape(image.native)[0]
    except AttributeError:
        shape_nnn = np.shape(mask)[0]

    grid = Grid2D.uniform(
        shape_native=(shape_nnn, shape_nnn),
        pixel_scales=pixel_scales,
    )

    x1d_hb, y1d_hb = grid_hilbert_order_from(
        length=hilbert_length, mask_radius=mask_radius
    )

    grid_hb = np.stack((y1d_hb, x1d_hb), axis=-1)
    grid_hb_radius = np.sqrt(grid_hb[:, 0] ** 2.0 + grid_hb[:, 1] ** 2.0)
    new_grid = grid_hb[grid_hb_radius <= mask_radius]

    new_img = griddata(
        points=grid,
        values=image.native.ravel(),
        xi=new_grid,
        fill_value=0.0,
        method="linear",
    )

    return new_img, new_grid


def inverse_transform_sampling_interpolated(probabilities, n_samples, gridx, gridy):
    """
    Given a 1d cumulative probability function, this code will generate points following the
    probability distribution.

    probabilities: 1D normalized cumulative probablity curve.
    n_samples: the number of points to draw.
    """

    cdf = np.cumsum(probabilities)
    npixels = len(probabilities)
    id_range = np.arange(0, npixels)
    cdf[0] = 0.0
    intp_func = interp1d(cdf, id_range, kind="linear")
    intp_func_x = interp1d(id_range, gridx, kind="linear")
    intp_func_y = interp1d(id_range, gridy, kind="linear")
    linear_points = np.linspace(0, 0.99999999, int(n_samples))
    output_ids = intp_func(linear_points)
    output_x = intp_func_x(output_ids)
    output_y = intp_func_y(output_ids)

    return output_ids, output_x, output_y


class Hilbert(AbstractImageMeshWeighted):
    def __init__(
        self,
        pixels=10.0,
        weight_floor=0.0,
        weight_power=0.0,
    ):
        """
        Computes an image-mesh by computing the Hilbert curve of the adapt data and drawing points from it.

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
        """

        super().__init__(
            pixels=pixels,
            weight_floor=weight_floor,
            weight_power=weight_power,
        )

    def image_plane_mesh_grid_from(
        self,
        mask: Mask2D,
        adapt_data: Optional[np.ndarray],
        settings: SettingsInversion = None,
    ) -> Grid2DIrregular:
        """
        Returns an image mesh by running the Hilbert curve on the weight map.

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
        if not mask.is_circular:
            raise exc.PixelizationException(
                """
                Hilbert image-mesh has been called but the input grid does not use a circular mask.
                
                Ensure that analysis is using a circular mask via the Mask2D.circular classmethod.
                """
            )

        adapt_data_hb, grid_hb = image_and_grid_from(
            image=adapt_data,
            mask=mask,
            mask_radius=mask.circular_radius,
            pixel_scales=mask.pixel_scales,
            hilbert_length=193,
        )

        weight_map = self.weight_map_from(adapt_data=adapt_data_hb)

        weight_map /= np.sum(weight_map)

        (
            drawn_id,
            drawn_x,
            drawn_y,
        ) = inverse_transform_sampling_interpolated(
            probabilities=weight_map,
            n_samples=self.pixels,
            gridx=grid_hb[:, 1],
            gridy=grid_hb[:, 0],
        )

        mesh_grid = Grid2DIrregular(values=np.stack((drawn_y, drawn_x), axis=-1))

        self.check_mesh_pixels_per_image_pixels(
            mask=mask, mesh_grid=mesh_grid, settings=settings
        )

        self.check_adapt_background_pixels(
            mask=mask, mesh_grid=mesh_grid, adapt_data=adapt_data, settings=settings
        )

        return mesh_grid
