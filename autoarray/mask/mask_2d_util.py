import numpy as np
from typing import Tuple, Union
import warnings

from autoarray import exc
from autoarray import numba_util
from autoarray.structures.grids import grid_2d_util
from autoarray import type as ty


@numba_util.jit()
def mask_2d_centres_from(
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    centre: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Returns the (y,x) scaled central coordinates of a mask from its shape, pixel-scales and centre.

    The coordinate system is defined such that the positive y axis is up and positive x axis is right.

    Parameters
    ----------
    shape_native
        The (y,x) shape of the 2D array the scaled centre is computed for.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D array.
    centre : (float, flloat)
        The (y,x) centre of the 2D mask.

    Returns
    -------
    tuple (float, float)
        The (y,x) scaled central coordinates of the input array.

    Examples
    --------
    centres_scaled = centres_from(shape=(5,5), pixel_scales=(0.5, 0.5), centre=(0.0, 0.0))
    """
    y_centre_scaled = (float(shape_native[0] - 1) / 2) - (centre[0] / pixel_scales[0])
    x_centre_scaled = (float(shape_native[1] - 1) / 2) + (centre[1] / pixel_scales[1])

    return (y_centre_scaled, x_centre_scaled)


@numba_util.jit()
def total_pixels_2d_from(mask_2d: np.ndarray) -> int:
    """
    Returns the total number of unmasked pixels in a mask.

    Parameters
    ----------
    mask_2d
        A 2D array of bools, where `False` values are unmasked and included when counting pixels.

    Returns
    -------
    int
        The total number of pixels that are unmasked.

    Examples
    --------

    mask = np.array([[True, False, True],
                 [False, False, False]
                 [True, False, True]])

    total_regular_pixels = total_regular_pixels_from(mask=mask)
    """

    total_regular_pixels = 0

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):
            if not mask_2d[y, x]:
                total_regular_pixels += 1

    return total_regular_pixels


@numba_util.jit()
def mask_2d_circular_from(
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    radius: float,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Returns a circular mask from the 2D mask array shape and radius of the circle.

    This creates a 2D array where all values within the mask radius are unmasked and therefore `False`.

    Parameters
    ----------
    shape_native: Tuple[int, int]
        The (y,x) shape of the mask in units of pixels.
    pixel_scales
        The scaled units to pixel units conversion factor of each pixel.
    radius
        The radius (in scaled units) of the circle within which pixels unmasked.
    centre
            The centre of the circle used to mask pixels.

    Returns
    -------
    ndarray
        The 2D mask array whose central pixels are masked as a circle.

    Examples
    --------
    mask = mask_circular_from(
        shape=(10, 10), pixel_scales=0.1, radius=0.5, centre=(0.0, 0.0))
    """

    mask_2d = np.full(shape_native, True)

    centres_scaled = mask_2d_centres_from(
        shape_native=mask_2d.shape, pixel_scales=pixel_scales, centre=centre
    )

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):
            y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
            x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

            r_scaled = np.sqrt(x_scaled**2 + y_scaled**2)

            if r_scaled <= radius:
                mask_2d[y, x] = False

    return mask_2d


@numba_util.jit()
def mask_2d_circular_annular_from(
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    inner_radius: float,
    outer_radius: float,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Returns an circular annular mask from an input inner and outer mask radius and shape.

    This creates a 2D array where all values within the inner and outer radii are unmasked and therefore `False`.

    Parameters
    ----------
    shape_native
        The (y,x) shape of the mask in units of pixels.
    pixel_scales
        The scaled units to pixel units conversion factor of each pixel.
    inner_radius
        The radius (in scaled units) of the inner circle outside of which pixels are unmasked.
    outer_radius
        The radius (in scaled units) of the outer circle within which pixels are unmasked.
    centre
            The centre of the annulus used to mask pixels.

    Returns
    -------
    ndarray
        The 2D mask array whose central pixels are masked as a annulus.

    Examples
    --------
    mask = mask_annnular_from(
        shape=(10, 10), pixel_scales=0.1, inner_radius=0.5, outer_radius=1.5, centre=(0.0, 0.0))
    """

    mask_2d = np.full(shape_native, True)

    centres_scaled = mask_2d_centres_from(
        shape_native=mask_2d.shape, pixel_scales=pixel_scales, centre=centre
    )

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):
            y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
            x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

            r_scaled = np.sqrt(x_scaled**2 + y_scaled**2)

            if outer_radius >= r_scaled >= inner_radius:
                mask_2d[y, x] = False

    return mask_2d


@numba_util.jit()
def mask_2d_circular_anti_annular_from(
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    inner_radius: float,
    outer_radius: float,
    outer_radius_2_scaled: float,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Returns an anti-annular mask from an input inner and outer mask radius and shape. The anti-annular is analogous to
    the annular mask but inverted, whereby its unmasked values are those inside the annulus.

    This creates a 2D array where all values outside the inner and outer radii are unmasked and therefore `False`.

    Parameters
    ----------
    shape_native
        The (y,x) shape of the mask in units of pixels.
    pixel_scales
        The scaled units to pixel units conversion factor of each pixel.
    inner_radius
        The inner radius in scaled units of the annulus within which pixels are `False` and unmasked.
    outer_radius
        The first outer radius in scaled units of the annulus within which pixels are `True` and masked.
    outer_radius_2
        The second outer radius in scaled units of the annulus within which pixels are `False` and unmasked and
        outside of which all entries are `True` and masked.
    centre
            The centre of the annulus used to mask pixels.

    Returns
    -------
    ndarray
        The 2D mask array whose central pixels are masked as a annulus.

    Examples
    --------
    mask = mask_annnular_from(
        shape=(10, 10), pixel_scales=0.1, inner_radius=0.5, outer_radius=1.5, centre=(0.0, 0.0))

    """

    mask_2d = np.full(shape_native, True)

    centres_scaled = mask_2d_centres_from(
        shape_native=mask_2d.shape, pixel_scales=pixel_scales, centre=centre
    )

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):
            y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
            x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

            r_scaled = np.sqrt(x_scaled**2 + y_scaled**2)

            if (
                inner_radius >= r_scaled
                or outer_radius_2_scaled >= r_scaled >= outer_radius
            ):
                mask_2d[y, x] = False

    return mask_2d


def mask_2d_via_pixel_coordinates_from(
    shape_native: Tuple[int, int], pixel_coordinates: [list], buffer: int = 0
) -> np.ndarray:
    """
    Returns a mask where all unmasked `False` entries are defined from an input list of list of pixel coordinates.

    These may be buffed via an input ``buffer``, whereby all entries in all 8 neighboring directions by this
    amount.

    Parameters
    ----------
    shape_native (int, int)
        The (y,x) shape of the mask in units of pixels.
    pixel_coordinates : [[int, int]]
        The input lists of 2D pixel coordinates where `False` entries are created.
    buffer
        All input ``pixel_coordinates`` are buffed with `False` entries in all 8 neighboring directions by this
        amount.
    """

    mask_2d = np.full(shape=shape_native, fill_value=True)

    for y, x in pixel_coordinates:
        mask_2d[y, x] = False

    if buffer == 0:
        return mask_2d
    else:
        return buffed_mask_2d_from(mask_2d=mask_2d, buffer=buffer)


@numba_util.jit()
def elliptical_radius_from(
    y_scaled: float, x_scaled: float, angle: float, axis_ratio: float
) -> float:
    """
    Returns the elliptical radius of an ellipse from its (y,x) scaled centre, rotation angle `angle` defined in degrees
    counter-clockwise from the positive x-axis and its axis-ratio.

    This is used by the function `mask_elliptical_from` to determine the radius of every (y,x) coordinate in elliptical
    units when deciding if it is within the mask.

    Parameters
    ----------
    y_scaled
        The scaled y coordinate in Cartesian coordinates which is converted to elliptical coordinates.
    x_scaled
        The scaled x coordinate in Cartesian coordinates which is converted to elliptical coordinates.
    angle
            The rotation angle in degrees counter-clockwise from the positive x-axis
    axis_ratio
            The axis-ratio of the ellipse (minor axis / major axis).

    Returns
    -------
    float
        The radius of the input scaled (y,x) coordinate on the ellipse's ellipitcal coordinate system.
    """
    r_scaled = np.sqrt(x_scaled**2 + y_scaled**2)

    theta_rotated = np.arctan2(y_scaled, x_scaled) + np.radians(angle)

    y_scaled_elliptical = r_scaled * np.sin(theta_rotated)
    x_scaled_elliptical = r_scaled * np.cos(theta_rotated)

    return np.sqrt(
        x_scaled_elliptical**2.0 + (y_scaled_elliptical / axis_ratio) ** 2.0
    )


@numba_util.jit()
def mask_2d_elliptical_from(
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    major_axis_radius: float,
    axis_ratio: float,
    angle: float,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Returns an elliptical mask from an input major-axis mask radius, axis-ratio, rotational angle, shape and
    centre.

    This creates a 2D array where all values within the ellipse are unmasked and therefore `False`.

    Parameters
    ----------
    shape_native: Tuple[int, int]
        The (y,x) shape of the mask in units of pixels.
    pixel_scales
        The scaled units to pixel units conversion factor of each pixel.
    major_axis_radius
        The major-axis (in scaled units) of the ellipse within which pixels are unmasked.
    axis_ratio
            The axis-ratio of the ellipse within which pixels are unmasked.
    angle
            The rotation angle of the ellipse within which pixels are unmasked, (counter-clockwise from the positive
         x-axis).
    centre
            The centre of the ellipse used to mask pixels.

    Returns
    -------
    ndarray
        The 2D mask array whose central pixels are masked as an ellipse.

    Examples
    --------
    mask = mask_elliptical_from(
        shape=(10, 10), pixel_scales=0.1, major_axis_radius=0.5, ell_comps=(0.333333, 0.0), centre=(0.0, 0.0))
    """

    mask_2d = np.full(shape_native, True)

    centres_scaled = mask_2d_centres_from(
        shape_native=mask_2d.shape, pixel_scales=pixel_scales, centre=centre
    )

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):
            y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
            x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

            r_scaled_elliptical = elliptical_radius_from(
                y_scaled, x_scaled, angle, axis_ratio
            )

            if r_scaled_elliptical <= major_axis_radius:
                mask_2d[y, x] = False

    return mask_2d


@numba_util.jit()
def mask_2d_elliptical_annular_from(
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    inner_major_axis_radius: float,
    inner_axis_ratio: float,
    inner_phi: float,
    outer_major_axis_radius: float,
    outer_axis_ratio: float,
    outer_phi: float,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Returns an elliptical annular mask from an input major-axis mask radius, axis-ratio, rotational angle for
    both the inner and outer elliptical annuli and a shape and centre for the mask.

    This creates a 2D array where all values within the elliptical annuli are unmasked and therefore `False`.

    Parameters
    ----------
    shape_native: Tuple[int, int]
        The (y,x) shape of the mask in units of pixels.
    pixel_scales
        The scaled units to pixel units conversion factor of each pixel.
    inner_major_axis_radius
        The major-axis (in scaled units) of the inner ellipse within which pixels are masked.
    inner_axis_ratio
            The axis-ratio of the inner ellipse within which pixels are masked.
    inner_phi
        The rotation angle of the inner ellipse within which pixels are masked, (counter-clockwise from the
        positive x-axis).
    outer_major_axis_radius
        The major-axis (in scaled units) of the outer ellipse within which pixels are unmasked.
    outer_axis_ratio
            The axis-ratio of the outer ellipse within which pixels are unmasked.
    outer_phi
        The rotation angle of the outer ellipse within which pixels are unmasked, (counter-clockwise from the
        positive x-axis).
    centre
            The centre of the elliptical annuli used to mask pixels.

    Returns
    -------
    ndarray
        The 2D mask array whose elliptical annuli pixels are masked.

    Examples
    --------
    mask = mask_elliptical_annuli_from(
        shape=(10, 10), pixel_scales=0.1,
         inner_major_axis_radius=0.5, inner_axis_ratio=0.5, inner_phi=45.0,
         outer_major_axis_radius=1.5, outer_axis_ratio=0.8, outer_phi=90.0,
         centre=(0.0, 0.0))
    """

    mask_2d = np.full(shape_native, True)

    centres_scaled = mask_2d_centres_from(
        shape_native=mask_2d.shape, pixel_scales=pixel_scales, centre=centre
    )

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):
            y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
            x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

            inner_r_scaled_elliptical = elliptical_radius_from(
                y_scaled, x_scaled, inner_phi, inner_axis_ratio
            )

            outer_r_scaled_elliptical = elliptical_radius_from(
                y_scaled, x_scaled, outer_phi, outer_axis_ratio
            )

            if (
                inner_r_scaled_elliptical >= inner_major_axis_radius
                and outer_r_scaled_elliptical <= outer_major_axis_radius
            ):
                mask_2d[y, x] = False

    return mask_2d


@numba_util.jit()
def blurring_mask_2d_from(
    mask_2d: np.ndarray, kernel_shape_native: Tuple[int, int]
) -> np.ndarray:
    """
    Returns a blurring mask from an input mask and psf shape.

    The blurring mask corresponds to all pixels which are outside of the mask but will have a fraction of their
    light blur into the masked region due to PSF convolution. The PSF shape is used to determine which pixels these are.

    If a pixel is identified which is outside the 2D dimensions of the input mask, an error is raised and the user
    should pad the input mask (and associated images).

    Parameters
    ----------
    mask_2d
        A 2D array of bools, where `False` values are unmasked.
    kernel_shape_native
        The 2D shape of the PSF which is used to compute the blurring mask.

    Returns
    -------
    ndarray
        The 2D blurring mask array whose unmasked values (`False`) correspond to where the mask will have PSF light
        blurred into them.

    Examples
    --------
    mask = np.array([[True, True, True],
                     [True, False, True]
                     [True, True, True]])

    blurring_mask = blurring_from(mask=mask)

    """

    blurring_mask_2d = np.full(mask_2d.shape, True)

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):
            if not mask_2d[y, x]:
                for y1 in range(
                    (-kernel_shape_native[0] + 1) // 2,
                    (kernel_shape_native[0] + 1) // 2,
                ):
                    for x1 in range(
                        (-kernel_shape_native[1] + 1) // 2,
                        (kernel_shape_native[1] + 1) // 2,
                    ):
                        if (
                            0 <= x + x1 <= mask_2d.shape[1] - 1
                            and 0 <= y + y1 <= mask_2d.shape[0] - 1
                        ):
                            if mask_2d[y + y1, x + x1]:
                                blurring_mask_2d[y + y1, x + x1] = False
                        else:
                            raise exc.MaskException(
                                "setup_blurring_mask extends beyond the edge "
                                "of the mask - pad the datas array before masking"
                            )

    return blurring_mask_2d


@numba_util.jit()
def mask_2d_via_shape_native_and_native_for_slim(
    shape_native: Tuple[int, int], native_for_slim: np.ndarray
) -> np.ndarray:
    """
    For a slimmed set of data that was computed by mapping unmasked values from a native 2D array of shape
    (total_y_pixels, total_x_pixels), map its slimmed indexes back to the original 2D array to create the
    native 2D mask.

    This uses an array 'native_for_slim' of shape [total_masked_pixels[ where each index gives the native 2D pixel
    indexes of the slimmed array's unmasked pixels, for example:

    - If native_for_slim[0] = [0,0], the first value of the slimmed array maps to the pixel [0,0] of the native 2D array.
    - If native_for_slim[1] = [0,1], the second value of the slimmed array maps to the pixel [0,1] of the native 2D array.
    - If native_for_slim[4] = [1,1], the fifth value of the slimmed array maps to the pixel [1,1] of the native 2D array.

    Parameters
    ----------
    shape_native
        The shape of the 2D array which the pixels are defined on.
    native_for_slim
        An array describing the native 2D array index that every slimmed array index maps too.

    Returns
    -------
    ndarray
        A 2D mask array where unmasked values are `False`.

    Examples
    --------
    native_for_slim = np.array([[0,1], [1,0], [1,1], [1,2], [2,1]])

    mask = mask_from(shape=(3,3), native_for_slim=native_for_slim)
    """

    mask = np.ones(shape_native)

    for index in range(len(native_for_slim)):
        mask[native_for_slim[index, 0], native_for_slim[index, 1]] = False

    return mask


@numba_util.jit()
def mask_slim_indexes_from(
    mask_2d: np.ndarray, return_masked_indexes: bool = True
) -> np.ndarray:
    """
    Returns a 1D array listing all masked (`value=True`) or unmasked pixel indexes (`value=False`) in the mask.

    For example, for the following ``Mask2D``:

    ::
        [[True,  True,  True, True]
         [True, False, False, True],
         [True, False,  True, True],
         [True,  True,  True, True]]

    This has three unmasked (``False`` values) which have the ``slim`` indexes, there ``unmasked_slim`` is:

    ::
        [0, 1, 2]

    Parameters
    ----------
    mask_2d
        The mask for which the 1D unmasked pixel indexes are computed.
    return_masked_indexes
        Whether to return the masked index values (`value=True`) or the unmasked index values (`value=False`).

    Returns
    -------
    np.ndarray
        The 1D indexes of all unmasked pixels on the mask.
    """

    mask_pixel_total = 0

    for y in range(0, mask_2d.shape[0]):
        for x in range(0, mask_2d.shape[1]):
            if mask_2d[y, x] == return_masked_indexes:
                mask_pixel_total += 1

    mask_pixels = np.zeros(mask_pixel_total)
    mask_index = 0
    regular_index = 0

    for y in range(0, mask_2d.shape[0]):
        for x in range(0, mask_2d.shape[1]):
            if mask_2d[y, x] == return_masked_indexes:
                mask_pixels[mask_index] = regular_index
                mask_index += 1

            regular_index += 1

    return mask_pixels


@numba_util.jit()
def check_if_edge_pixel(mask_2d: np.ndarray, y: int, x: int) -> bool:
    """
    Checks if an input [y,x] pixel on the input `mask` is an edge-pixel.

    An edge pixel is defined as a pixel on the mask which is unmasked (has a `False`) value and at least 1 of its 8
    direct neighbors is masked (is `True`).

    Parameters
    ----------
    mask_2d
        The mask for which the input pixel is checked if it is an edge pixel.
    y
        The y pixel coordinate on the mask that is checked for if it is an edge pixel.
    x
        The x pixel coordinate on the mask that is checked for if it is an edge pixel.

    Returns
    -------
    bool
        If `True` the pixel on the mask is an edge pixel, else a `False` is returned because it is not.
    """

    if (
        mask_2d[y + 1, x]
        or mask_2d[y - 1, x]
        or mask_2d[y, x + 1]
        or mask_2d[y, x - 1]
        or mask_2d[y + 1, x + 1]
        or mask_2d[y + 1, x - 1]
        or mask_2d[y - 1, x + 1]
        or mask_2d[y - 1, x - 1]
    ):
        return True
    else:
        return False


@numba_util.jit()
def total_edge_pixels_from(mask_2d: np.ndarray) -> int:
    """
    Returns the total number of edge-pixels in a mask.

    An edge pixel is defined as a pixel on the mask which is unmasked (has a `False`) value and at least 1 of its 8
    direct neighbors is masked (is `True`).

    Parameters
    ----------
    mask_2d
        The mask for which the total number of edge pixels is computed.

    Returns
    -------
    int
        The total number of edge pixels.
    """

    edge_pixel_total = 0

    for y in range(1, mask_2d.shape[0] - 1):
        for x in range(1, mask_2d.shape[1] - 1):
            if not mask_2d[y, x]:
                if check_if_edge_pixel(mask_2d=mask_2d, y=y, x=x):
                    edge_pixel_total += 1

    return edge_pixel_total


@numba_util.jit()
def edge_1d_indexes_from(mask_2d: np.ndarray) -> np.ndarray:
    """
    Returns a 1D array listing all edge pixel indexes in the mask.

    An edge pixel is defined as a pixel on the mask which is unmasked (has a `False`) value and at least 1 of its 8
    direct neighbors is masked (is `True`).

    For example, for the following ``Mask2D``:

    ::
        [[True,  True,  True,  True, True],
         [True, False, False, False, True],
         [True, False, False, False, True],
         [True, False, False, False, True],
         [True,  True,  True,  True, True]]

    The `edge_slim` indexes (given via ``mask_2d.derive_indexes.edge_slim``) is given by:

    ::
         [0, 1, 2, 3, 5, 6, 7, 8]

    Note that index 4 is skipped, which corresponds to the ``False`` value in the centre of the mask, because it
    does not neighbor a ``True`` value in any one of the eight neighboring directions and is therefore not at
    an edge.

    Parameters
    ----------
    mask_2d
        The mask for which the 1D edge pixel indexes are computed.

    Returns
    -------
    np.ndarray
        The 1D indexes of all edge pixels on the mask.
    """

    edge_pixel_total = total_edge_pixels_from(mask_2d)

    edge_pixels = np.zeros(edge_pixel_total)
    edge_index = 0
    regular_index = 0

    for y in range(1, mask_2d.shape[0] - 1):
        for x in range(1, mask_2d.shape[1] - 1):
            if not mask_2d[y, x]:
                if (
                    mask_2d[y + 1, x]
                    or mask_2d[y - 1, x]
                    or mask_2d[y, x + 1]
                    or mask_2d[y, x - 1]
                    or mask_2d[y + 1, x + 1]
                    or mask_2d[y + 1, x - 1]
                    or mask_2d[y - 1, x + 1]
                    or mask_2d[y - 1, x - 1]
                ):
                    edge_pixels[edge_index] = regular_index
                    edge_index += 1

                regular_index += 1

    return edge_pixels


@numba_util.jit()
def check_if_border_pixel(
    mask_2d: np.ndarray, edge_pixel_slim: int, native_to_slim: np.ndarray
) -> bool:
    """
    Checks if an input [y,x] pixel on the input `mask` is a border-pixel.

    A borders pixel is a pixel which:

    1) is not fully surrounding by `False` mask values.
    2) Can reach the edge of the array without hitting a masked pixel in one of four directions (upwards, downwards,
       left, right).

    The borders pixels are thus pixels which are on the exterior edge of the mask. For example, the inner ring of edge
    pixels in an annular mask are edge pixels but not borders pixels.

    Parameters
    ----------
    mask_2d
        The mask for which the input pixel is checked if it is a border pixel.
    edge_pixel_slim
        The edge pixel index in 1D that is checked if it is a border pixel (this 1D index is mapped to 2d via the
        array `native_index_for_slim_index_2d`).
    native_to_slim
        An array describing the native 2D array index that every slimmed array index maps too.

    Returns
    -------
    bool
        If `True` the pixel on the mask is a border pixel, else a `False` is returned because it is not.
    """
    edge_pixel_index = int(edge_pixel_slim)

    y = int(native_to_slim[edge_pixel_index, 0])
    x = int(native_to_slim[edge_pixel_index, 1])

    if (
        np.sum(mask_2d[0:y, x]) == y
        or np.sum(mask_2d[y, x : mask_2d.shape[1]]) == mask_2d.shape[1] - x - 1
        or np.sum(mask_2d[y : mask_2d.shape[0], x]) == mask_2d.shape[0] - y - 1
        or np.sum(mask_2d[y, 0:x]) == x
    ):
        return True
    else:
        return False


@numba_util.jit()
def total_border_pixels_from(mask_2d, edge_pixels, native_to_slim):
    """
    Returns the total number of border-pixels in a mask.

    A borders pixel is a pixel which:

    1) is not fully surrounding by `False` mask values.
    2) Can reach the edge of the array without hitting a masked pixel in one of four directions (upwards, downwards,
       left, right).

    The borders pixels are thus pixels which are on the exterior edge of the mask. For example, the inner ring of edge
    pixels in an annular mask are edge pixels but not borders pixels.

    Parameters
    ----------
    mask_2d
        The mask for which the total number of border pixels is computed.
    edge_pixel_1d
        The edge pixel index in 1D that is checked if it is a border pixel (this 1D index is mapped to 2d via the
        array `native_index_for_slim_index_2d`).
    native_to_slim
        An array describing the 2D array index that every 1D array index maps too.

    Returns
    -------
    int
        The total number of border pixels.
    """

    border_pixel_total = 0

    for i in range(edge_pixels.shape[0]):
        if check_if_border_pixel(mask_2d, edge_pixels[i], native_to_slim):
            border_pixel_total += 1

    return border_pixel_total


@numba_util.jit()
def border_slim_indexes_from(mask_2d: np.ndarray) -> np.ndarray:
    """
    Returns a slim array of shape [total_unmasked_border_pixels] listing all borders pixel indexes in the mask.

    A borders pixel is a pixel which:

    1) is not fully surrounding by `False` mask values.
    2) Can reach the edge of the array without hitting a masked pixel in one of four directions (upwards, downwards,
       left, right).

    The borders pixels are thus pixels which are on the exterior edge of the mask. For example, the inner ring of edge
    pixels in an annular mask are edge pixels but not borders pixels.

    For example, for the following ``Mask2D``:

    ::
        [[True,  True,  True,  True,  True,  True,  True,  True, True],
         [True, False, False, False, False, False, False, False, True],
         [True, False,  True,  True,  True,  True,  True, False, True],
         [True, False,  True, False, False, False,  True, False, True],
         [True, False,  True, False,  True, False,  True, False, True],
         [True, False,  True, False, False, False,  True, False, True],
         [True, False,  True,  True,  True,  True,  True, False, True],
         [True, False, False, False, False, False, False, False, True],
         [True,  True,  True,  True,  True,  True,  True,  True, True]]

    The `border_slim` indexes (given via ``mask_2d.derive_indexes.border_slim``) is given by:

    ::
         [0, 1, 2, 3, 5, 6, 7, 11, 12, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

    The interior 8 ``False`` values are omitted, because although they are edge pixels (neighbor a ``True``) they
    are not on the extreme exterior edge.

    Parameters
    ----------
    mask_2d
        The mask for which the slimmed border pixel indexes are calculated.

    Returns
    -------
    np.ndarray
        The slimmed indexes of all border pixels on the mask.
    """

    edge_pixels = edge_1d_indexes_from(mask_2d=mask_2d)
    native_index_for_slim_index_2d = native_index_for_slim_index_2d_from(
        mask_2d=mask_2d,
    )

    border_pixel_total = total_border_pixels_from(
        mask_2d=mask_2d,
        edge_pixels=edge_pixels,
        native_to_slim=native_index_for_slim_index_2d,
    )

    border_pixels = np.zeros(border_pixel_total)

    border_pixel_index = 0

    for edge_pixel_index in range(edge_pixels.shape[0]):
        if check_if_border_pixel(
            mask_2d=mask_2d,
            edge_pixel_slim=edge_pixels[edge_pixel_index],
            native_to_slim=native_index_for_slim_index_2d,
        ):
            border_pixels[border_pixel_index] = edge_pixels[edge_pixel_index]
            border_pixel_index += 1

    return border_pixels


@numba_util.jit()
def buffed_mask_2d_from(mask_2d: np.ndarray, buffer: int = 1) -> np.ndarray:
    """
    Returns a buffed mask from an input mask, where the buffed mask is the input mask but all `False` entries in the
    mask are buffed by an integer amount in all 8 surrouning pixels.

    Parameters
    ----------
    mask_2d
        The mask whose `False` entries are buffed.
    buffer
        The number of pixels around each `False` entry that pixel are buffed in all 8 directions.

    Returns
    -------
    np.ndarray
        The buffed mask.
    """
    buffed_mask_2d = mask_2d.copy()

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):
            if not mask_2d[y, x]:
                for y0 in range(y - buffer, y + 1 + buffer):
                    for x0 in range(x - buffer, x + 1 + buffer):
                        if (
                            y0 >= 0
                            and x0 >= 0
                            and y0 <= mask_2d.shape[0] - 1
                            and x0 <= mask_2d.shape[1] - 1
                        ):
                            buffed_mask_2d[y0, x0] = False

    return buffed_mask_2d


def rescaled_mask_2d_from(mask_2d: np.ndarray, rescale_factor: float) -> np.ndarray:
    """
    Returns a rescaled mask from an input mask, where the rescaled mask is the input mask but rescaled to a larger or
    smaller size depending on the `rescale_factor`.

    For example, a `rescale_factor` of 0.5 would reduce a 10 x 10 mask to a 5x5 mask, where the `False` entries
    of the 5 x 5 mask corresponding to pixels which had at least one `False` entry in their corresponding location on the
    10 x 10 mask. A rescale factor of 2.0 would increase the 10 x 10 mask in size to a 20 x 20 mask, with `False`
    again wherever the original mask had those entries.

    The edge of the rescaled mask is automatically set to all ` True` values to prevent border issues.

    Parameters
    ----------
    mask_2d
        The mask that is increased or decreased in size via rescaling.
    rescale_factor
        The factor by which the mask is increased in size or decreased in size.

    Returns
    -------
    np.ndarray
        The rescaled mask.
    """

    from skimage.transform import rescale

    warnings.filterwarnings("ignore")

    try:
        rescaled_mask_2d = rescale(
            image=mask_2d,
            scale=rescale_factor,
            mode="edge",
            anti_aliasing=False,
            multichannel=False,
        )
    except TypeError:
        rescaled_mask_2d = rescale(
            image=mask_2d,
            scale=rescale_factor,
            mode="edge",
            anti_aliasing=False,
            #  multichannel=False,
        )

    rescaled_mask_2d[0, :] = 1
    rescaled_mask_2d[rescaled_mask_2d.shape[0] - 1, :] = 1
    rescaled_mask_2d[:, 0] = 1
    rescaled_mask_2d[:, rescaled_mask_2d.shape[1] - 1] = 1
    return np.isclose(rescaled_mask_2d, 1)


@numba_util.jit()
def native_index_for_slim_index_2d_from(
    mask_2d: np.ndarray,
) -> np.ndarray:
    """
    Returns an array of shape [total_unmasked_pixels] that maps every unmasked pixel to its
    corresponding native 2D pixel using its (y,x) pixel indexes.

    For example, for the following ``Mask2D``:

    ::
        [[True,  True,  True, True]
         [True, False, False, True],
         [True, False,  True, True],
         [True,  True,  True, True]]

    This has three unmasked (``False`` values) which have the ``slim`` indexes:

    ::
        [0, 1, 2]

    The array ``native_index_for_slim_index_2d`` is therefore:

    ::
        [[1,1], [1,2], [2,1]]

    Parameters
    ----------
    mask_2d
        A 2D array of bools, where `False` values are unmasked.

    Returns
    -------
    ndarray
        An array that maps pixels from a slimmed array of shape [total_unmasked_pixels] to its native array
        of shape [total_pixels, total_pixels].

    Examples
    --------
    mask_2d = np.array([[True, True, True],
                     [True, False, True]
                     [True, True, True]])

    native_index_for_slim_index_2d = native_index_for_slim_index_2d_from(mask_2d=mask_2d)
    """

    total_pixels = total_pixels_2d_from(mask_2d=mask_2d)
    native_index_for_slim_index_2d = np.zeros(shape=(total_pixels, 2))
    slim_index = 0

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):
            if not mask_2d[y, x]:
                native_index_for_slim_index_2d[slim_index, :] = y, x
                slim_index += 1

    return native_index_for_slim_index_2d
