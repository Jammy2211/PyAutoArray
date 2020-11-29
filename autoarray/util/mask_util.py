import numpy as np
from skimage.transform import rescale
import typing

from autoarray import decorator_util
from autoarray import exc
from autoarray.util import grid_util


@decorator_util.jit()
def mask_centres_from(
    shape: (int, int), pixel_scales: typing.Tuple[float, float], centre: (float, float)
) -> (float, float):
    """
    Returns the (y,x) scaled central coordinates of a mask from its shape, pixel-scales and centre.

    The coordinate system is defined such that the positive y axis is up and positive x axis is right.

    Parameters
    ----------
    shape : (int, int)
        The (y,x) shape of the 2D array the scaled centre is computed for.
    pixel_scales : (float, float)
        The (y,x) scaled units to pixel units conversion factor of the 2D array.
    centre : (float, flloat)
        The (y,x) centre of the 2D mask.

    Returns
    -------
    tuple (float, float)
        The (y,x) scaled central coordinates of the input array.

    Examples
    --------
    centres_scaled = centres_from_shape_pixel_scales_and_centre(shape=(5,5), pixel_scales=(0.5, 0.5), centre=(0.0, 0.0))
    """
    y_centre_scaled = (float(shape[0] - 1) / 2) - (centre[0] / pixel_scales[0])
    x_centre_scaled = (float(shape[1] - 1) / 2) + (centre[1] / pixel_scales[1])

    return (y_centre_scaled, x_centre_scaled)


@decorator_util.jit()
def total_pixels_from(mask: np.ndarray) -> int:
    """
    Returns the total number of unmasked pixels in a mask.

    Parameters
    ----------
    mask : np.ndarray
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

    total_regular_pixels = total_regular_pixels_from_mask(mask=mask)
    """

    total_regular_pixels = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                total_regular_pixels += 1

    return total_regular_pixels


@decorator_util.jit()
def total_sub_pixels_from(mask: np.ndarray, sub_size: int) -> int:
    """
    Returns the total number of sub-pixels in unmasked pixels in a mask.

    Parameters
    ----------
    mask : np.ndarray
        A 2D array of bools, where `False` values are unmasked and included when counting sub pixels.
    sub_size : int
        The size of the sub-grid that each pixel of the 2D mask array is divided into.

    Returns
    -------
    int
        The total number of sub pixels that are unmasked.

    Examples
    --------

    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])

    total_sub_pixels = total_sub_pixels_from_mask(mask=mask, sub_size=2)
    """
    return total_pixels_from(mask) * sub_size ** 2


@decorator_util.jit()
def total_sparse_pixels_from(
    mask: np.ndarray, unmasked_sparse_grid_pixel_centres: np.ndarray
) -> int:
    """Given the full (i.e. without removing pixels which are outside the mask) pixelization grid's pixel
    center and the mask, compute the total number of pixels which are within the mask and thus used
    by the pixelization grid.

    Parameters
    -----------
    mask : np.ndarray
        The mask within which pixelization pixels must be inside
    unmasked_sparse_grid_pixel_centres : np.ndarray
        The centres of the unmasked pixelization grid pixels.
    """

    total_sparse_pixels = 0

    for unmasked_sparse_pixel_index in range(
        unmasked_sparse_grid_pixel_centres.shape[0]
    ):

        y = unmasked_sparse_grid_pixel_centres[unmasked_sparse_pixel_index, 0]
        x = unmasked_sparse_grid_pixel_centres[unmasked_sparse_pixel_index, 1]

        if not mask[y, x]:
            total_sparse_pixels += 1

    return total_sparse_pixels


@decorator_util.jit()
def mask_circular_from(
    shape_2d: (int, int),
    pixel_scales: typing.Tuple[float, float],
    radius: float,
    centre: typing.Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Returns a circular mask from the 2D mask array shape and radius of the circle.

    This creates a 2D array where all values within the mask radius are unmasked and therefore `False`.

    Parameters
    ----------
    shape_2d: (int, int)
        The (y,x) shape of the mask in units of pixels.
    pixel_scales: float
        The scaled units to pixel units conversion factor of each pixel.
    radius : float
        The radius (in scaled units) of the circle within which pixels unmasked.
    centre: (float, float)
        The centre of the circle used to mask pixels.

    Returns
    -------
    ndarray
        The 2D mask array whose central pixels are masked as a circle.

    Examples
    --------
    mask = mask_circular_from_shape_pixel_scale_and_radius(
        shape=(10, 10), pixel_scales=0.1, radius=0.5, centre=(0.0, 0.0))
    """

    mask = np.full(shape_2d, True)

    centres_scaled = mask_centres_from(
        shape=mask.shape, pixel_scales=pixel_scales, centre=centre
    )

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
            x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

            r_scaled = np.sqrt(x_scaled ** 2 + y_scaled ** 2)

            if r_scaled <= radius:
                mask[y, x] = False

    return mask


@decorator_util.jit()
def mask_circular_annular_from(
    shape_2d: (int, int),
    pixel_scales: typing.Tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    centre: typing.Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Returns an circular annular mask from an input inner and outer mask radius and shape.

    This creates a 2D array where all values within the inner and outer radii are unmasked and therefore `False`.

    Parameters
    ----------
    shape_2d : (int, int)
        The (y,x) shape of the mask in units of pixels.
    pixel_scales : (float, float)
        The scaled units to pixel units conversion factor of each pixel.
    inner_radius : float
        The radius (in scaled units) of the inner circle outside of which pixels are unmasked.
    outer_radius : float
        The radius (in scaled units) of the outer circle within which pixels are unmasked.
    centre: (float, float)
        The centre of the annulus used to mask pixels.

    Returns
    -------
    ndarray
        The 2D mask array whose central pixels are masked as a annulus.

    Examples
    --------
    mask = mask_annnular_from_shape_pixel_scale_and_radius(
        shape=(10, 10), pixel_scales=0.1, inner_radius=0.5, outer_radius=1.5, centre=(0.0, 0.0))
    """

    mask = np.full(shape_2d, True)

    centres_scaled = mask_centres_from(
        shape=mask.shape, pixel_scales=pixel_scales, centre=centre
    )

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
            x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

            r_scaled = np.sqrt(x_scaled ** 2 + y_scaled ** 2)

            if outer_radius >= r_scaled >= inner_radius:
                mask[y, x] = False

    return mask


@decorator_util.jit()
def mask_circular_anti_annular_from(
    shape_2d: (int, int),
    pixel_scales: typing.Tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    outer_radius_2_scaled: float,
    centre: typing.Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Returns an anti-annular mask from an input inner and outer mask radius and shape. The anti-annular is analogous to
    the annular mask but inverted, whereby its unmasked values are those inside the annulus.

    This creates a 2D array where all values outside the inner and outer radii are unmasked and therefore `False`.

    Parameters
    ----------
    shape_2d : (int, int)
        The (y,x) shape of the mask in units of pixels.
    pixel_scales : (float, float)
        The scaled units to pixel units conversion factor of each pixel.
    inner_radius : float
        The inner radius in scaled units of the annulus within which pixels are `False` and unmasked.
    outer_radius : float
        The first outer radius in scaled units of the annulus within which pixels are `True` and masked.
    outer_radius_2 : float
        The second outer radius in scaled units of the annulus within which pixels are `False` and unmasked and
        outside of which all entries are `True` and masked.
    centre: (float, float)
        The centre of the annulus used to mask pixels.

    Returns
    -------
    ndarray
        The 2D mask array whose central pixels are masked as a annulus.

    Examples
    --------
    mask = mask_annnular_from_shape_pixel_scale_and_radius(
        shape=(10, 10), pixel_scales=0.1, inner_radius=0.5, outer_radius=1.5, centre=(0.0, 0.0))

    """

    mask = np.full(shape_2d, True)

    centres_scaled = mask_centres_from(
        shape=mask.shape, pixel_scales=pixel_scales, centre=centre
    )

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
            x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

            r_scaled = np.sqrt(x_scaled ** 2 + y_scaled ** 2)

            if (
                inner_radius >= r_scaled
                or outer_radius_2_scaled >= r_scaled >= outer_radius
            ):
                mask[y, x] = False

    return mask


def mask_via_pixel_coordinates_from(
    shape_2d: (int, int), pixel_coordinates: [list], buffer: int = 0
) -> np.ndarray:
    """
    Returns a mask where all unmasked `False` entries are defined from an input list of list of pixel coordinates.

    These may be buffed via an input ``buffer``, whereby all entries in all 8 neighboring directions by this
    amount.

    Parameters
    ----------
    shape_2d (int, int)
        The (y,x) shape of the mask in units of pixels.
    pixel_coordinates : [[int, int]]
        The input lists of 2D pixel coordinates where `False` entries are created.
    buffer : int
        All input ``pixel_coordinates`` are buffed with `False` entries in all 8 neighboring directions by this
        amount.
    """

    mask = np.full(shape=shape_2d, fill_value=True)

    for y, x in pixel_coordinates:

        mask[y, x] = False

    if buffer == 0:
        return mask
    else:
        return buffed_mask_from(mask=mask, buffer=buffer)


@decorator_util.jit()
def elliptical_radius_from(
    y_scaled: float, x_scaled: float, phi: float, axis_ratio: float
) -> float:
    """
    Returns the elliptical radius of an ellipse from its (y,x) scaled centre, rotation angle `phi` defined in degrees
    counter-clockwise from the positive x-axis and its axis-ratio.

    This is used by the function `mask_elliptical_from` to determine the radius of every (y,x) coordinate in elliptical
    units when deciding if it is within the mask.

    Parameters
    ----------
    y_scaled : float
        The scaled y coordinate in Cartesian coordinates which is converted to elliptical coordinates.
    x_scaled : float
        The scaled x coordinate in Cartesian coordinates which is converted to elliptical coordinates.
    phi : float
        The rotation angle in degrees counter-clockwise from the positive x-axis
    axis_ratio : float
        The axis-ratio of the ellipse (minor axis / major axis).

    Returns
    -------
    float
        The radius of the input scaled (y,x) coordinate on the ellipse's ellipitcal coordinate system.
    """
    r_scaled = np.sqrt(x_scaled ** 2 + y_scaled ** 2)

    theta_rotated = np.arctan2(y_scaled, x_scaled) + np.radians(phi)

    y_scaled_elliptical = r_scaled * np.sin(theta_rotated)
    x_scaled_elliptical = r_scaled * np.cos(theta_rotated)

    return np.sqrt(
        x_scaled_elliptical ** 2.0 + (y_scaled_elliptical / axis_ratio) ** 2.0
    )


@decorator_util.jit()
def mask_elliptical_from(
    shape_2d: (int, int),
    pixel_scales: typing.Tuple[float, float],
    major_axis_radius: float,
    axis_ratio: float,
    phi: float,
    centre: typing.Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Returns an elliptical mask from an input major-axis mask radius, axis-ratio, rotational angle phi, shape and
    centre.

    This creates a 2D array where all values within the ellipse are unmasked and therefore `False`.

    Parameters
    ----------
    shape_2d: (int, int)
        The (y,x) shape of the mask in units of pixels.
    pixel_scales : (float, float)
        The scaled units to pixel units conversion factor of each pixel.
    major_axis_radius : float
        The major-axis (in scaled units) of the ellipse within which pixels are unmasked.
    axis_ratio : float
        The axis-ratio of the ellipse within which pixels are unmasked.
    phi : float
        The rotation angle of the ellipse within which pixels are unmasked, (counter-clockwise from the positive
         x-axis).
    centre: (float, float)
        The centre of the ellipse used to mask pixels.

    Returns
    -------
    ndarray
        The 2D mask array whose central pixels are masked as an ellipse.

    Examples
    --------
    mask = mask_elliptical_from_shape_pixel_scale_and_radius(
        shape=(10, 10), pixel_scales=0.1, major_axis_radius=0.5, elliptical_comps=(0.333333, 0.0), centre=(0.0, 0.0))
    """

    mask = np.full(shape_2d, True)

    centres_scaled = mask_centres_from(
        shape=mask.shape, pixel_scales=pixel_scales, centre=centre
    )

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
            x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

            r_scaled_elliptical = elliptical_radius_from(
                y_scaled, x_scaled, phi, axis_ratio
            )

            if r_scaled_elliptical <= major_axis_radius:
                mask[y, x] = False

    return mask


@decorator_util.jit()
def mask_elliptical_annular_from(
    shape_2d: (int, int),
    pixel_scales: typing.Tuple[float, float],
    inner_major_axis_radius: float,
    inner_axis_ratio: float,
    inner_phi: float,
    outer_major_axis_radius: float,
    outer_axis_ratio: float,
    outer_phi: float,
    centre: typing.Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Returns an elliptical annular mask from an input major-axis mask radius, axis-ratio, rotational angle phi for
    both the inner and outer elliptical annuli and a shape and centre for the mask.

    This creates a 2D array where all values within the elliptical annuli are unmasked and therefore `False`.

    Parameters
    ----------
    shape_2d: (int, int)
        The (y,x) shape of the mask in units of pixels.
    pixel_scales : (float, float)
        The scaled units to pixel units conversion factor of each pixel.
    inner_major_axis_radius : float
        The major-axis (in scaled units) of the inner ellipse within which pixels are masked.
    inner_axis_ratio : float
        The axis-ratio of the inner ellipse within which pixels are masked.
    inner_phi : float
        The rotation angle of the inner ellipse within which pixels are masked, (counter-clockwise from the
        positive x-axis).
    outer_major_axis_radius : float
        The major-axis (in scaled units) of the outer ellipse within which pixels are unmasked.
    outer_axis_ratio : float
        The axis-ratio of the outer ellipse within which pixels are unmasked.
    outer_phi : float
        The rotation angle of the outer ellipse within which pixels are unmasked, (counter-clockwise from the
        positive x-axis).
    centre: (float, float)
        The centre of the elliptical annuli used to mask pixels.

    Returns
    -------
    ndarray
        The 2D mask array whose elliptical annuli pixels are masked.

    Examples
    --------
    mask = mask_elliptical_annuli_from_shape_pixel_scale_and_radius(
        shape=(10, 10), pixel_scales=0.1,
         inner_major_axis_radius=0.5, inner_axis_ratio=0.5, inner_phi=45.0,
         outer_major_axis_radius=1.5, outer_axis_ratio=0.8, outer_phi=90.0,
         centre=(0.0, 0.0))
    """

    mask = np.full(shape_2d, True)

    centres_scaled = mask_centres_from(
        shape=mask.shape, pixel_scales=pixel_scales, centre=centre
    )

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

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
                mask[y, x] = False

    return mask


@decorator_util.jit()
def blurring_mask_from(mask: np.ndarray, kernel_shape_2d: (int, int)) -> np.ndarray:
    """
    Returns a blurring mask from an input mask and psf shape.

    The blurring mask corresponds to all pixels which are outside of the mask but will have a fraction of their
    light blur into the masked region due to PSF convolution. The PSF shape is used to determine which pixels these are.

    If a pixel is identified which is outside the 2D dimensions of the input mask, an error is raised and the user
    should pad the input mask (and associated images).

    Parameters
    -----------
    mask : np.ndarray
        A 2D array of bools, where `False` values are unmasked.
    kernel_shape_2d : (int, int)
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

    blurring_mask = blurring_mask_from_mask_and_psf_shape(mask=mask, psf_shape_2d=(3,3))

    """

    blurring_mask = np.full(mask.shape, True)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                for y1 in range(
                    (-kernel_shape_2d[0] + 1) // 2, (kernel_shape_2d[0] + 1) // 2
                ):
                    for x1 in range(
                        (-kernel_shape_2d[1] + 1) // 2, (kernel_shape_2d[1] + 1) // 2
                    ):
                        if (
                            0 <= x + x1 <= mask.shape[1] - 1
                            and 0 <= y + y1 <= mask.shape[0] - 1
                        ):
                            if mask[y + y1, x + x1]:
                                blurring_mask[y + y1, x + x1] = False
                        else:
                            raise exc.MaskException(
                                "setup_blurring_mask extends beyond the sub_size "
                                "of the mask - pad the datas array before masking"
                            )

    return blurring_mask


@decorator_util.jit()
def mask_via_shape_2d_and_mask_index_for_mask_1d_index_from(
    shape_2d: (int, int), mask_index_for_mask_1d_index: np.ndarray
) -> np.ndarray:
    """
    For a 1D array that was computed by util unmasked values from a 2D array of shape (total_y_pixels, total_x_pixels),
    map its indexes back to the original 2D array to create the origianl 2D mask.

    This uses a 1D array 'one_to_two' where each index gives the 2D pixel indexes of the 1D array's unmasked pixels,
    for example:

    - If one_to_two[0] = [0,0], the first value of the 1D array maps to the pixel [0,0] of the 2D array.
    - If one_to_two[1] = [0,1], the second value of the 1D array maps to the pixel [0,1] of the 2D array.
    - If one_to_two[4] = [1,1], the fifth value of the 1D array maps to the pixel [1,1] of the 2D array.

    Parameters
    ----------
    shape_2d : (int, int)
        The shape of the 2D array which the pixels are defined on.
    mask_index_for_mask_1d_index : np.ndarray
        An array describing the 2D array index that every 1D array index maps too.

    Returns
    -------
    ndarray
        A 2D mask array where unmasked values are `False`.

    Examples
    --------
    one_to_two = np.array([[0,1], [1,0], [1,1], [1,2], [2,1]])

    mask = mask_from_shape_and_one_to_two(shape=(3,3), one_to_two=one_to_two)
    """

    mask = np.ones(shape_2d)

    for index in range(len(mask_index_for_mask_1d_index)):
        mask[
            mask_index_for_mask_1d_index[index, 0],
            mask_index_for_mask_1d_index[index, 1],
        ] = False

    return mask


@decorator_util.jit()
def check_if_edge_pixel(mask: np.ndarray, y: int, x: int) -> bool:
    """
    Checks if an input [y,x] pixel on the input `mask` is an edge-pixel.

    An edge pixel is defined as a pixel on the mask which is unmasked (has a `False`) value and at least 1 of its 8
    direct neighbors is masked (is `True`).

    Parameters
    ----------
    mask : np.ndarray
        The mask for which the input pixel is checked if it is an edge pixel.
    y : int
        The y pixel coordinate on the mask that is checked for if it is an edge pixel.
    x : int
        The x pixel coordinate on the mask that is checked for if it is an edge pixel.

    Returns
    -------
    bool
        If `True` the pixel on the mask is an edge pixel, else a `False` is returned because it is not.
    """

    if (
        mask[y + 1, x]
        or mask[y - 1, x]
        or mask[y, x + 1]
        or mask[y, x - 1]
        or mask[y + 1, x + 1]
        or mask[y + 1, x - 1]
        or mask[y - 1, x + 1]
        or mask[y - 1, x - 1]
    ):
        return True
    else:
        return False


@decorator_util.jit()
def total_edge_pixels_from(mask: np.ndarray) -> int:
    """
    Returns the total number of edge-pixels in a mask.

    An edge pixel is defined as a pixel on the mask which is unmasked (has a `False`) value and at least 1 of its 8
    direct neighbors is masked (is `True`).

    Parameters
    ----------
    mask : np.ndarray
        The mask for which the total number of edge pixels is computed.

    Returns
    -------
    int
        The total number of edge pixels.
    """

    edge_pixel_total = 0

    for y in range(1, mask.shape[0] - 1):
        for x in range(1, mask.shape[1] - 1):
            if not mask[y, x]:
                if check_if_edge_pixel(mask=mask, y=y, x=x):
                    edge_pixel_total += 1

    return edge_pixel_total


@decorator_util.jit()
def edge_1d_indexes_from(mask: np.ndarray) -> np.ndarray:
    """
    Returns a 1D array listing all edge pixel indexes in the mask.

    An edge pixel is defined as a pixel on the mask which is unmasked (has a `False`) value and at least 1 of its 8
    direct neighbors is masked (is `True`).

    Parameters
    ----------
    mask : np.ndarray
        The mask for which the 1D edge pixel indexes are computed.

    Returns
    -------
    np.ndarray
        The 1D indexes of all edge pixels on the mask.
    """

    edge_pixel_total = total_edge_pixels_from(mask)

    edge_pixels = np.zeros(edge_pixel_total)
    edge_index = 0
    regular_index = 0

    for y in range(1, mask.shape[0] - 1):
        for x in range(1, mask.shape[1] - 1):
            if not mask[y, x]:
                if (
                    mask[y + 1, x]
                    or mask[y - 1, x]
                    or mask[y, x + 1]
                    or mask[y, x - 1]
                    or mask[y + 1, x + 1]
                    or mask[y + 1, x - 1]
                    or mask[y - 1, x + 1]
                    or mask[y - 1, x - 1]
                ):
                    edge_pixels[edge_index] = regular_index
                    edge_index += 1

                regular_index += 1

    return edge_pixels


@decorator_util.jit()
def check_if_border_pixel(
    mask: np.ndarray, edge_pixel_1d: int, mask_index_for_mask_1d_index: np.ndarray
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
    mask : np.ndarray
        The mask for which the input pixel is checked if it is a border pixel.
    edge_pixel_1d : int
        The edge pixel index in 1D that is checked if it is a border pixel (this 1D index is mapped to 2d via the
        array `mask_index_for_mask_1d_index`).
    mask_index_for_mask_1d_index : np.ndarray
        An array describing the 2D array index that every 1D array index maps too.

    Returns
    -------
    bool
        If `True` the pixel on the mask is a border pixel, else a `False` is returned because it is not.
    """
    edge_pixel_index = int(edge_pixel_1d)

    y = int(mask_index_for_mask_1d_index[edge_pixel_index, 0])
    x = int(mask_index_for_mask_1d_index[edge_pixel_index, 1])

    if (
        np.sum(mask[0:y, x]) == y
        or np.sum(mask[y, x : mask.shape[1]]) == mask.shape[1] - x - 1
        or np.sum(mask[y : mask.shape[0], x]) == mask.shape[0] - y - 1
        or np.sum(mask[y, 0:x]) == x
    ):
        return True
    else:
        return False


@decorator_util.jit()
def total_border_pixels_from(mask, edge_pixels, mask_index_for_mask_1d_index):
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
    mask : np.ndarray
        The mask for which the total number of border pixels is computed.
    edge_pixel_1d : int
        The edge pixel index in 1D that is checked if it is a border pixel (this 1D index is mapped to 2d via the
        array `mask_index_for_mask_1d_index`).
    mask_index_for_mask_1d_index : np.ndarray
        An array describing the 2D array index that every 1D array index maps too.

    Returns
    -------
    int
        The total number of border pixels.
    """

    border_pixel_total = 0

    for i in range(edge_pixels.shape[0]):

        if check_if_border_pixel(mask, edge_pixels[i], mask_index_for_mask_1d_index):
            border_pixel_total += 1

    return border_pixel_total


@decorator_util.jit()
def border_1d_indexes_from(mask: np.ndarray) -> np.ndarray:
    """
    Returns a 1D array listing all borders pixel indexes in the mask.

    A borders pixel is a pixel which:

    1) is not fully surrounding by `False` mask values.
    2) Can reach the edge of the array without hitting a masked pixel in one of four directions (upwards, downwards,
       left, right).

    The borders pixels are thus pixels which are on the exterior edge of the mask. For example, the inner ring of edge
    pixels in an annular mask are edge pixels but not borders pixels.

    Parameters
    ----------
    mask : np.ndarray
        The mask for which the 1D border pixel indexes are calculated.

    Returns
    -------
    np.ndarray
        The 1D indexes of all border pixels on the mask.
    """

    edge_pixels = edge_1d_indexes_from(mask=mask)
    mask_index_for_mask_1d_index = sub_mask_index_for_sub_mask_1d_index_via_mask_from(
        mask=mask, sub_size=1
    )

    border_pixel_total = total_border_pixels_from(
        mask=mask,
        edge_pixels=edge_pixels,
        mask_index_for_mask_1d_index=mask_index_for_mask_1d_index,
    )

    border_pixels = np.zeros(border_pixel_total)

    border_pixel_index = 0

    for edge_pixel_index in range(edge_pixels.shape[0]):

        if check_if_border_pixel(
            mask=mask,
            edge_pixel_1d=edge_pixels[edge_pixel_index],
            mask_index_for_mask_1d_index=mask_index_for_mask_1d_index,
        ):

            border_pixels[border_pixel_index] = edge_pixels[edge_pixel_index]
            border_pixel_index += 1

    return border_pixels


def sub_border_pixel_1d_indexes_from(mask: np.ndarray, sub_size: int) -> np.ndarray:
    """
    Returns a 1D array listing all sub-borders pixel indexes in the mask.

    A borders pixel is a pixel which:

    1) is not fully surrounding by `False` mask values.
    2) Can reach the edge of the array without hitting a masked pixel in one of four directions (upwards, downwards,
       left, right).

    The borders pixels are thus pixels which are on the exterior edge of the mask. For example, the inner ring of
    edge pixels in an annular mask are edge pixels but not borders pixels.

    A sub-border pixel is, for a border-pixel, the pixel within that border pixel which is futherest from the origin
    of the mask.

    Parameters
    ----------
    mask : np.ndarray
        The mask for which the 1D border pixel indexes are calculated.
    sub_size : int
        The size of the sub-grid in each mask pixel.

    Returns
    -------
    np.ndarray
        The 1D indexes of all border sub-pixels on the mask.
    """

    border_pixels = border_1d_indexes_from(mask=mask)

    sub_border_pixels = np.zeros(shape=border_pixels.shape[0])

    mask_1d_index_to_sub_mask_indexes = sub_mask_1d_indexes_for_mask_1d_index_via_mask_from(
        mask=mask, sub_size=sub_size
    )

    masked_sub_grid_1d = grid_util.grid_1d_via_mask_from(
        mask=mask, pixel_scales=(1.0, 1.0), sub_size=sub_size, origin=(0.0, 0.0)
    )
    mask_centre = grid_util.grid_centre_from(grid_1d=masked_sub_grid_1d)

    for (border_1d_index, border_pixel) in enumerate(border_pixels):
        sub_border_pixels_of_border_pixel = mask_1d_index_to_sub_mask_indexes[
            int(border_pixel)
        ]

        sub_border_pixels[border_1d_index] = grid_util.furthest_grid_1d_index_from(
            grid_1d=masked_sub_grid_1d,
            grid_1d_indexes=sub_border_pixels_of_border_pixel,
            coordinate=mask_centre,
        )

    return sub_border_pixels


@decorator_util.jit()
def buffed_mask_from(mask: np.ndarray, buffer: int = 1) -> np.ndarray:
    """
    Returns a buffed mask from an input mask, where the buffed mask is the input mask but all `False` entries in the
    mask are buffed by an integer amount in all 8 surrouning pixels.

    Parameters
    ----------
    mask : np.ndarray
        The mask whose `False` entries are buffed.
    buffer : int
        The number of pixels around each `False` entry that pixel are buffed in all 8 directions.

    Returns
    -------
    np.ndarray
        The buffed mask.
    """
    buffed_mask = mask.copy()

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                for y0 in range(y - buffer, y + 1 + buffer):
                    for x0 in range(x - buffer, x + 1 + buffer):

                        if (
                            y0 >= 0
                            and x0 >= 0
                            and y0 <= mask.shape[0] - 1
                            and x0 <= mask.shape[1] - 1
                        ):
                            buffed_mask[y0, x0] = False

    return buffed_mask


def rescaled_mask_from(mask: np.ndarray, rescale_factor: float) -> np.ndarray:
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
    mask : np.ndarray
        The mask that is increased or decreased in size via rescaling.
    rescale_factor : float
        The factor by which the mask is increased in size or decreased in size.

    Returns
    -------
    np.ndarray
        The rescaled mask.
    """
    rescaled_mask = rescale(
        image=mask,
        scale=rescale_factor,
        mode="edge",
        anti_aliasing=False,
        multichannel=False,
    )

    rescaled_mask[0, :] = 1
    rescaled_mask[rescaled_mask.shape[0] - 1, :] = 1
    rescaled_mask[:, 0] = 1
    rescaled_mask[:, rescaled_mask.shape[1] - 1] = 1
    return np.isclose(rescaled_mask, 1)


@decorator_util.jit()
def mask_1d_index_for_sub_mask_1d_index_via_mask_from(
    mask: np.ndarray, sub_size: int
) -> np.ndarray:
    """ "
    For pixels on a 2D array of shape (total_y_pixels, total_x_pixels), compute a 1D array which, for every unmasked
    pixel on this 2D array, maps the 1D sub-pixel indexes to their 1D pixel indexes.

    For example, for a sub-grid size of 2, the following mappings from sub-pixels to 2D array pixels are:

    - mask_1d_index_for_sub_mask_1d_index[0] = 0 -> The first sub-pixel maps to the first unmasked pixel on the 2D array.
    - mask_1d_index_for_sub_mask_1d_index[3] = 0 -> The fourth sub-pixel maps to the first unmasked pixel on the 2D array.
    - mask_1d_index_for_sub_mask_1d_index[7] = 1 -> The eighth sub-pixel maps to the second unmasked pixel on the 2D array.

    Parameters
    ----------
    mask : np.ndarray
        The mask whose indexes are mapped.
    sub_size : int
        The sub-size of the grid on the mask, so that the sub-mask indexes can be computed correctly.

    Returns
    -------
    np.ndarray
        The 1D ndarray mapping every unmasked pixel on the 2D mask array to its 1D index on the sub-mask array.

    Examples
    --------
    mask = np.array([[True, False, True]])
    mask_1d_index_for_sub_mask_1d_index = mask_1d_index_for_sub_mask_1d_index_from_mask(mask=mask, sub_size=2)
    """

    total_sub_pixels = total_sub_pixels_from(mask=mask, sub_size=sub_size)

    mask_1d_index_for_sub_mask_1d_index = np.zeros(shape=total_sub_pixels)
    mask_1d_index = 0
    sub_mask_1d_index = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                for y1 in range(sub_size):
                    for x1 in range(sub_size):
                        mask_1d_index_for_sub_mask_1d_index[
                            sub_mask_1d_index
                        ] = mask_1d_index
                        sub_mask_1d_index += 1

                mask_1d_index += 1

    return mask_1d_index_for_sub_mask_1d_index


def sub_mask_1d_indexes_for_mask_1d_index_via_mask_from(
    mask: np.ndarray, sub_size: int
) -> [list]:
    """ "
    For pixels on a 2D array of shape (total_y_pixels, total_x_pixels), compute a list oof lists which, for every
    unmasked pixel gives the 1D pixel indexes of its corresponding sub-pixels.

    For example, for a sub-grid size of 2, the following mappings from sub-pixels to 2D array pixels are:

    - sub_mask_1d_index_for_mask_1d_index[0] = [0, 1, 2, 3] -> The first pixel maps to the first 4 subpixels in 1D.
    - sub_mask_1d_index_for_mask_1d_index[1] = [4, 5, 6, 7] -> The seond pixel maps to the next 4 subpixels in 1D.

    Parameters
    ----------
    mask : np.ndarray
        The mask whose indexes are mapped.
    sub_size : int
        The sub-size of the grid on the mask, so that the sub-mask indexes can be computed correctly.

    Returns
    -------
    [list]
        The lists of the 1D sub-pixel indexes in every unmasked pixel in the mask.
    The term 'grid' is used because the grid is defined as the grid of coordinates on the centre of every
    pixel on the 2D array. Thus, this array maps sub-pixels on a sub-grid to pixels on a grid.

    Examples
    --------
    mask = ([[True, False, True]])
    sub_mask_1d_indexes_for_mask_1d_index = sub_mask_1d_indexes_for_mask_1d_index_from(mask=mask, sub_size=2)
    """

    total_pixels = total_pixels_from(mask=mask)

    sub_mask_1d_indexes_for_mask_1d_index = [[] for _ in range(total_pixels)]

    mask_1d_index_for_sub_mask_1d_index = mask_1d_index_for_sub_mask_1d_index_via_mask_from(
        mask=mask, sub_size=sub_size
    ).astype(
        "int"
    )

    for sub_mask_1d_index, mask_1d_index in enumerate(
        mask_1d_index_for_sub_mask_1d_index
    ):
        sub_mask_1d_indexes_for_mask_1d_index[mask_1d_index].append(sub_mask_1d_index)

    return sub_mask_1d_indexes_for_mask_1d_index


@decorator_util.jit()
def sub_mask_1d_index_for_sub_mask_index_from_sub_mask_from(sub_mask: np.ndarray):
    """
    Returns a 2D array which maps every `False` entry of a 2D mask to its 1D mask array index 2D sub mask. Every
    True entry is given a value -1.

    This is used as a convenience tool for creating structures util between different grids and structures.

    For example, if we had a 3x4:

    [[False, True, False, False],
     [False, True, False, False],
     [False, False, False, True]]]

    The mask_to_mask_1d array would be:

    [[0, -1, 2, 3],
     [4, -1, 5, 6],
     [7, 8, 9, -1]]

    Parameters
    ----------
    sub_mask : np.ndarray
        The 2D mask that the util array is created for.

    Returns
    -------
    ndarray
        The 2D array mapping 2D mask entries to their 1D masked array indexes.

    Examples
    --------
    mask = np.full(fill_value=False, shape=(9,9))
    sub_two_to_one = mask_to_mask_1d_index_from_mask(mask=mask)
    """

    sub_mask_1d_index_for_sub_mask_index = np.full(fill_value=-1, shape=sub_mask.shape)

    sub_mask_1d_index = 0

    for sub_mask_y in range(sub_mask.shape[0]):
        for sub_mask_x in range(sub_mask.shape[1]):
            if sub_mask[sub_mask_y, sub_mask_x] == False:
                sub_mask_1d_index_for_sub_mask_index[
                    sub_mask_y, sub_mask_x
                ] = sub_mask_1d_index
                sub_mask_1d_index += 1

    return sub_mask_1d_index_for_sub_mask_index


@decorator_util.jit()
def sub_mask_index_for_sub_mask_1d_index_via_mask_from(
    mask: np.ndarray, sub_size: int
) -> np.ndarray:
    """
    Returns a 1D array that maps every unmasked sub-pixel to its corresponding 2d pixel using its (y,x) pixel indexes.

    For example, for a sub-grid size of 2, f pixel [2,5] corresponds to the first pixel in the masked 1D array:

    - The first sub-pixel in this pixel on the 1D array is grid_to_pixel[4] = [2,5]
    - The second sub-pixel in this pixel on the 1D array is grid_to_pixel[5] = [2,6]
    - The third sub-pixel in this pixel on the 1D array is grid_to_pixel[5] = [3,5]

    Parameters
    -----------
    mask : np.ndarray
        A 2D array of bools, where `False` values are unmasked.
    sub_size : int
        The size of the sub-grid in each mask pixel.

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

    blurring_mask = blurring_mask_from_mask_and_psf_shape(mask=mask, psf_shape_2d=(3,3))

    """

    total_sub_pixels = total_sub_pixels_from(mask=mask, sub_size=sub_size)
    sub_mask_index_for_sub_mask_1d_index = np.zeros(shape=(total_sub_pixels, 2))
    sub_mask_1d_index = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                for y1 in range(sub_size):
                    for x1 in range(sub_size):
                        sub_mask_index_for_sub_mask_1d_index[sub_mask_1d_index, :] = (
                            (y * sub_size) + y1,
                            (x * sub_size) + x1,
                        )
                        sub_mask_1d_index += 1

    return sub_mask_index_for_sub_mask_1d_index


@decorator_util.jit()
def mask_neighbors_from(mask: np.ndarray) -> np.ndarray:
    """
    Returns a 1D array that maps every unmasked pixel to the 1D index of a neighboring unmasked pixel.

    Neighbors are chosen to the right of every unmasked pixel, and then down, left and up if there is no unmasked pixel
    in each location.

    Parameters
    -----------
    mask : np.ndarray
        A 2D array of bools, where `False` values are unmasked.

    Returns
    -------
    ndarray
        A 1D array mapping every unmasked pixel to the 1D index of a neighboring unmasked pixel.

    Examples
    --------
    mask = np.array(
        [
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ]
    )

    mask_neighbors = util.mask.mask_neighbors_from_mask(mask=mask)

    """

    total_pixels = total_pixels_from(mask=mask)

    mask_neighbors = -1 * np.ones(shape=total_pixels)

    sub_mask_1d_index_for_sub_mask_index = sub_mask_1d_index_for_sub_mask_index_from_sub_mask_from(
        sub_mask=mask
    )

    mask_index = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:

                flag = True

                if x + 1 < mask.shape[1]:
                    if not mask[y, x + 1]:
                        mask_neighbors[
                            mask_index
                        ] = sub_mask_1d_index_for_sub_mask_index[y, x + 1]
                        flag = False

                if y + 1 < mask.shape[0] and flag:
                    if not mask[y + 1, x]:
                        mask_neighbors[
                            mask_index
                        ] = sub_mask_1d_index_for_sub_mask_index[y + 1, x]
                        flag = False

                if x - 1 >= 0 and flag:
                    if not mask[y, x - 1]:
                        mask_neighbors[
                            mask_index
                        ] = sub_mask_1d_index_for_sub_mask_index[y, x - 1]
                        flag = False

                if y - 1 >= 0 and flag:
                    if not mask[y - 1, x]:
                        mask_neighbors[
                            mask_index
                        ] = sub_mask_1d_index_for_sub_mask_index[y - 1, x]

                mask_index += 1

    return mask_neighbors
