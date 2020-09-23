import numpy as np
from skimage.transform import rescale

from autoarray import decorator_util
from autoarray import exc
from autoarray.util import grid_util


@decorator_util.jit()
def mask_centres_from(shape, pixel_scales, centre):
    """Determine the (y,x) arc-second central coordinates of a mask from its shape, pixel-scales and centre.

     The coordinate system is defined such that the positive y axis is up and positive x axis is right.

    Parameters
     ----------
    shape : (int, int)
        The (y,x) shape of the 2D array the arc-second centre is computed for.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the 2D array.
    centre : (float, flloat)
        The (y,x) centre of the 2D mask.

    Returns
    --------
    tuple (float, float)
        The (y,x) arc-second central coordinates of the input array.

    Examples
    --------
    centres_arcsec = centres_from_shape_pixel_scales_and_centre(shape=(5,5), pixel_scales=(0.5, 0.5), centre=(0.0, 0.0))
    """
    y_centre_arcsec = (float(shape[0] - 1) / 2) - (centre[0] / pixel_scales[0])
    x_centre_arcsec = (float(shape[1] - 1) / 2) + (centre[1] / pixel_scales[1])

    return (y_centre_arcsec, x_centre_arcsec)


@decorator_util.jit()
def total_pixels_from(mask):
    """Compute the total number of unmasked pixels in a mask.

    Parameters
     ----------
    mask : ndarray
        A 2D array of bools, where *False* values are unmasked and included when counting pixels.

    Returns
    --------
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
def total_sub_pixels_from(mask, sub_size):
    """Compute the total number of sub-pixels in unmasked pixels in a mask.
    
    Parameters
     ----------
    mask : ndarray
        A 2D array of bools, where *False* values are unmasked and included when counting sub pixels.
    sub_size : int
        The size of the sub-grid that each pixel of the 2D mask array is divided into.

    Returns
    --------
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
def total_sparse_pixels_from(mask, unmasked_sparse_grid_pixel_centres):
    """Given the full (i.e. without removing pixels which are outside the mask) pixelization grid's pixel \ 
    center and the mask, compute the total number of pixels which are within the mask and thus used \ \
    by the pixelization grid.

    Parameters
    -----------
    mask : imaging.mask.Mask
        The mask within which pixelization pixels must be inside
    unmasked_sparse_grid_pixel_centres : ndarray
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
def mask_circular_from(shape_2d, pixel_scales, radius, centre=(0.0, 0.0)):
    """Compute a circular mask from the 2D mask array shape and radius of the circle.

    This creates a 2D array where all values within the mask radius are unmasked and therefore *False*.

    Parameters
     ----------
    shape_2d: (int, int)
        The (y,x) shape of the mask in units of pixels.
    pixel_scales: float
        The arc-second to pixel conversion factor of each pixel.
    radius : float
        The radius (in arc seconds) of the circle within which pixels unmasked.
    centre: (float, float)
        The centre of the circle used to mask pixels.

    Returns
    --------
    ndarray
        The 2D mask array whose central pixels are masked as a circle.

    Examples
    --------
    mask = mask_circular_from_shape_pixel_scale_and_radius( \
        shape=(10, 10), pixel_scales=0.1, radius=0.5, centre=(0.0, 0.0))
    """

    mask = np.full(shape_2d, True)

    centres_arcsec = mask_centres_from(
        shape=mask.shape, pixel_scales=pixel_scales, centre=centre
    )

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            y_arcsec = (y - centres_arcsec[0]) * pixel_scales[0]
            x_arcsec = (x - centres_arcsec[1]) * pixel_scales[1]

            r_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

            if r_arcsec <= radius:
                mask[y, x] = False

    return mask


@decorator_util.jit()
def mask_circular_annular_from(
    shape_2d, pixel_scales, inner_radius, outer_radius, centre=(0.0, 0.0)
):
    """Compute an annular mask from an input inner and outer mask radius and shape.

    This creates a 2D array where all values within the inner and outer radii are unmasked and therefore *False*.

    Parameters
     ----------
    shape_2d : (int, int)
        The (y,x) shape of the mask in units of pixels.
    pixel_scales : (float, float)
        The arc-second to pixel conversion factor of each pixel.
    inner_radius : float
        The radius (in arc seconds) of the inner circle outside of which pixels are unmasked.
    outer_radius : float
        The radius (in arc seconds) of the outer circle within which pixels are unmasked.
    centre: (float, float)
        The centre of the annulus used to mask pixels.

    Returns
    --------
    ndarray
        The 2D mask array whose central pixels are masked as a annulus.

    Examples
    --------
    mask = mask_annnular_from_shape_pixel_scale_and_radius( \
        shape=(10, 10), pixel_scales=0.1, inner_radius=0.5, outer_radius=1.5, centre=(0.0, 0.0))
    """

    mask = np.full(shape_2d, True)

    centres_arcsec = mask_centres_from(
        shape=mask.shape, pixel_scales=pixel_scales, centre=centre
    )

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            y_arcsec = (y - centres_arcsec[0]) * pixel_scales[0]
            x_arcsec = (x - centres_arcsec[1]) * pixel_scales[1]

            r_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

            if outer_radius >= r_arcsec >= inner_radius:
                mask[y, x] = False

    return mask


@decorator_util.jit()
def mask_circular_anti_annular_from(
    shape_2d,
    pixel_scales,
    inner_radius,
    outer_radius,
    outer_radius_2_scaled,
    centre=(0.0, 0.0),
):
    """Compute an annular mask from an input inner and outer mask radius and shape."""

    mask = np.full(shape_2d, True)

    centres_arcsec = mask_centres_from(
        shape=mask.shape, pixel_scales=pixel_scales, centre=centre
    )

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            y_arcsec = (y - centres_arcsec[0]) * pixel_scales[0]
            x_arcsec = (x - centres_arcsec[1]) * pixel_scales[1]

            r_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

            if (
                inner_radius >= r_arcsec
                or outer_radius_2_scaled >= r_arcsec >= outer_radius
            ):
                mask[y, x] = False

    return mask


def mask_via_pixel_coordinates_from(shape_2d, pixel_coordinates, buffer=0):
    """Compute an annular mask from an input inner and outer mask radius and shape."""

    mask = np.full(shape=shape_2d, fill_value=True)

    for y, x in pixel_coordinates:

        mask[y, x] = False

    if buffer == 0:
        return mask
    else:
        return buffed_mask_from(mask=mask, buffer=buffer)


@decorator_util.jit()
def elliptical_radius_from(y_arcsec, x_arcsec, phi, axis_ratio):
    r_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

    theta_rotated = np.arctan2(y_arcsec, x_arcsec) + np.radians(phi)

    y_arcsec_elliptical = r_arcsec * np.sin(theta_rotated)
    x_arcsec_elliptical = r_arcsec * np.cos(theta_rotated)

    return np.sqrt(
        x_arcsec_elliptical ** 2.0 + (y_arcsec_elliptical / axis_ratio) ** 2.0
    )


@decorator_util.jit()
def mask_elliptical_from(
    shape_2d, pixel_scales, major_axis_radius, axis_ratio, phi, centre=(0.0, 0.0)
):
    """Compute an elliptical mask from an input major-axis mask radius, axis-ratio, rotational angle phi, shape and \
    centre.

    This creates a 2D array where all values within the ellipse are unmasked and therefore *False*.

    Parameters
     ----------
    shape_2d: (int, int)
        The (y,x) shape of the mask in units of pixels.
    pixel_scales : (float, float)
        The arc-second to pixel conversion factor of each pixel.
    major_axis_radius : float
        The major-axis (in arc seconds) of the ellipse within which pixels are unmasked.
    axis_ratio : float
        The axis-ratio of the ellipse within which pixels are unmasked.
    phi : float
        The rotation angle of the ellipse within which pixels are unmasked, (counter-clockwise from the positive \
         x-axis).
    centre: (float, float)
        The centre of the ellipse used to mask pixels.

    Returns
    --------
    ndarray
        The 2D mask array whose central pixels are masked as an ellipse.

    Examples
    --------
    mask = mask_elliptical_from_shape_pixel_scale_and_radius( \
        shape=(10, 10), pixel_scales=0.1, major_axis_radius=0.5, elliptical_comps=(0.333333, 0.0), centre=(0.0, 0.0))
    """

    mask = np.full(shape_2d, True)

    centres_arcsec = mask_centres_from(
        shape=mask.shape, pixel_scales=pixel_scales, centre=centre
    )

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            y_arcsec = (y - centres_arcsec[0]) * pixel_scales[0]
            x_arcsec = (x - centres_arcsec[1]) * pixel_scales[1]

            r_arcsec_elliptical = elliptical_radius_from(
                y_arcsec, x_arcsec, phi, axis_ratio
            )

            if r_arcsec_elliptical <= major_axis_radius:
                mask[y, x] = False

    return mask


@decorator_util.jit()
def mask_elliptical_annular_from(
    shape_2d,
    pixel_scales,
    inner_major_axis_radius,
    inner_axis_ratio,
    inner_phi,
    outer_major_axis_radius,
    outer_axis_ratio,
    outer_phi,
    centre=(0.0, 0.0),
):
    """Compute an elliptical annular mask from an input major-axis mask radius, axis-ratio, rotational angle phi for \
     both the inner and outer elliptical annuli and a shape and centre for the mask.

    This creates a 2D array where all values within the elliptical annuli are unmasked and therefore *False*.

    Parameters
     ----------
    shape_2d: (int, int)
        The (y,x) shape of the mask in units of pixels.
    pixel_scales : (float, float)
        The arc-second to pixel conversion factor of each pixel.
    inner_major_axis_radius : float
        The major-axis (in arc seconds) of the inner ellipse within which pixels are masked.
    inner_axis_ratio : float
        The axis-ratio of the inner ellipse within which pixels are masked.
    inner_phi : float
        The rotation angle of the inner ellipse within which pixels are masked, (counter-clockwise from the \
        positive x-axis).
    outer_major_axis_radius : float
        The major-axis (in arc seconds) of the outer ellipse within which pixels are unmasked.
    outer_axis_ratio : float
        The axis-ratio of the outer ellipse within which pixels are unmasked.
    outer_phi : float
        The rotation angle of the outer ellipse within which pixels are unmasked, (counter-clockwise from the \
        positive x-axis).
    centre: (float, float)
        The centre of the elliptical annuli used to mask pixels.

    Returns
    --------
    ndarray
        The 2D mask array whose elliptical annuli pixels are masked

    Examples
    --------
    mask = mask_elliptical_annuli_from_shape_pixel_scale_and_radius( \
        shape=(10, 10), pixel_scales=0.1,
         inner_major_axis_radius=0.5, inner_axis_ratio=0.5, inner_phi=45.0,
         outer_major_axis_radius=1.5, outer_axis_ratio=0.8, outer_phi=90.0,
         centre=(0.0, 0.0))
    """

    mask = np.full(shape_2d, True)

    centres_arcsec = mask_centres_from(
        shape=mask.shape, pixel_scales=pixel_scales, centre=centre
    )

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            y_arcsec = (y - centres_arcsec[0]) * pixel_scales[0]
            x_arcsec = (x - centres_arcsec[1]) * pixel_scales[1]

            inner_r_arcsec_elliptical = elliptical_radius_from(
                y_arcsec, x_arcsec, inner_phi, inner_axis_ratio
            )

            outer_r_arcsec_elliptical = elliptical_radius_from(
                y_arcsec, x_arcsec, outer_phi, outer_axis_ratio
            )

            if (
                inner_r_arcsec_elliptical >= inner_major_axis_radius
                and outer_r_arcsec_elliptical <= outer_major_axis_radius
            ):
                mask[y, x] = False

    return mask


@decorator_util.jit()
def blurring_mask_from(mask, kernel_shape_2d):
    """Compute a blurring mask from an input mask and psf shape.

    The blurring mask corresponds to all pixels which are outside of the mask but will have a fraction of their \
    light blur into the masked region due to PSF convolution. The PSF shape is used to determine which pixels these are.
    
    If a pixel is identified which is outside the 2D dimensionos of the input mask, an error is raised and the user \
    should pad the input mask (and associated images).
    
    Parameters
    -----------
    mask : ndarray
        A 2D array of bools, where *False* values are unmasked.
    kernel_shape_2d : (int, int)
        The 2D shape of the PSF which is used to compute the blurring mask.
        
    Returns
    --------
    ndarray
        The 2D blurring mask array whose unmasked values (*False*) correspond to where the mask will have PSF light \
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
    shape_2d, mask_index_for_mask_1d_index
):
    """For a 1D array that was computed by util unmasked values from a 2D array of shape (rows, columns), map its \
    indexes back to the original 2D array to create the origianl 2D mask.

    This uses a 1D array 'one_to_two' where each index gives the 2D pixel indexes of the 1D array's unmasked pixels, \
    for example:

    - If one_to_two[0] = [0,0], the first value of the 1D array maps to the pixel [0,0] of the 2D array.
    - If one_to_two[1] = [0,1], the second value of the 1D array maps to the pixel [0,1] of the 2D array.
    - If one_to_two[4] = [1,1], the fifth value of the 1D array maps to the pixel [1,1] of the 2D array.

    Parameters
     ----------
    shape_2d : (int, int)
        The shape of the 2D array which the pixels are defined on.
    mask_index_for_mask_1d_index : ndarray
        An array describing the 2D array index that every 1D array index maps too.

    Returns
    --------
    ndarray
        A 2D mask array where unmasked values are *False*.

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
def check_if_edge_pixel(mask, y, x):

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
def total_edge_pixels_from(mask):
    """Compute the total number of borders-pixels in a mask."""

    edge_pixel_total = 0

    for y in range(1, mask.shape[0] - 1):
        for x in range(1, mask.shape[1] - 1):
            if not mask[y, x]:
                if check_if_edge_pixel(mask=mask, y=y, x=x):
                    edge_pixel_total += 1

    return edge_pixel_total


@decorator_util.jit()
def edge_1d_indexes_from(mask):
    """Compute a 1D array listing all edge pixel indexes in the mask. An edge pixel is a pixel which is not fully \
    surrounding by False mask values i.e. it is on an edge."""

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
def check_if_border_pixel(mask, edge_pixel_1d, mask_index_for_mask_1d_index):
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
    """Compute the total number of borders-pixels in a mask."""

    border_pixel_total = 0

    for i in range(edge_pixels.shape[0]):

        if check_if_border_pixel(mask, edge_pixels[i], mask_index_for_mask_1d_index):
            border_pixel_total += 1

    return border_pixel_total


@decorator_util.jit()
def border_1d_indexes_from(mask):
    """Compute a 1D array listing all borders pixel indexes in the mask. A borders pixel is a pixel which:

     1) is not fully surrounding by False mask values.
     2) Can reach the edge of the array without hitting a masked pixel in one of four directions (upwards, downwards,
     left, right).

     The borders pixels are thus pixels which are on the exterior edge of the mask. For example, the inner ring of edge \
     pixels in an annular mask are edge pixels but not borders pixels."""

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


def sub_border_pixel_1d_indexes_from(mask, sub_size):
    """Compute a 1D array listing all borders pixel indexes in the mask. A borders pixel is a pixel which:

     1) is not fully surrounding by False mask values.
     2) Can reach the edge of the array without hitting a masked pixel in one of four directions (upwards, downwards,
     left, right).

     The borders pixels are thus pixels which are on the exterior edge of the mask. For example, the inner ring of edge \
     pixels in an annular mask are edge pixels but not borders pixels."""

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
def buffed_mask_from(mask, buffer=1):

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


def rescaled_mask_from(mask, rescale_factor):
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
def mask_1d_index_for_sub_mask_1d_index_via_mask_from(mask, sub_size):
    """"For pixels on a 2D array of shape (rows, colums), compute a 1D array which, for every unmasked pixel on \
    this 2D array, maps the 1D sub-pixel indexes to their 1D pixel indexes.

    For example, for a sub-grid size of 2, the following mappings from sub-pixels to 2D array pixels are:

    - mask_1d_index_for_sub_mask_1d_index[0] = 0 -> The first sub-pixel maps to the first unmasked pixel on the 2D array.
    - mask_1d_index_for_sub_mask_1d_index[3] = 0 -> The fourth sub-pixel maps to the first unmasked pixel on the 2D array.
    - mask_1d_index_for_sub_mask_1d_index[7] = 1 -> The eighth sub-pixel maps to the second unmasked pixel on the 2D array.

    The term 'grid' is used because the grid is defined as the grid of coordinates on the centre of every \
    pixel on the 2D array. Thus, this array maps sub-pixels on a sub-grid to pixels on a grid.


                     [True, False, True]])
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


def sub_mask_1d_indexes_for_mask_1d_index_via_mask_from(mask, sub_size):
    """"For pixels on a 2D array of shape (rows, colums), compute a 1D array which, for every unmasked pixel on \
    this 2D array, maps the 1D sub-pixel indexes to their 1D pixel indexes.

    For example, for a sub-grid size of 2, the following mappings from sub-pixels to 2D array pixels are:

    - mask_1d_index_for_sub_mask_1d_index[0] = 0 -> The first sub-pixel maps to the first unmasked pixel on the 2D array.
    - mask_1d_index_for_sub_mask_1d_index[3] = 0 -> The fourth sub-pixel maps to the first unmasked pixel on the 2D array.
    - mask_1d_index_for_sub_mask_1d_index[7] = 1 -> The eighth sub-pixel maps to the second unmasked pixel on the 2D array.

    The term 'grid' is used because the grid is defined as the grid of coordinates on the centre of every \
    pixel on the 2D array. Thus, this array maps sub-pixels on a sub-grid to pixels on a grid.


                     [True, False, True]])
    mask_1d_index_for_sub_mask_1d_index = mask_1d_index_for_sub_mask_1d_index_from_mask(mask=mask, sub_size=2)
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
def sub_mask_1d_index_for_sub_mask_index_from_sub_mask_from(sub_mask):
    """Create a 2D array which maps every False entry of a 2D mask to its 1D mask array index 2D binned mask. Every \
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
    sub_mask : ndarray
        The 2D mask that the util array is created for.

    Returns
    -------
    ndarray
        The 2D array util 2D mask entries to their 1D masked array indexes.

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
def sub_mask_index_for_sub_mask_1d_index_via_mask_from(mask, sub_size):
    """Compute a 1D array that maps every unmasked sub-pixel to its corresponding 2d pixel using its (y,x) pixel indexes.

    For example, for a sub-grid size of 2, f pixel [2,5] corresponds to the first pixel in the masked 1D array:

    - The first sub-pixel in this pixel on the 1D array is grid_to_pixel[4] = [2,5]
    - The second sub-pixel in this pixel on the 1D array is grid_to_pixel[5] = [2,6]
    - The third sub-pixel in this pixel on the 1D array is grid_to_pixel[5] = [3,5]

    Parameters
    -----------
    mask : ndarray
        A 2D array of bools, where *False* values are unmasked.
    sub_size : int
        The size of the sub-grid in each mask pixel.

    Returns
    --------
    ndarray
        The 2D blurring mask array whose unmasked values (*False*) correspond to where the mask will have PSF light \
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
def mask_neighbors_from(mask):
    """Compute a 1D array that maps every unmasked pixel to the 1D index of a neighboring unmasked pixel.

    Neighbors are chosen to the right of every unmasked pixel, and then down, left and up if there is no unmasked pixel
    in each location.

    Parameters
    -----------
    mask : ndarray
        A 2D array of bools, where *False* values are unmasked.

    Returns
    --------
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
