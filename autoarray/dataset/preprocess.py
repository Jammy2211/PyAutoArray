import numpy as np
from scipy.stats import norm

from autoarray.structures import arrays
from autoarray import exc


def array_with_new_shape(array, new_shape):
    """Resize an input array around its centre to a new input shape.

    If a new_shape dimension is smaller than the array's current dimension, the data at the edges is trimmed and
    removedd. If it is larger, the data is padded with zeros.

    If the array has even sized dimensions, the central pixel around which data is trimmed / padded is chosen as
    the top-left pixel of the central quadrant of pixels.

    Parameters
    -----------
    array : aa.Array
        The array which is trimmed / padded to a new 2D shape.
    new_shape : (int, int)
        The new 2D shape of the array.
    """

    return array.resized_from_new_shape(new_shape=new_shape)


def array_eps_to_counts(array_eps, exposure_time_map):
    """
    Convert an array in units of electrons per second to counts, using an exposure time map containing the exposure
    time at every point in the array.

    The conversion from electrons per second to counts is:

    [Counts] = [EPS] * [Exposure_time]

    Parameters
    ----------
    array_eps : aa.Array
        The array which is converted from electrons per seconds to counts.
    exposure_time_map : aa.Array
        The exposure time at every data-point of the array.
    """
    return np.multiply(array_eps, exposure_time_map)


def array_counts_to_eps(array_counts, exposure_time_map):
    """
    Convert an array in units of electrons per second to counts, using an exposure time map containing the exposure
    time at every point in the array.

    The conversion from counts to electrons per second is:

    [EPS] = [Counts] / [Exposure_time]

    Parameters
    ----------
    array_counts  : aa.Array
        The array which is converted from counts to electrons per seconds.
    exposure_time_map : aa.Array
        The exposure time at every data-point of the array.
    """
    return np.divide(array_counts, exposure_time_map)


def array_eps_to_adus(array_eps, exposure_time_map, gain):
    """
    Convert an array in units of electrons per second to adus, using an exposure time map containing the exposure
    time at every point in the array and the instrument gain.

    The conversion from electrons per second to ADUs is:

    [ADUs] = [EPS] * [Exposure_time] / [Gain]

    Parameters
    ----------
    array_eps  : aa.Array
        The array which is converted from electrons per seconds to adus.
    exposure_time_map : aa.Array
        The exposure time at every data-point of the array.
    gain : float
        The gain of the instrument used in the conversion to / from counts and ADUs.
    """
    return np.multiply(array_eps, exposure_time_map) / gain


def array_adus_to_eps(array_adus, exposure_time_map, gain):
    """
    Convert an array in units of electrons per second to adus, using an exposure time map containing the exposure
    time at every point in the array and the instrument gain.

    The conversion from ADUs to electrons per second is:

    [EPS] = [Counts] * [Gain] / [Exposure_time]

    Parameters
    ----------
    array_adus  : aa.Array
        The array which is converted from adus to electrons per seconds
    exposure_time_map : aa.Array
        The exposure time at every data-point of the array.
    gain : float
        The gain of the instrument used in the conversion to / from counts and ADUs.
    """
    return np.divide(gain * array_adus, exposure_time_map)


def array_counts_to_counts_per_second(array_counts, exposure_time):

    if exposure_time is None:
        raise exc.FrameException(
            "Cannot convert a Frame to units counts per second without an exposure time attribute (exposure_time = None)."
        )

    return array_counts / exposure_time


def array_with_random_uniform_values_added(array, upper_limit=0.001):
    """ Add random values drawn from a uniform distribution between zero and an input upper limit to an array.

    The current use-case of this function is adding small random values to a noise-map that is constant (e.g. all noise
    map values are the same). Constant noise-maps have been found to create "broken" inversions where the source is
    reconstructed as a set of constant values.

    Parameters
    ----------
    data : aa.Array
        The array that the uniform noise values are added to.
    upper_limit : float
        The upper limit of the uniform distribution from which the values are drawn.
    """
    return array + upper_limit * np.random.uniform(size=array.shape_1d)


def noise_map_from_data_eps_and_exposure_time_map(data_eps, exposure_time_map):
    """ Estimate the noise-map value in every data-point, by converting the data to units of counts and taking the
    square root of these values.
    
    For datasets that may have a background noise component, this function does not return the overall noise-map if the
    data is background subtracted. In this case, the returned noise-map is the Poisson noise-map.

    This function assumes the input data is in electrons per second and returns the noise-map in electrons per second.

    Parameters
    ----------
    data_eps : aa.Array
        The data in electrons second used to estimate the Poisson noise in every data point.
    exposure_time_map : aa.Array
        The exposure time at every data-point of the data.
    """
    return np.sqrt(np.abs(data_eps * exposure_time_map)) / exposure_time_map


def noise_map_from_weight_map(weight_map):
    """Setup the noise-map from a weight map, which is a form of noise-map that comes via HST image-reduction and \
    the software package MultiDrizzle.

    The variance in each pixel is computed as:

    Variance = 1.0 / sqrt(weight_map).

    The weight map may contain zeros, in which case the variances are converted to large values to omit them from \
    the analysis.

    Parameters
    -----------
    pixel_scales : float
        The size of each pixel in arc seconds.
    weight_map : ndarray
        The weight-value of each pixel which is converted to a variance.
    """
    np.seterr(divide="ignore")
    noise_map = 1.0 / np.sqrt(weight_map)
    noise_map[noise_map > 1.0e8] = 1.0e8
    return noise_map


def noise_map_from_inverse_noise_map(inverse_noise_map):
    """Setup the noise-map from an inverse noise-map.

    The variance in each pixel is computed as:

    Variance = 1.0 / inverse_noise_map.

    Parameters
    -----------
    inverse_noise_map : ndarray
        The inverse noise_map value of each pixel which is converted to a variance.
    """
    return 1.0 / inverse_noise_map


def noise_map_from_data_eps_exposure_time_map_and_background_noise_map(
    data_eps, exposure_time_map, background_noise_map
):
    """ Estimate the noise-map values in every data-point, by converting the data to units of counts, adding the
    background noise-map and taking the square root of these values.

    This function assumes the input data is in electrons per second and returns the noise-map in electrons per second.

    Parameters
    ----------
    data_eps : aa.Array
        The data in electrons second used to estimate the Poisson noise in every data point.
    exposure_time_map : aa.Array
        The exposure time at every data-point of the data.
    background_noise_map : aa.Array
        The RMS standard deviation error in every data point due to a background component of the noise-map in units
        of electrons per second.        
    """
    return (
        np.sqrt(
            (
                np.abs(data_eps * exposure_time_map)
                + np.square(background_noise_map * exposure_time_map)
            )
        )
        / exposure_time_map
    )


def background_noise_map_from_edges_of_image(image, no_edges):
    """
    Estimate the background noise level in an image using the data values at its edges. These edge values are binned
    into a histogram, with a Gaussian profile fitted to this histogram, such that its standard deviation (sigma) gives
    an estimate of the background noise.

    The background noise-map is returned on a 2D array the same dimensions as the image.

    Parameters
    ----------
    image : aa.Array
        The image whose edge values are used to estimate the background noise.
    no_edges : int
        Number of edges used to estimate the background level.
    """

    edges = []

    for edge_no in range(no_edges):
        top_edge = image.in_2d[edge_no, edge_no : image.shape_2d[1] - edge_no]
        bottom_edge = image.in_2d[
            image.shape_2d[0] - 1 - edge_no, edge_no : image.shape_2d[1] - edge_no
        ]
        left_edge = image.in_2d[edge_no + 1 : image.shape_2d[0] - 1 - edge_no, edge_no]
        right_edge = image.in_2d[
            edge_no + 1 : image.shape_2d[0] - 1 - edge_no,
            image.shape_2d[1] - 1 - edge_no,
        ]

        edges = np.concatenate((edges, top_edge, bottom_edge, right_edge, left_edge))

    return arrays.Array.full(fill_value=norm.fit(edges)[1], shape_2d=image.shape_2d)


def psf_with_odd_dimensions_from_psf(psf):
    """
    If the PSF kernel has one or two even-sized dimensions, return a PSF object where the kernel has odd-sized
    dimensions (odd-sized dimensions are required by a *Convolver*).

    Kernels are rescaled using the scikit-image routine rescale, which performs rescaling via an interpolation
    routine. This may lead to loss of accuracy in the PSF kernel and it is advised that users, where possible,
    create their PSF on an odd-sized array using their data reduction pipelines that remove this approximation.

    Parameters
    ----------
    rescale_factor : float
        The factor by which the kernel is rescaled. If this has a value of 1.0, the kernel is rescaled to the
        closest odd-sized dimensions (e.g. 20 -> 19). Higher / lower values scale to higher / lower dimensions.
    renormalize : bool
        Whether the PSF should be renormalized after being rescaled.
    """
    return psf.rescaled_with_odd_dimensions_from_rescale_factor(rescale_factor=1.0)


def exposure_time_map_from_exposure_time_and_background_noise_map(
    exposure_time, background_noise_map
):
    """
    Compute the exposure time map from the exposure time of the observation and the background noise-map.
    
    This function assumes the only source of noise in the background noise-map is due to a variable exposure time in
    every pixel due to effects like dithering, cosmic rays, etc.

    Parameters
    ----------
    exposure_time : float
        The total exposure time of the observation.
    background_noise_map : aa.Array
        The RMS standard deviation error in every data point due to a background component of the noise-map in units
        of electrons per second.
    """
    inverse_background_noise_map = 1.0 / background_noise_map
    relative_background_noise_map = inverse_background_noise_map / np.max(
        inverse_background_noise_map
    )
    return np.abs(exposure_time * (relative_background_noise_map))


def setup_random_seed(seed):
    """Setup the random seed. If the input seed is -1, the code will use a random seed for every run. If it is \
    positive, that seed is used for all runs, thereby giving reproducible results.

    Parameters
    ----------
    seed : int
        The seed of the random number generator.
    """
    if seed == -1:
        seed = np.random.randint(
            0, int(1e9)
        )  # Use one seed, so all regions have identical column non-uniformity.
    np.random.seed(seed)


def poisson_noise_from_data_eps(data_eps, exposure_time_map, seed=-1):
    """
    Generate a two-dimensional poisson noise_maps-mappers from an image.

    Values are computed from a Poisson distribution using the image's input values in units of counts.

    Parameters
    ----------
    data_eps : ndarray
        The 2D image, whose values in counts are used to draw Poisson noise_maps values.
    exposure_time_map : Union(ndarray, int)
        2D array of the exposure time in each pixel used to convert to / from counts and electrons per second.
    seed : int
        The seed of the random number generator, used for the random noise_maps maps.

    Returns
    -------
    poisson_noise_map: ndarray
        An array describing simulated poisson noise_maps
    """
    setup_random_seed(seed)
    image_counts = np.multiply(data_eps, exposure_time_map)
    return data_eps - np.divide(
        np.random.poisson(image_counts, data_eps.shape), exposure_time_map
    )


def data_eps_with_poisson_noise_added(data_eps, exposure_time_map, seed=-1):
    """
    Generate a two-dimensional poisson noise_maps-mappers from an image.

    Values are computed from a Poisson distribution using the image's input values in units of counts.

    Parameters
    ----------
    data_eps : ndarray
        The 2D image, whose values in counts are used to draw Poisson noise_maps values.
    exposure_time_map : Union(ndarray, int)
        2D array of the exposure time in each pixel used to convert to / from counts and electrons per second.
    seed : int
        The seed of the random number generator, used for the random noise_maps maps.

    Returns
    -------
    poisson_noise_map: ndarray
        An array describing simulated poisson noise_maps
    """
    return data_eps + poisson_noise_from_data_eps(
        data_eps=data_eps, exposure_time_map=exposure_time_map, seed=seed
    )


def gaussian_noise_from_shape_and_sigma(shape, sigma, seed=-1):
    """Generate a two-dimensional read noises-map, generating values from a Gaussian distribution with mean 0.0.

    Params
    ----------
    shape : (int, int)
        The (x,y) image_shape of the generated Gaussian noises map.
    read_noise : float
        Standard deviation of the 1D Gaussian that each noises value is drawn from
    seed : int
        The seed of the random number generator, used for the random noises maps.
    """
    if seed == -1:
        # Use one seed, so all regions have identical column non-uniformity.
        seed = np.random.randint(0, int(1e9))
    np.random.seed(seed)
    read_noise_map = np.random.normal(loc=0.0, scale=sigma, size=shape)
    return read_noise_map


def data_with_gaussian_noise_added(data, sigma, seed=-1):
    return data + gaussian_noise_from_shape_and_sigma(
        shape=data.shape, sigma=sigma, seed=seed
    )
