from autoarray import decorator_util
import numpy as np
from autoarray import exc
from autoarray.util import mask_util


class Convolver(object):
    def __init__(self, mask_2d, kernel_2d):
        """ Class to setup the 1D convolution of an / util matrix.

        Take a simple 3x3 and masks:

        [[2, 8, 2],
        [5, 7, 5],
        [3, 1, 4]]

        [[True, False, True],   (True means that the value is masked)
        [False, False, False],
        [True, False, True]]

        A set of values in a corresponding 1d array of this might be represented as:

        [2, 8, 2, 5, 7, 5, 3, 1, 4]

        and after masking as:

        [8, 5, 7, 5, 1]

        Setup is required to perform 2D real-space convolution on the masked array. This module finds the \
        relationship between the unmasked 2D datas, masked datas and kernel, so that 2D real-space convolutions \
        can be efficiently applied to reduced 1D masked structures.

        This calculation also accounts for the blurring of light outside of the masked regions which blurs into \
        the masked region.

        IMAGE FRAMES:
        ------------

        For a masked in 2D, one can compute for every pixel all of the unmasked pixels it will blur light into for \
        a given PSF kernel size, e.g.:

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an imaging.Mask, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     x = True (Pixel is masked and excluded from lens)
        |x|x|x|o|o|o|x|x|x|x|     o = False (Pixel is not masked and included in lens)
        |x|x|x|o|o|o|x|x|x|x|
        |x|x|x|o|o|o|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|

        Here, there are 9 unmasked pixels. Indexing of each unmasked pixel goes from the top-left corner right and \
        downwards, therefore:

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|0|1|2|x|x|x|x|
        |x|x|x|3|4|5|x|x|x|x|
        |x|x|x|6|7|8|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|

        For every unmasked pixel, the Convolver over-lays the PSF and computes three quantities;

        image_frame_indexes - The indexes of all masked pixels it will blur light into.
        image_frame_kernels - The kernel values that overlap each masked pixel it will blur light into.
        image_frame_length - The number of masked pixels it will blur light into (unmasked pixels are excluded)

        For example, if we had the following 3x3 kernel:

        |0.1|0.2|0.3|
        |0.4|0.5|0.6|
        |0.7|0.8|0.9|

        For pixel 0 above, when we overlap the kernel 4 unmasked pixels overlap this kernel, such that:

        image_frame_indexes = [0, 1, 3, 4]
        image_frame_kernels = [0.5, 0.6, 0.8, 0.9]
        image_frame_length = 4

        Noting that the other 5 kernel values (0.1, 0.2, 0.3, 0.4, 0.7) overlap masked pixels and are thus discarded.

        For pixel 1, we get the following results:

        image_frame_indexes = [0, 1, 2, 3, 4, 5]
        image_frame_kernels = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        image_frame_lengths = 6

        In the majority of cases, the kernel will overlap only unmasked pixels. This is the case above for \
        central pixel 4, where:

        image_frame_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        image_frame_kernels = [0,1, 0.2, 0,3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        image_frame_lengths = 9

        Once we have set up all these quantities, the convolution routine simply uses them to convolve a 1D array of a
        masked or the masked of a util in the inversion module.

        BLURRING FRAMES:
        --------------

        Whilst the scheme above accounts for all blurred light within the masks, it does not account for the fact that \
        pixels outside of the masks will also blur light into it. This effect is accounted for using blurring frames.

        It is omitted for util matrix blurring, as an inversion does not fit datas outside of the masks.

        First, a blurring masks is computed from a masks, which describes all pixels which are close enough to the masks \
        to blur light into it for a given kernel size. Following the example above, the following blurring masks is \
        computed:

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an example grid.Mask, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|o|o|o|o|o|x|x|x|     x = True (Pixel is masked and excluded from lens)
        |x|x|o|x|x|x|o|x|x|x|     o = False (Pixel is not masked and included in lens)
        |x|x|o|x|x|x|o|x|x|x|
        |x|x|o|x|x|x|o|x|x|x|
        |x|x|o|o|o|o|o|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|

        Indexing again goes from the top-left corner right and downwards:

        |x|x| x| x| x| x| x|x|x|x|
        |x|x| x| x| x| x| x|x|x|x|
        |x|x| x| x| x| x| x|x|x|x|
        |x|x| 0| 1| 2| 3| 4|x|x|x|
        |x|x| 5| x| x| x| 6|x|x|x|
        |x|x| 7| x| x| x| 8|x|x|x|
        |x|x| 9| x| x| x|10|x|x|x|
        |x|x|11|12|13|14|15|x|x|x|
        |x|x| x| x| x| x| x|x|x|x|
        |x|x| x| x| x| x| x|x|x|x|

        For every unmasked blurring-pixel, the Convolver over-lays the PSF kernel and computes three quantities;

        blurring_frame_indexes - The indexes of all unmasked pixels (not unmasked blurring pixels) it will \
        blur light into.
        bluring_frame_kernels - The kernel values that overlap each pixel it will blur light into.
        blurring_frame_length - The number of pixels it will blur light into.

        The blurring frame therefore does not perform any blurring which blurs light into other blurring pixels. \
        It only performs computations which add light inside of the masks.

        For pixel 0 above, when we overlap the 3x3 kernel above only 1 unmasked pixels overlaps the kernel, such that:

        blurring_frame_indexes = [0] (This 0 refers to pixel 0 within the masks, not blurring_frame_pixel 0)
        blurring_frame_kernels = [0.9]
        blurring_frame_length = 1

        For pixel 1 above, when we overlap the 3x3 kernel above 2 unmasked pixels overlap the kernel, such that:

        blurring_frame_indexes = [0, 1]  (This 0 and 1 refer to pixels 0 and 1 within the masks)
        blurring_frame_kernels = [0.8, 0.9]
        blurring_frame_length = 2

        For pixel 3 above, when we overlap the 3x3 kernel above 3 unmasked pixels overlap the kernel, such that:

        blurring_frame_indexes = [0, 1, 2]  (Again, these are pixels 0, 1 and 2)
        blurring_frame_kernels = [0.7, 0.8, 0.9]
        blurring_frame_length = 3

        Parameters
        ----------
        mask_2d : Mask
            The masks, where True eliminates datas.
        blurring_mask : Mask
            A masks of pixels outside the masks but whose light blurs into it after PSF convolution.
        kernel_2d : grid.PSF or ndarray
            An array representing a PSF.
        """
        if kernel_2d.shape[0] % 2 == 0 or kernel_2d.shape[1] % 2 == 0:
            raise exc.ConvolutionException("PSF kernel must be odd")

        self.mask = mask_2d

        self.mask_index_array = np.full(mask_2d.shape, -1)
        self.pixels_in_mask = int(np.size(mask_2d) - np.sum(mask_2d))

        count = 0
        for x in range(mask_2d.shape[0]):
            for y in range(mask_2d.shape[1]):
                if not mask_2d[x, y]:
                    self.mask_index_array[x, y] = count
                    count += 1

        self.kernel = kernel_2d
        self.kernel_max_size = self.kernel.shape[0] * self.kernel.shape[1]

        mask_1d_index = 0
        self.image_frame_1d_indexes = np.zeros(
            (self.pixels_in_mask, self.kernel_max_size), dtype="int"
        )
        self.image_frame_1d_kernels = np.zeros(
            (self.pixels_in_mask, self.kernel_max_size)
        )
        self.image_frame_1d_lengths = np.zeros((self.pixels_in_mask), dtype="int")
        for x in range(self.mask_index_array.shape[0]):
            for y in range(self.mask_index_array.shape[1]):
                if not mask_2d[x][y]:
                    image_frame_1d_indexes, image_frame_1d_kernels = self.frame_at_coordinates_jit(
                        (x, y), mask_2d, self.mask_index_array, self.kernel[:, :]
                    )
                    self.image_frame_1d_indexes[
                        mask_1d_index, :
                    ] = image_frame_1d_indexes
                    self.image_frame_1d_kernels[
                        mask_1d_index, :
                    ] = image_frame_1d_kernels
                    self.image_frame_1d_lengths[mask_1d_index] = image_frame_1d_indexes[
                        image_frame_1d_indexes >= 0
                    ].shape[0]
                    mask_1d_index += 1

        self.blurring_mask = mask_util.blurring_mask_2d_from_mask_2d_and_kernel_shape_2d(
            mask_2d=mask_2d, kernel_shape_2d=kernel_2d.shape
        )

        self.pixels_in_blurring_mask = int(
            np.size(self.blurring_mask) - np.sum(self.blurring_mask)
        )

        mask_1d_index = 0
        self.blurring_frame_1d_indexes = np.zeros(
            (self.pixels_in_blurring_mask, self.kernel_max_size), dtype="int"
        )
        self.blurring_frame_1d_kernels = np.zeros(
            (self.pixels_in_blurring_mask, self.kernel_max_size)
        )
        self.blurring_frame_1d_lengths = np.zeros(
            (self.pixels_in_blurring_mask), dtype="int"
        )
        for x in range(mask_2d.shape[0]):
            for y in range(mask_2d.shape[1]):
                if mask_2d[x][y] and not self.blurring_mask[x, y]:
                    image_frame_1d_indexes, image_frame_1d_kernels = self.frame_at_coordinates_jit(
                        (x, y), mask_2d, self.mask_index_array, self.kernel
                    )
                    self.blurring_frame_1d_indexes[
                        mask_1d_index, :
                    ] = image_frame_1d_indexes
                    self.blurring_frame_1d_kernels[
                        mask_1d_index, :
                    ] = image_frame_1d_kernels
                    self.blurring_frame_1d_lengths[
                        mask_1d_index
                    ] = image_frame_1d_indexes[image_frame_1d_indexes >= 0].shape[0]
                    mask_1d_index += 1

    @staticmethod
    @decorator_util.jit()
    def frame_at_coordinates_jit(coordinates, mask, mask_index_array, kernel):
        """ Compute the frame (indexes of pixels light is blurred into) and kernel_frame (kernel kernel values of those \
        pixels) for a given coordinate in a masks and its PSF.

        Parameters
        ----------
        coordinates: (int, int)
            The coordinates of mask_index_array on which the frame should be centred
        kernel_shape: (int, int)
            The shape of the kernel for which this frame will be used
        """

        kernel_shape = kernel.shape
        kernel_max_size = kernel_shape[0] * kernel_shape[1]

        half_x = int(kernel_shape[0] / 2)
        half_y = int(kernel_shape[1] / 2)

        frame = -1 * np.ones((kernel_max_size))
        kernel_frame = -1.0 * np.ones((kernel_max_size))

        count = 0
        for i in range(kernel_shape[0]):
            for j in range(kernel_shape[1]):
                x = coordinates[0] - half_x + i
                y = coordinates[1] - half_y + j
                if (
                    0 <= x < mask_index_array.shape[0]
                    and 0 <= y < mask_index_array.shape[1]
                ):
                    value = mask_index_array[x, y]
                    if value >= 0 and not mask[x, y]:
                        frame[count] = value
                        kernel_frame[count] = kernel[i, j]
                        count += 1

        return frame, kernel_frame

    def convolved_image_1d_from_image_array_and_blurring_array(
        self, image_array, blurring_array
    ):
        """For a given 1D array and blurring array, convolve the two using this convolver.

        Parameters
        -----------
        image_array : ndarray
            1D array of the values which are to be blurred with the convolver's PSF.
        blurring_array : ndarray
            1D array of the blurring values which blur into the array after PSF convolution.
        """

        if self.blurring_mask is None:
            raise exc.ConvolutionException(
                "You cannot use the convolve_image function of a Convolver if the Convolver was"
                "not created with a blurring_mask."
            )

        convolved_image_1d = self.convolve_jit(
            image_1d_array=image_array,
            image_frame_1d_indexes=self.image_frame_1d_indexes,
            image_frame_1d_kernels=self.image_frame_1d_kernels,
            image_frame_1d_lengths=self.image_frame_1d_lengths,
            blurring_1d_array=blurring_array,
            blurring_frame_1d_indexes=self.blurring_frame_1d_indexes,
            blurring_frame_1d_kernels=self.blurring_frame_1d_kernels,
            blurring_frame_1d_lengths=self.blurring_frame_1d_lengths,
        )

        return self.mask.mapping.array_from_array_1d(array_1d=convolved_image_1d)

    def convolved_scaled_array_from_image_array_and_blurring_array(
        self, image_array, blurring_array
    ):
        """For a given 1D array and blurring array, convolve the two using this convolver.

        Parameters
        -----------
        image_array : ndarray
            1D array of the values which are to be blurred with the convolver's PSF.
        blurring_array : ndarray
            1D array of the blurring values which blur into the array after PSF convolution.
        """
        convolved_image_1d = self.convolved_image_1d_from_image_array_and_blurring_array(
            image_array=image_array, blurring_array=blurring_array
        )
        return self.mask.mapping.array_from_array_1d(array_1d=convolved_image_1d)

    @staticmethod
    @decorator_util.jit()
    def convolve_jit(
        image_1d_array,
        image_frame_1d_indexes,
        image_frame_1d_kernels,
        image_frame_1d_lengths,
        blurring_1d_array,
        blurring_frame_1d_indexes,
        blurring_frame_1d_kernels,
        blurring_frame_1d_lengths,
    ):

        blurred_image_1d = np.zeros(image_1d_array.shape)

        for image_1d_index in range(len(image_1d_array)):

            frame_1d_indexes = image_frame_1d_indexes[image_1d_index]
            frame_1d_kernel = image_frame_1d_kernels[image_1d_index]
            frame_1d_length = image_frame_1d_lengths[image_1d_index]
            image_value = image_1d_array[image_1d_index]

            for kernel_1d_index in range(frame_1d_length):

                vector_index = frame_1d_indexes[kernel_1d_index]
                kernel_value = frame_1d_kernel[kernel_1d_index]
                blurred_image_1d[vector_index] += image_value * kernel_value

        for blurring_1d_index in range(len(blurring_1d_array)):

            frame_1d_indexes = blurring_frame_1d_indexes[blurring_1d_index]
            frame_1d_kernel = blurring_frame_1d_kernels[blurring_1d_index]
            frame_1d_length = blurring_frame_1d_lengths[blurring_1d_index]
            image_value = blurring_1d_array[blurring_1d_index]

            for kernel_1d_index in range(frame_1d_length):
                vector_index = frame_1d_indexes[kernel_1d_index]
                kernel_value = frame_1d_kernel[kernel_1d_index]
                blurred_image_1d[vector_index] += image_value * kernel_value

        return blurred_image_1d

    def convolve_mapping_matrix(self, mapping_matrix):
        """For a given inversion util matrix, convolve every pixel's mapped with the PSF kernel.

        A util matrix provides non-zero entries in all elements which map two pixels to one another
        (see *inversions.mappers*).

        For example, lets take an which is masked using a 'cross' of 5 pixels:

        [[ True, False,  True]],
        [[False, False, False]],
        [[ True, False,  True]]

        As example util matrix of this cross is as follows (5 pixels x 3 source pixels):

        [1, 0, 0] [0->0]
        [1, 0, 0] [1->0]
        [0, 1, 0] [2->1]
        [0, 1, 0] [3->1]
        [0, 0, 1] [4->2]

        For each source-pixel, we can create an of its unit-surface brightnesses by util the non-zero
        entries back to masks. For example, doing this for source pixel 1 gives:

        [[0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0]]
        [[0.0, 0.0, 0.0]]

        And source pixel 2:

        [[0.0, 0.0, 0.0]],
        [[0.0, 1.0, 1.0]]
        [[0.0, 0.0, 0.0]]

        We then convolve each of these with our PSF kernel, in 2 dimensions, like we would a hyper grid. For
        example, using the kernel below:

        kernel:

        [[0.0, 0.1, 0.0]]
        [[0.1, 0.6, 0.1]]
        [[0.0, 0.1, 0.0]]

        Blurred Source Pixel 1 (we don't need to perform the convolution into masked pixels):

        [[0.0, 0.6, 0.0]],
        [[0.6, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]]

        Blurred Source pixel 2:

        [[0.0, 0.0, 0.0]],
        [[0.0, 0.7, 0.7]],
        [[0.0, 0.0, 0.0]]

        Finally, we map each of these blurred back to a blurred util matrix, which is analogous to the
        util matrix.

        [0.6, 0.0, 0.0] [0->0]
        [0.6, 0.0, 0.0] [1->0]
        [0.0, 0.7, 0.0] [2->1]
        [0.0, 0.7, 0.0] [3->1]
        [0.0, 0.0, 0.6] [4->2]

        If the util matrix is sub-gridded, we perform the convolution on the fractional surface brightnesses in an
        identical fashion to above.

        Parameters
        -----------
        mapping_matrix : ndarray
            The 2D util matix describing how every inversion pixel maps to an datas_ pixel.
        """
        return self.convolve_matrix_jit(
            mapping_matrix=mapping_matrix,
            image_frame_1d_indexes=self.image_frame_1d_indexes,
            image_frame_1d_kernels=self.image_frame_1d_kernels,
            image_frame_1d_lengths=self.image_frame_1d_lengths,
        )

    @staticmethod
    @decorator_util.jit()
    def convolve_matrix_jit(
        mapping_matrix,
        image_frame_1d_indexes,
        image_frame_1d_kernels,
        image_frame_1d_lengths,
    ):

        blurred_mapping_matrix = np.zeros(mapping_matrix.shape)

        for pixel_1d_index in range(mapping_matrix.shape[1]):
            for image_1d_index in range(mapping_matrix.shape[0]):

                value = mapping_matrix[image_1d_index, pixel_1d_index]

                if value > 0:

                    frame_1d_indexes = image_frame_1d_indexes[image_1d_index]
                    frame_1d_kernel = image_frame_1d_kernels[image_1d_index]
                    frame_1d_length = image_frame_1d_lengths[image_1d_index]

                    for kernel_1d_index in range(frame_1d_length):
                        vector_index = frame_1d_indexes[kernel_1d_index]
                        kernel_value = frame_1d_kernel[kernel_1d_index]
                        blurred_mapping_matrix[vector_index, pixel_1d_index] += (
                            value * kernel_value
                        )

        return blurred_mapping_matrix
