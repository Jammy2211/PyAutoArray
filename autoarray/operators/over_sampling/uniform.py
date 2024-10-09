import numpy as np
from typing import List, Tuple, Union

from autoconf import conf
from autoconf import cached_property

from autoarray.mask.mask_2d import Mask2D
from autoarray.operators.over_sampling.abstract import AbstractOverSampling
from autoarray.operators.over_sampling.abstract import AbstractOverSampler
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray import exc
from autoarray.operators.over_sampling import over_sample_util


class OverSamplingUniform(AbstractOverSampling):
    def __init__(self, sub_size: Union[int, Array2D]):
        """
        Over samples grid calculations using a uniform sub-grid.

        When a 2D grid of (y,x) coordinates is input into a function, the result is evaluated at every coordinate
        on the grid. When the grid is paired to a 2D image (e.g. an `Array2D`) the solution needs to approximate
        the 2D integral of that function in each pixel. Over sample objects define how this over-sampling is performed.

        This object inputs a uniform sub-grid, where every image-pixel is split into a uniform grid of sub-pixels. The
        function is evaluated at every sub-pixel, and the final value in each pixel is computed by summing the
        contribution from all sub-pixels.

        This is the simplest over-sampling method, but may not provide precise solutions for functions that vary
        significantly within a pixel. To achieve precision in these pixels a high `sub_size` is required, which can
        be computationally expensive as it is applied to every pixel.

        **Example**

        If the mask's `sub_size` is > 1, the grid is defined as a sub-grid where each entry corresponds to the (y,x)
        coordinates at the centre of each sub-pixel of an unmasked pixel. The Grid2D is therefore stored as an ndarray
        of shape [total_unmasked_coordinates*sub_size**2, 2]

        The sub-grid indexes are ordered such that pixels begin from the first (top-left) sub-pixel in the first
        unmasked pixel. Indexes then go over the sub-pixels in each unmasked pixel, for every unmasked pixel.
        Therefore, the sub-grid is an ndarray of shape [total_unmasked_coordinates*(sub_grid_shape)**2, 2].

        For example:

        - grid[9, 1] - using a 2x2 sub-grid, gives the 3rd unmasked pixel's 2nd sub-pixel x-coordinate.
        - grid[9, 1] - using a 3x3 sub-grid, gives the 2nd unmasked pixel's 1st sub-pixel x-coordinate.
        - grid[27, 0] - using a 3x3 sub-grid, gives the 4th unmasked pixel's 1st sub-pixel y-coordinate.

        Below is a visual illustration of a sub grid. Indexing of each sub-pixel goes from the top-left corner. In
        contrast to the grid above, our illustration below restricts the mask to just 2 pixels, to keep the
        illustration brief.

        .. code-block:: bash

             x x x x x x x x x x
             x x x x x x x x x x     This is an example mask.Mask2D, where:
             x x x x x x x x x x
             x x x x x x x x x x     x = `True` (Pixel is masked and excluded from lens)
             x x x x O O x x x x     O = `False` (Pixel is not masked and included in lens)
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x

        Our grid with a sub-size looks like it did before:

        .. code-block:: bash

            pixel_scales = 1.0"

            <--- -ve  x  +ve -->

             x x x x x x x x x x  ^
             x x x x x x x x x x  I
             x x x x x x x x x x  I                        y     x
             x x x x x x x x x x +ve  grid[0] = [0.5,  -1.5]
             x x x x 0 1 x x x x  y   grid[1] = [0.5,  -0.5]
             x x x x x x x x x x -ve
             x x x x x x x x x x  I
             x x x x x x x x x x  I
             x x x x x x x x x x \/
             x x x x x x x x x x

        However, if the sub-size is 2, we go to each unmasked pixel and allocate sub-pixel coordinates for it. For
        example, for pixel 0, if *sub_size=2*, we use a 2x2 sub-grid:

        .. code-block:: bash

            Pixel 0 - (2x2):
                                y      x
                   grid[0] = [0.66, -1.66]
            I0I1I  grid[1] = [0.66, -1.33]
            I2I3I  grid[2] = [0.33, -1.66]
                   grid[3] = [0.33, -1.33]

        If we used a sub_size of 3, for the pixel we we would create a 3x3 sub-grid:

        .. code-block:: bash

                                  y      x
                     grid[0] = [0.75, -0.75]
                     grid[1] = [0.75, -0.5]
                     grid[2] = [0.75, -0.25]
            I0I1I2I  grid[3] = [0.5,  -0.75]
            I3I4I5I  grid[4] = [0.5,  -0.5]
            I6I7I8I  grid[5] = [0.5,  -0.25]
                     grid[6] = [0.25, -0.75]
                     grid[7] = [0.25, -0.5]
                     grid[8] = [0.25, -0.25]

        All sub-pixels in masked pixels have values (0.0, 0.0).

        __Adaptive Oversampling__

        By default, the sub-grid is the same size in every pixel (e.g. the value of `sub_size` is an integer that
        defines the size of the sub-grid for every pixel).

        However, the `sub_size` can also be input as an `Array2D`, with varying integer values for each pixel.
        This is called adaptive over-sampling and is used to adapt the over-sampling to the bright regions of the
        data, saving computational time.

        __Pixelization__

        For pixelizations performed in the inversion module, over sampling is equally important. Now, the over
        sampling maps multiple data sub-pixels to pixels in the pixelization, where mappings are performed fractionally
        based on the sub-grid sizes.

        The over sampling class has functions dedicated to mapping between the sub-grid and pixel-grid, for example
        `sub_mask_native_for_sub_mask_slim` and `slim_for_sub_slim`.

        Parameters
        ----------
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        """

        self.sub_size = sub_size

    @classmethod
    def from_radial_bins(
        cls,
        grid: Grid2D,
        sub_size_list: List[int],
        radial_list: List[float],
        centre_list: List[Tuple] = None,
    ) -> "OverSamplingUniform":
        """
        Returns an adaptive sub-grid size based on the radial distance of every pixel from the centre of the mask.

        The adaptive sub-grid size is computed as follows:

        1) Compute the radial distance of every pixel in the mask from the centre of the mask (or input centres).
        2) For every pixel, determine the sub-grid size based on the radial distance of that pixel. For example, if
        the first entry in `radial_list` is 0.5 and the first entry in `sub_size_list` 8, all pixels with a radial
        distance less than 0.5 will have a sub-grid size of 8x8.

        This scheme can produce high sub-size values towards the centre of the mask, where the galaxy is brightest and
        has the most rapidly changing light profile which requires a high sub-grid size to resolve accurately.

        If the data has multiple galaxies, the `centre_list` can be used to define the centre of each galaxy
        and therefore increase the sub-grid size based on the light profile of each individual galaxy.

        Parameters
        ----------
        mask
            The mask defining the 2D region where the over-sampled grid is computed.
        sub_size_list
            The sub-grid size for every radial bin.
        radial_list
            The radial distance defining each bin, which are refeneced based on the previous entry. For example, if
            the first entry is 0.5, the second 1.0 and the third 1.5, the adaptive sub-grid size will be between 0.5
            and 1.0 for the first sub-grid size, between 1.0 and 1.5 for the second sub-grid size, etc.
        centre_list
            A list of centres for each galaxy whose centres require higher sub-grid sizes.

        Returns
        -------
        A uniform over-sampling object with an adaptive sub-grid size based on the radial distance of every pixel from
        the centre of the mask.
        """

        if centre_list is None:
            centre_list = [grid.mask.mask_centre]

        sub_size = np.zeros(grid.shape_slim)

        for centre in centre_list:
            radial_grid = grid.distances_to_coordinate_from(coordinate=centre)

            sub_size_of_centre = over_sample_util.sub_size_radial_bins_from(
                radial_grid=np.array(radial_grid),
                sub_size_list=np.array(sub_size_list),
                radial_list=np.array(radial_list),
            )

            sub_size = np.where(
                sub_size_of_centre > sub_size, sub_size_of_centre, sub_size
            )

        sub_size = Array2D(values=sub_size, mask=grid.mask)

        return cls(sub_size=sub_size)

    @classmethod
    def from_adaptive_scheme(
        cls, grid: Grid2D, name: str, centre: Tuple[float, float]
    ) -> "OverSamplingUniform":
        """
        Returns a 2D grid whose over sampling is adaptive, placing a high number of sub-pixels in the regions of the
        grid closest to the centre input (y,x) coordinates.

        This adaptive over sampling is primarily used in PyAutoGalaxy, to evaluate the image of a light profile
        (e.g. a Sersic function) with high levels of sub gridding in its centre and lower levels of sub gridding
        further away from the centre (saving computational time).

        The `autogalaxy_workspace` and `autolens_workspace` packages have guides called `over_sampling.ipynb`
        which describe over sampling in more detail.

        The inputs `name` and `centre` typically correspond to the name of the light profile (e.g. `Sersic`) and
        its `centre`, so that the adaptive over sampling parameters for that light profile are loaded from config
        files and used to setup the grid.

        Parameters
        ----------
        name
            The name of the light profile, which is used to load the adaptive over sampling parameters from config files.
        centre
            The (y,x) centre of the adaptive over sampled grid, around which the highest sub-pixel resolution is placed.

        Returns
        -------
        A new Grid2D with adaptive over sampling.

        """

        if not grid.is_uniform:
            raise exc.GridException(
                "You cannot make an adaptive over-sampled grid from a non-uniform grid."
            )

        sub_size_list = conf.instance["grids"]["over_sampling"]["sub_size_list"][name]
        radial_factor_list = conf.instance["grids"]["over_sampling"][
            "radial_factor_list"
        ][name]

        centre = grid.geometry.scaled_coordinate_2d_to_scaled_at_pixel_centre_from(
            scaled_coordinate_2d=centre
        )

        return OverSamplingUniform.from_radial_bins(
            grid=grid,
            sub_size_list=sub_size_list,
            radial_list=[
                min(grid.pixel_scales) * radial_factor
                for radial_factor in radial_factor_list
            ],
            centre_list=[centre],
        )

    @classmethod
    def from_adapt(
        cls,
        data: Array2D,
        noise_map: Array2D,
        signal_to_noise_cut: float = 5.0,
        sub_size_lower: int = 2,
        sub_size_upper: int = 4,
    ):
        """
        Returns an adaptive sub-grid size based on the signal-to-noise of the data.

        The adaptive sub-grid size is computed as follows:

        1) The signal-to-noise of every pixel is computed as the data divided by the noise-map.
        2) For all pixels with signal-to-noise above the signal-to-noise cut, the sub-grid size is set to the upper
          value. For all other pixels, the sub-grid size is set to the lower value.

        This scheme can produce low sub-size values over entire datasets if the data has a low signal-to-noise. However,
        just because the data has a low signal-to-noise does not mean that the sub-grid size should be low.

        To mitigate this, the signal-to-noise cut is set to the maximum signal-to-noise of the data divided by 2.0 if
        it this value is below the signal-to-noise cut.

        Parameters
        ----------
        data
            The data which is to be fitted via a calculation using this over-sampling sub-grid.
        noise_map
            The noise-map of the data.
        signal_to_noise_cut
            The signal-to-noise cut which defines whether the sub-grid size is the upper or lower value.
        sub_size_lower
            The sub-grid size for pixels with signal-to-noise below the signal-to-noise cut.
        sub_size_upper
            The sub-grid size for pixels with signal-to-noise above the signal-to-noise cut.

        Returns
        -------
        The adaptive sub-grid sizes.
        """
        signal_to_noise = data / noise_map

        if np.max(signal_to_noise) < (2.0 * signal_to_noise_cut):
            signal_to_noise_cut = np.max(signal_to_noise) / 2.0

        sub_size = np.where(
            signal_to_noise > signal_to_noise_cut, sub_size_upper, sub_size_lower
        )

        sub_size = Array2D(values=sub_size, mask=data.mask)

        return cls(sub_size=sub_size)

    def over_sampler_from(self, mask: Mask2D) -> "OverSamplerUniform":
        return OverSamplerUniform(
            mask=mask,
            sub_size=self.sub_size,
        )


class OverSamplerUniform(AbstractOverSampler):
    def __init__(self, mask: Mask2D, sub_size: Union[int, Array2D]):
        """
         Over samples grid calculations using a uniform sub-grid.

         See the class `OverSamplingUniform` for a description of how the over-sampling works in full.

         The class `OverSamplingUniform` is used for the high level API, whereby this is where users input their
         preferred over-sampling configuration. This class, `OverSamplerUniform`, contains the functionality
         which actually performs the over-sampling calculations, but is hidden from the user.

        Parameters
        ----------
        mask
            The mask defining the 2D region where the over-sampled grid is computed.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        """
        self.mask = mask

        if isinstance(sub_size, int):
            sub_size = Array2D(
                values=np.full(fill_value=sub_size, shape=mask.shape_slim), mask=mask
            )

        self.sub_size = sub_size

    @property
    def sub_total(self):
        """
        The total number of sub-pixels in the entire mask.
        """
        return int(np.sum(self.sub_size**2))

    @property
    def sub_length(self) -> Array2D:
        """
        The total number of sub-pixels in a give pixel,

        For example, a sub-size of 3x3 means every pixel has 9 sub-pixels.
        """
        return self.sub_size**self.mask.dimensions

    @property
    def sub_fraction(self) -> Array2D:
        """
        The fraction of the area of a pixel every sub-pixel contains.

        For example, a sub-size of 3x3 mean every pixel contains 1/9 the area.
        """

        return 1.0 / self.sub_length

    @property
    def sub_pixel_areas(self) -> np.ndarray:
        """
        The area of every sub-pixel in the mask.
        """
        sub_pixel_areas = np.zeros(self.sub_total)

        k = 0

        pixel_area = self.mask.pixel_scales[0] * self.mask.pixel_scales[1]

        for i in range(self.sub_size.shape[0]):
            for j in range(self.sub_size[i] ** 2):
                sub_pixel_areas[k] = pixel_area / self.sub_size[i] ** 2
                k += 1

        return sub_pixel_areas

    @cached_property
    def over_sampled_grid(self) -> Grid2DIrregular:
        grid = over_sample_util.grid_2d_slim_over_sampled_via_mask_from(
            mask_2d=np.array(self.mask),
            pixel_scales=self.mask.pixel_scales,
            sub_size=np.array(self.sub_size).astype("int"),
            origin=self.mask.origin,
        )

        return Grid2DIrregular(values=grid)

    def binned_array_2d_from(self, array: Array2D) -> "Array2D":
        """
        Convenience method to access the binned-up array in its 1D representation, which is a Grid2D stored as an
        ``ndarray`` of shape [total_unmasked_pixels, 2].

        The binning up process converts a array from (y,x) values where each value is a coordinate on the sub-array to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a array with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the array is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D.

        In **PyAutoCTI** all `Array2D` objects are used in their `native` representation without sub-gridding.
        Significant memory can be saved by only store this format, thus the `native_binned_only` config override
        can force this behaviour. It is recommended users do not use this option to avoid unexpected behaviour.
        """
        if conf.instance["general"]["structures"]["native_binned_only"]:
            return self

        try:
            array = array.slim
        except AttributeError:
            pass

        binned_array_2d = over_sample_util.binned_array_2d_from(
            array_2d=np.array(array),
            mask_2d=np.array(self.mask),
            sub_size=np.array(self.sub_size).astype("int"),
        )

        return Array2D(
            values=binned_array_2d,
            mask=self.mask,
        )

    def array_via_func_from(self, func, obj, *args, **kwargs):
        over_sampled_grid = self.over_sampled_grid

        if obj is not None:
            values = func(obj, over_sampled_grid, *args, **kwargs)
        else:
            values = func(over_sampled_grid, *args, **kwargs)

        return self.binned_array_2d_from(array=values)

    @cached_property
    def sub_mask_native_for_sub_mask_slim(self) -> np.ndarray:
        """
        Derives a 1D ``ndarray`` which maps every subgridded 1D ``slim`` index of the ``Mask2D`` to its
        subgridded 2D ``native`` index.

        For example, for the following ``Mask2D`` for ``sub_size=1``:

        ::
            [[True,  True,  True, True]
             [True, False, False, True],
             [True, False,  True, True],
             [True,  True,  True, True]]

        This has three unmasked (``False`` values) which have the ``slim`` indexes:

        ::
            [0, 1, 2]

        The array ``sub_mask_native_for_sub_mask_slim`` is therefore:

        ::
            [[1,1], [1,2], [2,1]]

        For a ``Mask2D`` with ``sub_size=2`` each unmasked ``False`` entry is split into a sub-pixel of size 2x2 and
        there are therefore 12 ``slim`` indexes:

        ::
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        The array ``native_for_slim`` is therefore:

        ::
            [[2,2], [2,3], [2,4], [2,5], [3,2], [3,3], [3,4], [3,5], [4,2], [4,3], [5,2], [5,3]]

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[[True,  True,  True, True]
                      [True, False, False, True],
                      [True, False,  True, True],
                      [True,  True,  True, True]]
                pixel_scales=1.0,
            )

            derive_indexes_2d = aa.DeriveIndexes2D(mask=mask_2d)

            print(derive_indexes_2d.sub_mask_native_for_sub_mask_slim)
        """
        return over_sample_util.native_sub_index_for_slim_sub_index_2d_from(
            mask_2d=self.mask.array, sub_size=np.array(self.sub_size)
        ).astype("int")

    @cached_property
    def slim_for_sub_slim(self) -> np.ndarray:
        """
        Derives a 1D ``ndarray`` which maps every subgridded 1D ``slim`` index of the ``Mask2D`` to its
        non-subgridded 1D ``slim`` index.

        For example, for the following ``Mask2D`` for ``sub_size=1``:

        ::
            [[True,  True,  True, True]
             [True, False, False, True],
             [True, False,  True, True],
             [True,  True,  True, True]]

        This has three unmasked (``False`` values) which have the ``slim`` indexes:

        ::
            [0, 1, 2]

        The array ``slim_for_sub_slim`` is therefore:

        ::
            [0, 1, 2]

        For a ``Mask2D`` with ``sub_size=2`` each unmasked ``False`` entry is split into a sub-pixel of size 2x2.
        Therefore the array ``slim_for_sub_slim`` becomes:

        ::
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[[True,  True,  True, True]
                      [True, False, False, True],
                      [True, False,  True, True],
                      [True,  True,  True, True]]
                pixel_scales=1.0,
            )

            derive_indexes_2d = aa.DeriveIndexes2D(mask=mask_2d)

            print(derive_indexes_2d.slim_for_sub_slim)
        """
        return over_sample_util.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask_2d=np.array(self.mask), sub_size=np.array(self.sub_size)
        ).astype("int")
