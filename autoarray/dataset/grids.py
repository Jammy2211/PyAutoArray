from typing import Optional, Union

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.arrays.kernel_2d import Kernel2D
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.inversion.pixelization.border_relocator import BorderRelocator
from autoconf import cached_property


class GridsDataset:
    def __init__(
        self,
        mask: Mask2D,
        over_sample_size_lp: Union[int, Array2D],
        over_sample_size_pixelization: Union[int, Array2D],
        psf: Optional[Kernel2D] = None,
    ):
        """
        Contains grids of (y,x) Cartesian coordinates at the centre of every pixel in the dataset's image and
        mask, which are used for performing calculations on the datas.

        The following grids are contained:

        - `uniform`: A grids of (y,x) coordinates which aligns with the centre of every image pixel of the image data,
        which is used for most normal calculations (e.g. evaluating the amount of light that falls in an pixel
        from a light profile).

        - `pixelization`: A grid of (y,x) coordinates which again align with the centre of every image pixel of
        the image data. This grid is used specifically for pixelizations computed via the `inversion` module, which
        can benefit from using different oversampling schemes than the normal grid.

        - `blurring`: A grid of (y,x) coordinates which are used to compute the blurring of the image data. This image
        contains the light of all pixels that are masked, but are close enough to the unmasked pixels that their light
        will be convolved into the unmasked pixels by the PSF.

        Every grid has its own `over_sampling` class which defines how over sampling is performed for these grids and
        is described in the corresponding `over_sampling.py` module.

        This is used in the project PyAutoGalaxy to load imaging data of a galaxy and fit it with galaxy light profiles.
        It is used in PyAutoLens to load imaging data of a strong lens and fit it with a lens model.

        Parameters
        ----------
        over_sample_size_lp
            The over sampling scheme size, which divides the grid into a sub grid of smaller pixels when computing
            values (e.g. images) from the grid to approximate the 2D line integral of the amount of light that falls
            into each pixel.
        over_sample_size_pixelization
            How over sampling is performed for the grid which is associated with a pixelization, which is therefore
            passed into the calculations performed in the `inversion` module.
        psf
            The Point Spread Function kernel of the image which accounts for diffraction due to the telescope optics
            via 2D convolution.
        """
        self.mask = mask
        self.over_sample_size_lp = over_sample_size_lp
        self.over_sample_size_pixelization = over_sample_size_pixelization
        self.psf = psf

    @cached_property
    def lp(self) -> Union[Grid1D, Grid2D]:
        """
        Returns the grid of (y,x) Cartesian coordinates at the centre of every pixel in the masked data, which is used
        to perform most normal calculations (e.g. evaluating the amount of light that falls in an pixel from a light
        profile).

        This grid is computed based on the mask, in particular its pixel-scale and sub-grid size.

        Returns
        -------
        The (y,x) coordinates of every pixel in the data.
        """
        return Grid2D.from_mask(
            mask=self.mask,
            over_sample_size=self.over_sample_size_lp,
        )

    @cached_property
    def pixelization(self) -> Grid2D:
        """
        Returns the grid of (y,x) Cartesian coordinates of every pixel in the masked data which is used
        specifically for calculations associated with a pixelization.

        The `pixelization` grid is identical to the `uniform` grid but often uses a different over sampling scheme
        when performing calculations. For example, the pixelization may benefit from using a a higher `sub_size` than
        the `uniform` grid, in order to better prevent aliasing effects.

        This grid is computed based on the mask, in particular its pixel-scale and sub-grid size.

        Returns
        -------
        The (y,x) coordinates of every pixel in the data, used for pixelization / inversion calculations.
        """
        return Grid2D.from_mask(
            mask=self.mask,
            over_sample_size=self.over_sample_size_pixelization,
        )

    @cached_property
    def blurring(self) -> Optional[Grid2D]:
        """
        Returns a blurring-grid from a mask and the 2D shape of the PSF kernel.

        A blurring grid consists of all pixels that are masked (and therefore have their values set to (0.0, 0.0)),
        but are close enough to the unmasked pixels that their values will be convolved into the unmasked those pixels.
        This when computing images from light profile objects.

        This uses lazy allocation such that the calculation is only performed when the blurring grid is used, ensuring
        efficient set up of the `Imaging` class.

        Returns
        -------
        The blurring grid given the mask of the imaging data.
        """

        if self.psf is None:
            return None

        return self.lp.blurring_grid_via_kernel_shape_from(
            kernel_shape_native=self.psf.shape_native,
        )

    @cached_property
    def border_relocator(self) -> BorderRelocator:
        return BorderRelocator(
            mask=self.mask, sub_size=self.over_sample_size_pixelization
        )


class GridsInterface:
    def __init__(
        self,
        lp=None,
        pixelization=None,
        blurring=None,
        border_relocator=None,
    ):
        self.lp = lp
        self.pixelization = pixelization
        self.blurring = blurring
        self.border_relocator = border_relocator
