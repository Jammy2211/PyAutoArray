from __future__ import annotations

from astropy.io import fits
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union


from autoarray.mask.abstract_mask import Mask

from autoarray.mask.derive.grid_1d import DeriveGrid1D
from autoarray.mask.derive.mask_1d import DeriveMask1D
from autoarray.geometry.geometry_1d import Geometry1D
from autoarray.structures.abstract_structure import Structure
from autoarray.structures.arrays import array_1d_util

from autoarray import exc
from autoarray import type as ty

logging.basicConfig()
logger = logging.getLogger(__name__)


class Mask1D(Mask):
    @property
    def native(self) -> Structure:
        raise NotImplemented()

    def __init__(
        self,
        mask: Union[np.ndarray, List],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float,] = (0.0,),
        invert: bool = False,
    ):
        """
        A 1D mask, representing 1D data on a uniform line of pixels with equal spacing.

        When applied to 1D data it extracts or masks the unmasked image pixels corresponding to mask entries that
        are `False` or 0).

        The mask also defines the geometry of the 1D data structure it is paired to, for example how every pixel
        coordinate on the 1D line of data converts to physical units via the `pixel_scales` and `origin`
        parameters and a sub-grid which is used for performing calculations via super-sampling.

        Parameters
        ----------
        mask
            The ndarray of shape [total_pixels] containing the bool's representing the mask, where `False`
            signifies an entry is unmasked and used in calculations.
        pixel_scales
             The scaled units to pixel units conversion factor of each pixel.
        origin
            The x origin of the mask's coordinate system in scaled units.
        """

        if type(mask) is list:
            mask = np.asarray(mask).astype("bool")

        if invert:
            mask = np.invert(mask)

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales,)

        if len(mask.shape) != 1:
            raise exc.MaskException("The input mask is not a one dimensional array")

        # noinspection PyArgumentList
        super().__init__(
            mask=mask,
            pixel_scales=pixel_scales,
            origin=origin,
        )

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj=obj)

        if isinstance(obj, Mask1D):
            pass
        else:
            self.origin = (0.0,)

    @property
    def geometry(self) -> Geometry1D:
        """
        Return the 1D geometry of the mask, representing its uniform rectangular grid of (x) coordinates defined by
        its ``shape_native``.
        """
        return Geometry1D(
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    @property
    def derive_mask(self) -> DeriveMask1D:
        return DeriveMask1D(mask=self)

    @property
    def derive_grid(self) -> DeriveGrid1D:
        return DeriveGrid1D(mask=self)

    @classmethod
    def all_false(
        cls,
        shape_slim,
        pixel_scales: ty.PixelScales,
        origin: Tuple[float] = (0.0,),
        invert: bool = False,
    ) -> "Mask1D":
        """
        Setup a 1D mask where all pixels are unmasked.

        Parameters
        ----------
        shape_slim
            The (y,x) shape of the mask in units of pixels.
        pixel_scales
            The scaled units to pixel units conversion factor of each pixel.
        """
        return cls(
            mask=np.full(shape=shape_slim, fill_value=False),
            pixel_scales=pixel_scales,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def from_fits(
        cls,
        file_path: Union[Path, str],
        pixel_scales: ty.PixelScales,
        hdu: int = 0,
        origin: Tuple[float] = (0.0,),
    ) -> "Mask1D":
        """
        Loads the 1D mask from a .fits file.

        Parameters
        ----------
        file_path
            The full path of the fits file.
        hdu
            The HDU number in the fits file containing the image image.
        pixel_scales
            The scaled units to pixel units conversion factor of each pixel.
        """

        return cls(
            array_1d_util.numpy_array_1d_via_fits_from(file_path=file_path, hdu=hdu),
            pixel_scales=pixel_scales,
            origin=origin,
        )

    @classmethod
    def from_primary_hdu(
        cls,
        primary_hdu: fits.PrimaryHDU,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "Mask1D":
        """
        Returns an ``Mask1D`` by from a `PrimaryHDU` object which has been loaded via `astropy.fits`

        This assumes that the `header` of the `PrimaryHDU` contains an entry named `PIXSCALE` which gives the
        pixel-scale of the array.

        For a full description of ``Mask1D`` objects, including a description of the ``slim`` and ``native`` attribute
        used by the API, see
        the :meth:`Mask1D class API documentation <autoarray.structures.arrays.uniform_1d.AbstractMask1D.__new__>`.

        Parameters
        ----------
        primary_hdu
            The `PrimaryHDU` object which has already been loaded from a .fits file via `astropy.fits` and contains
            the array data and the pixel-scale in the header with an entry named `PIXSCALE`.
        origin
            The (y,x) scaled units origin of the coordinate system.

        Examples
        --------

        .. code-block:: python

            from astropy.io import fits
            import autoarray as aa

            primary_hdu = fits.open("path/to/file.fits")

            array_1d = aa.Mask1D.from_primary_hdu(
                primary_hdu=primary_hdu,
            )
        """
        return cls(
            mask=primary_hdu.data.astype("bool"),
            pixel_scales=primary_hdu.header["PIXSCALE"],
            origin=origin,
        )

    @property
    def shape_native(self) -> Tuple[int]:
        return self.shape

    @property
    def shape_slim(self) -> Tuple[int]:
        return self.shape

    @property
    def hdu_for_output(self) -> fits.PrimaryHDU:
        """
        The mask as a HDU object, which can be output to a .fits file.

        The header of the HDU is used to store the `pixel_scale` of the array, which is used by the `Array1D.from_hdu`.

        This method is used in other projects (E.g. PyAutoGalaxy, PyAutoLens) to conveniently output the array to .fits
        files.

        Returns
        -------
        The HDU containing the data and its header which can then be written to .fits.
        """
        return array_1d_util.hdu_for_output_from(
            array_1d=self.astype("float"), header_dict=self.pixel_scale_header
        )

    def output_to_fits(self, file_path: Union[Path, str], overwrite: bool = False):
        """
        Write the 1D mask to a .fits file.

        Parameters
        ----------
        file_path
            The full path of the file that is output, including the file name and .fits extension.
        overwrite
            If `True` and a file already exists with the input file_path the .fits file is overwritten. If `False`,
            an error is raised.

        Returns
        -------
        None

        Examples
        --------
        mask = Mask1D(mask=np.full(shape=(5,), fill_value=False))
        mask.output_to_fits(file_path='/path/to/file/filename.fits', overwrite=True)
        """
        array_1d_util.numpy_array_1d_to_fits(
            array_1d=self.astype("float"),
            file_path=file_path,
            overwrite=overwrite,
            header_dict=self.pixel_scale_header,
        )
