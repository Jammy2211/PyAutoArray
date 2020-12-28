from autoconf import conf
from autoarray.structures import arrays, grids, lines as l, vector_fields
from autoarray.plot.plotter import visuals as ps
from functools import wraps


def include_key_from_dictionary(dictionary):
    include_key = None

    for key, value in dictionary.items():
        if isinstance(value, Include):
            include_key = key

    return include_key


def set_include(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        include_key = include_key_from_dictionary(dictionary=kwargs)

        if include_key is not None:
            include = kwargs[include_key]
        else:
            include = Include()
            include_key = "include"

        kwargs[include_key] = include

        return func(*args, **kwargs)

    return wrapper


class Include:
    def __init__(
        self,
        origin=None,
        mask=None,
        grid=None,
        border=None,
        inversion_pixelization_grid=None,
        inversion_grid=None,
        inversion_border=None,
        inversion_image_pixelization_grid=None,
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
    ):

        section = conf.instance["visualize"]["include"]["include"]

        self.origin = section["origin"] if origin is None else origin
        self.mask = section["mask"] if mask is None else mask
        self.grid = section["grid"] if grid is None else grid
        self.border = section["border"] if border is None else border
        self.inversion_pixelization_grid = (
            section["inversion_pixelization_grid"]
            if inversion_pixelization_grid is None
            else inversion_pixelization_grid
        )
        self.inversion_grid = (
            section["inversion_grid"] if inversion_grid is None else inversion_grid
        )
        self.inversion_border = (
            section["inversion_border"]
            if inversion_border is None
            else inversion_border
        )
        self.inversion_image_pixelization_grid = (
            section["inversion_image_pixelization_grid"]
            if inversion_image_pixelization_grid is None
            else inversion_image_pixelization_grid
        )
        self.parallel_overscan = (
            section["parallel_overscan"]
            if parallel_overscan is None
            else parallel_overscan
        )
        self.serial_prescan = (
            section["serial_prescan"] if serial_prescan is None else serial_prescan
        )
        self.serial_overscan = (
            section["serial_overscan"] if serial_overscan is None else serial_overscan
        )

    def visuals_from_structure(self, structure):

        origin = grids.GridIrregular(grid=[structure.origin]) if self.origin else None

        mask = structure.mask if self.mask else None

        border = (
            structure.mask.geometry.border_grid_sub_1.in_1d_binned
            if self.border
            else None
        )

        return ps.Visuals(origin=origin, mask=mask, border=border)

    def visuals_from_array(self, array):

        return self.visuals_from_structure(structure=array)

    def visuals_from_grid(self, grid):

        return self.visuals_from_structure(structure=grid)

    def visuals_from_frame(self, frame):

        visuals_structure = self.visuals_from_structure(structure=frame)

        parallel_overscan = (
            frame.scans.parallel_overscan if self.parallel_overscan else None
        )
        serial_prescan = frame.scans.serial_prescan if self.serial_prescan else None
        serial_overscan = frame.scans.serial_overscan if self.serial_overscan else None

        return visuals_structure + ps.Visuals(
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

    def mask_from_masked_dataset(self, masked_dataset):

        if self.mask:
            return masked_dataset.mask
        else:
            return None

    def mask_from_fit(self, fit):
        """Get the masks of the fit if the masks should be plotted on the fit.

        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractLensHyperFit
            The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
        include_mask : bool
            If `True`, the masks is plotted on the fit's datas.
        """
        if self.mask:
            return fit.mask
        else:
            return None

    def real_space_mask_from_fit(self, fit):
        """Get the masks of the fit if the masks should be plotted on the fit.

        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractLensHyperFit
            The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
        include_mask : bool
            If `True`, the masks is plotted on the fit's datas.
        """
        if self.mask:
            return fit.settings_masked_dataset.real_space_mask
        else:
            return None
