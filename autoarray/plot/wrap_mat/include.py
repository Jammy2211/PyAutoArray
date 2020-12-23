from autoconf import conf


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

        self.origin = self.load_include(value=origin, name="origin")
        self.mask = self.load_include(value=mask, name="mask")
        self.grid = self.load_include(value=grid, name="grid")
        self.border = self.load_include(value=border, name="border")
        self.inversion_pixelization_grid = self.load_include(
            value=inversion_pixelization_grid, name="inversion_pixelization_grid"
        )
        self.inversion_grid = self.load_include(
            value=inversion_grid, name="inversion_grid"
        )
        self.inversion_border = self.load_include(
            value=inversion_border, name="inversion_border"
        )
        self.inversion_image_pixelization_grid = self.load_include(
            value=inversion_image_pixelization_grid,
            name="inversion_image_pixelization_grid",
        )
        self.parallel_overscan = self.load_include(
            value=parallel_overscan, name="parallel_overscan"
        )
        self.serial_prescan = self.load_include(
            value=serial_prescan, name="serial_prescan"
        )
        self.serial_overscan = self.load_include(
            value=serial_overscan, name="serial_overscan"
        )

    @staticmethod
    def load_include(value, name):
        if value is not None:
            """
            Let is be known that Jam did this - I merely made this horror more efficient
            """
            return value
        return conf.instance["visualize"]["general"]["include"][name]

    def grid_from_grid(self, grid):

        if self.grid:
            return grid
        else:
            return None

    def mask_from_grid(self, grid):

        if self.mask:
            return grid.mask
        else:
            return None

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

    def parallel_overscan_from_frame(self, frame):

        if self.parallel_overscan:
            return frame.scans.parallel_overscan
        else:
            return None

    def serial_prescan_from_frame(self, frame):

        if self.serial_prescan:
            return frame.scans.serial_prescan
        else:
            return None

    def serial_overscan_from_frame(self, frame):

        if self.serial_overscan:
            return frame.scans.serial_overscan
        else:
            return None
