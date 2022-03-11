from typing import Optional

from autoconf import conf


class AbstractInclude:
    def __init__(self, origin: Optional[bool] = None, mask: Optional[bool] = None):
        """
        Sets which `Visuals` are included on a figure that is plotted using a `Plotter`.

        The `Include` object is used to extract the visuals of the plotted data structure (e.g. `Array2D`, `Grid2D`) so
        they can be used in plot functions. Only visuals with a `True` entry in the `Include` object are extracted and t
        plotted.

        If an entry is not input into the class (e.g. it retains its default entry of `None`) then the bool is
        loaded from the `config/visualize/include.ini` config file. This means the default visuals of a project
        can be specified in a config file.

        Parameters
        ----------
        origin
            If `True`, the `origin` of the plotted data structure (e.g. `Array2D`, `Grid2D`)  is included on the figure.
        mask
            if `True`, the `mask` of the plotted data structure (e.g. `Array2D`, `Grid2D`)  is included on the figure.
        """

        self._origin = origin
        self._mask = mask

    def load(self, value, name):
        if value is True:
            return True
        elif value is False:
            return False
        elif value is None:
            return conf.instance["visualize"]["include"][self.section][name]

    @property
    def section(self):
        raise NotImplementedError

    @property
    def origin(self):
        return self.load(value=self._origin, name="origin")

    @property
    def mask(self):
        return self.load(value=self._mask, name="mask")


class Include1D(AbstractInclude):
    def __init__(self, origin: Optional[bool] = None, mask: Optional[bool] = None):
        """
        Sets which `Visuals1D` are included on a figure plotting 1D data that is plotted using a `Plotter1D`.

        The `Include` object is used to extract the visuals of the plotted 1D data structures so they can be used in 
        plot functions. Only visuals with a `True` entry in the `Include` object are extracted and plotted.

        If an entry is not input into the class (e.g. it retains its default entry of `None`) then the bool is
        loaded from the `config/visualize/include.ini` config file. This means the default visuals of a project
        can be specified in a config file.

        Parameters
        ----------
        origin
            If `True`, the `origin` of the plotted data structure (e.g. `Line`)  is included on the figure.
        mask
            if `True`, the `mask` of the plotted data structure (e.g. `Line`)  is included on the figure.
        """
        super().__init__(origin=origin, mask=mask)

    @property
    def section(self):
        return "include_1d"


class Include2D(AbstractInclude):
    def __init__(
        self,
        origin: Optional[bool] = None,
        mask: Optional[bool] = None,
        border: Optional[bool] = None,
        grid: Optional[bool] = None,
        mapper_data_pixelization_grid: Optional[bool] = None,
        mapper_source_pixelization_grid: Optional[bool] = None,
        mapper_source_grid_slim: Optional[bool] = None,
        parallel_overscan: Optional[bool] = None,
        serial_prescan: Optional[bool] = None,
        serial_overscan: Optional[bool] = None,
    ):
        """
        Sets which `Visuals2D` are included on a figure plotting 2D data that is plotted using a `Plotter`.

        The `Include` object is used to extract the visuals of the plotted 2D data structures so they can be used in 
        plot functions. Only visuals with a `True` entry in the `Include` object are extracted and plotted.

        If an entry is not input into the class (e.g. it retains its default entry of `None`) then the bool is
        loaded from the `config/visualize/include.ini` config file. This means the default visuals of a project
        can be specified in a config file.

        Parameters
        ----------
        origin
            If `True`, the `origin` of the plotted data structure (e.g. `Array2D`, `Grid2D`)  is included on the figure.
        mask
            if `True`, the `mask` of the plotted data structure (e.g. `Array2D`, `Grid2D`)  is included on the figure.
        border
            If `True`, the `border` of the plotted data structure (e.g. `Array2D`, `Grid2D`)  is included on the figure.
        mapper_data_pixelization_grid
            If `True`, the pixelization grid in the data plane of a plotted `Mapper` is included on the figure.
        mapper_source_pixelization_grid
            If `True`, the pixelization grid in the source plane of a plotted `Mapper` is included on the figure.
        parallel_overscan
            If `True`, the parallel overscan of a plotted `Frame2D` is included on the figure.
        serial_prescan
            If `True`, the serial prescan of a plotted `Frame2D` is included on the figure.
        serial_overscan
            If `True`, the serial overscan of a plotted `Frame2D` is included on the figure.
        """

        super().__init__(origin=origin, mask=mask)

        self._border = border
        self._grid = grid
        self._mapper_data_pixelization_grid = mapper_data_pixelization_grid
        self._mapper_source_pixelization_grid = mapper_source_pixelization_grid
        self._mapper_source_grid_slim = mapper_source_grid_slim
        self._parallel_overscan = parallel_overscan
        self._serial_prescan = serial_prescan
        self._serial_overscan = serial_overscan

    @property
    def section(self):
        return "include_2d"

    @property
    def border(self):
        return self.load(value=self._border, name="border")

    @property
    def grid(self):
        return self.load(value=self._grid, name="grid")

    @property
    def mapper_data_pixelization_grid(self):
        return self.load(
            value=self._mapper_data_pixelization_grid,
            name="mapper_data_pixelization_grid",
        )

    @property
    def mapper_source_pixelization_grid(self):
        return self.load(
            value=self._mapper_source_pixelization_grid,
            name="mapper_source_pixelization_grid",
        )

    @property
    def mapper_source_grid_slim(self):
        return self.load(
            value=self._mapper_source_grid_slim, name="mapper_source_grid_slim"
        )

    @property
    def parallel_overscan(self):
        return self.load(value=self._parallel_overscan, name="parallel_overscan")

    @property
    def serial_prescan(self):
        return self.load(value=self._serial_prescan, name="serial_prescan")

    @property
    def serial_overscan(self):
        return self.load(value=self._serial_overscan, name="serial_overscan")
