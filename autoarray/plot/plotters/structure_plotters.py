from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot as mp
from autoarray.structures.arrays.one_d import array_1d
from autoarray.structures.arrays.two_d import array_2d
from autoarray.structures.frames import frames
from autoarray.structures.grids.two_d import grid_2d
from autoarray.structures.grids.two_d import grid_2d_irregular
from autoarray.inversion import mappers
import typing


class Array2DPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        array: array_2d.Array2D,
        mat_plot_2d: mp.MatPlot2D = mp.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):

        super().__init__(
            visuals_2d=visuals_2d, include_2d=include_2d, mat_plot_2d=mat_plot_2d
        )

        self.array = array

    @property
    def visuals_with_include_2d(self):
        """
        Extracts from an `Array2D` attributes that can be plotted and returns them in a `Visuals` object.

        Only attributes already in `self.visuals_2d` or with `True` entries in the `Include` object are extracted
        for plotting.

        From an `Array2D` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the structure's coordinate system.
        - mask: the mask of the structure.
        - border: the border of the structure's mask.

        Parameters
        ----------
        array : array_2d.Array2D
            The array whose attributes are extracted for plotting.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """
        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin", grid_2d_irregular.Grid2DIrregular(grid=[self.array.origin])
            ),
            mask=self.extract_2d("mask", self.array.mask),
            border=self.extract_2d(
                "border", self.array.mask.border_grid_sub_1.binned
            ),
        )

    def figure(self):

        self.mat_plot_2d.plot_array(
            array=self.array,
            visuals_2d=self.visuals_with_include_2d,
            auto_labels=mp.AutoLabels(title="Array2D", filename="array"),
        )


class Frame2DPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        frame: frames.Frame2D,
        mat_plot_2d: mp.MatPlot2D = mp.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):
        super().__init__(
            visuals_2d=visuals_2d, include_2d=include_2d, mat_plot_2d=mat_plot_2d
        )

        self.frame = frame

    @property
    def visuals_with_include_2d(self):
        """
        Extracts from a `Frame2D` attributes that can be plotted and return them in a `Visuals` object.

        Only attributes already in `self.visuals_2d` or with `True` entries in the `Include` object are extracted
        for plotting.

        From an `Frame2D` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the structure's coordinate system.
        - mask: the mask of the structure.
        - border: the border of the structure's mask.
        - parallel_overscan: the parallel overscan of the frame data.
        - serial_prescan: the serial prescan of the frame data.
        - serial_overscan: the serial overscan of the frame data.

        Parameters
        ----------
        frame : frames.Frame2D
            The frame whose attributes are extracted for plotting.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """
        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin", grid_2d_irregular.Grid2DIrregular(grid=[self.frame.origin])
            ),
            mask=self.extract_2d("mask", self.frame.mask),
            border=self.extract_2d(
                "border", self.frame.mask.border_grid_sub_1.binned
            ),
            parallel_overscan=self.extract_2d(
                "parallel_overscan", self.frame.scans.parallel_overscan
            ),
            serial_prescan=self.extract_2d(
                "serial_prescan", self.frame.scans.serial_prescan
            ),
            serial_overscan=self.extract_2d(
                "serial_overscan", self.frame.scans.serial_overscan
            ),
        )

    def figure(self):

        self.mat_plot_2d.plot_frame(
            frame=self.frame,
            visuals_2d=self.visuals_with_include_2d,
            auto_labels=mp.AutoLabels(title="Frame2D", filename="frame"),
        )


class Grid2DPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        grid: grid_2d.Grid2D,
        mat_plot_2d: mp.MatPlot2D = mp.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):
        super().__init__(
            visuals_2d=visuals_2d, include_2d=include_2d, mat_plot_2d=mat_plot_2d
        )

        self.grid = grid

    @property
    def visuals_with_include_2d(self):
        """
        Extracts from a `Grid2D` attributes that can be plotted and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `Grid2D` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the grid's coordinate system.
        - mask: the mask of the grid.
        - border: the border of the grid's mask.

        Parameters
        ----------
        grid : abstract_grid_2d.AbstractGrid2D
            The grid whose attributes are extracted for plotting.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """
        if not isinstance(self.grid, grid_2d.Grid2D):
            return self.visuals_2d

        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin", grid_2d_irregular.Grid2DIrregular(grid=[self.grid.origin])
            )
        )

    def figure(self, color_array=None):

        self.mat_plot_2d.plot_grid(
            grid=self.grid,
            visuals_2d=self.visuals_with_include_2d,
            auto_labels=mp.AutoLabels(title="Grid2D", filename="grid"),
            color_array=color_array,
        )


class MapperPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        mapper: typing.Union[mappers.MapperRectangular, mappers.MapperVoronoi],
        mat_plot_2d: mp.MatPlot2D = mp.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):
        super().__init__(
            visuals_2d=visuals_2d, include_2d=include_2d, mat_plot_2d=mat_plot_2d
        )

        self.mapper = mapper

    @property
    def visuals_data_with_include_2d(self):
        """
        Extracts from a `Mapper` attributes that can be plotted for figures in its data-plane (e.g. the reconstructed
        data) and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `Mapper` the following attributes can be extracted for plotting in the data-plane:

        - origin: the (y,x) origin of the `Array2D`'s coordinate system in the data plane.
        - mask : the `Mask` defined in the data-plane containing the data that is used by the `Mapper`.
        - mapper_data_pixelization_grid: the `Mapper`'s pixelization grid in the data-plane.
        - mapper_border_grid: the border of the `Mapper`'s full grid in the data-plane.

        Parameters
        ----------
        mapper : mappers.Mapper
            The mapper whose data-plane attributes are extracted for plotting.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """
        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin",
                grid_2d_irregular.Grid2DIrregular(
                    grid=[self.mapper.source_grid_slim.mask.origin]
                ),
            ),
            mask=self.extract_2d("mask", self.mapper.source_grid_slim.mask),
            border=self.extract_2d(
                "border",
                self.mapper.source_grid_slim.mask.border_grid_sub_1.binned,
            ),
            pixelization_grid=self.extract_2d(
                "pixelization_grid",
                self.mapper.data_pixelization_grid,
                "mapper_data_pixelization_grid",
            ),
        )

    @property
    def visuals_source_with_include_2d(self):
        """
        Extracts from a `Mapper` attributes that can be plotted for figures in its source-plane (e.g. the reconstruction
        and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `Mapper` the following attributes can be extracted for plotting in the source-plane:

        - origin: the (y,x) origin of the coordinate system in the source plane.
        - mapper_source_pixelization_grid: the `Mapper`'s pixelization grid in the source-plane.
        - mapper_source_grid_slim: the `Mapper`'s full grid in the source-plane.
        - mapper_border_grid: the border of the `Mapper`'s full grid in the data-plane.

        Parameters
        ----------
        mapper : mappers.Mapper
            The mapper whose source-plane attributes are extracted for plotting.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """

        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin",
                grid_2d_irregular.Grid2DIrregular(
                    grid=[self.mapper.source_pixelization_grid.origin]
                ),
            ),
            grid=self.extract_2d(
                "grid", self.mapper.source_grid_slim, "mapper_source_grid_slim"
            ),
            border=self.extract_2d(
                "border", self.mapper.source_grid_slim.sub_border_grid
            ),
            pixelization_grid=self.extract_2d(
                "pixelization_grid",
                self.mapper.source_pixelization_grid,
                "mapper_source_pixelization_grid",
            ),
        )

    def figure(self, source_pixelilzation_values=None):

        self.mat_plot_2d.plot_mapper(
            mapper=self.mapper,
            visuals_2d=self.visuals_source_with_include_2d,
            source_pixelilzation_values=source_pixelilzation_values,
            auto_labels=mp.AutoLabels(title="Mapper", filename="mapper"),
        )

    def subplot_image_and_mapper(self, image):

        self.open_subplot_figure(number_subplots=2)

        self.mat_plot_2d.plot_array(
            array=image,
            visuals_2d=self.visuals_data_with_include_2d,
            auto_labels=mp.AutoLabels(title="Image"),
        )

        if self.visuals_2d.pixelization_indexes is not None:

            indexes = self.mapper.slim_indexes_from_pixelization_indexes(
                pixelization_indexes=self.visuals_2d.pixelization_indexes
            )

            self.mat_plot_2d.index_scatter.scatter_grid_indexes(
                grid=self.mapper.source_grid_slim.mask.masked_grid, indexes=indexes
            )

        self.figure()

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_image_and_mapper"
        )
        self.close_subplot_figure()


class YX1DPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        y,
        x,
        mat_plot_1d: mp.MatPlot1D = mp.MatPlot1D(),
        visuals_1d: vis.Visuals1D = vis.Visuals1D(),
        include_1d: inc.Include1D = inc.Include1D(),
    ):

        super().__init__(
            visuals_1d=visuals_1d, include_1d=include_1d, mat_plot_1d=mat_plot_1d
        )

        self.y = y
        self.x = x

    @property
    def visuals_with_include_1d(self) -> vis.Visuals1D:

        return self.visuals_1d + self.visuals_1d.__class__(
            origin=self.extract_1d("origin", self.x.origin),
            mask=self.extract_1d("mask", self.x.mask),
        )

    def figure(self,):

        self.mat_plot_1d.plot_yx(
            y=self.y, x=self.x, visuals_1d=self.visuals_1d, auto_labels=mp.AutoLabels()
        )
