import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Union

from autoconf import conf

from autoarray.inversion.pixelization.mappers.rectangular import (
    MapperRectangularNoInterp,
)
from autoarray.inversion.pixelization.mappers.delaunay import MapperDelaunay
from autoarray.inversion.pixelization.mappers.voronoi import MapperVoronoiNoInterp
from autoarray.plot.mat_plot.abstract import AbstractMatPlot
from autoarray.plot.auto_labels import AutoLabels
from autoarray.plot.mat_wrap.visuals import Visuals2D
from autoarray.structures.arrays.uniform_2d import Array2D

from autoarray.structures.arrays import array_2d_util

from autoarray import exc
from autoarray.plot.wrap import base as wb
from autoarray.plot.wrap import two_d as w2d


class MatPlot2D(AbstractMatPlot):
    def __init__(
        self,
        units: Optional[wb.Units] = None,
        figure: Optional[wb.Figure] = None,
        axis: Optional[wb.Axis] = None,
        cmap: Optional[wb.Cmap] = None,
        colorbar: Optional[wb.Colorbar] = None,
        colorbar_tickparams: Optional[wb.ColorbarTickParams] = None,
        tickparams: Optional[wb.TickParams] = None,
        yticks: Optional[wb.YTicks] = None,
        xticks: Optional[wb.XTicks] = None,
        title: Optional[wb.Title] = None,
        ylabel: Optional[wb.YLabel] = None,
        xlabel: Optional[wb.XLabel] = None,
        text: Optional[Union[wb.Text, List[wb.Text]]] = None,
        legend: Optional[wb.Legend] = None,
        output: Optional[wb.Output] = None,
        array_overlay: Optional[w2d.ArrayOverlay] = None,
        grid_scatter: Optional[w2d.GridScatter] = None,
        grid_plot: Optional[w2d.GridPlot] = None,
        grid_errorbar: Optional[w2d.GridErrorbar] = None,
        vector_yx_quiver: Optional[w2d.VectorYXQuiver] = None,
        patch_overlay: Optional[w2d.PatchOverlay] = None,
        interpolated_reconstruction: Optional[w2d.InterpolatedReconstruction] = None,
        voronoi_drawer: Optional[w2d.VoronoiDrawer] = None,
        origin_scatter: Optional[w2d.OriginScatter] = None,
        mask_scatter: Optional[w2d.MaskScatter] = None,
        border_scatter: Optional[w2d.BorderScatter] = None,
        positions_scatter: Optional[w2d.PositionsScatter] = None,
        index_scatter: Optional[w2d.IndexScatter] = None,
        mesh_grid_scatter: Optional[w2d.MeshGridScatter] = None,
        parallel_overscan_plot: Optional[w2d.ParallelOverscanPlot] = None,
        serial_prescan_plot: Optional[w2d.SerialPrescanPlot] = None,
        serial_overscan_plot: Optional[w2d.SerialOverscanPlot] = None,
    ):
        """
        Visualizes 2D data structures (e.g an `Array2D`, `Grid2D`, `VectorField`, etc.) using Matplotlib.

        The `Plotter` is passed objects from the `wrap` package which wrap matplotlib plot functions and customize
        the appearance of the plots of the data structure. If the values of these matplotlib wrapper objects are not
        manually specified, they assume the default values provided in the `config.visualize.mat_*` `.ini` config files.

        The following 2D data structures can be plotted using the following matplotlib functions:

        - `Array2D`:, using `plt.imshow`.
        - `Grid2D`: using `plt.scatter`.
        - `Line`: using `plt.plot`, `plt.semilogy`, `plt.loglog` or `plt.scatter`.
        - `VectorField`: using `plt.quiver`.
        - `RectangularMapper`: using `plt.imshow`.
        - `MapperVoronoiNoInterp`: using `plt.fill`.

        Parameters
        ----------
        units
            The units of the figure used to plot the data structure which sets the y and x ticks and labels.
        figure
            Opens the matplotlib figure before plotting via `plt.figure` and closes it once plotting is complete
            via `plt.close`.
        axis
            Sets the extent of the figure axis via `plt.axis` and allows for a manual axis range.
        cmap
            Customizes the colormap of the plot and its normalization via matplotlib `colors` objects such
            as `colors.Normalize` and `colors.LogNorm`.
        colorbar
            Plots the colorbar of the plot via `plt.colorbar` and customizes its tick labels and values using method
            like `cb.set_yticklabels`.
        colorbar_tickparams
            Customizes the yticks of the colorbar plotted via `plt.colorbar`.
        tickparams
            Customizes the appearances of the y and x ticks on the plot, (e.g. the fontsize), using `plt.tick_params`.
        yticks
            Sets the yticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.yticks`.
        xticks
            Sets the xticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.xticks`.
        title
            Sets the figure title and customizes its appearance using `plt.title`.
        ylabel
            Sets the figure ylabel and customizes its appearance using `plt.ylabel`.
        xlabel
            Sets the figure xlabel and customizes its appearance using `plt.xlabel`.
        legend
            Sets whether the plot inclues a legend and customizes its appearance and labels using `plt.legend`.
        output
            Sets if the figure is displayed on the user's screen or output to `.png` using `plt.show` and `plt.savefig`
        array_overlay
            Overlays an input `Array2D` over the figure using `plt.imshow`.
        grid_scatter
            Scatters a `Grid2D` of (y,x) coordinates over the figure using `plt.scatter`.
        grid_plot
            Plots lines of data (e.g. a y versus x plot via `plt.plot`, vertical lines via `plt.avxline`, etc.)
        vector_yx_quiver
            Plots a `VectorField` object using the matplotlib function `plt.quiver`.
        patch_overlay
            Overlays matplotlib `patches.Patch` objects over the figure, such as an `Ellipse`.
        voronoi_drawer
            Draws a colored Voronoi mesh of pixels using `plt.fill`.
        interpolated_reconstruction
            Draws a colored Delaunay mesh of pixels using `plt.fill`.
        origin_scatter
            Scatters the (y,x) origin of the data structure on the figure.
        mask_scatter
            Scatters an input `Mask2d` over the plotted data structure's figure.
        border_scatter
            Scatters the border of an input `Mask2d` over the plotted data structure's figure.
        positions_scatter
            Scatters specific (y,x) coordinates input as a `Grid2DIrregular` object over the figure.
        index_scatter
            Scatters specific coordinates of an input `Grid2D` based on input values of the `Grid2D`'s 1D or 2D indexes.
        mesh_grid_scatter
            Scatters the `PixelizationGrid` of a `Mesh` object.
        parallel_overscan_plot
            Plots the parallel overscan on an `Array2D` data structure representing a CCD imaging via `plt.plot`.
        serial_prescan_plot
            Plots the serial prescan on an `Array2D` data structure representing a CCD imaging via `plt.plot`.
        serial_overscan_plot
            Plots the serial overscan on an `Array2D` data structure representing a CCD imaging via `plt.plot`.
        """

        super().__init__(
            units=units,
            figure=figure,
            axis=axis,
            cmap=cmap,
            colorbar=colorbar,
            colorbar_tickparams=colorbar_tickparams,
            tickparams=tickparams,
            yticks=yticks,
            xticks=xticks,
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
            text=text,
            legend=legend,
            output=output,
        )

        self.array_overlay = array_overlay or w2d.ArrayOverlay(is_default=True)

        self.grid_scatter = grid_scatter or w2d.GridScatter(is_default=True)
        self.grid_plot = grid_plot or w2d.GridPlot(is_default=True)
        self.grid_errorbar = grid_errorbar or w2d.GridErrorbar(is_default=True)

        self.vector_yx_quiver = vector_yx_quiver or w2d.VectorYXQuiver(is_default=True)
        self.patch_overlay = patch_overlay or w2d.PatchOverlay(is_default=True)

        self.interpolated_reconstruction = (
            interpolated_reconstruction
            or w2d.InterpolatedReconstruction(is_default=True)
        )
        self.voronoi_drawer = voronoi_drawer or w2d.VoronoiDrawer(is_default=True)

        self.origin_scatter = origin_scatter or w2d.OriginScatter(is_default=True)
        self.mask_scatter = mask_scatter or w2d.MaskScatter(is_default=True)
        self.border_scatter = border_scatter or w2d.BorderScatter(is_default=True)
        self.positions_scatter = positions_scatter or w2d.PositionsScatter(
            is_default=True
        )
        self.index_scatter = index_scatter or w2d.IndexScatter(is_default=True)
        self.mesh_grid_scatter = mesh_grid_scatter or w2d.MeshGridScatter(
            is_default=True
        )

        self.parallel_overscan_plot = (
            parallel_overscan_plot or w2d.ParallelOverscanPlot(is_default=True)
        )
        self.serial_prescan_plot = serial_prescan_plot or w2d.SerialPrescanPlot(
            is_default=True
        )
        self.serial_overscan_plot = serial_overscan_plot or w2d.SerialOverscanPlot(
            is_default=True
        )

        self.is_for_subplot = False

    def plot_array(
        self,
        array: Array2D,
        visuals_2d: Visuals2D,
        auto_labels: AutoLabels,
        bypass: bool = False,
    ):
        """
        Plot an `Array2D` data structure as a figure using the matplotlib wrapper objects and tools.

        This `Array2D` is plotted using `plt.imshow`.

        Parameters
        -----------
        array
            The 2D array of data_type which is plotted.
        visuals_2d
            Contains all the visuals that are plotted over the `Array2D` (e.g. the origin, mask, grids, etc.).
        bypass
            If `True`, `plt.close` is omitted and the matplotlib figure remains open. This is used when making subplots.
        """

        if array is None or np.all(array == 0):
            return

        if array.pixel_scales is None and self.units.use_scaled:
            raise exc.ArrayException(
                "You cannot plot an array using its scaled unit_label if the input array does not have "
                "a pixel scales attribute."
            )

        array = array.binned

        if array.mask.is_all_false:
            buffer = 0
        else:
            buffer = 1

        if conf.instance["visualize"]["general"]["general"]["zoom_around_mask"]:

            extent = array.extent_of_zoomed_array(buffer=buffer)
            array = array.zoomed_around_mask(buffer=buffer)

        else:

            extent = array.extent

        ax = None

        if not self.is_for_subplot:
            fig, ax = self.figure.open()
        else:
            if not bypass:
                ax = self.setup_subplot()

        aspect = self.figure.aspect_from(shape_native=array.shape_native)
        norm_scale = self.cmap.norm_from(array=array)

        origin = conf.instance["visualize"]["general"]["general"]["imshow_origin"]

        plt.imshow(
            X=array.native,
            aspect=aspect,
            cmap=self.cmap.cmap,
            norm=norm_scale,
            extent=extent,
            origin=origin,
        )

        if visuals_2d.array_overlay is not None:
            self.array_overlay.overlay_array(
                array=visuals_2d.array_overlay, figure=self.figure
            )

        extent_axis = self.axis.config_dict.get("extent")

        if extent_axis is None:

            extent_axis = extent

        self.axis.set(extent=extent_axis)

        self.tickparams.set()

        self.yticks.set(
            array=array,
            min_value=extent_axis[2],
            max_value=extent_axis[3],
            units=self.units,
        )

        self.xticks.set(
            array=array,
            min_value=extent_axis[0],
            max_value=extent_axis[1],
            units=self.units,
        )

        self.title.set(auto_title=auto_labels.title)
        self.ylabel.set(units=self.units, include_brackets=True)
        self.xlabel.set(units=self.units, include_brackets=True)

        if not isinstance(self.text, list):
            self.text.set()
        else:
            [text.set() for text in self.text]

        if self.colorbar is not False:
            cb = self.colorbar.set(ax=ax)
            self.colorbar_tickparams.set(cb=cb)

        grid_indexes = None

        if visuals_2d.indexes is not None or visuals_2d.pix_indexes is not None:
            grid_indexes = array.mask.masked_grid

        visuals_2d.plot_via_plotter(plotter=self, grid_indexes=grid_indexes)

        if not self.is_for_subplot and not bypass:
            self.output.to_figure(structure=array, auto_filename=auto_labels.filename)
            self.figure.close()

    def plot_grid(
        self,
        grid,
        visuals_2d: Visuals2D,
        auto_labels: AutoLabels,
        color_array=None,
        y_errors=None,
        x_errors=None,
        buffer=1.0,
    ):
        """Plot a grid of (y,x) Cartesian coordinates as a scatter plotter of points.

        Parameters
        -----------
        grid
            The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2).
        indexes
            A set of points that are plotted in a different colour for emphasis (e.g. to show the mappings between \
            different planes).
        """

        if not self.is_for_subplot:
            fig, ax = self.figure.open()
        else:
            ax = self.setup_subplot()

        if color_array is None:

            if y_errors is None and x_errors is None:
                self.grid_scatter.scatter_grid(grid=grid)
            else:
                self.grid_errorbar.errorbar_grid(
                    grid=grid, y_errors=y_errors, x_errors=x_errors
                )

        elif color_array is not None:

            cmap = plt.get_cmap(self.cmap.cmap)

            if y_errors is None and x_errors is None:
                self.grid_scatter.scatter_grid_colored(
                    grid=grid, color_array=color_array, cmap=cmap
                )
            else:
                self.grid_errorbar.errorbar_grid_colored(
                    grid=grid,
                    cmap=cmap,
                    color_array=color_array,
                    y_errors=y_errors,
                    x_errors=x_errors,
                )

            if self.colorbar is not None:

                colorbar = self.colorbar.set_with_color_values(
                    cmap=self.cmap.cmap, color_values=color_array, ax=ax
                )
                if colorbar is not None and self.colorbar_tickparams is not None:
                    self.colorbar_tickparams.set(cb=colorbar)

        self.title.set(auto_title=auto_labels.title)
        self.ylabel.set(units=self.units, include_brackets=True)
        self.xlabel.set(units=self.units, include_brackets=True)

        if not isinstance(self.text, list):
            self.text.set()
        else:
            [text.set() for text in self.text]

        extent = self.axis.config_dict.get("extent")

        if extent is None:

            extent = grid.extent + (buffer * grid.extent)

        self.axis.set(extent=extent, grid=grid)

        self.tickparams.set()

        if not self.axis.symmetric_around_centre:
            self.yticks.set(
                array=None, min_value=extent[2], max_value=extent[3], units=self.units
            )
            self.xticks.set(
                array=None, min_value=extent[0], max_value=extent[1], units=self.units
            )

        visuals_2d.plot_via_plotter(plotter=self, grid_indexes=grid)

        if not self.is_for_subplot:
            self.output.to_figure(structure=grid, auto_filename=auto_labels.filename)
            self.figure.close()

    def plot_mapper(
        self,
        mapper: Union[MapperRectangularNoInterp, MapperVoronoiNoInterp],
        visuals_2d: Visuals2D,
        auto_labels: AutoLabels,
        interpolate_to_uniform: bool = True,
        source_pixelilzation_values=None,
    ):

        if isinstance(mapper, MapperRectangularNoInterp):

            self._plot_rectangular_mapper(
                mapper=mapper,
                visuals_2d=visuals_2d,
                auto_labels=auto_labels,
                interpolate_to_uniform=interpolate_to_uniform,
                source_pixelilzation_values=source_pixelilzation_values,
            )

        elif isinstance(mapper, MapperDelaunay):
            self._plot_delaunay_mapper(
                mapper=mapper,
                visuals_2d=visuals_2d,
                auto_labels=auto_labels,
                interpolate_to_uniform=interpolate_to_uniform,
                source_pixelilzation_values=source_pixelilzation_values,
            )
        else:

            self._plot_voronoi_mapper(
                mapper=mapper,
                visuals_2d=visuals_2d,
                auto_labels=auto_labels,
                interpolate_to_uniform=interpolate_to_uniform,
                source_pixelilzation_values=source_pixelilzation_values,
            )

    def _plot_rectangular_mapper(
        self,
        mapper: MapperRectangularNoInterp,
        visuals_2d: Visuals2D,
        auto_labels: AutoLabels,
        interpolate_to_uniform: bool = True,
        source_pixelilzation_values=None,
    ):

        if source_pixelilzation_values is not None:

            solution_array_2d = array_2d_util.array_2d_native_from(
                array_2d_slim=source_pixelilzation_values,
                mask_2d=np.full(
                    fill_value=False, shape=mapper.source_mesh_grid.shape_native
                ),
                sub_size=1,
            )

            source_pixelilzation_values = Array2D.manual(
                array=solution_array_2d,
                sub_size=1,
                pixel_scales=mapper.source_mesh_grid.pixel_scales,
                origin=mapper.source_mesh_grid.origin,
            )

        extent = self.axis.config_dict.get("extent")
        extent = extent if extent is not None else mapper.source_mesh_grid.extent

        aspect_inv = self.figure.aspect_for_subplot_from(extent=extent)

        if not self.is_for_subplot:
            fig, ax = self.figure.open()
        else:
            ax = self.setup_subplot(aspect=aspect_inv)

        if source_pixelilzation_values is not None:
            self.plot_array(
                array=source_pixelilzation_values,
                visuals_2d=visuals_2d,
                auto_labels=auto_labels,
                bypass=True,
            )

        self.axis.set(extent=extent, grid=mapper.source_mesh_grid)

        self.yticks.set(
            array=None, min_value=extent[2], max_value=extent[3], units=self.units
        )
        self.xticks.set(
            array=None, min_value=extent[0], max_value=extent[1], units=self.units
        )

        if not isinstance(self.text, list):
            self.text.set()
        else:
            [text.set() for text in self.text]

        self.grid_plot.plot_rectangular_grid_lines(
            extent=mapper.source_mesh_grid.extent, shape_native=mapper.shape_native
        )

        self.title.set(auto_title=auto_labels.title)
        self.tickparams.set()
        self.ylabel.set(units=self.units, include_brackets=True)
        self.xlabel.set(units=self.units, include_brackets=True)

        visuals_2d.plot_via_plotter(
            plotter=self, grid_indexes=mapper.source_grid_slim, mapper=mapper
        )

        if not self.is_for_subplot:
            self.output.to_figure(structure=None, auto_filename=auto_labels.filename)
            self.figure.close()

    def _plot_delaunay_mapper(
        self,
        mapper: MapperDelaunay,
        visuals_2d: Visuals2D,
        auto_labels: AutoLabels,
        interpolate_to_uniform: bool = True,
        source_pixelilzation_values=None,
    ):

        extent = self.axis.config_dict.get("extent")
        extent = extent if extent is not None else mapper.source_mesh_grid.extent

        aspect_inv = self.figure.aspect_for_subplot_from(extent=extent)

        if not self.is_for_subplot:
            fig, ax = self.figure.open()
        else:
            ax = self.setup_subplot(aspect=aspect_inv)

        self.axis.set(extent=extent, grid=mapper.source_mesh_grid)

        self.tickparams.set()
        self.yticks.set(
            array=None, min_value=extent[2], max_value=extent[3], units=self.units
        )
        self.xticks.set(
            array=None, min_value=extent[0], max_value=extent[1], units=self.units
        )

        if not isinstance(self.text, list):
            self.text.set()
        else:
            [text.set() for text in self.text]

        self.interpolated_reconstruction.imshow_reconstruction(
            mapper=mapper,
            pixel_values=source_pixelilzation_values,
            cmap=self.cmap,
            colorbar=self.colorbar,
            colorbar_tickparams=self.colorbar_tickparams,
            aspect=aspect_inv,
            ax=ax,
        )

        self.title.set(auto_title=auto_labels.title)
        self.ylabel.set(units=self.units, include_brackets=True)
        self.xlabel.set(units=self.units, include_brackets=True)

        visuals_2d.plot_via_plotter(
            plotter=self, grid_indexes=mapper.source_grid_slim, mapper=mapper
        )

        if not self.is_for_subplot:
            self.output.to_figure(structure=None, auto_filename=auto_labels.filename)
            self.figure.close()

    def _plot_voronoi_mapper(
        self,
        mapper: MapperVoronoiNoInterp,
        visuals_2d: Visuals2D,
        auto_labels: AutoLabels,
        interpolate_to_uniform: bool = True,
        source_pixelilzation_values=None,
    ):

        extent = self.axis.config_dict.get("extent")
        extent = extent if extent is not None else mapper.source_mesh_grid.extent

        aspect_inv = self.figure.aspect_for_subplot_from(extent=extent)

        if not self.is_for_subplot:
            fig, ax = self.figure.open()
        else:
            ax = self.setup_subplot(aspect=aspect_inv)

        self.axis.set(extent=extent, grid=mapper.source_mesh_grid)

        plt.gca().set_aspect(aspect_inv)

        self.tickparams.set()
        self.yticks.set(
            array=None, min_value=extent[2], max_value=extent[3], units=self.units
        )
        self.xticks.set(
            array=None, min_value=extent[0], max_value=extent[1], units=self.units
        )

        if not isinstance(self.text, list):
            self.text.set()
        else:
            [text.set() for text in self.text]

        if not interpolate_to_uniform:

            self.voronoi_drawer.draw_voronoi_pixels(
                mapper=mapper,
                pixel_values=source_pixelilzation_values,
                cmap=self.cmap,
                colorbar=self.colorbar,
                colorbar_tickparams=self.colorbar_tickparams,
                ax=ax,
            )

        else:

            self.interpolated_reconstruction.imshow_reconstruction(
                mapper=mapper,
                pixel_values=source_pixelilzation_values,
                cmap=self.cmap,
                colorbar=self.colorbar,
                colorbar_tickparams=self.colorbar_tickparams,
                aspect=aspect_inv,
                ax=ax,
            )

        self.title.set(auto_title=auto_labels.title)
        self.ylabel.set(units=self.units, include_brackets=True)
        self.xlabel.set(units=self.units, include_brackets=True)

        visuals_2d.plot_via_plotter(
            plotter=self, grid_indexes=mapper.source_grid_slim, mapper=mapper
        )

        if not self.is_for_subplot:
            self.output.to_figure(structure=None, auto_filename=auto_labels.filename)
            self.figure.close()
