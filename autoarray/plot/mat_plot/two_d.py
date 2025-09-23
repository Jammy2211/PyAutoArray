import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Union

from autoconf import conf

from autoarray.inversion.pixelization.mappers.rectangular import (
    MapperRectangular,
)
from autoarray.inversion.pixelization.mappers.delaunay import MapperDelaunay
from autoarray.inversion.pixelization.mappers.voronoi import MapperVoronoi
from autoarray.mask.derive.zoom_2d import Zoom2D
from autoarray.plot.mat_plot.abstract import AbstractMatPlot
from autoarray.plot.auto_labels import AutoLabels
from autoarray.plot.visuals.two_d import Visuals2D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.arrays.rgb import Array2DRGB

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
        annotate: Optional[Union[wb.Annotate, List[wb.Annotate]]] = None,
        legend: Optional[wb.Legend] = None,
        output: Optional[wb.Output] = None,
        array_overlay: Optional[w2d.ArrayOverlay] = None,
        fill: Optional[w2d.Fill] = None,
        contour: Optional[w2d.Contour] = None,
        grid_scatter: Optional[w2d.GridScatter] = None,
        grid_plot: Optional[w2d.GridPlot] = None,
        grid_errorbar: Optional[w2d.GridErrorbar] = None,
        vector_yx_quiver: Optional[w2d.VectorYXQuiver] = None,
        patch_overlay: Optional[w2d.PatchOverlay] = None,
        interpolated_reconstruction: Optional[w2d.InterpolatedReconstruction] = None,
        delaunay_drawer: Optional[w2d.DelaunayDrawer] = None,
        voronoi_drawer: Optional[w2d.VoronoiDrawer] = None,
        origin_scatter: Optional[w2d.OriginScatter] = None,
        mask_scatter: Optional[w2d.MaskScatter] = None,
        border_scatter: Optional[w2d.BorderScatter] = None,
        positions_scatter: Optional[w2d.PositionsScatter] = None,
        index_scatter: Optional[w2d.IndexScatter] = None,
        index_plot: Optional[w2d.IndexPlot] = None,
        mesh_grid_scatter: Optional[w2d.MeshGridScatter] = None,
        parallel_overscan_plot: Optional[w2d.ParallelOverscanPlot] = None,
        serial_prescan_plot: Optional[w2d.SerialPrescanPlot] = None,
        serial_overscan_plot: Optional[w2d.SerialOverscanPlot] = None,
        use_log10: bool = False,
        plot_mask: bool = True,
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
        - `MapperVoronoi`: using `plt.fill`.

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
        text
            Sets any text on the figure and customizes its appearance using `plt.text`.
        annotate
            Sets any annotations on the figure and customizes its appearance using `plt.annotate`.
        legend
            Sets whether the plot inclues a legend and customizes its appearance and labels using `plt.legend`.
        output
            Sets if the figure is displayed on the user's screen or output to `.png` using `plt.show` and `plt.savefig`
        array_overlay
            Overlays an input `Array2D` over the figure using `plt.imshow`.
        fill
            Sets the fill of the figure using `plt.fill` and customizes its appearance, such as the color and alpha.
        contour
            Overlays contours of an input `Array2D` over the figure using `plt.contour`.
        grid_scatter
            Scatters a `Grid2D` of (y,x) coordinates over the figure using `plt.scatter`.
        grid_plot
            Plots lines of data (e.g. a y versus x plot via `plt.plot`, vertical lines via `plt.avxline`, etc.)
        vector_yx_quiver
            Plots a `VectorField` object using the matplotlib function `plt.quiver`.
        patch_overlay
            Overlays matplotlib `patches.Patch` objects over the figure, such as an `Ellipse`.
        delaunay_drawer
            Draws a colored Delaunay mesh of pixels using `plt.tripcolor`.
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
        use_log10
            If True, the plot has a log10 colormap, colorbar and contours showing the values.
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
            annotate=annotate,
            legend=legend,
            output=output,
        )

        self.array_overlay = array_overlay or w2d.ArrayOverlay(is_default=True)
        self.fill = fill or w2d.Fill(is_default=True)

        self.contour = contour or w2d.Contour(is_default=True)

        self.grid_scatter = grid_scatter or w2d.GridScatter(is_default=True)
        self.grid_plot = grid_plot or w2d.GridPlot(is_default=True)
        self.grid_errorbar = grid_errorbar or w2d.GridErrorbar(is_default=True)

        self.vector_yx_quiver = vector_yx_quiver or w2d.VectorYXQuiver(is_default=True)
        self.patch_overlay = patch_overlay or w2d.PatchOverlay(is_default=True)

        self.interpolated_reconstruction = (
            interpolated_reconstruction
            or w2d.InterpolatedReconstruction(is_default=True)
        )
        self.delaunay_drawer = delaunay_drawer or w2d.DelaunayDrawer(is_default=True)
        self.voronoi_drawer = voronoi_drawer or w2d.VoronoiDrawer(is_default=True)

        self.origin_scatter = origin_scatter or w2d.OriginScatter(is_default=True)
        self.mask_scatter = mask_scatter or w2d.MaskScatter(is_default=True)
        self.border_scatter = border_scatter or w2d.BorderScatter(is_default=True)
        self.positions_scatter = positions_scatter or w2d.PositionsScatter(
            is_default=True
        )
        self.index_scatter = index_scatter or w2d.IndexScatter(is_default=True)
        self.index_plot = index_plot or w2d.IndexPlot(is_default=True)
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

        self.use_log10 = use_log10
        self.plot_mask = plot_mask

        self.is_for_subplot = False

    def plot_array(
        self,
        array: Array2D,
        visuals_2d: Visuals2D,
        auto_labels: AutoLabels,
        grid_indexes=None,
        bypass: bool = False,
    ):
        """
        Plot an `Array2D` data structure as a figure using the matplotlib wrapper objects and tools.

        This `Array2D` is plotted using `plt.imshow`.

        Parameters
        ----------
        array
            The 2D array of data_type which is plotted.
        visuals_2d
            Contains all the visuals that are plotted over the `Array2D` (e.g. the origin, mask, grids, etc.).
        bypass
            If `True`, `plt.close` is omitted and the matplotlib figure remains open. This is used when making subplots.
        """

        if array is None or np.all(array == 0):
            return

        if self.use_log10 and (np.all(array == array[0]) or np.all(array < 0)):
            return

        if array.pixel_scales is None and self.units.use_scaled:
            raise exc.ArrayException(
                "You cannot plot an array using its scaled unit_label if the input array does not have "
                "a pixel scales attribute."
            )

        if conf.instance["visualize"]["general"]["general"]["zoom_around_mask"]:

            zoom = Zoom2D(mask=array.mask)

            buffer = 0 if array.mask.is_all_false else 1

            array = zoom.array_2d_from(array=array, buffer=buffer)

        extent = array.geometry.extent

        ax = None

        if not self.is_for_subplot:
            fig, ax = self.figure.open()
        else:
            if not bypass:
                ax = self.setup_subplot()

        aspect = self.figure.aspect_from(shape_native=array.shape_native)

        norm = self.cmap.norm_from(array=array.array, use_log10=self.use_log10)

        origin = conf.instance["visualize"]["general"]["general"]["imshow_origin"]

        if isinstance(array, Array2DRGB):

            plt.imshow(
                X=array.native.array,
                aspect=aspect,
                extent=extent,
                origin=origin,
            )

        else:

            plt.imshow(
                X=array.native.array,
                aspect=aspect,
                cmap=self.cmap.cmap,
                norm=norm,
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
            min_value=extent_axis[2],
            max_value=extent_axis[3],
            units=self.units,
            pixels=array.shape_native[0],
        )

        self.xticks.set(
            min_value=extent_axis[0],
            max_value=extent_axis[1],
            units=self.units,
            pixels=array.shape_native[1],
        )

        if isinstance(array, Array2DRGB):
            title = "RGB"
        else:
            title = auto_labels.title

        self.title.set(auto_title=title, use_log10=self.use_log10)
        self.ylabel.set()
        self.xlabel.set()

        if not isinstance(self.text, list):
            self.text.set()
        else:
            [text.set() for text in self.text]

        if not isinstance(self.annotate, list):
            self.annotate.set()
        else:
            [annotate.set() for annotate in self.annotate]

        if self.colorbar is not False:
            cb = self.colorbar.set(
                units=self.units,
                ax=ax,
                norm=norm,
                cb_unit=auto_labels.cb_unit,
                use_log10=self.use_log10,
            )
            self.colorbar_tickparams.set(cb=cb)

        if self.contour is not False:
            try:
                self.contour.set(array=array, extent=extent, use_log10=self.use_log10)
            except ValueError:
                pass

        if self.plot_mask and visuals_2d.mask is None:

            if not array.mask.is_all_false:

                self.mask_scatter.scatter_grid(grid=array.mask.derive_grid.edge.array)

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
        plot_grid_lines=False,
        plot_over_sampled_grid=False,
        buffer=0.1,
    ):
        """Plot a grid of (y,x) Cartesian coordinates as a scatter plotter of points.

        Parameters
        ----------
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

        if plot_over_sampled_grid:
            grid_plot = grid.over_sampled
        else:
            grid_plot = grid

        if color_array is None:
            if y_errors is None and x_errors is None:
                self.grid_scatter.scatter_grid(grid=grid_plot.array)
            else:
                self.grid_errorbar.errorbar_grid(
                    grid=grid_plot.array, y_errors=y_errors, x_errors=x_errors
                )

        elif color_array is not None:
            cmap = plt.get_cmap(self.cmap.cmap)

            if y_errors is None and x_errors is None:
                self.grid_scatter.scatter_grid_colored(
                    grid=grid.array, color_array=color_array, cmap=cmap
                )
            else:
                self.grid_errorbar.errorbar_grid_colored(
                    grid=grid.array,
                    cmap=cmap,
                    color_array=color_array,
                    y_errors=y_errors,
                    x_errors=x_errors,
                )

            if self.colorbar is not None:

                colorbar = self.colorbar.set_with_color_values(
                    units=self.units,
                    cmap=self.cmap.cmap,
                    color_values=color_array,
                    ax=ax,
                )
                if colorbar is not None and self.colorbar_tickparams is not None:
                    self.colorbar_tickparams.set(cb=colorbar)

        self.title.set(auto_title=auto_labels.title)
        self.ylabel.set()
        self.xlabel.set()

        if not isinstance(self.text, list):
            self.text.set()
        else:
            [text.set() for text in self.text]

        if not isinstance(self.annotate, list):
            self.annotate.set()
        else:
            [annotate.set() for annotate in self.annotate]

        extent = self.axis.config_dict.get("extent")

        if extent is None:
            extent = grid.extent_with_buffer_from(buffer=buffer)

        if plot_grid_lines:
            self.grid_plot.plot_rectangular_grid_lines(
                extent=grid.geometry.extent,
                shape_native=grid.shape_native,
            )

        self.axis.set(extent=extent, grid=grid)

        self.tickparams.set()

        if not self.axis.symmetric_around_centre:
            self.yticks.set(min_value=extent[2], max_value=extent[3], units=self.units)
            self.xticks.set(min_value=extent[0], max_value=extent[1], units=self.units)

        if self.contour is not False:
            self.contour.set(array=color_array, extent=extent, use_log10=self.use_log10)

        visuals_2d.plot_via_plotter(plotter=self, grid_indexes=grid.array)

        if not self.is_for_subplot:
            self.output.to_figure(structure=grid, auto_filename=auto_labels.filename)
            self.figure.close()

    def plot_mapper(
        self,
        mapper: MapperRectangular,
        visuals_2d: Visuals2D,
        auto_labels: AutoLabels,
        interpolate_to_uniform: bool = False,
        pixel_values: np.ndarray = Optional[None],
        zoom_to_brightest: bool = True,
    ):
        if isinstance(mapper, MapperRectangular):
            self._plot_rectangular_mapper(
                mapper=mapper,
                visuals_2d=visuals_2d,
                auto_labels=auto_labels,
                pixel_values=pixel_values,
                zoom_to_brightest=zoom_to_brightest,
            )

        elif isinstance(mapper, MapperDelaunay):
            self._plot_delaunay_mapper(
                mapper=mapper,
                visuals_2d=visuals_2d,
                auto_labels=auto_labels,
                interpolate_to_uniform=interpolate_to_uniform,
                pixel_values=pixel_values,
                zoom_to_brightest=zoom_to_brightest,
            )
        else:
            self._plot_voronoi_mapper(
                mapper=mapper,
                visuals_2d=visuals_2d,
                auto_labels=auto_labels,
                interpolate_to_uniform=interpolate_to_uniform,
                pixel_values=pixel_values,
                zoom_to_brightest=zoom_to_brightest,
            )

    def _plot_rectangular_mapper(
        self,
        mapper: MapperRectangular,
        visuals_2d: Visuals2D,
        auto_labels: AutoLabels,
        pixel_values: np.ndarray = Optional[None],
        zoom_to_brightest: bool = True,
    ):
        if pixel_values is not None:
            solution_array_2d = array_2d_util.array_2d_native_from(
                array_2d_slim=pixel_values,
                mask_2d=np.full(
                    fill_value=False, shape=mapper.source_plane_mesh_grid.shape_native
                ),
            )

            pixel_values = Array2D.no_mask(
                values=solution_array_2d,
                pixel_scales=mapper.source_plane_mesh_grid.pixel_scales,
                origin=mapper.source_plane_mesh_grid.origin,
            )

        extent = self.axis.config_dict.get("extent")
        if extent is None:
            extent = mapper.extent_from(
                values=pixel_values, zoom_to_brightest=zoom_to_brightest
            )

        aspect_inv = self.figure.aspect_for_subplot_from(extent=extent)

        if not self.is_for_subplot:
            fig, ax = self.figure.open()
        else:
            ax = self.setup_subplot(aspect=aspect_inv)

            shape_native = mapper.source_plane_mesh_grid.shape_native

        if pixel_values is not None:

            from autoarray.inversion.pixelization.mappers.rectangular_uniform import (
                MapperRectangularUniform,
            )
            from autoarray.inversion.pixelization.mappers.rectangular import (
                MapperRectangular,
            )

            if isinstance(mapper, MapperRectangularUniform):

                self.plot_array(
                    array=pixel_values,
                    visuals_2d=visuals_2d,
                    auto_labels=auto_labels,
                    bypass=True,
                )

            else:

                norm = self.cmap.norm_from(
                    array=pixel_values.array, use_log10=self.use_log10
                )

                edges_transformed = mapper.edges_transformed

                edges_transformed_dense = np.moveaxis(
                    np.stack(np.meshgrid(*edges_transformed.T)), 0, 2
                )

                plt.pcolormesh(
                    edges_transformed_dense[..., 0],
                    edges_transformed_dense[..., 1],
                    pixel_values.array.reshape(shape_native),
                    shading="flat",
                    norm=norm,
                    cmap=self.cmap.cmap,
                )

                if self.colorbar is not False:

                    cb = self.colorbar.set(
                        units=self.units,
                        ax=ax,
                        norm=norm,
                        cb_unit=auto_labels.cb_unit,
                        use_log10=self.use_log10,
                    )
                    self.colorbar_tickparams.set(cb=cb)

                extent_axis = self.axis.config_dict.get("extent")

                if extent_axis is None:
                    extent_axis = extent

                self.axis.set(extent=extent_axis)

                self.tickparams.set()
                self.yticks.set(
                    min_value=extent_axis[2],
                    max_value=extent_axis[3],
                    units=self.units,
                    pixels=shape_native[0],
                )

                self.xticks.set(
                    min_value=extent_axis[0],
                    max_value=extent_axis[1],
                    units=self.units,
                    pixels=shape_native[1],
                )

        if not isinstance(self.text, list):
            self.text.set()
        else:
            [text.set() for text in self.text]

        if not isinstance(self.annotate, list):
            self.annotate.set()
        else:
            [annotate.set() for annotate in self.annotate]

        # self.grid_plot.plot_rectangular_grid_lines(
        #     extent=mapper.source_plane_mesh_grid.geometry.extent,
        #     shape_native=mapper.shape_native,
        # )

        self.title.set(auto_title=auto_labels.title)
        self.ylabel.set()
        self.xlabel.set()

        visuals_2d.plot_via_plotter(
            plotter=self, grid_indexes=mapper.source_plane_data_grid.over_sampled
        )

        if not self.is_for_subplot:
            self.output.to_figure(structure=None, auto_filename=auto_labels.filename)
            self.figure.close()

    def _plot_delaunay_mapper(
        self,
        mapper: MapperDelaunay,
        visuals_2d: Visuals2D,
        auto_labels: AutoLabels,
        interpolate_to_uniform: bool = False,
        pixel_values: np.ndarray = Optional[None],
        zoom_to_brightest: bool = True,
    ):
        extent = self.axis.config_dict.get("extent")
        if extent is None:
            extent = mapper.extent_from(
                values=pixel_values, zoom_to_brightest=zoom_to_brightest
            )

        aspect_inv = self.figure.aspect_for_subplot_from(extent=extent)

        if not self.is_for_subplot:
            fig, ax = self.figure.open()
        else:
            ax = self.setup_subplot(aspect=aspect_inv)

        self.axis.set(extent=extent, grid=mapper.source_plane_mesh_grid)

        plt.gca().set_aspect(aspect_inv)

        self.tickparams.set()
        self.yticks.set(min_value=extent[2], max_value=extent[3], units=self.units)
        self.xticks.set(min_value=extent[0], max_value=extent[1], units=self.units)

        if not isinstance(self.text, list):
            self.text.set()
        else:
            [text.set() for text in self.text]

        if not isinstance(self.annotate, list):
            self.annotate.set()
        else:
            [annotate.set() for annotate in self.annotate]

        interpolation_array = None

        if interpolate_to_uniform:
            interpolation_array = (
                self.interpolated_reconstruction.imshow_reconstruction(
                    mapper=mapper,
                    pixel_values=pixel_values,
                    units=self.units,
                    cmap=self.cmap,
                    colorbar=self.colorbar,
                    colorbar_tickparams=self.colorbar_tickparams,
                    aspect=aspect_inv,
                    ax=ax,
                    use_log10=self.use_log10,
                )
            )

        else:
            self.delaunay_drawer.draw_delaunay_pixels(
                mapper=mapper,
                pixel_values=pixel_values,
                units=self.units,
                cmap=self.cmap,
                colorbar=self.colorbar,
                colorbar_tickparams=self.colorbar_tickparams,
                ax=ax,
                use_log10=self.use_log10,
            )

        self.title.set(auto_title=auto_labels.title)
        self.ylabel.set()
        self.xlabel.set()

        visuals_2d.plot_via_plotter(
            plotter=self, grid_indexes=mapper.source_plane_data_grid.over_sampled
        )

        if not self.is_for_subplot:
            self.output.to_figure(
                structure=interpolation_array, auto_filename=auto_labels.filename
            )
            self.figure.close()

    def _plot_voronoi_mapper(
        self,
        mapper: MapperVoronoi,
        visuals_2d: Visuals2D,
        auto_labels: AutoLabels,
        interpolate_to_uniform: bool = False,
        pixel_values: np.ndarray = Optional[None],
        zoom_to_brightest: bool = True,
    ):
        extent = self.axis.config_dict.get("extent")

        if extent is None:
            extent = mapper.extent_from(
                values=pixel_values, zoom_to_brightest=zoom_to_brightest
            )

        aspect_inv = self.figure.aspect_for_subplot_from(extent=extent)

        if not self.is_for_subplot:
            fig, ax = self.figure.open()
        else:
            ax = self.setup_subplot(aspect=aspect_inv)

        self.axis.set(extent=extent, grid=mapper.source_plane_mesh_grid)

        plt.gca().set_aspect(aspect_inv)

        self.tickparams.set()
        self.yticks.set(min_value=extent[2], max_value=extent[3], units=self.units)
        self.xticks.set(min_value=extent[0], max_value=extent[1], units=self.units)

        if not isinstance(self.text, list):
            self.text.set()
        else:
            [text.set() for text in self.text]

        if not isinstance(self.annotate, list):
            self.annotate.set()
        else:
            [annotate.set() for annotate in self.annotate]

        if not interpolate_to_uniform:
            self.voronoi_drawer.draw_voronoi_pixels(
                mapper=mapper,
                units=self.units,
                pixel_values=pixel_values,
                cmap=self.cmap,
                colorbar=self.colorbar,
                colorbar_tickparams=self.colorbar_tickparams,
                ax=ax,
                use_log10=self.use_log10,
            )

        else:
            self.interpolated_reconstruction.imshow_reconstruction(
                mapper=mapper,
                pixel_values=pixel_values,
                units=self.units,
                cmap=self.cmap,
                colorbar=self.colorbar,
                colorbar_tickparams=self.colorbar_tickparams,
                aspect=aspect_inv,
                ax=ax,
                use_log10=self.use_log10,
            )

        self.title.set(auto_title=auto_labels.title)
        self.ylabel.set()
        self.xlabel.set()

        visuals_2d.plot_via_plotter(
            plotter=self, grid_indexes=mapper.source_plane_data_grid.over_sampled
        )

        if pixel_values is not None:
            interpolation_array = mapper.interpolated_array_from(values=pixel_values)
        else:
            interpolation_array = None

        if not self.is_for_subplot:
            self.output.to_figure(
                structure=interpolation_array, auto_filename=auto_labels.filename
            )
            self.figure.close()
