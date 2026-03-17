import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Union

from autoconf import conf

from autoarray.inversion.mesh.interpolator.rectangular import (
    InterpolatorRectangular,
)
from autoarray.inversion.mesh.interpolator.rectangular_uniform import (
    InterpolatorRectangularUniform,
)
from autoarray.inversion.mesh.interpolator.delaunay import InterpolatorDelaunay
from autoarray.inversion.mesh.interpolator.knn import (
    InterpolatorKNearestNeighbor,
)
from autoarray.mask.derive.zoom_2d import Zoom2D
from autoarray.plot.mat_plot.abstract import AbstractMatPlot
from autoarray.plot.auto_labels import AutoLabels
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
        delaunay_drawer: Optional[w2d.DelaunayDrawer] = None,
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
        quick_update: bool = False,
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

        self.delaunay_drawer = delaunay_drawer or w2d.DelaunayDrawer(is_default=True)

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
        self.quick_update = quick_update

