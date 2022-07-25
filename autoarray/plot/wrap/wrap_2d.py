from autoarray.plot.wrap.wrap_base import set_backend

set_backend()

import matplotlib.pyplot as plt
from matplotlib import patches as ptch
from matplotlib.collections import PatchCollection
import numpy as np
import itertools
from typing import List, Union, Optional, Tuple

from autoarray.plot.wrap import wrap_base as wb

from autoarray.plot.wrap.wrap_base import AbstractMatWrap
from autoarray.inversion.pixelization.mappers.voronoi import MapperVoronoiNoInterp
from autoarray.inversion.pixelization.mappers.voronoi import MapperVoronoi
from autoarray.inversion.pixelization.mappers.delaunay import MapperDelaunay
from autoarray.inversion.pixelization.mesh import mesh_util
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.vectors.irregular import VectorYX2DIrregular

from autoarray import exc


class AbstractMatWrap2D(AbstractMatWrap):
    """
    An abstract base class for wrapping matplotlib plotting methods which take as input and plot data structures. For
    example, the `ArrayOverlay` object specifically plots `Array2D` data structures.

    As full description of the matplotlib wrapping can be found in `mat_base.AbstractMatWrap`.
    """

    @property
    def config_folder(self):
        return "mat_wrap_2d"


class ArrayOverlay(AbstractMatWrap2D):
    """
    Overlays an `Array2D` data structure over a figure.

    This object wraps the following Matplotlib method:

    - plt.imshow: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html

    This uses the `Units` and coordinate system of the `Array2D` to overlay it on on the coordinate system of the
    figure that is plotted.
    """

    def overlay_array(self, array, figure):

        aspect = figure.aspect_from(shape_native=array.shape_native)
        extent = array.extent_of_zoomed_array(buffer=0)

        plt.imshow(X=array.native, aspect=aspect, extent=extent, **self.config_dict)


class GridScatter(AbstractMatWrap2D):
    """
    Scatters an input set of grid points, for example (y,x) coordinates or data structures representing 2D (y,x)
    coordinates like a `Grid2D` or `Grid2DIrregular`. List of (y,x) coordinates are plotted with varying colors.

    This object wraps the following Matplotlib methods:

    - plt.scatter: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html

    There are a number of children of this method in the `mat_obj.py` module that plot specific sets of (y,x)
    points. Each of these objects uses uses their own config file and settings so that each has a unique appearance
    on every figure:

    - `OriginScatter`: plots the (y,x) coordinates of the origin of a data structure (e.g. as a black cross).
    - `MaskScatter`: plots a mask over an image, using the `Mask2d` object's (y,x)  `edge_grid_sub_1` property.
    - `BorderScatter: plots a border over an image, using the `Mask2d` object's (y,x) `border_grid_sub_1` property.
    - `PositionsScatter`: plots the (y,x) coordinates that are input in a plotter via the `positions` input.
    - `IndexScatter`: plots specific (y,x) coordinates of a grid (or grids) via their 1d or 2d indexes.
    - `MeshGridScatter`: plots the grid of a `Mesh` object (see `autoarray.inversion`).

    Parameters
    ----------
    colors : [str]
        The color or list of colors that the grid is plotted using. For plotting indexes or a grid list, a
        list of colors can be specified which the plot cycles through.
    """

    def scatter_grid(self, grid: Union[np.ndarray, Grid2D]):
        """
        Plot an input grid of (y,x) coordinates using the matplotlib method `plt.scatter`.

        Parameters
        ----------
        grid : Grid2D
            The grid of (y,x) coordinates that is plotted.
        errors
            The error on every point of the grid that is plotted.
        """

        config_dict = self.config_dict

        if len(config_dict["c"]) > 1:
            config_dict["c"] = config_dict["c"][0]

        try:
            plt.scatter(y=grid[:, 0], x=grid[:, 1], **config_dict)
        except (IndexError, TypeError):
            return self.scatter_grid_list(grid_list=grid)

    def scatter_grid_list(self, grid_list: Union[List[Grid2D], List[Grid2DIrregular]]):
        """
         Plot an input list of grids of (y,x) coordinates using the matplotlib method `plt.scatter`.

         This method colors each grid in each entry of the list the same, so that the different grids are visible in
         the plot.

         Parameters
         ----------
         grid_list
             The list of grids of (y,x) coordinates that are plotted.
         """
        if len(grid_list) == 0:
            return

        color = itertools.cycle(self.config_dict["c"])
        config_dict = self.config_dict
        config_dict.pop("c")

        try:
            for grid in grid_list:

                plt.scatter(y=grid[:, 0], x=grid[:, 1], c=next(color), **config_dict)
        except IndexError:
            return None

    def scatter_grid_colored(
        self, grid: Union[np.ndarray, Grid2D], color_array: np.ndarray, cmap: str
    ):
        """
        Plot an input grid of (y,x) coordinates using the matplotlib method `plt.scatter`.

        The method colors the scattered grid according to an input ndarray of color values, using an input colormap.

        Parameters
        ----------
        grid : Grid2D
            The grid of (y,x) coordinates that is plotted.
        color_array : ndarray
            The array of RGB color values used to color the grid.
        cmap
            The Matplotlib colormap used for the grid point coloring.
        """

        config_dict = self.config_dict
        config_dict.pop("c")

        plt.scatter(y=grid[:, 0], x=grid[:, 1], c=color_array, cmap=cmap, **config_dict)

    def scatter_grid_indexes(
        self, grid: Union[np.ndarray, Grid2D], indexes: np.ndarray
    ):
        """
        Plot specific points of an input grid of (y,x) coordinates, which are specified according to the 1D or 2D
        indexes of the `Grid2D`.

        This method allows us to color in points on grids that map between one another.

        Parameters
        ----------
        grid : Grid2D
            The grid of (y,x) coordinates that is plotted.
        indexes
            The 1D indexes of the grid that are colored in when plotted.
        """
        if not isinstance(grid, np.ndarray):
            raise exc.PlottingException(
                "The grid passed into scatter_grid_indexes is not a ndarray and thus its"
                "1D indexes cannot be marked and plotted."
            )

        if len(grid.shape) != 2:
            raise exc.PlottingException(
                "The grid passed into scatter_grid_indexes is not 2D (e.g. a flattened 1D"
                "grid) and thus its 1D indexes cannot be marked."
            )

        if isinstance(indexes, list):
            if not any(isinstance(i, list) for i in indexes):
                indexes = [indexes]

        color = itertools.cycle(self.config_dict["c"])
        config_dict = self.config_dict
        config_dict.pop("c")

        for index_list in indexes:

            if all([isinstance(index, float) for index in index_list]) or all(
                [isinstance(index, int) for index in index_list]
            ):

                plt.scatter(
                    y=grid[index_list, 0],
                    x=grid[index_list, 1],
                    color=next(color),
                    **config_dict,
                )

            elif all([isinstance(index, tuple) for index in index_list]) or all(
                [isinstance(index, list) for index in index_list]
            ):

                ys, xs = map(list, zip(*index_list))

                plt.scatter(
                    y=grid.native[ys, xs, 0],
                    x=grid.native[ys, xs, 1],
                    color=next(color),
                    **config_dict,
                )

            else:

                raise exc.PlottingException(
                    "The indexes input into the grid_scatter_index method do not conform to a "
                    "useable type"
                )


class GridPlot(AbstractMatWrap2D):
    """
    Plots `Grid2D` data structure that are better visualized as solid lines, for example rectangular lines that are
    plotted over an image and grids of (y,x) coordinates as lines (as opposed to a scatter of points
    using the `GridScatter` object).

    This object wraps the following Matplotlib methods:

    - plt.plot: https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.plot.html

    Parameters
    ----------
    colors : [str]
        The color or list of colors that the grid is plotted using. For plotting indexes or a grid list, a
        list of colors can be specified which the plot cycles through.
    """

    def plot_rectangular_grid_lines(
        self, extent: Tuple[float, float, float, float], shape_native: Tuple[int, int]
    ):
        """
        Plots a rectangular grid of lines on a plot, using the coordinate system of the figure.

        The size and shape of the grid is specified by the `extent` and `shape_native` properties of a data structure
        which will provide the rectangaular grid lines on a suitable coordinate system for the plot.

        Parameters
        ----------
        extent : (float, float, float, float)
            The extent of the rectangualr grid, with format [xmin, xmax, ymin, ymax]
        shape_native
            The 2D shape of the mask the array is paired with.
        """

        ys = np.linspace(extent[2], extent[3], shape_native[1] + 1)
        xs = np.linspace(extent[0], extent[1], shape_native[0] + 1)

        # grid lines
        for x in xs:
            plt.plot([x, x], [ys[0], ys[-1]], **self.config_dict)
        for y in ys:
            plt.plot([xs[0], xs[-1]], [y, y], **self.config_dict)

    def plot_grid(self, grid: Union[np.ndarray, Grid2D]):
        """
        Plot an input grid of (y,x) coordinates using the matplotlib method `plt.scatter`.

        Parameters
        ----------
        grid : Grid2D
            The grid of (y,x) coordinates that is plotted.
        """

        try:

            color = self.config_dict["c"]

            if isinstance(color, list):
                color = color[0]

            config_dict = self.config_dict
            config_dict.pop("c")

            plt.plot(grid[:, 1], grid[:, 0], c=color, **config_dict)
        except (IndexError, TypeError):
            return self.plot_grid_list(grid_list=grid)

    def plot_grid_list(self, grid_list: Union[List[Grid2D], List[Grid2DIrregular]]):
        """
         Plot an input list of grids of (y,x) coordinates using the matplotlib method `plt.line`.

        This method colors each grid in the list the same, so that the different grids are visible in the plot.

         This provides an alternative to `GridScatter.scatter_grid_list` where the plotted grids appear as lines
         instead of scattered points.

         Parameters
         ----------
         grid_list : Grid2DIrregular
             The list of grids of (y,x) coordinates that are plotted.
         """

        if len(grid_list) == 0:
            return None

        color = itertools.cycle(self.config_dict["c"])
        config_dict = self.config_dict
        config_dict.pop("c")

        try:
            for grid in grid_list:
                plt.plot(grid[:, 1], grid[:, 0], c=next(color), **config_dict)
        except IndexError:
            return None


class GridErrorbar(AbstractMatWrap2D):
    """
    Plots an input set of grid points with 2D errors, for example (y,x) coordinates or data structures representing 2D
    (y,x) coordinates like a `Grid2D` or `Grid2DIrregular`. Multiple lists of (y,x) coordinates are plotted with
    varying colors.

    This object wraps the following Matplotlib methods:

    - plt.errorbar: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.errorbar.html

    Parameters
    ----------
    colors : [str]
        The color or list of colors that the grid is plotted using. For plotting indexes or a grid list, a
        list of colors can be specified which the plot cycles through.
    """

    def errorbar_grid(
        self,
        grid: Union[np.ndarray, Grid2D],
        y_errors: Optional[Union[np.ndarray, List]] = None,
        x_errors: Optional[Union[np.ndarray, List]] = None,
    ):
        """
        Plot an input grid of (y,x) coordinates using the matplotlib method `plt.errorbar`.

        The (y,x) coordinates are plotted as dots, with a line / cross for its errors.

        Parameters
        ----------
        grid : Grid2D
            The grid of (y,x) coordinates that is plotted.
        y_errors
            The y values of the error on every point of the grid that is plotted (e.g. vertically).
        x_errors
            The x values of the error on every point of the grid that is plotted (e.g. horizontally).
        """

        config_dict = self.config_dict

        if len(config_dict["c"]) > 1:
            config_dict["c"] = config_dict["c"][0]

        try:
            plt.errorbar(
                y=grid[:, 0], x=grid[:, 1], yerr=y_errors, xerr=x_errors, **config_dict
            )
        except (IndexError, TypeError):
            return self.errorbar_grid_list(grid_list=grid)

    def errorbar_grid_list(
        self,
        grid_list: Union[List[Grid2D], List[Grid2DIrregular]],
        y_errors: Optional[Union[np.ndarray, List]] = None,
        x_errors: Optional[Union[np.ndarray, List]] = None,
    ):
        """
        Plot an input list of grids of (y,x) coordinates using the matplotlib method `plt.errorbar`.

        The (y,x) coordinates are plotted as dots, with a line / cross for its errors.

        This method colors each grid in each entry of the list the same, so that the different grids are visible in
        the plot.

        Parameters
        ----------
        grid_list
            The list of grids of (y,x) coordinates that are plotted.
         """
        if len(grid_list) == 0:
            return

        color = itertools.cycle(self.config_dict["c"])
        config_dict = self.config_dict
        config_dict.pop("c")

        try:
            for grid in grid_list:
                plt.errorbar(
                    y=grid[:, 0],
                    x=grid[:, 1],
                    yerr=np.asarray(y_errors),
                    xerr=np.asarray(x_errors),
                    c=next(color),
                    **config_dict,
                )
        except IndexError:
            return None

    def errorbar_grid_colored(
        self,
        grid: Union[np.ndarray, Grid2D],
        color_array: np.ndarray,
        cmap: str,
        y_errors: Optional[Union[np.ndarray, List]] = None,
        x_errors: Optional[Union[np.ndarray, List]] = None,
    ):
        """
        Plot an input grid of (y,x) coordinates using the matplotlib method `plt.errorbar`.

        The method colors the errorbared grid according to an input ndarray of color values, using an input colormap.

        Parameters
        ----------
        grid : Grid2D
            The grid of (y,x) coordinates that is plotted.
        color_array : ndarray
            The array of RGB color values used to color the grid.
        cmap
            The Matplotlib colormap used for the grid point coloring.
        """

        config_dict = self.config_dict
        config_dict.pop("c")

        plt.scatter(y=grid[:, 0], x=grid[:, 1], c=color_array, cmap=cmap)

        plt.errorbar(
            y=grid[:, 0],
            x=grid[:, 1],
            yerr=np.asarray(y_errors),
            xerr=np.asarray(x_errors),
            zorder=0.0,
            **self.config_dict,
        )


class VectorYXQuiver(AbstractMatWrap2D):
    """
    Plots a `VectorField` data structure. A vector field is a set of 2D vectors on a grid of 2d (y,x) coordinates.
    These are plotted as arrows representing the (y,x) components of each vector at each (y,x) coordinate of it
    grid.

    This object wraps the following Matplotlib method:

    https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.quiver.html
    """

    def quiver_vectors(self, vectors: VectorYX2DIrregular):
        """
         Plot a vector field using the matplotlib method `plt.quiver` such that each vector appears as an arrow whose
         direction depends on the y and x magnitudes of the vector.

         Parameters
         ----------
         vectors : VectorYX2DIrregular
             The vector field that is plotted using `plt.quiver`.
         """
        plt.quiver(
            vectors.grid[:, 1],
            vectors.grid[:, 0],
            vectors[:, 1],
            vectors[:, 0],
            **self.config_dict,
        )


class PatchOverlay(AbstractMatWrap2D):
    """
    Adds patches to a plotted figure using matplotlib `patches` objects.

    The coordinate system of each `Patch` uses that of the figure, which is typically set up using the plotted
    data structure. This makes it straight forward to add patches in specific locations.

    This object wraps methods described in below:

    https://matplotlib.org/3.3.2/api/collections_api.html
    """

    def overlay_patches(self, patches: Union[ptch.Patch]):
        """
        Overlay a list of patches on a figure, for example an `Ellipse`.
        `
        Parameters
        ----------
        patches : [Patch]
            The patches that are laid over the figure.
        """

        # patch_collection = PatchCollection(patches=patches, **self.config_dict)
        # plt.gcf().gca().add_collection(patch_collection)


class VoronoiDrawer(AbstractMatWrap2D):
    """
    Draws Voronoi pixels from a `MapperVoronoiNoInterp` object (see `inversions.mapper`). This includes both drawing
    each Voronoi cell and coloring it according to a color value.

    The mapper contains the grid of (y,x) coordinate where the centre of each Voronoi cell is plotted.

    This object wraps methods described in below:

    https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.fill.html
    """

    def draw_voronoi_pixels(
        self,
        mapper: MapperVoronoiNoInterp,
        pixel_values: Optional[np.ndarray],
        cmap: Optional[wb.Cmap],
        colorbar: Optional[wb.Colorbar],
        colorbar_tickparams: Optional[wb.ColorbarTickParams] = None,
    ):
        """
        Draws the Voronoi pixels of the input `mapper` using its `mesh_grid` which contains the (y,x)
        coordinate of the centre of every Voronoi cell. This uses the method `plt.fill`.
        
        Parameters
        ----------
        mapper
            A mapper object which contains the Voronoi mesh.
        pixel_values
            An array used to compute the color values that every Voronoi cell is plotted using.
        cmap
            The colormap used to plot each Voronoi cell.
        colorbar
            The `Colorbar` object in `mat_base` used to set the colorbar of the figure the Voronoi mesh is plotted on.
        """
        regions, vertices = mesh_util.voronoi_revised_from(voronoi=mapper.voronoi)

        if pixel_values is not None:

            vmin = cmap.vmin_from(array=pixel_values)
            vmax = cmap.vmax_from(array=pixel_values)

            color_values = np.where(pixel_values > vmax, vmax, pixel_values)
            color_values = np.where(pixel_values < vmin, vmin, color_values)

            if vmax != vmin:
                color_array = (color_values - vmin) / (vmax - vmin)
            else:
                color_array = np.ones(color_values.shape[0])

            cmap = plt.get_cmap(cmap.cmap)

            if colorbar is not None:

                cb = colorbar.set_with_color_values(
                    cmap=cmap, color_values=color_values
                )

                if cb is not None and colorbar_tickparams is not None:
                    colorbar_tickparams.set(cb=cb)

        else:
            cmap = plt.get_cmap("Greys")
            color_array = np.zeros(shape=mapper.pixels)

        for region, index in zip(regions, range(mapper.pixels)):
            polygon = vertices[region]
            color = cmap(color_array[index])

            plt.fill(*zip(*polygon), facecolor=color, zorder=-1, **self.config_dict)


class InterpolatedReconstruction(AbstractMatWrap2D):
    """
    Given a `Mapper` and a corresponding array of `pixel_values` (e.g. the reconstruction values of a Delaunay
    triangulation) plot the values using `plt.imshow()`.

    The `pixel_values` are an ndarray of values which correspond to the irregular pixels of the mesh (e.g. for
    a Delaunay triangulation they are the connecting corners of each triangle or Voronoi mesh). This cannot be plotted
    with `imshow()`, therefore this class first converts the `pixel_values` from this irregular grid to a uniform 2D
    array of square pixels via interpolation.

    The interpolation routine depends on the `Mapper`, with most mappers having their own built-in interpolation
    routine specific to that pixelization's mesh.

    This object wraps methods described in below:

    - plt.imshow: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html
    """

    def imshow_reconstruction(
        self,
        mapper: Union[MapperDelaunay, MapperVoronoiNoInterp, MapperVoronoi],
        pixel_values: np.ndarray,
        cmap: wb.Cmap,
        colorbar: wb.Colorbar,
        colorbar_tickparams: wb.ColorbarTickParams = None,
        aspect=None,
    ):
        """
        Given a `Mapper` and a corresponding array of `pixel_values` (e.g. the reconstruction values of a Delaunay
        triangulation) plot the values using `plt.imshow()`.

        The `pixel_values` are an ndarray of values which correspond to the irregular pixels of the mesh (e.g. for
        a Delaunay triangulation they are the connecting corners of each triangle or Voronoi mesh). This cannot be plotted
        with `imshow()`, therefore this class first converts the `pixel_values` from this irregular grid to a uniform 2D
        array of square pixels via interpolation.

        The interpolation routine depends on the `Mapper`, with most mappers having their own built-in interpolation
        routine specific to that pixelization's mesh.

        This object wraps methods described in below:

        - plt.imshow: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html
        
        Parameters
        ----------
        mapper
            An object which contains a 2D mesh (e.g. Voronoi mesh cells) and defines how to
            interpolate values from the pixelization's mesh.
        pixel_values
            The pixel values of the pixelization's mesh (e.g. a Voronoi mesh) which are interpolated to a uniform square
            array for plotting with `imshow()`.
        cmap
            The colormap used by `imshow()` to plot the pixelization's mesh values.
        colorbar
            The `Colorbar` object in `mat_base` used to set the colorbar of the figure the interpolated pixelization's mesh
            values (e.g. values interpolated from the Voronoi mesh) are plotted on.
        colorbar_tickparams
            Controls the tick parameters of the colorbar.
        """

        if pixel_values is None:
            return

        vmin = cmap.vmin_from(array=pixel_values)
        vmax = cmap.vmax_from(array=pixel_values)

        color_values = np.where(pixel_values > vmax, vmax, pixel_values)
        color_values = np.where(pixel_values < vmin, vmin, color_values)

        cmap = plt.get_cmap(cmap.cmap)

        if colorbar is not None:

            colorbar = colorbar.set_with_color_values(
                cmap=cmap, color_values=color_values
            )
            if colorbar is not None and colorbar_tickparams is not None:
                colorbar_tickparams.set(cb=colorbar)

        interpolation_array = mapper.interpolated_array_from(values=pixel_values)

        plt.imshow(
            X=interpolation_array.native,
            cmap=cmap,
            extent=mapper.source_mesh_grid.extent_square,
            aspect=aspect,
        )


class OriginScatter(GridScatter):
    """
    Plots the (y,x) coordinates of the origin of a data structure (e.g. as a black cross).

    See `mat_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class MaskScatter(GridScatter):
    """
    Plots a mask over an image, using the `Mask2d` object's (y,x) `edge_grid_sub_1` property.

    See `mat_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class BorderScatter(GridScatter):
    """
    Plots a border over an image, using the `Mask2d` object's (y,x) `border_grid_sub_1` property.

    See `mat_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class PositionsScatter(GridScatter):
    """
    Plots the (y,x) coordinates that are input in a plotter via the `positions` input.

    See `mat_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class IndexScatter(GridScatter):
    """
    Plots specific (y,x) coordinates of a grid (or grids) via their 1d or 2d indexes.

    See `mat_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class MeshGridScatter(GridScatter):
    """
    Plots the grid of a `Mesh` object (see `autoarray.inversion`).

    See `mat_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class ParallelOverscanPlot(GridPlot):
    pass


class SerialPrescanPlot(GridPlot):
    pass


class SerialOverscanPlot(GridPlot):
    pass
