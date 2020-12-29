from autoarray.plot.mat_wrap.wrap import wrap_base

wrap_base.set_backend()

from autoarray.plot.mat_wrap import wrap

import matplotlib.pyplot as plt
from matplotlib import patches as ptch
from matplotlib.collections import PatchCollection
import numpy as np
import itertools
import typing

from autoarray.inversion import mappers
from autoarray.structures import grids, vector_fields
from autoarray import exc


class AbstractMatWrap2D(wrap_base.AbstractMatWrap):
    """
    An abstract base class for wrapping matplotlib plotting methods which take as input and plot data structures. For
    example, the `ArrayOverlay` object specifically plots `Array` data structures.

    As full description of the matplotlib wrapping can be found in `mat_base.AbstractMatWrap`.
    """

    @property
    def config_folder(self):
        return "mat_wrap_2d"


class ArrayOverlay(AbstractMatWrap2D):
    def __init__(self, **kwargs):
        """
        Overlays an `Array` data structure over a figure.

        This object wraps the following Matplotlib method:

        - plt.imshow: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html

        This uses the `Units` and coordinate system of the `Array` to overlay it on on the coordinate system of the
        figure that is plotted.
        """
        super().__init__(kwargs=kwargs)

    def overlay_array(self, array, figure):

        aspect = figure.aspect_from_shape_2d(shape_2d=array.shape_2d)
        extent = array.extent_of_zoomed_array(buffer=0)

        plt.imshow(
            X=array.in_2d, aspect=aspect, extent=extent, **self.config_dict_imshow
        )


class GridScatter(AbstractMatWrap2D, wrap_base.AbstractMatWrapColored):
    def __init__(self, colors=None, **kwargs):
        """
        Scatters an input set of grid points, for example (y,x) coordinates or data structures representing 2D (y,x)
        coordinates like a `Grid` or `GridIrregular`. If the object groups (y,x) coordinates they are plotted with
        varying colors according to their group.

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
        - `PixelizationGridScatter`: plots the grid of a `Pixelization` object (see `autoarray.inversion`).

        Parameters
        ----------
        colors : [str]
            The color or list of colors that the grid is plotted using. For plotting indexes or a grouped grid, a
            list of colors can be specified which the plot cycles through.
        """
        super().__init__(kwargs=kwargs)
        wrap_base.AbstractMatWrapColored.__init__(self=self, colors=colors)

    def scatter_grid(self, grid: typing.Union[np.ndarray, grids.Grid]):
        """
        Plot an input grid of (y,x) coordinates using the matplotlib method `plt.scatter`.

        Parameters
        ----------
        grid : Grid
            The grid of (y,x) coordinates that is plotted.
        """
        plt.scatter(
            y=np.asarray(grid)[:, 0],
            x=np.asarray(grid)[:, 1],
            c=self.colors[0],
            **self.config_dict_scatter,
        )

    def scatter_grid_colored(
        self,
        grid: typing.Union[np.ndarray, grids.Grid],
        color_array: np.ndarray,
        cmap: str,
    ):
        """
        Plot an input grid of (y,x) coordinates using the matplotlib method `plt.scatter`.

        The method colors the scattered grid according to an input ndarray of color values, using an input colormap.

        Parameters
        ----------
        grid : Grid
            The grid of (y,x) coordinates that is plotted.
        color_array : ndarray
            The array of RGB color values used to color the grid.
        cmap : str
            The Matplotlib colormap used for the grid point coloring.
        """
        plt.scatter(
            y=np.asarray(grid)[:, 0],
            x=np.asarray(grid)[:, 1],
            c=color_array,
            cmap=cmap,
            **self.config_dict_scatter,
        )

    def scatter_grid_indexes(
        self, grid: typing.Union[np.ndarray, grids.Grid], indexes: np.ndarray
    ):
        """
        Plot specific points of an input grid of (y,x) coordinates, which are specified according to the 1D or 2D
        indexes of the `Grid`.

        This method allows us to color in points on grids that map between one another.

        Parameters
        ----------
        grid : Grid
            The grid of (y,x) coordinates that is plotted.
        indexes : np.ndarray
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

        color = itertools.cycle(self.colors)
        for index_list in indexes:

            if all([isinstance(index, float) for index in index_list]) or all(
                [isinstance(index, int) for index in index_list]
            ):

                plt.scatter(
                    y=np.asarray(grid[index_list, 0]),
                    x=np.asarray(grid[index_list, 1]),
                    color=next(color),
                    **self.config_dict_scatter,
                )

            elif all([isinstance(index, tuple) for index in index_list]) or all(
                [isinstance(index, list) for index in index_list]
            ):

                ys = [index[0] for index in index_list]
                xs = [index[1] for index in index_list]

                plt.scatter(
                    y=np.asarray(grid.in_2d[ys, xs, 0]),
                    x=np.asarray(grid.in_2d[ys, xs, 1]),
                    color=next(color),
                    **self.config_dict_scatter,
                )

            else:

                raise exc.PlottingException(
                    "The indexes input into the grid_scatter_index method do not conform to a "
                    "useable type"
                )

    def scatter_grid_grouped(self, grid_grouped: grids.GridIrregularGrouped):
        """
         Plot an input grid of grouped (y,x) coordinates using the matplotlib method `plt.scatter`.

         Coordinates are grouped when they share a common origin or feature. This method colors each group the same,
         so that the grouping is visible in the plot.

         Parameters
         ----------
         grid_grouped : GridIrregularGrouped
             The grid of grouped (y,x) coordinates that is plotted.
         """
        if len(grid_grouped) == 0:
            return

        color = itertools.cycle(self.colors)

        for group in grid_grouped.in_grouped_list:

            plt.scatter(
                y=np.asarray(group)[:, 0],
                x=np.asarray(group)[:, 1],
                c=next(color),
                **self.config_dict_scatter,
            )


class GridPlot(AbstractMatWrap2D, wrap_base.AbstractMatWrapColored):
    def __init__(self, colors=None, **kwargs):
        """
        Plots `Grid` data structure that are better visualized as solid lines, for example rectangular lines that are
        plotted over an image and grids of (y,x) coordinates as lines (as opposed to a scatter of points
        using the `GridScatter` object).

        This object wraps the following Matplotlib methods:

        - plt.plot: https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.plot.html

        Parameters
        ----------
        colors : [str]
            The color or list of colors that the grid is plotted using. For plotting indexes or a grouped grid, a
            list of colors can be specified which the plot cycles through.
        """
        super().__init__(kwargs=kwargs)
        wrap_base.AbstractMatWrapColored.__init__(self=self, colors=colors)

    def plot_rectangular_grid_lines(
        self,
        extent: typing.Tuple[float, float, float, float],
        shape_2d: typing.Tuple[int, int],
    ):
        """
        Plots a rectangular grid of lines on a plot, using the coordinate system of the figure.

        The size and shape of the grid is specified by the `extent` and `shape_2d` properties of a data structure
        which will provide the rectangaular grid lines on a suitable coordinate system for the plot.

        Parameters
        ----------
        extent : (float, float, float, float)
            The extent of the rectangualr grid, with format [xmin, xmax, ymin, ymax]
        shape_2d : (int, int)
            The 2D shape of the mask the array is paired with.
        """

        ys = np.linspace(extent[2], extent[3], shape_2d[1] + 1)
        xs = np.linspace(extent[0], extent[1], shape_2d[0] + 1)

        # grid lines
        for x in xs:
            plt.plot([x, x], [ys[0], ys[-1]], c=self.colors, **self.config_dict_plot)
        for y in ys:
            plt.plot([xs[0], xs[-1]], [y, y], c=self.colors, **self.config_dict_plot)

    def plot_grid_grouped(self, grid_grouped: grids.GridIrregularGrouped):
        """
         Plot an input grid of grouped (y,x) coordinates using the matplotlib method `plt.line`.

         Coordinates are grouped when they share a common origin or feature. This method colors each group the same,
         so that the grouping is visible in the plot.

         This provides an alternative to `GridScatter.scatter_grid_grouped` where the plotted grids appear as lines
         instead of scattered points.

         Parameters
         ----------
         grid_grouped : GridIrregularGrouped
             The grid of grouped (y,x) coordinates that are plotted.
         """

        if len(grid_grouped) == 0:
            return

        color = itertools.cycle(self.colors)

        for grid_group in grid_grouped.in_grouped_list:

            plt.plot(
                np.asarray(grid_group)[:, 1],
                np.asarray(grid_group)[:, 0],
                c=next(color),
                **self.config_dict_plot,
            )


class VectorFieldQuiver(AbstractMatWrap2D):
    def __init__(self, **kwargs):
        """
        Plots a `VectorField` data structure. A vector field is a set of 2D vectors on a grid of 2d (y,x) coordinates.
        These are plotted as arrows representing the (y,x) components of each vector at each (y,x) coordinate of it
        grid.

        This object wraps the following Matplotlib method:

        https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.quiver.html
        """
        super().__init__(kwargs=kwargs)

    def quiver_vector_field(self, vector_field: vector_fields.VectorFieldIrregular):
        """
         Plot a vector field using the matplotlib method `plt.quiver` such that each vector appears as an arrow whose
         direction depends on the y and x magnitudes of the vector.

         Parameters
         ----------
         vector_field : VectorFieldIrregular
             The vector field that is plotted using `plt.quiver`.
         """
        plt.quiver(
            vector_field.grid[:, 1],
            vector_field.grid[:, 0],
            vector_field[:, 1],
            vector_field[:, 0],
            **self.config_dict_quiver,
        )


class PatchOverlay(AbstractMatWrap2D):
    def __init__(self, **kwargs):
        """
        Adds patches to a plotted figure using matplotlib `patches` objects.

        The coordinate system of each `Patch` uses that of the figure, which is typically set up using the plotted
        data structure. This makes it straight forward to add patches in specific locations.

        This object wraps methods described in below:

        https://matplotlib.org/3.3.2/api/collections_api.html
        """
        super().__init__(kwargs=kwargs)

    def overlay_patches(self, patches: typing.Union[ptch.Patch]):
        """
        Overlay a list of patches on a figure, for example an `Ellipse`.
        `
        Parameters
        ----------
        patches : [Patch]
            The patches that are laid over the figure.
        """
        patch_collection = PatchCollection(
            patches=patches, **self.config_dict_patch_collection
        )

        plt.gcf().gca().add_collection(patch_collection)


class VoronoiDrawer(AbstractMatWrap2D):
    def __init__(self, **kwargs):
        """
        Draws Voronoi pixels from a `MapperVoronoi` object (see `inversions.mapper`). This includes both drawing
        each Voronoi cell and coloring it according to a color value.

        The mapper contains the grid of (y,x) coordinate where the centre of each Voronoi cell is plotted.

        This object wraps methods described in below:

        https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.fill.html
        """
        super().__init__(kwargs=kwargs)

    def draw_voronoi_pixels(
        self,
        mapper: mappers.MapperVoronoi,
        values: np.ndarray,
        cmap: str,
        cb: wrap.Colorbar,
    ):
        """
        Draws the Voronoi pixels of the input `mapper` using its `pixelization_grid` which contains the (y,x) 
        coordinate of the centre of every Voronoi cell. This uses the method `plt.fill`.
        
        Parameters
        ----------
        mapper : MapperVoronoi
            An object which contains the (y,x) grid of Voronoi cell centres.
        values : np.ndarray
            An array used to compute the color values that every Voronoi cell is plotted using.
        cmap : str
            The colormap used to plot each Voronoi cell.
        cb : Colorbar
            The `Colorbar` object in `mat_base` used to set the colorbar of the figure the Voronoi mesh is plotted on.
        """
        regions, vertices = self.voronoi_polygons(voronoi=mapper.voronoi)

        if values is not None:
            color_array = values[:] / np.max(values)
            cmap = plt.get_cmap(cmap)
            cb.set_with_color_values(cmap=cmap, color_values=values)
        else:
            cmap = plt.get_cmap("Greys")
            color_array = np.zeros(shape=mapper.pixels)

        for region, index in zip(regions, range(mapper.pixels)):
            polygon = vertices[region]
            col = cmap(color_array[index])
            plt.fill(*zip(*polygon), facecolor=col, **self.config_dict_fill)

    def voronoi_polygons(self, voronoi, radius=None):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite regions.

        Parameters
        ----------
        voronoi : Voronoi
            The input Voronoi diagram that is being plotted.
        radius : float, optional
            Distance to 'points at infinity'.

        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            GridIrregularGrouped for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.
        """

        if voronoi.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = voronoi.vertices.tolist()

        center = voronoi.points.mean(axis=0)
        if radius is None:
            radius = voronoi.points.ptp().max() * 2

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(voronoi.ridge_points, voronoi.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(voronoi.point_region):
            vertices = voronoi.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = voronoi.points[p2] - voronoi.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # hyper

                midpoint = voronoi.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = voronoi.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)


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


class PixelizationGridScatter(GridScatter):
    """
    Plots the grid of a `Pixelization` object (see `autoarray.inversion`).

    See `mat_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class ParallelOverscanPlot(GridPlot):
    pass


class SerialPrescanPlot(GridPlot):
    pass


class SerialOverscanPlot(GridPlot):
    pass
