from autoconf import conf
import matplotlib

from matplotlib.collections import PatchCollection

backend = conf.get_matplotlib_backend()

if backend not in "default":
    matplotlib.use(backend)

try:
    hpc_mode = conf.instance["general"]["hpc"]["hpc_mode"]
except KeyError:
    hpc_mode = False

if hpc_mode:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches as ptch
import numpy as np
import itertools
import typing

from autoarray.plot.mat_wrap import mat_base
from autoarray.inversion import mappers
from autoarray.structures import grids, lines, vector_fields
from autoarray import exc


class AbstractMatStructure(mat_base.AbstractMatBase):
    """
    An abstract base class for wrapping matplotlib plotting methods which take as input and plot data structures. For
    example, the `ArrayOverlay` object specifically plots `Array` data structures.

    As full description of the matplotlib wrapping can be found in `mat_base.AbstractMatBase`.
    """

    @property
    def config_folder(self):
        return "mat_structure"


class ArrayOverlay(AbstractMatStructure):
    def __init__(self, use_subplot_defaults=False, **kwargs):
        """
        Overlays an `Array` data structure over a figure.

        This object wraps the following Matplotlib method:

        - plt.imshow: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html

        This uses the `Units` and coordinate system of the `Array` to overlay it on on the coordinate system of the
        figure that is plotted.

        Parameters
        ----------
        use_subplot_defaults : bool
            `Mat` objects load settings from the [figure] section of its .ini config file by default. If
            `use_subplot_defaults=True` settings from the [subplot] section are loaded instead.
        """
        super().__init__(use_subplot_defaults=use_subplot_defaults, kwargs=kwargs)

    @property
    def kwargs_imshow(self):
        """Creates a kwargs dict of valid inputs of the method `plt.imshow` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="imshow")

    def overlay_array(self, array, figure):

        aspect = figure.aspect_from_shape_2d(shape_2d=array.shape_2d)
        extent = array.extent_of_zoomed_array(buffer=0)

        plt.imshow(X=array.in_2d, aspect=aspect, extent=extent, **self.kwargs_imshow)


class GridScatter(AbstractMatStructure):
    def __init__(self, use_subplot_defaults=False, colors=None, **kwargs):
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
        use_subplot_defaults : bool
            `Mat` objects load settings from the [figure] section of its .ini config file by default. If
            `use_subplot_defaults=True` settings from the [subplot] section are loaded instead.
        colors : [str]
            The color or list of colors that the grid is plotted using. For plotting indexes or a grouped grid, a
            list of colors can be specified which the plot cycles through.
        """
        super().__init__(use_subplot_defaults=use_subplot_defaults, kwargs=kwargs)

        if colors is None:
            self.kwargs["colors"] = remove_spaces_and_commas_from_colors(
                colors=self.kwargs["colors"]
            )
        else:
            self.kwargs["colors"] = colors

        if isinstance(self.kwargs["colors"], str):
            self.kwargs["colors"] = [self.kwargs["colors"]]

    @property
    def kwargs_scatter(self) -> dict:
        """Creates a kwargs dict of valid inputs of the method `plt.scatter` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="scatter")

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
            c=self.kwargs["colors"][0],
            **self.kwargs_scatter,
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
            **self.kwargs_scatter,
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

        color = itertools.cycle(self.kwargs["colors"])
        for index_list in indexes:

            if all([isinstance(index, float) for index in index_list]) or all(
                [isinstance(index, int) for index in index_list]
            ):

                plt.scatter(
                    y=np.asarray(grid[index_list, 0]),
                    x=np.asarray(grid[index_list, 1]),
                    color=next(color),
                    **self.kwargs_scatter,
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
                    **self.kwargs_scatter,
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

        color = itertools.cycle(self.kwargs["colors"])

        for group in grid_grouped.in_grouped_list:

            plt.scatter(
                y=np.asarray(group)[:, 0],
                x=np.asarray(group)[:, 1],
                c=next(color),
                **self.kwargs_scatter,
            )


class LinePlot(AbstractMatStructure):
    def __init__(self, use_subplot_defaults=False, colors=None, **kwargs):
        """
        Plots `Line` data structure, including y vs x figures, plotting rectangular lines over an image and plotting
        grids of (y,x) coordinates as lines (as opposed to a scatter of points using the `GridScatter` object).

        This object wraps the following Matplotlib methods:

        - plt.plot: https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.plot.html
        - plt.scatter: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html
        - plt.semilogy: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.semilogy.html
        - plt.loglog: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.loglog.html
        - plt.avxline: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.axvline.html

        Parameters
        ----------
        use_subplot_defaults : bool
            `Mat` objects load settings from the [figure] section of its .ini config file by default. If
            `use_subplot_defaults=True` settings from the [subplot] section are loaded instead.
        colors : [str]
            The color or list of colors that the grid is plotted using. For plotting indexes or a grouped grid, a
            list of colors can be specified which the plot cycles through.
        """
        super().__init__(use_subplot_defaults=use_subplot_defaults, kwargs=kwargs)

        if colors is None:
            self.kwargs["colors"] = remove_spaces_and_commas_from_colors(
                colors=self.kwargs["colors"]
            )
        else:
            self.kwargs["colors"] = colors

        if isinstance(self.kwargs["colors"], str):
            self.kwargs["colors"] = [self.kwargs["colors"]]

    @property
    def kwargs_plot(self) -> dict:
        """Creates a kwargs dict of valid inputs of the method `plt.quiver` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="plot")

    @property
    def kwargs_scatter(self) -> dict:
        """Creates a kwargs dict of valid inputs of the method `plt.quiver` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="scatter")

    def plot_y_vs_x(
        self,
        y: typing.Union[np.ndarray, lines.Line],
        x: typing.Union[np.ndarray, lines.Line],
        plot_axis_type: str,
        label: str = None,
    ):
        """
        Plots 1D y-data against 1D x-data using the matplotlib method `plt.plot`, `plt.semilogy`, `plt.loglog`,
        or `plt.scatter`.

        Parameters
        ----------
        y : np.ndarray or lines.Line
            The ydata that is plotted.
        x : np.ndarray or lines.Line
            The xdata that is plotted.
        plot_axis_type : str
            The method used to make the plot that defines the scale of the axes {"linear", "semilogy", "loglog",
            "scatter"}.
        label : str
            Optionally include a label on the plot for a `Legend` to display.
        """

        if plot_axis_type == "linear":
            plt.plot(x, y, c=self.kwargs["colors"], label=label, **self.kwargs_plot)
        elif plot_axis_type == "semilogy":
            plt.semilogy(x, y, c=self.kwargs["colors"], label=label, **self.kwargs_plot)
        elif plot_axis_type == "loglog":
            plt.loglog(x, y, c=self.kwargs["colors"], label=label, **self.kwargs_plot)
        elif plot_axis_type == "scatter":
            plt.scatter(
                x, y, c=self.kwargs["colors"][0], label=label, **self.kwargs_scatter
            )
        else:
            raise exc.PlottingException(
                "The plot_axis_type supplied to the plotter is not a valid string (must be linear "
                "{semilogy, loglog})"
            )

    def plot_vertical_lines(
        self,
        vertical_lines: typing.List[np.ndarray],
        vertical_line_labels: typing.List[str] = None,
    ):
        """
        Plots vertical lines on 1D plot of y versus x using the method `plt.axvline`.

        This method is typically called after `plot_y_vs_x` to add vertical lines to the figure.

        Parameters
        ----------
        vertical_lines : [np.ndarray]
            The vertical lines of data that are plotted on the figure.
        vertical_line_labels : [str]
            Labels for each vertical line used by a `Legend`.
        """

        if vertical_lines is [] or vertical_lines is None:
            return

        if vertical_line_labels is None:
            vertical_line_labels = [None for i in range(len(vertical_lines))]

        for vertical_line, vertical_line_label in zip(
            vertical_lines, vertical_line_labels
        ):

            plt.axvline(
                x=vertical_line,
                label=vertical_line_label,
                c=self.kwargs["colors"],
                **self.kwargs_plot,
            )

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
            plt.plot(
                [x, x], [ys[0], ys[-1]], c=self.kwargs["colors"], **self.kwargs_plot
            )
        for y in ys:
            plt.plot(
                [xs[0], xs[-1]], [y, y], c=self.kwargs["colors"], **self.kwargs_plot
            )

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

        color = itertools.cycle(self.kwargs["colors"])

        for grid_group in grid_grouped.in_grouped_list:

            plt.plot(
                np.asarray(grid_group)[:, 1],
                np.asarray(grid_group)[:, 0],
                c=next(color),
                **self.kwargs_plot,
            )


class VectorFieldQuiver(AbstractMatStructure):
    def __init__(self, use_subplot_defaults=False, **kwargs):
        """
        Plots a `VectorField` data structure. A vector field is a set of 2D vectors on a grid of 2d (y,x) coordinates.
        These are plotted as arrows representing the (y,x) components of each vector at each (y,x) coordinate of it
        grid.

        This object wraps the following Matplotlib method:

        https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.quiver.html

        Parameters
        ----------
        use_subplot_defaults : bool
            `Mat` objects load settings from the [figure] section of its .ini config file by default. If
            `use_subplot_defaults=True` settings from the [subplot] section are loaded instead.
        """
        super().__init__(use_subplot_defaults=use_subplot_defaults, kwargs=kwargs)

    @property
    def kwargs_quiver(self) -> dict:
        """Creates a kwargs dict of valid inputs of the method `plt.quiver` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="quiver")

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
            **self.kwargs_quiver,
        )


class PatchOverlay(AbstractMatStructure):
    def __init__(self, use_subplot_defaults=False, **kwargs):
        """
        Adds patches to a plotted figure using matplotlib `patches` objects.

        The coordinate system of each `Patch` uses that of the figure, which is typically set up using the plotted
        data structure. This makes it straight forward to add patches in specific locations.

        This object wraps methods described in below:

        https://matplotlib.org/3.3.2/api/collections_api.html

        Parameters
        ----------
        use_subplot_defaults : bool
            `Mat` objects load settings from the [figure] section of its .ini config file by default. If
            `use_subplot_defaults=True` settings from the [subplot] section are loaded instead.
        """
        super().__init__(use_subplot_defaults=use_subplot_defaults, kwargs=kwargs)

        if self.kwargs["facecolor"] is None:
            self.kwargs["facecolor"] = "none"

    @property
    def kwargs_patch_collection(self) -> dict:
        """Creates a kwargs dict of valid inputs of the method `plt.quiver` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="patch_collection")

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
            patches=patches, **self.kwargs_patch_collection
        )

        plt.gcf().gca().add_collection(patch_collection)


class VoronoiDrawer(AbstractMatStructure):
    def __init__(self, use_subplot_defaults=False, **kwargs):
        """
        Draws Voronoi pixels from a `MapperVoronoi` object (see `inversions.mapper`). This includes both drawing
        each Voronoi cell and coloring it according to a color value.

        The mapper contains the grid of (y,x) coordinate where the centre of each Voronoi cell is plotted.

        This object wraps methods described in below:

        https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.fill.html

        Parameters
        ----------
        use_subplot_defaults : bool
            `Mat` objects load settings from the [figure] section of its .ini config file by default. If
            `use_subplot_defaults=True` settings from the [subplot] section are loaded instead.
        """
        super().__init__(use_subplot_defaults=use_subplot_defaults, kwargs=kwargs)

    @property
    def kwargs_fill(self) -> dict:
        """Creates a kwargs dict of valid inputs of the method `plt.fill` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="fill")

    def draw_voronoi_pixels(
        self,
        mapper: mappers.MapperVoronoi,
        values: np.ndarray,
        cmap: str,
        cb: mat_base.Colorbar,
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
            plt.fill(*zip(*polygon), facecolor=col, **self.kwargs_fill)

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


def remove_spaces_and_commas_from_colors(colors):

    colors = [color.strip(",") for color in colors]
    colors = [color.strip(" ") for color in colors]
    return list(filter(None, colors))
