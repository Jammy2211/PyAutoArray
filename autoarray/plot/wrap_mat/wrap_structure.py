from autoconf import conf
import matplotlib

from matplotlib.collections import PatchCollection
from typing import Callable

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
import numpy as np
import itertools

from autoarray.plot.wrap_mat import wrap_mat
from autoarray.structures import abstract_structure, arrays, grids, vector_fields
from autoarray import exc


class AbstractWrapStructure(wrap_mat.AbstractWrapMat):
    @property
    def config_folder(self):
        return "wrap_structure"


class ArrayOverlay(AbstractWrapStructure):
    def __init__(self, use_subplot_defaults=False, **kwargs):

        super().__init__(use_subplot_defaults=use_subplot_defaults, kwargs=kwargs)

    @property
    def kwargs_imshow(self):
        """Creates a kwargs dict of valid inputs of the method `plt.imshow` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="imshow")

    def overlay_array(self, array, figure):

        aspect = figure.aspect_from_shape_2d(shape_2d=array.shape_2d)
        extent = array.extent_of_zoomed_array(buffer=0)

        plt.imshow(X=array.in_2d, aspect=aspect, extent=extent, **self.kwargs_imshow)


class GridScatter(AbstractWrapStructure):
    def __init__(self, colors=None, use_subplot_defaults=False, **kwargs):
        """
        An object for scattering an input set of grid points, for example (y,x) coordinates or a data structures
        representing 2D (y,x) coordinates like a `Grid` or `GridIrregular`. If the object groups (y,x) coordinates
        they are plotted with colors according to their group.

        This object wraps the following Matplotlib methods:

        - plt.scatter: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html

        There are a number of children of this method in the `include.py` module that plot specific sets of (y,x)
        points, where each uses theirown settings so that the property they plot appears unique on every figure:

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
    def kwargs_scatter(self):
        """Creates a kwargs dict of valid inputs of the method `plt.scatter` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="scatter")

    def scatter_grid(self, grid):
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

    def scatter_grid_colored(self, grid, color_array, cmap):
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

    def scatter_grid_indexes(self, grid, indexes):
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

    def scatter_grid_grouped(self, grid_grouped):
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


class LinePlot(AbstractWrapStructure):
    def __init__(self, use_subplot_defaults=False, colors=None, **kwargs):
        """
        An object for

        This object wraps the following Matplotlib methods:

        - plt.plot: https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.plot.html
        - plt.scatter: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html
        - plt.semilogy: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.semilogy.html
        - plt.loglog: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.loglog.html
        - plt.avxline: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.axvline.html

        There are a number of children of this method in the `include.py` module that plot specific sets of (y,x)
        points, where each uses theirown settings so that the property they plot appears unique on every figure:

        - `OriginScatter`: plots the (y,x) coordinates of the origin of a data structure (e.g. as a black cross).
        Parameters
        ----------
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
    def kwargs_plot(self):
        """Creates a kwargs dict of valid inputs of the method `plt.quiver` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="plot")

    @property
    def kwargs_scatter(self):
        """Creates a kwargs dict of valid inputs of the method `plt.quiver` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="scatter")

    def plot_y_vs_x(self, y, x, plot_axis_type, label=None):

        if plot_axis_type == "linear":
            plt.plot(x, y, c=self.kwargs["colors"], label=label, **self.kwargs_plot)
        elif plot_axis_type == "semilogy":
            plt.semilogy(x, y, c=self.kwargs["colors"], label=label, **self.kwargs_plot)
        elif plot_axis_type == "loglog":
            plt.loglog(x, y, c=self.kwargs["colors"], label=label, **self.kwargs_plot)
        elif plot_axis_type == "scatter":
            print(self.kwargs_scatter)
            plt.scatter(
                x, y, c=self.kwargs["colors"], label=label, **self.kwargs_scatter
            )
        else:
            raise exc.PlottingException(
                "The plot_axis_type supplied to the plotter is not a valid string (must be linear "
                "{semilogy, loglog})"
            )

    def plot_vertical_lines(self, vertical_lines, vertical_line_labels=None):

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

    def plot_rectangular_grid_lines(self, extent, shape_2d):

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

    def plot_grid_grouped(self, grid_grouped):
        """Plot the liness of the mask or the array on the figure.

        Parameters
        -----------t.
        mask : np.ndarray of data_type.array.mask.Mask2D
            The mask applied to the array, the edge of which is plotted as a set of points over the plotted array.
        plot_lines : bool
            If a mask is supplied, its liness pixels (e.g. the exterior edge) is plotted if this is `True`.
        unit_label : str
            The unit_label of the y / x axis of the plots.
        kpc_per_scaled : float or None
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        lines_pointsize : int
            The size of the points plotted to show the liness.
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


class VectorFieldQuiver(AbstractWrapStructure):
    def __init__(self, use_subplot_defaults=False, **kwargs):

        super().__init__(use_subplot_defaults=use_subplot_defaults, kwargs=kwargs)

    @property
    def kwargs_quiver(self):
        """Creates a kwargs dict of valid inputs of the method `plt.quiver` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="quiver")

    def quiver_vector_field(self, vector_field: vector_fields.VectorFieldIrregular):

        plt.quiver(
            vector_field.grid[:, 1],
            vector_field.grid[:, 0],
            vector_field[:, 1],
            vector_field[:, 0],
            **self.kwargs_quiver,
        )


class Patcher(AbstractWrapStructure):
    def __init__(self, use_subplot_defaults=False, **kwargs):

        super().__init__(use_subplot_defaults=use_subplot_defaults, kwargs=kwargs)

        if self.kwargs["facecolor"] is None:
            self.kwargs["facecolor"] = "none"

    @property
    def kwargs_patch_collection(self):
        """Creates a kwargs dict of valid inputs of the method `plt.quiver` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="patch_collection")

    def add_patches(self, patches):

        patch_collection = PatchCollection(
            patches=patches, **self.kwargs_patch_collection
        )

        plt.gcf().gca().add_collection(patch_collection)


class VoronoiDrawer(AbstractWrapStructure):
    def __init__(self, use_subplot_defaults=False, **kwargs):

        super().__init__(use_subplot_defaults=use_subplot_defaults, kwargs=kwargs)

    @property
    def kwargs_fill(self):
        """Creates a kwargs dict of valid inputs of the method `plt.fill` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="fill")

    def draw_voronoi_pixels(self, mapper, values, cmap, cb):

        regions, vertices = self.voronoi_polygons(voronoi=mapper.voronoi)

        if values is not None:
            color_array = values[:] / np.max(values)
            cmap = plt.get_cmap(cmap)
            cb.set_with_values(cmap=cmap, color_values=values)
        else:
            cmap = plt.get_cmap("Greys")
            color_array = np.zeros(shape=mapper.pixels)

        for region, index in zip(regions, range(mapper.pixels)):
            polygon = vertices[region]
            col = cmap(color_array[index])
            plt.fill(*zip(*polygon), facecolor=col, **self.kwargs_fill)

    def voronoi_polygons(self, voronoi, radius=None):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.
        Parameters
        ----------
        voronoi : Voronoi
            Input diagram
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
