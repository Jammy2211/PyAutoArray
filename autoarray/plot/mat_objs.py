from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import inspect
import itertools
import os

from autoarray import exc


class Units(object):
    def __init__(self, use_scaled=None, conversion_factor=None, in_kpc=None):

        self.use_scaled = use_scaled
        self.conversion_factor = conversion_factor
        self.in_kpc = in_kpc

    @classmethod
    def from_instance_and_config(cls, units):

        if units.use_scaled is not None:
            use_scaled = units.use_scaled
        else:
            try:
                conf.instance.visualize_general.get("general", "use_scaled", bool)
            except:
                use_scaled = True

        try:
            in_kpc = (
                units.in_kpc
                if units.in_kpc is not None
                else conf.instance.visualize_general.get("units", "in_kpc", bool)
            )
        except:
            in_kpc = None

        return Units(
            use_scaled=use_scaled,
            conversion_factor=units.conversion_factor,
            in_kpc=in_kpc,
        )


class Figure(object):
    def __init__(self, figsize=None, aspect=None):

        self.figsize = figsize
        self.aspect = aspect

    @classmethod
    def from_instance_and_config(cls, figure, load_func):

        figsize = (
            figure.figsize
            if figure.figsize is not None
            else load_func("figures", "figsize", str)
        )
        if figsize == "auto":
            figsize = None
        elif isinstance(figsize, str):
            figsize = tuple(map(int, figsize[1:-1].split(",")))

        aspect = (
            figure.aspect
            if figure.aspect is not None
            else load_func("figures", "aspect", str)
        )

        return Figure(figsize=figsize, aspect=aspect)

    def aspect_from_shape_2d(self, shape_2d):

        if self.aspect in "square":
            return float(shape_2d[1]) / float(shape_2d[0])
        else:
            return self.aspect

    def open(self):
        if not plt.fignum_exists(num=1):
            plt.figure(figsize=self.figsize)

    def close(self):
        if plt.fignum_exists(num=1):
            plt.close()


class ColorMap(object):
    def __init__(
        self,
        cmap=None,
        norm=None,
        norm_max=None,
        norm_min=None,
        linthresh=None,
        linscale=None,
    ):

        self.cmap = cmap
        self.norm = norm
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.linthresh = linthresh
        self.linscale = linscale

    @classmethod
    def from_instance_and_config(cls, colormap, load_func):

        cmap = (
            colormap.cmap
            if colormap.cmap is not None
            else load_func("colormap", "cmap", str)
        )
        norm = (
            colormap.norm
            if colormap.norm is not None
            else load_func("colormap", "norm", str)
        )
        norm_min = (
            colormap.norm_min
            if colormap.norm_min is not None
            else load_func("colormap", "norm_min", float)
        )
        norm_max = (
            colormap.norm_max
            if colormap.norm_max is not None
            else load_func("colormap", "norm_max", float)
        )
        linthresh = (
            colormap.linthresh
            if colormap.linthresh is not None
            else load_func("colormap", "linthresh", float)
        )
        linscale = (
            colormap.linscale
            if colormap.linscale is not None
            else load_func("colormap", "linscale", float)
        )

        return ColorMap(
            cmap=cmap,
            norm=norm,
            norm_min=norm_min,
            norm_max=norm_max,
            linthresh=linthresh,
            linscale=linscale,
        )

    def norm_from_array(self, array):
        """Get the normalization scale of the colormap. This will be hyper based on the input min / max normalization \
        values.

        For a 'symmetric_log' colormap, linthesh and linscale also change the colormap.

        If norm_min / norm_max are not supplied, the minimum / maximum values of the array of data_type are used.

        Parameters
        -----------
        array : data_type.array.aa.Scaled
            The 2D array of data_type which is plotted.
        norm_min : float or None
            The minimum array value the colormap map spans (all values below this value are plotted the same color).
        norm_max : float or None
            The maximum array value the colormap map spans (all values above this value are plotted the same color).
        linthresh : float
            For the 'symmetric_log' colormap normalization ,this specifies the range of values within which the colormap \
            is linear.
        linscale : float
            For the 'symmetric_log' colormap normalization, this allowws the linear range set by linthresh to be stretched \
            relative to the logarithmic range.
        """

        if self.norm_min is None:
            norm_min = array.min()
        else:
            norm_min = self.norm_min

        if self.norm_max is None:
            norm_max = array.max()
        else:
            norm_max = self.norm_max

        if self.norm in "linear":
            return colors.Normalize(vmin=norm_min, vmax=norm_max)
        elif self.norm in "log":
            if norm_min == 0.0:
                norm_min = 1.0e-4
            return colors.LogNorm(vmin=norm_min, vmax=norm_max)
        elif self.norm in "symmetric_log":
            return colors.SymLogNorm(
                linthresh=self.linthresh,
                linscale=self.linscale,
                vmin=norm_min,
                vmax=norm_max,
            )
        else:
            raise exc.PlottingException(
                "The normalization (norm) supplied to the plotter is not a valid string (must be "
                "linear | log | symmetric_log"
            )


class ColorBar(object):
    def __init__(
        self, ticksize=None, fraction=None, pad=None, tick_values=None, tick_labels=None
    ):

        self.ticksize = ticksize
        self.fraction = fraction
        self.pad = pad
        self.tick_values = tick_values
        self.tick_labels = tick_labels

    @classmethod
    def from_instance_and_config(cls, cb, load_func):

        ticksize = (
            cb.ticksize
            if cb.ticksize is not None
            else load_func("colorbar", "ticksize", int)
        )
        fraction = (
            cb.fraction
            if cb.fraction is not None
            else load_func("colorbar", "fraction", float)
        )
        pad = cb.pad if cb.pad is not None else load_func("colorbar", "pad", float)
        tick_values = cb.tick_values
        tick_labels = cb.tick_labels

        return ColorBar(
            ticksize=ticksize,
            fraction=fraction,
            pad=pad,
            tick_values=tick_values,
            tick_labels=tick_labels,
        )

    def set(self):
        """Setup the colorbar of the figure, specifically its ticksize and the size is appears relative to the figure.

        Parameters
        -----------
        cb_ticksize : int
            The size of the tick labels on the colorbar.
        cb_fraction : float
            The fraction of the figure that the colorbar takes up, which resizes the colorbar relative to the figure.
        cb_pad : float
            Pads the color bar in the figure, which resizes the colorbar relative to the figure.
        cb_tick_values : [float]
            Manually specified values of where the colorbar tick labels appear on the colorbar.
        cb_tick_labels : [float]
            Manually specified labels of the color bar tick labels, which appear where specified by cb_tick_values.
        """

        if self.tick_values is None and self.tick_labels is None:
            cb = plt.colorbar(fraction=self.fraction, pad=self.pad)
        elif self.tick_values is not None and self.tick_labels is not None:
            cb = plt.colorbar(
                fraction=self.fraction, pad=self.pad, ticks=self.tick_values
            )
            cb.ax.set_yticklabels(labels=self.tick_labels)
        else:
            raise exc.PlottingException(
                "Only 1 entry of tick_values or tick_labels was input. You must either supply"
                "both the values and labels, or neither."
            )

        cb.ax.tick_params(labelsize=self.ticksize)

    def set_with_values(self, cmap, color_values):

        cax = cm.ScalarMappable(cmap=cmap)
        cax.set_array(color_values)

        if self.tick_values is None and self.tick_labels is None:
            plt.colorbar(mappable=cax, fraction=self.fraction, pad=self.pad)
        elif self.tick_values is not None and self.tick_labels is not None:
            cb = plt.colorbar(
                mappable=cax,
                fraction=self.fraction,
                pad=self.pad,
                ticks=self.tick_values,
            )
            cb.ax.set_yticklabels(self.tick_labels)


class Ticks(object):
    def __init__(
        self, ysize=None, xsize=None, y_manual=None, x_manual=None, units=Units()
    ):

        self.ysize = ysize

        self.xsize = xsize
        self.y_manual = y_manual
        self.x_manual = x_manual
        self.units = units

    @classmethod
    def from_instance_and_config(cls, ticks, load_func, units=Units()):

        ysize = (
            ticks.ysize if ticks.ysize is not None else load_func("ticks", "ysize", int)
        )

        xsize = (
            ticks.xsize if ticks.xsize is not None else load_func("ticks", "xsize", int)
        )

        return Ticks(
            ysize=ysize,
            xsize=xsize,
            y_manual=ticks.y_manual,
            x_manual=ticks.x_manual,
            units=units,
        )

    def set_yticks(self, array, extent, symmetric_around_centre=False):
        """Get the extent of the dimensions of the array in the unit_label of the figure (e.g. arc-seconds or kpc).

        This is used to set the extent of the array and thus the y / x axis limits.

        Parameters
        -----------
        array : data_type.array.aa.Scaled
            The 2D array of data_type which is plotted.
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        xticks_manual :  [] or None
            If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
        yticks_manual :  [] or None
            If input, the yticks do not use the array's default yticks but instead overwrite them as these values.
        """

        plt.tick_params(labelsize=self.ysize)

        if symmetric_around_centre:
            return

        yticks = np.linspace(extent[2], extent[3], 5)

        if self.y_manual is not None:
            ytick_labels = np.asarray([self.y_manual[0], self.y_manual[3]])
        elif not self.units.use_scaled:
            ytick_labels = np.linspace(0, array.shape_2d[0], 5).astype("int")
        elif self.units.use_scaled and self.units.conversion_factor is None:
            ytick_labels = np.round(np.linspace(extent[2], extent[3], 5), 2)
        elif self.units.use_scaled and self.units.conversion_factor is not None:
            ytick_labels = np.round(
                np.linspace(
                    extent[2] * self.units.conversion_factor,
                    extent[3] * self.units.conversion_factor,
                    5,
                ),
                2,
            )

        else:
            raise exc.PlottingException(
                "The y and y ticks cannot be set using the input options."
            )

        plt.yticks(ticks=yticks, labels=ytick_labels)

    def set_xticks(self, array, extent, symmetric_around_centre=False):
        """Get the extent of the dimensions of the array in the unit_label of the figure (e.g. arc-seconds or kpc).

        This is used to set the extent of the array and thus the y / x axis limits.

        Parameters
        -----------
        array : data_type.array.aa.Scaled
            The 2D array of data_type which is plotted.
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        xticks_manual :  [] or None
            If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
        xticks_manual :  [] or None
            If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
        """

        plt.tick_params(labelsize=self.xsize)

        if symmetric_around_centre:
            return

        xticks = np.linspace(extent[0], extent[1], 5)

        if self.x_manual is not None:
            xtick_labels = np.asarray([self.x_manual[0], self.x_manual[3]])
        elif not self.units.use_scaled:
            xtick_labels = np.linspace(0, array.shape_2d[0], 5).astype("int")
        elif self.units.use_scaled and self.units.conversion_factor is None:
            xtick_labels = np.round(np.linspace(extent[0], extent[1], 5), 2)
        elif self.units.use_scaled and self.units.conversion_factor is not None:
            xtick_labels = np.round(
                np.linspace(
                    extent[0] * self.units.conversion_factor,
                    extent[1] * self.units.conversion_factor,
                    5,
                ),
                2,
            )

        else:
            raise exc.PlottingException(
                "The y and y ticks cannot be set using the input options."
            )

        plt.xticks(ticks=xticks, labels=xtick_labels)


class Labels(object):
    def __init__(
        self,
        title=None,
        yunits=None,
        xunits=None,
        titlesize=None,
        ysize=None,
        xsize=None,
        units=Units(),
    ):

        self.title = title
        self._yunits = yunits
        self._xunits = xunits

        self.titlesize = titlesize
        self.ysize = ysize
        self.xsize = xsize

        self.units = units

    @classmethod
    def from_instance_and_config(cls, labels, load_func, units=Units()):

        titlesize = (
            labels.titlesize
            if labels.titlesize is not None
            else load_func("labels", "titlesize", int)
        )

        ysize = (
            labels.ysize
            if labels.ysize is not None
            else load_func("labels", "ysize", int)
        )

        xsize = (
            labels.xsize
            if labels.xsize is not None
            else load_func("labels", "xsize", int)
        )

        return Labels(
            title=labels.title,
            yunits=labels._yunits,
            xunits=labels._xunits,
            titlesize=titlesize,
            ysize=ysize,
            xsize=xsize,
            units=units,
        )

    def title_from_func(self, func):
        if self.title is None:

            return func.__name__.capitalize()

        else:

            return self.title

    def yunits_from_func(self, func):

        if self._yunits is None:

            args = inspect.getfullargspec(func).args
            defaults = inspect.getfullargspec(func).defaults

            if defaults is not None:
                non_default_args = len(args) - len(defaults)
            else:
                non_default_args = 0

            if "label_yunits" in args:
                return defaults[args.index("label_yunits") - non_default_args]
            else:
                return None

        else:

            return self._yunits

    def xunits_from_func(self, func):

        if self._xunits is None:

            args = inspect.getfullargspec(func).args
            defaults = inspect.getfullargspec(func).defaults

            if defaults is not None:
                non_default_args = len(args) - len(defaults)
            else:
                non_default_args = 0

            if "label_xunits" in args:
                return defaults[args.index("label_xunits") - non_default_args]
            else:
                return None

        else:

            return self._xunits

    @property
    def yunits(self):

        if self._yunits is None:

            if self.units.in_kpc is not None:
                if self.units.in_kpc:
                    return "kpc"
                else:
                    return "arcsec"

            if self.units.use_scaled:
                return "scaled"
            else:
                return "pixels"

        else:

            return self._yunits

    @property
    def xunits(self):

        if self._xunits is None:

            if self.units.in_kpc is not None:
                if self.units.in_kpc:
                    return "kpc"
                else:
                    return "arcsec"

            if self.units.use_scaled:
                return "scaled"
            else:
                return "pixels"

        else:

            return self._xunits

    def set_title(self):
        """Set the title and title size of the figure.

        Parameters
        -----------
        title : str
            The text of the title.
        titlesize : int
            The size of of the title of the figure.
        """
        plt.title(label=self.title, fontsize=self.titlesize)

    def set_yunits(self, include_brackets):
        """Set the x and y labels of the figure, and set the fontsize of those self.label_

        The x and y labels are always the distance scales, thus the labels are either arc-seconds or kpc and depend on the \
        unit_label the figure is plotted in.

        Parameters
        -----------
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        xsize : int
            The fontsize of the x axes label.
        ysize : int
            The fontsize of the y axes label.
        xyticksize : int
            The font size of the x and y ticks on the figure axes.
        """
        if include_brackets:
            plt.ylabel("y (" + self.yunits + ")", fontsize=self.ysize)
        else:
            plt.ylabel(self.yunits, fontsize=self.ysize)

    def set_xunits(self, include_brackets):
        """Set the x and y labels of the figure, and set the fontsize of those self.label_

        The x and y labels are always the distance scales, thus the labels are either arc-seconds or kpc and depend on the \
        unit_label the figure is plotted in.

        Parameters
        -----------
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        xsize : int
            The fontsize of the x axes label.
        ysize : int
            The fontsize of the y axes label.
        xyticksize : int
            The font size of the x and y ticks on the figure axes.
        """
        if include_brackets:
            plt.xlabel("x (" + self.xunits + ")", fontsize=self.xsize)
        else:
            plt.xlabel(self.xunits, fontsize=self.xsize)


class Legend(object):
    def __init__(self, include=None, fontsize=None):

        self.include = include
        self.fontsize = fontsize

    @classmethod
    def from_instance_and_config(cls, legend, load_func):

        include = (
            legend.include
            if legend.include is not None
            else load_func("legend", "include", bool)
        )

        fontsize = (
            legend.fontsize
            if legend.fontsize is not None
            else load_func("legend", "fontsize", int)
        )

        return Legend(include=include, fontsize=fontsize)

    def set(self):
        if self.include:
            plt.legend(fontsize=self.fontsize)


class Output(object):
    def __init__(self, path=None, filename=None, format=None, bypass=False):

        self.path = path

        if path is not None and path:
            try:
                os.makedirs(path)
            except FileExistsError:
                pass

        self.filename = filename
        self._format = format
        self.bypass = bypass

    @classmethod
    def from_instance_and_config(cls, output, load_func, is_sub_plotter):

        return Output(
            path=output.path,
            format=output._format,
            filename=output.filename,
            bypass=is_sub_plotter,
        )

    @property
    def format(self):
        if self._format is None:
            return "show"
        else:
            return self._format

    def filename_from_func(self, func):

        if self.filename is None:
            return func.__name__
        else:

            return self.filename

    def to_figure(self, structure):
        """Output the figure, either as an image on the screen or to the hard-disk as a .png or .fits file.

        Parameters
        -----------
        structure : ndarray
            The 2D array of image to be output, required for outputting the image as a fits file.
        as_subplot : bool
            Whether the figure is part of subplot, in which case the figure is not output so that the entire subplot can \
            be output instead using the *output.output_figure(structure=None, is_sub_plotter=False)* function.
        output_path : str
            The path on the hard-disk where the figure is output.
        output_filename : str
            The filename of the figure that is output.
        output_format : str
            The format the figue is output:
            'show' - display on computer screen.
            'png' - output to hard-disk as a png.
            'fits' - output to hard-disk as a fits file.'
        """
        if not self.bypass:
            if self.format is "show":
                plt.show()
            elif self.format is "png":
                plt.savefig(self.path + self.filename + ".png", bbox_inches="tight")
            elif self.format is "fits":
                if structure is not None:
                    structure.output_to_fits(
                        file_path=self.path + self.filename + ".fits"
                    )

    def subplot_to_figure(self):
        """Output the figure, either as an image on the screen or to the hard-disk as a .png or .fits file.

        Parameters
        -----------
        structure : ndarray
            The 2D array of image to be output, required for outputting the image as a fits file.
        as_subplot : bool
            Whether the figure is part of subplot, in which case the figure is not output so that the entire subplot can \
            be output instead using the *output.output_figure(structure=None, is_sub_plotter=False)* function.
        output_path : str
            The path on the hard-disk where the figure is output.
        output_filename : str
            The filename of the figure that is output.
        output_format : str
            The format the figue is output:
            'show' - display on computer screen.
            'png' - output to hard-disk as a png.
            'fits' - output to hard-disk as a fits file.'
        """
        if self.format is "show":
            plt.show()
        elif self.format is "png":
            plt.savefig(self.path + self.filename + ".png", bbox_inches="tight")


def is_grids_list_of_grids(grids):

    if len(grids) == 0:
        return "pass"

    if isinstance(grids, list):
        if any(isinstance(i, tuple) for i in grids):
            return False
        elif any(
            isinstance(i, np.ndarray) for i in grids
        ):
            if len(grids) == 1:
                return False
            else:
                return True
        elif any(isinstance(i, list) for i in grids):
            return True
        else:
            raise exc.PlottingException(
                "The grid entered into scatter_grid is a list of values, but its data-structure"
                "cannot be determined so as to make a scatter plot"
            )
    elif isinstance(grids, np.ndarray):
        if len(grids.shape) == 2:
            return False
        else:
            raise exc.PlottingException(
                "The input grid into scatter_Grid is not 2D and therefore "
                "cannot be plotted using scatter."
            )
    else:
        raise exc.PlottingException(
            "The grid passed into scatter_grid is not a list or a ndarray."
        )

def remove_spaces_and_commas_from_colors(colors):

    colors = [color.strip(",") for color in colors]
    colors = [color.strip(" ") for color in colors]
    return list(filter(None, colors))

class Scatterer(object):
    def __init__(self, size=None, marker=None, colors=None):

        self.size = size
        self.marker = marker
        if isinstance(colors, str):
            colors = [colors]
        self.colors = colors

    @classmethod
    def from_instance_and_config(cls, scatterer, section, load_func):

        size = (
            scatterer.size
            if scatterer.size is not None
            else load_func(section, "size", int)
        )

        marker = (
            scatterer.marker
            if scatterer.marker is not None
            else load_func(section, "marker", str)
        )

        colors = (
            scatterer.colors
            if scatterer.colors is not None
            else load_func(section, "colors", list)
        )

        colors = remove_spaces_and_commas_from_colors(colors=colors)

        return Scatterer(size=size, marker=marker, colors=colors)

    def scatter_grids(self, grids):

        list_of_grids = is_grids_list_of_grids(grids=grids)

        if not list_of_grids:

            plt.scatter(
                y=np.asarray(grids)[:, 0],
                x=np.asarray(grids)[:, 1],
                s=self.size,
                c=self.colors[0],
                marker=self.marker,
            )

        else:

            color = itertools.cycle(self.colors)

            for grid in grids:

                if not None in grid:
                    if len(grid) != 0:
                        plt.scatter(
                            y=np.asarray(grid)[:, 0],
                            x=np.asarray(grid)[:, 1],
                            s=self.size,
                            c=next(color),
                            marker=self.marker,
                        )

    def scatter_colored_grid(self, grid, color_array, cmap):

        list_of_grids = is_grids_list_of_grids(grids=grid)

        if not list_of_grids:

            plt.scatter(
                y=np.asarray(grid)[:, 0],
                x=np.asarray(grid)[:, 1],
                s=self.size,
                c=color_array,
                marker=self.marker,
                cmap=cmap,
            )

        else:

            raise exc.PlottingException(
                "Cannot plot colorred grid if input grid is a list of grids."
            )

    def scatter_grid_indexes(self, grid, indexes):

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
                    s=self.size,
                    color=next(color),
                    marker=self.marker,
                )

            elif all([isinstance(index, tuple) for index in index_list]) or all(
                [isinstance(index, list) for index in index_list]
            ):

                ys = [index[0] for index in index_list]
                xs = [index[1] for index in index_list]

                plt.scatter(
                    y=np.asarray(grid.in_2d[ys, xs, 0]),
                    x=np.asarray(grid.in_2d[ys, xs, 1]),
                    s=self.size,
                    color=next(color),
                    marker=self.marker,
                )

            else:

                raise exc.PlottingException(
                    "The indexes input into the grid_scatter_index method do not conform to a "
                    "useable type"
                )


class Liner(object):
    def __init__(self, width=None, style=None, colors=None, pointsize=None):

        self.width = width
        self.style = style
        if isinstance(colors, str):
            colors = [colors]
        self.colors = colors
        self.pointsize = pointsize

    @classmethod
    def from_instance_and_config(cls, liner, section, load_func):

        width = (
            liner.width if liner.width is not None else load_func(section, "width", int)
        )

        style = (
            liner.style if liner.style is not None else load_func(section, "style", str)
        )

        colors = (
            liner.colors if liner.colors is not None else load_func(section, "colors", list)
        )

        colors = remove_spaces_and_commas_from_colors(colors=colors)

        pointsize = (
            liner.pointsize
            if liner.pointsize is not None
            else load_func(section, "pointsize", int)
        )

        return Liner(width=width, style=style, colors=colors, pointsize=pointsize)

    def draw_y_vs_x(self, y, x, plot_axis_type, label=None):

        if plot_axis_type is "linear":
            plt.plot(x, y, c=self.colors[0], lw=self.width, ls=self.style, label=label)
        elif plot_axis_type is "semilogy":
            plt.semilogy(x, y, c=self.colors[0], lw=self.width, ls=self.style, label=label)
        elif plot_axis_type is "loglog":
            plt.loglog(x, y, c=self.colors[0], lw=self.width, ls=self.style, label=label)
        elif plot_axis_type is "scatter":
            plt.scatter(x, y, c=self.colors[0], s=self.pointsize, label=label)
        else:
            raise exc.PlottingException(
                "The plot_axis_type supplied to the plotter is not a valid string (must be linear "
                "| semilogy | loglog)"
            )

    def draw_vertical_lines(self, vertical_lines, vertical_line_labels=None):

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
                c=self.colors[0],
                lw=self.width,
                ls=self.style,
            )

    def draw_grids(self, grids):
        """Plot the liness of the mask or the array on the figure.

        Parameters
        -----------t.
        mask : ndarray of data_type.array.mask.Mask
            The mask applied to the array, the edge of which is plotted as a set of points over the plotted array.
        plot_lines : bool
            If a mask is supplied, its liness pixels (e.g. the exterior edge) is plotted if this is *True*.
        unit_label : str
            The unit_label of the y / x axis of the plots.
        kpc_per_arcsec : float or None
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        lines_pointsize : int
            The size of the points plotted to show the liness.
        """

        list_of_grids = is_grids_list_of_grids(grids=grids)

        if not list_of_grids:

            plt.plot(
                np.asarray(grids)[:, 1],
                np.asarray(grids)[:, 0],
                c=self.colors[0],
                lw=self.width,
                ls=self.style,
            )

        else:

            color = itertools.cycle(self.colors)

            for grid in grids:

                if not None in grid:
                    if len(grid) != 0:
                        plt.plot(
                            np.asarray(grid)[:, 1],
                            np.asarray(grid)[:, 0],
                            c=next(color),
                            lw=self.width,
                            ls=self.style,
                        )

    def draw_rectangular_grid_lines(self, extent, shape_2d):

        ys = np.linspace(extent[0], extent[1], shape_2d[0] + 1)
        xs = np.linspace(extent[2], extent[3], shape_2d[1] + 1)

        # grid lines
        for x in xs:
            plt.plot(
                [x, x], [ys[0], ys[-1]], color=self.colors[0], lw=self.width, ls=self.style
            )
        for y in ys:
            plt.plot(
                [xs[0], xs[-1]], [y, y], color=self.colors[0], lw=self.width, ls=self.style
            )


class VoronoiDrawer(object):
    def __init__(self, edgewidth=None, edgecolor=None, alpha=None):

        self.edgewidth = edgewidth
        self.edgecolor = edgecolor
        self.alpha = alpha

    @classmethod
    def from_instance_and_config(cls, voronoi_drawer, section, load_func):

        edgewidth = (
            voronoi_drawer.edgewidth
            if voronoi_drawer.edgewidth is not None
            else load_func(section, "edgewidth", float)
        )

        edgecolor = (
            voronoi_drawer.edgecolor
            if voronoi_drawer.edgecolor is not None
            else load_func(section, "edgecolor", str)
        )

        alpha = (
            voronoi_drawer.alpha
            if voronoi_drawer.alpha is not None
            else load_func(section, "alpha", float)
        )

        return VoronoiDrawer(edgewidth=edgewidth, edgecolor=edgecolor, alpha=alpha)

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
            plt.fill(
                *zip(*polygon),
                edgecolor=self.edgecolor,
                alpha=self.alpha,
                facecolor=col,
                lw=self.edgewidth
            )

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
            Coordinates for revised Voronoi vertices. Same as coordinates
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
