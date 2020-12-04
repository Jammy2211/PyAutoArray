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
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import inspect
import itertools
from os import path
import os
import colorcet
import configparser
import typing

from autoarray.structures import abstract_structure, arrays, grids, vector_fields
from autoarray import exc


def load_setting(param, section, name, from_subplot_config):

    if param is not None:
        return param

    if not from_subplot_config:
        return load_figure_setting(section, name)
    else:
        return load_subplot_setting(section, name)


def load_figure_setting(section, name):
    return conf.instance["visualize"]["figures"][section][name]


def load_subplot_setting(section, name):
    return conf.instance["visualize"]["subplots"][section][name]


class AbstractMatObj:
    @property
    def section(self):
        raise NotImplementedError

    def load_setting(self, param, name, from_subplot_config):

        if param is not None:
            return param

        if not from_subplot_config:
            return conf.instance["visualize"]["figures"][self.section][name]
        else:
            return conf.instance["visualize"]["subplots"][self.section][name]


class Units(AbstractMatObj):
    def __init__(
        self,
        use_scaled: bool = None,
        conversion_factor: float = None,
        in_kpc: bool = None,
    ):
        """
        The units of the figure's y and x axis.

        Parameters
        ----------
        use_scaled : bool
            If True, plot the y and x axis labels of the `Array` as its scaled coordinates using its *pixel_scales*
            attribute. If `False` plot them in pixel units.
        conversion_factor : float
            If plotting the labels in scaled units, this factor multiplies the values that are used for the labels.
            This allows for additional unit conversions of the figure labels.
        in_kpc : bool
            If True, the scaled units are converted to kilo-parsecs via the input Comsology of the plot (this is only
            relevent for the projects PyAutoGalaxy / PyAutoLens).
        """

        self.use_scaled = use_scaled
        self.conversion_factor = conversion_factor
        self.in_kpc = in_kpc

        if use_scaled is not None:
            self.use_scaled = use_scaled
        else:
            try:
                self.use_scaled = conf.instance["visualize"]["general"]["general"][
                    "use_scaled"
                ]
            except:
                self.use_scaled = True

        try:
            self.in_kpc = (
                in_kpc
                if in_kpc is not None
                else conf.instance["visualize"]["general"]["units"]["in_kpc"]
            )
        except:
            self.in_kpc = None


class Figure(AbstractMatObj):
    def __init__(
        self,
        figsize: typing.Tuple[float, float] = None,
        aspect: typing.Union[float, str] = None,
        from_subplot_config: bool = False,
    ):
        """
        The settings used to set up the Matplotlib Figure before plotting.

        This object wraps the following Matplotlib methods:

        - plt.figure: https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.figure.html
        - plt.close()

        It also controls the aspect ratio of the figure plotted.

        Parameters
        ----------
        figsize : (float, float)
            Width, height in inches.
        aspect : float or str
            The aspect ratio of the figure.
        from_subplot_config : bool
            If True, load unspecified settings from the figures.ini visualization config, else use subplots.ini.
        """

        self.from_subplot_config = from_subplot_config

        self.figsize = load_setting(
            param=figsize,
            section="figures",
            name="figsize",
            from_subplot_config=from_subplot_config,
        )

        if self.figsize == "auto":
            self.figsize = None
        elif isinstance(self.figsize, str):
            self.figsize = tuple(map(int, self.figsize[1:-1].split(",")))

        self.aspect = load_setting(
            param=aspect,
            section="figures",
            name="aspect",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(
        cls,
        figsize: typing.Union[typing.Tuple[float, float]] = None,
        aspect: float or str = None,
    ):
        return Figure(figsize=figsize, aspect=aspect, from_subplot_config=True)

    def aspect_from_shape_2d(self, shape_2d: typing.Union[typing.Tuple[int, int]]):
        """
        Returns the aspect ratio of the figure from the 2D shape of an `Array`.

        This is primarily used to ensure that rectangular arrays are plotted as square figures on sub-plots.

        Parameters
        ----------
        shape_2d : (int, int)
            The two dimensional shape of an `Array` that is to be plotted.
        """
        if isinstance(self.aspect, str):
            if self.aspect in "square":
                return float(shape_2d[1]) / float(shape_2d[0])

        return self.aspect

    def open(self):
        """Wraps the Matplotlib method 'plt.figure' for opening a figure."""
        if not plt.fignum_exists(num=1):
            plt.figure(figsize=self.figsize)

    def close(self):
        """Wraps the Matplotlib method 'plt.close' for closing a figure."""
        if plt.fignum_exists(num=1):
            plt.close()


class ColorMap(AbstractMatObj):
    def __init__(
        self,
        module=None,
        cmap: str = None,
        norm: str = None,
        norm_max: float = None,
        norm_min: float = None,
        linthresh: float = None,
        linscale: float = None,
        from_subplot_config: bool = False,
    ):
        """
        The settings used to set up the Matplotlib colormap and its normalization.

        This object wraps the following Matplotlib methods:

        - colors.Linear: https://matplotlib.org/3.3.1/tutorials/colors/colormaps.html
        - colors.LogNorm: https://matplotlib.org/3.1.1/tutorials/colors/colormapnorms.html
        - colors.SymLogNorm: https://matplotlib.org/3.3.0/api/_as_gen/matplotlib.colors.SymLogNorm.html

        Parameters
        ----------
        cmap : str
            The colormap used to map normalized data values to RGBA colors (see
            https://matplotlib.org/3.3.1/api/cm_api.html).
        norm : str
            The Normalize object applied to the colormap (linear / log / symmetric_log)
        norm_max : float
            The maximum value of the normalization range, such that all values on a plotted `Array` above this value
            are the same color.
        norm_min : float
            The minimum value of the normalization range, such that all values on a plotted `Array` below this value
            are the same color.
        linthresh : float
            The range within which the plot is linear for a symmetric_log Normalization.
        linscale : float
            This allows the linear range (-linthresh to linthresh) to be stretched relative to the logarithmic range.
        from_subplot_config : bool
            If True, load unspecified settings from the figures.ini visualization config, else use subplots.ini.
        """

        self.from_subplot_config = from_subplot_config

        cmap = load_setting(
            param=cmap,
            section="colormap",
            name="cmap",
            from_subplot_config=from_subplot_config,
        )

        if module is not None:

            module_name = module.__name__.split(".")[-1]
            try:
                cmap = conf.instance["visualize"]["general"]["colormaps"][module_name]
            except configparser.NoOptionError:
                cmap = conf.instance["visualize"]["general"]["colormaps"]["default"]

        try:
            self.cmap = colorcet.cm[cmap]
        except KeyError:
            self.cmap = cmap

        self.norm = load_setting(
            param=norm,
            section="colormap",
            name="norm",
            from_subplot_config=from_subplot_config,
        )
        self.norm_min = load_setting(
            param=norm_min,
            section="colormap",
            name="norm_min",
            from_subplot_config=from_subplot_config,
        )
        self.norm_max = load_setting(
            param=norm_max,
            section="colormap",
            name="norm_max",
            from_subplot_config=from_subplot_config,
        )
        self.linthresh = load_setting(
            param=linthresh,
            section="colormap",
            name="linthresh",
            from_subplot_config=from_subplot_config,
        )
        self.linscale = load_setting(
            param=linscale,
            section="colormap",
            name="linscale",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(
        cls,
        cmap: str = None,
        norm: str = None,
        norm_max: float = None,
        norm_min: float = None,
        linthresh: float = None,
        linscale: float = None,
    ):
        return ColorMap(
            cmap=cmap,
            norm=norm,
            norm_min=norm_min,
            norm_max=norm_max,
            linthresh=linthresh,
            linscale=linscale,
            from_subplot_config=True,
        )

    def norm_from_array(self, array: np.ndarray):
        """
        Returns the `Normalization` object which scales of the colormap, using the input min / max normalization \
        values.

        If norm_min / norm_max are not supplied, the minimum / maximum values of the array of data_type are used.

        Parameters
        -----------
        array : data_type.array.aa.Scaled
            The 2D array of data_type which is plotted.
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
                "{linear, log, symmetric_log}"
            )


class ColorBar(AbstractMatObj):
    def __init__(
        self,
        ticksize: int = None,
        fraction: float = None,
        pad: float = None,
        tick_labels: typing.Union[typing.List[float]] = None,
        tick_values: typing.Union[typing.List[float]] = None,
        from_subplot_config: bool = False,
    ):
        """
        The settings used to set up the Matplotlib Colorbar.

        This object wraps the following Matplotlib methods:

        - plt.colorbar: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.colorbar.html

        Parameters
        ----------
        ticksize : int
            The font size of the colorbar ticks.
        fraction : float
            The fraction of the figure the colorbar occupies (equivalent to plt.colorbar(fraction=fraction).
        pad : float
            The padding around the colorbar in the figure (equivalent to plt.colorbar(pad=pad).
        tick_labels : [float]
            Manually override the colorbar tick labels to display the labels as the input list of float.
        tick_values : [float]
            If the colorbar tick labels are manually specified the locations on the colorbar they appear running 0 -> 1.
        from_subplot_config : bool
            If True, load unspecified settings from the figures.ini visualization config, else use subplots.ini.
        """

        self.from_subplot_config = from_subplot_config

        self.ticksize = load_setting(
            param=ticksize,
            section="colorbar",
            name="ticksize",
            from_subplot_config=from_subplot_config,
        )
        self.fraction = load_setting(
            param=fraction,
            section="colorbar",
            name="fraction",
            from_subplot_config=from_subplot_config,
        )
        self.pad = load_setting(
            param=pad,
            section="colorbar",
            name="pad",
            from_subplot_config=from_subplot_config,
        )

        self.tick_values = tick_values
        self.tick_labels = tick_labels

    @classmethod
    def sub(
        cls,
        ticksize: int = None,
        fraction: float = None,
        pad: float = None,
        tick_labels: typing.Union[typing.List[float]] = None,
        tick_values: typing.Union[typing.List[float]] = None,
    ):
        return ColorBar(
            ticksize=ticksize,
            fraction=fraction,
            pad=pad,
            tick_values=tick_values,
            tick_labels=tick_labels,
            from_subplot_config=True,
        )

    def set(self):
        """
        Setup the colorbar of the figure, specifically its ticksize, figure layout and optionally overriding
        the tick labels with manual inputs.
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

    def set_with_values(self, cmap: str, color_values: np.ndarray):
        """
        Set up the colorbar with a set of already known color values.

        This method is used for producing the color bar on a Voronoi mesh plot, which is unable to use the in-built
        Matplotlib colorbar method.

        Parameters
        ----------
        cmap : str
            The colormap used to map normalized data values to RGBA colors (see
            https://matplotlib.org/3.3.1/api/cm_api.html).
        color_values : np.ndarray
            The values of the pixels on the Voronoi mesh which are used to create the colorbar.
        """

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


class Ticks(AbstractMatObj):
    def __init__(
        self,
        ysize: int = None,
        xsize: int = None,
        y_manual: typing.Union[typing.List[float]] = None,
        x_manual: typing.Union[typing.List[float]] = None,
        from_subplot_config: bool = False,
    ):
        """
        The settings used to customize the figure's y and x ticks.

        This object wraps the following Matplotlib methods:

        - plt.tick_params
        - plt.yticks: https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.yticks.html
        - plt.xticks: https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.xticks.html

        Parameters
        ----------
        ysize : int
            The font size of y-axis ticks.
        xsize : int
            The font size of x-axis ticks.
        y_manual : [float]
            Manually override the y-axis tick labels to display the labels as the input list of floats.
        x_manual : [float]
            Manually override the x-axis tick labels to display the labels as the input list of floats.
        from_subplot_config : bool
            If True, load unspecified settings from the figures.ini visualization config, else use subplots.ini.
        """
        self.from_subplot_config = from_subplot_config

        self.ysize = load_setting(
            param=ysize,
            section="ticks",
            name="ysize",
            from_subplot_config=from_subplot_config,
        )
        self.xsize = load_setting(
            param=xsize,
            section="ticks",
            name="xsize",
            from_subplot_config=from_subplot_config,
        )

        self.y_manual = y_manual
        self.x_manual = x_manual

    @classmethod
    def sub(
        cls,
        ysize: int = None,
        xsize: int = None,
        y_manual: typing.Union[typing.List[float]] = None,
        x_manual: typing.Union[typing.List[float]] = None,
    ):
        return Ticks(
            ysize=ysize,
            xsize=xsize,
            y_manual=y_manual,
            x_manual=x_manual,
            from_subplot_config=True,
        )

    def set_yticks(
        self,
        array: arrays.Array,
        ymin: float,
        ymax: float,
        units: Units,
        symmetric_around_centre: bool = False,
    ):
        """
        Use the extent of an input `Array` object to set the y ticks of a figure.

        Parameters
        -----------
        array : arrays.Array
            The 2D array of data which is plotted.
        ymin : float
            The minimum y value of the ticks.
        ymax : float
            The maximum y value of the ticks.
        units : Units
            The units of the figure.
        symmetric_around_centre : bool
            If True, the figure is plotted symmetrically around a central value, which is the default behaviour of
            Matplotlib. This is used for plotting `Mapper`'s.
        """

        plt.tick_params(labelsize=self.ysize)

        if symmetric_around_centre:
            return

        yticks = np.linspace(ymin, ymax, 5)

        if self.y_manual is not None:
            ytick_labels = np.asarray(self.y_manual)
            yticks = np.linspace(ymin, ymax, len(self.y_manual))
        elif not units.use_scaled:
            ytick_labels = np.linspace(0, array.shape_2d[0], 5).astype("int")
        elif (units.use_scaled and units.conversion_factor is None) or not units.in_kpc:
            ytick_labels = np.round(np.linspace(ymin, ymax, 5), 2)
        elif units.use_scaled and units.conversion_factor is not None:
            ytick_labels = np.round(
                np.linspace(
                    ymin * units.conversion_factor, ymax * units.conversion_factor, 5
                ),
                2,
            )

        else:
            raise exc.PlottingException(
                "The y and y ticks cannot be set using the input options."
            )

        plt.yticks(ticks=yticks, labels=ytick_labels)

    def set_xticks(
        self,
        array: arrays.Array,
        xmin: float,
        xmax: float,
        units: Units,
        symmetric_around_centre: bool = False,
    ):
        """
        Use the extent of an input `Array` object to set the x ticks of a figure.

        Parameters
        -----------
        array : arrays.Array
            The 2D array of data which is plotted.
        xmin : float
            The minimum x value of the ticks.
        xmax : float
            The maximum x value of the ticks.
        units : Units
            The units of the figure.
        symmetric_around_centre : bool
            If True, the figure is plotted symmetrically around a central value, which is the default behaviour of
            Matplotlib. This is used for plotting `Mapper`'s.
        """

        plt.tick_params(labelsize=self.xsize)

        if symmetric_around_centre:
            return

        xticks = np.linspace(xmin, xmax, 5)

        if self.x_manual is not None:
            xtick_labels = np.asarray(self.x_manual)
            xticks = np.linspace(xmin, xmax, len(self.x_manual))
        elif not units.use_scaled:
            xtick_labels = np.linspace(0, array.shape_2d[0], 5).astype("int")
        elif (units.use_scaled and units.conversion_factor is None) or not units.in_kpc:
            xtick_labels = np.round(np.linspace(xmin, xmax, 5), 2)
        elif units.use_scaled and units.conversion_factor is not None:
            xtick_labels = np.round(
                np.linspace(
                    xmin * units.conversion_factor, xmax * units.conversion_factor, 5
                ),
                2,
            )

        else:
            raise exc.PlottingException(
                "The y and y ticks cannot be set using the input options."
            )

        plt.xticks(ticks=xticks, labels=xtick_labels)


class Labels(AbstractMatObj):
    def __init__(
        self,
        title: str = None,
        yunits: str = None,
        xunits: str = None,
        titlesize: int = None,
        ysize: int = None,
        xsize: int = None,
        from_subplot_config: bool = False,
    ):
        """The settings used to customize the figure's title and y and x labels.

        This object wraps the following Matplotlib methods:

        - plt.title: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.title.html
        - plt.ylabel: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.ylabel.html
        - plt.xlabel: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.xlabel.html

        The title and y and x labels will automatically be set if not specified, using the name of the functin
        used to plot the image and the _Unit_'s. object.

        Parameters
        ----------
        title : str
            Manually specify the figure's title text.
        ylabel : int
            Manually specify the figure's y label text.
        xlabel : int
            Manually specify the figure's xlabel text.
        titlesize : int
            The title text fontsize.
        ysize : int
            The ylabel text fontsize.
        xsize : int
            The xlabel text fontsize.
        """
        self.from_subplot_config = from_subplot_config

        self.title = title
        self._yunits = yunits
        self._xunits = xunits

        self.titlesize = load_setting(
            param=titlesize,
            section="labels",
            name="titlesize",
            from_subplot_config=from_subplot_config,
        )
        self.ysize = load_setting(
            param=ysize,
            section="labels",
            name="ysize",
            from_subplot_config=from_subplot_config,
        )
        self.xsize = load_setting(
            param=xsize,
            section="labels",
            name="xsize",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(
        cls,
        title: str = None,
        yunits: str = None,
        xunits: str = None,
        titlesize: int = None,
        ysize: int = None,
        xsize: int = None,
    ):
        return Labels(
            title=title,
            yunits=yunits,
            xunits=xunits,
            titlesize=titlesize,
            ysize=ysize,
            xsize=xsize,
            from_subplot_config=True,
        )

    def title_from_func(self, func: Callable) -> str:
        """If a title is not manually specified use the name of the function plotting the image to set the title.

        Parameters
        ----------
        func : func
           The function plotting the image.
        """
        if self.title is None:

            return func.__name__.capitalize()

        else:

            return self.title

    def yunits_from_func(self, func: Callable) -> str:
        """
        If the y label is not manually specified use the function plotting the image to y label, assuming that it
        represents spatial units.

        Parameters
        ----------
        func : func
           The function plotting the image.
        """

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

    def xunits_from_func(self, func: Callable) -> str:
        """
        If the x label is not manually specified use the function plotting the image to x label, assuming that it
        represents spatial units.

        Parameters
        ----------
        func : func
           The function plotting the image.
        """
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

    def yunits_from_units(self, units: Units) -> str:
        """
        Returns the units of the y-axis to create the y label text if it is not manually specified.

             Parameters
             ----------
             unit : Units
                The units of the image that is plotted which informs the appropriate y label text.
        """
        if self._yunits is None:

            if units.in_kpc is not None:
                if units.in_kpc:
                    return "kpc"
                else:
                    return "arcsec"

            if units.use_scaled:
                return "scaled"
            else:
                return "pixels"

        else:

            return self._yunits

    def xunits_from_units(self, units: Units) -> str:
        """
        Returns the units of the x-axis to create the x label text if it is not manually specified.

             Parameters
             ----------
             unit : Units
                The units of the image that is plotted which informs the the appropriate x label text.
        """
        if self._xunits is None:

            if units.in_kpc is not None:
                if units.in_kpc:
                    return "kpc"
                else:
                    return "arcsec"

            if units.use_scaled:
                return "scaled"
            else:
                return "pixels"

        else:

            return self._xunits

    def set_title(self):
        """Set the title and title size of the figure."""
        plt.title(label=self.title, fontsize=self.titlesize)

    def set_yunits(self, units: Units, include_brackets: bool):
        """
        Set the y labels of the figure, including the fontsize.

        The y labels are always the distance scales, thus the labels are either arc-seconds or kpc and depending on
        the unit_label the figure is plotted in.

        Parameters
        -----------
         unit : Units
            The units of the image that is plotted which informs the appropriate y label text.
        include_brackets : bool
            Whether to include brackets around the y label text of the units.
        """
        if include_brackets:
            plt.ylabel(
                "y (" + self.yunits_from_units(units=units) + ")", fontsize=self.ysize
            )
        else:
            plt.ylabel(self.yunits_from_units(units=units), fontsize=self.ysize)

    def set_xunits(self, units: Units, include_brackets: bool):
        """
        Set the x labels of the figure, including the fontsize.

        The x labels are always the distance scales, thus the labels are either arc-seconds or kpc and depending on
        the unit_label the figure is plotted in.

        Parameters
        -----------
         unit : Units
            The units of the image that is plotted which informs the appropriate x label text.
        include_brackets : bool
            Whether to include brackets around the x label text of the units.
        """
        if include_brackets:
            plt.xlabel(
                "x (" + self.xunits_from_units(units=units) + ")", fontsize=self.xsize
            )
        else:
            plt.xlabel(self.xunits_from_units(units=units), fontsize=self.xsize)


class Legend(AbstractMatObj):
    def __init__(
        self,
        include: bool = None,
        fontsize: int = None,
        from_subplot_config: bool = False,
    ):
        """
        The settings used to include a legend on a figure and customize it.

        This object wraps the following Matplotlib methods:

        - plt.legend: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html

        The title and y and x labels will automatically be set if not specified, using the name of the functin
        used to plot the image and the _Unit_'s. object.

        Parameters
        ----------
        include : bool
            Whether to include a legend on the figure oor not.
        fontsize : int
            The size of the font on the figure.
        """

        self.from_subplot_config = from_subplot_config

        self.include = load_setting(
            param=include,
            section="legend",
            name="include",
            from_subplot_config=from_subplot_config,
        )
        self.fontsize = load_setting(
            param=fontsize,
            section="legend",
            name="fontsize",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, include=None, fontsize=None):
        return Legend(include=include, fontsize=fontsize, from_subplot_config=True)

    def set(self):
        if self.include:
            plt.legend(fontsize=self.fontsize)


class Output(AbstractMatObj):
    def __init__(
        self,
        path: str = None,
        filename: str = None,
        format: str = None,
        bypass: bool = False,
    ):
        """
        An object for outputting a figure to the display or hard-disc.

        This object wraps the following Matplotlib methods:

        - plt.show: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.show.html
        - plt.savefig: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.savefig.html

        The default behaviour is the display the figure on the computer screen, as opposed to outputting to hard-disc
        as a file.

        Parameters
        ----------
        path : str
            If the figure is output to hard-disc the path of the folder it is saved to.
        filename : str
            If the figure is output to hard-disc the filename used to save it.
        format : str
            The format of the output, 'show' displays on the computer screen, 'png' outputs to .png, 'fits' outputs to
            .fits format.
        bypass : bool
            Whether to bypass the plt.show or plt.savefig methods, used when plotting a subplot.
        """
        self.path = path

        if path is not None and path:
            try:
                os.makedirs(path)
            except FileExistsError:
                pass

        self.filename = filename
        self._format = format
        self.bypass = bypass

    @property
    def format(self) -> str:
        if self._format is None:
            return "show"
        else:
            return self._format

    def filename_from_func(self, func) -> str:
        """If a filename is not manually specified use the name of the function plotting the image to set it.

        Parameters
        ----------
        func : func
           The function plotting the image.
        """
        if self.filename is None:
            return func.__name__
        else:
            return self.filename

    def to_figure(self, structure: abstract_structure.AbstractStructure):
        """Output the figure, either as an image on the screen or to the hard-disk as a .png or .fits file.

        Parameters
        -----------
        structure : abstract_structure.AbstractStructure
            The 2D array of image to be output, required for outputting the image as a fits file.
        """
        if not self.bypass:
            if self.format == "show":
                plt.show()
            elif self.format == "png":
                plt.savefig(
                    path.join(self.path, f"{self.filename}.png"), bbox_inches="tight"
                )
            elif self.format == "fits":
                if structure is not None:
                    structure.output_to_fits(
                        file_path=path.join(self.path, f"{self.filename}.fits"),
                        overwrite=True,
                    )

    def subplot_to_figure(self):
        """Output a subhplot figure, either as an image on the screen or to the hard-disk as a .png or .fits file."""
        if self.format == "show":
            plt.show()
        elif self.format == "png":
            plt.savefig(
                path.join(self.path, f"{self.filename}.png"), bbox_inches="tight"
            )


def remove_spaces_and_commas_from_colors(colors):

    colors = [color.strip(",") for color in colors]
    colors = [color.strip(" ") for color in colors]
    return list(filter(None, colors))


class Scatterer(AbstractMatObj):
    def __init__(
        self,
        size: int = None,
        marker: str = None,
        colors: typing.List[str] = None,
        section: str = None,
        from_subplot_config=False,
    ):
        """
        An object for scattering an input set of grid points, for example (y,x) coordinates or a data structures
        representing 2D (y,x) coordinates like a `Grid` or `GridIrregular`. If the object groups (y,x) coordinates
        they are plotted with colors according to their group.

        This object wraps the following Matplotlib methods:

        - plt.scatter: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html

        The default behaviour is the display the figure on the computer screen, as opposed to outputting to hard-disc
        as a file.

        Parameters
        ----------
        size : int
            The size of the points of the scatter plot.
        marker : str
            The style of the scattered point's mark (e.g. 'x', 'o', '.', etc.)
        colors : [str]
            The list of colors that the grouped plotted coordinates cycle through.
        """
        self.from_subplot_config = from_subplot_config

        self.size = load_setting(
            param=size,
            section=section,
            name="size",
            from_subplot_config=from_subplot_config,
        )
        self.marker = load_setting(
            param=marker,
            section=section,
            name="marker",
            from_subplot_config=from_subplot_config,
        )
        self.colors = load_setting(
            param=colors,
            section=section,
            name="colors",
            from_subplot_config=from_subplot_config,
        )

        self.colors = remove_spaces_and_commas_from_colors(colors=self.colors)

        if isinstance(self.colors, str):
            self.colors = [self.colors]

    def scatter_grid(self, grid):

        plt.scatter(
            y=np.asarray(grid)[:, 0],
            x=np.asarray(grid)[:, 1],
            s=self.size,
            c=self.colors[0],
            marker=self.marker,
        )

    def scatter_colored_grid(self, grid, color_array, cmap):

        plt.scatter(
            y=np.asarray(grid)[:, 0],
            x=np.asarray(grid)[:, 1],
            s=self.size,
            c=color_array,
            marker=self.marker,
            cmap=cmap,
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

    def scatter_coordinates(self, coordinates):

        if len(coordinates) == 0:
            return

        color = itertools.cycle(self.colors)

        for coordinate_group in coordinates.in_grouped_list:

            plt.scatter(
                y=np.asarray(coordinate_group)[:, 0],
                x=np.asarray(coordinate_group)[:, 1],
                s=self.size,
                c=next(color),
                marker=self.marker,
            )


class OriginScatterer(Scatterer):
    def __init__(self, size=None, marker=None, colors=None, from_subplot_config=False):
        """
        An object for scattering an input set of grid points, described fully in the `Scatterer` class.

        The `OriginScatterer` is used specifically to plot the (y,x) coordinate origin of a data structure. It uses
        its own settings (e.g, `size`, `marker`, etc.) in `config/visualize/mat_obj` config files.

        Parameters
        ----------
        size : int
            The size of the (y,x) origin of the scatter plot.
        marker : str
            The style of the scattered point's mark (e.g. 'x', 'o', '.', etc.)
        colors : [str]
            The list of colors that the grouped plotted coordinates cycle through.
        """
        super(OriginScatterer, self).__init__(
            size=size,
            marker=marker,
            colors=colors,
            section="origin",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, size=None, marker=None, colors=None):
        return OriginScatterer(
            size=size, marker=marker, colors=colors, from_subplot_config=True
        )


class MaskScatterer(Scatterer):
    def __init__(self, size=None, marker=None, colors=None, from_subplot_config=False):
        """
        An object for scattering an input set of grid points, described fully in the `Scatterer` class.

        The `MaskScatterer` is used specifically to plot a 2D mask over an image, using the mask's (y,x) grid of edge
        coordinates.

        It uses its own settings (e.g, `size`, `marker`, etc.) in `config/visualize/mat_obj` config files.

        Parameters
        ----------
        size : int
            The size of the (y,x) origin of the scatter plot.
        marker : str
            The style of the scattered point's mark (e.g. 'x', 'o', '.', etc.)
        colors : [str]
            The list of colors that the grouped plotted coordinates cycle through.
        """
        super(MaskScatterer, self).__init__(
            size=size,
            marker=marker,
            colors=colors,
            section="mask",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, size=None, marker=None, colors=None):
        return MaskScatterer(
            size=size, marker=marker, colors=colors, from_subplot_config=True
        )


class BorderScatterer(Scatterer):
    def __init__(self, size=None, marker=None, colors=None, from_subplot_config=False):
        """
        An object for scattering an input set of grid points, described fully in the `Scatterer` class.

        The `BorderScatterer` is used specifically to plot a 2D border over an image, using the border's (y,x) grid of
        border coordinates.

        It uses its own settings (e.g, `size`, `marker`, etc.) in `config/visualize/mat_obj` config files.

        Parameters
        ----------
        size : int
            The size of the (y,x) origin of the scatter plot.
        marker : str
            The style of the scattered point's mark (e.g. 'x', 'o', '.', etc.)
        colors : [str]
            The list of colors that the grouped plotted coordinates cycle through.
        """
        super(BorderScatterer, self).__init__(
            size=size,
            marker=marker,
            colors=colors,
            section="border",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, size=None, marker=None, colors=None):
        return BorderScatterer(
            size=size, marker=marker, colors=colors, from_subplot_config=True
        )


class GridScatterer(Scatterer):
    def __init__(self, size=None, marker=None, colors=None, from_subplot_config=False):
        """
        An object for scattering an input set of grid points, described fully in the `Scatterer` class.

        The `GridScatterer` is used specifically to plot a 2D grid of points.

        It uses its own settings (e.g, `size`, `marker`, etc.) in `config/visualize/mat_obj` config files.

        Parameters
        ----------
        size : int
            The size of the (y,x) origin of the scatter plot.
        marker : str
            The style of the scattered point's mark (e.g. 'x', 'o', '.', etc.)
        colors : [str]
            The list of colors that the grouped plotted coordinates cycle through.
        """
        super(GridScatterer, self).__init__(
            size=size,
            marker=marker,
            colors=colors,
            section="grid",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, size=None, marker=None, colors=None):
        return GridScatterer(
            size=size, marker=marker, colors=colors, from_subplot_config=True
        )


class PositionsScatterer(Scatterer):
    def __init__(self, size=None, marker=None, colors=None, from_subplot_config=False):
        """
        An object for scattering an input set of grid points, described fully in the `Scatterer` class.

        The `PositionsScatterer` is used specifically to plot a 2D irregular grid of (y,x) coordinates that are marked
        as the `positions` of a data structure. These will be colored according to grouping, if grouped.

        It uses its own settings (e.g, `size`, `marker`, etc.) in `config/visualize/mat_obj` config files.

        Parameters
        ----------
        size : int
            The size of the (y,x) origin of the scatter plot.
        marker : str
            The style of the scattered point's mark (e.g. 'x', 'o', '.', etc.)
        colors : [str]
            The list of colors that the grouped plotted coordinates cycle through.
        """
        super(PositionsScatterer, self).__init__(
            size=size,
            marker=marker,
            colors=colors,
            section="positions",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, size=None, marker=None, colors=None):
        return PositionsScatterer(
            size=size, marker=marker, colors=colors, from_subplot_config=True
        )


class IndexScatterer(Scatterer):
    def __init__(self, size=None, marker=None, colors=None, from_subplot_config=False):
        """
        An object for scattering an input set of grid points, described fully in the `Scatterer` class.

        The `IndexScatterer` is used specifically to plot a set of pixel indexes, which have been converted to a 2D
        grid of (y,x) coordinates from their original 1D or 2D image pixels. If the indexes were lists of grouped
        indexes they should be converted to a `GridIrregularGrouped` object so they are colored acoording to grouping.

        It uses its own settings (e.g, `size`, `marker`, etc.) in `config/visualize/mat_obj` config files.

        Parameters
        ----------
        size : int
            The size of the (y,x) origin of the scatter plot.
        marker : str
            The style of the scattered point's mark (e.g. 'x', 'o', '.', etc.)
        colors : [str]
            The list of colors that the grouped plotted coordinates cycle through.
        """
        super(IndexScatterer, self).__init__(
            size=size,
            marker=marker,
            colors=colors,
            section="index",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, size=None, marker=None, colors=None):
        return IndexScatterer(
            size=size, marker=marker, colors=colors, from_subplot_config=True
        )


class PixelizationGridScatterer(Scatterer):
    def __init__(self, size=None, marker=None, colors=None, from_subplot_config=False):
        """
        An object for scattering an input set of grid points, described fully in the `Scatterer` class.

        The `PixelizationScatterer` is used specifically to plot an irregular grid of (y,x) coordinates correspinding
        to the grid of a pixelization.

        It uses its own settings (e.g, `size`, `marker`, etc.) in `config/visualize/mat_obj` config files.

        Parameters
        ----------
        size : int
            The size of the (y,x) origin of the scatter plot.
        marker : str
            The style of the scattered point's mark (e.g. 'x', 'o', '.', etc.)
        colors : [str]
            The list of colors that the grouped plotted coordinates cycle through.
        """
        super(PixelizationGridScatterer, self).__init__(
            size=size,
            marker=marker,
            colors=colors,
            section="pixelization_grid",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, size=None, marker=None, colors=None):
        return PixelizationGridScatterer(
            size=size, marker=marker, colors=colors, from_subplot_config=True
        )


class VectorQuiverer(AbstractMatObj):
    def __init__(
        self,
        headlength=None,
        pivot=None,
        linewidth=None,
        units=None,
        angles=None,
        headwidth=None,
        alpha=None,
        section=None,
        from_subplot_config=False,
    ):

        if section is None:
            section = "vector_quiverer"

        self.from_subplot_config = from_subplot_config

        self.headlength = load_setting(
            param=headlength,
            section=section,
            name="headlength",
            from_subplot_config=from_subplot_config,
        )
        self.pivot = load_setting(
            param=pivot,
            section=section,
            name="pivot",
            from_subplot_config=from_subplot_config,
        )
        self.linewidth = load_setting(
            param=linewidth,
            section=section,
            name="linewidth",
            from_subplot_config=from_subplot_config,
        )
        self.units = load_setting(
            param=units,
            section=section,
            name="units",
            from_subplot_config=from_subplot_config,
        )
        self.angles = load_setting(
            param=angles,
            section=section,
            name="angles",
            from_subplot_config=from_subplot_config,
        )
        self.headwidth = load_setting(
            param=headwidth,
            section=section,
            name="headwidth",
            from_subplot_config=from_subplot_config,
        )
        self.alpha = load_setting(
            param=alpha,
            section=section,
            name="alpha",
            from_subplot_config=from_subplot_config,
        )

    def quiver_vector_field(self, vector_field: vector_fields.VectorFieldIrregular):

        plt.quiver(
            vector_field.grid[:, 1],
            vector_field.grid[:, 0],
            vector_field[:, 1],
            vector_field[:, 0],
            headlength=self.headlength,
            pivot=self.pivot,
            linewidth=self.linewidth,
            units=self.units,
            angles=self.angles,
            headwidth=self.headwidth,
            alpha=self.alpha,
        )

    @classmethod
    def sub(
        cls,
        headlength=None,
        pivot=None,
        linewidth=None,
        units=None,
        angles=None,
        headwidth=None,
        alpha=None,
        cmap=None,
        section=None,
    ):
        return VectorQuiverer(
            headlength=headlength,
            pivot=pivot,
            linewidth=linewidth,
            units=units,
            angles=angles,
            headwidth=headwidth,
            alpha=alpha,
            cmap=cmap,
            section=section,
            from_subplot_config=True,
        )


class Patcher(AbstractMatObj):
    def __init__(
        self, facecolor=None, edgecolor=None, section=None, from_subplot_config=False
    ):

        if section is None:
            section = "patcher"

        self.from_subplot_config = from_subplot_config

        self.facecolor = load_setting(
            param=facecolor,
            section=section,
            name="facecolor",
            from_subplot_config=from_subplot_config,
        )

        if self.facecolor is None:
            self.facecolor = "none"

        self.edgecolor = load_setting(
            param=edgecolor,
            section=section,
            name="edgecolor",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, facecolor=None, edgecolor=None, section=None):
        return Patcher(
            facecolor=facecolor,
            edgecolor=edgecolor,
            section=section,
            from_subplot_config=True,
        )

    def add_patches(self, patches):

        patch_collection = PatchCollection(
            patches=patches, facecolors=self.facecolor, edgecolors=self.edgecolor
        )

        plt.gcf().gca().add_collection(patch_collection)


class Liner(AbstractMatObj):
    def __init__(
        self,
        width=None,
        style=None,
        colors=None,
        pointsize=None,
        section=None,
        from_subplot_config=False,
    ):

        if section is None:
            section = "liner"

        self.from_subplot_config = from_subplot_config

        self.width = load_setting(
            param=width,
            section=section,
            name="width",
            from_subplot_config=from_subplot_config,
        )
        self.style = load_setting(
            param=style,
            section=section,
            name="style",
            from_subplot_config=from_subplot_config,
        )
        self.colors = load_setting(
            param=colors,
            section=section,
            name="colors",
            from_subplot_config=from_subplot_config,
        )

        self.colors = remove_spaces_and_commas_from_colors(colors=self.colors)

        self.pointsize = load_setting(
            param=pointsize,
            section=section,
            name="pointsize",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, width=None, style=None, colors=None, pointsize=None, section=None):
        return Liner(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            section=section,
            from_subplot_config=True,
        )

    def draw_y_vs_x(self, y, x, plot_axis_type, label=None):

        if plot_axis_type == "linear":
            plt.plot(x, y, c=self.colors[0], lw=self.width, ls=self.style, label=label)
        elif plot_axis_type == "semilogy":
            plt.semilogy(
                x, y, c=self.colors[0], lw=self.width, ls=self.style, label=label
            )
        elif plot_axis_type == "loglog":
            plt.loglog(
                x, y, c=self.colors[0], lw=self.width, ls=self.style, label=label
            )
        elif plot_axis_type == "scatter":
            plt.scatter(x, y, c=self.colors[0], s=self.pointsize, label=label)
        else:
            raise exc.PlottingException(
                "The plot_axis_type supplied to the plotter is not a valid string (must be linear "
                "{semilogy, loglog})"
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

    def draw_grid(self, grid):
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

        plt.plot(
            np.asarray(grid)[:, 1],
            np.asarray(grid)[:, 0],
            c=self.colors[0],
            lw=self.width,
            ls=self.style,
        )

    def draw_rectangular_grid_lines(self, extent, shape_2d):

        ys = np.linspace(extent[2], extent[3], shape_2d[1] + 1)
        xs = np.linspace(extent[0], extent[1], shape_2d[0] + 1)

        # grid lines
        for x in xs:
            plt.plot(
                [x, x],
                [ys[0], ys[-1]],
                color=self.colors[0],
                lw=self.width,
                ls=self.style,
            )
        for y in ys:
            plt.plot(
                [xs[0], xs[-1]],
                [y, y],
                color=self.colors[0],
                lw=self.width,
                ls=self.style,
            )

    def draw_coordinates(self, coordinates):
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

        if len(coordinates) == 0:
            return

        color = itertools.cycle(self.colors)

        for coordinate_group in coordinates.in_grouped_list:

            plt.plot(
                np.asarray(coordinate_group)[:, 1],
                np.asarray(coordinate_group)[:, 0],
                c=next(color),
                lw=self.width,
                ls=self.style,
            )


class ParallelOverscanLiner(Liner):
    def __init__(
        self,
        width=None,
        style=None,
        colors=None,
        pointsize=None,
        from_subplot_config=False,
    ):

        super(ParallelOverscanLiner, self).__init__(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            section="parallel_overscan",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, width=None, style=None, colors=None, pointsize=None):
        return ParallelOverscanLiner(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            from_subplot_config=True,
        )


class SerialPrescanLiner(Liner):
    def __init__(
        self,
        width=None,
        style=None,
        colors=None,
        pointsize=None,
        from_subplot_config=False,
    ):

        super(SerialPrescanLiner, self).__init__(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            section="serial_prescan",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, width=None, style=None, colors=None, pointsize=None):
        return SerialPrescanLiner(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            from_subplot_config=True,
        )


class SerialOverscanLiner(Liner):
    def __init__(
        self,
        width=None,
        style=None,
        colors=None,
        pointsize=None,
        from_subplot_config=False,
    ):

        super(SerialOverscanLiner, self).__init__(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            section="serial_overscan",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, width=None, style=None, colors=None, pointsize=None):
        return SerialOverscanLiner(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            from_subplot_config=True,
        )


class ArrayOverlayer(AbstractMatObj):
    def __init__(self, alpha=None, section=None, from_subplot_config=False):

        if section is None:
            section = "array_overlayer"

        self.from_subplot_config = from_subplot_config

        self.alpha = load_setting(
            param=alpha,
            section=section,
            name="alpha",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, alpha, section=None):
        return ArrayOverlayer(alpha=alpha, section=section, from_subplot_config=True)

    def overlay_array(self, array_overlay, figure):

        aspect_overlay = figure.aspect_from_shape_2d(shape_2d=array_overlay.shape_2d)
        extent_overlay = array_overlay.extent_of_zoomed_array(buffer=0)

        plt.imshow(
            X=array_overlay.in_2d,
            aspect=aspect_overlay,
            extent=extent_overlay,
            alpha=self.alpha,
        )


class VoronoiDrawer(AbstractMatObj):
    def __init__(
        self, edgewidth=None, edgecolor=None, alpha=None, from_subplot_config=False
    ):

        self.from_subplot_config = from_subplot_config

        self.edgewidth = load_setting(
            param=edgewidth,
            section="voronoi_drawer",
            name="edgewidth",
            from_subplot_config=from_subplot_config,
        )
        self.edgecolor = load_setting(
            param=edgecolor,
            section="voronoi_drawer",
            name="edgecolor",
            from_subplot_config=from_subplot_config,
        )
        self.alpha = load_setting(
            param=alpha,
            section="voronoi_drawer",
            name="alpha",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, edgewidth=None, edgecolor=None, alpha=None):
        return VoronoiDrawer(
            edgewidth=edgewidth,
            edgecolor=edgecolor,
            alpha=alpha,
            from_subplot_config=True,
        )

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
                lw=self.edgewidth,
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
