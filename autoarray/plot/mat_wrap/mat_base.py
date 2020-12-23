from autoconf import conf
import matplotlib

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
from os import path
import os
import colorcet
import configparser
import typing

from autoarray.structures import abstract_structure, arrays
from autoarray import exc

import inspect


class Units:
    def __init__(
        self,
        use_scaled: bool = None,
        conversion_factor: float = None,
        in_kpc: bool = None,
    ):
        """
        This object controls the units of a plotted figure, and performs multiple tasks when making the plot:

        1) Species the units of the plot (e.g. meters, kilometers) and contains a conversion factor which is used to
           converts the plotted data from its true units (e.g. meters) to the units to plotted (e.g. kilometeters).
           Pixel units can also be used if use_scaled=False.

        2) Uses the conversion above to manually override the yticks and xticks of the figure, so it appears in the
           converted units.

        3) Sets the ylabel and xlabel to include a string containing the units.

        Parameters
        ----------
        use_scaled : bool
            If True, plot the 2D data with y and x ticks corresponding to its scaled coordinates (its `pixel_scales`
            attribute is used as the conversion factor). If `False` plot them in pixel units.
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


class AbstractMatBase:
    def __init__(self, use_subplot_defaults, kwargs):

        self.use_subplot_defaults = use_subplot_defaults

        if not use_subplot_defaults:

            config_dict = conf.instance["visualize"][self.config_folder][
                self.__class__.__name__
            ]["figure"]._dict

        else:

            config_dict = conf.instance["visualize"][self.config_folder][
                self.__class__.__name__
            ]["subplot"]._dict

        self.kwargs = {**config_dict, **kwargs}

    @property
    def config_folder(self):
        return "mat_base"

    def kwargs_of_method(self, method_name, cls_name=None):

        if cls_name is None:
            cls_name = self.__class__.__name__

        args = conf.instance["visualize"][self.config_folder][cls_name]["args"][
            method_name
        ]

        args = args.replace(" ", "")
        args = args.split(",")

        return {key: self.kwargs[key] for key in args if key in self.kwargs}


class Figure(AbstractMatBase):
    def __init__(self, use_subplot_defaults: bool = False, **kwargs):
        """
        Sets up the Matplotlib figure before plotting (this is used when plotting individual figures and subplots).

        This object wraps the following Matplotlib methods:

        - plt.figure: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.figure.html
        - plt.close: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.close.html

        It also controls the aspect ratio of the figure plotted.

        Parameters
        ----------
        use_subplot_defaults : bool
            `WrapMat` objects load settings from the [figure] section of its .ini config file by default. If
            `use_subplot_defaults=True` settings from the [subplot] section are loaded instead.
        """

        super().__init__(use_subplot_defaults=use_subplot_defaults, kwargs=kwargs)

        if self.kwargs["figsize"] == "auto":
            self.kwargs["figsize"] = None
        elif isinstance(self.kwargs["figsize"], str):
            self.kwargs["figsize"] = tuple(
                map(int, self.kwargs["figsize"][1:-1].split(","))
            )

    @property
    def kwargs_figure(self):
        """Creates a kwargs dict of valid inputs of the method `plt.figure` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="figure")

    @property
    def kwargs_imshow(self):
        """Creates a kwargs dict of valid inputs of the method `plt.imshow` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="imshow")

    def aspect_from_shape_2d(self, shape_2d: typing.Union[typing.Tuple[int, int]]):
        """
        Returns the aspect ratio of the figure from the 2D shape of an `Array`.

        This is used to ensure that rectangular arrays are plotted as square figures on sub-plots.

        Parameters
        ----------
        shape_2d : (int, int)
            The two dimensional shape of an `Array` that is to be plotted.
        """
        if isinstance(self.kwargs["aspect"], str):
            if self.kwargs["aspect"] in "square":
                return float(shape_2d[1]) / float(shape_2d[0])

        return self.kwargs["aspect"]

    def open(self):
        """Wraps the Matplotlib method 'plt.figure' for opening a figure."""
        if not plt.fignum_exists(num=1):
            plt.figure(**self.kwargs_figure)

    def close(self):
        """Wraps the Matplotlib method 'plt.close' for closing a figure."""
        if plt.fignum_exists(num=1):
            plt.close()


class Cmap(AbstractMatBase):
    def __init__(
        self, use_subplot_defaults: bool = False, module: str = None, **kwargs
    ):
        """
        Customizes the Matplotlib colormap and its normalization.

        This object wraps the following Matplotlib methods:

        - colors.Linear: https://matplotlib.org/3.3.2/tutorials/colors/colormaps.html
        - colors.LogNorm: https://matplotlib.org/3.3.2/tutorials/colors/colormapnorms.html
        - colors.SymLogNorm: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.colors.SymLogNorm.html

        The cmap that is created is passed into various Matplotlib methods, most notably imshow:

         https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html

        Parameters
        ----------
        module : str
            The module from which the plot is called, which is used to customize the colormap for figures of different
            categories.
          use_subplot_defaults : bool
            `WrapMat` objects load settings from the [figure] section of its .ini config file by default. If
            `use_subplot_defaults=True` settings from the [subplot] section are loaded instead.
        """

        super().__init__(use_subplot_defaults=use_subplot_defaults, kwargs=kwargs)

        if module is not None:

            module_name = module.__name__.split(".")[-1]
            try:
                cmap = conf.instance["visualize"]["general"]["colormaps"][module_name]
            except configparser.NoOptionError:
                cmap = conf.instance["visualize"]["general"]["colormaps"]["default"]

        else:

            cmap = self.kwargs["cmap"]

        try:
            self.kwargs["cmap"] = colorcet.cm[cmap]
        except KeyError:
            pass

    def norm_from_array(self, array: np.ndarray):
        """
        Returns the `Normalization` object which scales of the colormap.

        If vmin / vmax are not manually input by the user, the minimum / maximum values of the data being plotted
        are used.

        Parameters
        -----------
        array : np.ndarray
            The array of data which is to be plotted.
        """

        if self.kwargs["vmin"] is None:
            vmin = array.min()
        else:
            vmin = self.kwargs["vmin"]

        if self.kwargs["vmax"] is None:
            vmax = array.max()
        else:
            vmax = self.kwargs["vmax"]

        if self.kwargs["norm"] in "linear":
            return colors.Normalize(vmin=vmin, vmax=vmax)
        elif self.kwargs["norm"] in "log":
            if vmin == 0.0:
                vmin = 1.0e-4
            return colors.LogNorm(vmin=vmin, vmax=vmax)
        elif self.kwargs["norm"] in "symmetric_log":
            return colors.SymLogNorm(
                vmin=vmin,
                vmax=vmax,
                linthresh=self.kwargs["linthresh"],
                linscale=self.kwargs["linscale"],
            )
        else:
            raise exc.PlottingException(
                "The normalization (norm) supplied to the plotter is not a valid string (must be "
                "{linear, log, symmetric_log}"
            )


class Colorbar(AbstractMatBase):
    def __init__(
        self,
        use_subplot_defaults: bool = False,
        manual_tick_labels: typing.Union[typing.List[float]] = None,
        manual_tick_values: typing.Union[typing.List[float]] = None,
        **kwargs,
    ):
        """
        Customizes the colorbar of the plotted figure.

        This object wraps the following Matplotlib method:

         plt.colorbar: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.colorbar.html

        The colorbar object `cb` that is created is also customized using the following methods:

         cb.set_yticklabels: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.axes.Axes.set_yticklabels.html
         cb.tick_params: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.axes.Axes.tick_params.html

        Parameters
        ----------
        use_subplot_defaults : bool
            `WrapMat` objects load settings from the [figure] section of its .ini config file by default. If
            `use_subplot_defaults=True` settings from the [subplot] section are loaded instead.
        manual_tick_labels : [float]
            Manually override the colorbar tick labels to an input list of float.
        manual_tick_values : [float]
            If the colorbar tick labels are manually specified the locations on the colorbar they appear running 0 -> 1.
         """

        super().__init__(use_subplot_defaults=use_subplot_defaults, kwargs=kwargs)

        self.manual_tick_labels = manual_tick_labels
        self.manual_tick_values = manual_tick_values

    @property
    def kwargs_colorbar(self):
        """Creates a kwargs dict of valid inputs of the method `plt.colorbar` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="colorbar")

    @property
    def kwargs_tick_params(self):
        """Creates a kwargs dict of valid inputs of the method `plt.tick_params` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="tick_params", cls_name="TickParams")

    def set(self):
        """ Set the figure's colorbar, optionally overriding the tick labels and values with manual inputs. """

        if self.manual_tick_values is None and self.manual_tick_labels is None:
            cb = plt.colorbar(**self.kwargs_colorbar)
        elif (
            self.manual_tick_values is not None and self.manual_tick_labels is not None
        ):
            cb = plt.colorbar(ticks=self.manual_tick_values, **self.kwargs_colorbar)
            cb.ax.set_yticklabels(labels=self.manual_tick_labels)
        else:
            raise exc.PlottingException(
                "Only 1 entry of tick_values or tick_labels was input. You must either supply"
                "both the values and labels, or neither."
            )

        cb.ax.tick_params(**self.kwargs_tick_params)

    def set_with_values(self, cmap: str, color_values: np.ndarray):
        """
        Set the figure's colorbar using an array of already known color values.

        This method is used for producing the color bar on a Voronoi mesh plot, which is unable to use the in-built
        Matplotlib colorbar method.

        Parameters
        ----------
        cmap : str
            The colormap used to map normalized data values to RGBA colors (see
            https://matplotlib.org/3.3.2/api/cm_api.html).
        color_values : np.ndarray
            The values of the pixels on the Voronoi mesh which are used to create the colorbar.
        """

        cax = cm.ScalarMappable(cmap=cmap)
        cax.set_array(color_values)

        if self.manual_tick_values is None and self.manual_tick_labels is None:
            plt.colorbar(mappable=cax, **self.kwargs_colorbar)
        elif (
            self.manual_tick_values is not None and self.manual_tick_labels is not None
        ):
            cb = plt.colorbar(
                mappable=cax, ticks=self.manual_tick_values, **self.kwargs
            )
            cb.ax.set_yticklabels(self.manual_tick_labels)


class TickParams(AbstractMatBase):
    def __init__(self, use_subplot_defaults: bool = False, **kwargs):
        """
        The settings used to customize a figure's y and x ticks parameters.

        This object wraps the following Matplotlib methods:

        - plt.tick_params: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.tick_params.html

        Parameters
        ----------
        use_subplot_defaults : bool
            `WrapMat` objects load settings from the [figure] section of its .ini config file by default. If
            `use_subplot_defaults=True` settings from the [subplot] section are loaded instead.
        """
        super().__init__(use_subplot_defaults=use_subplot_defaults, kwargs=kwargs)

    @property
    def kwargs_tick_params(self):
        """Creates a kwargs dict of valid inputs of the method `plt.tick_params` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="tick_params")

    def set(self,):
        """Set the tick_params of the figure using the method `plt.tick_params`."""
        plt.tick_params(**self.kwargs_tick_params)


class AbstractTicks(AbstractMatBase):
    def __init__(
        self,
        use_subplot_defaults: bool = False,
        manual_values: typing.Union[typing.List[float]] = None,
        **kwargs,
    ):
        """
        The settings used to customize a figure's y and x ticks using the `YTicks` and `XTicks` objects.

        This object wraps the following Matplotlib methods:

        - plt.yticks: https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.yticks.html
        - plt.xticks: https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.xticks.html

        Parameters
        ----------
        use_subplot_defaults : bool
            `WrapMat` objects load settings from the [figure] section of its .ini config file by default. If
            `use_subplot_defaults=True` settings from the [subplot] section are loaded instead.
        manual_values : [float]
            Manually override the tick labels to display the labels as the input list of floats.
        """
        super().__init__(use_subplot_defaults=use_subplot_defaults, kwargs=kwargs)

        self.manual_values = manual_values

    @property
    def kwargs_ticks(self):
        """Creates a kwargs dict of valid inputs of the methods `plt.yticks` and `plt.xticks` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="ticks")

    def tick_values_from(
        self, min_value: float, max_value: float, use_defaults: bool = False
    ):
        """
        Calculate the ticks used for the yticks or xticks from input values of the minimum and maximum coordinate
        values of the y and x axis.

        Certain figures may display better using the default ticks. This method has the option `use_defaults` to return
        None such that the defaults are used.

        Parameters
        ----------
        min_value : float
            the minimum value of the ticks that figure is plotted using.
        max_value : float
            the maximum value of the ticks that figure is plotted using.
        use_defaults : bool
            If `True`, the function does not return tick_values such that a plotter uses the default ticks.
        """
        if not use_defaults:
            if self.manual_values is not None:
                return np.linspace(min_value, max_value, len(self.manual_values))
            else:
                return np.linspace(min_value, max_value, 5)

    def tick_values_in_units_from(
        self,
        array: arrays.Array,
        min_value: float,
        max_value: float,
        units: Units,
        use_defaults: bool = False,
    ):
        """
        Calculate the labels used for the yticks or xticks from input values of the minimum and maximum coordinate
        values of the y and x axis.

        The values are converted to the `Units` of the figure, via its conversion factor or using data properties.

        Certain figures may display better using the default ticks. This method has the option `use_defaults` to return
        None such that the defaults are used.

        Parameters
        ----------
        array : arrays.Array
            The array of data that is to be plotted, whose 2D shape is used to determine the tick values in units of
            pixels if this is the units specified by `units`.
        min_value : float
            the minimum value of the ticks that figure is plotted using.
        max_value : float
            the maximum value of the ticks that figure is plotted using.
        units : Units
            The units the tick values are plotted using.
        use_defaults : bool
            If `True`, the function does not return tick_values such that a plotter uses the default ticks.
        """
        if use_defaults:
            return

        if self.manual_values is not None:
            return np.asarray(self.manual_values)
        elif not units.use_scaled:
            return np.linspace(0, array.shape_2d[0], 5).astype("int")
        elif (units.use_scaled and units.conversion_factor is None) or not units.in_kpc:
            return np.round(np.linspace(min_value, max_value, 5), 2)
        elif units.use_scaled and units.conversion_factor is not None:
            return np.round(
                np.linspace(
                    min_value * units.conversion_factor,
                    max_value * units.conversion_factor,
                    5,
                ),
                2,
            )

        else:
            raise exc.PlottingException(
                "The tick labels cannot be computed using the input options."
            )


class YTicks(AbstractTicks):
    def set(
        self,
        array: arrays.Array,
        min_value: float,
        max_value: float,
        units: Units,
        use_defaults: bool = False,
    ):
        """
        Set the y ticks of a figure using the shape of an input `Array` object and input units.

        Parameters
        -----------
        array : arrays.Array
            The 2D array of data which is plotted.
        min_value : float
            the minimum value of the yticks that figure is plotted using.
        max_value : float
            the maximum value of the yticks that figure is plotted using.
        units : Units
            The units of the figure.
        use_defaults : bool
            If True, the figure is plotted symmetrically around a central value, which is the default behaviour of
            Matplotlib. This is used for plotting `Mapper`'s.
        """

        ticks = self.tick_values_from(
            min_value=min_value, max_value=max_value, use_defaults=use_defaults
        )
        labels = self.tick_values_in_units_from(
            array=array,
            min_value=min_value,
            max_value=max_value,
            units=units,
            use_defaults=use_defaults,
        )
        plt.yticks(ticks=ticks, labels=labels, **self.kwargs_ticks)


class XTicks(AbstractTicks):
    def set(
        self,
        array: arrays.Array,
        min_value: float,
        max_value: float,
        units: Units,
        use_defaults: bool = False,
    ):
        """
        Set the x ticks of a figure using the shape of an input `Array` object and input units.

        Parameters
        -----------
        array : arrays.Array
            The 2D array of data which is plotted.
        min_value : float
            the minimum value of the xticks that figure is plotted using.
        max_value : float
            the maximum value of the xticks that figure is plotted using.
        units : Units
            The units of the figure.
        use_defaults : bool
            If True, the figure is plotted symmetrically around a central value, which is the default behaviour of
            Matplotlib. This is used for plotting `Mapper`'s.
        """

        ticks = self.tick_values_from(
            min_value=min_value, max_value=max_value, use_defaults=use_defaults
        )
        labels = self.tick_values_in_units_from(
            array=array,
            min_value=min_value,
            max_value=max_value,
            units=units,
            use_defaults=use_defaults,
        )
        plt.xticks(ticks=ticks, labels=labels, **self.kwargs_ticks)


class Title(AbstractMatBase):
    def __init__(self, use_subplot_defaults: bool = False, **kwargs):
        """The settings used to customize the figure's title.

        This object wraps the following Matplotlib methods:

        - plt.title: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.title.html

        The title will automatically be set if not specified, using the name of the function used to plot the data.

        Parameters
        ----------
        use_subplot_defaults : bool
            `WrapMat` objects load settings from the [figure] section of its .ini config file by default. If
            `use_subplot_defaults=True` settings from the [subplot] section are loaded instead.
        """
        self.use_subplot_defaults = use_subplot_defaults

        super().__init__(use_subplot_defaults=use_subplot_defaults, kwargs=kwargs)

        if "label" not in self.kwargs:
            self.kwargs["label"] = None

    @property
    def kwargs_title(self):
        """Creates a kwargs dict of valid inputs of the methods `plt.title` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="title")

    def title_from_func(self, func: Callable) -> str:
        """If a title is not manually specified use the name of the function plotting the image to set the title.

        Parameters
        ----------
        func : func
           The function plotting the image.
        """
        if self.kwargs_title["label"] is None:
            return func.__name__.capitalize()
        else:
            return self.kwargs_title["label"]

    def set(self):
        plt.title(**self.kwargs_title)


class AbstractLabel(AbstractMatBase):
    def __init__(
        self,
        use_subplot_defaults: bool = False,
        units: "Units" = None,
        manual_label: str = None,
        **kwargs,
    ):
        """The settings used to customize the figure's title and y and x labels.

        This object wraps the following Matplotlib methods:

        - plt.ylabel: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.ylabel.html
        - plt.xlabel: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.xlabel.html

        The y and x labels will automatically be set if not specified, using the input `Unit`'s. object.

        Parameters
        ----------
        use_subplot_defaults : bool
            `WrapMat` objects load settings from the [figure] section of its .ini config file by default. If
            `use_subplot_defaults=True` settings from the [subplot] section are loaded instead.
        units : Units
            The units the data is plotted using.
        manual_label : str
            A manual label which overrides the default computed via the units if input.
        """

        super().__init__(use_subplot_defaults=use_subplot_defaults, kwargs=kwargs)

        self.manual_label = manual_label
        self._units = units

    @property
    def kwargs_label(self):
        """Creates a kwargs dict of valid inputs of the methods `plt.ylabel` and `plt.xlabel` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="label")

    def units_from_func(self, func: Callable, for_ylabel=True) -> "Units":
        """
        If the x label is not manually specified use the function plotting the image to x label, assuming that it
        represents spatial units.

        Parameters
        ----------
        func : func
           The function plotting the image.
        """

        if self._units is None:

            args = inspect.getfullargspec(func).args
            defaults = inspect.getfullargspec(func).defaults

            if defaults is not None:
                non_default_args = len(args) - len(defaults)
            else:
                non_default_args = 0

            if (not for_ylabel) and "label_xunits" in args:
                return defaults[args.index("label_xunits") - non_default_args]
            elif for_ylabel and "label_yunits" in args:
                return defaults[args.index("label_yunits") - non_default_args]
            else:
                return None

        else:

            return self._units

    def label_from_units(self, units: Units) -> str:
        """
        Returns the label of an object, by determining it from the figure units if the label is not manually specified.

        Parameters
        ----------
        units : Units
           The units of the image that is plotted which informs the appropriate y label text.
        """
        if self._units is None:

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

            return self._units


class YLabel(AbstractLabel):
    def set(self, units: Units, include_brackets: bool):
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

        if self.manual_label is not None:
            plt.ylabel(ylabel=self.manual_label, **self.kwargs_label)
        else:
            if include_brackets:
                plt.ylabel(
                    ylabel="y (" + self.label_from_units(units=units) + ")",
                    **self.kwargs_label,
                )
            else:
                plt.ylabel(
                    ylabel=self.label_from_units(units=units), **self.kwargs_label
                )


class XLabel(AbstractLabel):
    def set(self, units: Units, include_brackets: bool):
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
        if self.manual_label is not None:
            plt.xlabel(xlabel=self.manual_label, **self.kwargs_label)
        else:
            if include_brackets:
                plt.xlabel(
                    xlabel="x (" + self.label_from_units(units=units) + ")",
                    **self.kwargs_label,
                )
            else:
                plt.xlabel(
                    xlabel=self.label_from_units(units=units), **self.kwargs_label
                )


class Legend(AbstractMatBase):
    def __init__(self, use_subplot_defaults: bool = False, include=False, **kwargs):
        """
        The settings used to include and customize a legend on a figure.

        This object wraps the following Matplotlib methods:

        - plt.legend: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.legend.html

        Parameters
        ----------
        use_subplot_defaults : bool
            `WrapMat` objects load settings from the [figure] section of its .ini config file by default. If
            `use_subplot_defaults=True` settings from the [subplot] section are loaded instead.
        include : bool
            If the legend should be plotted and therefore included on the figure.
        """

        super().__init__(use_subplot_defaults=use_subplot_defaults, kwargs=kwargs)

        self.include = include

    @property
    def kwargs_legend(self):
        """Creates a kwargs dict of valid inputs of the method `plt.legend` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="legend")

    def set(self):
        if self.include:
            plt.legend(**self.kwargs_legend)


class Output:
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

        The default behaviour is the display the figure on the computer screen, as opposed to outputting to hard-disk
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
        """Output the figure, by either displaying it on the user's screen or to the hard-disk as a .png or .fits file.

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
