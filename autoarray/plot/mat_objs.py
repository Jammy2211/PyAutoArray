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

import inspect

# def ignore_unmatched_kwargs(func):
#
#
#
#     return inner


class AbstractMatObj:
    def __init__(self, from_subplot_config, kwargs):

        self.from_subplot_config = from_subplot_config

        if not from_subplot_config:

            config_dict = conf.instance["visualize"]["mat_objs"][
                self.__class__.__name__
            ]["figure"]._dict

        else:

            config_dict = conf.instance["visualize"]["mat_objs"][
                self.__class__.__name__
            ]["subplot"]._dict

        self.kwargs = {**config_dict, **kwargs}

    def load_setting(self, param, name, from_subplot_config):

        if param is not None:
            return param

        if not from_subplot_config:
            return conf.instance["visualize"]["mat_objs"][self.__class__.__name__][
                "figure"
            ][name]
        else:
            return conf.instance["visualize"]["mat_objs"][self.__class__.__name__][
                "subplot"
            ][name]

    def kwargs_of_method(self, method_name, cls_name=None):

        if cls_name is None:
            cls_name = self.__class__.__name__

        args = conf.instance["visualize"]["mat_objs"][cls_name]["args"][method_name]

        args = args.replace(" ", "")
        args = args.split(",")

        return {key: self.kwargs[key] for key in args if key in self.kwargs}


class Units(AbstractMatObj):
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


class Figure(AbstractMatObj):
    def __init__(self, from_subplot_config: bool = False, **kwargs):
        """
        Sets up the Matplotlib figure before plotting (this is used when plotting individual figures and subplots).

        This object wraps the following Matplotlib methods:

        - plt.figure: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.figure.html
        - plt.close: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.close.html

        It also controls the aspect ratio of the figure plotted.

        Parameters
        ----------
        from_subplot_config : bool
            If True, load unspecified settings from the figures.ini visualization config, else use subplots.ini.
        """

        super().__init__(from_subplot_config=from_subplot_config, kwargs=kwargs)

    @property
    def kwargs_figure(self):
        """Creates a kwargs dict of valid inputs of the method `plt.figure` from the object's kwargs dict."""

        kwargs_figure = self.kwargs_of_method(method_name="figure")

        if kwargs_figure["figsize"] == "auto":
            kwargs_figure["figsize"] = None
        elif isinstance(kwargs_figure["figsize"], str):
            kwargs_figure["figsize"] = tuple(
                map(int, kwargs_figure["figsize"][1:-1].split(","))
            )

        return kwargs_figure

    @property
    def kwargs_imshow(self):
        """Creates a kwargs dict of valid inputs of the method `plt.imshow` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="imshow")

    @classmethod
    def sub(cls,):
        return Figure(from_subplot_config=True)

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


class Cmap(AbstractMatObj):
    def __init__(self, module: str = None, from_subplot_config: bool = False, **kwargs):
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
        from_subplot_config : bool
            If True, load unspecified settings from the figures.ini visualization config, else use subplots.ini.
        """

        super().__init__(from_subplot_config=from_subplot_config, kwargs=kwargs)

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

    @classmethod
    def sub(cls,):
        return Cmap(from_subplot_config=True)

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


class Colorbar(AbstractMatObj):
    def __init__(
        self,
        manual_tick_labels: typing.Union[typing.List[float]] = None,
        manual_tick_values: typing.Union[typing.List[float]] = None,
        from_subplot_config: bool = False,
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
        manual_tick_labels : [float]
            Manually override the colorbar tick labels to an input list of float.
        manual_tick_values : [float]
            If the colorbar tick labels are manually specified the locations on the colorbar they appear running 0 -> 1.
        from_subplot_config : bool
            If True, load unspecified settings from the figures.ini visualization config, else use subplots.ini.
        """

        super().__init__(from_subplot_config=from_subplot_config, kwargs=kwargs)

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

    @classmethod
    def sub(
        cls,
        manual_tick_labels: typing.Union[typing.List[float]] = None,
        manual_tick_values: typing.Union[typing.List[float]] = None,
        **kwargs,
    ):
        return Colorbar(
            manual_tick_values=manual_tick_values,
            manual_tick_labels=manual_tick_labels,
            from_subplot_config=True,
            **kwargs,
        )

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


class TickParams(AbstractMatObj):
    def __init__(self, from_subplot_config: bool = False, **kwargs):
        """
        The settings used to customize a figure's y and x ticks parameters.

        This object wraps the following Matplotlib methods:

        - plt.tick_params: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.tick_params.html

        Parameters
        ----------
        from_subplot_config : bool
            If True, load unspecified settings from the figures.ini visualization config, else use subplots.ini.
        """
        super().__init__(from_subplot_config=from_subplot_config, kwargs=kwargs)

    @property
    def kwargs_tick_params(self):
        """Creates a kwargs dict of valid inputs of the method `plt.tick_params` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="tick_params")

    def set(self,):
        """Set the tick_params of the figure using the method `plt.tick_params`."""
        plt.tick_params(**self.kwargs_tick_params)


class AbstractTicks(AbstractMatObj):
    def __init__(
        self,
        manual_values: typing.Union[typing.List[float]] = None,
        from_subplot_config: bool = False,
        **kwargs,
    ):
        """
        The settings used to customize a figure's y and x ticks using the `YTicks` and `XTicks` objects.

        This object wraps the following Matplotlib methods:

        - plt.yticks: https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.yticks.html
        - plt.xticks: https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.xticks.html

        Parameters
        ----------
        manual_values : [float]
            Manually override the tick labels to display the labels as the input list of floats.
        from_subplot_config : bool
            If True, load unspecified settings from the figures.ini visualization config, else use subplots.ini.
        """
        super().__init__(from_subplot_config=from_subplot_config, kwargs=kwargs)

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
    @classmethod
    def sub(cls, manual_values: typing.Union[typing.List[float]] = None):
        return YTicks(manual_values=manual_values, from_subplot_config=True)

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
    @classmethod
    def sub(cls, manual_values: typing.Union[typing.List[float]] = None):
        return XTicks(manual_values=manual_values, from_subplot_config=True)

    def set(
        self,
        array: arrays.Array,
        min_value: float,
        max_value: float,
        units: Units,
        symmetric_around_centre: bool = False,
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
            min_value=min_value,
            max_value=max_value,
            use_defaults=symmetric_around_centre,
        )
        labels = self.tick_values_in_units_from(
            array=array,
            min_value=min_value,
            max_value=max_value,
            units=units,
            use_defaults=symmetric_around_centre,
        )
        plt.xticks(ticks=ticks, labels=labels, **self.kwargs_ticks)


class Title(AbstractMatObj):
    def __init__(self, from_subplot_config: bool = False, **kwargs):
        """The settings used to customize the figure's title.

        This object wraps the following Matplotlib methods:

        - plt.title: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.title.html

        The title will automatically be set if not specified, using the name of the function used to plot the data.

        Parameters
        ----------
        from_subplot_config : bool
            If True, load unspecified settings from the figures.ini visualization config, else use subplots.ini.
        """
        self.from_subplot_config = from_subplot_config

        super().__init__(from_subplot_config=from_subplot_config, kwargs=kwargs)

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


class AbstractLabel(AbstractMatObj):
    def __init__(
        self,
        units: "Units" = None,
        manual_label: str = None,
        from_subplot_config: bool = False,
        **kwargs,
    ):
        """The settings used to customize the figure's title and y and x labels.

        This object wraps the following Matplotlib methods:

        - plt.ylabel: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.ylabel.html
        - plt.xlabel: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.xlabel.html

        The y and x labels will automatically be set if not specified, using the input `Unit`'s. object.

        Parameters
        ----------
        units : Units
            The units the data is plotted using.
        manual_label : str
            A manual label which overrides the default computed via the units if input.

        """

        super().__init__(from_subplot_config=from_subplot_config, kwargs=kwargs)

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
    @classmethod
    def sub(cls, units: str = None, manual_label: str = None):
        return YLabel(units=units, manual_label=manual_label, from_subplot_config=True)

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
    @classmethod
    def sub(cls, units: str = None, manual_label: str = None):
        return YLabel(units=units, manual_label=manual_label, from_subplot_config=True)

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


class Legend(AbstractMatObj):
    def __init__(self, include=False, from_subplot_config: bool = False, **kwargs):
        """
        The settings used to include and customize a legend on a figure.

        This object wraps the following Matplotlib methods:

        - plt.legend: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.legend.html

        Parameters
        ----------
        include : bool
            If the legend should be plotted and therefore included on the figure.
        """

        super().__init__(from_subplot_config=from_subplot_config, kwargs=kwargs)

        self.kwargs["include"] = include

    @property
    def kwargs_legend(self):
        """Creates a kwargs dict of valid inputs of the method `plt.legend` from the object's kwargs dict."""
        return self.kwargs_of_method(method_name="label")

    @classmethod
    def sub(cls, include: bool = False):
        return Legend(include=include, from_subplot_config=True)

    def set(self):
        if self.kwargs["include"]:
            plt.legend(**self.kwargs_legend)


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


class Scatter(AbstractMatObj):
    def __init__(self, colors=None, from_subplot_config=False, **kwargs):
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
        - `GridScatter`: plots an input grid of points which are passed in as a `Grid` structure.
        - `PositionsScatter`: plots the (y,x) coordinates that are input in a plotter via the `positions` input.
        - `IndexScatter`: plots specific (y,x) coordinates of a grid (or grids) via their 1d or 2d indexes.
        - `PixelizationGridScatter`: plots the grid of a `Pixelization` object (see `autoarray.inversion`).

        Parameters
        ----------
        colors : [str]
            The color or list of colors that the grid is plotted using. For plotting indexes or a grouped grid, a
            list of colors can be specified which the plot cycles through.
        """
        super().__init__(from_subplot_config=from_subplot_config, kwargs=kwargs)

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


class Quiver(AbstractMatObj):
    def __init__(
        self,
        from_subplot_config=False,
        **kwargs
    ):

        super().__init__(from_subplot_config=from_subplot_config, kwargs=kwargs)

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
            **self.kwargs_quiver
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
    ):
        return Quiver(
            headlength=headlength,
            pivot=pivot,
            linewidth=linewidth,
            units=units,
            angles=angles,
            headwidth=headwidth,
            alpha=alpha,
            cmap=cmap,
            from_subplot_config=True,
        )


class Patcher(AbstractMatObj):
    def __init__(self, facecolor=None, edgecolor=None, from_subplot_config=False):

        self.from_subplot_config = from_subplot_config

        self.facecolor = self.load_setting(
            param=facecolor, name="facecolor", from_subplot_config=from_subplot_config
        )

        if self.facecolor is None:
            self.facecolor = "none"

        self.edgecolor = self.load_setting(
            param=edgecolor, name="edgecolor", from_subplot_config=from_subplot_config
        )

    @classmethod
    def sub(cls, facecolor=None, edgecolor=None):
        return Patcher(
            facecolor=facecolor, edgecolor=edgecolor, from_subplot_config=True
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
        from_subplot_config=False,
    ):

        self.from_subplot_config = from_subplot_config

        self.width = self.load_setting(
            param=width, name="width", from_subplot_config=from_subplot_config
        )
        self.style = self.load_setting(
            param=style, name="style", from_subplot_config=from_subplot_config
        )
        self.colors = self.load_setting(
            param=colors, name="colors", from_subplot_config=from_subplot_config
        )

        self.colors = remove_spaces_and_commas_from_colors(colors=self.colors)

        self.pointsize = self.load_setting(
            param=pointsize, name="pointsize", from_subplot_config=from_subplot_config
        )

    @classmethod
    def sub(cls, width=None, style=None, colors=None, pointsize=None):
        return Liner(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
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
    def __init__(self, alpha=None, from_subplot_config=False):

        self.from_subplot_config = from_subplot_config

        self.alpha = self.load_setting(
            param=alpha, name="alpha", from_subplot_config=from_subplot_config
        )

    @classmethod
    def sub(cls, alpha):
        return ArrayOverlayer(alpha=alpha, from_subplot_config=True)

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

        self.edgewidth = self.load_setting(
            param=edgewidth, name="edgewidth", from_subplot_config=from_subplot_config
        )
        self.edgecolor = self.load_setting(
            param=edgecolor, name="edgecolor", from_subplot_config=from_subplot_config
        )
        self.alpha = self.load_setting(
            param=alpha, name="alpha", from_subplot_config=from_subplot_config
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


def remove_spaces_and_commas_from_colors(colors):

    colors = [color.strip(",") for color in colors]
    colors = [color.strip(" ") for color in colors]
    return list(filter(None, colors))
