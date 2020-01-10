from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps
import copy
import inspect

from autoarray import exc


def setting(section, name, python_type):
    return conf.instance.visualize.get(section, name, python_type)


def load_setting(section, value, name, python_type):
    return (
        value
        if value is not None
        else setting(section=section, name=name, python_type=python_type)
    )


class Ticks(object):
    def __init__(
        self, ysize=None, xsize=None, y_manual=None, x_manual=None, is_sub_plotter=False
    ):

        if not is_sub_plotter:

            self.ysize = load_setting(
                section="figures_ticks", value=ysize, name="ysize", python_type=int
            )

            self.xsize = load_setting(
                section="figures_ticks", value=xsize, name="xsize", python_type=int
            )

        else:

            self.ysize = load_setting(
                "subplots_ticks", value=ysize, name="ysize", python_type=int
            )

            self.xsize = load_setting(
                "subplots_ticks", value=xsize, name="xsize", python_type=int
            )

        self.y_manual = y_manual
        self.x_manual = x_manual

    def set_yticks(
        self,
        array,
        extent,
        use_scaled_units,
        unit_conversion_factor,
        symmetric_around_centre=False,
    ):
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
        elif not use_scaled_units:
            ytick_labels = np.linspace(0, array.shape_2d[0], 5).astype("int")
        elif use_scaled_units and unit_conversion_factor is None:
            ytick_labels = np.round(np.linspace(extent[2], extent[3], 5), 2)
        elif use_scaled_units and unit_conversion_factor is not None:
            ytick_labels = np.round(
                np.linspace(
                    extent[2] * unit_conversion_factor,
                    extent[3] * unit_conversion_factor,
                    5,
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
        array,
        extent,
        use_scaled_units,
        unit_conversion_factor,
        symmetric_around_centre=False,
    ):
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
        elif not use_scaled_units:
            xtick_labels = np.linspace(0, array.shape_2d[0], 5).astype("int")
        elif use_scaled_units and unit_conversion_factor is None:
            xtick_labels = np.round(np.linspace(extent[0], extent[1], 5), 2)
        elif use_scaled_units and unit_conversion_factor is not None:
            xtick_labels = np.round(
                np.linspace(
                    extent[0] * unit_conversion_factor,
                    extent[1] * unit_conversion_factor,
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
        use_scaled_units=None,
        plot_in_kpc=None,
        is_sub_plotter=False,
    ):

        self.title = title
        self._yunits = yunits
        self._xunits = xunits

        if not is_sub_plotter:

            self.titlesize = load_setting(
                section="figures_labels",
                value=titlesize,
                name="titlesize",
                python_type=int,
            )
            self.ysize = load_setting(
                section="figures_labels", value=ysize, name="ysize", python_type=int
            )
            self.xsize = load_setting(
                section="figures_labels", value=xsize, name="xsize", python_type=int
            )

        else:

            self.titlesize = load_setting(
                section="subplots_labels",
                value=titlesize,
                name="titlesize",
                python_type=int,
            )
            self.ysize = load_setting(
                section="subplots_labels", value=ysize, name="ysize", python_type=int
            )
            self.xsize = load_setting(
                section="subplots_labels", value=xsize, name="xsize", python_type=int
            )

        self.plot_in_kpc = plot_in_kpc
        self.use_scaled_units = use_scaled_units

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

            if self.plot_in_kpc is not None:
                if self.plot_in_kpc:
                    return "kpc"
                else:
                    return "arcsec"

            if self.use_scaled_units:
                return "scaled"
            else:
                return "pixels"

        else:

            return self._yunits

    @property
    def xunits(self):

        if self._xunits is None:

            if self.plot_in_kpc is not None:
                if self.plot_in_kpc:
                    return "kpc"
                else:
                    return "arcsec"

            if self.use_scaled_units:
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


class Output(object):
    def __init__(self, path=None, filename=None, format="show"):

        self.path = path
        self.filename = filename
        self.format = format

    def filename_from_func(self, func):

        if self.filename is None:
            return func.__name__
        else:

            return self.filename

    def to_figure(self, structure, is_sub_plotter):
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
        if not is_sub_plotter:
            if self.format is "show":
                plt.show()
            elif self.format is "png":
                plt.savefig(self.path + self.filename + ".png", bbox_inches="tight")
            elif self.format is "fits":
                if structure is not None:
                    structure.output_to_fits(
                        file_path=self.path + self.filename + ".fits"
                    )


class Plotter(object):
    def __init__(
        self,
        is_sub_plotter=False,
        use_scaled_units=None,
        unit_conversion_factor=None,
        plot_in_kpc=None,
        figsize=None,
        aspect=None,
        cmap=None,
        norm=None,
        norm_min=None,
        norm_max=None,
        linthresh=None,
        linscale=None,
        cb_ticksize=None,
        cb_fraction=None,
        cb_pad=None,
        cb_tick_values=None,
        cb_tick_labels=None,
        mask_pointsize=None,
        border_pointsize=None,
        point_pointsize=None,
        grid_pointsize=None,
        ticks=Ticks(),
        labels=Labels(),
        output=Output(),
    ):

        self.is_sub_plotter = is_sub_plotter

        if not is_sub_plotter:

            self.figsize = load_setting(
                section="figures", value=figsize, name="figsize", python_type=str
            )
            if isinstance(self.figsize, str):
                self.figsize = tuple(map(int, self.figsize[1:-1].split(",")))
            self.aspect = load_setting(
                section="figures", value=aspect, name="aspect", python_type=str
            )

        else:

            self.figsize = load_setting(
                section="subplots", value=figsize, name="figsize", python_type=str
            )
            self.figsize = None if self.figsize == "auto" else self.figsize
            if isinstance(self.figsize, str):
                self.figsize = tuple(map(int, self.figsize[1:-1].split(",")))
            self.aspect = load_setting(
                section="subplots", value=aspect, name="aspect", python_type=str
            )

        self.use_scaled_units = load_setting(
            section="settings",
            value=use_scaled_units,
            name="use_scaled_units",
            python_type=bool,
        )
        self.unit_conversion_factor = unit_conversion_factor
        try:
            self.plot_in_kpc = load_setting(
                section="general",
                value=plot_in_kpc,
                name="plot_in_kpc",
                python_type=bool,
            )
        except:
            self.plot_in_kpc = None

        self.cmap = load_setting(
            section="settings", value=cmap, name="cmap", python_type=str
        )
        self.norm = load_setting(
            section="settings", value=norm, name="norm", python_type=str
        )
        self.norm_min = load_setting(
            section="settings", value=norm_min, name="norm_min", python_type=float
        )
        self.norm_max = load_setting(
            section="settings", value=norm_max, name="norm_max", python_type=float
        )
        self.linthresh = load_setting(
            section="settings", value=linthresh, name="linthresh", python_type=float
        )
        self.linscale = load_setting(
            section="settings", value=linscale, name="linscale", python_type=float
        )

        self.cb_ticksize = load_setting(
            section="settings", value=cb_ticksize, name="cb_ticksize", python_type=int
        )
        self.cb_fraction = load_setting(
            section="settings", value=cb_fraction, name="cb_fraction", python_type=float
        )
        self.cb_pad = load_setting(
            section="settings", value=cb_pad, name="cb_pad", python_type=float
        )
        self.cb_tick_values = cb_tick_values
        self.cb_tick_labels = cb_tick_labels

        self.mask_pointsize = load_setting(
            section="settings",
            value=mask_pointsize,
            name="mask_pointsize",
            python_type=int,
        )
        self.border_pointsize = load_setting(
            section="settings",
            value=border_pointsize,
            name="border_pointsize",
            python_type=int,
        )
        self.point_pointsize = load_setting(
            section="settings",
            value=point_pointsize,
            name="point_pointsize",
            python_type=int,
        )
        self.grid_pointsize = load_setting(
            section="settings",
            value=grid_pointsize,
            name="grid_pointsize",
            python_type=int,
        )

        self.ticks = Ticks(
            ysize=ticks.ysize,
            xsize=ticks.xsize,
            y_manual=ticks.y_manual,
            x_manual=ticks.x_manual,
        )

        self.labels = Labels(
            title=labels.title,
            yunits=labels._yunits,
            xunits=labels._xunits,
            titlesize=labels.titlesize,
            ysize=labels.ysize,
            xsize=labels.xsize,
            use_scaled_units=use_scaled_units,
        )

        self.output = Output(
            path=output.path, format=output.format, filename=output.filename
        )

    def setup_figure(self):
        """Setup a figure for plotting an image.

        Parameters
        -----------
        figsize : (int, int)
            The size of the figure in (rows, columns).
        as_subplot : bool
            If the figure is a subplot, the setup_figure function is omitted to ensure that each subplot does not create a \
            new figure and so that it can be output using the *output.output_figure(structure=None, is_sub_plotter=False)* function.
        """
        if not self.is_sub_plotter:
            fig = plt.figure(figsize=self.figsize)
            return fig

    def set_colorbar(self):
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

        if self.cb_tick_values is None and self.cb_tick_labels is None:
            cb = plt.colorbar(fraction=self.cb_fraction, pad=self.cb_pad)
        elif self.cb_tick_values is not None and self.cb_tick_labels is not None:
            cb = plt.colorbar(
                fraction=self.cb_fraction, pad=self.cb_pad, ticks=self.cb_tick_values
            )
            cb.ax.set_yticklabels(labels=self.cb_tick_labels)
        else:
            raise exc.PlottingException(
                "Only 1 entry of cb_tick_values or cb_tick_labels was input. You must either supply"
                "both the values and labels, or neither."
            )

        cb.ax.tick_params(labelsize=self.cb_ticksize)

    @staticmethod
    def plot_lines(line_lists):
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
        if line_lists is not None:
            for line_list in line_lists:
                if line_list is not None:
                    for line in line_list:
                        if len(line) != 0:
                            plt.plot(line[:, 1], line[:, 0], c="w", lw=2.0, zorder=200)

    def close_figure(self):
        """After plotting and outputting a figure, close the matplotlib figure instance (omit if a subplot).

        Parameters
        -----------
        as_subplot : bool
            Whether the figure is part of subplot, in which case the figure is not closed so that the entire figure can \
            be closed later after output.
        """
        if not self.is_sub_plotter:
            plt.close()

    @staticmethod
    def get_subplot_rows_columns_figsize(number_subplots):
        """Get the size of a sub plotters in (rows, columns), based on the number of subplots that are going to be plotted.

        Parameters
        -----------
        number_subplots : int
            The number of subplots that are to be plotted in the figure.
        """
        if number_subplots <= 2:
            return 1, 2, (18, 8)
        elif number_subplots <= 4:
            return 2, 2, (13, 10)
        elif number_subplots <= 6:
            return 2, 3, (18, 12)
        elif number_subplots <= 9:
            return 3, 3, (25, 20)
        elif number_subplots <= 12:
            return 3, 4, (25, 20)
        elif number_subplots <= 16:
            return 4, 4, (25, 20)
        elif number_subplots <= 20:
            return 4, 5, (25, 20)
        else:
            return 6, 6, (25, 20)

    def plotter_as_sub_plotter(self):

        if self.is_sub_plotter:
            return self

        plotter = copy.deepcopy(self)
        plotter.is_sub_plotter = True

        plotter.figsize = load_setting(
            section="subplots", value=None, name="figsize", python_type=str
        )
        plotter.figsize = None if plotter.figsize == "auto" else plotter.figsize
        if isinstance(plotter.figsize, str):
            plotter.figsize = tuple(map(int, plotter.figsize[1:-1].split(",")))
        plotter.aspect = load_setting(
            section="subplots", value=None, name="aspect", python_type=str
        )

        plotter.labels.titlesize = load_setting(
            section="subplots_labels", value=None, name="titlesize", python_type=int
        )
        plotter.labels.ysize = load_setting(
            section="subplots_labels", value=None, name="ysize", python_type=int
        )
        plotter.labels.xsize = load_setting(
            section="subplots_labels", value=None, name="xsize", python_type=int
        )

        plotter.ticks.ysize = load_setting(
            "subplots_ticks", value=None, name="ysize", python_type=int
        )

        plotter.ticks.xsize = load_setting(
            "subplots_ticks", value=None, name="xsize", python_type=int
        )

        return plotter

    def plotter_with_new_labels(self, labels=Labels()):

        plotter = copy.deepcopy(self)

        plotter.labels.title = (
            labels.title if labels.title is not None else self.labels.title
        )
        plotter.labels._yunits = (
            labels._yunits if labels._yunits is not None else self.labels._yunits
        )
        plotter.labels._xunits = (
            labels._xunits if labels._xunits is not None else self.labels._xunits
        )

        return plotter

    def plotter_with_new_output_filename(self, output_filename=None):

        plotter = copy.deepcopy(self)

        plotter.output.filename = (
            output_filename if output_filename is not None else self.output.filename
        )

        return plotter


class Include(object):
    def __init__(
        self,
        origin=None,
        mask=None,
        grid=None,
        border=None,
        inversion_centres=None,
        inversion_grid=None,
        inversion_border=None,
    ):

        self.origin = self.load_include(value=origin, name="origin")
        self.mask = self.load_include(value=mask, name="mask")
        self.grid = self.load_include(value=grid, name="grid")
        self.border = self.load_include(value=border, name="border")
        self.inversion_centres = self.load_include(
            value=inversion_centres, name="inversion_centres"
        )
        self.inversion_grid = self.load_include(
            value=inversion_grid, name="inversion_grid"
        )
        self.inversion_border = self.load_include(
            value=inversion_border, name="inversion_border"
        )

    @staticmethod
    def load_include(value, name):

        return (
            setting(section="include", name=name, python_type=bool)
            if value is None
            else value
        )

    def mask_from_fit(self, fit):
        """Get the masks of the fit if the masks should be plotted on the fit.

        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractLensHyperFit
            The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
        include_mask : bool
            If *True*, the masks is plotted on the fit's datas.
        """
        if self.mask:
            return fit.mask
        else:
            return None


def plotter_key_from_dictionary(dictionary):

    plotter_key = None

    for key, value in dictionary.items():
        if isinstance(value, Plotter):
            plotter_key = key

    if plotter_key is None:
        raise exc.PlottingException(
            "The plot function called could not locate a Plotter in the kwarg arguments"
            "in order to set the labels."
        )

    return plotter_key


def set_labels(func):
    """
    Decorate a profile method that accepts a coordinate grid and returns a data_type grid.

    If an interpolator attribute is associated with the input grid then that interpolator is used to down sample the
    coordinate grid prior to calling the function and up sample the result of the function.

    If no interpolator attribute is associated with the input grid then the function is called as hyper.

    Parameters
    ----------
    func
        Some method that accepts a grid

    Returns
    -------
    decorated_function
        The function with optional interpolation
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        plotter_key = plotter_key_from_dictionary(dictionary=kwargs)
        plotter = kwargs[plotter_key]

        title = plotter.labels.title_from_func(func=func)
        yunits = plotter.labels.yunits_from_func(func=func)
        xunits = plotter.labels.xunits_from_func(func=func)

        plotter = plotter.plotter_with_new_labels(
            labels=Labels(title=title, yunits=yunits, xunits=xunits)
        )

        filename = plotter.output.filename_from_func(func=func)

        plotter = plotter.plotter_with_new_output_filename(output_filename=filename)

        kwargs[plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper
