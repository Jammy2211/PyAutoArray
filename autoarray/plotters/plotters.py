from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps

from autoarray import exc
from autoarray.util import array_util
from autoarray.plotters import plotters_util

class Plotter(object):

    def __init__(
        self,
        is_sub_plotter=False,
        use_scaled_units=None,
        unit_conversion_factor=None,
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
        titlesize=None,
        xlabelsize=None,
        ylabelsize=None,
        xyticksize=None,
        label_title=None,
        label_yunits=None,
        label_xunits=None,
        label_yticks=None,
        label_xticks=None,
        output_path=None,
        output_format="show",
        output_filename=None,
    ):

        self.is_sub_plotter = is_sub_plotter

        if not is_sub_plotter:

            self.figsize = plotters_util.load_figure_setting(value=figsize, name="figsize", python_type=str)
            if isinstance(self.figsize, str):
                self.figsize = tuple(map(int, self.figsize[1:-1].split(",")))
            self.aspect = plotters_util.load_figure_setting(value=aspect, name="aspect", python_type=str)
            self.titlesize = plotters_util.load_figure_setting(
                value=titlesize, name="titlesize", python_type=int
            )
            self.ylabelsize = plotters_util.load_figure_setting(
                value=ylabelsize, name="ylabelsize", python_type=int
            )
            self.xlabelsize = plotters_util.load_figure_setting(
                value=xlabelsize, name="xlabelsize", python_type=int
            )
            self.xyticksize = plotters_util.load_figure_setting(
                value=xyticksize, name="xyticksize", python_type=int
            )

        else:

            self.figsize = plotters_util.load_subplot_setting(value=figsize, name="figsize", python_type=str)
            self.figsize = None if self.figsize == "auto" else self.figsize
            if isinstance(self.figsize, str):
                self.figsize = tuple(map(int, self.figsize[1:-1].split(",")))
            self.aspect = plotters_util.load_subplot_setting(value=aspect, name="aspect", python_type=str)
            self.titlesize = plotters_util.load_subplot_setting(
                value=titlesize, name="titlesize", python_type=int
            )
            self.ylabelsize = plotters_util.load_subplot_setting(
                value=ylabelsize, name="ylabelsize", python_type=int
            )
            self.xlabelsize = plotters_util.load_subplot_setting(
                value=xlabelsize, name="xlabelsize", python_type=int
            )
            self.xyticksize = plotters_util.load_subplot_setting(
                value=xyticksize, name="xyticksize", python_type=int
            )

        self.use_scaled_units = plotters_util.load_setting(
            value=use_scaled_units, name="use_scaled_units", python_type=bool
        )
        self.unit_conversion_factor = unit_conversion_factor

        self.cmap = plotters_util.load_setting(value=cmap, name="cmap", python_type=str)
        self.norm = plotters_util.load_setting(value=norm, name="norm", python_type=str)
        self.norm_min = plotters_util.load_setting(value=norm_min, name="norm_min", python_type=float)
        self.norm_max = plotters_util.load_setting(value=norm_max, name="norm_max", python_type=float)
        self.linthresh = plotters_util.load_setting(
            value=linthresh, name="linthresh", python_type=float
        )
        self.linscale = plotters_util.load_setting(value=linscale, name="linscale", python_type=float)

        self.cb_ticksize = plotters_util.load_setting(
            value=cb_ticksize, name="cb_ticksize", python_type=int
        )
        self.cb_fraction = plotters_util.load_setting(
            value=cb_fraction, name="cb_fraction", python_type=float
        )
        self.cb_pad = plotters_util.load_setting(value=cb_pad, name="cb_pad", python_type=float)
        self.cb_tick_values = cb_tick_values
        self.cb_tick_labels = cb_tick_labels

        self.mask_pointsize = plotters_util.load_setting(
            value=mask_pointsize, name="mask_pointsize", python_type=int
        )
        self.border_pointsize = plotters_util.load_setting(
            value=border_pointsize, name="border_pointsize", python_type=int
        )
        self.point_pointsize = plotters_util.load_setting(
            value=point_pointsize, name="point_pointsize", python_type=int
        )
        self.grid_pointsize = plotters_util.load_setting(
            value=grid_pointsize, name="grid_pointsize", python_type=int
        )

        self.label_title = label_title
        self.label_yunits = label_yunits
        self.label_xunits = label_xunits
        self.label_yticks = label_yticks
        self.label_xticks = label_xticks

        self.output_path = output_path
        self.output_format = output_format
        self.output_filename = output_filename

    def setup_figure(self):
        """Setup a figure for plotting an image.

        Parameters
        -----------
        figsize : (int, int)
            The size of the figure in (rows, columns).
        as_subplot : bool
            If the figure is a subplot, the setup_figure function is omitted to ensure that each subplot does not create a \
            new figure and so that it can be output using the *output_subplot_array* function.
        """
        if not self.is_sub_plotter:
            fig = plt.figure(figsize=self.figsize)
            return fig

    def set_title(self):
        """Set the title and title size of the figure.

        Parameters
        -----------
        title : str
            The text of the title.
        titlesize : int
            The size of of the title of the figure.
        """
        plt.title(label=self.label_title, fontsize=self.titlesize)

    def set_yx_labels_and_ticksize(self):
        """Set the x and y labels of the figure, and set the fontsize of those self.label_

        The x and y labels are always the distance scales, thus the labels are either arc-seconds or kpc and depend on the \
        unit_label the figure is plotted in.

        Parameters
        -----------
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        xlabelsize : int
            The fontsize of the x axes label.
        ylabelsize : int
            The fontsize of the y axes label.
        xyticksize : int
            The font size of the x and y ticks on the figure axes.
        """

        plt.ylabel("y (" + self.label_yunits + ")", fontsize=self.ylabelsize)
        plt.xlabel("x (" + self.label_xunits + ")", fontsize=self.xlabelsize)

        plt.tick_params(labelsize=self.xyticksize)

    def set_yxticks(self, array, extent, symmetric_around_centre=False):
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

        if symmetric_around_centre:
            return

        yticks = np.linspace(extent[2], extent[3], 5)
        xticks = np.linspace(extent[0], extent[1], 5)

        if self.label_xticks is not None and self.label_yticks is not None:
            ytick_labels = np.asarray([self.label_yticks[0], self.label_yticks[3]])
            xtick_labels = np.asarray([self.label_xticks[0], self.label_xticks[3]])
        elif not self.use_scaled_units:
            ytick_labels = np.linspace(0, array.shape_2d[0], 5).astype("int")
            xtick_labels = np.linspace(0, array.shape_2d[1], 5).astype("int")
        elif self.use_scaled_units and self.unit_conversion_factor is None:
            ytick_labels = np.round(np.linspace(extent[2], extent[3], 5), 2)
            xtick_labels = np.round(np.linspace(extent[0], extent[1], 5), 2)
        elif self.use_scaled_units and self.unit_conversion_factor is not None:
            ytick_labels = np.round(
                np.linspace(
                    extent[2] * self.unit_conversion_factor,
                    extent[3] * self.unit_conversion_factor,
                    5,
                ),
                2,
            )
            xtick_labels = np.round(
                np.linspace(
                    extent[0] * self.unit_conversion_factor,
                    extent[1] * self.unit_conversion_factor,
                    5,
                ),
                2,
            )
        else:
            raise exc.PlottingException(
                "The y and y ticks cannot be set using the input options."
            )

        plt.yticks(ticks=yticks, labels=ytick_labels)
        plt.xticks(ticks=xticks, labels=xtick_labels)

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

    def output_figure(self, array):
        """Output the figure, either as an image on the screen or to the hard-disk as a .png or .fits file.

        Parameters
        -----------
        array : ndarray
            The 2D array of image to be output, required for outputting the image as a fits file.
        as_subplot : bool
            Whether the figure is part of subplot, in which case the figure is not output so that the entire subplot can \
            be output instead using the *output_subplot_array* function.
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
        if not self.is_sub_plotter:
            if self.output_format is "show":
                plt.show()
            elif self.output_format is "png":
                plt.savefig(
                    self.output_path + self.output_filename + ".png",
                    bbox_inches="tight",
                )
            elif self.output_format is "fits":
                array_util.numpy_array_2d_to_fits(
                    array_2d=array,
                    file_path=self.output_path + self.output_filename + ".fits",
                    overwrite=True,
                )

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

    def output_subplot_array(self):
        """Output a figure which consists of a set of subplot,, either as an image on the screen or to the hard-disk as a \
        .png file.

        Parameters
        -----------
        output_path : str
            The path on the hard-disk where the figure is output.
        output_filename : str
            The filename of the figure that is output.
        output_format : str
            The format the figue is output:
            'show' - display on computer screen.
            'png' - output to hard-disk as a png.
        """
        if self.output_format is "show":
            plt.show()
        elif self.output_format is "png":
            plt.savefig(
                self.output_path + self.output_filename + ".png", bbox_inches="tight"
            )
        elif self.output_format is "fits":
            raise exc.PlottingException(
                "You cannot output a subplots with format .fits"
            )


def set_includes(func):
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

        includes = ["include_origin", "include_mask", "include_grid", "include_centres", "include_border"]

        for include in includes:
            if include in kwargs:
                if kwargs[include] is None:

                    kwargs[include] = plotters_util.setting(
                        section="include", name=include[8:], python_type=bool)

        return func(*args, **kwargs)

    return wrapper

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

        plotter_key = plotters_util.plotter_key_from_dictionary(dictionary=kwargs)
        plotter = kwargs[plotter_key]

        label_title = plotters_util.label_title_from_plotter(plotter=plotter, func=func)
        label_yunits = plotters_util.label_yunits_from_plotter(plotter=plotter)
        label_xunits = plotters_util.label_xunits_from_plotter(plotter=plotter)
        output_filename = plotters_util.output_filename_from_plotter_and_func(plotter=plotter, func=func)

        kwargs[plotter_key] = plotter.plotter_with_new_labels_and_filename(
            label_title=label_title,
            label_yunits=label_yunits,
            label_xunits=label_xunits,
            output_filename=output_filename,
        )

        return func(*args, **kwargs)

    return wrapper