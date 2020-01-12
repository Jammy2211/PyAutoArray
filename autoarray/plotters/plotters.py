from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
from functools import wraps
import copy

from autoarray import exc
from autoarray.plotters import mat_objs
import itertools
from autoarray.operators.inversion import mappers


def setting(section, name, python_type):
    return conf.instance.visualize_figures.get(section, name, python_type)


def load_setting(section, value, name, python_type):
    return (
        value
        if value is not None
        else setting(section=section, name=name, python_type=python_type)
    )


def load_figure_setting(section, name, python_type):
    return conf.instance.visualize_figures.get(section, name, python_type)

def load_subplot_setting(section, name, python_type):
    return conf.instance.visualize_subplots.get(section, name, python_type)


class AbstractPlotter(object):
    def __init__(
        self,
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
            line_pointsize=None,
            include_legend=None,
            legend_fontsize=None,
            ticks=mat_objs.Ticks(),
            labels=mat_objs.Labels(),
            output=mat_objs.Output(),
    ):

        if use_scaled_units is not None:
            self.use_scaled_units = use_scaled_units
        else:
            try:
                conf.instance.visualize_general.get("general", "use_scaled_units", bool)
            except:
                self.use_scaled_units = True

        self.unit_conversion_factor = unit_conversion_factor
        try:
            self.plot_in_kpc = plot_in_kpc if plot_in_kpc is not None else conf.instance.visualize_general.get(
                "general",
                "plot_in_kpc",
                bool,
            )
        except:
            self.plot_in_kpc = None

        if not self.is_sub_plotter:
            load_setting_func = load_figure_setting
        else:
            load_setting_func = load_subplot_setting

        self.figsize = figsize if figsize is not None else load_setting_func("figures", "figsize", str)
        if self.figsize == "auto":
            self.figsize = None
        elif isinstance(self.figsize, str):
            self.figsize = tuple(map(int, self.figsize[1:-1].split(",")))
        self.aspect = aspect if aspect is not None else load_setting_func(
            "figures", "aspect", str
        )

        self.cmap = cmap if cmap is not None else load_setting_func(
            "settings", "cmap", str
        )
        self.norm = norm if norm is not None else load_setting_func(
            "settings", "norm", str
        )
        self.norm_min = norm_min if norm_min is not None else load_setting_func(
            "settings", "norm_min", float
        )
        self.norm_max = norm_max if norm_max is not None else load_setting_func(
            "settings", "norm_max", float
        )
        self.linthresh = linthresh if linthresh is not None else load_setting_func(
            "settings", "linthresh", float
        )
        self.linscale = linscale if linscale is not None else load_setting_func(
            "settings", "linscale", float
        )

        self.cb_ticksize = cb_ticksize if cb_ticksize is not None else load_setting_func(
            "settings", "cb_ticksize", int
        )
        self.cb_fraction = cb_fraction if cb_fraction is not None else load_setting_func(
            "settings", "cb_fraction", float
        )
        self.cb_pad = cb_pad if cb_pad is not None else load_setting_func(
            "settings", "cb_pad", float
        )
        self.cb_tick_values = cb_tick_values
        self.cb_tick_labels = cb_tick_labels

        self.mask_pointsize = mask_pointsize if mask_pointsize is not None else load_setting_func(
            "settings",
            "mask_pointsize",
            int,
        )
        self.border_pointsize = border_pointsize if border_pointsize is not None else load_setting_func(
            "settings",
            "border_pointsize",
            int,
        )
        self.point_pointsize = point_pointsize if point_pointsize is not None else load_setting_func(
            "settings",
            "point_pointsize",
            int,
        )
        self.grid_pointsize = grid_pointsize if grid_pointsize is not None else load_setting_func(
            "settings",
            "grid_pointsize",
            int,
        )

        ticks_ysize = ticks.ysize if ticks.ysize is not None else load_setting_func(
            "ticks",
            "ysize",
            int,
        )

        ticks_xsize = ticks.xsize if ticks.xsize is not None else load_setting_func(
            "ticks",
            "xsize",
            int,
        )

        self.ticks = mat_objs.Ticks(
            ysize=ticks_ysize,
            xsize=ticks_xsize,
            y_manual=ticks.y_manual,
            x_manual=ticks.x_manual,
        )

        labels_titlesize = labels.titlesize if labels.titlesize is not None else load_setting_func(
            "labels",
            "titlesize",
            int,
        )

        labels_ysize = labels.ysize if labels.ysize is not None else load_setting_func(
            "labels",
            "ysize",
            int,
        )

        labels_xsize = labels.xsize if labels.xsize is not None else load_setting_func(
            "labels",
            "xsize",
            int,
        )

        self.labels = mat_objs.Labels(
            title=labels.title,
            yunits=labels._yunits,
            xunits=labels._xunits,
            titlesize=labels_titlesize,
            ysize=labels_ysize,
            xsize=labels_xsize,
            use_scaled_units=use_scaled_units,
        )

        self.output = mat_objs.Output(
            path=output.path, format=output.format, filename=output.filename, bypass=self.is_sub_plotter
        )

        self.line_pointsize = line_pointsize
        self.include_legend = include_legend
        self.legend_fontsize = legend_fontsize

    @property
    def array(self):
        return ArrayPlotter(
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
            plot_in_kpc=self.plot_in_kpc,
            figsize=self.figsize,
            aspect=self.aspect,
            cmap=self.cmap,
            norm=self.norm,
            norm_min=self.norm_min,
            norm_max=self.norm_max,
            linthresh=self.linthresh,
            linscale=self.linscale,
            cb_ticksize=self.cb_ticksize,
            cb_fraction=self.cb_fraction,
            cb_pad=self.cb_pad,
            cb_tick_values=self.cb_tick_values,
            cb_tick_labels=self.cb_tick_labels,
            mask_pointsize=self.mask_pointsize,
            border_pointsize=self.border_pointsize,
            point_pointsize=self.point_pointsize,
            grid_pointsize=self.grid_pointsize,
            ticks=self.ticks,
            labels=self.labels,
            output=self.output,
        )

    @property
    def grid(self):
        return GridPlotter(
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
            plot_in_kpc=self.plot_in_kpc,
            figsize=self.figsize,
            aspect=self.aspect,
            cmap=self.cmap,
            norm=self.norm,
            norm_min=self.norm_min,
            norm_max=self.norm_max,
            linthresh=self.linthresh,
            linscale=self.linscale,
            cb_ticksize=self.cb_ticksize,
            cb_fraction=self.cb_fraction,
            cb_pad=self.cb_pad,
            cb_tick_values=self.cb_tick_values,
            cb_tick_labels=self.cb_tick_labels,
            grid_pointsize=self.grid_pointsize,
            grid_pointcolor="k",
            ticks=self.ticks,
            labels=self.labels,
            output=self.output,
        )

    @property
    def mapper(self):
        return MapperPlotter(
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
            plot_in_kpc=self.plot_in_kpc,
            figsize=self.figsize,
            aspect=self.aspect,
            cmap=self.cmap,
            norm=self.norm,
            norm_min=self.norm_min,
            norm_max=self.norm_max,
            linthresh=self.linthresh,
            linscale=self.linscale,
            cb_ticksize=self.cb_ticksize,
            cb_fraction=self.cb_fraction,
            cb_pad=self.cb_pad,
            cb_tick_values=self.cb_tick_values,
            cb_tick_labels=self.cb_tick_labels,
            grid_pointsize=self.grid_pointsize,
            grid_pointcolor="k",
            ticks=self.ticks,
            labels=self.labels,
            output=self.output,
        )

    @property
    def line(self):
        return LinePlotter(
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
            plot_in_kpc=self.plot_in_kpc,
            figsize=self.figsize,
            aspect=self.aspect,
            line_pointsize=self.line_pointsize,
            include_legend=self.include_legend,
            legend_fontsize=self.legend_fontsize,
            ticks=self.ticks,
            labels=self.labels,
            output=self.output,
        )

    def setup_figure(self):
        if not plt.fignum_exists(num=1):
            plt.figure(figsize=self.figsize)

    def close_figure(self):
        if plt.fignum_exists(num=1):
            plt.close()

    @property
    def is_sub_plotter(self):
        raise NotImplementedError()

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

            if not any(isinstance(el, list) for el in line_lists):

                for line in line_lists:
                    if len(line) != 0:
                        plt.plot(line[:, 1], line[:, 0], c="w", lw=2.0, zorder=200)

            else:

                for line_list in line_lists:
                    if line_list is not None:
                        for line in line_list:
                            if len(line) != 0:
                                plt.plot(
                                    line[:, 1], line[:, 0], c="w", lw=2.0, zorder=200
                                )

    def plotter_with_new_labels(self, labels=mat_objs.Labels()):

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
        plotter.labels.titlesize = (
            labels.titlesize if labels.titlesize is not None else self.labels.titlesize
        )
        plotter.labels.ysize = (
            labels.ysize if labels.ysize is not None else self.labels.ysize
        )
        plotter.labels.xsize = (
            labels.xsize if labels.xsize is not None else self.labels.xsize
        )

        return plotter

    def plotter_with_new_unit_conversion_factor(self, unit_conversion_factor=None):

        plotter = copy.deepcopy(self)

        plotter.unit_conversion_factor = (
            unit_conversion_factor
            if unit_conversion_factor is not None
            else self.unit_conversion_factor
        )

        plotter.ticks.unit_conversion_factor = (
            unit_conversion_factor
            if unit_conversion_factor is not None
            else self.unit_conversion_factor
        )

        return plotter

    def plotter_with_new_output(self, output=mat_objs.Output()):

        plotter = copy.deepcopy(self)

        plotter.output.path = (
            output.path if output.path is not None else self.output.path
        )

        plotter.output.filename = (
            output.filename if output.filename is not None else self.output.filename
        )

        return plotter


class Plotter(AbstractPlotter):
    def __init__(
        self,
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
            line_pointsize=None,
            include_legend=None,
            legend_fontsize=None,
        ticks=mat_objs.Ticks(),
        labels=mat_objs.Labels(),
        output=mat_objs.Output(),
    ):
        
        super(Plotter, self).__init__(
            use_scaled_units=use_scaled_units,
            unit_conversion_factor=unit_conversion_factor,
            plot_in_kpc=plot_in_kpc,
            figsize=figsize,
            aspect=aspect,
            cmap=cmap,
            norm=norm,
            norm_min=norm_min,
            norm_max=norm_max,
            linthresh=linthresh,
            linscale=linscale,
            cb_ticksize=cb_ticksize,
            cb_fraction=cb_fraction,
            cb_pad=cb_pad,
            cb_tick_values=cb_tick_values,
            cb_tick_labels=cb_tick_labels,
            mask_pointsize=mask_pointsize,
            border_pointsize=border_pointsize,
            point_pointsize=point_pointsize,
            grid_pointsize=grid_pointsize,
            line_pointsize=line_pointsize,
            include_legend=include_legend,
            legend_fontsize=legend_fontsize,
            ticks=ticks,
            labels=labels,
            output=output,
        )

    @property
    def is_sub_plotter(self):
        return False


class SubPlotter(AbstractPlotter):
    def __init__(
        self,
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
            line_pointsize=None,
            include_legend=None,
            legend_fontsize=None,
        ticks=mat_objs.Ticks(),
        labels=mat_objs.Labels(),
        output=mat_objs.Output(),
    ):

        super(SubPlotter, self).__init__(
            use_scaled_units=use_scaled_units,
            unit_conversion_factor=unit_conversion_factor,
            plot_in_kpc=plot_in_kpc,
            figsize=figsize,
            aspect=aspect,
            cmap=cmap,
            norm=norm,
            norm_min=norm_min,
            norm_max=norm_max,
            linthresh=linthresh,
            linscale=linscale,
            cb_ticksize=cb_ticksize,
            cb_fraction=cb_fraction,
            cb_pad=cb_pad,
            cb_tick_values=cb_tick_values,
            cb_tick_labels=cb_tick_labels,
            mask_pointsize=mask_pointsize,
            border_pointsize=border_pointsize,
            point_pointsize=point_pointsize,
            grid_pointsize=grid_pointsize,
            line_pointsize=line_pointsize,
            include_legend=include_legend,
            legend_fontsize=legend_fontsize,
            ticks=ticks,
            labels=labels,
            output=output,
        )

    def setup_subplot_figure(self, number_subplots):
        """Setup a figure for plotting an image.

        Parameters
        -----------
        figsize : (int, int)
            The size of the figure in (rows, columns).
        as_subplot : bool
            If the figure is a subplot, the setup_figure function is omitted to ensure that each subplot does not create a \
            new figure and so that it can be output using the *output.output_figure(structure=None, is_sub_plotter=False)* function.
        """

        figsize = self.get_subplot_figsize(number_subplots=number_subplots)
        plt.figure(figsize=figsize)

    def setup_subplot(self, number_subplots, subplot_index):
        rows, columns = self.get_subplot_rows_columns(number_subplots=number_subplots)
        plt.subplot(rows, columns, subplot_index)

    def get_subplot_rows_columns(self, number_subplots):
        """Get the size of a sub plotters in (rows, columns), based on the number of subplots that are going to be plotted.

        Parameters
        -----------
        number_subplots : int
            The number of subplots that are to be plotted in the figure.
        """
        if number_subplots <= 2:
            return 1, 2
        elif number_subplots <= 4:
            return 2, 2
        elif number_subplots <= 6:
            return 2, 3
        elif number_subplots <= 9:
            return 3, 3
        elif number_subplots <= 12:
            return 3, 4
        elif number_subplots <= 16:
            return 4, 4
        elif number_subplots <= 20:
            return 4, 5
        else:
            return 6, 6

    def get_subplot_figsize(self, number_subplots):
        """Get the size of a sub plotters in (rows, columns), based on the number of subplots that are going to be plotted.

        Parameters
        -----------
        number_subplots : int
            The number of subplots that are to be plotted in the figure.
        """

        if self.figsize is not None:
            return self.figsize

        if number_subplots <= 2:
            return (18, 8)
        elif number_subplots <= 4:
            return (13, 10)
        elif number_subplots <= 6:
            return (18, 12)
        elif number_subplots <= 9:
            return (25, 20)
        elif number_subplots <= 12:
            return (25, 20)
        elif number_subplots <= 16:
            return (25, 20)
        elif number_subplots <= 20:
            return (25, 20)
        else:
            return (25, 20)

    @property
    def is_sub_plotter(self):
        return True


class ArrayPlotter(AbstractPlotter):
    def __init__(
        self,
        use_scaled_units,
        unit_conversion_factor,
        plot_in_kpc,
        figsize,
        aspect,
        cmap,
        norm,
        norm_min,
        norm_max,
        linthresh,
        linscale,
        cb_ticksize,
        cb_fraction,
        cb_pad,
        cb_tick_values,
        cb_tick_labels,
        mask_pointsize,
        border_pointsize,
        point_pointsize,
        grid_pointsize,
        ticks,
        labels,
        output,
    ):


        self.figsize = figsize
        self.aspect = aspect

        self.use_scaled_units = use_scaled_units
        self.unit_conversion_factor = unit_conversion_factor

        self.plot_in_kpc = plot_in_kpc

        self.cmap = cmap
        self.norm = norm
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.linthresh = linthresh
        self.linscale = linscale

        self.cb_ticksize = cb_ticksize
        self.cb_fraction = cb_fraction
        self.cb_pad = cb_pad
        self.cb_tick_values = cb_tick_values
        self.cb_tick_labels = cb_tick_labels

        self.mask_pointsize = mask_pointsize
        self.border_pointsize = border_pointsize
        self.point_pointsize = point_pointsize
        self.grid_pointsize = grid_pointsize

        self.ticks = ticks

        self.labels = labels

        self.output = output

    def plot(
        self,
        array,
        include_origin=False,
        mask=None,
        border=None,
        lines=None,
        points=None,
        centres=None,
        grid=None,
    ):
        """Plot an array of data_type as a figure.

        Parameters
        -----------
        settings : PlotterSettings
            Settings
        include : PlotterInclude
            Include
        labels : PlotterLabels
            labels
        outputs : PlotterOutputs
            outputs
        array : data_type.array.aa.Scaled
            The 2D array of data_type which is plotted.
        origin : (float, float).
            The origin of the coordinate system of the array, which is plotted as an 'x' on the image if input.
        mask : data_type.array.mask.Mask
            The mask applied to the array, the edge of which is plotted as a set of points over the plotted array.
        extract_array_from_mask : bool
            The plotter array is extracted using the mask, such that masked values are plotted as zeros. This ensures \
            bright features outside the mask do not impact the color map of the plotters.
        zoom_around_mask : bool
            If True, the 2D region of the array corresponding to the rectangle encompassing all unmasked values is \
            plotted, thereby zooming into the region of interest.
        border : bool
            If a mask is supplied, its borders pixels (e.g. the exterior edge) is plotted if this is *True*.
        points : [[]]
            Lists of (y,x) coordinates on the image which are plotted as colored dots, to highlight specific pixels.
        grid : data_type.array.aa.Grid
            A grid of (y,x) coordinates which may be plotted over the plotted array.
        as_subplot : bool
            Whether the array is plotted as part of a subplot, in which case the grid figure is not opened / closed.
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float or None
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        figsize : (int, int)
            The size of the figure in (rows, columns).
        aspect : str
            The aspect ratio of the array, specifically whether it is forced to be square ('equal') or adapts its size to \
            the figure size ('auto').
        cmap : str
            The colormap the array is plotted using, which may be chosen from the standard matplotlib colormaps.
        norm : str
            The normalization of the colormap used to plotters the image, specifically whether it is linear ('linear'), log \
            ('log') or a symmetric log normalization ('symmetric_log').
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
        cb_ticksize : int
            The size of the tick labels on the colorbar.
        cb_fraction : float
            The fraction of the figure that the colorbar takes up, which resizes the colorbar relative to the figure.
        cb_pad : float
            Pads the color bar in the figure, which resizes the colorbar relative to the figure.
        xsize : int
            The fontsize of the x axes label.
        ysize : int
            The fontsize of the y axes label.
        xyticksize : int
            The font size of the x and y ticks on the figure axes.
        mask_pointsize : int
            The size of the points plotted to show the mask.
        border_pointsize : int
            The size of the points plotted to show the borders.
        point_pointsize : int
            The size of the points plotted to show points on the image.
        grid_pointsize : int
            The size of the points plotted to show the grid.
        xticks_manual :  [] or None
            If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
        yticks_manual :  [] or None
            If input, the yticks do not use the array's default yticks but instead overwrite them as these values.
        output_path : str
            The path on the hard-disk where the figure is output.
        output_filename : str
            The filename of the figure that is output.
        output_format : str
            The format the figue is output:
            'show' - display on computer screen.
            'png' - output to hard-disk as a png.
            'fits' - output to hard-disk as a fits file.'

        Returns
        --------
        None

        Examples
        --------
            plotters.plot_array(
            array=image, origin=(0.0, 0.0), mask=circular_mask,
            border=False, points=[[1.0, 1.0], [2.0, 2.0]], grid=None, as_subplot=False,
            unit_label='scaled', kpc_per_arcsec=None, figsize=(7,7), aspect='auto',
            cmap='jet', norm='linear, norm_min=None, norm_max=None, linthresh=None, linscale=None,
            cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
            title='Image', titlesize=16, xsize=16, ysize=16, xyticksize=16,
            mask_pointsize=10, border_pointsize=2, position_pointsize=10, grid_pointsize=10,
            xticks_manual=None, yticks_manual=None,
            output_path='/path/to/output', output_format='png', output_filename='image')
        """

        if array is None or np.all(array == 0):
            return

        if array.pixel_scales is None and self.use_scaled_units:
            raise exc.ArrayException(
                "You cannot plot an array using its scaled unit_label if the input array does not have "
                "a pixel scales attribute."
            )

        array = array.in_1d_binned

        if array.mask.is_all_false:
            buffer = 0
        else:
            buffer = 1

        extent = array.extent_of_zoomed_array(buffer=buffer)
        array = array.zoomed_around_mask(buffer=buffer)

        self.plot_figure(array=array, extent=extent)

        self.ticks.set_yticks(
            array=array,
            extent=extent,
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
        )
        self.ticks.set_xticks(
            array=array,
            extent=extent,
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
        )

        self.labels.set_title()
        self.labels.set_yunits(include_brackets=True)
        self.labels.set_xunits(include_brackets=True)

        self.set_colorbar()
        self.plot_origin(array=array, include_origin=include_origin)
        self.plot_mask(mask=mask)
        self.plot_lines(line_lists=lines)
        self.plot_border(mask=mask, border=border)
        self.plot_points(points=points)
        self.plot_grid(grid=grid)
        self.plot_centres(centres=centres)
        self.output.to_figure(structure=array)
        self.close_figure()

    def plot_figure(self, array, extent):
        """Open a matplotlib figure and plotters the array of data_type on it.

        Parameters
        -----------
        array : data_type.array.aa.Scaled
            The 2D array of data_type which is plotted.
        as_subplot : bool
            Whether the array is plotted as part of a subplot, in which case the grid figure is not opened / closed.
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float or None
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        figsize : (int, int)
            The size of the figure in (rows, columns).
        aspect : str
            The aspect ratio of the array, specifically whether it is forced to be square ('equal') or adapts its size to \
            the figure size ('auto').
        cmap : str
            The colormap the array is plotted using, which may be chosen from the standard matplotlib colormaps.
        norm : str
            The normalization of the colormap used to plotters the image, specifically whether it is linear ('linear'), log \
            ('log') or a symmetric log normalization ('symmetric_log').
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
        xticks_manual :  [] or None
            If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
        yticks_manual :  [] or None
            If input, the yticks do not use the array's default yticks but instead overwrite them as these values.
        """

        self.setup_figure()

        norm_scale = self.get_normalization_scale(array=array)

        if self.aspect in "square":
            aspect = float(array.shape_2d[1]) / float(array.shape_2d[0])
        else:
            aspect = self.aspect

        plt.imshow(
            X=array.in_2d, aspect=aspect, cmap=self.cmap, norm=norm_scale, extent=extent
        )

    def get_normalization_scale(self, array):
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
            if self.norm_min == 0.0:
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

    def plot_origin(self, array, include_origin):
        """Plot the (y,x) origin ofo the array's coordinates as a 'x'.

        Parameters
        -----------
        array : data_type.array.aa.Scaled
            The 2D array of data_type which is plotted.
        origin : (float, float).
            The origin of the coordinate system of the array, which is plotted as an 'x' on the image if input.
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float or None
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        """
        if include_origin:
            plt.scatter(
                y=np.asarray(array.origin[0]),
                x=np.asarray(array.origin[1]),
                s=80,
                c="k",
                marker="x",
            )

    def plot_centres(self, centres):
        """Plot the (y,x) centres (e.g. of a mass profile) on the array as an 'x'.

        Parameters
        -----------
        array : data_type.array.aa.Scaled
            The 2D array of data_type which is plotted.
        centres : [[tuple]]
            The list of centres; centres in the same list entry are colored the same.
        use_scaled_units_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float or None
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        """
        if centres is not None:

            if not any(isinstance(el, list) for el in centres):

                for centre in centres:
                    plt.scatter(y=centre[0], x=centre[1], s=300, marker="x")

            else:

                colors = itertools.cycle(["m", "y", "r", "w", "c", "b", "g", "k"])

                for centres_of_galaxy in centres:
                    color = next(colors)
                    for centre in centres_of_galaxy:
                        plt.scatter(
                            y=centre[0], x=centre[1], s=300, c=color, marker="x"
                        )

    def plot_mask(self, mask):
        """Plot the mask of the array on the figure.

        Parameters
        -----------
        mask : ndarray of data_type.array.mask.Mask
            The mask applied to the array, the edge of which is plotted as a set of points over the plotted array.
        use_scaled_units_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float or None
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        pointsize : int
            The size of the points plotted to show the mask.
        """

        if mask is not None:
            plt.gca()
            edge_pixels = (
                mask.regions._mask_2d_index_for_mask_1d_index[
                    mask.regions._edge_1d_indexes
                ]
                + 0.5
            )

            edge_scaled = mask.geometry.grid_scaled_from_grid_pixels_1d(
                grid_pixels_1d=edge_pixels
            )

            plt.scatter(
                y=np.asarray(edge_scaled[:, 0]),
                x=np.asarray(edge_scaled[:, 1]),
                s=self.mask_pointsize,
                c="k",
            )

    def plot_border(self, mask, border):
        """Plot the borders of the mask or the array on the figure.

        Parameters
        -----------t.
        mask : ndarray of data_type.array.mask.Mask
            The mask applied to the array, the edge of which is plotted as a set of points over the plotted array.
        border : bool
            If a mask is supplied, its borders pixels (e.g. the exterior edge) is plotted if this is *True*.
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        kpc_per_arcsec : float or None
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        border_pointsize : int
            The size of the points plotted to show the borders.
        """
        if border and mask is not None:
            plt.gca()
            border_grid = mask.geometry.border_grid.in_1d_binned

            plt.scatter(
                y=np.asarray(border_grid[:, 0]),
                x=np.asarray(border_grid[:, 1]),
                s=self.border_pointsize,
                c="y",
            )

    def plot_points(self, points):
        """Plot a set of points over the array of data_type on the figure.

        Parameters
        -----------
        points : [[]]
            Lists of (y,x) coordinates on the image which are plotted as colored dots, to highlight specific pixels.
        array : data_type.array.aa.Scaled
            The 2D array of data_type which is plotted.
        use_scaled_units_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float or None
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        pointsize : int
            The size of the points plotted to show the input points.
        """

        if points is not None:

            points = list(map(lambda position_set: np.asarray(position_set), points))
            point_colors = itertools.cycle(["m", "y", "r", "w", "c", "b", "g", "k"])
            for point_set in points:
                plt.scatter(
                    y=point_set[:, 0],
                    x=point_set[:, 1],
                    color=next(point_colors),
                    s=self.point_pointsize,
                )

    def plot_grid(self, grid):
        """Plot a grid of points over the array of data_type on the figure.

         Parameters
         -----------.
         grid_arcsec : ndarray or data_type.array.aa.Grid
             A grid of (y,x) coordinates in arc-seconds which may be plotted over the array.
         array : data_type.array.aa.Scaled
            The 2D array of data_type which is plotted.
         unit_label : str
             The label for the unit_label of the y / x axis of the plots.
         kpc_per_arcsec : float or None
             The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
         grid_pointsize : int
             The size of the points plotted to show the grid.
         """
        if grid is not None:

            plt.scatter(
                y=np.asarray(grid[:, 0]),
                x=np.asarray(grid[:, 1]),
                s=self.grid_pointsize,
                c="k",
            )


class GridPlotter(AbstractPlotter):
    def __init__(
        self,
        use_scaled_units,
        unit_conversion_factor,
        plot_in_kpc,
        figsize,
        aspect,
        cmap,
        norm,
        norm_min,
        norm_max,
        linthresh,
        linscale,
        cb_ticksize,
        cb_fraction,
        cb_pad,
        cb_tick_values,
        cb_tick_labels,
        grid_pointsize,
        grid_pointcolor,
        ticks,
        labels,
        output,
    ):

        self.figsize = figsize
        self.aspect = aspect

        self.use_scaled_units = use_scaled_units
        self.unit_conversion_factor = unit_conversion_factor

        self.plot_in_kpc = plot_in_kpc

        self.cmap = cmap
        self.norm = norm
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.linthresh = linthresh
        self.linscale = linscale

        self.cb_ticksize = cb_ticksize
        self.cb_fraction = cb_fraction
        self.cb_pad = cb_pad
        self.cb_tick_values = cb_tick_values
        self.cb_tick_labels = cb_tick_labels

        self.grid_pointsize = grid_pointsize
        self.grid_pointcolor = grid_pointcolor

        self.ticks = ticks

        self.labels = labels

        self.output = output

        self.grid_pointcolor = grid_pointcolor

    def plot(
        self,
        grid,
        colors=None,
        axis_limits=None,
        points=None,
        lines=None,
        symmetric_around_centre=True,
        bypass_limits=False,
    ):
        """Plot a grid of (y,x) Cartesian coordinates as a scatter plotters of points.

        Parameters
        -----------
        grid : data_type.array.aa.Grid
            The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2).
        axis_limits : []
            The axis limits of the figure on which the grid is plotted, following [xmin, xmax, ymin, ymax].
        points : []
            A set of points that are plotted in a different colour for emphasis (e.g. to show the mappings between \
            different planes).
        as_subplot : bool
            Whether the grid is plotted as part of a subplot, in which case the grid figure is not opened / closed.
        label_yunits : str
            The label of the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        figsize : (int, int)
            The size of the figure in (rows, columns).
        pointsize : int
            The size of the points plotted on the grid.
        xyticksize : int
            The font size of the x and y ticks on the figure axes.
        title : str
            The text of the title.
        titlesize : int
            The size of of the title of the figure.
        xsize : int
            The fontsize of the x axes label.
        ysize : int
            The fontsize of the y axes label.
        output_path : str
            The path on the hard-disk where the figure is output.
        output_filename : str
            The filename of the figure that is output.
        output_format : str
            The format the figue is output:
            'show' - display on computer screen.
            'png' - output to hard-disk as a png.
        """

        self.setup_figure()

        if colors is not None:

            plt.cm.get_cmap(self.cmap)

        plt.scatter(
            y=np.asarray(grid[:, 0]),
            x=np.asarray(grid[:, 1]),
            c=colors,
            s=self.grid_pointsize,
            marker=".",
            cmap=self.cmap,
        )

        if colors is not None:

            self.set_colorbar()

        self.labels.set_title()
        self.labels.set_yunits(include_brackets=True)
        self.labels.set_xunits(include_brackets=True)

        if not bypass_limits:

            self.set_axis_limits(
                axis_limits=axis_limits,
                grid=grid,
                symmetric_around_centre=symmetric_around_centre,
            )

        self.ticks.set_yticks(
            array=None,
            extent=grid.extent,
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
            symmetric_around_centre=symmetric_around_centre,
        )
        self.ticks.set_xticks(
            array=None,
            extent=grid.extent,
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
            symmetric_around_centre=symmetric_around_centre,
        )

        self.plot_points(grid=grid, points=points)
        self.plot_lines(line_lists=lines)

        self.output.to_figure(structure=grid)
        self.close_figure()

    def set_axis_limits(self, axis_limits, grid, symmetric_around_centre):
        """Set the axis limits of the figure the grid is plotted on.

        Parameters
        -----------
        axis_limits : []
            The axis limits of the figure on which the grid is plotted, following [xmin, xmax, ymin, ymax].
        """
        if axis_limits is not None:
            plt.axis(axis_limits)
        elif symmetric_around_centre:
            ymin = np.min(grid[:, 0])
            ymax = np.max(grid[:, 0])
            xmin = np.min(grid[:, 1])
            xmax = np.max(grid[:, 1])
            x = np.max([np.abs(xmin), np.abs(xmax)])
            y = np.max([np.abs(ymin), np.abs(ymax)])
            axis_limits = [-x, x, -y, y]
            plt.axis(axis_limits)

    def plot_points(self, grid, points):
        """Plot a subset of points in a different color, to highlight a specifc region of the grid (e.g. how certain \
        pixels map between different planes).

        Parameters
        -----------
        grid : ndarray or data_type.array.aa.Grid
            The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2).
        points : []
            A set of points that are plotted in a different colour for emphasis (e.g. to show the mappings between \
            different planes).
        pointcolor : str or None
            The color the points should be plotted. If None, the points are iterated through a cycle of colors.
        """
        if points is not None:

            if self.grid_pointcolor is None:

                point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
                for point_set in points:
                    plt.scatter(
                        y=np.asarray(grid[point_set, 0]),
                        x=np.asarray(grid[point_set, 1]),
                        s=8,
                        color=next(point_colors),
                    )

            else:

                for point_set in points:
                    plt.scatter(
                        y=np.asarray(grid[point_set, 0]),
                        x=np.asarray(grid[point_set, 1]),
                        s=8,
                        color=self.grid_pointcolor,
                    )


class MapperPlotter(GridPlotter):
    def __init__(
        self,
        use_scaled_units,
        unit_conversion_factor,
        plot_in_kpc,
        figsize,
        aspect,
        cmap,
        norm,
        norm_min,
        norm_max,
        linthresh,
        linscale,
        cb_ticksize,
        cb_fraction,
        cb_pad,
        cb_tick_values,
        cb_tick_labels,
        grid_pointsize,
        grid_pointcolor,
        ticks,
        labels,
        output,
    ):

        self.figsize = figsize
        self.aspect = aspect

        self.use_scaled_units = use_scaled_units
        self.unit_conversion_factor = unit_conversion_factor

        self.plot_in_kpc = plot_in_kpc

        self.cmap = cmap
        self.norm = norm
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.linthresh = linthresh
        self.linscale = linscale

        self.cb_ticksize = cb_ticksize
        self.cb_fraction = cb_fraction
        self.cb_pad = cb_pad
        self.cb_tick_values = cb_tick_values
        self.cb_tick_labels = cb_tick_labels

        self.grid_pointsize = grid_pointsize
        self.grid_pointcolor = grid_pointcolor

        self.ticks = ticks

        self.labels = labels

        self.output = output

    def plot(
        self,
        mapper,
        include_centres=False,
        include_grid=False,
        include_border=False,
        image_pixels=None,
        source_pixels=None,
    ):

        if isinstance(mapper, mappers.MapperRectangular):

            self.plot_rectangular_mapper(
                mapper=mapper,
                include_centres=include_centres,
                include_grid=include_grid,
                include_border=include_border,
                image_pixels=image_pixels,
                source_pixels=source_pixels,
            )

        else:

            self.plot_voronoi_mapper(
                mapper=mapper,
                include_centres=include_centres,
                include_grid=include_grid,
                include_border=include_border,
                image_pixels=image_pixels,
                source_pixels=source_pixels,
            )

    def plot_rectangular_mapper(
        self,
        mapper,
        include_centres=False,
        include_grid=False,
        include_border=False,
        image_pixels=None,
        source_pixels=None,
    ):

        self.setup_figure()

        self.set_axis_limits(
            axis_limits=mapper.pixelization_grid.extent,
            grid=None,
            symmetric_around_centre=False,
        )

        self.ticks.set_yticks(
            array=None,
            extent=mapper.pixelization_grid.extent,
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
        )
        self.ticks.set_xticks(
            array=None,
            extent=mapper.pixelization_grid.extent,
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
        )

        self.plot_rectangular_pixelization_lines(mapper=mapper)

        self.labels.set_title()
        self.labels.set_yunits(include_brackets=True)
        self.labels.set_xunits(include_brackets=True)

        self.plot_centres(mapper=mapper, include_centres=include_centres)

        self.plot_mapper_grid(include_grid=include_grid, mapper=mapper)

        self.plot_border(include_border=include_border, mapper=mapper)

        point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
        self.plot_source_plane_image_pixels(
            grid=mapper.grid, image_pixels=image_pixels, point_colors=point_colors
        )
        self.plot_source_plane_source_pixels(
            grid=mapper.grid,
            mapper=mapper,
            source_pixels=source_pixels,
            point_colors=point_colors,
        )

        self.output.to_figure(structure=None)
        self.close_figure()

    def plot_voronoi_mapper(
        self,
        mapper,
        source_pixel_values,
        include_centres=True,
        include_grid=True,
        include_border=False,
        lines=None,
        image_pixels=None,
        source_pixels=None,
    ):

        self.setup_figure()

        self.set_axis_limits(
            axis_limits=mapper.pixelization_grid.extent,
            grid=None,
            symmetric_around_centre=False,
        )

        self.ticks.set_yticks(
            array=None,
            extent=mapper.pixelization_grid.extent,
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
        )
        self.ticks.set_xticks(
            array=None,
            extent=mapper.pixelization_grid.extent,
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
        )

        regions_SP, vertices_SP = self.voronoi_finite_polygons_2d(
            voronoi=mapper.voronoi
        )

        color_values = source_pixel_values[:] / np.max(source_pixel_values)
        cmap = plt.get_cmap("jet")

        self.set_colorbar(cmap=cmap, color_values=source_pixel_values)

        for region, index in zip(regions_SP, range(mapper.pixels)):
            polygon = vertices_SP[region]
            col = cmap(color_values[index])
            plt.fill(*zip(*polygon), alpha=0.7, facecolor=col, lw=0.0)

        self.labels.set_title()
        self.labels.set_yunits(include_brackets=True)
        self.labels.set_xunits(include_brackets=True)

        self.plot_centres(mapper=mapper, include_centres=include_centres)

        self.plot_mapper_grid(include_grid=include_grid, mapper=mapper)

        self.plot_border(include_border=include_border, mapper=mapper)

        self.plot_lines(line_lists=lines)

        point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
        self.plot_source_plane_image_pixels(
            grid=mapper.grid, image_pixels=image_pixels, point_colors=point_colors
        )
        self.plot_source_plane_source_pixels(
            grid=mapper.grid,
            mapper=mapper,
            source_pixels=source_pixels,
            point_colors=point_colors,
        )

        self.output.to_figure(structure=None)
        self.close_figure()

    def voronoi_finite_polygons_2d(self, vor, radius=None):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.
        Parameters
        ----------
        vor : Voronoi
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

        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max() * 2

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

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

                t = vor.points[p2] - vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # hyper

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

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

    def plot_rectangular_pixelization_lines(self, mapper):

        ys = np.linspace(
            mapper.pixelization_grid.scaled_minima[0],
            mapper.pixelization_grid.scaled_maxima[0],
            mapper.pixelization_grid.shape_2d[0] + 1,
        )
        xs = np.linspace(
            mapper.pixelization_grid.scaled_minima[1],
            mapper.pixelization_grid.scaled_maxima[1],
            mapper.pixelization_grid.shape_2d[1] + 1,
        )

        # grid lines
        for x in xs:
            plt.plot([x, x], [ys[0], ys[-1]], color="black", linestyle="-")
        for y in ys:
            plt.plot([xs[0], xs[-1]], [y, y], color="black", linestyle="-")

    def set_colorbar(self, color_values):

        cax = cm.ScalarMappable(cmap=self.cmap)
        cax.set_array(color_values)

        if self.cb_tick_values is None and self.cb_tick_labels is None:
            plt.colorbar(mappable=cax, fraction=self.cb_fraction, pad=self.cb_pad)
        elif self.cb_tick_values is not None and self.cb_tick_labels is not None:
            cb = plt.colorbar(
                mappable=cax,
                fraction=self.cb_fraction,
                pad=self.cb_pad,
                ticks=self.cb_tick_values,
            )
            cb.ax.set_yticklabels(self.cb_tick_labels)

    def plot_centres(self, mapper, include_centres):

        if include_centres:

            pixelization_grid = mapper.pixelization_grid

            plt.scatter(
                y=pixelization_grid[:, 0], x=pixelization_grid[:, 1], s=3, c="r"
            )

    def plot_mapper_grid(self, mapper, include_grid):

        if include_grid:

            super(MapperPlotter, self).plot(grid=mapper.grid, bypass_limits=True)

    def plot_border(self, mapper, include_border):

        if include_border:

            border = mapper.grid[mapper.grid.mask.regions._sub_border_1d_indexes]

            self.plot(grid=border)

    def plot_image_pixels(self, grid, image_pixels, point_colors):

        if image_pixels is not None:

            for image_pixel_set in image_pixels:
                color = next(point_colors)
                plt.scatter(
                    y=np.asarray(grid[image_pixel_set, 0]),
                    x=np.asarray(grid[image_pixel_set, 1]),
                    color=color,
                    s=10.0,
                )

    def plot_image_plane_source_pixels(self, grid, mapper, source_pixels, point_colors):

        if source_pixels is not None:

            for source_pixel_set in source_pixels:
                color = next(point_colors)
                for source_pixel in source_pixel_set:
                    plt.scatter(
                        y=np.asarray(
                            grid[
                                mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index[
                                    source_pixel
                                ],
                                0,
                            ]
                        ),
                        x=np.asarray(
                            grid[
                                mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index[
                                    source_pixel
                                ],
                                1,
                            ]
                        ),
                        s=8,
                        color=color,
                    )

    def plot_source_plane_image_pixels(self, grid, image_pixels, point_colors):

        if image_pixels is not None:

            for image_pixel_set in image_pixels:
                color = next(point_colors)
                plt.scatter(
                    y=np.asarray(grid[[image_pixel_set], 0]),
                    x=np.asarray(grid[[image_pixel_set], 1]),
                    s=8,
                    color=color,
                )

    def plot_source_plane_source_pixels(
        self, grid, mapper, source_pixels, point_colors
    ):

        if source_pixels is not None:

            for source_pixel_set in source_pixels:
                color = next(point_colors)
                for source_pixel in source_pixel_set:
                    plt.scatter(
                        y=np.asarray(
                            grid[
                                mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index[
                                    source_pixel
                                ],
                                0,
                            ]
                        ),
                        x=np.asarray(
                            grid[
                                mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index[
                                    source_pixel
                                ],
                                1,
                            ]
                        ),
                        s=8,
                        color=color,
                    )


class LinePlotter(AbstractPlotter):
    def __init__(
        self,
        use_scaled_units,
        unit_conversion_factor,
        plot_in_kpc,
        include_legend,
        legend_fontsize,
        figsize,
        aspect,
        line_pointsize,
        ticks,
        labels,
        output,
    ):

        self.figsize = figsize
        self.aspect = aspect

        self.use_scaled_units = use_scaled_units
        self.unit_conversion_factor = unit_conversion_factor

        self.plot_in_kpc = plot_in_kpc

        self.ticks = ticks

        self.labels = labels

        self.output = output

        self.line_pointsize = line_pointsize
        self.include_legend = include_legend
        self.legend_fontsize = legend_fontsize

    def plot(
        self,
        y,
        x,
        label=None,
        plot_axis_type="semilogy",
        vertical_lines=None,
        vertical_line_labels=None,
    ):

        if y is None:
            return

        self.setup_figure()
        self.labels.set_title()

        if x is None:
            x = np.arange(len(y))

        self.plot_y_vs_x(y=y, x=x, plot_axis_type=plot_axis_type, label=label)

        self.labels.set_yunits(include_brackets=False)
        self.labels.set_xunits(include_brackets=False)

        self.plot_vertical_lines(
            vertical_lines=vertical_lines, vertical_line_labels=vertical_line_labels
        )

        self.set_legend()

        self.ticks.set_xticks(
            array=None,
            extent=[np.min(x), np.max(x)],
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
        )

        self.output.to_figure(structure=None)

        self.close_figure()

    def plot_y_vs_x(self, y, x, plot_axis_type, label):

        if plot_axis_type is "linear":
            plt.plot(x, y, label=label)
        elif plot_axis_type is "semilogy":
            plt.semilogy(x, y, label=label)
        elif plot_axis_type is "loglog":
            plt.loglog(x, y, label=label)
        elif plot_axis_type is "scatter":
            plt.scatter(x, y, label=label, s=self.line_pointsize)
        else:
            raise exc.PlottingException(
                "The plot_axis_type supplied to the plotter is not a valid string (must be linear "
                "| semilogy | loglog)"
            )

    def plot_vertical_lines(self, vertical_lines, vertical_line_labels):

        if vertical_lines is [] or vertical_lines is None:
            return

        for vertical_line, vertical_line_label in zip(
            vertical_lines, vertical_line_labels
        ):

            if self.unit_conversion_factor is None:
                x_value_plot = vertical_line
            else:
                x_value_plot = vertical_line * self.unit_conversion_factor

            plt.axvline(x=x_value_plot, label=vertical_line_label, linestyle="--")

    def set_legend(self):
        if self.include_legend:
            plt.legend(fontsize=self.legend_fontsize)


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
        inversion_image_pixelization_grid=None,
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
        self.inversion_image_pixelization_grid = self.load_include(
            value=inversion_image_pixelization_grid, name="inversion_image_pixelization_grid"
        )

    @staticmethod
    def load_include(value, name):

        return (
            conf.instance.visualize_general.get(section_name="include", attribute_name=name, attribute_type=bool)
            if value is None
            else value
        )

    def grid_from_grid(self, grid):

        if self.grid:
            return grid
        else:
            return None

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

    def real_space_mask_from_fit(self, fit):
        """Get the masks of the fit if the masks should be plotted on the fit.

        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractLensHyperFit
            The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
        include_mask : bool
            If *True*, the masks is plotted on the fit's datas.
        """
        if self.mask:
            return fit.masked_dataset.real_space_mask
        else:
            return None


def plotter_key_from_dictionary(dictionary):

    plotter_key = None

    for key, value in dictionary.items():
        if isinstance(value, AbstractPlotter):
            plotter_key = key

    if plotter_key is None:
        raise exc.PlottingException(
            "The plot function called could not locate a Plotter in the kwarg arguments"
            "in order to set the labels."
        )

    return plotter_key


def kpc_per_arcsec_of_object_from_dictionary(dictionary):

    kpc_per_arcsec = None

    for key, value in dictionary.items():
        if hasattr(value, "kpc_per_arcsec"):
            return value.kpc_per_arcsec

    return kpc_per_arcsec


def set_subplot_title(func):
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

        if not isinstance(plotter, SubPlotter):
            raise exc.PlottingException("The decorator set_subplot_title was applied to a function without a SubPlotter class")

        filename = plotter.output.filename_from_func(func=func)

        plotter = plotter.plotter_with_new_output(mat_objs.Output(filename=filename))

        kwargs[plotter_key] = plotter

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

        plotter_key = plotter_key_from_dictionary(dictionary=kwargs)
        plotter = kwargs[plotter_key]

        title = plotter.labels.title_from_func(func=func)
        yunits = plotter.labels.yunits_from_func(func=func)
        xunits = plotter.labels.xunits_from_func(func=func)

        plotter = plotter.plotter_with_new_labels(
            labels=mat_objs.Labels(title=title, yunits=yunits, xunits=xunits)
        )

        filename = plotter.output.filename_from_func(func=func)

        plotter = plotter.plotter_with_new_output(mat_objs.Output(filename=filename))

        kpc_per_arcsec = kpc_per_arcsec_of_object_from_dictionary(dictionary=kwargs)

        plotter = plotter.plotter_with_new_unit_conversion_factor(
            unit_conversion_factor=kpc_per_arcsec
        )

        kwargs[plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper
