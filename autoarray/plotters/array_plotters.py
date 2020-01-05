from autoarray import conf
from autoarray import exc
import matplotlib
from functools import wraps

backend = conf.get_matplotlib_backend()

matplotlib.use(backend)
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import itertools

from autoarray.plotters import plotters


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

        array_plotter = kwargs["array_plotter"]

        label_title = plotters.label_title_from_plotter(plotter=array_plotter, func=func)
        label_yunits = plotters.label_yunits_from_plotter(plotter=array_plotter)
        label_xunits = plotters.label_xunits_from_plotter(plotter=array_plotter)
        output_filename = plotters.output_filename_from_plotter_and_func(plotter=array_plotter, func=func)

        kwargs["array_plotter"] = array_plotter.plotter_with_new_labels_and_filename(
            label_title=label_title,
            label_yunits=label_yunits,
            label_xunits=label_xunits,
            output_filename=output_filename,
        )

        return func(*args, **kwargs)

    return wrapper


class ArrayPlotter(plotters.Plotter):
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
        titlesize=None,
        xlabelsize=None,
        ylabelsize=None,
        xyticksize=None,
        mask_pointsize=None,
        border_pointsize=None,
        point_pointsize=None,
        grid_pointsize=None,
        label_title=None,
        label_yunits=None,
        label_xunits=None,
        label_yticks=None,
        label_xticks=None,
        output_path=None,
        output_format="show",
        output_filename=None,
    ):

        super(ArrayPlotter, self).__init__(
            is_sub_plotter=is_sub_plotter,
            use_scaled_units=use_scaled_units,
            unit_conversion_factor=unit_conversion_factor,
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
            titlesize=titlesize,
            xlabelsize=xlabelsize,
            ylabelsize=ylabelsize,
            xyticksize=xyticksize,
            mask_pointsize=mask_pointsize,
            border_pointsize=border_pointsize,
            point_pointsize=point_pointsize,
            grid_pointsize=grid_pointsize,
            label_title=label_title,
            label_yunits=label_yunits,
            label_xunits=label_xunits,
            label_yticks=label_yticks,
            label_xticks=label_xticks,
            output_path=output_path,
            output_format=output_format,
            output_filename=output_filename,
        )


    def plotter_as_sub_plotter(
        self,
    ):

        return ArrayPlotter(
            is_sub_plotter=True,
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
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
            titlesize=self.titlesize,
            xlabelsize=self.xlabelsize,
            ylabelsize=self.ylabelsize,
            xyticksize=self.xyticksize,
            mask_pointsize=self.mask_pointsize,
            border_pointsize=self.border_pointsize,
            point_pointsize=self.point_pointsize,
            grid_pointsize=self.grid_pointsize,
            label_title=self.label_title,
            label_yunits=self.label_yunits,
            label_xunits=self.label_xunits,
            label_yticks=self.label_yticks,
            label_xticks=self.label_xticks,
            output_path=self.output_path,
            output_format=self.output_format,
            output_filename=self.output_filename,
        )

    def plotter_with_new_labels_and_filename(
        self,
        label_title=None,
        label_yunits=None,
        label_xunits=None,
        output_filename=None,
    ):

        label_title = self.label_title if label_title is None else label_title
        label_yunits = self.label_yunits if label_yunits is None else label_yunits
        label_xunits = self.label_xunits if label_xunits is None else label_xunits
        output_filename = (
            self.output_filename if output_filename is None else output_filename
        )

        return ArrayPlotter(
            is_sub_plotter=self.is_sub_plotter,
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
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
            titlesize=self.titlesize,
            xlabelsize=self.xlabelsize,
            ylabelsize=self.ylabelsize,
            xyticksize=self.xyticksize,
            mask_pointsize=self.mask_pointsize,
            border_pointsize=self.border_pointsize,
            point_pointsize=self.point_pointsize,
            grid_pointsize=self.grid_pointsize,
            label_title=label_title,
            label_yunits=label_yunits,
            label_xunits=label_xunits,
            label_yticks=self.label_yticks,
            label_xticks=self.label_xticks,
            output_path=self.output_path,
            output_format=self.output_format,
            output_filename=output_filename,
        )

    def plot_array(
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
        xlabelsize : int
            The fontsize of the x axes label.
        ylabelsize : int
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
            array_plotters.plot_array(
            array=image, origin=(0.0, 0.0), mask=circular_mask,
            border=False, points=[[1.0, 1.0], [2.0, 2.0]], grid=None, as_subplot=False,
            unit_label='scaled', kpc_per_arcsec=None, figsize=(7,7), aspect='auto',
            cmap='jet', norm='linear, norm_min=None, norm_max=None, linthresh=None, linscale=None,
            cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
            title='Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
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

        self.set_title()
        self.set_yx_labels_and_ticksize()

        self.set_colorbar()
        self.plot_origin(array=array, include_origin=include_origin)
        self.plot_mask(mask=mask)
        self.plot_lines(line_lists=lines)
        self.plot_border(mask=mask, border=border)
        self.plot_points(points=points)
        self.plot_grid(grid=grid)
        self.plot_centres(centres=centres)
        self.output_figure(array)
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

        fig = self.setup_figure()

        norm_scale = self.get_normalization_scale(array=array)

        if self.aspect in "square":
            aspect = float(array.shape_2d[1]) / float(array.shape_2d[0])
        else:
            aspect = self.aspect

        plt.imshow(
            X=array.in_2d, aspect=aspect, cmap=self.cmap, norm=norm_scale, extent=extent
        )
        self.set_yxticks(array=array, extent=extent)

        return fig

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

            colors = itertools.cycle(["m", "y", "r", "w", "c", "b", "g", "k"])

            for centres_of_galaxy in centres:
                color = next(colors)
                for centre in centres_of_galaxy:
                    plt.scatter(y=centre[0], x=centre[1], s=300, c=color, marker="x")

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
