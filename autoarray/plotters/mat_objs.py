from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
import matplotlib.pyplot as plt
import numpy as np
import inspect

from autoarray import exc


class Ticks(object):
    def __init__(
            self,
            ysize=None,
            xsize=None,
            unit_conversion_factor=None,
            y_manual=None,
            x_manual=None,
    ):

        self.ysize = ysize

        self.xsize = xsize

        self.unit_conversion_factor = unit_conversion_factor
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
    ):

        self.title = title
        self._yunits = yunits
        self._xunits = xunits

        self.titlesize = titlesize
        self.ysize = ysize
        self.xsize = xsize

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
    def __init__(self, path=None, filename=None, format="show", bypass=False):

        self.path = path
        self.filename = filename
        self.format = format
        self.bypass = bypass

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