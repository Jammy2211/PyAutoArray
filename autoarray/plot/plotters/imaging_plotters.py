from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot
from autoarray.plot.plotters import abstract_plotters
from autoarray.dataset import imaging as im
from autoarray.structures import grids


class AbstractImagingPlotter(abstract_plotters.AbstractPlotter):
    def __init__(self, imaging, mat_plot_2d, visuals_2d, include_2d):

        self.imaging = imaging

        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

    @property
    def visuals_with_include_2d(self):

        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin", grids.GridIrregular(grid=[self.imaging.image.origin])
            ),
            mask=self.extract_2d("mask", self.imaging.image.mask),
            border=self.extract_2d(
                "border",
                self.imaging.image.mask.geometry.border_grid_sub_1.in_1d_binned,
            ),
        )

    @abstract_plotters.for_figure
    def figure_image(self):
        """Plot the observed data_type of the imaging data_type.

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        image : data_type.ImagingData
            The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
        include_origin : True
            If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
        image_plane_pix_grid : np.ndarray or data_type.array.grid_stacks.PixGrid
            If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
            over the immage.
        """
        self.mat_plot_2d.plot_array(
            array=self.imaging.image, visuals_2d=self.visuals_with_include_2d
        )

    @abstract_plotters.for_figure
    def figure_noise_map(self):
        """Plot the noise_map of the imaging data_type.

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        image : data_type.ImagingData
            The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
        include_origin : True
            If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
        """
        self.mat_plot_2d.plot_array(
            array=self.imaging.noise_map, visuals_2d=self.visuals_with_include_2d
        )

    @abstract_plotters.for_figure
    def figure_psf(self):
        """Plot the PSF of the imaging data_type.

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        image : data_type.ImagingData
            The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
        include_origin : True
            If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
        """
        self.mat_plot_2d.plot_array(
            array=self.imaging.psf, visuals_2d=self.visuals_with_include_2d
        )

    @abstract_plotters.for_figure
    def figure_inverse_noise_map(self):
        """Plot the noise_map of the imaging data_type.

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        image : data_type.ImagingData
            The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
        include_origin : True
            If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
        """
        self.mat_plot_2d.plot_array(
            array=self.imaging.inverse_noise_map,
            visuals_2d=self.visuals_with_include_2d,
        )

    @abstract_plotters.for_figure
    def figure_signal_to_noise_map(self):
        """Plot the signal-to-noise_map of the imaging data_type.

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        image : data_type.ImagingData
            The imaging data_type, which includes the observed image, noise_map, PSF, signal-to-noise_map, etc.
        include_origin : True
            If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
        """
        self.mat_plot_2d.plot_array(
            array=self.imaging.signal_to_noise_map,
            visuals_2d=self.visuals_with_include_2d,
        )

    @abstract_plotters.for_figure
    def figure_absolute_signal_to_noise_map(self):
        """Plot the signal-to-noise_map of the imaging data_type.

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        image : data_type.ImagingData
            The imaging data_type, which includes the observed image, noise_map, PSF, signal-to-noise_map, etc.
        include_origin : True
            If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
        """
        self.mat_plot_2d.plot_array(
            array=self.imaging.absolute_signal_to_noise_map,
            visuals_2d=self.visuals_with_include_2d,
        )

    @abstract_plotters.for_figure
    def figure_potential_chi_squared_map(self):
        """Plot the signal-to-noise_map of the imaging data_type.

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        image : data_type.ImagingData
            The imaging data_type, which includes the observed image, noise_map, PSF, signal-to-noise_map, etc.
        include_origin : True
            If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
        """
        self.mat_plot_2d.plot_array(
            array=self.imaging.potential_chi_squared_map,
            visuals_2d=self.visuals_with_include_2d,
        )

    def figure_individuals(
        self,
        plot_image=False,
        plot_noise_map=False,
        plot_psf=False,
        plot_inverse_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_absolute_signal_to_noise_map=False,
        plot_potential_chi_squared_map=False,
    ):
        """Plot each attribute of the imaging data_type as individual figures one by one (e.g. the dataset, noise_map, PSF, \
         Signal-to_noise-map, etc).

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        imaging : data_type.ImagingData
            The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
        include_origin : True
            If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
        """

        if plot_image:
            self.figure_image()
        if plot_noise_map:
            self.figure_noise_map()
        if plot_psf:
            self.figure_psf()
        if plot_inverse_noise_map:
            self.figure_inverse_noise_map()
        if plot_signal_to_noise_map:
            self.figure_signal_to_noise_map()
        if plot_absolute_signal_to_noise_map:
            self.figure_absolute_signal_to_noise_map()
        if plot_potential_chi_squared_map:
            self.figure_potential_chi_squared_map()

    @abstract_plotters.for_subplot
    def subplot_imaging(self):
        """Plot the imaging data_type as a sub-mat_plot_2d of all its quantites (e.g. the dataset, noise_map, PSF, Signal-to_noise-map, \
         etc).

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        imaging : data_type.ImagingData
            The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
        include_origin : True
            If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
        image_plane_pix_grid : np.ndarray or data_type.array.grid_stacks.PixGrid
            If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
            over the immage.
        ignore_config : bool
            If `False`, the config file general.ini is used to determine whether the subpot is plotted. If `True`, the \
            config file is ignored.
        """

        number_subplots = 6

        self.open_subplot_figure(number_subplots=number_subplots)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=1)
        self.figure_image()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=2)
        self.figure_noise_map()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=3)
        self.figure_psf()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=4)
        self.figure_signal_to_noise_map()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=5)
        self.figure_inverse_noise_map()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=6)
        self.figure_potential_chi_squared_map()

        self.mat_plot_2d.output.subplot_to_figure()
        self.mat_plot_2d.figure.close()


class ImagingPlotter(AbstractImagingPlotter):
    def __init__(
        self,
        imaging: im.Imaging,
        mat_plot_2d: mat_plot.MatPlot2D = mat_plot.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):

        super().__init__(
            imaging=imaging,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )
