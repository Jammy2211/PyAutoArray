from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot
from autoarray.plot.plotters import abstract_plotters


class ImagingPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        mat_plot_2d: mat_plot.MatPlot2D = mat_plot.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):

        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

    def subplot_imaging(self, imaging):
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

        mat_plot_2d = self.mat_plot_2d.plotter_for_subplot_from(
            func=self.subplot_imaging
        )

        number_subplots = 6

        mat_plot_2d.open_subplot_figure(number_subplots=number_subplots)

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

        self.image(imaging=imaging)

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=2)

        self.noise_map(imaging=imaging)

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=3)

        self.psf(imaging=imaging)

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=4)

        self.signal_to_noise_map(imaging=imaging)

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=5)

        self.inverse_noise_map(imaging=imaging)

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=6)

        self.potential_chi_squared_map(imaging=imaging)

        mat_plot_2d.output.subplot_to_figure()

        mat_plot_2d.figure.close()

    def individual(
        self,
        imaging,
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
            self.image(imaging=imaging)
        if plot_noise_map:
            self.noise_map(imaging=imaging)
        if plot_psf:
            self.psf(imaging=imaging)
        if plot_inverse_noise_map:
            self.inverse_noise_map(imaging=imaging)
        if plot_signal_to_noise_map:
            self.signal_to_noise_map(imaging=imaging)
        if plot_absolute_signal_to_noise_map:
            self.absolute_signal_to_noise_map(imaging=imaging)
        if plot_potential_chi_squared_map:
            self.potential_chi_squared_map(imaging=imaging)

    @abstract_plotters.set_labels
    def image(self, imaging):
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
            array=imaging.image,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_from_dataset(dataset=imaging),
        )

    @abstract_plotters.set_labels
    def noise_map(self, imaging):
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
            array=imaging.noise_map,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_from_dataset(dataset=imaging),
        )

    @abstract_plotters.set_labels
    def psf(self, imaging):
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
            array=imaging.psf,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_from_dataset(dataset=imaging),
        )

    @abstract_plotters.set_labels
    def inverse_noise_map(self, imaging):
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
            array=imaging.inverse_noise_map,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_from_dataset(dataset=imaging),
        )

    @abstract_plotters.set_labels
    def signal_to_noise_map(self, imaging):
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
            array=imaging.signal_to_noise_map,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_from_dataset(dataset=imaging),
        )

    @abstract_plotters.set_labels
    def absolute_signal_to_noise_map(self, imaging):
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
            array=imaging.absolute_signal_to_noise_map,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_from_dataset(dataset=imaging),
        )

    @abstract_plotters.set_labels
    def potential_chi_squared_map(self, imaging):
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
            array=imaging.potential_chi_squared_map,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_from_dataset(dataset=imaging),
        )
