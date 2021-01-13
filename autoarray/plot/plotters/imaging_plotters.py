from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot as mp
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

    def figures(
        self,
        image=False,
        noise_map=False,
        psf=False,
        inverse_noise_map=False,
        signal_to_noise_map=False,
        absolute_signal_to_noise_map=False,
        potential_chi_squared_map=False,
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

        if image:
            self.plot_array(
                array=self.imaging.image,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(title="Image", filename="image"),
            )
        if noise_map:
            self.plot_array(
                array=self.imaging.noise_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels("Noise-Map", filename="noise_map"),
            )
        if psf:
            self.plot_array(
                array=self.imaging.psf,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="Point Spread Function", filename="psf"
                ),
            )
        if inverse_noise_map:
            self.plot_array(
                array=self.imaging.inverse_noise_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="Inverse Noise-Map", filename="inverse_noise_map"
                ),
            )
        if signal_to_noise_map:
            self.plot_array(
                array=self.imaging.signal_to_noise_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="Signal-To-Noise Map", filename="signal_to_noise_map"
                ),
            )
        if absolute_signal_to_noise_map:
            self.plot_array(
                array=self.imaging.absolute_signal_to_noise_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="Absolute Signal-To-Noise Map",
                    filename="absolute_signal_to_noise_map",
                ),
            )
        if potential_chi_squared_map:
            self.plot_array(
                array=self.imaging.potential_chi_squared_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="Potential Chi-Squared Map",
                    filename="potential_chi_squared_map",
                ),
            )

    def subplot(
        self,
        image=False,
        noise_map=False,
        psf=False,
        signal_to_noise_map=False,
        inverse_noise_map=False,
        absolute_signal_to_noise_map=False,
        potential_chi_squared_map=False,
        auto_filename="subplot_imaging",
    ):

        self._subplot_custom_plot(
            image=image,
            noise_map=noise_map,
            psf=psf,
            signal_to_noise_map=signal_to_noise_map,
            inverse_noise_map=inverse_noise_map,
            absolute_signal_to_noise_map=absolute_signal_to_noise_map,
            potential_chi_squared_map=potential_chi_squared_map,
            auto_labels=mp.AutoLabels(filename=auto_filename),
        )

    def subplot_imaging(self):
        self.subplot(
            image=True,
            noise_map=True,
            psf=True,
            signal_to_noise_map=True,
            inverse_noise_map=True,
            potential_chi_squared_map=True,
        )


class ImagingPlotter(AbstractImagingPlotter):
    def __init__(
        self,
        imaging: im.Imaging,
        mat_plot_2d: mp.MatPlot2D = mp.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):

        super().__init__(
            imaging=imaging,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )
