from autoarray.plot.mat_wrap.visuals import Visuals2D
from autoarray.plot.mat_wrap.include import Include2D
from autoarray.plot.mat_wrap.mat_plot import MatPlot2D
from autoarray.plot.mat_wrap.mat_plot import AutoLabels
from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.dataset.imaging import Imaging
from autoarray.structures.grids.two_d.grid_2d_irregular import Grid2DIrregular


class AbstractImagingPlotter(AbstractPlotter):
    def __init__(
        self,
        imaging: Imaging,
        mat_plot_2d: MatPlot2D,
        visuals_2d: Visuals2D,
        include_2d: Include2D,
    ):

        self.imaging = imaging

        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

    @property
    def visuals_with_include_2d(self) -> Visuals2D:

        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin", Grid2DIrregular(grid=[self.imaging.image.origin])
            ),
            mask=self.extract_2d("mask", self.imaging.image.mask),
            border=self.extract_2d(
                "border", self.imaging.image.mask.border_grid_sub_1.binned
            ),
        )

    def figures_2d(
        self,
        image: bool = False,
        noise_map: bool = False,
        psf: bool = False,
        inverse_noise_map: bool = False,
        signal_to_noise_map: bool = False,
        absolute_signal_to_noise_map: bool = False,
        potential_chi_squared_map: bool = False,
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

            self.mat_plot_2d.plot_array(
                array=self.imaging.image,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=AutoLabels(title="Image", filename="image_2d"),
            )

        if noise_map:

            self.mat_plot_2d.plot_array(
                array=self.imaging.noise_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=AutoLabels("Noise-Map", filename="noise_map"),
            )

        if psf:

            self.mat_plot_2d.plot_array(
                array=self.imaging.psf,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=AutoLabels(title="Point Spread Function", filename="psf"),
            )

        if inverse_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.imaging.inverse_noise_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=AutoLabels(
                    title="Inverse Noise-Map", filename="inverse_noise_map"
                ),
            )

        if signal_to_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.imaging.signal_to_noise_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=AutoLabels(
                    title="Signal-To-Noise Map", filename="signal_to_noise_map"
                ),
            )

        if absolute_signal_to_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.imaging.absolute_signal_to_noise_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=AutoLabels(
                    title="Absolute Signal-To-Noise Map",
                    filename="absolute_signal_to_noise_map",
                ),
            )

        if potential_chi_squared_map:

            self.mat_plot_2d.plot_array(
                array=self.imaging.potential_chi_squared_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=AutoLabels(
                    title="Potential Chi-Squared Map",
                    filename="potential_chi_squared_map",
                ),
            )

    def subplot(
        self,
        image: bool = False,
        noise_map: bool = False,
        psf: bool = False,
        signal_to_noise_map: bool = False,
        inverse_noise_map: bool = False,
        absolute_signal_to_noise_map: bool = False,
        potential_chi_squared_map: bool = False,
        auto_filename: str = "subplot_imaging",
    ):

        self._subplot_custom_plot(
            image=image,
            noise_map=noise_map,
            psf=psf,
            signal_to_noise_map=signal_to_noise_map,
            inverse_noise_map=inverse_noise_map,
            absolute_signal_to_noise_map=absolute_signal_to_noise_map,
            potential_chi_squared_map=potential_chi_squared_map,
            auto_labels=AutoLabels(filename=auto_filename),
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
        imaging: Imaging,
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):

        super().__init__(
            imaging=imaging,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )
