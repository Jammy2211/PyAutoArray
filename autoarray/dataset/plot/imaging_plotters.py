from autoarray.plot.mat_wrap.visuals import Visuals2D
from autoarray.plot.mat_wrap.include import Include2D
from autoarray.plot.mat_wrap.mat_plot import MatPlot2D
from autoarray.plot.mat_wrap.mat_plot import AutoLabels
from autoarray.plot.abstract_plotters import Plotter
from autoarray.dataset.imaging import Imaging


class ImagingPlotter(Plotter):
    def __init__(
        self,
        imaging: Imaging,
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):
        """
        Plots the attributes of `Imaging` objects using the matplotlib method `imshow()` and many other matplotlib
        functions which customize the plot's appearance.

        The `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot2d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals2D` object. Attributes may be extracted from
        the `Array2D` and plotted via the visuals object, if the corresponding entry is `True` in the `Include2D`
        object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        imaging
            The imaging dataset the plotter plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make the plot.
        visuals_2d
            Contains visuals that can be overlaid on the plot.
        include_2d
            Specifies which attributes of the `Array2D` are extracted and plotted as visuals.
        """
        self.imaging = imaging

        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

    def get_visuals_2d(self):
        return self.get_2d.via_mask_from(mask=self.imaging.mask)

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
        """
        Plots the individual attributes of the plotter's `Imaging` object in 2D.
        
        The API is such that every plottable attribute of the `Imaging` object is an input parameter of type bool of 
        the function, which if switched to `True` means that it is plotted.
        
        Parameters
        ----------
        image
            Whether or not to make a 2D plot (via `imshow`) of the image data.
        noise_map
            Whether or not to make a 2D plot (via `imshow`) noise map.
        psf
            Whether or not to make a 2D plot (via `imshow`) psf.          
        inverse_noise_map
            Whether or not to make a 2D plot (via `imshow`) inverse noise map.
        signal_to_noise_map
            Whether or not to make a 2D plot (via `imshow`) signal-to-noise map.
        absolute_signal_to_noise_map
            Whether or not to make a 2D plot (via `imshow`) absolute signal to noise map.  
        potential_chi_squared_map
            Whether or not to make a 2D plot (via `imshow`) potential chi squared map.
        """

        if image:

            self.mat_plot_2d.plot_array(
                array=self.imaging.image,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(title="Image", filename="image_2d"),
            )

        if noise_map:

            self.mat_plot_2d.plot_array(
                array=self.imaging.noise_map,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels("Noise-Map", filename="noise_map"),
            )

        if psf:

            self.mat_plot_2d.plot_array(
                array=self.imaging.psf,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(title="Point Spread Function", filename="psf"),
            )

        if inverse_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.imaging.inverse_noise_map,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(
                    title="Inverse Noise-Map", filename="inverse_noise_map"
                ),
            )

        if signal_to_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.imaging.signal_to_noise_map,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(
                    title="Signal-To-Noise Map", filename="signal_to_noise_map"
                ),
            )

        if absolute_signal_to_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.imaging.absolute_signal_to_noise_map,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(
                    title="Absolute Signal-To-Noise Map",
                    filename="absolute_signal_to_noise_map",
                ),
            )

        if potential_chi_squared_map:

            self.mat_plot_2d.plot_array(
                array=self.imaging.potential_chi_squared_map,
                visuals_2d=self.get_visuals_2d(),
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
