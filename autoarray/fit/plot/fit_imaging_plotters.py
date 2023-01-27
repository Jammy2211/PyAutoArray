from typing import Callable

from autoarray.plot.abstract_plotters import Plotter
from autoarray.plot.visuals.two_d import Visuals2D
from autoarray.plot.include.two_d import Include2D
from autoarray.plot.mat_plot.two_d import MatPlot2D
from autoarray.plot.auto_labels import AutoLabels
from autoarray.fit.fit_imaging import FitImaging


class FitImagingPlotterMeta(Plotter):
    def __init__(
        self,
        fit,
        get_visuals_2d: Callable,
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
        residuals_symmetric_cmap: bool = True,
    ):
        """
        Plots the attributes of `FitImaging` objects using the matplotlib method `imshow()` and many other matplotlib
        functions which customize the plot's appearance.

        The `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot2d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals2D` object. Attributes may be extracted from
        the `FitImaging` and plotted via the visuals object, if the corresponding entry is `True` in the `Include2D`
        object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        fit
            The fit to an imaging dataset the plotter plots.
        get_visuals_2d
            A function which extracts from the `FitImaging` the 2D visuals which are plotted on figures.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make the plot.
        visuals_2d
            Contains visuals that can be overlaid on the plot.
        include_2d
            Specifies which attributes of the `Array2D` are extracted and plotted as visuals.
        residuals_symmetric_cmap
            If true, the `residual_map` and `normalized_residual_map` are plotted with a symmetric color map such
            that `abs(vmin) = abs(vmax)`.
        """
        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

        self.fit = fit
        self.get_visuals_2d = get_visuals_2d
        self.residuals_symmetric_cmap = residuals_symmetric_cmap

    def figures_2d(
        self,
        image: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_image: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
        suffix: str = "",
    ):
        """
        Plots the individual attributes of the plotter's `FitImaging` object in 2D.

        The API is such that every plottable attribute of the `FitImaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether to make a 2D plot (via `imshow`) of the image data.
        noise_map
            Whether to make a 2D plot (via `imshow`) of the noise map.
        signal_to_noise_map
            Whether to make a 2D plot (via `imshow`) of the signal-to-noise map.
        model_image
            Whether to make a 2D plot (via `imshow`) of the model image.
        residual_map
            Whether to make a 2D plot (via `imshow`) of the residual map.
        normalized_residual_map
            Whether to make a 2D plot (via `imshow`) of the normalized residual map.
        chi_squared_map
            Whether to make a 2D plot (via `imshow`) of the chi-squared map.
        """

        if image:

            self.mat_plot_2d.plot_array(
                array=self.fit.data,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(title="Image", filename=f"image_2d{suffix}"),
            )

        if noise_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.noise_map,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(
                    title="Noise-Map", filename=f"noise_map{suffix}"
                ),
            )

        if signal_to_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.signal_to_noise_map,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(
                    title="Signal-To-Noise Map", filename=f"signal_to_noise_map{suffix}"
                ),
            )

        if model_image:

            self.mat_plot_2d.plot_array(
                array=self.fit.model_data,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(
                    title="Model Image", filename=f"model_image{suffix}"
                ),
            )

        cmap_original = self.mat_plot_2d.cmap

        if self.residuals_symmetric_cmap:

            self.mat_plot_2d.cmap = self.mat_plot_2d.cmap.symmetric

        if residual_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.residual_map,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(
                    title="Residual Map", filename=f"residual_map{suffix}"
                ),
            )

        if normalized_residual_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.normalized_residual_map,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(
                    title="Normalized Residual Map",
                    filename=f"normalized_residual_map{suffix}",
                ),
            )

        self.mat_plot_2d.cmap = cmap_original

        if chi_squared_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.chi_squared_map,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(
                    title="Chi-Squared Map", filename=f"chi_squared_map{suffix}"
                ),
            )

    def subplot(
        self,
        image: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_image: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
        auto_filename: str = "subplot_fit_imaging",
    ):
        """
        Plots the individual attributes of the plotter's `FitImaging` object in 2D on a subplot.

        The API is such that every plottable attribute of the `FitImaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is included on the subplot.

        Parameters
        ----------
        image
            Whether or not to include a 2D plot (via `imshow`) of the image data.
        noise_map
            Whether or not to include a 2D plot (via `imshow`) of the noise map.
        psf
            Whether or not to include a 2D plot (via `imshow`) of the psf.
        signal_to_noise_map
            Whether or not to include a 2D plot (via `imshow`) of the signal-to-noise map.
        model_image
            Whether or not to include a 2D plot (via `imshow`) of the model image.
        residual_map
            Whether or not to include a 2D plot (via `imshow`) of the residual map.
        normalized_residual_map
            Whether or not to include a 2D plot (via `imshow`) of the normalized residual map.
        chi_squared_map
            Whether or not to include a 2D plot (via `imshow`) of the chi-squared map.
        auto_filename
            The default filename of the output subplot if written to hard-disk.
        """
        self._subplot_custom_plot(
            image=image,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            model_image=model_image,
            residual_map=residual_map,
            normalized_residual_map=normalized_residual_map,
            chi_squared_map=chi_squared_map,
            auto_labels=AutoLabels(filename=auto_filename),
        )

    def subplot_fit_imaging(self):
        """
        Standard subplot of the attributes of the plotter's `FitImaging` object.
        """
        return self.subplot(
            image=True,
            signal_to_noise_map=True,
            model_image=True,
            residual_map=True,
            normalized_residual_map=True,
            chi_squared_map=True,
        )


class FitImagingPlotter(Plotter):
    def __init__(
        self,
        fit: FitImaging,
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):
        """
        Plots the attributes of `FitImaging` objects using the matplotlib method `imshow()` and many other matplotlib
        functions which customize the plot's appearance.

        The `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot2d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals2D` object. Attributes may be extracted from
        the `FitImaging` and plotted via the visuals object, if the corresponding entry is `True` in the `Include2D`
        object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        fit
            The fit to an imaging dataset the plotter plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make the plot.
        visuals_2d
            Contains visuals that can be overlaid on the plot.
        include_2d
            Specifies which attributes of the `Array2D` are extracted and plotted as visuals.
        """
        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

        self.fit = fit

        self._fit_imaging_meta_plotter = FitImagingPlotterMeta(
            fit=self.fit,
            get_visuals_2d=self.get_visuals_2d,
            mat_plot_2d=self.mat_plot_2d,
            include_2d=self.include_2d,
            visuals_2d=self.visuals_2d,
        )

        self.figures_2d = self._fit_imaging_meta_plotter.figures_2d
        self.subplot = self._fit_imaging_meta_plotter.subplot
        self.subplot_fit_imaging = self._fit_imaging_meta_plotter.subplot_fit_imaging

    def get_visuals_2d(self) -> Visuals2D:
        return self.get_2d.via_fit_imaging_from(fit=self.fit)
