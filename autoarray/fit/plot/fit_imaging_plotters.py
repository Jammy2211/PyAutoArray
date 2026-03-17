import numpy as np
from typing import Callable

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.visuals.two_d import Visuals2D
from autoarray.plot.mat_plot.two_d import MatPlot2D
from autoarray.plot.auto_labels import AutoLabels
from autoarray.plot.plots.array import plot_array
from autoarray.fit.fit_imaging import FitImaging
from autoarray.structures.plot.structure_plotters import (
    _mask_edge_from,
    _grid_from_visuals,
    _lines_from_visuals,
    _positions_from_visuals,
    _output_for_mat_plot,
)


class FitImagingPlotterMeta(AbstractPlotter):
    def __init__(
        self,
        fit,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
        residuals_symmetric_cmap: bool = True,
    ):
        """
        Plots the attributes of `FitImaging` objects using the matplotlib method `imshow()` and many other matplotlib
        functions which customize the plot's appearance.

        The `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot2d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals2D` object. Attributes may be extracted from
        the `FitImaging` and plotted via the visuals object.

        Parameters
        ----------
        fit
            The fit to an imaging dataset the plotter plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make the plot.
        visuals_2d
            Contains visuals that can be overlaid on the plot.
        residuals_symmetric_cmap
            If true, the `residual_map` and `normalized_residual_map` are plotted with a symmetric color map such
            that `abs(vmin) = abs(vmax)`.
        """
        super().__init__(mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d)

        self.fit = fit
        self.residuals_symmetric_cmap = residuals_symmetric_cmap

    def _plot_array(self, array, auto_labels, visuals_2d=None):
        """Helper: plot an Array2D using the new direct-matplotlib function."""
        if array is None:
            return

        v2d = visuals_2d if visuals_2d is not None else self.visuals_2d
        is_sub = self.mat_plot_2d.is_for_subplot
        ax = self.mat_plot_2d.setup_subplot() if is_sub else None

        output_path, filename, fmt = _output_for_mat_plot(
            self.mat_plot_2d,
            is_sub,
            auto_labels.filename if auto_labels else "array",
        )

        try:
            arr = array.native.array
            extent = array.geometry.extent
        except AttributeError:
            arr = np.asarray(array)
            extent = None

        plot_array(
            array=arr,
            ax=ax,
            extent=extent,
            mask=_mask_edge_from(array if hasattr(array, "mask") else None, v2d),
            grid=_grid_from_visuals(v2d),
            positions=_positions_from_visuals(v2d),
            lines=_lines_from_visuals(v2d),
            title=auto_labels.title if auto_labels else "",
            colormap=self.mat_plot_2d.cmap.cmap,
            use_log10=self.mat_plot_2d.use_log10,
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
        )

    def figures_2d(
        self,
        data: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_image: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
        residual_flux_fraction_map: bool = False,
        suffix: str = "",
    ):
        """
        Plots the individual attributes of the plotter's `FitImaging` object in 2D.

        The API is such that every plottable attribute of the `FitImaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        data
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
        residual_flux_fraction_map
            Whether to make a 2D plot (via `imshow`) of the residual flux fraction map.
        """

        if data:
            self._plot_array(
                array=self.fit.data,
                auto_labels=AutoLabels(title="Data", filename=f"data{suffix}"),
            )

        if noise_map:
            self._plot_array(
                array=self.fit.noise_map,
                auto_labels=AutoLabels(
                    title="Noise-Map", filename=f"noise_map{suffix}"
                ),
            )

        if signal_to_noise_map:
            self._plot_array(
                array=self.fit.signal_to_noise_map,
                auto_labels=AutoLabels(
                    title="Signal-To-Noise Map", filename=f"signal_to_noise_map{suffix}"
                ),
            )

        if model_image:
            self._plot_array(
                array=self.fit.model_data,
                auto_labels=AutoLabels(
                    title="Model Image", filename=f"model_image{suffix}"
                ),
            )

        cmap_original = self.mat_plot_2d.cmap
        if self.residuals_symmetric_cmap:
            self.mat_plot_2d.cmap = self.mat_plot_2d.cmap.symmetric_cmap_from()

        if residual_map:
            self._plot_array(
                array=self.fit.residual_map,
                auto_labels=AutoLabels(
                    title="Residual Map", filename=f"residual_map{suffix}"
                ),
            )

        if normalized_residual_map:
            self._plot_array(
                array=self.fit.normalized_residual_map,
                auto_labels=AutoLabels(
                    title="Normalized Residual Map",
                    filename=f"normalized_residual_map{suffix}",
                ),
            )

        self.mat_plot_2d.cmap = cmap_original

        if chi_squared_map:
            self._plot_array(
                array=self.fit.chi_squared_map,
                auto_labels=AutoLabels(
                    title="Chi-Squared Map", filename=f"chi_squared_map{suffix}"
                ),
            )

        if residual_flux_fraction_map:
            self._plot_array(
                array=self.fit.residual_map,
                auto_labels=AutoLabels(
                    title="Residual Flux Fraction Map",
                    filename=f"residual_flux_fraction_map{suffix}",
                ),
            )

    def subplot(
        self,
        data: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_image: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
        residual_flux_fraction_map: bool = False,
        auto_filename: str = "subplot_fit",
    ):
        """
        Plots the individual attributes of the plotter's `FitImaging` object in 2D on a subplot.

        The API is such that every plottable attribute of the `FitImaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is included on the subplot.

        Parameters
        ----------
        data
            Whether to include a 2D plot (via `imshow`) of the image data.
        noise_map
            Whether to include a 2D plot (via `imshow`) of the noise map.
        psf
            Whether to include a 2D plot (via `imshow`) of the psf.
        signal_to_noise_map
            Whether to include a 2D plot (via `imshow`) of the signal-to-noise map.
        model_image
            Whether to include a 2D plot (via `imshow`) of the model image.
        residual_map
            Whether to include a 2D plot (via `imshow`) of the residual map.
        normalized_residual_map
            Whether to include a 2D plot (via `imshow`) of the normalized residual map.
        chi_squared_map
            Whether to include a 2D plot (via `imshow`) of the chi-squared map.
        residual_flux_fraction_map
            Whether to include a 2D plot (via `imshow`) of the residual flux fraction map.
        auto_filename
            The default filename of the output subplot if written to hard-disk.
        """
        self._subplot_custom_plot(
            data=data,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            model_image=model_image,
            residual_map=residual_map,
            normalized_residual_map=normalized_residual_map,
            chi_squared_map=chi_squared_map,
            residual_flux_fraction_map=residual_flux_fraction_map,
            auto_labels=AutoLabels(filename=auto_filename),
        )

    def subplot_fit(self):
        """
        Standard subplot of the attributes of the plotter's `FitImaging` object.
        """
        return self.subplot(
            data=True,
            signal_to_noise_map=True,
            model_image=True,
            residual_map=True,
            normalized_residual_map=True,
            chi_squared_map=True,
        )


class FitImagingPlotter(AbstractPlotter):
    def __init__(
        self,
        fit: FitImaging,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
    ):
        """
        Plots the attributes of `FitImaging` objects using the matplotlib method `imshow()` and many other matplotlib
        functions which customize the plot's appearance.

        The `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot2d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals2D` object. Attributes may be extracted from
        the `FitImaging` and plotted via the visuals object.

        Parameters
        ----------
        fit
            The fit to an imaging dataset the plotter plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make the plot.
        visuals_2d
            Contains visuals that can be overlaid on the plot.
        """
        super().__init__(mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d)

        self.fit = fit

        self._fit_imaging_meta_plotter = FitImagingPlotterMeta(
            fit=self.fit,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
        )

        self.figures_2d = self._fit_imaging_meta_plotter.figures_2d
        self.subplot = self._fit_imaging_meta_plotter.subplot
        self.subplot_fit = self._fit_imaging_meta_plotter.subplot_fit
