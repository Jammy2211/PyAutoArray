from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot as mp
from autoarray.fit import fit as f
from autoarray.structures import grids


class AbstractFitImagingPlotter(abstract_plotters.AbstractPlotter):
    def __init__(self, fit, mat_plot_2d, visuals_2d, include_2d):
        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

        self.fit = fit

    @property
    def visuals_with_include_2d(self):

        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin", grids.GridIrregular(grid=[self.fit.mask.origin])
            ),
            mask=self.extract_2d("mask", self.fit.mask),
            border=self.extract_2d(
                "border", self.fit.mask.geometry.border_grid_sub_1.in_1d_binned
            ),
        )

    def figures(
        self,
        image=False,
        noise_map=False,
        signal_to_noise_map=False,
        model_image=False,
        residual_map=False,
        normalized_residual_map=False,
        chi_squared_map=False,
    ):
        """Plot the model datas_ of an analysis, using the *Fitter* class object.

        The visualization and output type can be fully customized.

        Parameters
        -----------
        fit : autolens.lens.fitting.Fitter
            Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
        output_path : str
            The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
        output_format : str
            How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
            in the python interpreter window.
        """

        if image:

            self.mat_plot_2d.plot_array(
                array=self.fit.data,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(title="Image", filename="image"),
            )

        if noise_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.noise_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(title="Noise-Map", filename="noise_map"),
            )

        if signal_to_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.signal_to_noise_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="Signal-To-Noise Map", filename="signal_to_noise_map"
                ),
            )

        if model_image:

            self.mat_plot_2d.plot_array(
                array=self.fit.model_data,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(title="Model Image", filename="model_image"),
            )

        if residual_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.residual_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="Residual Map", filename="residual_map"
                ),
            )

        if normalized_residual_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.normalized_residual_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="Normalized Residual Map", filename="normalized_residual_map"
                ),
            )

        if chi_squared_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.chi_squared_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="Chi-Squared Map", filename="chi_squared_map"
                ),
            )

    def subplot(
        self,
        image=False,
        noise_map=False,
        signal_to_noise_map=False,
        model_image=False,
        residual_map=False,
        normalized_residual_map=False,
        chi_squared_map=False,
        auto_filename="subplot_fit_imaging",
    ):

        self._subplot_custom_plot(
            image=image,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            model_image=model_image,
            residual_map=residual_map,
            normalized_residual_map=normalized_residual_map,
            chi_squared_map=chi_squared_map,
            auto_labels=mp.AutoLabels(filename=auto_filename),
        )

    def subplot_fit_imaging(self):
        return self.subplot(
            image=True,
            signal_to_noise_map=True,
            model_image=True,
            residual_map=True,
            normalized_residual_map=True,
            chi_squared_map=True,
        )


class FitImagingPlotter(AbstractFitImagingPlotter):
    def __init__(
        self,
        fit: f.FitImaging,
        mat_plot_2d: mp.MatPlot2D = mp.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):

        super().__init__(
            fit=fit,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )
