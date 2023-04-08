from autoarray.plot.abstract_plotters import Plotter
from autoarray.plot.visuals.one_d import Visuals1D
from autoarray.plot.visuals.two_d import Visuals2D
from autoarray.plot.include.one_d import Include1D
from autoarray.plot.include.two_d import Include2D
from autoarray.plot.mat_plot.one_d import MatPlot1D
from autoarray.plot.mat_plot.two_d import MatPlot2D
from autoarray.plot.auto_labels import AutoLabels
from autoarray.dataset.interferometer.interferometer import Interferometer
from autoarray.structures.grids.irregular_2d import Grid2DIrregular


class InterferometerPlotter(Plotter):
    def __init__(
        self,
        interferometer: Interferometer,
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        include_1d: Include1D = Include1D(),
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):
        """
        Plots the attributes of `Interferometer` objects using the matplotlib methods `plot()`, `scatter()` and
        `imshow()` and other matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2d` to
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be
        extracted from the `LightProfile` and plotted via the visuals object, if the corresponding entry is `True` in
        the `Include1D` or `Include2D` object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        interferometer
            The interferometer dataset the plotter plots.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        include_1d
            Specifies which attributes of the `Interferometer` are extracted and plotted as visuals for 1D plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        include_2d
            Specifies which attributes of the `Interferometer` are extracted and plotted as visuals for 2D plots.
        """
        self.interferometer = interferometer

        super().__init__(
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

    def get_visuals_2d_real_space(self):
        return self.get_2d.via_mask_from(mask=self.interferometer.real_space_mask)

    def figures_2d(
        self,
        data: bool = False,
        noise_map: bool = False,
        u_wavelengths: bool = False,
        v_wavelengths: bool = False,
        uv_wavelengths: bool = False,
        amplitudes_vs_uv_distances: bool = False,
        phases_vs_uv_distances: bool = False,
        dirty_image: bool = False,
        dirty_noise_map: bool = False,
        dirty_signal_to_noise_map: bool = False,
        dirty_inverse_noise_map: bool = False,
    ):
        """
        Plots the individual attributes of the plotter's `Interferometer` object in 1D and 2D.

        The API is such that every plottable attribute of the `Interferometer` object is an input parameter of type
        bool of the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        data
            Whether to make a 2D plot (via `scatter`) of the visibility data.
        noise_map
            Whether to make a 2D plot (via `scatter`) of the noise-map.
        u_wavelengths
            Whether to make a 1D plot (via `plot`) of the u-wavelengths.
        v_wavelengths
            Whether to make a 1D plot (via `plot`) of the v-wavelengths.
        amplitudes_vs_uv_distances
            Whether to make a 1D plot (via `plot`) of the amplitudes versis the uv distances.
        phases_vs_uv_distances
            Whether to make a 1D plot (via `plot`) of the phases versis the uv distances.
        dirty_image
            Whether to make a 2D plot (via `imshow`) of the dirty image.
        dirty_noise_map
            Whether to make a 2D plot (via `imshow`) of the dirty noise map.
        dirty_signal_to_noise_map
            Whether to make a 2D plot (via `imshow`) of the dirty signal-to-noise map.
        dirty_inverse_noise_map
            Whether to make a 2D plot (via `imshow`) of the dirty inverse noise map.
        """

        if data:

            self.mat_plot_2d.plot_grid(
                grid=self.interferometer.visibilities.in_grid,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(title="Visibilities", filename="visibilities"),
            )

        if noise_map:

            self.mat_plot_2d.plot_grid(
                grid=self.interferometer.visibilities.in_grid,
                visuals_2d=self.visuals_2d,
                color_array=self.interferometer.noise_map.real,
                auto_labels=AutoLabels(title="Noise-Map", filename="noise_map"),
            )

        if u_wavelengths:

            self.mat_plot_1d.plot_yx(
                y=self.interferometer.uv_wavelengths[:, 0],
                x=None,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="U-Wavelengths",
                    filename="u_wavelengths",
                    ylabel="Wavelengths",
                ),
                plot_axis_type_override="linear",
            )

        if v_wavelengths:

            self.mat_plot_1d.plot_yx(
                y=self.interferometer.uv_wavelengths[:, 1],
                x=None,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="V-Wavelengths",
                    filename="v_wavelengths",
                    ylabel="Wavelengths",
                ),
                plot_axis_type_override="linear",
            )

        if uv_wavelengths:

            self.mat_plot_2d.plot_grid(
                grid=Grid2DIrregular.from_yx_1d(
                    y=self.interferometer.uv_wavelengths[:, 1] / 10**3.0,
                    x=self.interferometer.uv_wavelengths[:, 0] / 10**3.0,
                ),
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(
                    title="UV-Wavelengths", filename="uv_wavelengths"
                ),
            )

        if amplitudes_vs_uv_distances:

            self.mat_plot_1d.plot_yx(
                y=self.interferometer.amplitudes,
                x=self.interferometer.uv_distances / 10**3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="Amplitudes vs UV-distances",
                    filename="amplitudes_vs_uv_distances",
                    ylabel="amplitude (Jy)",
                    xlabel="U-Wavelengths ($\lambda$)",
                ),
                plot_axis_type_override="scatter",
            )

        if phases_vs_uv_distances:

            self.mat_plot_1d.plot_yx(
                y=self.interferometer.phases,
                x=self.interferometer.uv_distances / 10**3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="Phases vs UV-distances",
                    filename="phases_vs_uv_distances",
                    ylabel="phase (deg)",
                    xlabel=r"UV$_{distance}$ (k$\lambda$)",
                ),
                plot_axis_type_override="scatter",
            )

        if dirty_image:

            self.mat_plot_2d.plot_array(
                array=self.interferometer.dirty_image,
                visuals_2d=self.get_visuals_2d_real_space(),
                auto_labels=AutoLabels(title="Dirty Image", filename="dirty_image_2d"),
            )

        if dirty_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.interferometer.dirty_noise_map,
                visuals_2d=self.get_visuals_2d_real_space(),
                auto_labels=AutoLabels(
                    title="Dirty Noise Map", filename="dirty_noise_map_2d"
                ),
            )

        if dirty_signal_to_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.interferometer.dirty_signal_to_noise_map,
                visuals_2d=self.get_visuals_2d_real_space(),
                auto_labels=AutoLabels(
                    title="Dirty Signal-To-Noise Map",
                    filename="dirty_signal_to_noise_map_2d",
                ),
            )

        if dirty_inverse_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.interferometer.dirty_inverse_noise_map,
                visuals_2d=self.get_visuals_2d_real_space(),
                auto_labels=AutoLabels(
                    title="Dirty Inverse Noise Map",
                    filename="dirty_inverse_noise_map_2d",
                ),
            )

    def subplot(
        self,
        data: bool = False,
        noise_map: bool = False,
        u_wavelengths: bool = False,
        v_wavelengths: bool = False,
        uv_wavelengths: bool = False,
        amplitudes_vs_uv_distances: bool = False,
        phases_vs_uv_distances: bool = False,
        dirty_image: bool = False,
        dirty_noise_map: bool = False,
        dirty_signal_to_noise_map: bool = False,
        dirty_inverse_noise_map: bool = False,
        auto_filename: str = "subplot_interferometer",
    ):
        """
        Plots the individual attributes of the plotter's `Interferometer` object in 1D and 2D on a subplot.

        The API is such that every plottable attribute of the `Interferometer` object is an input parameter of type
        bool of the function, which if switched to `True` means that it is included on the subplot.

        Parameters
        ----------
        data
            Whether or not to include a 2D plot (via `scatter`) of the visibility data.
        noise_map
            Whether or not to include a 2D plot (via `scatter`) of the noise-map.
        u_wavelengths
            Whether or not to include a 1D plot (via `plot`) of the u-wavelengths.
        v_wavelengths
            Whether or not to include a 1D plot (via `plot`) of the v-wavelengths.
        amplitudes_vs_uv_distances
            Whether or not to include a 1D plot (via `plot`) of the amplitudes versis the uv distances.
        phases_vs_uv_distances
            Whether or not to include a 1D plot (via `plot`) of the phases versis the uv distances.
        dirty_image
            Whether or not to include a 2D plot (via `imshow`) of the dirty image.
        dirty_noise_map
            Whether or not to include a 2D plot (via `imshow`) of the dirty noise map.
        dirty_signal_to_noise_map
            Whether or not to include a 2D plot (via `imshow`) of the dirty signal-to-noise map.
        dirty_inverse_noise_map
            Whether or not to include a 2D plot (via `imshow`) of the dirty inverse noise map.
        """
        self._subplot_custom_plot(
            data=data,
            noise_map=noise_map,
            u_wavelengths=u_wavelengths,
            v_wavelengths=v_wavelengths,
            uv_wavelengths=uv_wavelengths,
            amplitudes_vs_uv_distances=amplitudes_vs_uv_distances,
            phases_vs_uv_distances=phases_vs_uv_distances,
            dirty_image=dirty_image,
            dirty_noise_map=dirty_noise_map,
            dirty_signal_to_noise_map=dirty_signal_to_noise_map,
            dirty_inverse_noise_map=dirty_inverse_noise_map,
            auto_labels=AutoLabels(filename=auto_filename),
        )

    def subplot_interferometer(self):
        """
        Standard subplot of the attributes of the plotter's `Interferometer` object.
        """
        return self.subplot(
            data=True,
            uv_wavelengths=True,
            amplitudes_vs_uv_distances=True,
            phases_vs_uv_distances=True,
            auto_filename="subplot_interferometer",
        )

    def subplot_dirty_images(self):
        """
        Standard subplot of the dirty attributes of the plotter's `Interferometer` object.
        """
        return self.subplot(
            dirty_image=True,
            dirty_noise_map=True,
            dirty_signal_to_noise_map=True,
            dirty_inverse_noise_map=True,
            auto_filename="subplot_dirty_images",
        )
