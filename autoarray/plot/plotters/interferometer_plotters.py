from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot as mp
from autoarray.dataset import interferometer as inter
from autoarray.structures.grids.two_d import grid_2d_irregular
import numpy as np


class InterferometerPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        interferometer: inter.Interferometer,
        mat_plot_1d: mp.MatPlot1D = mp.MatPlot1D(),
        visuals_1d: vis.Visuals1D = vis.Visuals1D(),
        include_1d: inc.Include1D = inc.Include1D(),
        mat_plot_2d: mp.MatPlot2D = mp.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):

        self.interferometer = interferometer

        super().__init__(
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

    @property
    def visuals_with_include_2d(self):
        return self.visuals_2d + self.visuals_2d.__class__()

    def figures(
        self,
        visibilities=False,
        noise_map=False,
        u_wavelengths=False,
        v_wavelengths=False,
        uv_wavelengths=False,
        amplitudes_vs_uv_distances=False,
        phases_vs_uv_distances=False,
    ):
        """
        Plot each attribute of the interferometer data_type as individual figures one by one (e.g. the dataset, noise_map, PSF, \
         Signal-to_noise-map, etc).

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        interferometer : data_type.UVPlaneData
            The interferometer data_type, which include the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        """

        if visibilities:

            self.mat_plot_2d.plot_grid(
                grid=self.interferometer.visibilities.in_grid,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="Visibilities", filename="visibilities"
                ),
            )

        if noise_map:

            self.mat_plot_2d.plot_grid(
                grid=self.interferometer.visibilities.in_grid,
                visuals_2d=self.visuals_with_include_2d,
                color_array=self.interferometer.noise_map.real,
                auto_labels=mp.AutoLabels(title="Noise-Map", filename="noise_map"),
            )

        if u_wavelengths:

            self.mat_plot_1d.plot_yx(
                y=self.interferometer.uv_wavelengths[:, 0],
                x=None,
                visuals_1d=self.visuals_1d,
                auto_labels=mp.AutoLabels(
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
                auto_labels=mp.AutoLabels(
                    title="V-Wavelengths",
                    filename="v_wavelengths",
                    ylabel="Wavelengths",
                ),
                plot_axis_type_override="linear",
            )

        if uv_wavelengths:

            self.mat_plot_2d.plot_grid(
                grid=grid_2d_irregular.Grid2DIrregular.from_yx_1d(
                    y=self.interferometer.uv_wavelengths[:, 1] / 10 ** 3.0,
                    x=self.interferometer.uv_wavelengths[:, 0] / 10 ** 3.0,
                ),
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="UV-Wavelengths", filename="uv_wavelengths"
                ),
            )

        if amplitudes_vs_uv_distances:

            self.mat_plot_1d.plot_yx(
                y=self.interferometer.amplitudes,
                x=self.interferometer.uv_distances / 10 ** 3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=mp.AutoLabels(
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
                x=self.interferometer.uv_distances / 10 ** 3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=mp.AutoLabels(
                    title="Phases vs UV-distances",
                    filename="phases_vs_uv_distances",
                    ylabel="phase (deg)",
                    xlabel=r"UV$_{distance}$ (k$\lambda$)",
                ),
                plot_axis_type_override="scatter",
            )

    def subplot(
        self,
        visibilities=False,
        noise_map=False,
        u_wavelengths=False,
        v_wavelengths=False,
        uv_wavelengths=False,
        amplitudes_vs_uv_distances=False,
        phases_vs_uv_distances=False,
        auto_filename="subplot_interferometer",
    ):

        self._subplot_custom_plot(
            visibilities=visibilities,
            noise_map=noise_map,
            u_wavelengths=u_wavelengths,
            v_wavelengths=v_wavelengths,
            uv_wavelengths=uv_wavelengths,
            amplitudes_vs_uv_distances=amplitudes_vs_uv_distances,
            phases_vs_uv_distances=phases_vs_uv_distances,
            auto_labels=mp.AutoLabels(filename=auto_filename),
        )

    def subplot_interferometer(self):
        """Plot the interferometer data_type as a sub-mat_plot_2d of all its quantites (e.g. the dataset, noise_map, PSF, Signal-to_noise-map, \
         etc).

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        interferometer : data_type.UVPlaneData
            The interferometer data_type, which include the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        image_plane_pix_grid : np.ndarray or data_type.array.grid_stacks.PixGrid
            If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
            over the immage.
        ignore_config : bool
            If `False`, the config file general.ini is used to determine whether the subpot is plotted. If `True`, the \
            config file is ignored.
        """
        return self.subplot(
            visibilities=True,
            uv_wavelengths=True,
            amplitudes_vs_uv_distances=True,
            phases_vs_uv_distances=True,
        )
