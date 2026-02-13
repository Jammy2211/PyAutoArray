from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray.inversion.regularization.matern_kernel import matern_cov_matrix_from


class AdaptiveBrightnessMatern(AbstractRegularization):
    def __init__(
            self,
            coefficient: float = 1.0,
            scale: float = 1.0,
            nu: float = 0.5,
            rho: float = 1.0,
    ):
        super().__init__(coefficient=coefficient, scale=scale, rho=rho)
        self.nu = nu

    def covariance_kernel_weights_from(self, linear_obj: LinearObj) -> np.ndarray:
        pixel_signals = linear_obj.pixel_signals_from(signal_scale=1.0)
        return np.exp(-self.rho * (1 - pixel_signals / pixel_signals.max()))

    def regularization_matrix_from(self, linear_obj: LinearObj) -> np.ndarray:
        kernel_weights = self.covariance_kernel_weights_from(linear_obj=linear_obj)

        covariance_matrix = matern_cov_matrix_from(
            scale=self.scale,
            pixel_points=linear_obj.source_plane_mesh_grid,
            nu=self.nu,
            weights=kernel_weights,
        )

        return self.coefficient * np.linalg.inv(covariance_matrix)

    def regularization_weights_from(self, linear_obj: LinearObj) -> np.ndarray:
        """
        Returns the regularization weights of this regularization scheme.

        The regularization weights define the level of regularization applied to each parameter in the linear object
        (e.g. the ``pixels`` in a ``Mapper``).

        For standard regularization (e.g. ``Constant``) are weights are equal, however for adaptive schemes
        (e.g. ``AdaptiveBrightness``) they vary to adapt to the data being reconstructed.

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses these weights when performing regularization.

        Returns
        -------
        The regularization weights.
        """
        return 1.0/self.covariance_kernel_weights_from(linear_obj=linear_obj) #meaningless, but consistent with other regularization schemes