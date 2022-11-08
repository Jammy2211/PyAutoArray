from autoarray import exc


class AbstractWTilde:
    def __init__(self, curvature_preload, noise_map_value):
        """
        Packages together all derived data quantities necessary to fit `data (e.g. `Imaging`, Interferometer`) using
        an ` Inversion` via the w_tilde formalism.

        The w_tilde formalism performs linear algebra formalism in a way that speeds up the construction of the
        simultaneous linear equations by bypassing the construction of a `mapping_matrix` and precomputing
        operations like blurring or a Fourier transform.

        Parameters
        ----------
        curvature_preload
            A matrix which uses the imaging's noise-map and PSF to preload as much of the computation of the
            curvature matrix as possible.
        noise_map_value
            The first value of the noise-map used to construct the curvature preload, which is used as a sanity
            check when performing the inversion to ensure the preload corresponds to the data being fitted.
        """
        self.curvature_preload = curvature_preload
        self.noise_map_value = noise_map_value

    def check_noise_map(self, noise_map):

        if noise_map[0] != self.noise_map_value:
            raise exc.InversionException(
                "The preloaded values of WTildeImaging are not consistent with the noise-map passed to them, thus "
                "they cannot be used for the inversion."
                ""
                f"The value of the noise map is {noise_map[0]} whereas in WTildeImaging it is {self.noise_map_value}"
                ""
                "Update WTildeImaging or do not use the w_tilde formalism to perform the Inversion."
            )