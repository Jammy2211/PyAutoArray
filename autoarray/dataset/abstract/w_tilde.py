from autoarray import exc


class AbstractWTilde:
    def __init__(self, curvature_preload):
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
        """
        self.curvature_preload = curvature_preload
