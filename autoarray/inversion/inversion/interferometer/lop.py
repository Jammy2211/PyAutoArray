from scipy import sparse

import numpy as np
from typing import Dict

from autoconf import cached_property

from autoarray.inversion.inversion.interferometer.abstract import (
    AbstractInversionInterferometer,
)
from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.structures.visibilities import Visibilities

from autoarray.numba_util import profile_func


class InversionInterferometerMappingPyLops(AbstractInversionInterferometer):
    """
    Constructs linear equations (via vectors and matrices) which allow for sets of simultaneous linear equations
    to be solved (see `inversion.inversion.abstract.AbstractInversion` for a full description).

    A linear object describes the mappings between values in observed `data` and the linear object's model via its
    `mapping_matrix`. This class constructs linear equations for `Interferometer` objects, where the data is an
    an array of visibilities and the mappings include a non-uniform fast Fourier transform operation described by
    the interferometer dataset's transformer.

    This class uses the mapping formalism, which constructs the simultaneous linear equations using the
    `mapping_matrix` of every linear object. This is performed using the library PyLops, which uses linear
    operators to avoid these matrices being created explicitly in memory, making the calculation more efficient.
    """

    @cached_property
    @profile_func
    def reconstruction(self):
        """
        Solve the linear system [F + reg_coeff*H] S = D -> S = [F + reg_coeff*H]^-1 D given by equation (12)
        of https://arxiv.org/pdf/astro-ph/0302587.pdf

        S is the vector of reconstructed inversion values.
        """

        import pylops

        Aop = pylops.MatrixMult(
            sparse.bsr_matrix(self.linear_obj_list[0].mapping_matrix)
        )

        Fop = self.transformer

        Op = Fop * Aop

        MOp = pylops.MatrixMult(sparse.bsr_matrix(self.preconditioner_matrix_inverse))

        return pylops.NormalEquationsInversion(
            Op=Op,
            Regs=None,
            epsNRs=[1.0],
            data=self.data.ordered_1d,
            Weight=pylops.Diagonal(diag=self.noise_map.weight_list_ordered_1d),
            NRegs=[pylops.MatrixMult(sparse.bsr_matrix(self.regularization_matrix))],
            M=MOp,
            tol=self.settings.tolerance,
            atol=self.settings.tolerance,
            **dict(maxiter=self.settings.maxiter),
        )

    @property
    @profile_func
    def mapped_reconstructed_data_dict(
        self,
    ) -> Dict[LinearObj, Visibilities]:
        """
        When constructing the simultaneous linear equations (via vectors and matrices) the quantities of each individual
        linear object (e.g. their `mapping_matrix`) are combined into single ndarrays. This does not track which
        quantities belong to which linear objects, therefore the linear equation's solutions (which are returned as
        ndarrays) do not contain information on which linear object(s) they correspond to.

        For example, consider if two `Mapper` objects with 50 and 100 source pixels are used in an `Inversion`.
        The `reconstruction` (which contains the solved for source pixels values) is an ndarray of shape [150], but
        the ndarray itself does not track which values belong to which `Mapper`.

        This function converts an ndarray of a `reconstruction` to a dictionary of ndarrays containing each linear
        object's reconstructed images, where the keys are the instances of each mapper in the inversion.

        The PyLops calculation bypasses the calculation of the `mapping_matrix` and it therefore cannot be used to map
        the reconstruction's values to the image-plane. Instead, the unique data-to-pixelization mappings are used,
        including the 2D non-uniform fast Fourier transform operation after mapping is complete.

        Parameters
        ----------
        reconstruction
            The reconstruction (in the source frame) whose values are mapped to a dictionary of values for each
            individual mapper (in the image-plane).
        """

        mapped_reconstructed_image_dict = self.mapped_reconstructed_image_dict

        return {
            linear_obj: self.transformer.visibilities_from(image=image)
            for linear_obj, image in mapped_reconstructed_image_dict.items()
        }

    @cached_property
    @profile_func
    def preconditioner_matrix(self):

        curvature_matrix_approx = np.multiply(
            np.sum(self.noise_map.weight_list_ordered_1d),
            self.linear_obj_list[0].mapping_matrix.T
            @ self.linear_obj_list[0].mapping_matrix,
        )

        return np.add(curvature_matrix_approx, self.regularization_matrix)

    @cached_property
    @profile_func
    def preconditioner_matrix_inverse(self):
        return np.linalg.inv(self.preconditioner_matrix)

    @cached_property
    @profile_func
    def log_det_curvature_reg_matrix_term(self):
        return 2.0 * np.sum(
            np.log(np.diag(np.linalg.cholesky(self.preconditioner_matrix)))
        )

    @property
    def errors(self):
        return None
