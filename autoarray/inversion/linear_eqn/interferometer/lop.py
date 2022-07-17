import numpy as np
from typing import Dict, List, Optional

from autoarray.inversion.linear_eqn.interferometer.abstract import (
    AbstractLEqInterferometer,
)
from autoarray.inversion.linear_obj.func_list import LinearObj
from autoarray.operators.transformer import TransformerNUFFT
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap

from autoarray.numba_util import profile_func


class LEqInterferometerMappingPyLops(AbstractLEqInterferometer):
    def __init__(
        self,
        noise_map: VisibilitiesNoiseMap,
        transformer: TransformerNUFFT,
        linear_obj_list: List[LinearObj],
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Constructs linear equations (via vectors and matrices) which allow for sets of simultaneous linear equations
        to be solved (see `inversion.linear_eqn.abstract.AbstractLEq` for a full description).

        A linear object describes the mappings between values in observed `data` and the linear object's model via its
        `mapping_matrix`. This class constructs linear equations for `Interferometer` objects, where the data is an
        an array of visibilities and the mappings include a non-uniform fast Fourier transform operation described by
        the interferometer dataset's transformer.

        This class uses the mapping formalism, which constructs the simultaneous linear equations using the
        `mapping_matrix` of every linear object. This is performed using the library PyLops, which uses linear
        operators to avoid these matrices being created explicitly in memory, making the calculation more efficient.

        Parameters
        -----------
        noise_map
            The noise-map of the observed interferometer data which values are solved for.
        transformer
            The transformer which performs a non-uniform fast Fourier transform operations on the mapping matrix
            with the interferometer data's transformer.
        linear_obj_list
            The linear objects used to reconstruct the data's observed values. If multiple linear objects are passed
            the simultaneous linear equations are combined and solved simultaneously.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """

        super().__init__(
            noise_map=noise_map,
            transformer=transformer,
            linear_obj_list=linear_obj_list,
            profiling_dict=profiling_dict,
        )

    @profile_func
    def mapped_reconstructed_data_dict_from(
        self, reconstruction: np.ndarray
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
        the reconstruction's values to the data frame. Instead, the unique data-to-pixelization mappings are used,
        including the 2D non-uniform fast Fourier transform operation after mapping is complete.

        Parameters
        ----------
        reconstruction
            The reconstruction (in the source frame) whose values are mapped to a dictionary of values for each
            individual mapper (in the data frame).
        """

        mapped_reconstructed_image_dict = self.mapped_reconstructed_image_dict_from(
            reconstruction=reconstruction
        )

        return {
            linear_obj: self.transformer.visibilities_from(image=image)
            for linear_obj, image in mapped_reconstructed_image_dict.items()
        }
