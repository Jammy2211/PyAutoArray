from typing import Dict, List, Optional

from autoarray.inversion_new.linear_eqn.abstract import AbstractLEq
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.preloads import Preloads
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.operators.convolver import Convolver


class AbstractLEqImaging(AbstractLEq):
    def __init__(
        self,
        data: Array2D,
        noise_map: Array2D,
        convolver: Convolver,
        settings: SettingsInversion = SettingsInversion(),
        preloads: Optional[Preloads] = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Constructs linear equations (via vectors and matrices) which allow for sets of simultaneous linear equations
        to be solved (see `inversion.linear_eqn.abstract.AbstractLEq` for a full description).

        A linear object describes the mappings between values in observed `data` and the linear object's model via its
        `mapping_matrix`. This class constructs linear equations for `Imaging` objects, where the data is an image
        and the mappings may include a convolution operation described by the imaging data's PSF.

        Parameters
        -----------
        noise_map
            The noise-map of the observed imaging data which values are solved for.
        convolver
            The convolver which performs a 2D convolution on the mapping matrix with the imaging data's PSF.
        linear_obj_list
            The linear objects used to reconstruct the data's observed values. If multiple linear objects are passed
            the simultaneous linear equations are combined and solved simultaneously.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """

        self.convolver = convolver

        super().__init__(
            data=data,
            noise_map=noise_map,
            settings=settings,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

    @property
    def mask(self) -> Array2D:
        return self.noise_map.mask
