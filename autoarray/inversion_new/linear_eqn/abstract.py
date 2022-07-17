import numpy as np
from typing import Dict, List, Optional, Union

from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.preloads import Preloads
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap
from autoarray.structures.arrays.uniform_2d import Array2D


class AbstractLEq:
    def __init__(
        self,
        data: Union[Array2D, Visibilities],
        noise_map: Union[Array2D, VisibilitiesNoiseMap],
        settings: SettingsInversion = SettingsInversion(),
        preloads: Optional[Preloads] = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Constructs linear equations (via vectors and matrices) which allow for sets of simultaneous linear equations
        to be solved.

        A linear object describes the mappings between values in observed `data` and the linear object's model via its
        `mapping_matrix`. For example:

        - `Mapper` objects describe the mappings between the data's values and pixels another pixelizaiton
        (e.g. a rectangular grid, Voronoi mesh, etc.).

        - `LinearObjFuncListImaging` objects describe the mappings between the data's values and a functional form.

        From the `mapping_matrix` a system of linear equations can be constructed, which can then be solved for using
        the `Inversion` object. This class provides functions for setting up the system of linear equations.

        Parameters
        ----------
        noise_map
            The noise-map of the observed data which values are solved for.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """
        self.data = data
        self.noise_map = noise_map
        self.settings = settings
        self.preloads = preloads
        self.profiling_dict = profiling_dict

    @property
    def data_vector(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def curvature_matrix(self) -> np.ndarray:
        raise NotImplementedError
