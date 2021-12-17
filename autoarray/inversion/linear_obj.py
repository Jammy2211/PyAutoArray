import numpy as np


class LinearObj:
    @property
    def mapping_matrix(self) -> np.ndarray:
        raise NotImplementedError
