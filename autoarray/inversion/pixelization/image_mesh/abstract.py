from typing import Optional

from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

class AbstractImageMesh:

    def __init__(self):
        """
        An abstract image mesh, which is used by pixelizations to determine the (y,x) mesh coordinates from image
        data.
        """
        pass

    def image_mesh_from(self, grid: Grid2D, weight_map : Optional[Array2D]) -> Grid2DIrregular:
        raise NotImplementedError