from typing import Optional

from autoarray.inversion.pixelization.mesh.abstract import AbstractMesh
from autoarray.inversion.regularization.abstract import AbstractRegularization


class Pixelization:
    def __init__(self, mesh: AbstractMesh, regularization: Optional[AbstractRegularization] = None):

        self.mesh = mesh
        self.regularization = regularization
