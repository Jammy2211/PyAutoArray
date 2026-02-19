from autoarray.inversion.pixelization.mesh.rectangular_adapt_density import (
    RectangularAdaptDensity,
)


class RectangularUniform(RectangularAdaptDensity):

    @property
    def mapper_cls(self):
        from autoarray.inversion.pixelization.mappers.rectangular_uniform import (
            MapperRectangularUniform,
        )

        return MapperRectangularUniform
