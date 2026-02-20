from autoarray.inversion.mesh.mesh.rectangular_adapt_density import (
    RectangularAdaptDensity,
)


class RectangularUniform(RectangularAdaptDensity):

    @property
    def interpolator_cls(self):
        from autoarray.inversion.mesh.interpolator.rectangular_uniform import (
            InterpolatorRectangularUniform,
        )

        return InterpolatorRectangularUniform
