from autoarray.inversion.mesh.voronoi import VoronoiMagnification
from autoarray.inversion.mesh.voronoi import VoronoiBrightnessImage


class VoronoiNNMagnification(VoronoiMagnification):
    """
    A full description of this class is given for the class `VoronoiMagnification`.

    The only difference for this class is that when it is used by the `Mapper` to map coordinates from the data
    frame to source frame it uses interpolation. This means that every pixel in the data is mapped to multiple Voronoi
    pixels, where these mappings are weighted.

    This uses uses a natural neighbor interpolation scheme (https://en.wikipedia.org/wiki/Natural_neighbor_interpolation).
    """

    @property
    def uses_interpolation(self):
        return True


class VoronoiNNBrightnessImage(VoronoiBrightnessImage):
    """
    A full description of this class is given for the class `VoronoiBrightnessImage`.

    The only difference for this class is that when it is used by the `Mapper` to map coordinates from the data
    frame to source frame it uses interpolation. This means that every pixel in the data is mapped to multiple Voronoi
    pixels, where these mappings are weighted.

    This uses uses a natural neighbor interpolation scheme (https://en.wikipedia.org/wiki/Natural_neighbor_interpolation).
    """

    @property
    def uses_interpolation(self):
        return True

    @property
    def is_stochastic(self) -> bool:
        return True
