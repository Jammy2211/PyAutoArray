from autoarray.structures.grids.irregular import GridIrregularGrouped
from autoarray.structures.grids.irregular import GridIrregularGroupedUniform
from autoarray.structures.grids.iterate import GridIterate
from autoarray.structures.vector_fields.vector_field_irregular import (
    VectorFieldIrregular,
)
from . import exc
from . import plot
from . import util
from . import mock
from .dataset import preprocess
from .dataset.imaging import Imaging
from .dataset.imaging import MaskedImaging
from .dataset.imaging import SettingsMaskedImaging
from .dataset.imaging import SimulatorImaging
from .dataset.interferometer import Interferometer
from .dataset.interferometer import MaskedInterferometer
from .dataset.interferometer import SettingsMaskedInterferometer
from .dataset.interferometer import SimulatorInterferometer
from .fit.fit import FitImaging
from .fit.fit import FitInterferometer
from .instruments import acs
from .instruments import euclid
from .inversion import pixelizations as pix
from .inversion import regularization as reg
from .inversion.inversions import SettingsInversion
from .inversion.inversions import inversion as Inversion
from .inversion.mappers import mapper as Mapper
from .inversion.pixelizations import SettingsPixelization
from .mask.mask_1d import Mask1D
from .mask.mask_2d import Mask2D
from .mock import mock
from .mock import fixtures
from .operators.convolver import Convolver
from .operators.convolver import Convolver
from .operators.transformer import TransformerDFT
from .operators.transformer import TransformerNUFFT
from .structures.arrays import Array
from .structures.arrays import Values
from .structures.arrays.abstract_array import ExposureInfo
from .structures.frames import Frame
from .structures.frames.abstract_frame import Scans
from .structures.grids import Grid
from .structures.grids import GridInterpolate
from .structures.grids import GridRectangular
from .structures.grids import GridVoronoi
from .structures.grids import GridIrregular
from .structures.grids import GridIrregularGrouped
from .structures.grids import GridIrregularGroupedUniform
from .structures.lines.lines import Line
from .structures.lines.lines import LineCollection
from .structures.region import Region
from .structures.kernel import Kernel
from .structures.region import Region
from .structures.visibilities import Visibilities
from .structures.visibilities import VisibilitiesNoiseMap

from autoconf import conf

conf.instance.register(__file__)

__version__ = '0.16.3'
