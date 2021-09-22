import pylops
from scipy import sparse
from typing import Dict, Optional, Union

from autoarray.inversion.inversion.abstract import AbstractInversion
from autoarray.inversion.linear_eqn.interferometer import LinearEqnInterferometerMapping
from autoarray.inversion.linear_eqn.interferometer import (
    LinearEqnInterferometerLinearOperator,
)
from autoarray.inversion.mappers.rectangular import MapperRectangular
from autoarray.inversion.mappers.voronoi import MapperVoronoi
from autoarray.inversion.regularizations.abstract import AbstractRegularization
from autoarray.operators.transformer import TransformerDFT
from autoarray.operators.transformer import TransformerNUFFT
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap


def inversion_interferometer_from(
    dataset,
    mapper: Union[MapperRectangular, MapperVoronoi],
    regularization,
    settings=SettingsInversion(),
    profiling_dict: Optional[Dict] = None,
):

    return inversion_interferometer_unpacked_from(
        visibilities=dataset.visibilities,
        noise_map=dataset.noise_map,
        transformer=dataset.transformer,
        mapper=mapper,
        regularization=regularization,
        settings=settings,
        profiling_dict=profiling_dict,
    )


def inversion_interferometer_unpacked_from(
    visibilities: Visibilities,
    noise_map: VisibilitiesNoiseMap,
    transformer: Union[TransformerDFT, TransformerNUFFT],
    mapper: Union[MapperRectangular, MapperVoronoi],
    regularization: AbstractRegularization,
    settings: SettingsInversion = SettingsInversion(),
    profiling_dict: Optional[Dict] = None,
):
    if not settings.use_linear_operators:

        linear_eqn = LinearEqnInterferometerMapping(
            noise_map=noise_map,
            transformer=transformer,
            mapper=mapper,
            regularization=regularization,
            profiling_dict=profiling_dict,
        )

    else:

        linear_eqn = LinearEqnInterferometerLinearOperator(
            noise_map=noise_map,
            transformer=transformer,
            mapper=mapper,
            regularization=regularization,
            profiling_dict=profiling_dict,
        )

    return InversionInterferometer(
        data=visibilities,
        linear_eqn=linear_eqn,
        settings=settings,
        profiling_dict=profiling_dict,
    )


class InversionInterferometer(AbstractInversion):
    def __init__(
        self,
        data: Union[Visibilities],
        linear_eqn: Union[
            LinearEqnInterferometerMapping, LinearEqnInterferometerLinearOperator
        ],
        settings: SettingsInversion = SettingsInversion(),
        profiling_dict: Optional[Dict] = None,
    ):
        """
        An inversion, which given an input image and noise-map reconstructs the image using a linear inversion, \
        including a convolution that accounts for blurring.

        The inversion uses a 2D pixelization to perform the reconstruction by util each pixelization pixel to a \
        set of image pixels via a mapper. The reconstructed pixelization is smoothed via a regularization scheme to \
        prevent over-fitting noise.

        Parameters
        -----------
        image_1d
            Flattened 1D array of the observed image the inversion is fitting.
        noise_map
            Flattened 1D array of the noise-map used by the inversion during the fit.
        convolver : imaging.convolution.Convolver
            The convolver used to blur the mapping matrix with the PSF.
        mapper : inversion.Mapper
            The util between the image-pixels (via its / sub-grid) and pixelization pixels.
        regularization : inversion.regularization.Regularization
            The regularization scheme applied to smooth the pixelization used to reconstruct the image for the \
            inversion

        Attributes
        -----------
        blurred_mapping_matrix
            The matrix representing the blurred mappings between the image's sub-grid of pixels and the pixelization \
            pixels.
        regularization_matrix
            The matrix defining how the pixelization's pixels are regularized with one another for smoothing (H).
        curvature_matrix
            The curvature_matrix between each pixelization pixel and all other pixelization pixels (F).
        curvature_reg_matrix
            The curvature_matrix + regularization matrix.
        solution_vector
            The vector containing the reconstructed fit to the hyper_galaxies.
        """

        super().__init__(
            data=data,
            linear_eqn=linear_eqn,
            settings=settings,
            profiling_dict=profiling_dict,
        )

    @property
    def visibilities(self):
        return self.data

    @property
    def mapped_reconstructed_visibilities(self):

        return self.linear_eqn.mapped_reconstructed_visibilities_from(
            reconstruction=self.reconstruction
        )

    @property
    def residual_map(self):
        return None

    @property
    def normalized_residual_map(self):
        return None

    @property
    def chi_squared_map(self):
        return None
