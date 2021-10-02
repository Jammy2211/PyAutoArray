from typing import Dict, List, Optional, Union

from autoarray.dataset.imaging import Imaging
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap
from autoarray.operators.convolver import Convolver
from autoarray.operators.transformer import TransformerDFT
from autoarray.operators.transformer import TransformerNUFFT
from autoarray.inversion.linear_eqn.imaging import LinearEqnImagingWTilde
from autoarray.inversion.linear_eqn.imaging import LinearEqnImagingMapping
from autoarray.inversion.inversion.matrices import InversionMatrices
from autoarray.inversion.inversion.linear_operator import InversionLinearOperator
from autoarray.inversion.linear_eqn.interferometer import LinearEqnInterferometerMapping
from autoarray.inversion.linear_eqn.interferometer import (
    LinearEqnInterferometerLinearOperator,
)
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.mappers.rectangular import MapperRectangular
from autoarray.inversion.mappers.voronoi import MapperVoronoi
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.preloads import Preloads


def inversion_from(
    dataset,
    mapper_list: List[Union[MapperRectangular, MapperVoronoi]],
    regularization_list: List[AbstractRegularization],
    settings: SettingsInversion = SettingsInversion(),
    preloads: Preloads = Preloads(),
    profiling_dict: Optional[Dict] = None,
):

    if isinstance(dataset, Imaging):

        return inversion_imaging_unpacked_from(
            image=dataset.image,
            noise_map=dataset.noise_map,
            convolver=dataset.convolver,
            w_tilde=dataset.w_tilde,
            mapper_list=mapper_list,
            regularization_list=regularization_list,
            settings=settings,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

    return inversion_interferometer_unpacked_from(
        visibilities=dataset.visibilities,
        noise_map=dataset.noise_map,
        transformer=dataset.transformer,
        mapper_list=mapper_list,
        regularization_list=regularization_list,
        settings=settings,
        profiling_dict=profiling_dict,
    )


def inversion_imaging_unpacked_from(
    image: Array2D,
    noise_map: Array2D,
    convolver: Convolver,
    w_tilde,
    mapper_list: List[Union[MapperRectangular, MapperVoronoi]],
    regularization_list: List[AbstractRegularization],
    settings: SettingsInversion = SettingsInversion(),
    preloads: Preloads = Preloads(),
    profiling_dict: Optional[Dict] = None,
):

    if preloads.use_w_tilde is not None:
        use_w_tilde = preloads.use_w_tilde
    else:
        use_w_tilde = settings.use_w_tilde

    if use_w_tilde:

        linear_eqn = LinearEqnImagingWTilde(
            noise_map=noise_map,
            convolver=convolver,
            w_tilde=w_tilde,
            mapper_list=mapper_list,
            profiling_dict=profiling_dict,
        )

    else:

        linear_eqn = LinearEqnImagingMapping(
            noise_map=noise_map,
            convolver=convolver,
            mapper_list=mapper_list,
            profiling_dict=profiling_dict,
        )

    return InversionMatrices(
        data=image,
        linear_eqn=linear_eqn,
        regularization_list=regularization_list,
        settings=settings,
        preloads=preloads,
        profiling_dict=profiling_dict,
    )


def inversion_interferometer_unpacked_from(
    visibilities: Visibilities,
    noise_map: VisibilitiesNoiseMap,
    transformer: Union[TransformerDFT, TransformerNUFFT],
    mapper_list: List[Union[MapperRectangular, MapperVoronoi]],
    regularization_list: List[AbstractRegularization],
    settings: SettingsInversion = SettingsInversion(),
    preloads: Preloads = Preloads(),
    profiling_dict: Optional[Dict] = None,
):
    if not settings.use_linear_operators:

        linear_eqn = LinearEqnInterferometerMapping(
            noise_map=noise_map,
            transformer=transformer,
            mapper_list=mapper_list,
            profiling_dict=profiling_dict,
        )

    else:

        linear_eqn = LinearEqnInterferometerLinearOperator(
            noise_map=noise_map,
            transformer=transformer,
            mapper_list=mapper_list,
            profiling_dict=profiling_dict,
        )

    if not settings.use_linear_operators:

        return InversionMatrices(
            data=visibilities,
            linear_eqn=linear_eqn,
            regularization_list=regularization_list,
            settings=settings,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

    return InversionLinearOperator(
        data=visibilities,
        linear_eqn=linear_eqn,
        regularization_list=regularization_list,
        settings=settings,
        profiling_dict=profiling_dict,
    )
