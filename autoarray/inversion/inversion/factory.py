import numpy as np
from typing import Dict, List, Optional, Union

from autoarray.dataset.imaging import Imaging
from autoarray.dataset.imaging import WTildeImaging
from autoarray.dataset.interferometer import WTildeInterferometer
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap
from autoarray.operators.convolver import Convolver
from autoarray.operators.transformer import TransformerDFT
from autoarray.operators.transformer import TransformerNUFFT
from autoarray.inversion.linear_obj.func_list import LinearObj
from autoarray.inversion.linear_obj.func_list import LinearObjFuncList
from autoarray.inversion.inversion.imaging.w_tilde import InversionImagingWTilde
from autoarray.inversion.inversion.imaging.mapping import InversionImagingMapping
from autoarray.inversion.inversion.interferometer.mapping import (
    InversionInterferometerMapping,
)
from autoarray.inversion.inversion.interferometer.w_tilde import (
    InversionInterferometerWTilde,
)
from autoarray.inversion.inversion.interferometer.lop import (
    InversionInterferometerMappingPyLops,
)
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.preloads import Preloads


def inversion_from(
    dataset,
    linear_obj_list: List[LinearObj],
    regularization_list: List[Optional[AbstractRegularization]] = None,
    settings: SettingsInversion = SettingsInversion(),
    preloads: Preloads = Preloads(),
    profiling_dict: Optional[Dict] = None,
):

    if settings.use_w_tilde:
        w_tilde = dataset.w_tilde
    else:
        w_tilde = None

    if isinstance(dataset, Imaging):

        return inversion_imaging_unpacked_from(
            image=dataset.image,
            noise_map=dataset.noise_map,
            convolver=dataset.convolver,
            w_tilde=w_tilde,
            linear_obj_list=linear_obj_list,
            regularization_list=regularization_list,
            settings=settings,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

    return inversion_interferometer_unpacked_from(
        visibilities=dataset.visibilities,
        noise_map=dataset.noise_map,
        transformer=dataset.transformer,
        w_tilde=w_tilde,
        linear_obj_list=linear_obj_list,
        regularization_list=regularization_list,
        settings=settings,
        profiling_dict=profiling_dict,
    )


def inversion_imaging_unpacked_from(
    image: Array2D,
    noise_map: Array2D,
    convolver: Convolver,
    w_tilde: WTildeImaging,
    linear_obj_list: List[LinearObj],
    regularization_list: List[Optional[AbstractRegularization]] = None,
    settings: SettingsInversion = SettingsInversion(),
    preloads: Preloads = Preloads(),
    profiling_dict: Optional[Dict] = None,
):

    if any(isinstance(linear_obj, LinearObjFuncList) for linear_obj in linear_obj_list):
        use_w_tilde = False
    elif preloads.use_w_tilde is not None:
        use_w_tilde = preloads.use_w_tilde
    else:
        use_w_tilde = settings.use_w_tilde

    if not settings.use_w_tilde:
        use_w_tilde = False

    if preloads.w_tilde is not None:

        w_tilde = preloads.w_tilde

    if use_w_tilde:

        return InversionImagingWTilde(
            data=image,
            noise_map=noise_map,
            convolver=convolver,
            w_tilde=w_tilde,
            linear_obj_list=linear_obj_list,
            regularization_list=regularization_list,
            settings=settings,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

    return InversionImagingMapping(
        data=image,
        noise_map=noise_map,
        convolver=convolver,
        linear_obj_list=linear_obj_list,
        regularization_list=regularization_list,
        settings=settings,
        preloads=preloads,
        profiling_dict=profiling_dict,
    )


def inversion_interferometer_unpacked_from(
    visibilities: Visibilities,
    noise_map: VisibilitiesNoiseMap,
    transformer: Union[TransformerDFT, TransformerNUFFT],
    w_tilde: WTildeInterferometer,
    linear_obj_list: List[LinearObj],
    regularization_list: List[Optional[AbstractRegularization]] = None,
    settings: SettingsInversion = SettingsInversion(),
    preloads: Preloads = Preloads(),
    profiling_dict: Optional[Dict] = None,
):

    try:
        from autoarray.inversion.inversion import inversion_util_secret
    except ImportError:
        settings.use_w_tilde = False

    if any(isinstance(linear_obj, LinearObjFuncList) for linear_obj in linear_obj_list):
        use_w_tilde = False
    else:
        use_w_tilde = settings.use_w_tilde

    if not settings.use_linear_operators:

        if use_w_tilde:

            return InversionInterferometerWTilde(
                data=visibilities,
                noise_map=noise_map,
                transformer=transformer,
                w_tilde=w_tilde,
                linear_obj_list=linear_obj_list,
                regularization_list=regularization_list,
                settings=settings,
                preloads=preloads,
                profiling_dict=profiling_dict,
            )

        else:

            return InversionInterferometerMapping(
                data=visibilities,
                noise_map=noise_map,
                transformer=transformer,
                linear_obj_list=linear_obj_list,
                regularization_list=regularization_list,
                settings=settings,
                preloads=preloads,
                profiling_dict=profiling_dict,
            )

    else:

        return InversionInterferometerMappingPyLops(
            data=visibilities,
            noise_map=noise_map,
            transformer=transformer,
            linear_obj_list=linear_obj_list,
            regularization_list=regularization_list,
            settings=settings,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )
