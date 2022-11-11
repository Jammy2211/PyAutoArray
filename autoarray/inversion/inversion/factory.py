from typing import Dict, List, Optional, Union

from autoarray.dataset.imaging.imaging import Imaging
from autoarray.dataset.imaging.w_tilde import WTildeImaging
from autoarray.dataset.interferometer.w_tilde import WTildeInterferometer
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap
from autoarray.operators.convolver import Convolver
from autoarray.operators.transformer import TransformerDFT
from autoarray.operators.transformer import TransformerNUFFT
from autoarray.inversion.linear_obj.func_list import LinearObj
from autoarray.inversion.linear_obj.func_list import AbstractLinearObjFuncList
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
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.preloads import Preloads


def inversion_from(
    dataset,
    linear_obj_list: List[LinearObj],
    settings: SettingsInversion = SettingsInversion(),
    preloads: Preloads = Preloads(),
    profiling_dict: Optional[Dict] = None,
):
    """
    Factory which given an input dataset and list of linear objects, creates an `Inversion`.

    An `Inversion` reconstructs the input dataset using a list of linear objects (e.g. a list of analytic functions
    or a pixelized grid). The inversion solves for the values of these linear objects that best reconstruct the
    dataset, via linear matrix algebra.

    Different `Inversion` objects are used for different dataset types (e.g. `Imaging`, `Interferometer`) and
    for different linear algebra formalisms (determined via the input `settings`) which solve for the linear object
    parameters in different ways.

    This factory inspects the type of dataset input and settings of the inversion in order to create the appropriate
    inversion object.

    Parameters
    ----------
    dataset
        The dataset (e.g. `Imaging`, `Interferometer`) whose data is reconstructed via the `Inversion`.
    linear_obj_list
        The list of linear objects (e.g. analytic functions, a mapper with a pixelized grid) which reconstruct the
        input dataset's data and whose values are solved for via the inversion.
    settings
        Settings controlling how an inversion is fitted for example which linear algebra formalism is used.
    preloads
        Preloads in memory certain arrays which may be known beforehand in order to speed up the calculation,
        for example certain matrices used by the linear algebra could be preloaded.
    profiling_dict
        A dictionary which contains timing of certain functions calls which is used for profiling.

    Returns
    -------
    An `Inversion` whose type is determined by the input `dataset` and `settings`.
    """
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
        settings=settings,
        profiling_dict=profiling_dict,
    )


def inversion_unpacked_from(
    dataset,
    data: Union[Array2D, Visibilities],
    noise_map: Union[Array2D, VisibilitiesNoiseMap],
    w_tilde: Union[WTildeImaging, WTildeInterferometer],
    linear_obj_list: List[LinearObj],
    settings: SettingsInversion = SettingsInversion(),
    preloads: Preloads = Preloads(),
    profiling_dict: Optional[Dict] = None,
):
    """
    Factory which given an input dataset and list of linear objects, creates an `Inversion`.

    Unlike the `inversion_from` factory this function takes the `data`, `noise_map` and `w_tilde` objects as separate
    inputs, which facilitates certain computations where the `dataset` object is unpacked before the `Inversion` is
    performed (for example if the noise-map is scaled before the inversion to downweight certain regions of the
    data).

    An `Inversion` reconstructs the input dataset using a list of linear objects (e.g. a list of analytic functions
    or a pixelized grid). The inversion solves for the values of these linear objects that best reconstruct the
    dataset, via linear matrix algebra.

    Different `Inversion` objects are used for different dataset types (e.g. `Imaging`, `Interferometer`) and
    for different linear algebra formalisms (determined via the input `settings`) which solve for the linear object
    parameters in different ways.

    This factory inspects the type of dataset input and settings of the inversion in order to create the appropriate
    inversion object.

    Parameters
    ----------
    dataset
        The dataset (e.g. `Imaging`, `Interferometer`) whose data is reconstructed via the `Inversion`.
    data
        The data of the dataset (e.g. the `image` of `Imaging` data) which may have been changed.
    noise_map
        The noise_map of the noise_mapset (e.g. the `noise_map` of `Imaging` noise_map) which may have been changed.
    w_tilde
        Object which uses the dataset's operated (e.g. the PSF of `Imaging`) to perform the `Inversion` using the
        w-tilde formalism.
    linear_obj_list
        The list of linear objects (e.g. analytic functions, a mapper with a pixelized grid) which reconstruct the
        input dataset's data and whose values are solved for via the inversion.
    settings
        Settings controlling how an inversion is fitted for example which linear algebra formalism is used.
    preloads
        Preloads in memory certain arrays which may be known beforehand in order to speed up the calculation,
        for example certain matrices used by the linear algebra could be preloaded.
    profiling_dict
        A dictionary which contains timing of certain functions calls which is used for profiling.

    Returns
    -------
    An `Inversion` whose type is determined by the input `dataset` and `settings`.
    """
    if isinstance(dataset, Imaging):

        return inversion_imaging_unpacked_from(
            image=data,
            noise_map=noise_map,
            convolver=dataset.convolver,
            w_tilde=w_tilde,
            linear_obj_list=linear_obj_list,
            settings=settings,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

    return inversion_interferometer_unpacked_from(
        visibilities=data,
        noise_map=noise_map,
        transformer=dataset.transformer,
        w_tilde=w_tilde,
        linear_obj_list=linear_obj_list,
        settings=settings,
        profiling_dict=profiling_dict,
    )


def inversion_imaging_unpacked_from(
    image: Array2D,
    noise_map: Array2D,
    convolver: Convolver,
    w_tilde: WTildeImaging,
    linear_obj_list: List[LinearObj],
    settings: SettingsInversion = SettingsInversion(),
    preloads: Preloads = Preloads(),
    profiling_dict: Optional[Dict] = None,
):
    """
    Factory which given an input `Imaging` dataset and list of linear objects, creates an `InversionImaging`.

    Unlike the `inversion_from` factory this function takes the `data`, `noise_map` and `w_tilde` objects as separate
    inputs, which facilitates certain computations where the `dataset` object is unpacked before the `Inversion` is
    performed (for example if the noise-map is scaled before the inversion to downweight certain regions of the
    data).

    An `Inversion` reconstructs the input dataset using a list of linear objects (e.g. a list of analytic functions
    or a pixelized grid). The inversion solves for the values of these linear objects that best reconstruct the
    dataset, via linear matrix algebra.

    Different `Inversion` objects are used for different linear algebra formalisms (determined via the
    input `settings`) which solve for the linear object parameters in different ways.

    This factory inspects the type of dataset input and settings of the inversion in order to create the appropriate
    inversion object.

    Parameters
    ----------
    image
        The `image` data of the `Imaging` dataset which may have been changed.
    noise_map
        The noise_map of the `Imaging` dataset which may have been changed.
    w_tilde
        Object which uses the `Imaging` dataset's PSF / `Convolver` operateor to perform the `Inversion` using the
        w-tilde formalism.
    linear_obj_list
        The list of linear objects (e.g. analytic functions, a mapper with a pixelized grid) which reconstruct the
        input dataset's data and whose values are solved for via the inversion.
    settings
        Settings controlling how an inversion is fitted for example which linear algebra formalism is used.
    preloads
        Preloads in memory certain arrays which may be known beforehand in order to speed up the calculation,
        for example certain matrices used by the linear algebra could be preloaded.
    profiling_dict
        A dictionary which contains timing of certain functions calls which is used for profiling.

    Returns
    -------
    An `Inversion` whose type is determined by the input `dataset` and `settings`.
    """
    if any(
        isinstance(linear_obj, AbstractLinearObjFuncList)
        for linear_obj in linear_obj_list
    ):
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
            settings=settings,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

    return InversionImagingMapping(
        data=image,
        noise_map=noise_map,
        convolver=convolver,
        linear_obj_list=linear_obj_list,
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
    settings: SettingsInversion = SettingsInversion(),
    preloads: Preloads = Preloads(),
    profiling_dict: Optional[Dict] = None,
):
    """
    Factory which given an input `Interferometer` dataset and list of linear objects, creates
    an `InversionInterferometer`.

    Unlike the `inversion_from` factory this function takes the `data`, `noise_map` and `w_tilde` objects as separate
    inputs, which facilitates certain computations where the `dataset` object is unpacked before the `Inversion` is
    performed (for example if the noise-map is scaled before the inversion to downweight certain regions of the
    data).

    An `Inversion` reconstructs the input dataset using a list of linear objects (e.g. a list of analytic functions
    or a pixelized grid). The inversion solves for the values of these linear objects that best reconstruct the
    dataset, via linear matrix algebra.

    Different `Inversion` objects are used for different linear algebra formalisms (determined via the
    input `settings`) which solve for the linear object parameters in different ways.

    This factory inspects the type of dataset input and settings of the inversion in order to create the appropriate
    inversion object.

    Parameters
    ----------
    image
        The `image` data of the `Imaging` dataset which may have been changed.
    noise_map
        The noise_map of the `Imaging` dataset which may have been changed.
    w_tilde
        Object which uses the `Imaging` dataset's PSF / `Convolver` operateor to perform the `Inversion` using the
        w-tilde formalism.
    linear_obj_list
        The list of linear objects (e.g. analytic functions, a mapper with a pixelized grid) which reconstruct the
        input dataset's data and whose values are solved for via the inversion.
    settings
        Settings controlling how an inversion is fitted for example which linear algebra formalism is used.
    preloads
        Preloads in memory certain arrays which may be known beforehand in order to speed up the calculation,
        for example certain matrices used by the linear algebra could be preloaded.
    profiling_dict
        A dictionary which contains timing of certain functions calls which is used for profiling.

    Returns
    -------
    An `Inversion` whose type is determined by the input `dataset` and `settings`.
    """
    try:
        from autoarray.inversion.inversion import inversion_util_secret
    except ImportError:
        settings.use_w_tilde = False

    if any(
        isinstance(linear_obj, AbstractLinearObjFuncList)
        for linear_obj in linear_obj_list
    ):
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
            settings=settings,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )
