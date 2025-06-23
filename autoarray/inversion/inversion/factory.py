from typing import Dict, List, Optional, Union

from autoarray.dataset.imaging.dataset import Imaging
from autoarray.dataset.interferometer.dataset import Interferometer
from autoarray.inversion.inversion.imaging.mapping import InversionImagingMapping
from autoarray.inversion.inversion.interferometer.mapping import (
    InversionInterferometerMapping,
)
from autoarray.inversion.inversion.interferometer.w_tilde import (
    InversionInterferometerWTilde,
)
from autoarray.inversion.inversion.dataset_interface import DatasetInterface
from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.inversion.linear_obj.func_list import AbstractLinearObjFuncList
from autoarray.inversion.inversion.imaging.w_tilde import InversionImagingWTilde
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.structures.arrays.uniform_2d import Array2D


def inversion_from(
    dataset: Union[Imaging, Interferometer, DatasetInterface],
    linear_obj_list: List[LinearObj],
    settings: SettingsInversion = SettingsInversion(),
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

    Returns
    -------
    An `Inversion` whose type is determined by the input `dataset` and `settings`.
    """
    if isinstance(dataset.data, Array2D):
        return inversion_imaging_from(
            dataset=dataset,
            linear_obj_list=linear_obj_list,
            settings=settings,
        )

    return inversion_interferometer_from(
        dataset=dataset,
        linear_obj_list=linear_obj_list,
        settings=settings,
    )


def inversion_imaging_from(
    dataset,
    linear_obj_list: List[LinearObj],
    settings: SettingsInversion = SettingsInversion(),
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
    dataset
        The dataset (e.g. `Imaging`) whose data is reconstructed via the `Inversion`.
    linear_obj_list
        The list of linear objects (e.g. analytic functions, a mapper with a pixelized grid) which reconstruct the
        input dataset's data and whose values are solved for via the inversion.
    settings
        Settings controlling how an inversion is fitted for example which linear algebra formalism is used.

    Returns
    -------
    An `Inversion` whose type is determined by the input `dataset` and `settings`.
    """
    if all(
        isinstance(linear_obj, AbstractLinearObjFuncList)
        for linear_obj in linear_obj_list
    ):
        use_w_tilde = False
    else:
        use_w_tilde = settings.use_w_tilde

    if not settings.use_w_tilde:
        use_w_tilde = False

    if use_w_tilde:
        w_tilde = dataset.w_tilde

        return InversionImagingWTilde(
            dataset=dataset,
            w_tilde=w_tilde,
            linear_obj_list=linear_obj_list,
            settings=settings,
        )

    return InversionImagingMapping(
        dataset=dataset,
        linear_obj_list=linear_obj_list,
        settings=settings,
    )


def inversion_interferometer_from(
    dataset: Union[Interferometer, DatasetInterface],
    linear_obj_list: List[LinearObj],
    settings: SettingsInversion = SettingsInversion(),
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
    dataset
        The dataset (e.g. `Interferometer`) whose data is reconstructed via the `Inversion`.
    w_tilde
        Object which uses the `Imaging` dataset's PSF to perform the `Inversion` using the w-tilde formalism.
    linear_obj_list
        The list of linear objects (e.g. analytic functions, a mapper with a pixelized grid) which reconstruct the
        input dataset's data and whose values are solved for via the inversion.
    settings
        Settings controlling how an inversion is fitted for example which linear algebra formalism is used.

    Returns
    -------
    An `Inversion` whose type is determined by the input `dataset` and `settings`.
    """
    if any(
        isinstance(linear_obj, AbstractLinearObjFuncList)
        for linear_obj in linear_obj_list
    ):
        use_w_tilde = False
    else:
        use_w_tilde = settings.use_w_tilde

    if not settings.use_linear_operators:
        if use_w_tilde:
            w_tilde = dataset.w_tilde

            return InversionInterferometerWTilde(
                dataset=dataset,
                w_tilde=w_tilde,
                linear_obj_list=linear_obj_list,
                settings=settings,
            )

        else:
            return InversionInterferometerMapping(
                dataset=dataset,
                linear_obj_list=linear_obj_list,
                settings=settings,
            )
