import numpy as np

import autoarray as aa

from autoarray.mock.mock import MockMask
from autoarray.mock.mock import MockDataset
from autoarray.mock.mock import MockFit
from autoarray.mock.mock import MockMapper
from autoarray.mock.mock import MockRegularization
from autoarray.mock.mock import MockLEqMapper
from autoarray.mock.mock import MockLEqMapperImaging
from autoarray.mock.mock import MockInversion

# def test__set_w_tilde():
#
#     # fit inversion is None, so no need to bother with w_tilde.
#
#     fit_0 = MockFit(inversion=None)
#     fit_1 = MockFit(inversion=None)
#
#     preloads = aa.Preloads(w_tilde=1, use_w_tilde=1)
#     preloads.set_w_tilde_imaging(fit_0=fit_0, fit_1=fit_1)
#
#     assert preloads.w_tilde is None
#     assert preloads.use_w_tilde is False
#
#     # Noise maps of fit are different but there is an inversion, so we should not preload w_tilde and use w_tilde.
#
#     fit_0 = MockFit(
#         inversion=1, noise_map=aa.Array2D.zeros(shape_native=(3, 1), pixel_scales=0.1)
#     )
#     fit_1 = MockFit(
#         inversion=1, noise_map=aa.Array2D.ones(shape_native=(3, 1), pixel_scales=0.1)
#     )
#
#     preloads = aa.Preloads(w_tilde=1, use_w_tilde=1)
#     preloads.set_w_tilde_imaging(fit_0=fit_0, fit_1=fit_1)
#
#     assert preloads.w_tilde is None
#     assert preloads.use_w_tilde is False
#
#     # Noise maps of fits are the same so preload w_tilde and use it.
#
#     noise_map = aa.Array2D.ones(shape_native=(5, 5), pixel_scales=0.1, sub_size=1)
#
#     mask = MockMask(
#         native_index_for_slim_index=noise_map.mask.native_index_for_slim_index
#     )
#
#     dataset = MockDataset(psf=aa.Kernel2D.no_blur(pixel_scales=1.0), mask=mask)
#
#     fit_0 = MockFit(inversion=1, dataset=dataset, noise_map=noise_map)
#     fit_1 = MockFit(inversion=1, dataset=dataset, noise_map=noise_map)
#
#     preloads = aa.Preloads(w_tilde=1, use_w_tilde=1)
#     preloads.set_w_tilde_imaging(fit_0=fit_0, fit_1=fit_1)
#
#     curvature_preload, indexes, lengths = aa.util.inversion.w_tilde_curvature_preload_imaging_from(
#         noise_map_native=fit_0.noise_map.native,
#         kernel_native=fit_0.dataset.psf.native,
#         native_index_for_slim_index=fit_0.dataset.mask.native_index_for_slim_index,
#     )
#
#     assert (preloads.w_tilde.curvature_preload == curvature_preload).all()
#     assert (preloads.w_tilde.indexes == indexes).all()
#     assert (preloads.w_tilde.lengths == lengths).all()
#     assert preloads.w_tilde.noise_map_value == 1.0
#     assert preloads.use_w_tilde == True


def test__set_relocated_grid():

    # LEq is None so there is no mapper, thus preload mapper to None.

    fit_0 = MockFit(inversion=None)
    fit_1 = MockFit(inversion=None)

    preloads = aa.Preloads(relocated_grid=1)
    preloads.set_relocated_grid(fit_0=fit_0, fit_1=fit_1)

    assert preloads.relocated_grid is None

    # Mapper's mapping matrices are different, thus preload mapper to None.

    leq_0 = MockLEqMapper(mapper_list=[MockMapper(source_grid_slim=np.ones((3, 2)))])
    leq_1 = MockLEqMapper(
        mapper_list=[MockMapper(source_grid_slim=2.0 * np.ones((3, 2)))]
    )

    fit_0 = MockFit(inversion=MockInversion(leq=leq_0))
    fit_1 = MockFit(inversion=MockInversion(leq=leq_1))

    preloads = aa.Preloads(relocated_grid=1)
    preloads.set_relocated_grid(fit_0=fit_0, fit_1=fit_1)

    assert preloads.relocated_grid is None

    # Mapper's mapping matrices are the same, thus preload mapper.

    leq_0 = MockLEqMapper(mapper_list=[MockMapper(source_grid_slim=np.ones((3, 2)))])
    leq_1 = MockLEqMapper(mapper_list=[MockMapper(source_grid_slim=np.ones((3, 2)))])

    fit_0 = MockFit(inversion=MockInversion(leq=leq_0))
    fit_1 = MockFit(inversion=MockInversion(leq=leq_1))

    preloads = aa.Preloads(relocated_grid=1)
    preloads.set_relocated_grid(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.relocated_grid == np.ones((3, 2))).all()


def test__set_mapper_list():

    # LEq is None so there is no mapper, thus preload mapper to None.

    fit_0 = MockFit(inversion=None)
    fit_1 = MockFit(inversion=None)

    preloads = aa.Preloads(mapper_list=1)
    preloads.set_mapper_list(fit_0=fit_0, fit_1=fit_1)

    assert preloads.mapper_list is None

    # Mapper's mapping matrices are different, thus preload mapper to None.

    leq_0 = MockLEqMapper(mapper_list=[MockMapper(mapping_matrix=np.ones((3, 2)))])
    leq_1 = MockLEqMapper(
        mapper_list=[MockMapper(mapping_matrix=2.0 * np.ones((3, 2)))]
    )

    fit_0 = MockFit(inversion=MockInversion(leq=leq_0))
    fit_1 = MockFit(inversion=MockInversion(leq=leq_1))

    preloads = aa.Preloads(mapper_list=1)
    preloads.set_mapper_list(fit_0=fit_0, fit_1=fit_1)

    assert preloads.mapper_list is None

    # Mapper's mapping matrices are the same, thus preload mapper.

    leq_0 = MockLEqMapper(mapper_list=[MockMapper(mapping_matrix=np.ones((3, 2)))])
    leq_1 = MockLEqMapper(mapper_list=[MockMapper(mapping_matrix=np.ones((3, 2)))])

    fit_0 = MockFit(inversion=MockInversion(leq=leq_0))
    fit_1 = MockFit(inversion=MockInversion(leq=leq_1))

    preloads = aa.Preloads(mapper_list=1)
    preloads.set_mapper_list(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.mapper_list[0].mapping_matrix == np.ones((3, 2))).all()

    # Multiple mappers pre inversion still preloads full mapper list.

    leq_0 = MockLEqMapper(
        mapper_list=[
            MockMapper(mapping_matrix=np.ones((3, 2))),
            MockMapper(mapping_matrix=np.ones((3, 2))),
        ]
    )
    leq_1 = MockLEqMapper(
        mapper_list=[
            MockMapper(mapping_matrix=np.ones((3, 2))),
            MockMapper(mapping_matrix=np.ones((3, 2))),
        ]
    )

    fit_0 = MockFit(inversion=MockInversion(leq=leq_0))
    fit_1 = MockFit(inversion=MockInversion(leq=leq_1))

    preloads = aa.Preloads(mapper_list=1)
    preloads.set_mapper_list(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.mapper_list[0].mapping_matrix == np.ones((3, 2))).all()
    assert (preloads.mapper_list[1].mapping_matrix == np.ones((3, 2))).all()


def test__set_operated_mapping_matrix_with_preloads():

    curvature_matrix_preload = np.array([[1.0]])
    curvature_matrix_counts = np.array([1.0])

    # LEq is None thus preload it to None.

    fit_0 = MockFit(inversion=None)
    fit_1 = MockFit(inversion=None)

    preloads = aa.Preloads(
        operated_mapping_matrix=1,
        curvature_matrix_preload=np.array([[1.0]]),
        curvature_matrix_counts=np.array([1.0]),
    )
    preloads.set_operated_mapping_matrix_with_preloads(fit_0=fit_0, fit_1=fit_1)

    assert preloads.operated_mapping_matrix is None
    assert preloads.curvature_matrix_preload is None
    assert preloads.curvature_matrix_counts is None

    # LEq's blurred mapping matrices are different thus no preloading.

    blurred_mapping_matrix_0 = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    blurred_mapping_matrix_1 = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    leq_0 = MockLEqMapperImaging(blurred_mapping_matrix=blurred_mapping_matrix_0)
    leq_1 = MockLEqMapperImaging(blurred_mapping_matrix=blurred_mapping_matrix_1)

    fit_0 = MockFit(inversion=MockInversion(leq=leq_0))
    fit_1 = MockFit(inversion=MockInversion(leq=leq_1))

    preloads = aa.Preloads(
        operated_mapping_matrix=1,
        curvature_matrix_preload=curvature_matrix_preload,
        curvature_matrix_counts=curvature_matrix_counts,
    )
    preloads.set_operated_mapping_matrix_with_preloads(fit_0=fit_0, fit_1=fit_1)

    assert preloads.operated_mapping_matrix is None
    assert preloads.curvature_matrix_preload is None
    assert preloads.curvature_matrix_counts is None

    # LEq's blurred mapping matrices are the same therefore preload it and the curvature sparse terms.

    leq_0 = MockLEqMapperImaging(blurred_mapping_matrix=blurred_mapping_matrix_0)
    leq_1 = MockLEqMapperImaging(blurred_mapping_matrix=blurred_mapping_matrix_0)

    inversion_0 = MockInversion(
        leq=leq_0,
        curvature_matrix_preload=curvature_matrix_preload,
        curvature_matrix_counts=curvature_matrix_counts,
    )

    fit_0 = MockFit(inversion=inversion_0)
    fit_1 = MockFit(inversion=MockInversion(leq=leq_1))

    preloads = aa.Preloads(
        operated_mapping_matrix=1,
        curvature_matrix_preload=curvature_matrix_preload,
        curvature_matrix_counts=curvature_matrix_counts,
    )
    preloads.set_operated_mapping_matrix_with_preloads(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.operated_mapping_matrix == blurred_mapping_matrix_0).all()
    assert (
        preloads.curvature_matrix_preload == curvature_matrix_preload.astype("int")
    ).all()
    assert (
        preloads.curvature_matrix_counts == curvature_matrix_counts.astype("int")
    ).all()


def test__set_regularization_matrix_and_term():

    regularization = MockRegularization(regularization_matrix=np.eye(2))

    # Inversion is None thus preload log_det_regularization_matrix_term to None.

    fit_0 = MockFit(inversion=None)
    fit_1 = MockFit(inversion=None)

    preloads = aa.Preloads(log_det_regularization_matrix_term=1)
    preloads.set_regularization_matrix_and_term(fit_0=fit_0, fit_1=fit_1)

    assert preloads.regularization_matrix is None
    assert preloads.log_det_regularization_matrix_term is None

    # Inversion's log_det_regularization_matrix_term are different thus no preloading.

    fit_0 = MockFit(
        inversion=MockInversion(
            log_det_regularization_matrix_term=0, regularization_list=[regularization]
        )
    )
    fit_1 = MockFit(
        inversion=MockInversion(
            log_det_regularization_matrix_term=1, regularization_list=[regularization]
        )
    )

    preloads = aa.Preloads(log_det_regularization_matrix_term=1)
    preloads.set_regularization_matrix_and_term(fit_0=fit_0, fit_1=fit_1)

    assert preloads.regularization_matrix is None
    assert preloads.log_det_regularization_matrix_term is None

    # LEq's blurred mapping matrices are the same therefore preload it and the curvature sparse terms.

    preloads = aa.Preloads(log_det_regularization_matrix_term=2)

    fit_0 = MockFit(
        inversion=MockInversion(
            leq=MockLEqMapper(mapper_list=[MockMapper()]),
            log_det_regularization_matrix_term=1,
            regularization_list=[regularization],
            preloads=preloads,
        )
    )
    fit_1 = MockFit(
        inversion=MockInversion(
            leq=MockLEqMapper(mapper_list=[MockMapper()]),
            log_det_regularization_matrix_term=1,
            regularization_list=[regularization],
            preloads=preloads,
        )
    )

    preloads.set_regularization_matrix_and_term(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.regularization_matrix == np.eye(2)).all()
    assert preloads.log_det_regularization_matrix_term == 1
