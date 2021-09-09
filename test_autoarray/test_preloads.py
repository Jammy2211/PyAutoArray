import numpy as np

import autoarray as aa

from autoarray.mock.mock import MockMask
from autoarray.mock.mock import MockDataset
from autoarray.mock.mock import MockFit
from autoarray.mock.mock import MockMapper
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
#         signal_to_noise_map_native=fit_0.signal_to_noise_map.native,
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

    # Inversion is None so there is no mapper, thus preload mapper to None.

    fit_0 = MockFit(inversion=None)
    fit_1 = MockFit(inversion=None)

    preloads = aa.Preloads(relocated_grid=1)
    preloads.set_relocated_grid(fit_0=fit_0, fit_1=fit_1)

    assert preloads.relocated_grid is None

    # Mapper's mapping matrices are different, thus preload mapper to None.

    inversion_0 = MockInversion(mapper=MockMapper(source_grid_slim=np.ones((3, 2))))
    inversion_1 = MockInversion(
        mapper=MockMapper(source_grid_slim=2.0 * np.ones((3, 2)))
    )

    fit_0 = MockFit(inversion=inversion_0)
    fit_1 = MockFit(inversion=inversion_1)

    preloads = aa.Preloads(relocated_grid=1)
    preloads.set_relocated_grid(fit_0=fit_0, fit_1=fit_1)

    assert preloads.relocated_grid is None

    # Mapper's mapping matrices are the same, thus preload mapper.

    inversion_0 = MockInversion(mapper=MockMapper(source_grid_slim=np.ones((3, 2))))
    inversion_1 = MockInversion(mapper=MockMapper(source_grid_slim=np.ones((3, 2))))

    fit_0 = MockFit(inversion=inversion_0)
    fit_1 = MockFit(inversion=inversion_1)

    preloads = aa.Preloads(relocated_grid=1)
    preloads.set_relocated_grid(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.relocated_grid == np.ones((3, 2))).all()


def test__set_mapper():

    # Inversion is None so there is no mapper, thus preload mapper to None.

    fit_0 = MockFit(inversion=None)
    fit_1 = MockFit(inversion=None)

    preloads = aa.Preloads(mapper=1)
    preloads.set_mapper(fit_0=fit_0, fit_1=fit_1)

    assert preloads.mapper is None

    # Mapper's mapping matrices are different, thus preload mapper to None.

    inversion_0 = MockInversion(mapper=MockMapper(mapping_matrix=np.ones((3, 2))))
    inversion_1 = MockInversion(mapper=MockMapper(mapping_matrix=2.0 * np.ones((3, 2))))

    fit_0 = MockFit(inversion=inversion_0)
    fit_1 = MockFit(inversion=inversion_1)

    preloads = aa.Preloads(mapper=1)
    preloads.set_mapper(fit_0=fit_0, fit_1=fit_1)

    assert preloads.mapper is None

    # Mapper's mapping matrices are the same, thus preload mapper.

    inversion_0 = MockInversion(mapper=MockMapper(mapping_matrix=np.ones((3, 2))))
    inversion_1 = MockInversion(mapper=MockMapper(mapping_matrix=np.ones((3, 2))))

    fit_0 = MockFit(inversion=inversion_0)
    fit_1 = MockFit(inversion=inversion_1)

    preloads = aa.Preloads(mapper=1)
    preloads.set_mapper(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.mapper.mapping_matrix == np.ones((3, 2))).all()


def test__set_inversion():

    curvature_matrix_sparse_preload = np.array([[1.0]])
    curvature_matrix_preload_counts = np.array([1.0])

    # Inversion is None thus preload it to None.

    fit_0 = MockFit(inversion=None)
    fit_1 = MockFit(inversion=None)

    preloads = aa.Preloads(
        blurred_mapping_matrix=1,
        curvature_matrix_sparse_preload=np.array([[1.0]]),
        curvature_matrix_preload_counts=np.array([1.0]),
    )
    preloads.set_inversion(fit_0=fit_0, fit_1=fit_1)

    assert preloads.blurred_mapping_matrix is None
    assert preloads.curvature_matrix_sparse_preload is None
    assert preloads.curvature_matrix_preload_counts is None

    # Inversion's blurred mapping matrices are different thus no preloading.

    blurred_mapping_matrix_0 = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    blurred_mapping_matrix_1 = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    inversion_0 = MockInversion(blurred_mapping_matrix=blurred_mapping_matrix_0)
    inversion_1 = MockInversion(blurred_mapping_matrix=blurred_mapping_matrix_1)

    fit_0 = MockFit(inversion=inversion_0)
    fit_1 = MockFit(inversion=inversion_1)

    preloads = aa.Preloads(
        blurred_mapping_matrix=1,
        curvature_matrix_sparse_preload=curvature_matrix_sparse_preload,
        curvature_matrix_preload_counts=curvature_matrix_preload_counts,
    )
    preloads.set_inversion(fit_0=fit_0, fit_1=fit_1)

    assert preloads.blurred_mapping_matrix is None
    assert preloads.curvature_matrix_sparse_preload is None
    assert preloads.curvature_matrix_preload_counts is None

    # Inversion's blurred mapping matrices are the same therefore preload it and the curvature sparse terms.

    inversion_0 = MockInversion(
        blurred_mapping_matrix=blurred_mapping_matrix_0,
        curvature_matrix_sparse_preload=curvature_matrix_sparse_preload,
        curvature_matrix_preload_counts=curvature_matrix_preload_counts,
    )
    inversion_1 = MockInversion(blurred_mapping_matrix=blurred_mapping_matrix_0)

    fit_0 = MockFit(inversion=inversion_0)
    fit_1 = MockFit(inversion=inversion_1)

    preloads = aa.Preloads(
        blurred_mapping_matrix=1,
        curvature_matrix_sparse_preload=curvature_matrix_sparse_preload,
        curvature_matrix_preload_counts=curvature_matrix_preload_counts,
    )
    preloads.set_inversion(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.blurred_mapping_matrix == blurred_mapping_matrix_0).all()
    assert (
        preloads.curvature_matrix_sparse_preload
        == curvature_matrix_sparse_preload.astype("int")
    ).all()
    assert (
        preloads.curvature_matrix_preload_counts
        == curvature_matrix_preload_counts.astype("int")
    ).all()


def test__set_regularization_matrix_and_term():

    # Inversion is None thus preload log_det_regularization_matrix_term to None.

    fit_0 = MockFit(inversion=None)
    fit_1 = MockFit(inversion=None)

    preloads = aa.Preloads(log_det_regularization_matrix_term=1)
    preloads.set_regularization_matrix_and_term(fit_0=fit_0, fit_1=fit_1)

    assert preloads.regularization_matrix is None
    assert preloads.log_det_regularization_matrix_term is None

    # Inversion's log_det_regularization_matrix_term are different thus no preloading.

    inversion_0 = MockInversion(log_det_regularization_matrix_term=0)
    inversion_1 = MockInversion(log_det_regularization_matrix_term=1)

    fit_0 = MockFit(inversion=inversion_0)
    fit_1 = MockFit(inversion=inversion_1)

    preloads = aa.Preloads(log_det_regularization_matrix_term=1)
    preloads.set_regularization_matrix_and_term(fit_0=fit_0, fit_1=fit_1)

    assert preloads.regularization_matrix is None
    assert preloads.log_det_regularization_matrix_term is None

    # Inversion's blurred mapping matrices are the same therefore preload it and the curvature sparse terms.

    inversion_0 = MockInversion(log_det_regularization_matrix_term=1)
    inversion_1 = MockInversion(log_det_regularization_matrix_term=1)

    fit_0 = MockFit(inversion=inversion_0)
    fit_1 = MockFit(inversion=inversion_1)

    preloads = aa.Preloads(log_det_regularization_matrix_term=2)
    preloads.set_regularization_matrix_and_term(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.regularization_matrix == np.zeros((1, 1))).all()
    assert preloads.log_det_regularization_matrix_term == 1
