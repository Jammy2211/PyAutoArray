import autoarray as aa
import numpy as np
import pytest


def test__curvature_matrix_diag_via_psf_weighted_noise_from():
    psf_weighted_noise = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 2.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )

    mapping_matrix = np.array(
        [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    )

    curvature_matrix = (
        aa.util.inversion.curvature_matrix_diag_via_psf_weighted_noise_from(
            psf_weighted_noise=psf_weighted_noise, mapping_matrix=mapping_matrix
        )
    )

    assert (
        curvature_matrix
        == np.array([[6.0, 8.0, 0.0], [8.0, 8.0, 0.0], [0.0, 0.0, 0.0]])
    ).all()


def test__curvature_matrix_via_mapping_matrix_from():
    blurred_mapping_matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    noise_map = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    curvature_matrix = aa.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix, noise_map=noise_map
    )

    assert (
        curvature_matrix
        == np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 1.0]])
    ).all()

    blurred_mapping_matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    noise_map = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    curvature_matrix = aa.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix, noise_map=noise_map
    )

    assert (
        curvature_matrix
        == np.array([[1.25, 0.25, 0.0], [0.25, 2.25, 1.0], [0.0, 1.0, 1.0]])
    ).all()


def test__reconstruction_positive_negative_from():
    data_vector = np.array([1.0, 1.0, 2.0])

    curvature_reg_matrix = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 1.0]])

    reconstruction = aa.util.inversion.reconstruction_positive_negative_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature_reg_matrix,
    )

    assert reconstruction == pytest.approx(np.array([1.0, -1.0, 3.0]), 1.0e-4)


def test__mapped_reconstructed_data_via_mapping_matrix_from():
    mapping_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    reconstruction = np.array([1.0, 1.0, 2.0])

    mapped_reconstructed_operated_data = (
        aa.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix, reconstruction=reconstruction
        )
    )

    assert (mapped_reconstructed_operated_data == np.array([1.0, 1.0, 2.0])).all()

    mapping_matrix = np.array([[0.25, 0.50, 0.25], [0.0, 1.0, 0.0], [0.0, 0.25, 0.75]])

    reconstruction = np.array([1.0, 1.0, 2.0])

    mapped_reconstructed_operated_data = (
        aa.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix, reconstruction=reconstruction
        )
    )

    assert (mapped_reconstructed_operated_data == np.array([1.25, 1.0, 1.75])).all()


def test__mapped_reconstructed_data_via_image_to_pix_unique_from():
    pix_indexes_for_sub_slim_index = np.array([[0], [1], [2]])
    pix_indexes_for_sub_slim_index_sizes = np.array([1, 1, 1]).astype("int")
    pix_weights_for_sub_slim_index = np.array([[1.0], [1.0], [1.0]])

    (
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = aa.util.mapper_numba.data_slim_to_pixelization_unique_from(
        data_pixels=3,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_sizes_for_sub_slim_index=pix_indexes_for_sub_slim_index_sizes,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        pix_pixels=3,
        sub_size=np.array([1, 1, 1]),
    )

    reconstruction = np.array([1.0, 1.0, 2.0])

    mapped_reconstructed_operated_data = aa.util.inversion_imaging_numba.mapped_reconstructed_data_via_image_to_pix_unique_from(
        data_to_pix_unique=data_to_pix_unique.astype("int"),
        data_weights=data_weights,
        pix_lengths=pix_lengths.astype("int"),
        reconstruction=reconstruction,
    )

    assert (mapped_reconstructed_operated_data == np.array([1.0, 1.0, 2.0])).all()

    pix_indexes_for_sub_slim_index = np.array(
        [[0], [1], [1], [2], [1], [1], [1], [1], [1], [2], [2], [2]]
    )
    pix_indexes_for_sub_slim_index_sizes = np.ones(shape=(12,)).astype("int")
    pix_weights_for_sub_slim_index = np.ones(shape=(12, 1))

    (
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = aa.util.mapper_numba.data_slim_to_pixelization_unique_from(
        data_pixels=3,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_sizes_for_sub_slim_index=pix_indexes_for_sub_slim_index_sizes,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        pix_pixels=3,
        sub_size=np.array([2, 2, 2]),
    )

    reconstruction = np.array([1.0, 1.0, 2.0])

    mapped_reconstructed_operated_data = aa.util.inversion_imaging_numba.mapped_reconstructed_data_via_image_to_pix_unique_from(
        data_to_pix_unique=data_to_pix_unique.astype("int"),
        data_weights=data_weights,
        pix_lengths=pix_lengths.astype("int"),
        reconstruction=reconstruction,
    )

    assert (mapped_reconstructed_operated_data == np.array([1.25, 1.0, 1.75])).all()


def test__preconditioner_matrix_via_mapping_matrix_from():
    mapping_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    preconditioner_matrix = (
        aa.util.inversion.preconditioner_matrix_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            preconditioner_noise_normalization=1.0,
            regularization_matrix=np.zeros((3, 3)),
        )
    )

    assert (
        preconditioner_matrix
        == np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    ).all()

    preconditioner_matrix = (
        aa.util.inversion.preconditioner_matrix_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            preconditioner_noise_normalization=2.0,
            regularization_matrix=np.zeros((3, 3)),
        )
    )

    assert (
        preconditioner_matrix
        == np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    ).all()

    regularization_matrix = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )

    preconditioner_matrix = (
        aa.util.inversion.preconditioner_matrix_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            preconditioner_noise_normalization=2.0,
            regularization_matrix=regularization_matrix,
        )
    )

    assert (
        preconditioner_matrix
        == np.array([[5.0, 2.0, 3.0], [4.0, 9.0, 6.0], [7.0, 8.0, 13.0]])
    ).all()


def test__reconstruction_positive_only_from__jax_ill_conditioned_grad_is_finite():
    """
    On ill-conditioned curvature matrices the jaxnnls backward pass used to
    return NaN gradients, because the relaxed-KKT solver diverged. Jacobi
    preconditioning inside `reconstruction_positive_only_from` re-parameterises
    the NNLS problem so the solve converges and `jax.value_and_grad` produces
    finite gradients. Skip the test if jax / jaxnnls are not available.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    pytest.importorskip("jaxnnls")

    # A small deliberately ill-conditioned symmetric positive-definite Q,
    # cond(Q) ~ 1e7, which is enough to break the raw jaxnnls backward pass.
    rng = np.random.default_rng(0)
    n = 10
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.logspace(-4, 3, n)
    Q_np = (U * eigs) @ U.T
    Q_np = 0.5 * (Q_np + Q_np.T)
    q_np = rng.standard_normal(n)

    Q = jnp.array(Q_np)
    q = jnp.array(q_np)

    def loss(q_in):
        x = aa.util.inversion.reconstruction_positive_only_from(
            data_vector=q_in, curvature_reg_matrix=Q, xp=jnp,
        )
        return jnp.sum(x)

    value, grad = jax.value_and_grad(loss)(q)

    assert np.isfinite(float(value))
    grad_np = np.array(grad)
    assert np.all(np.isfinite(grad_np)), (
        f"gradient has {np.sum(~np.isfinite(grad_np))} non-finite entries"
    )


def test__reconstruction_positive_only_from__jax_matches_unpreconditioned_primal():
    """
    Jacobi preconditioning is a change of coordinates; the forward primal
    solution must match the raw jaxnnls solve to within solver tolerance for
    a moderately-conditioned problem where the raw solver also converges.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    jaxnnls = pytest.importorskip("jaxnnls")

    rng = np.random.default_rng(1)
    n = 8
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.linspace(0.5, 5.0, n)  # well-conditioned
    Q_np = (U * eigs) @ U.T
    Q_np = 0.5 * (Q_np + Q_np.T)
    q_np = rng.standard_normal(n)

    Q = jnp.array(Q_np)
    q = jnp.array(q_np)

    x_raw = np.array(jaxnnls.solve_nnls_primal(Q, q))
    x_pc = np.array(
        aa.util.inversion.reconstruction_positive_only_from(
            data_vector=q, curvature_reg_matrix=Q, xp=jnp,
        )
    )

    np.testing.assert_allclose(x_pc, x_raw, rtol=1e-6, atol=1e-8)
