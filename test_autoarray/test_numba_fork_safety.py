"""No numba parallel threads inside multiprocessing workers (``numba_util.parallel_kernel_allowed``).

PyAutoFit evaluates the likelihood once in the parent, then forks (or spawns) search workers that
evaluate it again. On numba's ``omp`` threading layer a forked child launching a ``prange`` kernel
after the parent used the pool is terminated, so workers run each kernel's serial compilation.
"""

import multiprocessing
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numba")

from autoarray import numba_util
from autoarray.inversion.mesh.interpolator import sibson
from autoarray.inversion.mesh.interpolator.delaunay import (
    pix_indexes_delaunay_walk_from,
    scipy_delaunay_tri_only,
)

_SCRIPT = textwrap.dedent(
    """
    import multiprocessing as mp
    import sys

    import numpy as np
    import numba

    from autoarray import numba_util
    from autoarray.inversion.mesh.interpolator import sibson
    from autoarray.inversion.mesh.interpolator.delaunay import (
        pix_indexes_delaunay_walk_from,
        scipy_delaunay_tri_only,
    )

    MODE = sys.argv[1]
    TIMEOUT = float(sys.argv[2])

    if MODE == "control":
        # Undo the fix in every process: children launch the parallel kernel too.
        numba_util.parallel_kernel_allowed = lambda: True

    rng = np.random.default_rng(1)
    points = rng.uniform(-1.0, 1.0, size=(60, 2))
    query = rng.uniform(-1.1, 1.1, size=(200, 2))
    simplices, neighbors, vertex_simplex = scipy_delaunay_tri_only(points)
    mappings, simplex_indexes = pix_indexes_delaunay_walk_from(
        query_points=query,
        points=points,
        simplices_padded=simplices,
        simplex_neighbors=neighbors,
        vertex_simplex=vertex_simplex,
        xp=np,
        return_simplex_indexes=True,
    )
    KWARGS = dict(
        query_points=query,
        points=points,
        simplices_padded=simplices,
        simplex_neighbors=neighbors,
        simplex_indexes=simplex_indexes,
        outside_fallback_indexes=mappings[:, 0],
        circumcircles=sibson.delaunay_circumcircles_from(points, simplices, xp=np),
        max_cavity_triangles=32,
        max_neighbors=32,
    )


    def threading_layer():
        try:
            return numba.threading_layer()
        except ValueError:  # no parallel region launched in this process
            return None


    def work(_):
        return sibson._sibson_mappings_weights_numba(**KWARGS), threading_layer()


    def check(results, parent, fresh_process):
        for outputs, layer in results:
            # A forked child inherits the parent's record of its layer; a spawned child
            # running the serial kernel never initialises one.
            if fresh_process:
                assert layer is None, layer
            for expected, actual in zip(parent, outputs):
                np.testing.assert_array_equal(expected, actual)


    if __name__ == "__main__":
        try:
            parent = sibson._sibson_mappings_weights_numba(**KWARGS)
        except ValueError as exc:  # requested threading layer unavailable
            print("LAYER_UNAVAILABLE", exc, flush=True)
            sys.exit(0)
        print("LAYER", threading_layer(), flush=True)

        with mp.get_context("fork").Pool(2) as pool:
            check(pool.map_async(work, range(4)).get(timeout=TIMEOUT), parent, False)
        print("FORK_OK", flush=True)

        with mp.get_context("spawn").Pool(2) as pool:
            check(pool.map_async(work, range(2)).get(timeout=TIMEOUT), parent, True)
        print("SPAWN_OK", flush=True)
    """
)


def _run_script(tmp_path, mode, timeout, layer="omp"):
    script = tmp_path / "fork_sibson.py"
    script.write_text(_SCRIPT)

    env = dict(os.environ)
    env.pop("NUMBA_NUM_THREADS", None)
    env.pop("NUMBA_THREADING_LAYER", None)
    if layer is not None:
        env["NUMBA_THREADING_LAYER"] = layer
    # The subprocess must import this checkout's autoarray, whatever is installed.
    package_root = str(Path(numba_util.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        [package_root] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )

    return subprocess.run(
        [sys.executable, str(script), mode, str(timeout)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=4 * timeout + 120,
    )


needs_fork = pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="needs the fork start method",
)


@needs_fork
def test__worker_processes_run_sibson_kernel_after_parent(tmp_path):
    """The RAL failure condition: parallel kernel on ``omp`` in the parent, then a fork pool."""
    result = _run_script(tmp_path, mode="dispatch", timeout=60)
    if "LAYER_UNAVAILABLE" in result.stdout:
        result = _run_script(tmp_path, mode="dispatch", timeout=60, layer=None)

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "FORK_OK" in result.stdout, output
    assert "SPAWN_OK" in result.stdout, output


@needs_fork
def test__control__parallel_kernel_in_forked_children_dies_on_omp(tmp_path):
    """With the predicate forced True the same script fails, so the test above is not vacuous."""
    result = _run_script(tmp_path, mode="control", timeout=10)

    if "LAYER_UNAVAILABLE" in result.stdout:
        pytest.skip("numba omp threading layer unavailable")

    output = result.stdout + result.stderr
    assert "LAYER omp" in result.stdout, output
    assert result.returncode != 0, output
    assert "FORK_OK" not in result.stdout, output


def test__sibson_serial_kernel_equals_parallel_kernel():
    rng = np.random.default_rng(3)
    points = rng.uniform(-1.0, 1.0, size=(80, 2))
    query = rng.uniform(-1.2, 1.2, size=(300, 2))
    simplices, neighbors, vertex_simplex = scipy_delaunay_tri_only(points)
    mappings, simplex_indexes = pix_indexes_delaunay_walk_from(
        query_points=query,
        points=points,
        simplices_padded=simplices,
        simplex_neighbors=neighbors,
        vertex_simplex=vertex_simplex,
        xp=np,
        return_simplex_indexes=True,
    )
    centres, radii_squared, valid = sibson.delaunay_circumcircles_from(
        points, simplices, xp=np
    )
    args = (
        query,
        np.asarray(simplex_indexes, dtype=np.int64),
        np.asarray(mappings[:, 0], dtype=np.int64),
        points,
        np.asarray(simplices, dtype=np.int64),
        np.asarray(neighbors, dtype=np.int64),
        np.asarray(centres, dtype=np.float64),
        np.asarray(radii_squared, dtype=np.float64),
        np.asarray(valid, dtype=np.bool_),
        32,
        32,
    )

    parallel = sibson.sibson_numba_kernel(parallel=True)(*args)
    serial = sibson.sibson_numba_kernel(parallel=False)(*args)

    assert sibson.sibson_numba_kernel(parallel=False) is not sibson.sibson_numba_kernel(
        parallel=True
    )
    for expected, actual in zip(parallel, serial):
        assert expected.dtype == actual.dtype
        np.testing.assert_array_equal(expected, actual)


@pytest.fixture(name="predicate")
def make_predicate(monkeypatch):
    monkeypatch.setattr(numba_util, "_parallel_pid", None)
    monkeypatch.setattr(numba_util.multiprocessing, "parent_process", lambda: None)
    return monkeypatch


def test__predicate__main_process_allowed_and_recorded(predicate):
    assert numba_util.parallel_kernel_allowed() is True
    assert numba_util.parallel_kernel_allowed() is True
    assert numba_util._parallel_pid == os.getpid()


def test__predicate__multiprocessing_child_disallowed(predicate):
    predicate.setattr(numba_util.multiprocessing, "parent_process", lambda: object())

    assert numba_util.parallel_kernel_allowed() is False


def test__predicate__forked_pid_disallowed(predicate):
    assert numba_util.parallel_kernel_allowed() is True

    predicate.setattr(os, "getpid", lambda: -12345)

    assert numba_util.parallel_kernel_allowed() is False


def test__function_copy__renames_without_changing_behaviour():
    def body(x, y=2):
        return x * y

    copy = numba_util.function_copy(body, "renamed")

    assert copy is not body
    assert copy.__name__ == "renamed"
    assert copy.__qualname__.endswith(".renamed")
    assert copy(3) == body(3) == 6


def test__direct_conv_parallel_route_off_in_worker_processes(monkeypatch):
    from autoarray.inversion.inversion.interferometer_numba import sparse

    class Config:
        def __getitem__(self, key):
            return {"numba": {"parallel": True}} if key == "general" else {}

    monkeypatch.setattr(sparse.conf, "instance", Config())

    monkeypatch.setattr(numba_util, "parallel_kernel_allowed", lambda: True)
    assert sparse._numba_parallel() is True

    monkeypatch.setattr(numba_util, "parallel_kernel_allowed", lambda: False)
    assert sparse._numba_parallel() is False
