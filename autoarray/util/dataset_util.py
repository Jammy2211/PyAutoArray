import os
import shutil


def should_simulate(dataset_path):
    """
    Returns True if the dataset at ``dataset_path`` needs to be simulated.

    When ``PYAUTO_WORKSPACE_SMALL_DATASETS=1`` is active, any existing dataset
    is deleted so the simulator re-creates it at the reduced resolution.  This
    avoids shape mismatches between full-resolution FITS files on disk and the
    15x15 mask/grid cap applied by the env var.

    Use this as a drop-in replacement for ``not path.exists(dataset_path)`` in
    the workspace auto-simulation pattern::

        if aa.util.dataset.should_simulate(dataset_path):
            subprocess.run([sys.executable, "scripts/.../simulator.py"], check=True)
    """
    if os.environ.get("PYAUTO_WORKSPACE_SMALL_DATASETS") == "1":
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)

    return not os.path.exists(dataset_path)
