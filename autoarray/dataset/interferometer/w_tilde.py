import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from autoarray.dataset.abstract.w_tilde import AbstractWTilde
from autoarray.mask.mask_2d import Mask2D


def _bbox_from_mask(mask_bool: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Return bbox (y_min, y_max, x_min, x_max) of the unmasked region.
    mask_bool: True=masked, False=unmasked
    """
    ys, xs = np.where(~mask_bool)
    if ys.size == 0:
        raise ValueError("Mask has no unmasked pixels; cannot compute bbox.")
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _mask_sha256(mask_bool: np.ndarray) -> str:
    """
    Stable hash of the full boolean mask content (not just bbox).
    """
    # Ensure contiguous, stable dtype
    arr = np.ascontiguousarray(mask_bool.astype(np.uint8))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _as_pixel_scales_tuple(pixel_scales) -> Tuple[float, float]:
    """
    Normalize pixel_scales to a stable 2-tuple of float.
    Works with AutoArray pixel_scales objects or raw tuples.
    """
    try:
        # autoarray typically stores pixel_scales as tuple-like
        return (float(pixel_scales[0]), float(pixel_scales[1]))
    except Exception:
        # fallback: treat as scalar
        s = float(pixel_scales)
        return (s, s)


def _np_float_tuple(x) -> Tuple[float, float]:
    return (float(x[0]), float(x[1]))


def curvature_preload_metadata_from(real_space_mask) -> Dict[str, Any]:
    """
    Build the minimal metadata required to decide whether a stored curvature_preload
    can be reused for the current WTildeInterferometer instance.

    The preload depends on:
      - the *rectangular FFT grid extent* used for offset evaluation (bbox / extent)
      - pixel scales (radians per pixel)
      - (usually) the exact mask shape and content (recommended to hash)

    Returns
    -------
    dict
        JSON-serializable metadata.
    """
    mask_bool = np.asarray(real_space_mask, dtype=bool)
    y_min, y_max, x_min, x_max = _bbox_from_mask(mask_bool)
    y_extent = y_max - y_min + 1
    x_extent = x_max - x_min + 1

    pixel_scales = _as_pixel_scales_tuple(real_space_mask.pixel_scales)

    meta = {
        "format": "autoarray.w_tilde.curvature_preload.v1",
        "mask_shape": tuple(mask_bool.shape),
        "pixel_scales": pixel_scales,
        "bbox_unmasked": (y_min, y_max, x_min, x_max),
        "rect_shape": (y_extent, x_extent),
        # full-content hash: safest way to prevent accidental reuse
        "mask_sha256": _mask_sha256(mask_bool),
    }
    return meta


def is_preload_metadata_compatible(
    real_space_mask,
    meta: Dict[str, Any],
    *,
    require_mask_hash: bool = True,
    atol: float = 0.0,
) -> Tuple[bool, str]:
    """
    Compare loaded metadata against current instance.

    Parameters
    ----------
    meta
        Metadata dict loaded from disk.
    require_mask_hash
        If True, require the full mask sha256 to match (safest).
        If False, only check bbox + shape + pixel scales.
    atol
        Tolerances for pixel scale comparisons (normally exact is fine
        because these are configuration constants, but tolerances allow
        for tiny float repr differences).

    Returns
    -------
    (ok, reason)
        ok: bool, True if compatible
        reason: str, human-readable mismatch reason if not ok.
    """
    current = curvature_preload_metadata_from(real_space_mask=real_space_mask)

    # 1) format version
    if meta.get("format") != current["format"]:
        return False, f"format mismatch: {meta.get('format')} != {current['format']}"

    # 2) mask shape
    if tuple(meta.get("mask_shape", ())) != tuple(current["mask_shape"]):
        return (
            False,
            f"mask_shape mismatch: {meta.get('mask_shape')} != {current['mask_shape']}",
        )

    # 3) pixel scales
    ps_saved = _np_float_tuple(meta.get("pixel_scales", (np.nan, np.nan)))
    ps_curr = _np_float_tuple(current["pixel_scales"])

    if not (
        np.isclose(ps_saved[0], ps_curr[0], atol=atol)
        and np.isclose(ps_saved[1], ps_curr[1], atol=atol)
    ):
        return False, f"pixel_scales mismatch: {ps_saved} != {ps_curr}"

    # 4) bbox / rect shape
    if tuple(meta.get("bbox_unmasked", ())) != tuple(current["bbox_unmasked"]):
        return (
            False,
            f"bbox_unmasked mismatch: {meta.get('bbox_unmasked')} != {current['bbox_unmasked']}",
        )

    if tuple(meta.get("rect_shape", ())) != tuple(current["rect_shape"]):
        return (
            False,
            f"rect_shape mismatch: {meta.get('rect_shape')} != {current['rect_shape']}",
        )

    # 5) full mask hash (optional but recommended)
    if require_mask_hash:
        if meta.get("mask_sha256") != current["mask_sha256"]:
            return False, "mask_sha256 mismatch (mask content differs)"

    return True, "ok"


def load_curvature_preload_if_compatible(
    file: Union[str, Path],
    real_space_mask,
    *,
    require_mask_hash: bool = True,
) -> Optional[np.ndarray]:
    """
    Load a saved curvature_preload if (and only if) it is compatible with the current mask geometry.

    Parameters
    ----------
    file
        Path to a previously saved NPZ.
    require_mask_hash
        If True, require the full mask content hash to match (safest).
        If False, only bbox + shape + pixel scales are checked.

    Returns
    -------
    np.ndarray
        The loaded curvature_preload if compatible, otherwise raises ValueError.
    """
    file = Path(file)
    if file.suffix.lower() != ".npz":
        file = file.with_suffix(".npz")

    if not file.exists():
        raise FileNotFoundError(str(file))

    with np.load(file, allow_pickle=False) as npz:
        if "curvature_preload" not in npz or "meta_json" not in npz:
            msg = f"File does not contain required fields: {file}"
            raise ValueError(msg)

        meta_json = str(npz["meta_json"].item())
        meta = json.loads(meta_json)

        ok, reason = is_preload_metadata_compatible(
            meta=meta,
            real_space_mask=real_space_mask,
            require_mask_hash=require_mask_hash,
            atol=1.0e-8,
        )

        if not ok:
            raise ValueError(f"curvature_preload incompatible: {reason}")

        return np.asarray(npz["curvature_preload"])


class WTildeInterferometer(AbstractWTilde):
    def __init__(
        self,
        curvature_preload: np.ndarray,
        dirty_image: np.ndarray,
        real_space_mask: Mask2D,
        batch_size: int = 128,
    ):
        """
        Packages together all derived data quantities necessary to fit `Interferometer` data using an ` Inversion` via
        the w_tilde formalism.

        The w_tilde formalism performs linear algebra formalism in a way that speeds up the construction of  the
        simultaneous linear equations by bypassing the construction of a `mapping_matrix` and precomputing the
        Fourier transform operations performed using the interferometer's `uv_wavelengths`.

        Parameters
        ----------
        w_matrix
            The w_tilde matrix used by the w-tilde formalism to construct the data vector and
            curvature matrix during an inversion efficiently..
        curvature_preload
            A matrix which uses the interferometer `uv_wavelengths` to preload as much of the computation of the
            curvature matrix as possible.
        dirty_image
            The real-space image of the visibilities computed via the transform, which is used to construct the
            curvature matrix.
        real_space_mask
            The 2D mask in real-space defining the area where the interferometer data's visibilities are observing
            a signal.
        batch_size
            The size of batches used to compute the w-tilde curvature matrix via FFT-based convolution,
            which can be reduced to produce lower memory usage at the cost of speed.
        """
        super().__init__(
            curvature_preload=curvature_preload,
        )

        self.dirty_image = dirty_image
        self.real_space_mask = real_space_mask

        from autoarray.inversion.inversion.interferometer import (
            inversion_interferometer_util,
        )

        self.fft_state = inversion_interferometer_util.w_tilde_fft_state_from(
            curvature_preload=self.curvature_preload, batch_size=batch_size
        )

    @property
    def mask_rectangular_w_tilde(self) -> np.ndarray:
        """
        Returns a rectangular boolean mask that tightly bounds the unmasked region
        of the interferometer mask.

        This rectangular mask is used for computing the W-tilde curvature matrix
        via FFT-based convolution, which requires a full rectangular grid.

        Pixels outside the bounding box of the original mask are set to True
        (masked), and pixels inside are False (unmasked).

        Returns
        -------
        np.ndarray
            Boolean mask of shape (Ny, Nx), where False denotes unmasked pixels.
        """
        mask = self.real_space_mask

        ys, xs = np.where(~mask)

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        rect_mask = np.ones(mask.shape, dtype=bool)
        rect_mask[y_min : y_max + 1, x_min : x_max + 1] = False

        return rect_mask

    @property
    def rect_index_for_mask_index(self) -> np.ndarray:
        """
        Mapping from masked-grid pixel indices to rectangular-grid pixel indices.

        This array enables extraction of a curvature matrix computed on a full
        rectangular grid back to the original masked grid.

        If:
            - C_rect is the curvature matrix computed on the rectangular grid
            - idx = rect_index_for_mask_index

        then the masked curvature matrix is:
            C_mask = C_rect[idx[:, None], idx[None, :]]

        Returns
        -------
        np.ndarray
            Array of shape (N_masked_pixels,), where each entry gives the
            corresponding index in the rectangular grid (row-major order).
        """
        mask = self.real_space_mask
        rect_mask = self.mask_rectangular_w_tilde

        # Bounding box of the rectangular region
        ys, xs = np.where(~rect_mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        rect_width = x_max - x_min + 1

        # Coordinates of unmasked pixels in the original mask (slim order)
        mask_ys, mask_xs = np.where(~mask)

        # Convert (y, x) → rectangular flat index
        rect_indices = ((mask_ys - y_min) * rect_width + (mask_xs - x_min)).astype(
            np.int32
        )

        return rect_indices

    def save_curvature_preload(
        self,
        file: Union[str, Path],
        *,
        overwrite: bool = False,
    ) -> Path:
        """
        Save curvature_preload plus enough metadata to ensure it is only reused when safe.

        Uses NPZ so we can store:
          - curvature_preload (array)
          - meta_json (string)

        Parameters
        ----------
        file
            Path to save to. Recommended suffix: ".npz".
            If you pass ".npy", we will still save an ".npz" next to it.
        overwrite
            If False and the file exists, raise FileExistsError.

        Returns
        -------
        Path
            The path actually written (will end with ".npz").
        """
        file = Path(file)

        # Force .npz (storing metadata safely)
        if file.suffix.lower() != ".npz":
            file = file.with_suffix(".npz")

        if file.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {file}")

        meta = curvature_preload_metadata_from(self.real_space_mask)

        meta_json = json.dumps(meta, sort_keys=True)

        np.savez_compressed(
            file,
            curvature_preload=np.asarray(self.curvature_preload),
            meta_json=np.asarray(meta_json),
        )
        return file
