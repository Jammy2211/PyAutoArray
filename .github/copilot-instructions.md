# Copilot Coding Agent Instructions

You are working on **PyAutoArray**, the low-level data structures and numerical utilities package for the PyAuto ecosystem.

## Key Rules

- Run tests after every change: `python -m pytest test_autoarray/`
- Format code with `black autoarray/`
- All files must use Unix line endings (LF, `\n`)
- Decorated functions (`@to_array`, `@to_grid`, `@to_vector_yx`) must return **raw arrays**, not autoarray wrappers
- The `xp` parameter controls NumPy (`xp=np`) vs JAX (`xp=jnp`) — never import JAX at module level
- If changing public API, clearly document what changed in your PR description — downstream packages depend on this

## Architecture

- `autoarray/structures/` — `Array2D`, `Grid2D`, `Grid2DIrregular`, `VectorYX2D`, decorators
- `autoarray/dataset/` — `Imaging`, `Interferometer` containers
- `autoarray/inversion/` — Pixelization and linear inversion machinery
- `autoarray/operators/` — `Convolver`, over-sampling utilities
- `test_autoarray/` — Test suite

## Sandboxed runs

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest test_autoarray/
```
