# Plan: Remove Visuals Classes and Pass Overlays Directly

## Current State

The codebase is in a *partial* refactoring state. Standalone `plot_array`, `plot_grid`,
`plot_yx`, `plot_inversion_reconstruction` functions already exist in
`autoarray/plot/plots/` and are called by the new-style plotters. However:

- `Visuals1D` and `Visuals2D` wrapper classes still exist
- Every plotter still accepts `visuals_2d` / `visuals_1d` constructor args and stores them
- Helper functions (`_lines_from_visuals`, `_positions_from_visuals`, `_mask_edge_from`,
  `_grid_from_visuals`) bridge old Visuals → new standalone functions
- `MatPlot2D.plot_array/plot_grid/plot_mapper` and `MatPlot1D.plot_yx` still exist and
  `InterferometerPlotter` still calls them directly
- `InversionPlotter.subplot_of_mapper` directly mutates `self.visuals_2d`

## Goal

Remove `Visuals1D`, `Visuals2D`, and `AbstractVisuals` entirely. Each plotter holds its
overlay data as plain attributes and passes them straight to the `plot_*` standalone
functions. Default overlays (e.g. mask derived from `array.mask`) are computed inline.

---

## Steps

### 1. Update `AbstractPlotter` (`abstract_plotters.py`)
- Remove `visuals_1d: Visuals1D` and `visuals_2d: Visuals2D` constructor parameters and
  their default instantiation (`self.visuals_1d = visuals_1d or Visuals1D()`, etc.)
- Remove the imports of `Visuals1D` and `Visuals2D`

### 2. Update each Plotter constructor to accept individual overlay objects

Replace `visuals_2d: Visuals2D = None` with explicit per-overlay kwargs. Plotters store
each overlay as a plain instance attribute (defaulting to `None`).

**`Array2DPlotter`** (`structures/plot/structure_plotters.py`):
```python
def __init__(self, array, mat_plot_2d=None,
             mask=None, origin=None, border=None, grid=None,
             positions=None, lines=None, vectors=None,
             patches=None, fill_region=None, array_overlay=None):
```

**`Grid2DPlotter`**:
```python
def __init__(self, grid, mat_plot_2d=None, lines=None, positions=None):
```

**`YX1DPlotter`**:
```python
def __init__(self, y, x=None, mat_plot_1d=None,
             shaded_region=None, vertical_line=None, points=None, ...):
```

**`MapperPlotter`** (`inversion/plot/mapper_plotters.py`):
```python
def __init__(self, mapper, mat_plot_2d=None,
             lines=None, grid=None, positions=None):
```

**`InversionPlotter`** (`inversion/plot/inversion_plotters.py`):
```python
def __init__(self, inversion, mat_plot_2d=None,
             lines=None, grid=None, positions=None,
             residuals_symmetric_cmap=True):
```

**`ImagingPlotterMeta` / `ImagingPlotter`** (`dataset/plot/imaging_plotters.py`):
```python
def __init__(self, dataset, mat_plot_2d=None,
             mask=None, grid=None, positions=None, lines=None):
```

**`FitImagingPlotterMeta` / `FitImagingPlotter`** (`fit/plot/fit_imaging_plotters.py`):
```python
def __init__(self, fit, mat_plot_2d=None,
             mask=None, grid=None, positions=None, lines=None,
             residuals_symmetric_cmap=True):
```

**`InterferometerPlotter`** (`dataset/plot/interferometer_plotters.py`):
```python
def __init__(self, dataset, mat_plot_1d=None, mat_plot_2d=None, lines=None):
```

### 3. Inline overlay logic inside each plotter's `_plot_*` / `figure_*` methods

Each plotter's internal plot helpers already call the standalone functions. Replace
calls like:
```python
mask=_mask_edge_from(array, self.visuals_2d),
lines=_lines_from_visuals(self.visuals_2d),
```
with direct access to the plotter's own attributes plus inline auto-extraction:
```python
mask=self.mask if self.mask is not None else _auto_mask_edge(array),
lines=self.lines,
```

Where `_auto_mask_edge(array)` is a tiny module-level helper (no Visuals dependency):
```python
def _auto_mask_edge(array):
    """Return edge-pixel (y,x) coords from array.mask, or None."""
    try:
        if not array.mask.is_all_false:
            return np.array(array.mask.derive_grid.edge.array)
    except AttributeError:
        pass
    return None
```

### 4. Fix `InversionPlotter.subplot_of_mapper` — drop the `visuals_2d` mutation

Currently this method does:
```python
self.visuals_2d += Visuals2D(mesh_grid=mapper.image_plane_mesh_grid)
```
Replace by passing `mesh_grid` directly to the specific `figures_2d_of_pixelization`
call that needs it, or by temporarily storing `self.mesh_grid` on the plotter and
checking it in `_plot_array`. The mutation and the `Visuals2D(...)` construction are
both removed.

Similarly remove `self.visuals_2d.indexes = indexes` in `subplot_mappings` — store as
`self._indexes` and pass through.

### 5. Update `InterferometerPlotter.figures_2d` — replace old MatPlot calls

`InterferometerPlotter` still calls `self.mat_plot_2d.plot_array(...)`,
`self.mat_plot_2d.plot_grid(...)`, and `self.mat_plot_1d.plot_yx(...)`.

Replace each with the equivalent standalone function call, deriving `ax`, `output_path`,
`filename`, `fmt` via `_output_for_mat_plot` (which already exists and has no Visuals
dependency).

### 6. Remove `MatPlot2D.plot_array`, `plot_grid`, `plot_mapper` (and private helpers)

Once no caller uses them, delete these methods from `mat_plot/two_d.py`:
- `plot_array`
- `plot_grid`
- `plot_mapper`
- `_plot_rectangular_mapper`
- `_plot_delaunay_mapper`

Remove the `from autoarray.plot.visuals.two_d import Visuals2D` import.

### 7. Remove `MatPlot1D.plot_yx`

Delete the method from `mat_plot/one_d.py` and remove the `Visuals1D` import.

### 8. Remove helper extraction functions from `structure_plotters.py`

Delete (no longer needed):
- `_lines_from_visuals`
- `_positions_from_visuals`
- `_mask_edge_from`
- `_grid_from_visuals`

Keep: `_zoom_array`, `_output_for_mat_plot` (neither depends on Visuals).

### 9. Delete `autoarray/plot/visuals/`

Remove:
- `autoarray/plot/visuals/__init__.py`
- `autoarray/plot/visuals/abstract.py`
- `autoarray/plot/visuals/one_d.py`
- `autoarray/plot/visuals/two_d.py`

### 10. Update `autoarray/plot/__init__.py`

Remove `Visuals1D` and `Visuals2D` exports (lines 45–46).

### 11. Check and update remaining plotters

Read and update:
- `fit/plot/fit_interferometer_plotters.py`
- `fit/plot/fit_vector_yx_plotters.py`

Both import `Visuals1D`/`Visuals2D`; apply the same pattern as above.

### 12. Run full test suite

```bash
python -m pytest test_autoarray/ -q --tb=short
```

Fix any failures before committing.

---

## Summary of files changed

| File | Change |
|------|--------|
| `autoarray/plot/abstract_plotters.py` | Remove `visuals_1d`, `visuals_2d` |
| `autoarray/plot/mat_plot/one_d.py` | Remove `plot_yx`, remove Visuals1D import |
| `autoarray/plot/mat_plot/two_d.py` | Remove `plot_array/grid/mapper` methods, remove Visuals2D import |
| `autoarray/plot/visuals/` | **Delete entire directory** |
| `autoarray/plot/__init__.py` | Remove Visuals exports |
| `autoarray/structures/plot/structure_plotters.py` | Replace visuals args with individual kwargs; remove helper functions |
| `autoarray/inversion/plot/mapper_plotters.py` | Replace visuals args with individual kwargs |
| `autoarray/inversion/plot/inversion_plotters.py` | Replace visuals args; fix subplot_of_mapper mutation |
| `autoarray/dataset/plot/imaging_plotters.py` | Replace visuals args with individual kwargs |
| `autoarray/dataset/plot/interferometer_plotters.py` | Replace visuals args; replace old MatPlot calls |
| `autoarray/fit/plot/fit_imaging_plotters.py` | Replace visuals args with individual kwargs |
| `autoarray/fit/plot/fit_interferometer_plotters.py` | Replace visuals args |
| `autoarray/fit/plot/fit_vector_yx_plotters.py` | Replace visuals args |
