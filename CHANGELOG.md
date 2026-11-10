# Changelog

## 0.2.0 — 2026-11-10

Published to PyPI on 2026-11-10. See `docs/release/0.2.0.md` for the tested
environments, the artifact hashes and the post-publication verification.

First public release.

- `PixelTransform`: the `Protocol`-based seam the package is built around. Any
  callable `transform(x_out, y_out) -> (x_in, y_in)` satisfies it
  structurally; `reprojax.core.PixelTransform` is for typing only. Conforming
  transforms are pure array operations, so they work under `jax.jit`,
  `jax.vmap` and `jax.grad`.
- `reproject_interp` and `reproject_interp_core`: bilinear (order 1) and
  nearest (order 0) reprojection, matching `reproject`'s call and return
  shape. `reproject_interp_batched` maps a batch of pre-built transforms.
- `reproject_adaptive` and `reproject_adaptive_core`: the DeForest (2004)
  adaptive anti-aliased algorithm with a `gaussian` or `hann` kernel. The
  Jacobian comes from JAX autodiff rather than finite differences; the search
  window is a fixed maximum size because `vmap` needs static shapes.
- `pixel_map(x_out, y_out, bounds_shape=(ny, nx))`: an optional transform
  capability returning finite coordinates plus a boolean geometric-validity
  mask. Interpolation uses it when present and falls back to the two-result
  callable contract otherwise. It keeps invalid coordinates out of the
  interpolation arithmetic, so gradients stay finite on mixed valid/invalid
  batches.
- `matrix_precision="f64" | "f32"`: the coordinate-precision choice, default
  `"f64"`. `"f64"` requires `jax_enable_x64`; `"f32"` needs no x64 and runs
  the per-pixel path in float32 while composing the celestial rotation in
  float64 on the host. f32 coordinate accuracy is capped at 6e-4 px forward
  and 5e-4 px reverse on the real SPHEREx pair — a coordinate bound, not an
  image-agreement or fp64-equivalence claim.
- Requires `mapprojax>=0.3.0,<0.4`. Runtime dependencies are `numpy>=2.4.6`,
  `jax>=0.10.2`, `astropy>=7.2.0`. `reproject>=0.21` is a **test-only**
  golden reference, pinned to 0.21.0 in `ci/`.
- The sdist ships the package only; `tests/` and its 16 MB of FITS fixtures
  stay in the repository.
- Accuracy evidence against `reproject` 0.21.0: the real SPHEREx pair agrees
  to `rtol=1e-6, atol=0.01` (interp, enforced) and `atol=0.005` (adaptive,
  enforced) over all jointly-valid pixels, with exact footprint equality;
  synthetic pairs agree to `1e-8/1e-9` (interp) and `1e-6/1e-8` (adaptive).
