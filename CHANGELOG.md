# Changelog

## 0.3.1 — 2026-11-20

### Added

- **ZEA, STG and ARC projections**, through mapprojax 0.5.0. reprojax builds
its kernels with `mapprojax.from_astropy`, so these headers -- previously
refused -- now reproject end to end: TAN<->ZEA in both directions and both
matrix precisions agree with the astropy round trip within the existing
tolerances (1e-4 px f64, 5e-3 px f32), and a ZEA image through
`reproject_interp` matches the original `reproject`. The parametrised
zenithals (AZP, SZP, ZPN, AIR) are still refused.

### Changed

- Requires `mapprojax[astropy]>=0.5.0,<0.6` (was `>=0.4.0,<0.5`).

## 0.3.0 — 2026-11-16

Host-side WCS handling now goes through
`mapprojax.from_astropy`, which fixes three latent defects and changes what
is refused.

### Fixed

- **Frames at different FK5 equinoxes were treated as identical.** Frame
equivalence was decided by comparing frame *names*, so two FK5 headers with
`EQUINOX=2000` and `EQUINOX=1950` returned the identity rotation and dropped
a 50-year precession: 2234.6 px at 1"/px over a 512" field. The rotation is
now computed whenever the frames are not `is_equivalent_frame`, and checked
to be orthonormal with determinant +1. ICRS<->FK5(J2000) already rotated and
still does; that pair is now covered by a regression test.
- **A `preserve_units=True` header was read as if its units were degrees.**
With `CUNIT='arcsec'` and `CRVAL=(10, 20)`, the old extractor fed arcsec
straight into the kernel, so the reference pixel landed **22.2687 deg** from
the origin on the sky where astropy puts it **0.006211 deg** away -- a
**22.2625 deg** disagreement, ≈80,145 px at 1"/px. Celestial units are now
converted per axis; the arcsec and degree encodings of the same header
agree to better than 1e-4 px, and to the astropy pixel oracle.
- **TPV headers were silently flattened to TAN.** Astropy reports
`ctype == 'RA---TAN'`, `has_distortion == False` and no PV cards for a TPV
header while still applying the distortion, and the transform cache keyed on
those raw attributes, so a TPV WCS shared a cache entry with a plain TAN
WCS. TPV is now refused.
- **A cube reported mapprojax's internal error.** A 3-D `output_projection`
now refuses with `pass a two-axis celestial WCS, e.g. wcs.celestial`, for a
`Header` and a `WCS`, before any output grid is allocated. (reprojax's own
guard; the message is not `from_astropy`'s.)

### Changed

- **Refusals raise `ValueError`, not `NotImplementedError`.** Projection and
FITS-convention refusals come from `mapprojax.from_astropy` and carry its
message. reprojax raises its own `ValueError` for the frame allow-list, a
non-rigid rotation, a non-2-D WCS, an unsupported `order` or `kernel`, an
uninferable `shape_out`, and a bad `matrix_precision`. A caller catching
`NotImplementedError` must catch `ValueError` instead. A malformed header can
still fail earlier, inside `astropy.wcs.WCS`, before reprojax runs.
- **New refusals: inputs 0.2.0 returned a result for.**
  - **any PV card on TAN**, including the identity `PV1_1 = 0.0` — 0.2.0
    ignored PV entirely;
  - **TAN at `CRVAL2 = +90`**, because wcslib's default there is
    `LONPOLE = 0`. Note the asymmetry: `CRVAL2 = -90` is still accepted,
    since its default is 180;
  - **lookup tables** (`cpdis1`/`cpdis2`, `det2im1`/`det2im2`);
  - **TPV, and TPD carried as `CQDIS`/`DQ` cards.** Astropy reports both as
    plain `RA---TAN` with `has_distortion == False` and no PV cards, so 0.2.0
    accepted them and silently dropped the distortion;
  - **FK4 and FK4-NO-E headers**. FK4's E-terms are not a rotation, so it is
    refused rather than approximated; FK4-NO-E is a rigid rotation and could
    be allow-listed later. The ecliptic axes (`ELON`/`ELAT`) are refused
    too: astropy resolves them to ICRS, which would be silently wrong.
- **Already refused by 0.2.0, now as `ValueError` with `from_astropy`'s
message.** Unsupported projection codes astropy can still construct — AZP, ZEA
and the like. 0.2.0 raised `NotImplementedError` for these; the type changed,
nothing else did.
- **Unreachable in both versions.** A literal `RA---TPD` CTYPE fails inside
`astropy.wcs.WCS` (`InconsistentAxisTypesError`), and `-TAB` without its
`WCS-TAB` HDU raises `MemoryError` there. Both die at `WCS(header)`, before
reprojax runs.
- **Dependency floor**: `mapprojax[astropy]>=0.4.0,<0.5`. Both CI
  constraint files move to `mapprojax==0.4.0`, and the extra's
  `astropy>=7.2.0` floor is already satisfied on both stacks.

### Internal

- `_astropy_wcs` now owns one validated conversion seam
(`validated_wcs`), an explicit frame allow-list (`frame_descriptor`) and a
validated rotation (`_rotation_from_descriptors`); `projection_code`,
`_wcs_fingerprint` and `_PROVIDERS` are gone.
- The transform cache key is built from the *validated* leaves plus a frame
  descriptor, after validation rather than before, so an unsupported input
  cannot be served a cached transform. Warm `build_transform` costs one
  eager conversion per WCS (~0.4 ms without SIP, ~1.5 ms with order-3 SIP).

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
