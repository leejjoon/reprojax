# reprojax

A JAX-native, differentiable port of astropy's [`reproject`](https://github.com/astropy/reproject) package.

reprojax reprojects astronomical images between WCS frames on top of
[mapprojax](https://github.com/leejjoon/mapprojax), a JAX port of CDS's
`mapproj` crate providing fast, differentiable pixel↔sky transforms. The goal
is GPU-accelerated, batchable, gradient-friendly reprojection, validated
against the original `reproject` package as the accuracy reference.

reprojax is a JAX port of `reproject`'s algorithms and API shape; see
`LICENSE` for `reproject`'s BSD-3-Clause license, carried alongside
reprojax's own.

## Status

0.2.0, first public release. The validated scope is deliberate:

- **Projections**: TAN and SIN, with SIP distortion. Anything else raises
  `NotImplementedError`.
- **Algorithms**: bilinear/nearest interpolation (`reproject_interp`) and the
  DeForest (2004) adaptive anti-aliased algorithm (`reproject_adaptive`).
- **Reference**: `reproject` 0.21.0 is the golden accuracy reference. It is a
  **test-time** dependency only; nothing in `reprojax` imports it at runtime.
- **Not included**: cubic interpolation, flux-conserving resampling, mosaic
  coadds, filename/HDU/`NDData` input forms. See the
  [issue tracker](https://github.com/leejjoon/reprojax/issues) for the plan.

## Precision: read this before your first call

`matrix_precision` selects the transform's coordinate precision, and the
**default is `"f64"`, which requires JAX x64**. Without it your first call
raises:

```
ValueError: precision='f64' requires float64. Call
jax.config.update('jax_enable_x64', True) before constructing it.
```

Two valid choices:

| Mode | Requirement | Use it for |
| --- | --- | --- |
| `"f64"` (default) | `jax.config.update("jax_enable_x64", True)` before importing/creating anything JAX | Full-precision work, astrometry, gradients with respect to WCS parameters |
| `"f32"` | none — works with x64 off | GPU throughput; the celestial rotation is still composed in float64 on the host, only the per-pixel path is float32 |

`"f64"` is the accurate default. `"f32"` is the fast path and is documented
in terms of its own measured caps, not as an equivalent to `"f64"` (below).

## Quick start

No data files needed. This runs exactly as written, from an installed package:

```python
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # required by the default matrix_precision="f64"

from astropy.io import fits
from astropy.wcs import WCS
from reprojax import reproject_interp

ny_in = nx_in = 64
hdr_in = fits.Header()
hdr_in["NAXIS"] = 2
hdr_in["NAXIS1"] = hdr_in["NAXIS2"] = ny_in
hdr_in["CTYPE1"], hdr_in["CTYPE2"] = "RA---TAN", "DEC--TAN"
hdr_in["CRPIX1"] = hdr_in["CRPIX2"] = ny_in / 2 + 0.5
hdr_in["CRVAL1"], hdr_in["CRVAL2"] = 210.0, 54.0
hdr_in["CDELT1"], hdr_in["CDELT2"] = -1e-3, 1e-3

y, x = np.mgrid[:ny_in, :nx_in]
data = np.exp(-(((x - 32) ** 2 + (y - 32) ** 2) / (2 * 6.0 ** 2))).astype(np.float32)

hdr_out = hdr_in.copy()
hdr_out["CRVAL1"] = 210.02  # shift the output grid

array, footprint = reproject_interp((data, WCS(hdr_in)), hdr_out, shape_out=(40, 40))

assert array.shape == footprint.shape == (40, 40)
assert np.isfinite(array).any()
print(float(np.nanmax(array)))
```

`array` holds the resampled image (NaN where no valid input pixel was
covered) and `footprint` is 1.0 where a valid input contributed, 0.0
otherwise. Adding `matrix_precision="f32"` runs the same call without x64.

## Architecture

- `reprojax.core` — the `PixelTransform` contract: a callable
  `transform(x_out, y_out) -> (x_in, y_in)` mapping output-image pixel
  coordinates to input-image pixel coordinates. This is the seam the rest of
  the package is built around.
- `reprojax.transform` — builds a `PixelTransform` from a pair of astropy
  WCS objects, using mapprojax for the projection math and astropy for
  celestial frame rotation.
- Mapprojax-backed transforms also expose an optional
  `transform.pixel_map(x_out, y_out, bounds_shape=(ny, nx))` capability. It
  supplies finite coordinates and geometric validity to interpolation while
  the two-result callable contract remains available for generic providers.

## Accuracy

Two kinds of numbers appear here, and they are labelled: an **observation**
is what the two implementations measured on a case, an **enforced tolerance**
is what the test suite requires.

Real SPHEREx pair (2040² ICRS TAN+SIP exposure onto the 1201² Galactic TAN
template, compared against `reproject` 0.21.0):

| Algorithm | Observation | Enforced |
| --- | --- | --- |
| `reproject_interp` | 1,345,444 of 1,345,460 jointly-valid pixels agree to `rtol <= 1e-8` | `rtol=1e-6, atol=0.01` over all jointly-valid pixels, plus exact footprint equality |
| `reproject_adaptive` | median disagreement ~`4e-8`; 1,334,193 of 1,334,200 jointly-valid pixels form the bulk | `rtol=1e-6, atol=0.005`, plus exact footprint equality |

In both algorithms the pixels outside the stated bulk are 16 (interp) and 7
(adaptive) pixels sitting at one real, un-flagged data spike: this fixture
drops the `FLAGS` extension that would normally mask it, so a sub-micropixel
position difference next to a ~10⁴-per-pixel local gradient is amplified into
a value difference of up to ~0.01 (interp) / ~0.003 (adaptive). The fixtures
are described in [`tests/data/README.md`](tests/data/README.md).

Synthetic TAN / TAN-SIP pairs: interpolated images agree within enforced
`rtol=1e-8, atol=1e-9`; adaptive within enforced `rtol=1e-6, atol=1e-8`.

`matrix_precision="f32"` has **enforced coordinate caps** of `6e-4` px
(forward) and `5e-4` px (reverse) per direction on the real pair. These bound
the coordinate map only — they are not image-agreement bounds, and they are
not a claim of float64 equivalence.

On a Strix Halo host (ROCm device, JAX 0.11.1 + rocm7 plugins 0.11.1),
batched and `jax.jit`-compiled reprojax reprojects 8 SPHEREx-scale
(2040² → 1201²) exposures in 0.026 s. The looped CPU `reproject_interp`
baseline for the same work measured 4.4–7.9 s across runs on this host, i.e.
roughly 170–300×. The reprojax figure is stable; the baseline is what varies.
This is a cross-backend batched-vs-looped comparison, not a like-for-like
backend comparison, and the CPU skips the batched GPU-only tests.

## Development

```bash
python -m pip install -e ".[test]" -c ci/latest.txt
python -m pytest tests/
```

Using a `ci/` constraints file matters: the golden reference is pinned to
`reproject==0.21.0` there, and the suite asserts the installed version. Use
`ci/minimum.txt` for the oldest supported stack (Python 3.11). If a
constraint conflicts with your environment, install `.[test]` without it and
expect the version guard to tell you which pin is off.

The test suite needs `tests/data/`, which is 16 MB of real SPHEREx
exposures and is therefore not shipped in the sdist. Install from a checkout
when you want to run the tests.

## License

BSD-3-Clause. The SIP and reprojection algorithms are ported from
[`reproject`](https://github.com/astropy/reproject); its license is carried in
[`LICENSE`](LICENSE) alongside reprojax's own.

## Test data

The integration tests use a real SPHEREx exposure and a Galactic template
header. Only the exposure is mission data: NASA's Science Mission
Directorate releases mission data into the public domain (SPD-41), while the
template header is project metadata under reprojax's own licence. If you
publish work that uses the exposure, include the acknowledgement IRSA
requests:

> This publication makes use of data products from the Spectro-Photometer for
the History of the Universe, Epoch of Reionization and Ices Explorer
> (SPHEREx), which is a joint project of the Jet Propulsion Laboratory and the
> California Institute of Technology, and is funded by the National
> Aeronautics and Space Administration.

Provenance, checksums and the citation DOIs are in
[`tests/data/README.md`](tests/data/README.md).
