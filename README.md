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

Early development. See the [issue tracker](https://github.com/leejjoon/reprojax/issues)
for the current milestone plan (spec in #1).

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

## Development

```bash
pip install -e ".[test]"
pytest tests/
```
