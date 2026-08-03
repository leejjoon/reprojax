"""Nearest/bilinear reprojection: the pure-JAX core and the reproject-shaped wrapper.

The core (:func:`reproject_interp_core`) consumes a :class:`~reprojax.core.PixelTransform`
and is a pure function of arrays — jit/vmap/grad-safe. The wrapper
(:func:`reproject_interp`) does the astropy-facing work (header/WCS parsing,
transform composition) and matches ``reproject``'s call/return shape.
"""

import jax.numpy as jnp
from astropy.io.fits import Header
from astropy.wcs import WCS
from jax.scipy.ndimage import map_coordinates

from .transform import build_transform

SUPPORTED_ORDERS = (0, 1)

# Below this coverage, at least one interpolation neighbor was outside the
# input array and contributed through `cval`. The threshold sits far below
# any real partial-coverage value (>= 1/(order+2)) and far above float32
# rounding noise, so it only ever separates "fully covered" from "not".
_COVERAGE_VALID_THRESHOLD = 1e-5


def _clip_border_pixels(coord, size):
    """Snap coordinates in the outer half-pixel border band to the border
    pixel's center index.

    ``map_coordinates`` (like plain scipy) treats pixel values as defined
    only at integer pixel centers, so a coordinate in the outer half of the
    border pixel is technically "outside" the defined domain even though it
    is still within that pixel. ``reproject`` corrects for this with the
    same clip (see ``reproject._array_utils._clip_coords``); we match it so
    our footprints agree with reproject's at the image edges.
    """
    lower = (coord < 0) & (coord >= -0.5)
    upper = (coord < size - 0.5) & (coord >= size - 1)
    coord = jnp.where(lower, 0.0, coord)
    coord = jnp.where(upper, size - 1.0, coord)
    return coord


def reproject_interp_core(data, x_out, y_out, transform, order=1):
    """Reproject ``data`` onto the pixel grid given by ``x_out``/``y_out``.

    Parameters
    ----------
    data : array, shape (ny_in, nx_in)
        Input image.
    x_out, y_out : arrays, any common broadcastable shape
        Output-image pixel coordinates to fill (0-based, pixel-center
        convention — see :mod:`reprojax.core`).
    transform : PixelTransform
        Maps ``(x_out, y_out) -> (x_in, y_in)``.
    order : int
        0 (nearest) or 1 (bilinear).

    Returns
    -------
    result : array, same shape as x_out
        Reprojected values, dtype matching ``data``. NaN where invalid.
    footprint : array, same shape as x_out
        1.0 where the value is valid, 0.0 where not.
    """
    if order not in SUPPORTED_ORDERS:
        raise ValueError(f"order must be one of {SUPPORTED_ORDERS} (bicubic and higher are not supported), got {order!r}")

    x_in, y_in = transform(x_out, y_out)

    ny, nx = data.shape
    y_clipped = _clip_border_pixels(y_in, ny)
    x_clipped = _clip_border_pixels(x_in, nx)
    coords = [y_clipped, x_clipped]

    result = map_coordinates(data, coords, order=order, mode="constant", cval=0.0)

    # A second interpolation of an all-ones array of the same dtype-neutral
    # precision as the coordinates: its value is exactly the fraction of
    # interpolation weight that landed on in-bounds neighbors. This avoids
    # relying on `cval=nan` propagating through interpolation (order 0 does
    # not respect a NaN coordinate at all, and order 1's weighted sum can
    # produce `nan * 0 = nan` for a neighbor at exactly zero weight — both
    # of which would corrupt otherwise-valid, fully in-bounds pixels).
    coverage_input = jnp.ones((ny, nx), dtype=jnp.float64)
    coverage = map_coordinates(coverage_input, coords, order=order, mode="constant", cval=0.0)

    valid = (coverage >= 1.0 - _COVERAGE_VALID_THRESHOLD) & ~(jnp.isnan(x_in) | jnp.isnan(y_in))

    result = jnp.where(valid, result, jnp.nan).astype(data.dtype)
    footprint = valid.astype(jnp.float32)
    return result, footprint


def _as_wcs(wcs_or_header):
    if isinstance(wcs_or_header, WCS):
        return wcs_or_header
    if isinstance(wcs_or_header, Header):
        return WCS(wcs_or_header)
    raise TypeError(f"expected an astropy WCS or Header, got {type(wcs_or_header)!r}")


def _shape_from_header(wcs_or_header):
    if isinstance(wcs_or_header, Header):
        header = wcs_or_header
    elif getattr(wcs_or_header, "array_shape", None) is not None:
        return tuple(wcs_or_header.array_shape)
    else:
        return None
    if "NAXIS2" in header and "NAXIS1" in header:
        return (header["NAXIS2"], header["NAXIS1"])
    return None


def reproject_interp(input_data, output_projection, shape_out=None, order=1):
    """Reproject an ``(array, WCS-or-header)`` pair onto ``output_projection``.

    Matches ``reproject.reproject_interp``'s call and return shape for the
    subset of input forms reprojax currently supports.

    Parameters
    ----------
    input_data : tuple(array, astropy.wcs.WCS or astropy.io.fits.Header)
    output_projection : astropy.wcs.WCS or astropy.io.fits.Header
    shape_out : tuple(int, int), optional
        Required unless it can be inferred from ``output_projection``
        (a Header's NAXIS1/NAXIS2, or a WCS with ``array_shape`` set).
    order : int
        0 (nearest) or 1 (bilinear).

    Returns
    -------
    array, footprint
    """
    data, wcs_or_header_in = input_data
    wcs_in = _as_wcs(wcs_or_header_in)
    wcs_out = _as_wcs(output_projection)

    if shape_out is None:
        shape_out = _shape_from_header(output_projection)
        if shape_out is None:
            raise ValueError(
                "shape_out could not be inferred from output_projection; pass it explicitly"
            )
    ny, nx = shape_out

    y_out, x_out = jnp.meshgrid(
        jnp.arange(ny, dtype=jnp.float64), jnp.arange(nx, dtype=jnp.float64), indexing="ij"
    )

    transform = build_transform(wcs_in, wcs_out)
    return reproject_interp_core(jnp.asarray(data), x_out, y_out, transform, order=order)
