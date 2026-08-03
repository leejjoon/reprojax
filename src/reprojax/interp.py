"""Nearest/bilinear reprojection: the pure-JAX core and the reproject-shaped wrapper.

The core (:func:`reproject_interp_core`) consumes a :class:`~reprojax.core.PixelTransform`
and is a pure function of arrays — jit/vmap/grad-safe. The wrapper
(:func:`reproject_interp`) does the astropy-facing work (header/WCS parsing,
transform composition) and matches ``reproject``'s call/return shape.
"""

import jax
import jax.numpy as jnp
import numpy as np
from astropy.io.fits import Header
from astropy.wcs import WCS
from jax.scipy.ndimage import map_coordinates

from ._sip_refine import DEFAULT_REFINE_STEPS
from .transform import build_transform, compose_transform

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
    coord_is_nan = jnp.isnan(x_in) | jnp.isnan(y_in)

    ny, nx = data.shape
    # NaN coordinates (from the PixelTransform contract's invalid-point
    # convention) must never reach map_coordinates, even though the final
    # result at those positions is masked to NaN regardless: its *gradient*
    # w.r.t. `data` is a scatter-add of (bilinear weight * upstream
    # cotangent) into the neighboring data pixels, and a NaN coordinate
    # makes that weight NaN. The upstream cotangent is correctly zeroed at
    # invalid positions by the jnp.where below (and by any further masking a
    # caller applies), but `nan * 0 = nan` in the scatter itself, corrupting
    # the gradient at whatever data pixels the NaN coordinate's (undefined)
    # floor happens to land on. Substituting a safe, finite, in-bounds
    # coordinate here — its value is discarded by the NaN masking either
    # way — keeps that scatter finite.
    x_safe = jnp.where(coord_is_nan, 0.0, x_in)
    y_safe = jnp.where(coord_is_nan, 0.0, y_in)
    y_clipped = _clip_border_pixels(y_safe, ny)
    x_clipped = _clip_border_pixels(x_safe, nx)
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

    valid_coord = (coverage >= 1.0 - _COVERAGE_VALID_THRESHOLD) & ~coord_is_nan

    result = jnp.where(valid_coord, result, jnp.nan).astype(data.dtype)
    # Footprint tracks the final result, not just coordinate validity: `data`
    # itself can contain NaN (real masked/bad pixels), which propagates
    # through interpolation even when the coordinate is fully in-bounds.
    # Matches reproject's own footprint definition (~isnan(output)).
    footprint = (~jnp.isnan(result)).astype(jnp.float32)
    return result, footprint


def reproject_interp_batched(
    data_batch,
    mwcs_in_batch,
    mwcs_out,
    rotation_batch,
    x_out,
    y_out,
    order=1,
    sip_inverse_refine_steps=DEFAULT_REFINE_STEPS,
):
    """Reproject a batch of exposures onto one shared output grid.

    Batching is `jax.vmap` over the whole per-exposure pipeline (build the
    transform, then reproject) — per the design, batching happens by
    vmapping the core, not by threading a batch dimension through a single
    transform. This is a thin, documented recipe: build ``mwcs_in_batch``
    (and ``rotation_batch``) yourself with mapprojax's JAX WCS classes and
    JAX pytrees, each leaf carrying a leading batch dimension, and everything
    else follows the same path as the single-exposure core.

    Parameters
    ----------
    data_batch : array, shape (n, ny_in, nx_in)
        Stacked input images. Individual exposures may differ in size only
        if you pad them to a common shape first — vmap requires uniform
        shapes across the batch.
    mwcs_in_batch : a mapprojax WCS (e.g. TanJax), batched
        Construct with each parameter (crpix, cd, crval, and SIP
        coefficients if used) carrying a leading dimension of size ``n`` —
        e.g. ``TanJax(crpix_batch, cd_batch, crval_batch, sip=sip_batch)``.
        Parameters that don't vary across the batch (e.g. a shared crpix for
        exposures from the same detector) still need that leading dimension;
        broadcast them yourself (``jnp.broadcast_to``) before constructing.
        Every exposure must share the same SIP-or-not structure (vmap
        requires a uniform pytree).
    mwcs_out : a mapprojax WCS
        The single, shared output projection (not batched).
    rotation_batch : array, shape (n, 3, 3)
        Per-exposure celestial frame rotation (see :func:`build_transform`).
        Often identical across the batch (same input/output frame pair for
        every exposure); broadcast to shape ``(n, 3, 3)`` in that case.
    x_out, y_out : arrays
        The shared output pixel grid (not batched).
    order : int
        0 (nearest) or 1 (bilinear).
    sip_inverse_refine_steps : int
        See :func:`build_transform`.

    Returns
    -------
    results, footprints : arrays, shape (n, *x_out.shape)
        Element-wise identical to calling :func:`reproject_interp_core` once
        per exposure with the same ``x_out``/``y_out``.
    """

    def _single(data, mwcs_in, rotation):
        transform = compose_transform(mwcs_in, mwcs_out, rotation, sip_inverse_refine_steps=sip_inverse_refine_steps)
        return reproject_interp_core(data, x_out, y_out, transform, order=order)

    return jax.vmap(_single, in_axes=(0, 0, 0))(data_batch, mwcs_in_batch, rotation_batch)


def _as_wcs(wcs_or_header):
    if isinstance(wcs_or_header, WCS):
        return wcs_or_header
    if isinstance(wcs_or_header, Header):
        return WCS(wcs_or_header)
    raise TypeError(f"expected an astropy WCS or Header, got {type(wcs_or_header)!r}")


def _to_native_byte_order(data):
    """FITS data is always big-endian; JAX only accepts native-byte-order
    arrays. A plain ``jnp.asarray`` on data read straight from
    ``astropy.io.fits`` fails outright, so the wrapper normalizes here
    before anything touches JAX.
    """
    data = np.asarray(data)
    if not data.dtype.isnative:
        data = data.astype(data.dtype.newbyteorder("="))
    return data


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
    data = jnp.asarray(_to_native_byte_order(data))
    return reproject_interp_core(data, x_out, y_out, transform, order=order)
