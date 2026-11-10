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

from .transform import build_transform, compose_transform

SUPPORTED_ORDERS = (0, 1)


def _output_grid_dtype(matrix_precision):
    """The dtype for the output pixel grid built by the wrappers.

    The grid must match the transform's coordinate dtype, not a fixed
    float64: under ``matrix_precision="f32"`` every per-pixel call runs in
    float32, so a float64 grid only costs accuracy and, with
    ``jax_enable_x64`` off, emits a truncation warning on every call.
    """
    return jnp.float32 if matrix_precision == "f32" else jnp.float64

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

    ny, nx = data.shape
    pixel_map = getattr(transform, "pixel_map", None)
    has_pixel_map = callable(pixel_map)
    if has_pixel_map:
        x_in, y_in, valid = pixel_map(x_out, y_out, bounds_shape=(ny, nx))
        coord_is_nan = ~valid
    else:
        x_in, y_in = transform(x_out, y_out)
        coord_is_nan = jnp.isnan(x_in) | jnp.isnan(y_in)
        valid = ~coord_is_nan
    # Generic providers may return NaN coordinates, which must never reach
    # map_coordinates: NaN interpolation weights can corrupt data gradients
    # even when the final result is masked invalid. pixel_map providers
    # already guarantee finite zero stand-ins, so they need no second
    # sanitization pass.
    if has_pixel_map:
        x_safe, y_safe = x_in, y_in
    else:
        x_safe = jnp.where(coord_is_nan, 0.0, x_in)
        y_safe = jnp.where(coord_is_nan, 0.0, y_in)
    y_clipped = _clip_border_pixels(y_safe, ny)
    x_clipped = _clip_border_pixels(x_safe, nx)
    coords = [y_clipped, x_clipped]

    result = map_coordinates(data, coords, order=order, mode="constant", cval=0.0)

    if has_pixel_map:
        valid_coord = valid
    else:
        # For generic providers, retain the existing coverage interpolation:
        # its value is the fraction of interpolation weight that landed on
        # in-bounds neighbors. A float32 fraction is exact enough for the
        # threshold below, and matches the float32 path under x64-off.
        coverage_input = jnp.ones((ny, nx), dtype=jnp.float32)
        coverage = map_coordinates(
            coverage_input, coords, order=order, mode="constant", cval=0.0
        )
        valid_coord = (
            valid
            & (coverage >= 1.0 - _COVERAGE_VALID_THRESHOLD)
            & ~coord_is_nan
        )

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
):
    """Reproject a batch of exposures onto one shared output grid.

    Batching is `jax.vmap` over the whole per-exposure pipeline (build the
    transform, then reproject) — per the design, batching happens by
    vmapping the core, not by threading a batch dimension through a single
    transform. This is a thin, documented recipe: build ``mwcs_in_batch``
    (and ``rotation_batch``) yourself with mapprojax's ``Tan``/``Sin`` (which are
    JAX pytrees), each leaf carrying a leading batch dimension, and everything
    else follows the same path as the single-exposure core.

    Parameters
    ----------
    data_batch : array, shape (n, ny_in, nx_in)
        Stacked input images. Individual exposures may differ in size only
        if you pad them to a common shape first — vmap requires uniform
        shapes across the batch.
    mwcs_in_batch : a mapprojax WCS (e.g. ``mapprojax.wcs.Tan``), batched
        Construct with each parameter (crpix, cd, crval, and SIP
        coefficients if used) carrying a leading dimension of size ``n`` —
        e.g. ``Tan(crpix_batch, cd_batch, crval_batch, sip=sip_batch,
        precision="f64")`` (``precision=`` is required).
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

    Returns
    -------
    results, footprints : arrays, shape (n, *x_out.shape)
        Element-wise identical to calling :func:`reproject_interp_core` once
        per exposure with the same ``x_out``/``y_out``.
    """

    def _single(data, mwcs_in, rotation):
        transform = compose_transform(mwcs_in, mwcs_out, rotation)
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


def reproject_interp(input_data, output_projection, shape_out=None, order=1, matrix_precision="f64"):
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
    matrix_precision : {"f64", "f32"}
        See :func:`~reprojax.transform.build_transform`. ``"f32"`` needs
        no ``jax_enable_x64`` and is faster on GPUs where float64
        throughput is much lower than float32's, at reduced accuracy
        (~1e-4 px for typical overlapping-field reprojections; see that
        function's docstring for the detailed tradeoff).

    Returns
    -------
    array, footprint

    Note
    ----
    Like :func:`~reprojax.adaptive.reproject_adaptive`, this calls
    :func:`~reprojax.transform.build_transform` internally, so repeated
    calls with an equivalent ``wcs_in``/``wcs_out`` pair reuse the same
    transform object rather than rebuilding (and retracing) it — see that
    function's docstring for the caching details.
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

    grid_dtype = _output_grid_dtype(matrix_precision)
    y_out, x_out = jnp.meshgrid(
        jnp.arange(ny, dtype=grid_dtype), jnp.arange(nx, dtype=grid_dtype), indexing="ij"
    )

    transform = build_transform(wcs_in, wcs_out, matrix_precision=matrix_precision)
    data = jnp.asarray(_to_native_byte_order(data))
    return reproject_interp_core(data, x_out, y_out, transform, order=order)
