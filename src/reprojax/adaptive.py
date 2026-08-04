"""Adaptive anti-aliased reprojection: DeForest (2004), ported from mapprojax.

The core (:func:`reproject_adaptive_core`) consumes a
:class:`~reprojax.core.PixelTransform` (no mapprojax-class coupling) and
returns ``(result, footprint)`` matching :mod:`reprojax.interp`'s
conventions. The wrapper (:func:`reproject_adaptive`) does the astropy-facing
work, matching ``reproject.reproject_adaptive``'s call/return shape.

This is a from-scratch reimplementation of mapprojax's ``adaptive_reproject``
against ``reproject_adaptive``'s actual reference algorithm
(``reproject/adaptive/deforest.pyx``), not a direct port: validating against
the real algorithm surfaced several places where mapprojax's original
approximated reproject's behavior rather than matching it, corrected here
(see the "Deliberate differences from reproject_adaptive" note below).

Deliberate differences from reproject_adaptive
-----------------------------------------------
- **Jacobian**: computed exactly via JAX autodiff (``jax.jacfwd`` through the
  transform), not reproject's finite-difference estimate between
  neighboring output pixels. Empirically these agree to ~1e-10 on every
  scenario tested (synthetic WCS pairs at ~1x and ~4x scale, with SIP, and
  the real SPHEREx pair) — exact autodiff is at least as accurate as a
  finite-difference approximation of the same quantity, and is the natural
  choice given reprojax's differentiable transform.
- **Search window is a fixed maximum size** (``max_window_radius``), not
  reproject's per-pixel dynamically-sized region: JAX/vmap requires a
  static shape across every output pixel. Cells outside the true dynamic
  window (computed per-pixel from the local Jacobian, same as reproject)
  get zero weight, matching reproject's hard truncation, but
  ``max_window_radius`` must be set large enough to contain that dynamic
  window for every pixel in the image — large enough for the most
  aggressive downsampling present, not just a typical pixel. Too small a
  value silently truncates the kernel (no error is raised); this is the
  cost of vmap's uniform per-pixel work requirement.
- **Only the algorithm's default parameters are supported**:
  ``kernel_width``, ``sample_region_width``, and ``kernel`` (Gaussian or
  Hann) are the only tunables. reproject_adaptive's ``boundary_mode`` and
  ``bad_value_mode`` are fixed at their defaults (``"strict"`` for both —
  any output pixel whose window extends beyond the input array, or whose
  window contains a NaN/Inf input value, is set to NaN as a whole) with no
  other mode supported; ``conserve_flux``, ``despike_jacobian``,
  ``center_jacobian``, and ``x_cyclic``/``y_cyclic`` are not implemented.

Validation
----------
Against ``reproject_adaptive`` with matching default parameters, on
non-degenerate (non-integer CRPIX/CRVAL/rotation) synthetic WCS pairs:
identical footprints and value agreement to ~1e-10, at both ~1x and ~4x
scale, with SIP, and with real masked (NaN) input pixels. See
tests/test_adaptive_synthetic.py and tests/test_real_spherex.py.

Two things were diagnosed and are *not* papered over with a looser
tolerance:

1. Exact-integer-tie sensitivity in ``boundary_mode="strict"``'s window
   check: reproject_adaptive's coordinate pipeline can return a
   mathematically-exact integer position for special-case inputs (e.g. an
   output WCS identical to the input WCS), while reprojax's composed
   transform (via trigonometric round-trip through mapprojax) generally
   lands ~1e-10 away from that exact integer even when the true answer is
   exact. Since the window's ceil/floor bounds are then computed from that
   position, a coincidental floating-point nudge across an exact integer
   can shift a search window's boundary by one pixel, occasionally
   disagreeing right at the input array's edge. This is specific to
   artificially "round" test parameters (integer CRPIX, exactly-matching
   in/out WCS) rather than real data — it was not observed on the real
   SPHEREx pair (see tests/test_real_spherex.py), and the synthetic test
   suite uses deliberately non-round parameters to avoid manufacturing it.
2. A genuine bug caught during this validation, not present in the shipped
   code: masking cells outside the per-pixel dynamic search window to zero
   *weight* is not sufficient on its own — the fixed-size window array can
   still contain a NaN there (e.g. a masked input pixel that happens to
   fall within ``max_window_radius`` of an output pixel's mapped position,
   but outside that pixel's actual, smaller dynamic window), and
   `0 * nan = nan` under IEEE 754 poisons the weighted sum regardless of
   the zero weight. Fixed by substituting a safe value outside the dynamic
   window before the multiply — the same "never multiply an excluded
   value by NaN" lesson as reprojax.interp's map_coordinates handling.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .interp import _as_wcs, _shape_from_header, _to_native_byte_order
from .transform import build_transform

SUPPORTED_KERNELS = ("gaussian", "hann")


@partial(
    jax.jit,
    static_argnames=("transform", "kernel", "kernel_width", "sample_region_width", "max_window_radius", "block_size"),
)
def reproject_adaptive_core(
    data,
    x_out,
    y_out,
    transform,
    kernel="gaussian",
    kernel_width=1.3,
    sample_region_width=4.0,
    max_window_radius=10,
    block_size=None,
):
    """Reproject ``data`` onto the pixel grid given by ``x_out``/``y_out``
    using the DeForest (2004) adaptive, anti-aliased algorithm.

    Parameters
    ----------
    data : array, shape (ny_in, nx_in)
        Input image.
    x_out, y_out : arrays, any common broadcastable shape
        Output-image pixel coordinates to fill (0-based, pixel-center
        convention — see :mod:`reprojax.core`).
    transform : PixelTransform
        Maps ``(x_out, y_out) -> (x_in, y_in)``.
    kernel : str
        ``"gaussian"`` (default, matches reproject_adaptive's default) or
        ``"hann"``.
    kernel_width : float
        Gaussian kernel width in pixels (unused for Hann).
    sample_region_width : float
        Width, in output pixels, of the sampling region for the Gaussian
        kernel (unused for Hann, whose extent is determined by the
        transform's local Jacobian alone).
    max_window_radius : int
        Fixed half-width (in input pixels) of the search window; must be
        large enough to contain the true, dynamically-sized search window
        for every output pixel (larger for more aggressive downsampling).
        See the module docstring's note on this being a JAX/vmap-driven
        departure from reproject_adaptive's per-pixel dynamic sizing.
    block_size : int, optional
        Process this many output pixels per `jax.lax.scan` step instead of
        vectorizing the whole output grid at once, to bound memory use.

    Returns
    -------
    result : array, same shape as x_out
        Reprojected values, dtype matching ``data``. NaN where invalid.
    footprint : array, same shape as x_out
        1.0 where the value is valid, 0.0 where not.
    """
    if kernel not in SUPPORTED_KERNELS:
        raise ValueError(f"kernel must be one of {SUPPORTED_KERNELS}, got {kernel!r}")

    ny_in, nx_in = data.shape
    radius = max_window_radius
    win_size = 2 * radius + 1
    padded = jnp.pad(data, radius, mode="constant", constant_values=0.0)

    def process_pixel(x0_out, y0_out):
        u0, v0 = transform(x0_out, y0_out)

        def t(xy):
            u, v = transform(xy[0], xy[1])
            return jnp.stack([u, v])

        J = jax.jacfwd(t)(jnp.array([x0_out, y0_out]))  # d(input)/d(output), (2, 2)

        U, S, Vt = jnp.linalg.svd(J)
        s_clamped = jnp.maximum(1.0, S)  # never sharpen beyond native resolution
        j_inv = (Vt.T * (1.0 / s_clamped)) @ U.T  # d(output)/d(input), using the clamped Jacobian

        if kernel == "gaussian":
            half_w = half_h = sample_region_width * jnp.max(s_clamped) / 2.0
        else:
            ji_padded = (U * s_clamped) @ Vt
            corners = jnp.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
            transformed_corners = corners @ ji_padded.T
            half_w = jnp.max(jnp.abs(transformed_corners[:, 0]))
            half_h = jnp.max(jnp.abs(transformed_corners[:, 1]))

        bottom = jnp.ceil(v0 - half_h)
        top = jnp.floor(v0 + half_h)
        left = jnp.ceil(u0 - half_w)
        right = jnp.floor(u0 + half_w)

        u0_int = jnp.rint(u0).astype(int)
        v0_int = jnp.rint(v0).astype(int)
        window = jax.lax.dynamic_slice(padded, (v0_int, u0_int), (win_size, win_size))

        win_v_idx, win_u_idx = jnp.meshgrid(jnp.arange(win_size), jnp.arange(win_size), indexing="ij")
        global_v = v0_int - radius + win_v_idx
        global_u = u0_int - radius + win_u_idx
        global_v_f = global_v.astype(jnp.float64)
        global_u_f = global_u.astype(jnp.float64)

        in_dynamic_window = (global_v_f >= bottom) & (global_v_f <= top) & (global_u_f >= left) & (global_u_f <= right)

        dv = global_v_f - v0
        du = global_u_f - u0
        tx = j_inv[0, 0] * du + j_inv[0, 1] * dv
        ty = j_inv[1, 0] * du + j_inv[1, 1] * dv

        if kernel == "gaussian":
            raw_weight = jnp.exp(-(tx**2 + ty**2) / kernel_width**2 * 2.0)
        else:
            hann_mask = (jnp.abs(tx) < 1.0) & (jnp.abs(ty) < 1.0)
            raw_weight = jnp.where(hann_mask, (jnp.cos(jnp.pi * tx) + 1.0) * (jnp.cos(jnp.pi * ty) + 1.0), 0.0)

        weight = jnp.where(in_dynamic_window, raw_weight, 0.0)

        # Cells outside the dynamic window carry zero weight, but the
        # fixed-size window can still contain a NaN there; `0 * nan = nan`
        # would poison the sum regardless of weight (see module docstring).
        # A NaN genuinely inside the dynamic window must still poison the
        # result (strict bad_value_mode) -- only substitute outside it.
        window_safe = jnp.where(in_dynamic_window, window, 0.0)

        weight_sum = jnp.sum(weight)
        target = jnp.sum(weight * window_safe)

        in_array_bounds = (global_v >= 0) & (global_v < ny_in) & (global_u >= 0) & (global_u < nx_in)
        # strict boundary_mode: any dynamic-window cell outside the input
        # array invalidates the whole output pixel.
        window_out_of_bounds = jnp.any(in_dynamic_window & ~in_array_bounds)
        invalid_transform = jnp.isnan(u0) | jnp.isnan(v0) | jnp.any(jnp.isnan(J))

        result = target / weight_sum
        return jnp.where(window_out_of_bounds | invalid_transform, jnp.nan, result)

    x_flat = x_out.ravel()
    y_flat = y_out.ravel()

    if block_size is None:
        flat_result = jax.vmap(process_pixel)(x_flat, y_flat)
    else:
        n = x_flat.size
        pad = (-n) % block_size
        x_padded = jnp.pad(x_flat, (0, pad))
        y_padded = jnp.pad(y_flat, (0, pad))
        x_blocks = x_padded.reshape(-1, block_size)
        y_blocks = y_padded.reshape(-1, block_size)

        def scan_body(carry, block):
            bx, by = block
            return carry, jax.vmap(process_pixel)(bx, by)

        _, blocks_out = jax.lax.scan(scan_body, None, (x_blocks, y_blocks))
        flat_result = blocks_out.ravel()
        if pad:
            flat_result = flat_result[:-pad]

    result = flat_result.reshape(x_out.shape).astype(data.dtype)
    footprint = (~jnp.isnan(result)).astype(jnp.float32)
    return result, footprint


def reproject_adaptive(
    input_data,
    output_projection,
    shape_out=None,
    kernel="gaussian",
    kernel_width=1.3,
    sample_region_width=4.0,
    max_window_radius=10,
    block_size=None,
    matrix_precision="f64",
):
    """Reproject an ``(array, WCS-or-header)`` pair onto ``output_projection``
    using the DeForest (2004) adaptive, anti-aliased algorithm.

    Matches ``reproject.reproject_adaptive``'s call and return shape for the
    parameters this port supports — see the module docstring for the
    (default-matching) scope this covers and what's out of scope.

    Parameters
    ----------
    input_data : tuple(array, astropy.wcs.WCS or astropy.io.fits.Header)
    output_projection : astropy.wcs.WCS or astropy.io.fits.Header
    shape_out : tuple(int, int), optional
        Required unless it can be inferred from ``output_projection``
        (a Header's NAXIS1/NAXIS2, or a WCS with ``array_shape`` set).
    kernel, kernel_width, sample_region_width, max_window_radius, block_size
        See :func:`reproject_adaptive_core`.
    matrix_precision : {"f64", "f32"}
        See :func:`~reprojax.transform.build_transform`. ``"f32"`` needs
        no ``jax_enable_x64`` and is faster on GPUs where float64
        throughput is much lower than float32's, at reduced accuracy
        (~1e-4 px for typical overlapping-field reprojections; see that
        function's docstring for the detailed tradeoff).

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

    transform = build_transform(wcs_in, wcs_out, matrix_precision=matrix_precision)
    data = jnp.asarray(_to_native_byte_order(data))
    return reproject_adaptive_core(
        data,
        x_out,
        y_out,
        transform,
        kernel=kernel,
        kernel_width=kernel_width,
        sample_region_width=sample_region_width,
        max_window_radius=max_window_radius,
        block_size=block_size,
    )
