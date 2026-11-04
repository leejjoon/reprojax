"""Build a :class:`~reprojax.core.PixelTransform` from a pair of astropy WCS objects.

mapprojax provides the projection math (TAN, SIN, with optional SIP
distortion); astropy provides celestial frame determination and rotation.
Everything astropy-facing runs once, eagerly, at build time in
:func:`build_transform`; the returned callable is pure JAX and safe under
``jit``/``vmap``/``grad``.
"""

from collections import OrderedDict

import jax.numpy as jnp
from astropy.wcs.utils import wcs_to_celestial_frame
from mapprojax.jax_projections import SinJax, TanJax
from mapprojax.projections import Sin, SinMixin, Tan
from mapprojax.reproject import Reprojector
from mapprojax.utils import apply_rotation

from ._astropy_wcs import frame_rotation_matrix, projection_code, wcs_params_from_astropy
from ._sip_refine import DEFAULT_REFINE_STEPS, refine_pixel_estimate

_PROVIDERS = {"TAN": TanJax, "SIN": SinJax}
#: NumPy-backed (not JAX) counterparts, for Reprojector/compose_relative_rotation --
#: see _mapprojax_wcs_numpy.
_PROVIDERS_NUMPY = {"TAN": Tan, "SIN": Sin}
MATRIX_PRECISIONS = ("f64", "f32")

# build_transform's cache: keyed on the numeric content of wcs_in/wcs_out
# (not object identity), so repeated calls with re-parsed-but-identical WCS
# (e.g. the same header read from disk each time, or the same pointing's
# multiple spectral channels) hit the cache too, not just calls reusing the
# same Python WCS object. Bounded and LRU-evicted since each entry pins a
# compiled transform closure (and jit will hold a compilation cache entry
# keyed on its identity) in memory for as long as the cache keeps it.
_TRANSFORM_CACHE_MAXSIZE = 32
_TRANSFORM_CACHE = OrderedDict()


def _mapprojax_wcs(astropy_wcs):
    code = projection_code(astropy_wcs)
    crpix, cd, crval, sip = wcs_params_from_astropy(astropy_wcs)
    return _PROVIDERS[code](crpix, cd, crval, sip=sip)


def _mapprojax_wcs_numpy(astropy_wcs):
    """Like :func:`_mapprojax_wcs`, but a plain NumPy-backed mapprojax WCS
    (``Tan``/``Sin``) instead of the JAX one. Needed for
    :class:`mapprojax.reproject.Reprojector`, whose one-time rotation
    composition must run in true float64 regardless of
    ``jax_enable_x64`` -- a JAX WCS's own rotation matrix may already be
    truncated to float32 if x64 was never enabled (see
    ``compose_relative_rotation``'s docstring in mapprojax).
    """
    code = projection_code(astropy_wcs)
    crpix, cd, crval, sip = wcs_params_from_astropy(astropy_wcs)
    return _PROVIDERS_NUMPY[code](crpix, cd, crval, sip=sip)


def _sip_fingerprint(sip):
    if sip is None:
        return None
    # A/B-only SIP (no AP/BP) is valid: a missing polynomial is part of the key.
    return tuple(None if arr is None else (arr.shape, arr.tobytes()) for arr in (sip.a, sip.b, sip.ap, sip.bp))


def _wcs_fingerprint(astropy_wcs):
    code = projection_code(astropy_wcs)
    crpix, cd, crval, sip = wcs_params_from_astropy(astropy_wcs)
    frame = repr(wcs_to_celestial_frame(astropy_wcs))
    return (code, tuple(crpix), tuple(map(tuple, cd)), tuple(crval), _sip_fingerprint(sip), frame)


def _forward_hemisphere(mwcs, xn):
    """Native-frame validity of the forward projection: TAN excludes its
    horizon (``xn == 0`` maps to infinity); SIN includes it (the unit-disc
    boundary is a finite, valid pixel)."""
    return xn >= 0 if isinstance(mwcs, SinMixin) else xn > 0


def compose_transform(mwcs_in, mwcs_out, rotation, sip_inverse_refine_steps=DEFAULT_REFINE_STEPS):
    """Build a ``PixelTransform`` from already-constructed mapprojax WCS
    objects and a fixed frame-rotation matrix — the pure-JAX half of
    :func:`build_transform`, with no astropy involved.

    ``mwcs_in``/``mwcs_out`` are JAX pytrees (mapprojax registers
    ``TanJax``/``SinJax``/``Sip`` for exactly this), so unlike
    ``build_transform`` (which takes non-traceable ``astropy.wcs.WCS``
    objects), this function's WCS parameters — CRVAL, CD, SIP coefficients —
    are differentiable: construct ``mwcs_in``/``mwcs_out`` inside a function
    you pass to ``jax.grad``, and gradients flow through here into whatever
    consumes the returned transform.

    Parameters
    ----------
    mwcs_in, mwcs_out : mapprojax WCS (e.g. TanJax, SinJax)
    rotation : array, shape (3, 3)
        Fixed celestial-frame rotation from ``mwcs_out``'s frame to
        ``mwcs_in``'s (not a differentiable parameter — it comes from
        astropy's frame machinery in ``build_transform``, but a constant
        array is equally valid to pass directly).
    sip_inverse_refine_steps : int
        See :func:`build_transform`. Applied only when ``mwcs_in.sip`` is
        not None.

    Returns
    -------
    PixelTransform
    """
    refine_steps = sip_inverse_refine_steps if mwcs_in.sip is not None else 0

    def transform(x_out, y_out):
        x_out = jnp.asarray(x_out)
        y_out = jnp.asarray(y_out)

        v_out = mwcs_out.unproj_xyz(x_out, y_out)
        v_in_celestial = apply_rotation(rotation, *v_out)

        # Inline mwcs_in.proj_xyz's body (celestial -> native rotation, then
        # native -> pixel) so we can mask the native x <= 0 case: the far
        # hemisphere of a zenithal projection (TAN, SIN) has no valid
        # forward projection, and the raw math produces huge or ambiguous
        # values there rather than the NaN the PixelTransform contract
        # requires for invalid points.
        xn, yn, zn = apply_rotation(mwcs_in.r_matrix, *v_in_celestial)
        valid = _forward_hemisphere(mwcs_in, xn)

        # Substitute the (always-safe) native pole direction at invalid
        # points *before* native_to_pix/refinement, rather than computing on
        # the real (possibly wildly out-of-domain) xn/yn/zn and masking
        # afterward. Under jax.grad, a discarded jnp.where branch that
        # merely evaluates to a huge-but-finite number is harmless, but one
        # that produces NaN/Inf (Newton refinement diverging from a bad
        # starting point on the far hemisphere) poisons the gradient of the
        # whole loss even at points where `valid` is True — `False * nan`
        # is `nan`, not `0`, and reprojax's WCS parameters (crval, SIP
        # coefficients) are shared across every output pixel.
        xn_safe = jnp.where(valid, xn, 1.0)
        yn_safe = jnp.where(valid, yn, 0.0)
        zn_safe = jnp.where(valid, zn, 0.0)
        x_in, y_in = mwcs_in.native_to_pix(xn_safe, yn_safe, zn_safe)

        if refine_steps > 0:
            x_in, y_in = refine_pixel_estimate(mwcs_in, x_in, y_in, xn_safe, yn_safe, zn_safe, n_steps=refine_steps)

        x_in = jnp.where(valid, x_in, jnp.nan)
        y_in = jnp.where(valid, y_in, jnp.nan)
        return x_in, y_in

    return transform


def compose_transform_fp32(reprojector, sip_inverse_refine_steps=DEFAULT_REFINE_STEPS):
    """Build a ``PixelTransform`` from a :class:`mapprojax.reproject.Reprojector`
    -- the fp64-free counterpart of :func:`compose_transform`.

    The celestial<->native rotation runs once, in true float64, at
    ``Reprojector`` construction time (not per pixel -- see its
    docstring); every per-pixel call after that (``pix_to_native -> apply
    the composed rotation -> native_to_pix``) runs in float32,
    independent of ``jax_enable_x64``.

    Note
    ----
    Unlike :func:`compose_transform`, this is *not* differentiable with
    respect to either WCS's ``crval``: ``Reprojector``'s rotation is
    precomputed outside JAX's trace (see its docstring). It remains
    differentiable with respect to the output pixel coordinates
    (``x_out``, ``y_out``) -- all ``reproject_adaptive``'s
    ``jax.jacfwd`` needs -- and, through the target leg, with respect to
    ``crpix``/``cd``/SIP coefficients.

    Parameters
    ----------
    reprojector : mapprojax.reproject.Reprojector
        Maps output pixels (its ``wcs_src``) to input pixels (its
        ``wcs_tgt``) -- see :func:`build_transform`.
    sip_inverse_refine_steps : int
        See :func:`build_transform`. Applied only when the target leg has
        SIP.

    Returns
    -------
    PixelTransform
    """
    tgt_leg = reprojector.tgt_leg
    refine_steps = sip_inverse_refine_steps if tgt_leg.sip is not None else 0

    def transform(x_out, y_out):
        x_out = jnp.asarray(x_out)
        y_out = jnp.asarray(y_out)

        xn, yn, zn = reprojector.pix_to_native_tgt(x_out, y_out)
        valid = _forward_hemisphere(tgt_leg, xn)

        # Same far-hemisphere masking as compose_transform -- see its
        # comment for why this substitutes a safe direction before
        # native_to_pix/refinement rather than masking the result after.
        xn_safe = jnp.where(valid, xn, 1.0)
        yn_safe = jnp.where(valid, yn, 0.0)
        zn_safe = jnp.where(valid, zn, 0.0)
        x_in, y_in = tgt_leg.native_to_pix(xn_safe, yn_safe, zn_safe)

        if refine_steps > 0:
            x_in, y_in = refine_pixel_estimate(tgt_leg, x_in, y_in, xn_safe, yn_safe, zn_safe, n_steps=refine_steps)

        x_in = jnp.where(valid, x_in, jnp.nan)
        y_in = jnp.where(valid, y_in, jnp.nan)
        return x_in, y_in

    return transform


def build_transform(wcs_in, wcs_out, sip_inverse_refine_steps=DEFAULT_REFINE_STEPS, matrix_precision="f64"):
    """Build a ``PixelTransform`` mapping output pixels to input pixels.

    The composed pipeline, per output pixel: output pixel -> output-frame
    unit vector (``wcs_out.unproj_xyz``) -> input-frame unit vector
    (celestial frame rotation) -> input pixel (``wcs_in.proj_xyz``).

    Parameters
    ----------
    wcs_in, wcs_out : astropy.wcs.WCS
        TAN or SIN projections, optionally with SIP distortion.
    sip_inverse_refine_steps : int
        When ``wcs_in`` has SIP, the number of Newton steps (see
        :mod:`reprojax._sip_refine`) used to refine the AP/BP-based pixel
        estimate. AP/BP is an independently-fitted approximation of the true
        inverse, and its quality varies file to file — on a real SPHEREx L2b
        header it disagreed with astropy's iterative inversion by up to
        0.03 px, while a couple of Newton steps against the (exact) forward
        polynomial brings that to ~1e-7 px. Set to 0 to disable (matching
        the plain AP/BP evaluation used before this was diagnosed).
    matrix_precision : {"f64", "f32"}
        - ``"f64"`` (default): current behavior, unchanged -- builds
          ``mwcs_in``/``mwcs_out`` as JAX WCS objects and composes the
          rotation via :func:`compose_transform`, entirely in float64
          when ``jax_enable_x64`` is enabled (required for correctness;
          see :mod:`reprojax`'s and mapprojax's warnings if it isn't).
        - ``"f32"``: uses ``mapprojax.reproject.Reprojector`` (see
          mapprojax's ``fo64_report_and_plan.md``, Phase 1) to compose
          the two WCSes' rotations into one matrix, once, in true
          float64 on the host, independent of ``jax_enable_x64``; every
          per-pixel call then runs in float32. No ``jax_enable_x64``
          needed, and faster on GPUs where float64 throughput is much
          lower than float32's. Less accurate than ``"f64"`` -- ~1e-4 px
          for typical overlapping-field reprojections at arcsecond plate
          scales, growing with field-center separation and with finer
          plate scales (see mapprojax's ``Reprojector`` docstring); not
          differentiable with respect to either WCS's ``crval`` (see
          :func:`compose_transform_fp32`).

    Returns
    -------
    PixelTransform

    Note
    ----
    This function itself is not traceable (it constructs and reads
    ``astropy.wcs.WCS`` objects, which are not JAX pytrees) — the returned
    *transform* is pure JAX, but you cannot ``jax.grad`` through
    ``build_transform`` with respect to ``wcs_in``/``wcs_out``. To
    differentiate with respect to WCS parameters (CRVAL, CD, SIP
    coefficients), construct mapprojax WCS objects directly and use
    :func:`compose_transform`.

    Note
    ----
    Results are cached on the numeric content of ``wcs_in``/``wcs_out`` (CD,
    CRPIX, CRVAL, SIP coefficients, projection code, celestial frame) plus
    ``sip_inverse_refine_steps`` — calling this again with an equivalent
    WCS pair, even freshly reconstructed from the same header, returns the
    *same* transform object rather than rebuilding it. This matters beyond
    saving the (cheap) astropy-side work here: the returned ``transform`` is
    passed as a static ``jax.jit`` argument by :mod:`reprojax.interp` and
    :mod:`reprojax.adaptive`, so a fresh object forces a full retrace on the
    next call — negligible for ``reproject_interp``, but expensive for
    ``reproject_adaptive`` (its core differentiates *through* ``transform``,
    and retracing an already-autodiff-heavy function, as SIP inversion is,
    is costly regardless of whether the compiled executable ends up cached).
    Reusing the same WCS pair across many calls (e.g. per-band exposures
    sharing one pointing) now pays that cost once.
    """
    if matrix_precision not in MATRIX_PRECISIONS:
        raise ValueError(f"matrix_precision must be one of {MATRIX_PRECISIONS}, got {matrix_precision!r}")

    cache_key = (_wcs_fingerprint(wcs_in), _wcs_fingerprint(wcs_out), sip_inverse_refine_steps, matrix_precision)
    cached = _TRANSFORM_CACHE.get(cache_key)
    if cached is not None:
        _TRANSFORM_CACHE.move_to_end(cache_key)
        return cached

    if matrix_precision == "f32":
        wcs_in_np = _mapprojax_wcs_numpy(wcs_in)
        wcs_out_np = _mapprojax_wcs_numpy(wcs_out)
        frame_rotation = frame_rotation_matrix(wcs_out, wcs_in)
        reprojector = Reprojector(wcs_out_np, wcs_in_np, frame_rotation=frame_rotation, matrix_precision="f32")
        transform = compose_transform_fp32(reprojector, sip_inverse_refine_steps=sip_inverse_refine_steps)
    else:
        mwcs_in = _mapprojax_wcs(wcs_in)
        mwcs_out = _mapprojax_wcs(wcs_out)
        rotation = jnp.asarray(frame_rotation_matrix(wcs_out, wcs_in))
        transform = compose_transform(mwcs_in, mwcs_out, rotation, sip_inverse_refine_steps=sip_inverse_refine_steps)

    _TRANSFORM_CACHE[cache_key] = transform
    if len(_TRANSFORM_CACHE) > _TRANSFORM_CACHE_MAXSIZE:
        _TRANSFORM_CACHE.popitem(last=False)
    return transform


def _clear_transform_cache():
    _TRANSFORM_CACHE.clear()


build_transform.cache_clear = _clear_transform_cache
