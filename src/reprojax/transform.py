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
from mapprojax.utils import apply_rotation

from ._astropy_wcs import frame_rotation_matrix, projection_code, wcs_params_from_astropy
from ._sip_refine import DEFAULT_REFINE_STEPS, refine_pixel_estimate

_PROVIDERS = {"TAN": TanJax, "SIN": SinJax}

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


def _sip_fingerprint(sip):
    if sip is None:
        return None
    return tuple((arr.shape, arr.tobytes()) for arr in (sip.a, sip.b, sip.ap, sip.bp))


def _wcs_fingerprint(astropy_wcs):
    code = projection_code(astropy_wcs)
    crpix, cd, crval, sip = wcs_params_from_astropy(astropy_wcs)
    frame = repr(wcs_to_celestial_frame(astropy_wcs))
    return (code, tuple(crpix), tuple(map(tuple, cd)), tuple(crval), _sip_fingerprint(sip), frame)


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

    Note
    ----
    Differentiating *twice* through this function (as ``jax.grad`` of a loss
    does when ``mwcs_in`` has SIP, since refinement's own Jacobian is itself
    autodiff) can produce NaN at an *exact* geometric symmetry point: e.g. an
    output pixel landing exactly on both WCSes' CRPIX with an axis-aligned CD
    matrix, no frame rotation, and forward SIP polynomials that each depend
    on only one of (u, v). This needs several exact coincidences at once and
    was not observed with real WCS parameters (real CRPIX values are not
    integers, real frame rotations are not the identity, real SIP always has
    cross-terms) or with randomized/offset synthetic ones — see
    tests/test_differentiability.py for both the reproduction and the
    realistic-parameter checks that pass cleanly.

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
        valid = xn > 0

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


def build_transform(wcs_in, wcs_out, sip_inverse_refine_steps=DEFAULT_REFINE_STEPS):
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
    cache_key = (_wcs_fingerprint(wcs_in), _wcs_fingerprint(wcs_out), sip_inverse_refine_steps)
    cached = _TRANSFORM_CACHE.get(cache_key)
    if cached is not None:
        _TRANSFORM_CACHE.move_to_end(cache_key)
        return cached

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
