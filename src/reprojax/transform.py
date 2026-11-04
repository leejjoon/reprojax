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
from mapprojax.wcs import Sin, Tan
from mapprojax.reproject import Reprojector
from mapprojax.utils import apply_rotation

from ._astropy_wcs import frame_rotation_matrix, projection_code, wcs_params_from_astropy

_PROVIDERS = {"TAN": Tan, "SIN": Sin}
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


def _mapprojax_wcs(astropy_wcs, precision="f64"):
    code = projection_code(astropy_wcs)
    crpix, cd, crval, sip = wcs_params_from_astropy(astropy_wcs)
    return _PROVIDERS[code](crpix, cd, crval, sip=sip, precision=precision)


def _sip_fingerprint(sip):
    if sip is None:
        return None
    return tuple(None if arr is None else (arr.shape, arr.tobytes()) for arr in (sip.a, sip.b, sip.ap, sip.bp))


def _wcs_fingerprint(astropy_wcs):
    code = projection_code(astropy_wcs)
    crpix, cd, crval, sip = wcs_params_from_astropy(astropy_wcs)
    frame = repr(wcs_to_celestial_frame(astropy_wcs))
    return (code, tuple(crpix), tuple(map(tuple, cd)), tuple(crval), _sip_fingerprint(sip), frame)


def compose_transform(mwcs_in, mwcs_out, rotation):
    """Build a ``PixelTransform`` from already-constructed mapprojax WCS
    objects and a fixed frame-rotation matrix — the pure-JAX half of
    :func:`build_transform`, with no astropy involved.

    ``mwcs_in``/``mwcs_out`` are JAX pytrees (mapprojax registers
    ``Tan``/``Sin``/``Sip`` for exactly this), so unlike
    ``build_transform`` (which takes non-traceable ``astropy.wcs.WCS``
    objects), this function's WCS parameters — CRVAL, CD, SIP coefficients —
    are differentiable: construct ``mwcs_in``/``mwcs_out`` inside a function
    you pass to ``jax.grad``, and gradients flow through here into whatever
    consumes the returned transform.

    The relative rotation ``mwcs_in.r_matrix @ rotation @ mwcs_out.r_matrix.T``
    is formed in JAX (so it stays differentiable with respect to CRVAL) and
    handed to :meth:`mapprojax.reproject.Reprojector.from_matrix` with
    ``matrix_precision="f64"``; the returned transform is that
    ``Reprojector``'s ``pix_to_pix``.

    Parameters
    ----------
    mwcs_in, mwcs_out : mapprojax WCS
        ``mapprojax.wcs.Tan`` or ``mapprojax.wcs.Sin``, constructed with
        ``precision="f64"`` (the ``precision=`` argument is required), and
        optionally carrying a ``mapprojax.sip.Sip``.
    rotation : array, shape (3, 3)
        Fixed celestial-frame rotation from ``mwcs_out``'s frame to
        ``mwcs_in``'s (not a differentiable parameter — it comes from
        astropy's frame machinery in ``build_transform``, but a constant
        array is equally valid to pass directly).
    Returns
    -------
    PixelTransform
    """

    m_rel = mwcs_in.r_matrix @ rotation @ jnp.swapaxes(mwcs_out.r_matrix, -1, -2)
    reprojector = Reprojector.from_matrix(
        mwcs_out, mwcs_in, m_rel, matrix_precision="f64"
    )

    def transform(x_out, y_out):
        return reprojector.pix_to_pix(x_out, y_out)

    return transform


def compose_transform_fp32(reprojector):
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
    Returns
    -------
    PixelTransform
    """
    def transform(x_out, y_out):
        return reprojector.pix_to_pix(x_out, y_out)

    return transform


def build_transform(wcs_in, wcs_out, matrix_precision="f64"):
    """Build a ``PixelTransform`` mapping output pixels to input pixels.

    The composed pipeline, per output pixel: output pixel -> output-frame
    unit vector (``wcs_out.unproj_xyz``) -> input-frame unit vector
    (celestial frame rotation) -> input pixel (``wcs_in.proj_xyz``).

    Parameters
    ----------
    wcs_in, wcs_out : astropy.wcs.WCS
        TAN or SIN projections, optionally with SIP distortion.
    matrix_precision : {"f64", "f32"}
        - ``"f64"`` (default): builds ``mwcs_in``/``mwcs_out`` as mapprojax
          ``Tan``/``Sin`` with ``precision="f64"`` and composes them via
          :func:`compose_transform` (``Reprojector.from_matrix`` with
          ``matrix_precision="f64"``), entirely in float64. Requires
          ``jax_enable_x64`` for correctness.
        - ``"f32"``: builds ``precision="f32"`` WCS objects and a
          :class:`mapprojax.reproject.Reprojector` with
          ``matrix_precision="f32"``, which composes the two WCSes'
          rotations with the frame rotation into one matrix, once, in true
          float64 on the host, independent of ``jax_enable_x64``; every
          per-pixel call then runs in float32. No ``jax_enable_x64``
          needed, and faster on GPUs where float64 throughput is much
          lower than float32's. Less accurate than ``"f64"`` -- a few
          e-4 px on the real SPHEREx ICRS<->Galactic pair in
          ``tests/test_real_spherex.py``, growing with field-center
          separation and with finer plate scales (see mapprojax's
          ``Reprojector`` docstring); not differentiable with respect to
          either WCS's ``crval`` (see :func:`compose_transform_fp32`).

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
    CRPIX, CRVAL, SIP coefficients, projection code, and celestial frame)
    — calling this again with an equivalent
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

    cache_key = (_wcs_fingerprint(wcs_in), _wcs_fingerprint(wcs_out), matrix_precision)
    cached = _TRANSFORM_CACHE.get(cache_key)
    if cached is not None:
        _TRANSFORM_CACHE.move_to_end(cache_key)
        return cached

    if matrix_precision == "f32":
        wcs_in_np = _mapprojax_wcs(wcs_in, precision="f32")
        wcs_out_np = _mapprojax_wcs(wcs_out, precision="f32")
        frame_rotation = frame_rotation_matrix(wcs_out, wcs_in)
        reprojector = Reprojector(wcs_out_np, wcs_in_np, frame_rotation=frame_rotation, matrix_precision="f32")
        transform = compose_transform_fp32(reprojector)
    else:
        mwcs_in = _mapprojax_wcs(wcs_in)
        mwcs_out = _mapprojax_wcs(wcs_out)
        rotation = jnp.asarray(frame_rotation_matrix(wcs_out, wcs_in))
        transform = compose_transform(mwcs_in, mwcs_out, rotation)

    _TRANSFORM_CACHE[cache_key] = transform
    if len(_TRANSFORM_CACHE) > _TRANSFORM_CACHE_MAXSIZE:
        _TRANSFORM_CACHE.popitem(last=False)
    return transform


def _clear_transform_cache():
    _TRANSFORM_CACHE.clear()


build_transform.cache_clear = _clear_transform_cache
