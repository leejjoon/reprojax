"""Build a :class:`~reprojax.core.PixelTransform` from a pair of astropy WCS objects.

mapprojax provides the projection math (the zenithal TAN, SIN, ZEA, STG and
ARC, with optional SIP distortion); astropy provides celestial frame determination and rotation.
Everything astropy-facing runs once, eagerly, at build time in
:func:`build_transform`; the returned callable is pure JAX and safe under
``jit``/``vmap``/``grad``.
"""

from collections import OrderedDict

import jax.numpy as jnp
import numpy as np
from mapprojax.reproject import Reprojector

from ._astropy_wcs import _rotation_from_descriptors, validated_wcs
from mapprojax import from_astropy

MATRIX_PRECISIONS = ("f64", "f32")

# build_transform's cache: keyed on the validated leaves of wcs_in/wcs_out
# (not object identity), so repeated calls with re-parsed-but-identical WCS
# (e.g. the same header read from disk each time, or the same pointing's
# multiple spectral channels) hit the cache too, not just calls reusing the
# same Python WCS object. Bounded and LRU-evicted since each entry pins a
# compiled transform closure (and jit will hold a compilation cache entry
# keyed on its identity) in memory for as long as the cache keeps it.
#
# The key is built from *validated* output, after from_astropy has accepted
# the pair: a raw-key lookup would let an unsupported input (a TPV header
# astropy normalises to plain TAN, a unit-mismatched header) be served a
# cached transform built from a different WCS. Validation therefore runs on
# every call, hit or miss, and costs one conversion per WCS.
_TRANSFORM_CACHE_MAXSIZE = 32
_TRANSFORM_CACHE = OrderedDict()


def _mapprojax_wcs(astropy_wcs, precision="f64"):
    """The mapprojax kernel for one WCS, with no frame-pair checks.

    Single-WCS callers (the batched path, tests, benchmarks) do not have a
    frame pair, so they must not be subjected to :func:`frame_descriptor`'s
    allow-list; only :func:`build_transform` decides a rotation.
    """
    return from_astropy(astropy_wcs, precision=precision)


def _leaf_key(array):
    """A hashable, dtype-aware key for one array leaf.

    Values are copied first so the ``-0.0`` normalisation never mutates a
    caller's array, and ``dtype.str`` is part of the key because an f32
    ``crpix``/``cd`` beside a host-f64 ``crval`` must not collide with the
    f64 entry for the same WCS.
    """
    owned = np.array(array, copy=True, order="C")
    canonical = np.where(owned == 0, 0.0, owned)
    return (canonical.dtype.str, canonical.shape, canonical.tobytes())


def _sip_key(sip):
    if sip is None:
        return None
    return (
        getattr(sip, "inverse_steps", None),
        tuple(None if leaf is None else _leaf_key(leaf) for leaf in (sip.a, sip.b, sip.ap, sip.bp)),
    )


def _validated_key(record, matrix_precision):
    """The cache key of one validated WCS.

    Everything here comes from the validated output, so an input that differs
    only in a convention astropy hides (TPV vs TAN) can never share a key.
    The descriptor contributes primitives only, never the live astropy frame.
    """
    mwcs = record.mwcs
    descriptor = record.descriptor
    return (
        type(mwcs).__name__,
        _leaf_key(mwcs.crpix),
        _leaf_key(mwcs.cd),
        _leaf_key(mwcs.crval),
        _sip_key(mwcs.sip),
        (descriptor.kind, descriptor.equinox, descriptor.axis_pair),
        matrix_precision,
    )


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
        A mapprojax projection class (``Tan``, ``Sin``, ``Zea``, ``Stg`` or
        ``Arc``), constructed with
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

    def pixel_map(x_out, y_out, *, bounds_shape=None):
        return reprojector.pixel_map(x_out, y_out, bounds_shape=bounds_shape)

    transform.pixel_map = pixel_map
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

    def pixel_map(x_out, y_out, *, bounds_shape=None):
        return reprojector.pixel_map(x_out, y_out, bounds_shape=bounds_shape)

    transform.pixel_map = pixel_map
    return transform


def build_transform(wcs_in, wcs_out, matrix_precision="f64"):
    """Build a ``PixelTransform`` mapping output pixels to input pixels.

    The composed pipeline, per output pixel: output pixel -> output-frame
    unit vector (``wcs_out.unproj_xyz``) -> input-frame unit vector
    (celestial frame rotation) -> input pixel (``wcs_in.proj_xyz``).

    Parameters
    ----------
    wcs_in, wcs_out : astropy.wcs.WCS
        TAN, SIN, ZEA, STG or ARC projections, optionally with SIP distortion.
    matrix_precision : {"f64", "f32"}
        - ``"f64"`` (default): builds ``mwcs_in``/``mwcs_out`` as mapprojax
          kernels (``Tan``, ``Sin``, ``Zea``, ``Stg`` or ``Arc``) with
          ``precision="f64"`` and composes them via
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
    Results are cached on the *validated* content of ``wcs_in``/``wcs_out``
    (projection type, CD, CRPIX, CRVAL, SIP coefficients, and a frame
    descriptor of kind, equinox and CTYPE axis pair) — calling this again
    with an equivalent WCS pair, even freshly reconstructed from the same
    header, returns the *same* transform object rather than rebuilding it.
    Reusing that object also keeps the compiled interpolation/adaptive call
    paths on their established cache key; a changed WCS key or static
    configuration can require retracing, especially for the autodiff-heavy
    adaptive core. Reusing the same WCS pair across many calls (e.g. per-band
    exposures sharing one pointing) pays the construction and compilation
    costs once.

    Validation runs on every call, hit or miss, so an unsupported input can
    never be served a cached transform: both WCSes go through
    :func:`mapprojax.from_astropy` before the cache is consulted, which costs
    one eager conversion per WCS (~0.4 ms without SIP, ~1.5 ms with order-3
    SIP) and is what the cache-key ordering exists to guarantee.
    """
    if matrix_precision not in MATRIX_PRECISIONS:
        raise ValueError(f"matrix_precision must be one of {MATRIX_PRECISIONS}, got {matrix_precision!r}")

    record_in = validated_wcs(wcs_in, precision=matrix_precision)
    record_out = validated_wcs(wcs_out, precision=matrix_precision)

    cache_key = (_validated_key(record_in, matrix_precision), _validated_key(record_out, matrix_precision))
    cached = _TRANSFORM_CACHE.get(cache_key)
    if cached is not None:
        _TRANSFORM_CACHE.move_to_end(cache_key)
        return cached

    frame_rotation = _rotation_from_descriptors(record_out.descriptor, record_in.descriptor)
    if matrix_precision == "f32":
        reprojector = Reprojector(
            record_out.mwcs, record_in.mwcs, frame_rotation=frame_rotation, matrix_precision="f32"
        )
        transform = compose_transform_fp32(reprojector)
    else:
        transform = compose_transform(record_in.mwcs, record_out.mwcs, jnp.asarray(frame_rotation))

    _TRANSFORM_CACHE[cache_key] = transform
    if len(_TRANSFORM_CACHE) > _TRANSFORM_CACHE_MAXSIZE:
        _TRANSFORM_CACHE.popitem(last=False)
    return transform


def _clear_transform_cache():
    _TRANSFORM_CACHE.clear()


build_transform.cache_clear = _clear_transform_cache
