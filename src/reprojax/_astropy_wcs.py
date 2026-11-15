"""Host-side WCS validation and frame rotations, on top of mapprojax.

Everything in this module runs eagerly in NumPy/astropy at transform-build
time — never inside the jit-compiled :class:`~reprojax.core.PixelTransform`
itself. :func:`mapprojax.from_astropy` owns the numerical conversion and its
refusals; this module adds the two things reprojax needs on top of it: a
classification of the celestial frame pair, and the rotation between them.
"""

from typing import NamedTuple

import numpy as np
from astropy.coordinates import FK5, Galactic, ICRS, SkyCoord
from astropy.wcs.utils import wcs_to_celestial_frame
from mapprojax import from_astropy

#: CTYPE prefixes of the two supported celestial axis pairs, in world-axis
#: order: (longitude, latitude).
_AXIS_PAIRS = {("RA--", "DEC-"): "equatorial", ("GLON", "GLAT"): "galactic"}

#: Tolerance for the rotation being a rigid rotation. The basis construction
#: measures ~2e-16 on this platform, so four orders of headroom.
_ROTATION_ATOL = 1e-12


class FrameDescriptor(NamedTuple):
    """A celestial frame reduced to the primitives that identify it.

    ``frame`` is the resolved astropy frame, kept so the rotation can be
    computed with :class:`~astropy.coordinates.SkyCoord` without resolving the
    WCS twice; it is deliberately excluded from the transform cache key, which
    uses only the primitive fields.
    """

    kind: str
    equinox: float | None
    axis_pair: tuple[str, str]
    frame: object


class ValidatedWcs(NamedTuple):
    """A validated mapprojax kernel plus the descriptor of its pixel grid."""

    mwcs: object
    descriptor: FrameDescriptor


def frame_descriptor(wcs):
    """Classify ``wcs``'s celestial frame, or raise ``ValueError``.

    Only pairs known to be related by a rigid rotation are accepted: ICRS,
    FK5 at an explicit equinox, and Galactic. FK4 (whose E-terms are not a
    rotation) and the ecliptic frames are refused, and so is any CTYPE axis
    pair that does not match the resolved frame — astropy resolves
    ``ELON``/``ELAT`` to ``ICRS``, and resolves by ``wcs.wcs.lng``, which
    accepts swapped axes.
    """
    if wcs.naxis != 2:
        raise ValueError(
            f"expected a two-axis celestial WCS, got naxis = {wcs.naxis}; "
            "pass a two-axis celestial WCS, e.g. wcs.celestial"
        )
    try:
        frame = wcs_to_celestial_frame(wcs)
    except ValueError as exc:
        raise ValueError(
            f"cannot resolve the celestial frame of CTYPE1 = {wcs.wcs.ctype[0]!r}, "
            f"CTYPE2 = {wcs.wcs.ctype[1]!r}: {exc}"
        ) from exc

    axis_pair = (str(wcs.wcs.ctype[0])[:4], str(wcs.wcs.ctype[1])[:4])
    if axis_pair not in _AXIS_PAIRS:
        raise ValueError(
            f"unsupported celestial axis pair {axis_pair} "
            f"(CTYPE {wcs.wcs.ctype[0]!r}, {wcs.wcs.ctype[1]!r}); reprojax needs "
            "RA---/DEC-- or GLON/GLAT"
        )

    kind = _AXIS_PAIRS[axis_pair]
    if type(frame) is ICRS and kind == "equatorial":
        return FrameDescriptor("icrs", None, axis_pair, frame)
    if type(frame) is FK5 and kind == "equatorial":
        equinox = float(frame.equinox.jyear)
        if not np.isfinite(equinox):
            raise ValueError(f"FK5 frame has a non-finite equinox ({frame.equinox!r})")
        return FrameDescriptor("fk5", equinox, axis_pair, frame)
    if type(frame) is Galactic and kind == "galactic":
        return FrameDescriptor("galactic", None, axis_pair, frame)

    raise ValueError(
        f"unsupported celestial frame {frame!r} for axis pair {axis_pair} "
        f"(CTYPE {wcs.wcs.ctype[0]!r}, {wcs.wcs.ctype[1]!r}); reprojax supports "
        "ICRS, FK5 at an explicit equinox, and Galactic"
    )


def validated_wcs(wcs, *, precision):
    """Validate ``wcs`` and return its kernel plus descriptor.

    ``from_astropy`` runs **first**: it checks the caller's object for
    non-finite parameters, and a later ``wcsset()`` trip can rewrite those
    values before the check sees them.
    """
    return ValidatedWcs(from_astropy(wcs, precision=precision), frame_descriptor(wcs))


def _rotation_from_descriptors(desc_from, desc_to):
    """3x3 matrix rotating Cartesian unit vectors from ``desc_from``'s frame to
    ``desc_to``'s, validated as a rigid rotation.
    """
    if desc_from.frame.is_equivalent_frame(desc_to.frame):
        return np.eye(3)

    basis_lon = [0.0, 90.0, 0.0]
    basis_lat = [0.0, 0.0, 90.0]
    cart = SkyCoord(basis_lon, basis_lat, unit="deg", frame=desc_from.frame).transform_to(
        desc_to.frame
    ).cartesian

    # Transformed basis vectors as columns: matrix @ v_from == v_to.
    rotation = np.stack([cart.x.value, cart.y.value, cart.z.value], axis=0)

    residual = np.abs(rotation @ rotation.T - np.eye(3)).max()
    determinant = np.linalg.det(rotation)
    if not np.isfinite(rotation).all() or residual > _ROTATION_ATOL or abs(determinant - 1.0) > _ROTATION_ATOL:
        raise ValueError(
            f"the rotation between {desc_from.kind} and {desc_to.kind} is not rigid "
            f"(orthonormality residual {residual:.3e}, determinant {determinant:.6f}); "
            "these frames are not related by a pure rotation"
        )
    return rotation


def frame_rotation_matrix(wcs_from, wcs_to):
    """3x3 matrix rotating Cartesian unit vectors from ``wcs_from``'s celestial
    frame to ``wcs_to``'s.

    Returns the exact identity when the two frames are equivalent (the common
    case), and a validated rotation otherwise.
    """
    return _rotation_from_descriptors(frame_descriptor(wcs_from), frame_descriptor(wcs_to))


__all__ = [
    "FrameDescriptor",
    "ValidatedWcs",
    "frame_descriptor",
    "frame_rotation_matrix",
    "validated_wcs",
]
