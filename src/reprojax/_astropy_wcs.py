"""Host-side extraction of mapprojax WCS parameters and frame rotations from astropy.

Everything in this module runs eagerly in NumPy/astropy, at transform-build
time — never inside the jit-compiled :class:`~reprojax.core.PixelTransform`
itself.
"""

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import wcs_to_celestial_frame
from mapprojax.sip import Sip

#: FITS projection code -> mapprojax provider key (see transform.py).
SUPPORTED_PROJECTIONS = ("TAN", "SIN")


def projection_code(wcs):
    """The 3-letter FITS projection code for an astropy WCS's first axis.

    Per the WCS papers, CTYPEia's first 8 characters are always
    ``"<axis>-<code>"`` with the code in the last 3 characters of that
    8-character field — true whether or not a further distortion suffix
    (``-SIP``, ``-TPV``, ...) follows.
    """
    ctype = wcs.wcs.ctype[0]
    code = ctype[:8][-3:].strip()
    if code not in SUPPORTED_PROJECTIONS:
        raise NotImplementedError(
            f"Unsupported projection {code!r} (from CTYPE {ctype!r}); "
            f"reprojax currently supports: {SUPPORTED_PROJECTIONS}"
        )
    return code


def wcs_params_from_astropy(wcs):
    """Extract ``(crpix, cd, crval, sip)`` from an astropy WCS.

    ``crpix`` is converted to mapprojax's 0-based convention (astropy's
    ``wcs.wcs.crpix`` is 1-based, matching the FITS ``CRPIXi`` keywords).
    ``cd`` is the CD-matrix-equivalent linear transform regardless of
    whether the header used ``CDi_j`` or ``PCi_j``/``CDELTi`` keywords.
    """
    crpix = (np.asarray(wcs.wcs.crpix, dtype=float) - 1.0).tolist()
    crval = np.asarray(wcs.wcs.crval, dtype=float).tolist()
    cd = np.asarray(wcs.pixel_scale_matrix, dtype=float).tolist()

    sip = None
    if wcs.sip is not None:
        sip = Sip(a=wcs.sip.a, b=wcs.sip.b, ap=wcs.sip.ap, bp=wcs.sip.bp)

    return crpix, cd, crval, sip


def frame_rotation_matrix(wcs_from, wcs_to):
    """3x3 matrix rotating Cartesian unit vectors from ``wcs_from``'s celestial
    frame to ``wcs_to``'s.

    Returns the identity when the two WCSes share the same celestial frame
    (the common case, and the only case that avoids an astropy frame
    transform).
    """
    frame_from = wcs_to_celestial_frame(wcs_from)
    frame_to = wcs_to_celestial_frame(wcs_to)

    if frame_from.name == frame_to.name:
        return np.eye(3)

    basis_lon = [0.0, 90.0, 0.0]
    basis_lat = [0.0, 0.0, 90.0]
    basis = SkyCoord(basis_lon, basis_lat, unit="deg", frame=frame_from)
    cart = basis.transform_to(frame_to).cartesian

    # Transformed basis vectors as columns: matrix @ v_from == v_to.
    return np.stack([cart.x.value, cart.y.value, cart.z.value], axis=0)
