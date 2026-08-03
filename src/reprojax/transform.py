"""Build a :class:`~reprojax.core.PixelTransform` from a pair of astropy WCS objects.

mapprojax provides the projection math (TAN, SIN, with optional SIP
distortion); astropy provides celestial frame determination and rotation.
Everything astropy-facing runs once, eagerly, at build time in
:func:`build_transform`; the returned callable is pure JAX and safe under
``jit``/``vmap``/``grad``.
"""

import jax.numpy as jnp
from mapprojax.jax_projections import SinJax, TanJax
from mapprojax.utils import apply_rotation

from ._astropy_wcs import frame_rotation_matrix, projection_code, wcs_params_from_astropy
from ._sip_refine import DEFAULT_REFINE_STEPS, refine_pixel_estimate

_PROVIDERS = {"TAN": TanJax, "SIN": SinJax}


def _mapprojax_wcs(astropy_wcs):
    code = projection_code(astropy_wcs)
    crpix, cd, crval, sip = wcs_params_from_astropy(astropy_wcs)
    return _PROVIDERS[code](crpix, cd, crval, sip=sip)


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
    """
    mwcs_in = _mapprojax_wcs(wcs_in)
    mwcs_out = _mapprojax_wcs(wcs_out)
    rotation = jnp.asarray(frame_rotation_matrix(wcs_out, wcs_in))
    refine_steps = sip_inverse_refine_steps if wcs_in.sip is not None else 0

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
        x_in, y_in = mwcs_in.native_to_pix(xn, yn, zn)

        if refine_steps > 0:
            x_in, y_in = refine_pixel_estimate(mwcs_in, x_in, y_in, xn, yn, zn, n_steps=refine_steps)

        x_in = jnp.where(valid, x_in, jnp.nan)
        y_in = jnp.where(valid, y_in, jnp.nan)
        return x_in, y_in

    return transform
