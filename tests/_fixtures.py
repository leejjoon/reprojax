"""Synthetic FITS header builders and SIP fixtures for the Tier-1 test suite.

Not part of the public package: real FITS files ship pre-fit AP/BP
coefficients, so production code never needs to fit a SIP inverse. Here we
fit one the same way real SIP pipelines do (least squares on sampled
forward-distorted points) purely to build realistic, honest test fixtures.
"""

import numpy as np
from astropy.io import fits


def make_header(
    crpix,
    crval,
    scale_arcsec,
    rot_deg=0.0,
    ctype=("RA---TAN", "DEC--TAN"),
    naxis=(1024, 1024),
    sip=None,
):
    """A minimal synthetic WCS header: CRPIX/CRVAL/CD (+ optional SIP)."""
    h = fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"], h["NAXIS2"] = naxis
    h["CTYPE1"], h["CTYPE2"] = ctype
    h["CRPIX1"], h["CRPIX2"] = crpix
    h["CRVAL1"], h["CRVAL2"] = crval

    scale = scale_arcsec / 3600.0
    theta = np.radians(rot_deg)
    cd = scale * np.array(
        [[-np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    h["CD1_1"], h["CD1_2"] = cd[0]
    h["CD2_1"], h["CD2_2"] = cd[1]

    if sip is not None:
        h["CTYPE1"] = h["CTYPE1"] + "-SIP"
        h["CTYPE2"] = h["CTYPE2"] + "-SIP"
        a, b, ap, bp = sip
        _write_sip_terms(h, "A", a)
        _write_sip_terms(h, "B", b)
        _write_sip_terms(h, "AP", ap)
        _write_sip_terms(h, "BP", bp)

    return h


def _write_sip_terms(header, prefix, coeffs):
    order = max(p + q for p, q in coeffs)
    header[f"{prefix}_ORDER"] = order
    for (p, q), value in coeffs.items():
        header[f"{prefix}_{p}_{q}"] = value


def _eval_poly(coeffs, u, v):
    return sum(c * u**p * v**q for (p, q), c in coeffs.items())


def fit_sip_reverse(order, a, b, u_range, v_range, n=40):
    """Least-squares fit of AP/BP (as functions of the distorted coords)
    approximating the inverse of forward SIP polynomials a, b — the same
    approach real SIP-fitting pipelines use to derive reverse coefficients.
    """
    us = np.linspace(*u_range, n)
    vs = np.linspace(*v_range, n)
    U, V = np.meshgrid(us, vs)
    U, V = U.ravel(), V.ravel()

    Up = U + _eval_poly(a, U, V)
    Vp = V + _eval_poly(b, U, V)

    terms = [(p, q) for p in range(order + 1) for q in range(order + 1 - p)]
    design = np.stack([Up**p * Vp**q for p, q in terms], axis=1)

    coef_ap, *_ = np.linalg.lstsq(design, U - Up, rcond=None)
    coef_bp, *_ = np.linalg.lstsq(design, V - Vp, rcond=None)

    ap = {t: c for t, c in zip(terms, coef_ap)}
    bp = {t: c for t, c in zip(terms, coef_bp)}
    return ap, bp


def make_sip(a, b, u_range, v_range):
    """Build a full (a, b, ap, bp) SIP fixture, fitting ap/bp from a, b."""
    order = max(p + q for p, q in {**a, **b})
    ap, bp = fit_sip_reverse(order, a, b, u_range, v_range)
    return a, b, ap, bp
