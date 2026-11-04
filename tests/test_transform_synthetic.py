"""Tier-1 validation: mapprojax-composed PixelTransforms vs astropy.wcs.

Two things are checked throughout:
  - pixel -> sky agreement for a single WCS, against astropy, to < 10 uas.
  - composed output-pixel -> input-pixel agreement for a *pair* of WCSes
    (as build_transform produces), against an astropy round-trip
    (all_pix2world -> [frame convert] -> all_world2pix), to < 1e-4 px.

Datasets span rotation/scale/offset, SIP, a frame change (ICRS<->Galactic),
a near-pole pointing, and RA wraparound, per the milestone-1 spec.
"""

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from reprojax._astropy_wcs import projection_code
from reprojax.transform import _mapprojax_wcs, build_transform

from _fixtures import make_header, make_sip

SKY_TOL_UAS = 10.0
PIX_TOL = 1e-4

N_DOMAIN = 1024


def _sample_pixels(n, seed, low=0.0, high=N_DOMAIN - 1.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=n), rng.uniform(low, high, size=n)


def assert_pixel_to_sky_matches_astropy(header, n=500, seed=0, tol_uas=SKY_TOL_UAS):
    wcs = WCS(header)
    mwcs = _mapprojax_wcs(wcs)

    x, y = _sample_pixels(n, seed)
    ra_ref, dec_ref = wcs.all_pix2world(x, y, 0)

    ra_rad, dec_rad = mwcs.unproj(np.asarray(x), np.asarray(y))
    ra, dec = np.degrees(np.asarray(ra_rad)), np.degrees(np.asarray(dec_rad))

    sep = SkyCoord(ra, dec, unit="deg").separation(SkyCoord(ra_ref, dec_ref, unit="deg"))
    assert sep.max().to("uas").value < tol_uas


def assert_composed_transform_matches_astropy(
    header_in, header_out, n=2000, seed=1, tol_px=PIX_TOL, matrix_precision="f64"
):
    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)
    transform = build_transform(wcs_in, wcs_out, matrix_precision=matrix_precision)

    x_out, y_out = _sample_pixels(n, seed)
    x_in, y_in = transform(x_out, y_out)
    x_in, y_in = np.asarray(x_in), np.asarray(y_in)

    sky = SkyCoord(
        *wcs_out.all_pix2world(x_out, y_out, 0), unit="deg", frame=_frame(wcs_out)
    )
    sky_in_frame = sky.transform_to(_frame(wcs_in))
    x_in_ref, y_in_ref = wcs_in.all_world2pix(
        sky_in_frame.spherical.lon.deg, sky_in_frame.spherical.lat.deg, 0
    )

    err = np.hypot(x_in - x_in_ref, y_in - y_in_ref)
    assert np.nanmax(err) < tol_px, f"max composed pixel error {np.nanmax(err):.3e} px"
    return err


def _frame(wcs):
    from astropy.wcs.utils import wcs_to_celestial_frame

    return wcs_to_celestial_frame(wcs)


# ---------------------------------------------------------------------------
# Rotation / scale / offset — no SIP, same frame.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "crpix_out,crval_out,scale_out,rot_out,crpix_in,crval_in,scale_in,rot_in",
    [
        ((512.0, 512.0), (180.0, 30.0), 1.0, 0.0, (512.0, 512.0), (180.0, 30.0), 1.0, 0.0),
        ((512.0, 512.0), (180.0, 30.0), 1.0, 15.0, (600.0, 400.0), (180.05, 30.02), 0.8, -7.0),
        ((300.0, 700.0), (10.0, -45.0), 0.5, 45.0, (512.0, 512.0), (10.0, -45.0), 2.0, 0.0),
    ],
)
def test_rotation_scale_offset(
    crpix_out, crval_out, scale_out, rot_out, crpix_in, crval_in, scale_in, rot_in
):
    h_out = make_header(crpix_out, crval_out, scale_out, rot_deg=rot_out)
    h_in = make_header(crpix_in, crval_in, scale_in, rot_deg=rot_in)
    assert_pixel_to_sky_matches_astropy(h_out)
    assert_pixel_to_sky_matches_astropy(h_in)
    assert_composed_transform_matches_astropy(h_in, h_out)


# ---------------------------------------------------------------------------
# SIP: forward-only path (SIP on the output WCS — unproj_xyz only evaluates
# the forward A/B polynomial, no inversion involved) at full tolerance.
# ---------------------------------------------------------------------------


def test_sip_forward_path_output_side():
    a = {(2, 0): 4e-6, (1, 1): -2e-6, (0, 2): 3e-6}
    b = {(2, 0): -3e-6, (1, 1): 2.5e-6, (0, 2): -2e-6}
    sip = make_sip(a, b, (-512, 512), (-512, 512))

    h_out = make_header((512.0, 512.0), (180.0, 30.0), 1.0, sip=sip)
    h_in = make_header((600.0, 400.0), (180.05, 30.02), 0.8, rot_deg=-7.0)

    assert_pixel_to_sky_matches_astropy(h_out)
    assert_composed_transform_matches_astropy(h_in, h_out)


# ---------------------------------------------------------------------------
# SIP: reverse-polynomial path (SIP on the input WCS, so the input WCS's
# proj applies the SIP inverse). mapprojax's Sip.foc_to_pix uses AP/BP only
# as the initial guess and Newton-solves against the exact forward A/B
# polynomial, which keeps this under the Tier-1 tolerance. The raw AP/BP
# shortfall was first diagnosed on a real SPHEREx L2b header (ticket #4: up
# to 0.03 px vs astropy's iterative inversion); see tests/test_real_spherex.py.
# ---------------------------------------------------------------------------


def _reverse_sip_fixture(scale=1e-6):
    a = {(2, 0): scale, (1, 1): -0.5 * scale, (0, 2): 0.75 * scale}
    b = {(2, 0): -0.75 * scale, (1, 1): 0.6 * scale, (0, 2): -0.5 * scale}
    sip = make_sip(a, b, (-512, 512), (-512, 512))
    h_in = make_header((512.0, 512.0), (180.0, 30.0), 1.0, sip=sip)
    h_out = make_header((600.0, 400.0), (180.05, 30.02), 0.8, rot_deg=-7.0)
    return h_in, h_out


def test_sip_reverse_path_input_side():
    h_in, h_out = _reverse_sip_fixture()
    assert_composed_transform_matches_astropy(h_in, h_out)


# ---------------------------------------------------------------------------
# Frame change: ICRS <-> Galactic.
# ---------------------------------------------------------------------------


def test_icrs_galactic_frame_change():
    icrs_center = SkyCoord(180.05, 30.02, unit="deg", frame="icrs")
    gal_center = icrs_center.galactic

    h_icrs = make_header((600.0, 400.0), (icrs_center.ra.deg, icrs_center.dec.deg), 0.8, rot_deg=-7.0)
    h_gal = make_header(
        (512.0, 512.0),
        (gal_center.l.deg, gal_center.b.deg),
        1.0,
        ctype=("GLON-TAN", "GLAT-TAN"),
    )

    assert_composed_transform_matches_astropy(h_icrs, h_gal)
    assert_composed_transform_matches_astropy(h_gal, h_icrs)


# ---------------------------------------------------------------------------
# Near celestial pole.
# ---------------------------------------------------------------------------


def test_near_pole():
    h_out = make_header((512.0, 512.0), (45.0, 89.5), 2.0)
    h_in = make_header((512.0, 512.0), (45.0, 89.5), 2.0, rot_deg=30.0)
    assert_composed_transform_matches_astropy(h_in, h_out)


# ---------------------------------------------------------------------------
# RA wraparound (crval near the 0/360 boundary).
# ---------------------------------------------------------------------------


def test_ra_wraparound():
    h_out = make_header((512.0, 512.0), (0.05, 10.0), 1.0)
    h_in = make_header((512.0, 512.0), (359.95, 10.0), 1.0)
    assert_composed_transform_matches_astropy(h_in, h_out)


# ---------------------------------------------------------------------------
# SIN projection.
# ---------------------------------------------------------------------------


def test_sin_projection():
    h_out = make_header((512.0, 512.0), (180.0, 30.0), 1.0, rot_deg=10.0, ctype=("RA---SIN", "DEC--SIN"))
    h_in = make_header((600.0, 400.0), (180.02, 30.01), 0.9, rot_deg=-15.0, ctype=("RA---SIN", "DEC--SIN"))
    assert_pixel_to_sky_matches_astropy(h_out)
    assert_composed_transform_matches_astropy(h_in, h_out)


# ---------------------------------------------------------------------------
# Pixel convention: 0-based, pixel-center — verified directly, not assumed.
# ---------------------------------------------------------------------------


def test_pixel_convention_is_0_based_verified_against_astropy():
    h = make_header((512.0, 512.0), (180.0, 30.0), 1.0, rot_deg=12.0)
    wcs = WCS(h)
    mwcs = _mapprojax_wcs(wcs)

    # CRPIX in the header is 1-based (FITS); mapprojax's crpix is CRPIX-1.
    # Evaluating mapprojax's unproj at that 0-based crpix must recover
    # CRVAL exactly, since a pixel->sky WCS must map CRPIX to CRVAL exactly
    # by construction (an unconditional identity, not a general property).
    crpix_0based = np.asarray(wcs.wcs.crpix) - 1.0
    ra, dec = mwcs.unproj(*crpix_0based)
    np.testing.assert_allclose(np.degrees(float(ra)), wcs.wcs.crval[0], atol=1e-10)
    np.testing.assert_allclose(np.degrees(float(dec)), wcs.wcs.crval[1], atol=1e-10)

    # And more generally: astropy's all_pix2world(x, y, origin=0) treats x, y
    # as 0-based already, so calling it with the SAME x, y we feed
    # mapprojax (no shift in either direction) must agree.
    assert_pixel_to_sky_matches_astropy(h)


# ---------------------------------------------------------------------------
# NaN for points outside the input WCS's visible hemisphere.
# ---------------------------------------------------------------------------


def test_invalid_points_return_nan():
    h_in = make_header((512.0, 512.0), (180.0, 30.0), 1.0)
    h_out = make_header((512.0, 512.0), (0.0, -30.0), 1.0, ctype=("RA---SIN", "DEC--SIN"))
    wcs_in, wcs_out = WCS(h_in), WCS(h_out)
    transform = build_transform(wcs_in, wcs_out)

    # output CRVAL (~150 deg from input's CRVAL) is far outside the input
    # TAN WCS's visible hemisphere.
    x_in, y_in = transform(512.0, 512.0)
    assert np.isnan(np.asarray(x_in))
    assert np.isnan(np.asarray(y_in))


# ---------------------------------------------------------------------------
# Unsupported projections raise a clear, informative error.
# ---------------------------------------------------------------------------


def test_unsupported_projection_raises():
    h = make_header((512.0, 512.0), (180.0, 30.0), 1.0, ctype=("RA---AZP", "DEC--AZP"))
    wcs = WCS(h)
    with pytest.raises(NotImplementedError, match="AZP"):
        projection_code(wcs)


# ---------------------------------------------------------------------------
# build_transform caches on WCS content, not object identity: a fresh WCS
# reconstructed from an equivalent header must reuse the same transform
# object (this is what lets reprojax.reproject_adaptive avoid retracing on
# every call for a repeated WCS pair — see transform.py's build_transform
# docstring).
# ---------------------------------------------------------------------------


def test_build_transform_caches_on_content_not_identity():
    build_transform.cache_clear()
    h_in = make_header((512.0, 512.0), (180.0, 30.0), 1.0, rot_deg=7.0)
    h_out = make_header((500.0, 500.0), (180.1, 30.1), 0.9, rot_deg=-3.0)

    t1 = build_transform(WCS(h_in), WCS(h_out))
    # Fresh WCS objects, reconstructed from copies of the same headers.
    t2 = build_transform(WCS(h_in.copy()), WCS(h_out.copy()))
    assert t1 is t2


def test_build_transform_does_not_collide_on_different_wcs():
    build_transform.cache_clear()
    h_in = make_header((512.0, 512.0), (180.0, 30.0), 1.0, rot_deg=7.0)
    h_out = make_header((500.0, 500.0), (180.1, 30.1), 0.9, rot_deg=-3.0)
    h_in_shifted = h_in.copy()
    h_in_shifted["CRVAL1"] += 0.01

    t1 = build_transform(WCS(h_in), WCS(h_out))
    t2 = build_transform(WCS(h_in_shifted), WCS(h_out))
    assert t1 is not t2

    x_in_1, y_in_1 = t1(250.0, 250.0)
    x_in_2, y_in_2 = t2(250.0, 250.0)
    assert not (np.isclose(float(x_in_1), float(x_in_2)) and np.isclose(float(y_in_1), float(y_in_2)))


def test_build_transform_cache_is_bounded():
    build_transform.cache_clear()
    from reprojax.transform import _TRANSFORM_CACHE, _TRANSFORM_CACHE_MAXSIZE

    h_out = make_header((500.0, 500.0), (180.0, 30.0), 0.9)
    for i in range(_TRANSFORM_CACHE_MAXSIZE + 5):
        h_in = make_header((512.0, 512.0), (180.0 + i * 0.01, 30.0), 1.0)
        build_transform(WCS(h_in), WCS(h_out))

    assert len(_TRANSFORM_CACHE) <= _TRANSFORM_CACHE_MAXSIZE


# ---------------------------------------------------------------------------
# matrix_precision="f32": mapprojax.reproject.Reprojector's fp64-free path
# (see transform.py's compose_transform_fp32). Bounds below have headroom
# over what was actually measured on these fixtures and on the real SPHEREx
# pair (tests/test_real_spherex.py) -- not copied from mapprojax's own
# synthetic report numbers, since this project's SIP/frame-change specifics
# hadn't been checked against that mode until now.
# ---------------------------------------------------------------------------


def test_f32_matches_astropy_looser_tolerance():
    h_out = make_header((512.0, 512.0), (180.0, 30.0), 1.0, rot_deg=15.0)
    h_in = make_header((600.0, 400.0), (180.05, 30.02), 0.8, rot_deg=-7.0)
    assert_composed_transform_matches_astropy(h_in, h_out, tol_px=5e-3, matrix_precision="f32")


def test_f32_matches_astropy_with_sip():
    h_in, h_out = _reverse_sip_fixture()
    assert_composed_transform_matches_astropy(h_in, h_out, tol_px=5e-3, matrix_precision="f32")


def test_f32_matches_astropy_across_frame_change():
    icrs_center = SkyCoord(180.05, 30.02, unit="deg", frame="icrs")
    gal_center = icrs_center.galactic
    h_icrs = make_header((600.0, 400.0), (icrs_center.ra.deg, icrs_center.dec.deg), 0.8, rot_deg=-7.0)
    h_gal = make_header(
        (512.0, 512.0), (gal_center.l.deg, gal_center.b.deg), 1.0, ctype=("GLON-TAN", "GLAT-TAN")
    )
    assert_composed_transform_matches_astropy(h_icrs, h_gal, tol_px=5e-3, matrix_precision="f32")
    assert_composed_transform_matches_astropy(h_gal, h_icrs, tol_px=5e-3, matrix_precision="f32")


def test_f32_works_without_jax_enable_x64():
    """The whole point of matrix_precision="f32": correct (within its own
    looser bound) regardless of jax_enable_x64, unlike "f64".
    """
    import jax

    h_out = make_header((512.0, 512.0), (180.0, 30.0), 1.0, rot_deg=15.0)
    h_in = make_header((600.0, 400.0), (180.05, 30.02), 0.8, rot_deg=-7.0)

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        build_transform.cache_clear()
        assert_composed_transform_matches_astropy(h_in, h_out, tol_px=5e-3, matrix_precision="f32")
    finally:
        jax.config.update("jax_enable_x64", prev)
        build_transform.cache_clear()


def test_f32_invalid_points_return_nan():
    """Same far-hemisphere NaN contract as the f64 path (compose_transform_fp32
    mirrors compose_transform's masking).
    """
    h_in = make_header((512.0, 512.0), (180.0, 30.0), 1.0)
    h_out = make_header((512.0, 512.0), (0.0, -30.0), 1.0, ctype=("RA---SIN", "DEC--SIN"))
    wcs_in, wcs_out = WCS(h_in), WCS(h_out)
    transform = build_transform(wcs_in, wcs_out, matrix_precision="f32")

    x_in, y_in = transform(512.0, 512.0)
    assert np.isnan(np.asarray(x_in))
    assert np.isnan(np.asarray(y_in))


def test_invalid_matrix_precision_raises():
    h = make_header((512.0, 512.0), (180.0, 30.0), 1.0)
    wcs = WCS(h)
    with pytest.raises(ValueError):
        build_transform(wcs, wcs, matrix_precision="fp16")


def test_cache_distinguishes_matrix_precision():
    """f64 and f32 transforms for the same WCS pair must not collide in the
    cache -- they compute genuinely different things.
    """
    build_transform.cache_clear()
    h_out = make_header((512.0, 512.0), (180.0, 30.0), 1.0, rot_deg=15.0)
    h_in = make_header((600.0, 400.0), (180.05, 30.02), 0.8, rot_deg=-7.0)
    wcs_in, wcs_out = WCS(h_in), WCS(h_out)

    t64 = build_transform(wcs_in, wcs_out, matrix_precision="f64")
    t32 = build_transform(wcs_in, wcs_out, matrix_precision="f32")
    assert t64 is not t32

    x64, y64 = t64(250.0, 250.0)
    x32, y32 = t32(250.0, 250.0)
    # Both should be close (same underlying geometry), but not bit-identical
    # -- if they were, the cache-key change wouldn't be doing anything.
    assert not (float(x64) == float(x32) and float(y64) == float(y32))
