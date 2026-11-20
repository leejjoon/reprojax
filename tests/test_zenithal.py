"""ZEA, STG and ARC end to end (mapprojax 0.5.0's zenithal projections).

reprojax builds its kernels through ``mapprojax.from_astropy``, so widening
the mapprojax cap to ``<0.6`` makes these projections reachable here. The
checks mirror the TAN ones in ``test_transform_synthetic.py``: TAN<->ZEA in
both directions and both matrix precisions against an astropy round trip, an
image through ``reproject_interp`` against the original ``reproject``, and
construction and cache smoke tests for ARC and STG.
"""

import numpy as np
import pytest
from astropy.wcs import WCS
from reproject import reproject_interp as original_reproject_interp

from reprojax import reproject_interp
from reprojax.transform import build_transform

from _fixtures import make_header
from test_transform_synthetic import (
    PIX_TOL,
    assert_composed_transform_matches_astropy,
    assert_pixel_to_sky_matches_astropy,
)

ZENITHAL = ("ZEA", "STG", "ARC")
F32_PIX_TOL = 5e-3  # as for TAN in test_transform_synthetic.py


def _header(code, crpix, crval, scale_arcsec, rot_deg=0.0, naxis=(1024, 1024)):
    return make_header(crpix, crval, scale_arcsec, rot_deg=rot_deg,
                       ctype=(f"RA---{code}", f"DEC--{code}"), naxis=naxis)


@pytest.mark.parametrize("code", ZENITHAL)
def test_pixel_to_sky_matches_astropy(code):
    assert_pixel_to_sky_matches_astropy(_header(code, (512.0, 512.0), (150.0, 30.0), 1.0, rot_deg=12.0))


@pytest.mark.parametrize("matrix_precision", ["f64", "f32"])
@pytest.mark.parametrize("direction", ["tan_to_zea", "zea_to_tan"])
def test_tan_zea_composed_transform_matches_astropy(direction, matrix_precision):
    tan = make_header((512.0, 512.0), (150.0, 30.0), 1.0, rot_deg=-5.0)
    zea = _header("ZEA", (600.0, 450.0), (150.05, 30.03), 1.3, rot_deg=10.0)
    h_in, h_out = (tan, zea) if direction == "tan_to_zea" else (zea, tan)
    tol = PIX_TOL if matrix_precision == "f64" else F32_PIX_TOL
    assert_composed_transform_matches_astropy(h_in, h_out, tol_px=tol,
                                              matrix_precision=matrix_precision)


@pytest.mark.parametrize("code", ["STG", "ARC"])
def test_other_zenithals_compose_against_tan(code):
    tan = make_header((512.0, 512.0), (150.0, 30.0), 1.0)
    other = _header(code, (500.0, 520.0), (150.02, 29.98), 1.1, rot_deg=4.0)
    assert_composed_transform_matches_astropy(tan, other)
    assert_composed_transform_matches_astropy(other, tan)


def test_zea_image_matches_original_reproject():
    h_in = _header("ZEA", (100.0, 100.0), (150.0, 30.0), 1.0, naxis=(200, 200))
    h_out = make_header((90.0, 110.0), (150.01, 30.005), 1.2, rot_deg=8.0, naxis=(180, 220))
    data = np.random.default_rng(0).standard_normal((200, 200))

    arr, footprint = reproject_interp((data, h_in), h_out)
    ref, ref_footprint = original_reproject_interp((data, h_in), h_out, shape_out=(220, 180),
                                                   order="bilinear")
    valid = np.asarray(footprint).astype(bool) & np.asarray(ref_footprint).astype(bool)
    assert valid.sum() > 1000
    np.testing.assert_allclose(np.asarray(arr)[valid], np.asarray(ref)[valid], rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("code", ["STG", "ARC"])
def test_construction_and_cache(code):
    build_transform.cache_clear()
    # 60"/px (a 17 degree field), so TAN and the zenithal clearly differ.
    h_in = _header(code, (512.0, 512.0), (150.0, 30.0), 60.0)
    h_out = make_header((500.0, 500.0), (151.0, 31.0), 50.0)
    t1 = build_transform(WCS(h_in), WCS(h_out))
    assert build_transform(WCS(h_in.copy()), WCS(h_out.copy())) is t1

    # Same numbers, different projection: must not share a cache entry.
    h_tan = h_in.copy()
    h_tan["CTYPE1"], h_tan["CTYPE2"] = "RA---TAN", "DEC--TAN"
    t_tan = build_transform(WCS(h_tan), WCS(h_out))
    assert t_tan is not t1
    x1, y1 = t1(np.asarray([900.0]), np.asarray([100.0]))
    x2, y2 = t_tan(np.asarray([900.0]), np.asarray([100.0]))
    assert np.hypot(float(x1[0]) - float(x2[0]), float(y1[0]) - float(y2[0])) > 0.1


@pytest.mark.parametrize("ctype", [("RA---AZP", "DEC--AZP"), ("RA---ZPN", "DEC--ZPN")])
def test_parametrised_zenithals_are_still_refused(ctype):
    h = make_header((512.0, 512.0), (150.0, 30.0), 1.0, ctype=ctype)
    tan = make_header((512.0, 512.0), (150.0, 30.0), 1.0)
    with pytest.raises(ValueError):
        build_transform(WCS(h), WCS(tan))
