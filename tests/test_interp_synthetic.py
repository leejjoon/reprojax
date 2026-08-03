"""Tier-2 validation: reprojax.reproject_interp vs astropy/reproject's reproject_interp.

Mirrors the Tier-1 synthetic suite (tests/test_transform_synthetic.py) at the
image level: for each scenario, footprints must match exactly and jointly-
valid pixel values must agree to rtol <= 1e-8 (the milestone-1 bar) — except
where a documented, understood reason applies (the SIP-reverse-polynomial
path; see test_sip_reverse_path_input_side below).
"""

import pathlib
import subprocess
import sys

import jax
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp as original_reproject_interp

from reprojax import reproject_interp
from reprojax.interp import reproject_interp_core
from reprojax.transform import build_transform

from _fixtures import make_header, make_sip

RTOL = 1e-8
ATOL = 1e-9

_ORDER_NAME = {0: "nearest-neighbor", 1: "bilinear"}


def _random_data(header, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(header["NAXIS2"], header["NAXIS1"]))


def assert_matches_original(header_in, header_out, shape_out, order=1, data=None, seed=0, rtol=RTOL, atol=ATOL):
    wcs_in, wcs_out = WCS(header_in), WCS(header_out)
    if data is None:
        data = _random_data(header_in, seed=seed)

    arr, footprint = reproject_interp((data, wcs_in), wcs_out, shape_out=shape_out, order=order)
    arr, footprint = np.asarray(arr), np.asarray(footprint)

    arr_ref, footprint_ref = original_reproject_interp(
        (data, wcs_in), wcs_out, shape_out=shape_out, order=_ORDER_NAME[order]
    )

    assert np.array_equal(footprint.astype(bool), footprint_ref.astype(bool)), "footprints differ"

    valid = footprint.astype(bool) & footprint_ref.astype(bool)
    assert valid.any(), "test scenario has no overlap between input and output fields"
    np.testing.assert_allclose(arr[valid], arr_ref[valid], rtol=rtol, atol=atol)
    return arr, footprint


# ---------------------------------------------------------------------------
# Rotation / scale / offset — no SIP, same frame.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "crpix_out,crval_out,scale_out,rot_out,crpix_in,crval_in,scale_in,rot_in",
    [
        ((100.0, 100.0), (180.0, 30.0), 1.0, 0.0, (100.0, 100.0), (180.0, 30.0), 1.0, 0.0),
        ((90.0, 110.0), (180.01, 30.005), 1.2, 8.0, (100.0, 100.0), (180.0, 30.0), 1.0, 0.0),
    ],
)
def test_rotation_scale_offset(
    crpix_out, crval_out, scale_out, rot_out, crpix_in, crval_in, scale_in, rot_in
):
    h_out = make_header(crpix_out, crval_out, scale_out, rot_deg=rot_out, naxis=(200, 200))
    h_in = make_header(crpix_in, crval_in, scale_in, rot_deg=rot_in, naxis=(200, 200))
    assert_matches_original(h_in, h_out, (200, 200))


# ---------------------------------------------------------------------------
# SIP: forward-only path (SIP on the output WCS) at full tolerance — see
# tests/test_transform_synthetic.py::test_sip_forward_path_output_side for
# why this path carries no inversion-approximation error.
# ---------------------------------------------------------------------------


def test_sip_forward_path_output_side():
    a = {(2, 0): 4e-6, (1, 1): -2e-6, (0, 2): 3e-6}
    b = {(2, 0): -3e-6, (1, 1): 2.5e-6, (0, 2): -2e-6}
    sip = make_sip(a, b, (-100, 100), (-100, 100))

    h_out = make_header((100.0, 100.0), (180.0, 30.0), 1.0, sip=sip, naxis=(200, 200))
    h_in = make_header((100.0, 100.0), (180.0, 30.0), 1.0, naxis=(200, 200))
    assert_matches_original(h_in, h_out, (200, 200))


# ---------------------------------------------------------------------------
# SIP: reverse-polynomial path (SIP on the input WCS) — documented, looser
# tolerance.
#
# Ticket #2 established that mapprojax's AP/BP evaluation (a direct, fitted
# approximation of the true SIP inverse — see reprojax._astropy_wcs) agrees
# with astropy's iterative all_world2pix to ~1.5e-5 px at this distortion
# magnitude, comfortably under the *transform*-level 1e-4 px bar. At the
# *image* level, that residual position error shows up as a value error of
# roughly the same order as the local pixel-to-pixel gradient of the data
# (here, differences of order 1 for uniform random data), i.e. ~1e-5 — three
# orders of magnitude above the 1e-8 rtol bar used elsewhere in this suite.
# This is not a bug: it is the direct, expected consequence of the SIP
# reverse-polynomial being an approximate inverse, diagnosed in ticket #2.
# Tightening this test would require either negligible SIP coefficients
# (defeating the point of testing the SIP path) or the AP/BP iterative
# refinement that ticket #4 adds when validating real SPHEREx headers, which
# have their own, independently-measured inversion quality. So we assert a
# tolerance set with margin above the diagnosed residual instead of the
# blanket rtol=1e-8.
# ---------------------------------------------------------------------------


def test_sip_reverse_path_input_side():
    scale = 2e-7
    a = {(2, 0): scale, (1, 1): -0.5 * scale, (0, 2): 0.75 * scale}
    b = {(2, 0): -0.75 * scale, (1, 1): 0.6 * scale, (0, 2): -0.5 * scale}
    sip = make_sip(a, b, (-512, 512), (-512, 512))

    h_in = make_header((512.0, 512.0), (180.0, 30.0), 1.0, sip=sip, naxis=(1024, 1024))
    h_out = make_header((100.0, 100.0), (180.02, 30.01), 0.9, rot_deg=-15.0, naxis=(200, 200))
    assert_matches_original(h_in, h_out, (200, 200), rtol=0, atol=2e-5)


# ---------------------------------------------------------------------------
# Frame change: ICRS <-> Galactic.
# ---------------------------------------------------------------------------


def test_icrs_galactic_frame_change():
    icrs_center = SkyCoord(180.05, 30.02, unit="deg", frame="icrs")
    gal_center = icrs_center.galactic

    h_icrs = make_header(
        (100.0, 100.0), (icrs_center.ra.deg, icrs_center.dec.deg), 0.8, rot_deg=-7.0, naxis=(200, 200)
    )
    h_gal = make_header(
        (100.0, 100.0), (gal_center.l.deg, gal_center.b.deg), 1.0, ctype=("GLON-TAN", "GLAT-TAN"), naxis=(200, 200)
    )
    assert_matches_original(h_icrs, h_gal, (200, 200))


# ---------------------------------------------------------------------------
# Near celestial pole.
# ---------------------------------------------------------------------------


def test_near_pole():
    h_out = make_header((100.0, 100.0), (45.0, 89.5), 2.0, naxis=(200, 200))
    h_in = make_header((100.0, 100.0), (45.0, 89.5), 2.0, rot_deg=30.0, naxis=(200, 200))
    assert_matches_original(h_in, h_out, (200, 200))


# ---------------------------------------------------------------------------
# RA wraparound (crval near the 0/360 boundary).
# ---------------------------------------------------------------------------


def test_ra_wraparound():
    h_out = make_header((100.0, 100.0), (0.01, 10.0), 1.0, naxis=(200, 200))
    h_in = make_header((100.0, 100.0), (359.99, 10.0), 1.0, naxis=(200, 200))
    assert_matches_original(h_in, h_out, (200, 200))


# ---------------------------------------------------------------------------
# SIN projection.
# ---------------------------------------------------------------------------


def test_sin_projection():
    h_out = make_header(
        (100.0, 100.0), (180.0, 30.0), 1.0, rot_deg=10.0, ctype=("RA---SIN", "DEC--SIN"), naxis=(200, 200)
    )
    h_in = make_header(
        (100.0, 100.0), (180.02, 30.01), 0.9, rot_deg=-15.0, ctype=("RA---SIN", "DEC--SIN"), naxis=(200, 200)
    )
    assert_matches_original(h_in, h_out, (200, 200))


# ---------------------------------------------------------------------------
# Nearest-neighbor (order 0).
# ---------------------------------------------------------------------------


def test_nearest_neighbor_order():
    h_out = make_header((90.0, 110.0), (180.01, 30.005), 1.2, rot_deg=8.0, naxis=(200, 200))
    h_in = make_header((100.0, 100.0), (180.0, 30.0), 1.0, naxis=(200, 200))
    assert_matches_original(h_in, h_out, (200, 200), order=0, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Raw FITS headers (not just WCS objects) are an accepted input/output form,
# with shape_out inferred from the output header's NAXIS1/NAXIS2 when omitted.
# ---------------------------------------------------------------------------


def test_accepts_raw_headers_and_infers_shape_out():
    h_in = make_header((100.0, 100.0), (180.0, 30.0), 1.0, naxis=(200, 200))
    # naxis=(NAXIS1, NAXIS2)=(nx, ny) -> inferred shape_out is (ny, nx)=(220, 180).
    h_out = make_header((90.0, 110.0), (180.01, 30.005), 1.2, rot_deg=8.0, naxis=(180, 220))
    data = _random_data(h_in)

    arr, footprint = reproject_interp((data, h_in), h_out)
    assert arr.shape == (220, 180)
    assert footprint.shape == (220, 180)

    arr_ref, footprint_ref = original_reproject_interp((data, h_in), h_out, shape_out=(220, 180), order="bilinear")
    valid = np.asarray(footprint).astype(bool) & np.asarray(footprint_ref).astype(bool)
    assert valid.any()
    np.testing.assert_allclose(np.asarray(arr)[valid], np.asarray(arr_ref)[valid], rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# dtype passthrough: image data keeps its incoming dtype; coordinates are
# float64 regardless.
# ---------------------------------------------------------------------------


def test_dtype_passthrough():
    h_in = make_header((100.0, 100.0), (180.0, 30.0), 1.0, naxis=(200, 200))
    h_out = make_header((90.0, 110.0), (180.01, 30.005), 1.2, rot_deg=8.0, naxis=(200, 200))
    data32 = _random_data(h_in).astype(np.float32)

    arr, footprint = reproject_interp((data32, WCS(h_in)), WCS(h_out), shape_out=(200, 200))
    assert arr.dtype == np.float32
    assert footprint.dtype == np.float32


# ---------------------------------------------------------------------------
# Footprint semantics: NaN <-> footprint 0, finite <-> footprint 1.
# ---------------------------------------------------------------------------


def test_footprint_semantics_are_self_consistent():
    h_in = make_header((100.0, 100.0), (180.0, 30.0), 1.0, naxis=(200, 200))
    h_out = make_header((10.0, 10.0), (180.03, 30.02), 1.0, naxis=(200, 200))  # mostly off-field
    data = _random_data(h_in)

    arr, footprint = reproject_interp((data, WCS(h_in)), WCS(h_out), shape_out=(200, 200))
    arr, footprint = np.asarray(arr), np.asarray(footprint)

    assert footprint.min() == 0.0 and footprint.max() == 1.0  # scenario actually exercises both
    np.testing.assert_array_equal(np.isnan(arr), footprint == 0.0)


# ---------------------------------------------------------------------------
# Unsupported order / projection raise informative errors.
# ---------------------------------------------------------------------------


def test_unsupported_order_raises():
    h_in = make_header((100.0, 100.0), (180.0, 30.0), 1.0, naxis=(200, 200))
    h_out = make_header((100.0, 100.0), (180.0, 30.0), 1.0, naxis=(200, 200))
    with pytest.raises(ValueError, match="order"):
        reproject_interp((_random_data(h_in), WCS(h_in)), WCS(h_out), shape_out=(200, 200), order=3)


def test_unsupported_projection_raises():
    h_in = make_header((100.0, 100.0), (180.0, 30.0), 1.0, ctype=("RA---AZP", "DEC--AZP"), naxis=(200, 200))
    h_out = make_header((100.0, 100.0), (180.0, 30.0), 1.0, naxis=(200, 200))
    with pytest.raises(NotImplementedError, match="AZP"):
        reproject_interp((_random_data(h_in), WCS(h_in)), WCS(h_out), shape_out=(200, 200))


# ---------------------------------------------------------------------------
# The core is a jit-able pure function of arrays + transform.
# ---------------------------------------------------------------------------


def test_core_is_jit_able():
    h_in = make_header((100.0, 100.0), (180.0, 30.0), 1.0, naxis=(200, 200))
    h_out = make_header((90.0, 110.0), (180.01, 30.005), 1.2, rot_deg=8.0, naxis=(200, 200))
    data = jax.numpy.asarray(_random_data(h_in))
    transform = build_transform(WCS(h_in), WCS(h_out))

    ny, nx = 200, 200
    y_out, x_out = jax.numpy.meshgrid(
        jax.numpy.arange(ny, dtype=jax.numpy.float64), jax.numpy.arange(nx, dtype=jax.numpy.float64), indexing="ij"
    )

    eager = reproject_interp_core(data, x_out, y_out, transform, order=1)
    jitted = jax.jit(reproject_interp_core, static_argnames=("transform", "order"))(
        data, x_out, y_out, transform, order=1
    )

    # rtol accounts for ordinary eager-vs-compiled floating-point fusion
    # differences (~1e-14 here), not a correctness gap.
    np.testing.assert_allclose(np.asarray(eager[0]), np.asarray(jitted[0]), rtol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(np.asarray(eager[1]), np.asarray(jitted[1]))


# ---------------------------------------------------------------------------
# x64-disabled: warns, does not raise; the library never mutates global
# JAX config.
# ---------------------------------------------------------------------------


def test_warns_on_x64_disabled_and_never_mutates_jax_config():
    result = subprocess.run(
        [sys.executable, "-c", "import warnings; warnings.simplefilter('always'); import reprojax"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "64-bit precision is not enabled" in result.stderr

    import reprojax

    pkg_dir = pathlib.Path(reprojax.__file__).parent
    for path in pkg_dir.rglob("*.py"):
        assert "config.update" not in path.read_text(), f"{path} mutates global JAX config"
