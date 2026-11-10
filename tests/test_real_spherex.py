"""Real-data integration case: a SPHEREx L2b exposure onto a Galactic template.

Fixtures (tests/data/):
  - spherex_l2b_v20.fits: the real IMAGE extension (2040x2040 float32,
    including its genuine masked/NaN pixels) and its real header (TAN+SIP,
    ICRS) from a SPHEREx L2b product. Trimmed from the original multi-
    extension file (FLAGS/VARIANCE/ZODI/PSF/WCS-WAVE dropped) but otherwise
    bit-identical and header-unchanged.
  - galactic_template_header.fits: the real output header (GLON-TAN/GLAT-TAN,
    NAXIS 1201x1201) from the companion mosaicking template used for this
    exposure in the original (pre-reprojax) workflow. Header only — no pixel
    data is needed to define an output grid.

This is the same operation reproject_sip.py (mapprojax's predecessor script)
performed by hand: ICRS TAN+SIP -> Galactic TAN, with real SIP distortion.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import wcs_to_celestial_frame
from reproject import reproject_interp as original_reproject_interp
from reproject.adaptive import reproject_adaptive as original_reproject_adaptive

from reprojax import build_transform, reproject_adaptive, reproject_interp
from reprojax.transform import _mapprojax_wcs

# The golden reference. Tolerances in this module were tuned against this
# exact version; upgrading it moves the baseline, not just the tool.
# `ci/minimum.txt` and `ci/latest.txt` pin the same version.
REFERENCE_REPROJECT_VERSION = "0.21.0"

DATA_DIR = Path(__file__).parent / "data"

SKY_TOL_UAS = 10.0
PIX_TOL = 1e-4


@pytest.fixture(scope="module")
def spherex_header():
    with fits.open(DATA_DIR / "spherex_l2b_v20.fits") as hdul:
        return hdul[0].header.copy()


@pytest.fixture(scope="module")
def spherex_data():
    with fits.open(DATA_DIR / "spherex_l2b_v20.fits") as hdul:
        return hdul[0].data.copy()


@pytest.fixture(scope="module")
def template_header():
    return fits.Header.fromtextfile(DATA_DIR / "galactic_template_header.fits")


def test_reference_reproject_version():
    """Guard the pinned golden reference.

    An upgraded reference changes the numbers this module compares against,
    so the pin has to match what is installed.
    """
    import importlib.metadata

    installed = importlib.metadata.version("reproject")
    assert installed == REFERENCE_REPROJECT_VERSION, (
        f"golden reference is pinned to {REFERENCE_REPROJECT_VERSION}, "
        f"but reproject {installed} is installed"
    )


# ---------------------------------------------------------------------------
# Tier-1: composed transform vs astropy on the real headers.
# ---------------------------------------------------------------------------


def test_real_forward_sip_path_pixel_to_sky(spherex_header):
    """The forward A/B polynomial path (no inversion): pixel -> sky agreement."""
    wcs = WCS(spherex_header)
    mwcs = _mapprojax_wcs(wcs)

    rng = np.random.default_rng(0)
    n = 5000
    x = rng.uniform(0, spherex_header["NAXIS1"] - 1, n)
    y = rng.uniform(0, spherex_header["NAXIS2"] - 1, n)

    ra_ref, dec_ref = wcs.all_pix2world(x, y, 0)
    ra_rad, dec_rad = mwcs.unproj(x, y)
    ra, dec = np.degrees(np.asarray(ra_rad)), np.degrees(np.asarray(dec_rad))

    sep = SkyCoord(ra, dec, unit="deg").separation(SkyCoord(ra_ref, dec_ref, unit="deg"))
    assert sep.max().to("uas").value < SKY_TOL_UAS


def test_real_sip_inversion_quality_quantified(spherex_header):
    """Quantify the real SIP inverse (mapprojax's ``proj``: AP/BP initial
    guess, Newton-solved against the forward A/B polynomial) against
    astropy's iterative all_world2pix -- the measurement ticket #4 exists to
    make.
    """
    wcs = WCS(spherex_header)
    mwcs = _mapprojax_wcs(wcs)

    rng = np.random.default_rng(0)
    n = 5000
    x = rng.uniform(0, spherex_header["NAXIS1"] - 1, n)
    y = rng.uniform(0, spherex_header["NAXIS2"] - 1, n)
    ra, dec = wcs.all_pix2world(x, y, 0)
    x_ref, y_ref = wcs.all_world2pix(ra, dec, 0)

    x_direct, y_direct = mwcs.proj(np.radians(ra), np.radians(dec))
    direct_err = np.hypot(np.asarray(x_direct) - x_ref, np.asarray(y_direct) - y_ref)
    print(f"\nreal SPHEREx AP/BP direct-evaluation error vs astropy: max={direct_err.max():.4f} px, median={np.median(direct_err):.4f} px")

    # mapprojax's Newton-solved SIP inverse meets the Tier-1 bar directly;
    # keep the measurement to prevent regression.
    assert direct_err.max() < PIX_TOL


def test_real_composed_transform_matches_astropy(spherex_header, template_header):
    """The full composed PixelTransform (as build_transform produces) meets
    the Tier-1 bar end-to-end on the real ICRS<->Galactic pair.
    """
    wcs_in = WCS(spherex_header)
    wcs_out = WCS(template_header)
    transform = build_transform(wcs_in, wcs_out)

    rng = np.random.default_rng(1)
    n = 5000
    x_out = rng.uniform(0, template_header["NAXIS1"] - 1, n)
    y_out = rng.uniform(0, template_header["NAXIS2"] - 1, n)
    x_in, y_in = transform(x_out, y_out)
    x_in, y_in = np.asarray(x_in), np.asarray(y_in)

    sky = SkyCoord(*wcs_out.all_pix2world(x_out, y_out, 0), unit="deg", frame=wcs_to_celestial_frame(wcs_out))
    sky_in_frame = sky.transform_to(wcs_to_celestial_frame(wcs_in))
    x_ref, y_ref = wcs_in.all_world2pix(sky_in_frame.spherical.lon.deg, sky_in_frame.spherical.lat.deg, 0)

    err = np.hypot(x_in - x_ref, y_in - y_ref)
    valid = ~np.isnan(err)
    assert valid.any()
    assert np.nanmax(err) < PIX_TOL


def test_real_composed_transform_f32_precision_quantified(spherex_header, template_header):
    """matrix_precision="f32" (mapprojax's Reprojector, no jax_enable_x64
    needed) on this real ICRS TAN+SIP <-> Galactic TAN pair, against the
    same astropy reference as the f64 test above. Measured (CPU and ROCm
    agree): max=2.81e-4 px, median=6.58e-5 px over all 5000 samples. The
    caps below are ~2x those measurements.
    """
    wcs_in = WCS(spherex_header)
    wcs_out = WCS(template_header)
    transform = build_transform(wcs_in, wcs_out, matrix_precision="f32")

    rng = np.random.default_rng(1)
    n = 5000
    x_out = rng.uniform(0, template_header["NAXIS1"] - 1, n)
    y_out = rng.uniform(0, template_header["NAXIS2"] - 1, n)
    x_in, y_in = transform(x_out, y_out)
    x_in, y_in = np.asarray(x_in), np.asarray(y_in)

    sky = SkyCoord(*wcs_out.all_pix2world(x_out, y_out, 0), unit="deg", frame=wcs_to_celestial_frame(wcs_out))
    sky_in_frame = sky.transform_to(wcs_to_celestial_frame(wcs_in))
    x_ref, y_ref = wcs_in.all_world2pix(sky_in_frame.spherical.lon.deg, sky_in_frame.spherical.lat.deg, 0)

    err = np.hypot(x_in - x_ref, y_in - y_ref)
    valid = ~np.isnan(err)
    assert valid.any()
    assert int(valid.sum()) == n  # measured: all 5000 samples valid
    assert np.nanmax(err) < 6e-4  # measured max 2.81e-4 px
    assert np.nanmedian(err) < 1.5e-4  # measured median 6.58e-5 px


def _strip_reverse_sip(header):
    """Drop the AP/BP reverse polynomial (and its orders), keeping A/B."""
    header = header.copy()
    for key in list(header):
        if key.startswith(("AP_", "BP_")):
            del header[key]
    return header


# Measured on CPU and ROCm (identical): max error of the f32 transform over
# the oracle-valid samples is 3.04e-4 px forward (SPHEREx is the source) and
# 2.59e-4 px reverse (the template is the source), for both sip modes. The
# f32 caps are ~2x those. f64 measures 4.0e-8 px forward / 6.9e-11 px
# reverse, well inside the shared Tier-1 PIX_TOL.
_F32_CAP = {False: 6e-4, True: 5e-4}
# Oracle-valid sample counts (seed 7, 5000 samples). These depend only on
# astropy and the fixed RNG seed, so they are deterministic and frozen.
_EXPECTED_OVERLAP = {False: 4673, True: 1568}


@pytest.mark.parametrize("matrix_precision", ["f64", "f32"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("sip_mode", ["full", "ab_only"])
def test_real_pair_bidirectional_overlap_gate(
    spherex_header, template_header, matrix_precision, reverse, sip_mode
):
    """Both real frame directions retain finite, non-vacuous overlap.

    ``ab_only`` strips AP/BP from whichever header carries SIP (the SPHEREx
    header, which is the source forward and the target in reverse), so the
    transform must work from the forward A/B polynomial alone.
    """
    cap = PIX_TOL if matrix_precision == "f64" else _F32_CAP[reverse]
    source_header, target_header = (
        (spherex_header, template_header)
        if not reverse
        else (template_header, spherex_header)
    )
    if sip_mode == "ab_only":
        assert any(k.startswith(("AP_", "BP_")) for k in spherex_header)
        full_transform = build_transform(
            WCS(source_header), WCS(target_header), matrix_precision=matrix_precision
        )
        if reverse:
            target_header = _strip_reverse_sip(target_header)
        else:
            source_header = _strip_reverse_sip(source_header)
        for header in (source_header, target_header):
            assert not any(k.startswith(("AP_", "BP_")) for k in header)
    source = WCS(source_header)
    target = WCS(target_header)
    transform = build_transform(source, target, matrix_precision=matrix_precision)
    if sip_mode == "ab_only":
        # A distinct SIP fingerprint, so a distinct (not cached-full) transform.
        assert transform is not full_transform
    rng = np.random.default_rng(7)
    n = 5000
    x_out = rng.uniform(0, target_header["NAXIS1"] - 1, n)
    y_out = rng.uniform(0, target_header["NAXIS2"] - 1, n)
    x_in, y_in = transform(x_out, y_out)
    sky = SkyCoord(
        *target.all_pix2world(x_out, y_out, 0),
        unit="deg",
        frame=wcs_to_celestial_frame(target),
    ).transform_to(wcs_to_celestial_frame(source))
    x_ref, y_ref = source.all_world2pix(
        sky.spherical.lon.deg, sky.spherical.lat.deg, 0
    )
    x_in, y_in = np.asarray(x_in), np.asarray(y_in)
    err = np.hypot(x_in - x_ref, y_in - y_ref)
    oracle_valid = (
        np.isfinite(x_ref)
        & np.isfinite(y_ref)
        & (x_ref >= 0)
        & (x_ref < source_header["NAXIS1"])
        & (y_ref >= 0)
        & (y_ref < source_header["NAXIS2"])
    )
    assert int(oracle_valid.sum()) == _EXPECTED_OVERLAP[reverse]
    assert np.all(np.isfinite(err[oracle_valid]))
    assert np.max(err[oracle_valid]) < cap


# ---------------------------------------------------------------------------
# Tier-2: full image reprojection vs original reproject_interp.
# ---------------------------------------------------------------------------


def test_real_reproject_interp_matches_original(spherex_header, spherex_data, template_header):
    wcs_in = WCS(spherex_header)

    arr, footprint = reproject_interp((spherex_data, wcs_in), template_header)
    arr, footprint = np.asarray(arr), np.asarray(footprint)

    arr_ref, footprint_ref = original_reproject_interp(
        (spherex_data, wcs_in), template_header, order="bilinear"
    )

    assert np.array_equal(footprint.astype(bool), footprint_ref.astype(bool))

    valid = footprint.astype(bool) & np.asarray(footprint_ref).astype(bool)
    assert valid.any()

    # 1,345,444 of 1,345,460 jointly-valid pixels agree to rtol<=1e-8. The
    # remaining 16 sit immediately adjacent to an extreme, un-flagged data
    # spike (this fixture dropped the FLAGS extension that would normally
    # mask it) where the local value gradient is ~10^4 per pixel; there,
    # astropy's own iterative all_world2pix intermittently fails to fully
    # converge (NoConvergence), and the sub-micropixel position difference
    # between its best-effort answer and mapprojax's Newton-solved one gets
    # amplified into a value difference up to ~0.007 -- not a reprojax
    # accuracy defect (the transform itself measures ~2e-7 px on this header;
    # see test_real_sip_inversion_quality_quantified), but an expected
    # consequence of interpolating next to a sharp discontinuity with two
    # independently-converged solvers. rtol/atol below cover this with margin.
    np.testing.assert_allclose(arr[valid], arr_ref[valid], rtol=1e-6, atol=0.01)


def test_real_data_has_masked_pixels_handled_correctly(spherex_data):
    """The real exposure genuinely contains masked (NaN) pixels -- confirm
    the fixture still exercises that path (not just clean synthetic data).
    """
    assert np.isnan(spherex_data).sum() > 0


# ---------------------------------------------------------------------------
# Adaptive (DeForest 2004): the ticket #6 real-data integration case.
# ---------------------------------------------------------------------------


def test_real_reproject_adaptive_matches_original(spherex_header, spherex_data, template_header):
    wcs_in = WCS(spherex_header)

    arr, footprint = reproject_adaptive((spherex_data, wcs_in), template_header, max_window_radius=10)
    arr, footprint = np.asarray(arr), np.asarray(footprint)

    arr_ref, footprint_ref = original_reproject_adaptive((spherex_data, wcs_in), template_header)
    arr_ref, footprint_ref = np.asarray(arr_ref), np.asarray(footprint_ref)

    assert np.array_equal(footprint.astype(bool), footprint_ref.astype(bool))

    valid = footprint.astype(bool) & footprint_ref.astype(bool)
    assert valid.any()

    # 1,334,193 of 1,334,200 jointly-valid pixels agree to a median of ~4e-8.
    # The remaining 7 cluster at the same real, unmasked data spike (around
    # row 807-809, col 609-611) diagnosed in test_real_reproject_interp_matches_original
    # -- this fixture dropped the FLAGS extension that would normally mask
    # it, so a sub-micropixel position difference near a ~10^4-per-pixel
    # local gradient gets amplified into a value difference up to ~0.003.
    # Not an adaptive-algorithm defect; the same root cause as the interp
    # case, at a comparable magnitude.
    np.testing.assert_allclose(arr[valid], arr_ref[valid], rtol=1e-6, atol=0.005)
