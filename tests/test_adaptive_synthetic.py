"""Adaptive (DeForest 2004) reprojection: core sanity checks + Tier-2
golden-reference validation against reproject_adaptive.

See reprojax/adaptive.py's module docstring for the algorithm's scope and
the two things diagnosed while building this harness (exact-integer-tie
window-boundary sensitivity, and a 0*nan window-masking bug this harness
caught before it shipped).

The core-sanity tests below are adapted from mapprojax's original
tests/test_adaptive_reproject.py and tests/test_adaptive_blocking.py — ported
to the (result, footprint) return convention and explicit x_out/y_out grids,
per ticket #6's "adaptive tests moved/adapted from mapprojax pass in
reprojax". test_nan_handling below intentionally asserts different behavior
than mapprojax's original: this port matches reproject_adaptive's actual
default (a NaN in the window poisons the whole output pixel), whereas
mapprojax's original approximated a different, non-default behavior
(exclude NaNs and renormalize) that reproject_adaptive doesn't actually use
by default.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from reproject.adaptive import reproject_adaptive as original_reproject_adaptive

from reprojax.adaptive import reproject_adaptive, reproject_adaptive_core

from _fixtures import make_header, make_sip

RTOL = 1e-6
ATOL = 1e-8


def _grid(ny, nx, dtype=jnp.float64):
    y_out, x_out = jnp.meshgrid(jnp.arange(ny, dtype=dtype), jnp.arange(nx, dtype=dtype), indexing="ij")
    return x_out, y_out


# ---------------------------------------------------------------------------
# Core sanity checks (adapted from mapprojax's original test suite).
# ---------------------------------------------------------------------------


def test_identity_preserves_peak():
    ny, nx = 30, 30
    y, x = jnp.meshgrid(jnp.arange(ny, dtype=jnp.float64), jnp.arange(nx, dtype=jnp.float64), indexing="ij")
    xc, yc = 15.0, 15.0
    image = jnp.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (3.0**2))

    def identity(x, y):
        return x, y

    x_out, y_out = _grid(ny, nx)
    result, footprint = reproject_adaptive_core(image, x_out, y_out, identity, kernel_width=0.5, max_window_radius=10)

    assert abs(float(jnp.nanmax(result)) - float(image.max())) < 0.1
    max_idx = jnp.unravel_index(jnp.nanargmax(result), result.shape)
    assert max_idx == (15, 15)
    # strict boundary_mode legitimately NaNs a border ring whose search
    # window extends past the array edge; check validity away from it.
    assert bool(jnp.all(footprint[5:25, 5:25] == 1.0))


def test_downsample_averages_constant_region():
    image = jnp.ones((40, 40))

    def scale_2x(x, y):
        return 2.0 * x, 2.0 * y

    x_out, y_out = _grid(20, 20)
    result, footprint = reproject_adaptive_core(image, x_out, y_out, scale_2x, max_window_radius=10)

    assert abs(float(result[10, 10]) - 1.0) < 0.01
    assert float(footprint[10, 10]) == 1.0


def test_coordinate_shift_moves_mass():
    ny, nx = 20, 20
    image = jnp.zeros((ny, nx)).at[10, 10].set(1.0)

    def shift(x, y):
        return x - 2.5, y - 2.5

    x_out, y_out = _grid(ny, nx)
    result, _ = reproject_adaptive_core(image, x_out, y_out, shift, max_window_radius=10)

    assert float(result[10, 10]) < 0.1
    assert float(jnp.sum(result[11:15, 11:15])) > 0.5


def test_nan_in_source_poisons_the_window_strict_default():
    """reproject_adaptive's actual default (bad_value_mode='strict') is that
    a NaN/Inf sample anywhere in an output pixel's window invalidates that
    whole pixel -- there is no implicit "ignore NaNs and renormalize"
    unless bad_value_mode='ignore' is requested (not implemented here; see
    the module docstring). This is the behavior to match, confirmed against
    the real reference in test_matches_original_with_nan_in_data below.
    """
    ny, nx = 20, 20
    image = jnp.ones((ny, nx)).at[10, 10].set(jnp.nan)

    def identity(x, y):
        return x, y

    x_out, y_out = _grid(ny, nx)
    result, footprint = reproject_adaptive_core(image, x_out, y_out, identity, kernel_width=1.0, max_window_radius=3)

    assert bool(jnp.isnan(result[10, 10]))
    assert float(footprint[10, 10]) == 0.0
    # far from both the NaN and the array edge (strict boundary_mode
    # legitimately NaNs a border ring too -- not what this test targets).
    assert not bool(jnp.isnan(result[4, 4]))
    assert float(footprint[4, 4]) == 1.0


# ---------------------------------------------------------------------------
# block_size tiling: element-wise identical to unblocked processing.
# ---------------------------------------------------------------------------


def test_block_processing_matches_unblocked():
    ny, nx = 40, 40
    y_grid, x_grid = jnp.meshgrid(jnp.arange(ny, dtype=jnp.float64), jnp.arange(nx, dtype=jnp.float64), indexing="ij")
    image = (x_grid + y_grid).astype(float)

    def identity(x, y):
        return x, y

    x_out, y_out = _grid(ny, nx)
    result_full, footprint_full = reproject_adaptive_core(image, x_out, y_out, identity, block_size=None)
    result_blocked, footprint_blocked = reproject_adaptive_core(image, x_out, y_out, identity, block_size=100)

    np.testing.assert_allclose(np.asarray(result_full), np.asarray(result_blocked), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(footprint_full), np.asarray(footprint_blocked))


def test_block_processing_pads_correctly_when_not_a_multiple():
    ny, nx = 21, 21  # 441 pixels, not a multiple of block_size=100
    image = jnp.ones((ny, nx))

    def identity(x, y):
        return x, y

    x_out, y_out = _grid(ny, nx)
    result, footprint = reproject_adaptive_core(image, x_out, y_out, identity, block_size=100, max_window_radius=5)

    assert result.shape == (ny, nx)
    # strict boundary_mode legitimately NaNs a border ring; the constant
    # region is preserved everywhere the window stays in bounds.
    assert bool(jnp.all(footprint[3:-3, 3:-3] == 1.0))
    assert bool(jnp.allclose(result[3:-3, 3:-3], 1.0))


# ---------------------------------------------------------------------------
# Tier-2 golden-reference validation against reproject_adaptive.
#
# Non-round CRPIX/CRVAL/rotation throughout (see reprojax/adaptive.py's
# module docstring): artificially "round" synthetic parameters can put a
# search window's true boundary exactly on an integer, which is where the
# tiny (~1e-10) trig round-trip difference between reprojax's composed
# transform and astropy's own coordinate pipeline can flip a ceil/floor
# decision. Realistic (non-round) parameters avoid manufacturing this.
# ---------------------------------------------------------------------------


def assert_matches_original(header_in, header_out, shape_out, data=None, seed=0, max_window_radius=10, **kernel_kwargs):
    """kernel_kwargs (kernel, kernel_width, sample_region_width) are passed
    to *both* reprojax and the reference call -- max_window_radius has no
    reproject_adaptive equivalent (see the module docstring) and is
    reprojax-only.
    """
    wcs_in, wcs_out = WCS(header_in), WCS(header_out)
    if data is None:
        rng = np.random.default_rng(seed)
        data = rng.uniform(0.0, 1.0, size=(header_in["NAXIS2"], header_in["NAXIS1"]))

    arr, footprint = reproject_adaptive(
        (data, wcs_in), wcs_out, shape_out=shape_out, max_window_radius=max_window_radius, **kernel_kwargs
    )
    arr, footprint = np.asarray(arr), np.asarray(footprint)

    arr_ref, footprint_ref = original_reproject_adaptive(
        (data, wcs_in), wcs_out, shape_out=shape_out, **kernel_kwargs
    )
    footprint_ref = np.asarray(footprint_ref)

    assert np.array_equal(footprint.astype(bool), footprint_ref.astype(bool)), "footprints differ"
    valid = footprint.astype(bool) & footprint_ref.astype(bool)
    assert valid.any()
    np.testing.assert_allclose(arr[valid], np.asarray(arr_ref)[valid], rtol=RTOL, atol=ATOL)
    return arr, footprint


def test_matches_original_rotation_scale_offset():
    h_in = make_header((50.37, 50.21), (180.013, 30.007), 1.03, naxis=(100, 100))
    h_out = make_header((49.62, 50.88), (180.0, 30.0), 0.97, rot_deg=3.7, naxis=(100, 100))
    assert_matches_original(h_in, h_out, (100, 100))


def test_matches_original_4x_downsample():
    h_in = make_header((200.4, 199.6), (180.011, 30.009), 1.01, naxis=(400, 400))
    h_out = make_header((49.7, 50.3), (180.0, 30.0), 4.03, rot_deg=2.1, naxis=(100, 100))
    assert_matches_original(h_in, h_out, (100, 100), max_window_radius=15)


def test_matches_original_with_sip_forward_path():
    a = {(2, 0): 4e-6, (1, 1): -2e-6, (0, 2): 3e-6}
    b = {(2, 0): -3e-6, (1, 1): 2.5e-6, (0, 2): -2e-6}
    sip = make_sip(a, b, (-50, 50), (-50, 50))
    h_out = make_header((50.37, 50.21), (180.013, 30.007), 1.03, sip=sip, naxis=(100, 100))
    h_in = make_header((49.62, 50.88), (180.0, 30.0), 0.97, rot_deg=3.7, naxis=(100, 100))
    assert_matches_original(h_in, h_out, (100, 100))


def test_matches_original_with_sip_reverse_path():
    scale = 1e-6
    a = {(2, 0): scale, (1, 1): -0.5 * scale, (0, 2): 0.75 * scale}
    b = {(2, 0): -0.75 * scale, (1, 1): 0.6 * scale, (0, 2): -0.5 * scale}
    sip = make_sip(a, b, (-50, 50), (-50, 50))
    h_in = make_header((50.37, 50.21), (180.013, 30.007), 1.03, sip=sip, naxis=(100, 100))
    h_out = make_header((49.62, 50.88), (180.0, 30.0), 0.97, rot_deg=3.7, naxis=(100, 100))
    assert_matches_original(h_in, h_out, (100, 100))


def test_matches_original_icrs_galactic_frame_change():
    icrs_center = SkyCoord(180.013, 30.007, unit="deg", frame="icrs")
    gal_center = icrs_center.galactic
    h_icrs = make_header((50.37, 50.21), (icrs_center.ra.deg, icrs_center.dec.deg), 1.03, naxis=(100, 100))
    h_gal = make_header(
        (49.62, 50.88), (gal_center.l.deg, gal_center.b.deg), 0.97, ctype=("GLON-TAN", "GLAT-TAN"), naxis=(100, 100)
    )
    assert_matches_original(h_icrs, h_gal, (100, 100))


def test_matches_original_with_nan_in_data():
    h_in = make_header((50.37, 50.21), (180.013, 30.007), 1.03, naxis=(100, 100))
    h_out = make_header((49.62, 50.88), (180.0, 30.0), 0.97, rot_deg=3.7, naxis=(100, 100))
    data = np.random.default_rng(0).uniform(0.0, 1.0, size=(100, 100))
    data[50, 50] = np.nan
    assert_matches_original(h_in, h_out, (100, 100), data=data)


def test_matches_original_hann_kernel():
    h_in = make_header((50.37, 50.21), (180.013, 30.007), 1.03, naxis=(100, 100))
    h_out = make_header((49.62, 50.88), (180.0, 30.0), 0.97, rot_deg=3.7, naxis=(100, 100))
    assert_matches_original(h_in, h_out, (100, 100), kernel="hann")


# ---------------------------------------------------------------------------
# Error handling.
# ---------------------------------------------------------------------------


def test_unsupported_kernel_raises():
    h_in = make_header((50.0, 50.0), (180.0, 30.0), 1.0, naxis=(100, 100))
    h_out = make_header((50.0, 50.0), (180.0, 30.0), 1.0, naxis=(100, 100))
    data = np.random.default_rng(0).uniform(0.0, 1.0, size=(100, 100))
    with pytest.raises(ValueError, match="kernel"):
        reproject_adaptive((data, WCS(h_in)), WCS(h_out), shape_out=(100, 100), kernel="bogus")
