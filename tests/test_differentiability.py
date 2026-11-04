"""Differentiability guarantee: jax.grad through the reproject_interp core.

Tests attach at the public core / PixelTransform seam
(reprojax.interp.reproject_interp_core + reprojax.transform.compose_transform)
rather than the astropy-facing wrapper (reprojax.reproject_interp /
build_transform): astropy.wcs.WCS objects are not JAX pytrees, so
differentiating with respect to WCS parameters (CRVAL, CD, SIP coefficients)
requires constructing mapprojax WCS objects directly and composing them with
compose_transform, which *is* pure JAX all the way through.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mapprojax.jax_projections import TanJax
from mapprojax.sip import Sip

from reprojax.interp import reproject_interp_core
from reprojax.transform import compose_transform

# A deliberately non-symmetric scene: offset (non-integer) CRPIX on both
# sides, SIP coefficients with genuine cross-terms (both u and v contribute
# to every coefficient position that's populated), and a small CRVAL offset
# between input and output within the field of view -- i.e. ordinary,
# unremarkable WCS parameters, not the exact-symmetry configuration
# documented in test_newton_refinement_nan_at_exact_symmetry_point_is_a_known_limitation
# below. No frame rotation here (identity): an arbitrary rotation not
# constructed to align the two WCSes' tangent points can shift the effective
# offset between them by degrees rather than arcseconds, which was an early,
# corrected mistake in building this scene (it silently made every pixel
# invalid) -- frame rotation itself is exercised by tests/test_transform_synthetic.py
# and tests/test_real_spherex.py, not needed again here.

CRPIX_IN = [10.3, 9.7]
CRPIX_OUT = [7.2, 6.8]
CRVAL_OUT = jnp.array([180.0, 30.0])
_ARCSEC = 1.0 / 3600.0
CD_OUT = jnp.array([[-1.1 * _ARCSEC, 0.01 * _ARCSEC], [0.008 * _ARCSEC, 0.97 * _ARCSEC]])
ROTATION = jnp.eye(3)

NY, NX = 15, 15
_Y_OUT, _X_OUT = jnp.meshgrid(
    jnp.arange(NY, dtype=jnp.float64) + 0.31, jnp.arange(NX, dtype=jnp.float64) + 0.17, indexing="ij"
)

DATA = jnp.asarray(np.random.default_rng(0).uniform(0.0, 1.0, size=(20, 20)))

SIP_SCALE = 1e-6


def _sip_coeffs(scale=SIP_SCALE, sign=1.0):
    a = jnp.zeros((3, 3)).at[2, 0].set(sign * scale).at[1, 1].set(-0.6 * sign * scale).at[0, 2].set(0.4 * sign * scale)
    b = jnp.zeros((3, 3)).at[2, 0].set(-0.7 * sign * scale).at[1, 1].set(0.5 * sign * scale).at[0, 2].set(-0.3 * sign * scale)
    return a, b


def _loss(crval_in, cd_in, sip_a, sip_b, sip_ap, sip_bp, data, order=1, refine_steps=0, x_out=_X_OUT, y_out=_Y_OUT):
    sip = Sip(a=sip_a, b=sip_b, ap=sip_ap, bp=sip_bp)
    mwcs_in = TanJax(CRPIX_IN, cd_in, crval_in, sip=sip)
    mwcs_out = TanJax(CRPIX_OUT, CD_OUT, CRVAL_OUT, sip=None)
    transform = compose_transform(mwcs_in, mwcs_out, ROTATION, sip_inverse_refine_steps=refine_steps)
    result, footprint = reproject_interp_core(data, x_out, y_out, transform, order=order)
    safe = jnp.where(footprint > 0, result, 0.0)
    return jnp.sum(safe**2)


def _default_params():
    sip_a, sip_b = _sip_coeffs()
    sip_ap, sip_bp = _sip_coeffs(sign=-1.0)
    # ~3 arcsec / ~2 arcsec offset from CRVAL_OUT -- well within the 20px
    # field at ~1"/px, so the two WCSes' fields genuinely overlap.
    crval_in = jnp.array([180.0 + 3.0 * _ARCSEC, 30.0 + 2.0 * _ARCSEC])
    cd_in = jnp.array([[-_ARCSEC, 0.02 * _ARCSEC], [0.015 * _ARCSEC, 1.03 * _ARCSEC]])
    return crval_in, cd_in, sip_a, sip_b, sip_ap, sip_bp


def _fd_grad(f, x, eps):
    """Central-difference gradient of a scalar function f(x) w.r.t. array x."""
    x_np = np.asarray(x, dtype=float)
    grad = np.zeros_like(x_np)
    it = np.nditer(x_np, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        x_plus = x_np.copy()
        x_plus[idx] += eps
        x_minus = x_np.copy()
        x_minus[idx] -= eps
        grad[idx] = (float(f(jnp.asarray(x_plus))) - float(f(jnp.asarray(x_minus)))) / (2 * eps)
    return grad


def _assert_matches_fd(analytic, fd, rtol=1e-4, atol=1e-6):
    """rtol is looser than a single-point gradient check would need: the
    loss sums over ~210 output pixels, and bilinear interpolation has a
    genuine gradient kink at each pixel crossing (see
    test_bilinear_gradient_has_kinks_at_pixel_crossings) -- a central
    difference at a fixed eps has *some* chance of straddling one of those
    crossings for one of the many summed pixels, which central-differences
    away from any crossing would not need to tolerate.
    """
    analytic, fd = np.asarray(analytic), np.asarray(fd)
    assert np.all(np.isfinite(analytic)), "analytic gradient is not finite"
    np.testing.assert_allclose(analytic, fd, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Gradients w.r.t. WCS parameters (CRVAL, CD) match finite differences.
# ---------------------------------------------------------------------------


def test_grad_crval_matches_finite_differences():
    crval_in, cd_in, sip_a, sip_b, sip_ap, sip_bp = _default_params()

    def f(crval):
        return _loss(crval, cd_in, sip_a, sip_b, sip_ap, sip_bp, DATA)

    g = jax.grad(f)(crval_in)
    fd = _fd_grad(f, crval_in, eps=1e-7)
    _assert_matches_fd(g, fd)


def test_grad_cd_matches_finite_differences():
    crval_in, cd_in, sip_a, sip_b, sip_ap, sip_bp = _default_params()

    def f(cd):
        return _loss(crval_in, cd, sip_a, sip_b, sip_ap, sip_bp, DATA)

    g = jax.grad(f)(cd_in)
    fd = _fd_grad(f, cd_in, eps=1e-9)
    _assert_matches_fd(g, fd)


# ---------------------------------------------------------------------------
# Gradients w.r.t. SIP coefficients match finite differences.
#
# Which coefficients carry gradient signal depends on whether Newton
# refinement is active (the default whenever SIP is present, see
# reprojax._sip_refine): refinement uses AP/BP purely as an initial guess and
# then converges against the *forward* A/B polynomial, so at convergence the
# result is (to the refinement's own precision) insensitive to the initial
# guess -- gradient w.r.t. AP/BP is near machine-zero, and A/B carries the
# real signal. Disable refinement to instead see AP/BP carry the signal
# (the plain evaluation path). Both are tested; both match finite differences
# where the signal is non-negligible.
# ---------------------------------------------------------------------------


# Central-difference step for the A/B check. Step sweep on this code (CPU,
# worst relative discrepancy over components with |fd| > 1e-6; A / B):
#   eps=1e-6: 8.2e-7 / 3.9e-5    eps=3e-7: 3.0e-7 / 3.5e-6
#   eps=1e-7: 2.1e-7 / 3.9e-7    eps=3e-8: 4.9e-6 / 1.2e-7
#   eps=1e-8: 1.9e-5 / 1.2e-6    eps=1e-9: 2.8e-4 / 2.0e-5
# The ~3e-4 mismatch the roadmap recorded is the old eps=1e-9 step: error
# grows at large steps (truncation) and at small steps (roundoff), the usual
# finite-difference trade-off, with no pixel-boundary kink involved (that
# would not shrink with the step). eps=1e-7 sits in the minimum, so
# rtol=1e-5 has >20x margin.
_SIP_FD_EPS = 1e-7
_SIP_FD_RTOL = 1e-5


def test_grad_sip_forward_coeffs_matches_finite_differences():
    crval_in, cd_in, sip_a, sip_b, sip_ap, sip_bp = _default_params()

    def f_a(a):
        return _loss(crval_in, cd_in, a, sip_b, sip_ap, sip_bp, DATA)

    def f_b(b):
        return _loss(crval_in, cd_in, sip_a, b, sip_ap, sip_bp, DATA)

    g_a = jax.grad(f_a)(sip_a)
    g_b = jax.grad(f_b)(sip_b)
    _assert_matches_fd(g_a, _fd_grad(f_a, sip_a, eps=_SIP_FD_EPS), rtol=_SIP_FD_RTOL)
    _assert_matches_fd(g_b, _fd_grad(f_b, sip_b, eps=_SIP_FD_EPS), rtol=_SIP_FD_RTOL)


@pytest.mark.parametrize("refine_steps", [0, 2])
def test_grad_sip_reverse_coeffs_vanishes_under_newton_inverse(refine_steps):
    """Not a bug: mapprojax's SIP inverse Newton-solves against A/B and uses
    AP/BP only as its initial guess, so the AP/BP gradient is near zero with
    or without reprojax's own (now redundant) refinement.
    """
    crval_in, cd_in, sip_a, sip_b, sip_ap, sip_bp = _default_params()

    def f(ap):
        return _loss(crval_in, cd_in, sip_a, sip_b, ap, sip_bp, DATA, refine_steps=refine_steps)

    g = np.asarray(jax.grad(f)(sip_ap))
    assert np.all(np.isfinite(g))
    assert np.max(np.abs(g)) < 1e-8


# ---------------------------------------------------------------------------
# Gradient w.r.t. input image data matches finite differences.
# ---------------------------------------------------------------------------


def test_grad_data_matches_finite_differences():
    crval_in, cd_in, sip_a, sip_b, sip_ap, sip_bp = _default_params()

    def f(data):
        return _loss(crval_in, cd_in, sip_a, sip_b, sip_ap, sip_bp, data)

    g = jax.grad(f)(DATA)
    assert np.all(np.isfinite(np.asarray(g)))

    # Spot-check a random subset of pixels rather than all 400 -- finite
    # differencing every element is unnecessary to establish correctness.
    rng = np.random.default_rng(1)
    idx = rng.choice(DATA.size, size=25, replace=False)
    data_np = np.asarray(DATA)
    for flat_i in idx:
        i, j = np.unravel_index(flat_i, data_np.shape)
        dp = data_np.copy()
        dp[i, j] += 1e-6
        dm = data_np.copy()
        dm[i, j] -= 1e-6
        fd = (float(f(jnp.asarray(dp))) - float(f(jnp.asarray(dm)))) / 2e-6
        assert np.asarray(g)[i, j] == pytest.approx(fd, rel=1e-4, abs=1e-6)


# ---------------------------------------------------------------------------
# Gradients flow through footprint masking without NaN contamination.
# ---------------------------------------------------------------------------


def test_grad_clean_through_footprint_masking_with_genuinely_invalid_pixels():
    """Output grid deliberately extends beyond the input array, so some
    pixels land outside its bounds (footprint 0 there, from
    reproject_interp_core's own coverage check) with SIP + Newton refinement
    active on the input WCS (the default whenever SIP is present). This
    should not poison the gradient w.r.t. WCS parameters or data, which are
    shared across every output pixel (a single NaN contribution would
    otherwise NaN the whole reduced loss).
    """
    crval_in, cd_in, sip_a, sip_b, sip_ap, sip_bp = _default_params()

    y_out, x_out = jnp.meshgrid(
        jnp.arange(NY, dtype=jnp.float64) * 3.0 + 0.31, jnp.arange(NX, dtype=jnp.float64) * 3.0 + 0.17, indexing="ij"
    )

    def f(crval_in, data):
        return _loss(crval_in, cd_in, sip_a, sip_b, sip_ap, sip_bp, data, x_out=x_out, y_out=y_out)

    # sanity: this scenario actually exercises invalid pixels
    sip = Sip(a=sip_a, b=sip_b, ap=sip_ap, bp=sip_bp)
    mwcs_in = TanJax(CRPIX_IN, cd_in, crval_in, sip=sip)
    mwcs_out = TanJax(CRPIX_OUT, CD_OUT, CRVAL_OUT, sip=None)
    transform = compose_transform(mwcs_in, mwcs_out, ROTATION)
    _, footprint = reproject_interp_core(DATA, x_out, y_out, transform, order=1)
    assert int(jnp.sum(footprint == 0)) > 0

    g_crval, g_data = jax.grad(f, argnums=(0, 1))(crval_in, DATA)
    assert np.all(np.isfinite(np.asarray(g_crval)))
    assert np.all(np.isfinite(np.asarray(g_data)))


def test_grad_clean_through_off_hemisphere_masking_without_sip():
    """The other, narrower source of invalid pixels: a point genuinely
    outside the input WCS's visible hemisphere (compose_transform's own
    xn > 0 domain guard, independent of reproject_interp_core's array-bounds
    coverage check and of any SIP/refinement). No SIP here, isolating this
    specific masking path.
    """
    crpix_in = [10.3, 9.7]
    crpix_out = [7.2, 6.8]
    crval_out = jnp.array([0.0, -30.0])
    cd_out = jnp.array([[-2.0, 0.0], [0.0, 2.0]])  # huge FOV: guaranteed to include far-hemisphere points
    rotation = jnp.eye(3)

    y_out, x_out = jnp.meshgrid(
        jnp.arange(NY, dtype=jnp.float64) + 0.31, jnp.arange(NX, dtype=jnp.float64) + 0.17, indexing="ij"
    )

    def f(crval_in, cd_in, data):
        mwcs_in = TanJax(crpix_in, cd_in, crval_in, sip=None)
        mwcs_out = TanJax(crpix_out, cd_out, crval_out, sip=None)
        transform = compose_transform(mwcs_in, mwcs_out, rotation)
        result, footprint = reproject_interp_core(data, x_out, y_out, transform, order=1)
        return jnp.sum(jnp.where(footprint > 0, result, 0.0) ** 2)

    crval_in = jnp.array([180.0, 30.0])
    cd_in = jnp.array([[-1.0 / 3600, 0.0], [0.0, 1.0 / 3600]])

    mwcs_in = TanJax(crpix_in, cd_in, crval_in, sip=None)
    mwcs_out = TanJax(crpix_out, cd_out, crval_out, sip=None)
    transform = compose_transform(mwcs_in, mwcs_out, rotation)
    x_in, _ = transform(x_out, y_out)
    assert bool(jnp.any(jnp.isnan(x_in))), "scenario should include off-hemisphere points"

    g_crval, g_cd, g_data = jax.grad(f, argnums=(0, 1, 2))(crval_in, cd_in, DATA)
    assert np.all(np.isfinite(np.asarray(g_crval)))
    assert np.all(np.isfinite(np.asarray(g_cd)))
    assert np.all(np.isfinite(np.asarray(g_data)))


# ---------------------------------------------------------------------------
# Bilinear's coordinate-gradient kinks at pixel crossings: documented as
# inherent to the algorithm, not something to be worked around.
# ---------------------------------------------------------------------------


def test_bilinear_gradient_has_kinks_at_pixel_crossings():
    """Bilinear interpolation is piecewise-linear in the sampled coordinate:
    its gradient w.r.t. anything that shifts that coordinate is
    discontinuous exactly where the sample crosses an input pixel boundary
    (a different pair of neighbors engages, with a different local slope),
    and smooth in between. This is inherent to order=1 interpolation, not a
    defect to work around -- demonstrated directly through
    reproject_interp_core (the public core seam) with a trivial shift
    transform, using data whose consecutive differences (1, 3, 5) guarantee
    a different local slope in each pixel interval.
    """
    data = jnp.array([[0.0, 1.0, 4.0, 9.0]])  # squares: slopes 1, 3, 5 between consecutive pixels
    y_out = jnp.array([0.0])
    x_out = jnp.array([1.0])

    def result_at(t):
        def shift_transform(xo, yo):
            return xo + t, yo

        result, _ = reproject_interp_core(data, x_out, y_out, shift_transform, order=1)
        return result[0]

    grad_fn = jax.grad(result_at)
    # x_in = x_out + t = 1 + t. Just below t=1, x_in is in [1,2) (slope 3);
    # just above, x_in is in [2,3) (slope 5, since x_out=1 stays fixed and
    # only t crosses the boundary at x_in=2).
    g_below = float(grad_fn(0.999999))
    g_above = float(grad_fn(1.000001))

    assert g_below == pytest.approx(3.0, abs=1e-4)
    assert g_above == pytest.approx(5.0, abs=1e-4)
    assert g_below != pytest.approx(g_above)  # the inherent kink


# ---------------------------------------------------------------------------
# An exact geometric symmetry point (output pixel on both CRPIXes, axis-aligned
# CD, identity frame rotation, SIP depending on u only) used to give a NaN
# gradient through reprojax's Newton refinement. Kept as a regression check
# now that the default path (mapprojax's own masked inverse) is finite there.
# ---------------------------------------------------------------------------


def test_sip_inverse_grad_is_finite_at_exact_symmetry_point():
    crpix_in = [10.0, 10.0]
    crpix_out = [7.0, 7.0]
    crval_out = jnp.array([180.0, 30.0])
    crval_in = jnp.array([180.0, 30.0])
    cd_in = jnp.array([[-1.0 / 3600, 0.0], [0.0, 1.0 / 3600]])
    cd_out = jnp.array([[-2.0 / 3600, 0.0], [0.0, 2.0 / 3600]])
    rotation = jnp.eye(3)
    # SIP depending on u only (no cross-term, no v-only term): an
    # unrealistic simplification real optical distortion never has.
    sip_a = jnp.zeros((3, 3)).at[2, 0].set(1e-6)
    sip_b = jnp.zeros((3, 3)).at[2, 0].set(-1e-6)
    sip_ap, sip_bp = -sip_a, -sip_b

    y_out, x_out = jnp.meshgrid(
        jnp.arange(NY, dtype=jnp.float64), jnp.arange(NX, dtype=jnp.float64), indexing="ij"
    )

    def f(crval):
        sip = Sip(a=sip_a, b=sip_b, ap=sip_ap, bp=sip_bp)
        mwcs_in = TanJax(crpix_in, cd_in, crval, sip=sip)
        mwcs_out = TanJax(crpix_out, cd_out, crval_out, sip=None)
        transform = compose_transform(mwcs_in, mwcs_out, rotation)
        result, footprint = reproject_interp_core(DATA, x_out, y_out, transform, order=1)
        safe = jnp.where(footprint > 0, result, 0.0)
        return jnp.sum(safe**2)

    # Formerly a documented NaN from reprojax's refinement. With refinement
    # off and mapprojax's masked Newton inverse, the gradient is finite.
    g = jax.grad(f)(crval_in)
    assert np.all(np.isfinite(np.asarray(g)))
