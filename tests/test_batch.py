"""Batched reprojection: jax.vmap over the core (ticket #5 / issue #6).

Batching stacks exposures and their per-exposure WCS parameters and vmaps
the whole per-exposure pipeline (build transform, then reproject) onto one
shared output grid — the throughput story the whole package is built for.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.wcs import WCS
from mapprojax.wcs import Tan
from mapprojax.sip import Sip
from reproject import reproject_interp as original_reproject_interp

from reprojax import reproject_interp_batched
from reprojax._astropy_wcs import frame_rotation_matrix
from reprojax.interp import reproject_interp_core
from reprojax.transform import compose_transform, wcs_params_from_astropy

from _fixtures import make_header, make_sip


def _make_batch(n, ny_in=30, nx_in=30, seed=0):
    """n exposures of the same (synthetic) instrument: shared CRPIX/CD/SIP,
    each pointed at a slightly different sky position, in ICRS.
    """
    crpix = [15.0, 15.0]
    cd = [[-1.0 / 3600, 0.0], [0.0, 1.0 / 3600]]
    a = {(2, 0): 1e-6, (1, 1): -0.6e-6, (0, 2): 0.4e-6}
    b = {(2, 0): -0.7e-6, (1, 1): 0.5e-6, (0, 2): -0.3e-6}
    sip = make_sip(a, b, (-15, 15), (-15, 15))

    rng = np.random.default_rng(seed)
    ra_centers = 180.0 + rng.uniform(-5.0 / 3600, 5.0 / 3600, size=n)
    dec_centers = 30.0 + rng.uniform(-5.0 / 3600, 5.0 / 3600, size=n)

    headers_in = [
        make_header(crpix, (ra_centers[i], dec_centers[i]), 1.0, sip=sip, naxis=(nx_in, ny_in)) for i in range(n)
    ]
    data_batch = jnp.asarray(rng.uniform(0.0, 1.0, size=(n, ny_in, nx_in)))
    return headers_in, data_batch


def _stack_mwcs_in(headers_in, wcs_out):
    """Build a batched mapprojax ``Tan`` + rotation stack from a list of astropy
    headers, using the same host-side parameter extraction reprojax's
    transform module uses for a single exposure.
    """
    crpixs, cds, crvals, sips, rotations = [], [], [], [], []
    for h in headers_in:
        wcs_in = WCS(h)
        crpix, cd, crval, sip = wcs_params_from_astropy(wcs_in)
        crpixs.append(crpix)
        cds.append(cd)
        crvals.append(crval)
        sips.append(sip)
        rotations.append(frame_rotation_matrix(wcs_out, wcs_in))

    sip_batch = Sip(
        a=jnp.stack([jnp.asarray(s.a) for s in sips]),
        b=jnp.stack([jnp.asarray(s.b) for s in sips]),
        ap=jnp.stack([jnp.asarray(s.ap) for s in sips]),
        bp=jnp.stack([jnp.asarray(s.bp) for s in sips]),
    )
    mwcs_in_batch = Tan(
        jnp.asarray(crpixs), jnp.asarray(cds), jnp.asarray(crvals), sip=sip_batch, precision="f64"
    )
    rotation_batch = jnp.stack([jnp.asarray(r) for r in rotations])
    return mwcs_in_batch, rotation_batch


# ---------------------------------------------------------------------------
# Batched results are element-wise identical to the per-exposure loop.
# ---------------------------------------------------------------------------


def test_batched_matches_looped_single_exposures():
    n = 5
    headers_in, data_batch = _make_batch(n)

    h_out = make_header((15.0, 15.0), (180.0, 30.0), 1.0, naxis=(30, 30))
    wcs_out = WCS(h_out)
    mwcs_out = Tan(*wcs_params_from_astropy(wcs_out)[:3], sip=None, precision="f64")

    mwcs_in_batch, rotation_batch = _stack_mwcs_in(headers_in, wcs_out)

    ny_out, nx_out = 30, 30
    y_out, x_out = jnp.meshgrid(
        jnp.arange(ny_out, dtype=jnp.float64), jnp.arange(nx_out, dtype=jnp.float64), indexing="ij"
    )

    results, footprints = reproject_interp_batched(data_batch, mwcs_in_batch, mwcs_out, rotation_batch, x_out, y_out)
    assert results.shape == (n, ny_out, nx_out)
    assert footprints.shape == (n, ny_out, nx_out)

    for i in range(n):
        wcs_in_i = WCS(headers_in[i])
        crpix_i, cd_i, crval_i, sip_i = wcs_params_from_astropy(wcs_in_i)
        mwcs_in_i = Tan(crpix_i, cd_i, crval_i, sip=sip_i, precision="f64")
        transform_i = compose_transform(mwcs_in_i, mwcs_out, rotation_batch[i])
        result_i, footprint_i = reproject_interp_core(data_batch[i], x_out, y_out, transform_i, order=1)

        np.testing.assert_array_equal(np.asarray(footprints[i]), np.asarray(footprint_i))
        np.testing.assert_array_equal(np.asarray(results[i]), np.asarray(result_i))


# ---------------------------------------------------------------------------
# Works under jit, on GPU, at SPHEREx-scale (2k x 2k) exposures.
# ---------------------------------------------------------------------------


def test_batched_jit_on_gpu_at_spherex_scale():
    if jax.default_backend() != "gpu":
        pytest.skip("no GPU backend available in this environment")

    n = 4
    ny_in = nx_in = 2040  # SPHEREx L2b detector size (see tests/test_real_spherex.py)
    headers_in, data_batch = _make_batch(n, ny_in=ny_in, nx_in=nx_in)

    ny_out = nx_out = 1201  # matches the real Galactic template scale
    h_out = make_header((600.0, 600.0), (180.0, 30.0), 1.0, naxis=(nx_out, ny_out))
    wcs_out = WCS(h_out)
    mwcs_out = Tan(*wcs_params_from_astropy(wcs_out)[:3], sip=None, precision="f64")
    mwcs_in_batch, rotation_batch = _stack_mwcs_in(headers_in, wcs_out)

    y_out, x_out = jnp.meshgrid(
        jnp.arange(ny_out, dtype=jnp.float64), jnp.arange(nx_out, dtype=jnp.float64), indexing="ij"
    )

    assert data_batch.device.platform == "gpu"

    jitted = jax.jit(reproject_interp_batched, static_argnames=("order",))
    results, footprints = jitted(data_batch, mwcs_in_batch, mwcs_out, rotation_batch, x_out, y_out)
    results.block_until_ready()

    assert results.shape == (n, ny_out, nx_out)
    assert results.device.platform == "gpu"
    assert bool(jnp.any(footprints > 0))  # some real overlap, not a vacuous all-invalid scenario


# ---------------------------------------------------------------------------
# Benchmark: wall-clock vs original reproject_interp looped over the batch.
# Numbers are printed and documented, not gated on a specific speedup.
# ---------------------------------------------------------------------------


def test_benchmark_batched_vs_looped_original(capsys):
    if jax.default_backend() != "gpu":
        pytest.skip("no GPU backend available in this environment")

    n = 8
    ny_in = nx_in = 2040
    headers_in, data_batch = _make_batch(n, ny_in=ny_in, nx_in=nx_in)
    data_np = np.asarray(data_batch)

    ny_out = nx_out = 1201
    h_out = make_header((600.0, 600.0), (180.0, 30.0), 1.0, naxis=(nx_out, ny_out))
    wcs_out = WCS(h_out)
    mwcs_out = Tan(*wcs_params_from_astropy(wcs_out)[:3], sip=None, precision="f64")
    mwcs_in_batch, rotation_batch = _stack_mwcs_in(headers_in, wcs_out)

    y_out, x_out = jnp.meshgrid(
        jnp.arange(ny_out, dtype=jnp.float64), jnp.arange(nx_out, dtype=jnp.float64), indexing="ij"
    )

    jitted = jax.jit(reproject_interp_batched, static_argnames=("order",))

    # Warm-up: exclude one-time JIT compilation from the timed call.
    jitted(data_batch, mwcs_in_batch, mwcs_out, rotation_batch, x_out, y_out)[0].block_until_ready()

    t0 = time.perf_counter()
    results, _ = jitted(data_batch, mwcs_in_batch, mwcs_out, rotation_batch, x_out, y_out)
    results.block_until_ready()
    reprojax_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(n):
        original_reproject_interp((data_np[i], WCS(headers_in[i])), h_out, shape_out=(ny_out, nx_out), order="bilinear")
    original_time = time.perf_counter() - t0

    with capsys.disabled():
        print(
            f"\n[benchmark] batch of {n} exposures, {ny_in}x{nx_in} -> {ny_out}x{nx_out}:\n"
            f"  reprojax (batched, jitted, GPU): {reprojax_time:.4f}s\n"
            f"  reproject_interp (looped, CPU):  {original_time:.4f}s\n"
            f"  speedup: {original_time / reprojax_time:.1f}x"
        )

    assert reprojax_time > 0 and original_time > 0
