"""Compare wall-clock performance: original reproject (single-core AND
genuinely multi-core) vs reprojax, across JAX backends (CPU / CUDA), for
both reproject_interp and reproject_adaptive, single-image and batched.

JAX picks its backend at process start (from JAX_PLATFORMS or whatever
accelerator is visible) and doesn't support switching mid-process, so this
script is meant to be run twice as separate processes:

    python benchmarks/benchmark_backends.py                  # default backend (GPU if present)
    JAX_PLATFORMS=cpu python benchmarks/benchmark_backends.py # forced CPU

Each run reports reprojax's numbers for whichever backend it's running
under, plus the original `reproject` package's numbers (which don't depend
on JAX at all, so they're recomputed in both runs for a self-contained
report, but aren't specific to either).

Multi-core reproject: `reproject_interp`/`reproject_adaptive`'s own
`parallel=` (dask-threaded, splits one image into blocks) turns out not to
help for a single image at this resolution -- overhead dominates unless the
block size is tuned (a quick sweep found block_size=(400, 400) to be a good
size at this resolution; see BEST_BLOCK_SIZE). For the batch, the fairer and
far more effective comparison is a process pool across the (embarrassingly
parallel) independent exposures, sidestepping the GIL entirely -- see
_run_interp_one/_run_adaptive_one and BATCH_WORKERS.

Workload: the real SPHEREx L2b fixture (2040x2040, TAN+SIP, ICRS) onto its
real Galactic TAN template (1201x1201) -- see tests/data/ and
tests/test_real_spherex.py. The batch is `BATCH_SIZE` synthetic exposures
at the same resolution (real data, repeated with slightly jittered
pointing, since a batch of independent real SPHEREx exposures with a
shared template isn't available here), reprojected onto the same template
in one call.
"""

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from mapprojax import Tan
from mapprojax.sip import Sip
from reproject import reproject_interp as original_reproject_interp
from reproject.adaptive import reproject_adaptive as original_reproject_adaptive

import reprojax
from reprojax._astropy_wcs import frame_rotation_matrix, wcs_params_from_astropy
from reprojax.adaptive import reproject_adaptive_core
from reprojax.interp import reproject_interp_core
from reprojax.transform import build_transform, compose_transform

DATA_DIR = Path(__file__).parent.parent / "tests" / "data"
BATCH_SIZE = 8
BATCH_WORKERS = 8  # one process per exposure; machine has far more cores available
BEST_BLOCK_SIZE = (400, 400)  # found by a quick sweep at this resolution; see module docstring

# Populated once per worker process by _init_worker (see main): forking the
# main process would be unsafe here (JAX is already multithreaded by the
# time the pool is created -- os.fork() after that is a documented deadlock
# risk, confirmed while building this script), so the pool uses the 'spawn'
# context instead, and each fresh worker loads its own copy of the fixture
# data once at startup rather than inheriting it.
_data = None
_header_out = None
_shape_out = None


def _init_worker():
    global _data, _header_out, _shape_out
    data, _header_in_unused, header_out = load_fixture()
    _data = data
    _header_out = header_out
    _shape_out = (header_out["NAXIS2"], header_out["NAXIS1"])


def _run_interp_one(header_in):
    return original_reproject_interp((_data, WCS(header_in)), _header_out, shape_out=_shape_out, order="bilinear")


def _run_adaptive_one(header_in):
    return original_reproject_adaptive((_data, WCS(header_in)), _header_out, shape_out=_shape_out)
# Vmapping reproject_adaptive_core over a batch multiplies its per-pixel
# (2*max_window_radius+1)^2 window by the batch size; at max_window_radius=10
# (used for the single-image case below) an 8-exposure batch at this
# resolution needs ~44GB, more than one GPU here has (32GB). A smaller
# window is a legitimate, real tuning choice (it only needs to cover the
# actual local downsampling factor, which is modest for this real pair) --
# not a benchmark-only shortcut -- but is called out here since it differs
# from the single-image number's window size.
BATCH_ADAPTIVE_MAX_WINDOW_RADIUS = 6


def load_fixture():
    with fits.open(DATA_DIR / "spherex_l2b_v20.fits") as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        header_in = hdul[0].header.copy()
    header_out = fits.Header.fromtextfile(DATA_DIR / "galactic_template_header.fits")
    return data, header_in, header_out


def make_batch(header_in, header_out, data, n=BATCH_SIZE, seed=0):
    """n copies of the real exposure at slightly jittered pointing (real
    independent SPHEREx exposures sharing one output template aren't
    available in this fixture set) -- same resolution and SIP as the real
    data, which is what matters for a throughput comparison.
    """
    rng = np.random.default_rng(seed)
    headers = []
    for _ in range(n):
        h = header_in.copy()
        h["CRVAL1"] = header_in["CRVAL1"] + rng.uniform(-3, 3) / 3600.0
        h["CRVAL2"] = header_in["CRVAL2"] + rng.uniform(-3, 3) / 3600.0
        headers.append(h)
    data_batch = jnp.asarray(np.stack([data] * n))

    wcs_out = WCS(header_out)
    crpixs, cds, crvals, sips, rotations = [], [], [], [], []
    for h in headers:
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
    mwcs_in_batch = Tan(jnp.asarray(crpixs), jnp.asarray(cds), jnp.asarray(crvals), sip=sip_batch,
                        precision="f64")
    rotation_batch = jnp.stack([jnp.asarray(r) for r in rotations])
    return headers, data_batch, mwcs_in_batch, rotation_batch


def reproject_adaptive_batched(data_batch, mwcs_in_batch, mwcs_out, rotation_batch, x_out, y_out, **kwargs):
    """Same jax.vmap recipe as reprojax.interp.reproject_interp_batched,
    for reproject_adaptive_core (not yet a library-level batched entry
    point -- see reprojax/interp.py's reproject_interp_batched docstring
    for the general pattern this follows).
    """
    def _single(data, mwcs_in, rotation):
        transform = compose_transform(mwcs_in, mwcs_out, rotation)
        return reproject_adaptive_core(data, x_out, y_out, transform, **kwargs)

    return jax.vmap(_single, in_axes=(0, 0, 0))(data_batch, mwcs_in_batch, rotation_batch)


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    elif isinstance(result, tuple) and hasattr(result[0], "block_until_ready"):
        result[0].block_until_ready()
    return time.perf_counter() - t0


def main():
    backend = jax.default_backend()
    print(f"=== JAX backend: {backend} ({jax.devices()}) ===\n")

    data, header_in, header_out = load_fixture()
    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)
    shape_out = (header_out["NAXIS2"], header_out["NAXIS1"])
    ny, nx = shape_out
    y_out, x_out = jnp.meshgrid(jnp.arange(ny, dtype=jnp.float64), jnp.arange(nx, dtype=jnp.float64), indexing="ij")

    print(f"--- Single image: {data.shape} -> {shape_out} ---")

    t_orig_interp = timed(original_reproject_interp, (data, wcs_in), header_out, shape_out=shape_out, order="bilinear")
    print(f"  original reproject_interp (1 core):         {t_orig_interp:.3f}s")

    t_orig_interp_mc = timed(
        original_reproject_interp,
        (data, wcs_in),
        header_out,
        shape_out=shape_out,
        order="bilinear",
        parallel=True,
        block_size=BEST_BLOCK_SIZE,
    )
    print(f"  original reproject_interp (parallel, tuned block_size): {t_orig_interp_mc:.3f}s")

    t_interp_compile = timed(reprojax.reproject_interp, (data, wcs_in), header_out)
    t_interp_warm = timed(reprojax.reproject_interp, (data, wcs_in), header_out)
    print(f"  reprojax reproject_interp:                  {t_interp_warm:.3f}s warm ({t_interp_compile:.3f}s incl. compile)")

    t_orig_adaptive = timed(original_reproject_adaptive, (data, wcs_in), header_out, shape_out=shape_out)
    print(f"  original reproject_adaptive (1 core):        {t_orig_adaptive:.3f}s")

    t_orig_adaptive_mc = timed(
        original_reproject_adaptive,
        (data, wcs_in),
        header_out,
        shape_out=shape_out,
        parallel=True,
        block_size=BEST_BLOCK_SIZE,
    )
    print(f"  original reproject_adaptive (parallel, tuned block_size): {t_orig_adaptive_mc:.3f}s")

    # build_transform() caches on the numeric content of (wcs_in, wcs_out),
    # so this repeated call returns the *same*
    # transform object the wrapper built internally last time rather than
    # rebuilding it -- avoiding a full jit retrace on every call. That
    # retrace was cheap for reproject_interp (~0.27s either way) but not for
    # reproject_adaptive, whose core differentiates *through* transform
    # (jax.jacfwd), and transform itself runs mapprojax's Newton-solved SIP
    # inverse whenever SIP is present (this real SPHEREx header has SIP) --
    # retracing that doubly-nested autodiff graph
    # used to cost ~4s per call before the cache existed.
    t_adaptive_compile = timed(reprojax.reproject_adaptive, (data, wcs_in), header_out, max_window_radius=10)
    t_adaptive_warm = timed(reprojax.reproject_adaptive, (data, wcs_in), header_out, max_window_radius=10)
    print(f"  reprojax reproject_adaptive:                 {t_adaptive_warm:.3f}s warm ({t_adaptive_compile:.3f}s incl. compile)")

    print(f"\n--- Batch of {BATCH_SIZE}: {data.shape} -> {shape_out} ---")

    headers, data_batch, mwcs_in_batch, rotation_batch = make_batch(header_in, header_out, data)
    mwcs_out = Tan(*wcs_params_from_astropy(wcs_out)[:3], sip=None, precision="f64")

    t0 = time.perf_counter()
    for h in headers:
        original_reproject_interp((data, WCS(h)), header_out, shape_out=shape_out, order="bilinear")
    t_orig_interp_batch = time.perf_counter() - t0
    print(f"  original reproject_interp (looped, 1 core):        {t_orig_interp_batch:.3f}s")

    # 'spawn', not the platform default 'fork': JAX is already multithreaded
    # by this point in the script, and forking a multithreaded process is a
    # documented deadlock risk (hit while building this: reproject's own
    # dask-threaded parallel= path was fine, but forking here intermittently
    # stalled with no worker processes ever appearing). One pool, reused for
    # both interp and adaptive below, so worker startup (~re-importing
    # astropy/reproject fresh per spawned worker) is paid once; a throwaway
    # warm-up call keeps that startup cost out of the timed measurement, the
    # same "warm" convention used for reprojax's JIT compilation above.
    with ProcessPoolExecutor(
        max_workers=BATCH_WORKERS, mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker
    ) as pool:
        list(pool.map(_run_interp_one, headers[:1]))  # warm-up: pay worker startup here
        t0 = time.perf_counter()
        list(pool.map(_run_interp_one, headers))
        t_orig_interp_batch_mc = time.perf_counter() - t0
        print(f"  original reproject_interp (process pool, {BATCH_WORKERS} workers): {t_orig_interp_batch_mc:.3f}s")

        jitted_interp_batch = jax.jit(reprojax.reproject_interp_batched)
        t_interp_batch_compile = timed(jitted_interp_batch, data_batch, mwcs_in_batch, mwcs_out, rotation_batch, x_out, y_out)
        t_interp_batch_warm = timed(jitted_interp_batch, data_batch, mwcs_in_batch, mwcs_out, rotation_batch, x_out, y_out)
        print(f"  reprojax reproject_interp_batched:                 {t_interp_batch_warm:.3f}s warm ({t_interp_batch_compile:.3f}s incl. compile)")

        t0 = time.perf_counter()
        for h in headers:
            original_reproject_adaptive((data, WCS(h)), header_out, shape_out=shape_out)
        t_orig_adaptive_batch = time.perf_counter() - t0
        print(f"  original reproject_adaptive (looped, 1 core):      {t_orig_adaptive_batch:.3f}s")

        list(pool.map(_run_adaptive_one, headers[:1]))  # warm-up
        t0 = time.perf_counter()
        list(pool.map(_run_adaptive_one, headers))
        t_orig_adaptive_batch_mc = time.perf_counter() - t0
        print(f"  original reproject_adaptive (process pool, {BATCH_WORKERS} workers): {t_orig_adaptive_batch_mc:.3f}s")

    jitted_adaptive_batch = jax.jit(reproject_adaptive_batched, static_argnames=("max_window_radius",))
    t_adaptive_batch_compile = timed(
        jitted_adaptive_batch,
        data_batch,
        mwcs_in_batch,
        mwcs_out,
        rotation_batch,
        x_out,
        y_out,
        max_window_radius=BATCH_ADAPTIVE_MAX_WINDOW_RADIUS,
    )
    t_adaptive_batch_warm = timed(
        jitted_adaptive_batch,
        data_batch,
        mwcs_in_batch,
        mwcs_out,
        rotation_batch,
        x_out,
        y_out,
        max_window_radius=BATCH_ADAPTIVE_MAX_WINDOW_RADIUS,
    )
    print(f"  reprojax reproject_adaptive (vmapped):             {t_adaptive_batch_warm:.3f}s warm ({t_adaptive_batch_compile:.3f}s incl. compile)")

    print("\n=== summary (backend=%s) ===" % backend)
    print(
        f"interp   single: 1core={t_orig_interp:.3f}s multicore={t_orig_interp_mc:.3f}s reprojax={t_interp_warm:.3f}s "
        f"| reprojax vs 1core={t_orig_interp/t_interp_warm:.1f}x vs multicore={t_orig_interp_mc/t_interp_warm:.1f}x"
    )
    print(
        f"adaptive single: 1core={t_orig_adaptive:.3f}s multicore={t_orig_adaptive_mc:.3f}s reprojax={t_adaptive_warm:.3f}s "
        f"| reprojax vs 1core={t_orig_adaptive/t_adaptive_warm:.1f}x vs multicore={t_orig_adaptive_mc/t_adaptive_warm:.1f}x"
    )
    print(
        f"interp   batch({BATCH_SIZE}): 1core={t_orig_interp_batch:.3f}s multicore={t_orig_interp_batch_mc:.3f}s reprojax={t_interp_batch_warm:.3f}s "
        f"| reprojax vs 1core={t_orig_interp_batch/t_interp_batch_warm:.1f}x vs multicore={t_orig_interp_batch_mc/t_interp_batch_warm:.1f}x"
    )
    print(
        f"adaptive batch({BATCH_SIZE}): 1core={t_orig_adaptive_batch:.3f}s multicore={t_orig_adaptive_batch_mc:.3f}s reprojax={t_adaptive_batch_warm:.3f}s "
        f"| reprojax vs 1core={t_orig_adaptive_batch/t_adaptive_batch_warm:.1f}x vs multicore={t_orig_adaptive_batch_mc/t_adaptive_batch_warm:.1f}x"
    )


if __name__ == "__main__":
    main()
