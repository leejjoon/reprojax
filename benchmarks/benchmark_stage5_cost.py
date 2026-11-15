"""The Stage 5 cost rows: warm cache-hit build and warm real-pair reprojection.

Run from the reprojax root, in an environment with ``jax_enable_x64``:

    .venv/bin/python benchmarks/benchmark_stage5_cost.py

Reports single-host medians (5 warm-ups + 100 reps). Both figures are
budgeted in the Stage 5 plan: the warm ``build_transform`` cache hit at
5.0 ms and the warm real-pair ``reproject_interp`` at 0.608 s, CPU only.
They are machine-specific; another host gives different numbers inside the
same budgets.

Every timed and untimed row is synchronized with ``jax.block_until_ready``.
The cache-hit row deliberately does not call the returned transform, because
one JAX dispatch (~27 ms for 1000 points) dwarfs the ~2 ms build; for the
``reproject_interp`` row the *whole* result tuple is awaited, not just the
first array, so the second array cannot leak into the next iteration.
"""

import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from reprojax import reproject_interp
from reprojax.transform import build_transform

WARMUPS = 5
REPS = 100


def _median(values):
    return float(np.median(values)), float(np.percentile(values, 90))


def _print_row(label, unit, times):
    median, p90 = _median(times)
    print(f"{label}: {median:.3f} {unit} (p90 {p90:.3f}, n={len(times)})")


def main():
    h_in = fits.getheader("tests/data/spherex_l2b_v20.fits")
    h_out = fits.Header.fromtextfile("tests/data/galactic_template_header.fits")
    wcs_in, wcs_out = WCS(h_in), WCS(h_out)

    print(f"jax backend: {jax.default_backend()} ({jax.devices()[0]})")

    # Warm build_transform cache hit on pre-built WCS objects. There is no JAX
    # work to await here -- build_transform is host-side -- so synchronization
    # applies only to the warm-up that compiles nothing.
    build_transform(wcs_in, wcs_out)
    for _ in range(WARMUPS):
        build_transform(wcs_in, wcs_out)
    times = []
    for _ in range(REPS):
        start = time.perf_counter()
        build_transform(wcs_in, wcs_out)
        times.append((time.perf_counter() - start) * 1e3)
    _print_row("warm build_transform cache hit, pre-built WCS", "ms", times)

    # The same call with two WCS(header) parses added, which is what a
    # per-image path actually pays. The parses were always there.
    times = []
    for _ in range(REPS):
        start = time.perf_counter()
        build_transform(WCS(h_in), WCS(h_out))
        times.append((time.perf_counter() - start) * 1e3)
    _print_row("warm build_transform cache hit, fresh WCS(header)", "ms", times)

    # Warm real-pair reproject_interp, 2040^2 -> 1201^2, native float32 data
    # (the FITS array's own dtype; this path is float32 regardless).
    data = fits.getdata("tests/data/spherex_l2b_v20.fits")
    print(f"real-pair input: {data.dtype} {data.shape}, output {h_out['NAXIS1']}x{h_out['NAXIS2']}")
    result = reproject_interp((data, WCS(h_in)), h_out)
    jax.block_until_ready(result)
    for _ in range(WARMUPS):
        jax.block_until_ready(reproject_interp((data, WCS(h_in)), h_out))
    times = []
    for _ in range(REPS):
        start = time.perf_counter()
        result = reproject_interp((data, WCS(h_in)), h_out)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)
    _print_row("warm real-pair reproject_interp", "s", times)


if __name__ == "__main__":
    main()
