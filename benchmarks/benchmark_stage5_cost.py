"""The Stage 5 cost rows: warm cache-hit build and warm real-pair reprojection.

Run from the reprojax root, in an environment with ``jax_enable_x64``:

    .venv/bin/python benchmarks/benchmark_stage5_cost.py

Reports single-host medians (5 warm-ups + 100 reps). Both figures are
budgeted in the Stage 5 plan: the warm ``build_transform`` cache hit at
5.0 ms and the warm real-pair ``reproject_interp`` at 0.608 s, CPU only.
They are machine-specific; another host gives different numbers inside the
same budgets. The cache-hit row deliberately does not call the returned
transform, because one JAX dispatch (~27 ms for 1000 points) dwarfs the
~2 ms build.
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


def main():
    h_in = fits.getheader("tests/data/spherex_l2b_v20.fits")
    h_out = fits.Header.fromtextfile("tests/data/galactic_template_header.fits")
    wcs_in, wcs_out = WCS(h_in), WCS(h_out)

    # Warm build_transform cache hit on pre-built WCS objects. This is the
    # cache-hit cost proper; do not call the returned transform inside the
    # timing, because a JAX dispatch dwarfs it (~27 ms for a 1000-point call).
    build_transform(wcs_in, wcs_out)
    for _ in range(WARMUPS):
        build_transform(wcs_in, wcs_out)
    times = []
    for _ in range(REPS):
        start = time.perf_counter()
        build_transform(wcs_in, wcs_out)
        times.append((time.perf_counter() - start) * 1e3)
    median, p90 = _median(times)
    print(f"warm build_transform cache hit, pre-built WCS: {median:.3f} ms (p90 {p90:.3f})")

    # The same call with two WCS(header) parses added, which is what a
    # per-image path actually pays. The parses were always there.
    times = []
    for _ in range(REPS):
        start = time.perf_counter()
        build_transform(WCS(h_in), WCS(h_out))
        times.append((time.perf_counter() - start) * 1e3)
    median, p90 = _median(times)
    print(f"warm build_transform cache hit, fresh WCS(header): {median:.3f} ms (p90 {p90:.3f})")

    # Warm real-pair reproject_interp, 2040^2 -> 1201^2.
    data = fits.getdata("tests/data/spherex_l2b_v20.fits").astype(np.float32)
    reproject_interp((data, WCS(h_in)), h_out)
    for _ in range(WARMUPS):
        reproject_interp((data, WCS(h_in)), h_out)
    times = []
    for _ in range(REPS):
        start = time.perf_counter()
        result = reproject_interp((data, WCS(h_in)), h_out)
        jax.block_until_ready(result[0])
        times.append(time.perf_counter() - start)
    median, p90 = _median(times)
    print(f"warm real-pair reproject_interp: {median:.3f} s (p90 {p90:.3f})")


if __name__ == "__main__":
    main()
