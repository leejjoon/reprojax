"""Reproject a real SPHEREx L2b exposure onto its Galactic mosaicking
template, using reprojax's public API directly -- no manual WCS/transform
construction needed.

This replaces mapprojax's original demo scripts (reproject_sip.py,
reproject_package_comparison.py, reproject_fits_example.py), which built
mapprojax WCS objects and rotation matrices by hand before mapprojax's own
reprojection API existed. What used to take ~40 lines of manual TanJax +
frame-rotation setup is now two calls: reprojax.reproject_interp and
reprojax.reproject_adaptive.

Run from the reprojax repo root:
    python examples/reproject_spherex.py
"""

import time
from pathlib import Path

import jax
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

jax.config.update("jax_enable_x64", True)

import reprojax  # noqa: E402  (must follow the x64 config update)

DATA_DIR = Path(__file__).parent.parent / "tests" / "data"


def main():
    with fits.open(DATA_DIR / "spherex_l2b_v20.fits") as hdul:
        data = hdul[0].data
        wcs_in = WCS(hdul[0].header)

    header_out = fits.Header.fromtextfile(DATA_DIR / "galactic_template_header.fits")
    shape_out = (header_out["NAXIS2"], header_out["NAXIS1"])

    print(f"Reprojecting {data.shape} (ICRS, TAN+SIP) -> {shape_out} (Galactic, TAN)")

    for name, reproject_fn in [("reproject_interp", reprojax.reproject_interp), ("reproject_adaptive", reprojax.reproject_adaptive)]:
        t0 = time.perf_counter()
        result, footprint = reproject_fn((data, wcs_in), header_out)
        result.block_until_ready()
        elapsed = time.perf_counter() - t0

        valid_frac = float(footprint.mean())
        print(f"  {name}: {elapsed:.2f}s, {valid_frac:.1%} footprint coverage")

        out_path = Path(__file__).parent / f"reprojected_{name}.fits"
        fits.writeto(out_path, np.asarray(result), header_out, overwrite=True)
        print(f"    -> saved {out_path}")


if __name__ == "__main__":
    main()
