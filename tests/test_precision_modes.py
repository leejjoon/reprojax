"""Precision-mode contract, asserted in fresh processes.

Each case runs in its own interpreter because ``jax_enable_x64`` is a
process-global flag: it can only be set once, before the first float64
value, and cannot be toggled mid-session. Only a fresh process can prove
what a real user sees.

Three cases, matching the documented contract:

1. ``f64`` with x64 on  -> runs, no warnings.
2. ``f64`` without x64  -> the documented ``ValueError``, from both entry
   points (the default mode is unusable without x64, which is why the
   README leads with the rule).
3. ``f32`` without x64  -> runs, **zero warnings**, on all three paths,
   including the generic-provider coverage branch that the pixel-map path
   skips.

The warning assertions are the point of this module: a hardcoded float64
constant on a float32 path is numerically harmless but makes every user
call emit a truncation warning, which is a real regression signal.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# A tiny synthetic pair: 64x64 TAN in, 40x40 TAN out, slightly offset.
# No tests/data dependency -- this must run from an installed wheel too.
_PREAMBLE = textwrap.dedent(
    """
    import sys, warnings
    import jax

    X64 = sys.argv[1] == "x64"
    if X64:
        jax.config.update("jax_enable_x64", True)

    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    import reprojax

    def header(crval1, n=64):
        h = fits.Header()
        h["NAXIS"] = 2
        h["NAXIS1"] = h["NAXIS2"] = n
        h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
        h["CRPIX1"] = h["CRPIX2"] = n / 2 + 0.5
        h["CRVAL1"], h["CRVAL2"] = crval1, 54.0
        h["CDELT1"], h["CDELT2"] = -1e-3, 1e-3
        return h

    def data(n=64):
        y, x = np.mgrid[:n, :n]
        return np.exp(-(((x - n / 2) ** 2 + (y - n / 2) ** 2) / (2 * 6.0 ** 2))).astype(
            np.float32
        )

    H_IN, H_OUT = header(210.0), header(210.02, n=40)
    DATA = data()
    SHAPE = (40, 40)

    def capture(fn):
        with warnings.catch_warnings(record=True) as got:
            warnings.simplefilter("always")
            out = fn()
        return out, [str(w.message) for w in got]
    """
)


def run(script_body, *args, expect_ok=True):
    """Run a snippet in a fresh interpreter with this checkout importable."""
    script = _PREAMBLE + textwrap.dedent(script_body)
    proc = subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    if expect_ok:
        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    else:
        assert proc.returncode != 0, f"expected failure, got stdout={proc.stdout}"
    return proc


def test_f64_with_x64_runs_without_warnings():
    proc = run(
        """
        print("x64", jax.config.jax_enable_x64)
        (out, fp), warns = capture(
            lambda: reprojax.reproject_interp(
                (DATA, WCS(H_IN)), H_OUT, shape_out=SHAPE
            )
        )
        assert warns == [], warns
        (a_out, a_fp), a_warns = capture(
            lambda: reprojax.reproject_adaptive(
                (DATA, WCS(H_IN)), H_OUT, shape_out=SHAPE, max_window_radius=4
            )
        )
        assert a_warns == [], a_warns
        assert np.isfinite(np.asarray(out)).any()
        print("ok")
        """,
        "x64",
    )
    assert "ok" in proc.stdout and "x64 True" in proc.stdout


def test_f64_without_x64_raises_the_documented_error():
    proc = run(
        """
        for name, fn in (
            ("interp", lambda: reprojax.reproject_interp(
                (DATA, WCS(H_IN)), H_OUT, shape_out=SHAPE)),
            ("adaptive", lambda: reprojax.reproject_adaptive(
                (DATA, WCS(H_IN)), H_OUT, shape_out=SHAPE, max_window_radius=4)),
        ):
            try:
                fn()
            except ValueError as exc:
                assert "requires float64" in str(exc), str(exc)
                print(name, "raised")
            else:
                raise AssertionError(f"{name} did not raise")
        """,
        "nox64",
    )
    assert "interp raised" in proc.stdout
    assert "adaptive raised" in proc.stdout


def test_f32_without_x64_emits_no_warnings_on_all_three_paths():
    proc = run(
        """
        assert jax.config.jax_enable_x64 is False

        (out, fp), warns = capture(
            lambda: reprojax.reproject_interp(
                (DATA, WCS(H_IN)), H_OUT, shape_out=SHAPE, matrix_precision="f32"
            )
        )
        assert warns == [], ("interp", warns)

        (a_out, a_fp), a_warns = capture(
            lambda: reprojax.reproject_adaptive(
                (DATA, WCS(H_IN)), H_OUT, shape_out=SHAPE,
                max_window_radius=4, matrix_precision="f32",
            )
        )
        assert a_warns == [], ("adaptive", a_warns)

        # Generic provider: a plain callable with no pixel_map, which takes
        # the coverage-interpolation branch the pixel-map path skips.
        def identity(x, y):
            return x, y

        import jax.numpy as jnp
        (g_out, g_fp), g_warns = capture(
            lambda: reprojax.reproject_interp_core(
                jnp.ones((8, 8), dtype=jnp.float32),
                jnp.array([[1.0, 2.5]]),
                jnp.array([[1.0, 2.5]]),
                identity,
            )
        )
        assert g_warns == [], ("generic", g_warns)
        print("ok")
        """,
        "nox64",
    )
    assert "ok" in proc.stdout
