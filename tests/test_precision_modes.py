"""Precision-mode contract, asserted in fresh processes.

Each case runs in its own interpreter because ``jax_enable_x64`` is a
process-global flag: it can only be set once, before the first float64
value, and cannot be toggled mid-session. Only a fresh process can prove
what a real user sees.

The children import ``reprojax`` exactly the way the suite did — no
``PYTHONPATH`` injection, no source overlay — and assert they resolved the
*same installation* as the parent. That is what makes these tests valid in
CI's wheel job, where the only ``reprojax`` on the path is the built wheel.

Cases:

1. ``f64`` with x64 on   -> runs, no warnings.
2. ``f64`` without x64   -> the documented ``ValueError``, from both entry
   points (the default mode is unusable without x64, which is why the README
   leads with the rule).
3. ``f32`` without x64   -> runs, **zero warnings**, on all three paths,
   including the generic-provider coverage branch the pixel-map path skips,
   with shape/dtype/finiteness/footprint and exact known values asserted.
4. ``f32`` vs ``f64`` coordinates with x64 on -> f32 coordinate error stays
   inside the published cap.

The warning assertions are the point of this module: a hardcoded float64
constant on a float32 path is numerically harmless but makes every user call
emit a truncation warning, which is a real regression signal.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

import reprojax

ROOT = Path(__file__).resolve().parent.parent

#: The parent's own installation. The children must resolve this same file,
#: otherwise they are testing something other than what this suite tests.
PARENT_REPROJAX = Path(reprojax.__file__).resolve()

#: Loosest published f32 cap, applied **per coordinate** (forward, real
#: SPHEREx pair: 6e-4 px forward / 5e-4 px reverse, read as a bound on each
#: of x and y separately, not on their Euclidean norm). The synthetic pair
#: here is far smaller, so this is a ceiling, not a target.
F32_PER_COORD_CAP_PX = 6e-4

# A tiny synthetic pair: 64x64 TAN in, 40x40 TAN out, slightly offset.
# No tests/data dependency -- this must run from an installed wheel too.
_PREAMBLE = textwrap.dedent(
    """
    import os, sys, warnings
    import jax

    EXPECT_X64 = sys.argv[1] == "x64"
    if EXPECT_X64:
        jax.config.update("jax_enable_x64", True)
    # Set explicitly by the parent; assert it so a stray ambient value in the
    # calling environment can never silently change what this case exercises.
    assert jax.config.jax_enable_x64 is EXPECT_X64, (
        f"x64 is {jax.config.jax_enable_x64}, expected {EXPECT_X64}")
    assert "PYTHONPATH" not in os.environ, os.environ.get("PYTHONPATH")

    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    import reprojax

    print("REPROJAX_FILE", reprojax.__file__)
    print("SYS_PREFIX", sys.prefix)

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


def run(script_body, x64, expect_ok=True):
    """Run a snippet in a fresh interpreter, with no path injection.

    ``JAX_ENABLE_X64`` is set explicitly (not inherited), so each case is
    hermetic. The child runs outside the repository too, so ``python -c``'s
    implicit cwd entry cannot supply a ``reprojax`` package.
    """
    script = _PREAMBLE + textwrap.dedent(script_body)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "JAX_ENABLE_X64")}
    env["JAX_ENABLE_X64"] = "1" if x64 else "0"
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(
            [sys.executable, "-c", script, "x64" if x64 else "nox64"],
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env,
        )
    if expect_ok:
        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
        # Both facts must hold for every successful child: same installation,
        # same interpreter prefix. In the wheel job that means the wheel.
        assert f"REPROJAX_FILE {PARENT_REPROJAX}" in proc.stdout, (
            f"child imported a different reprojax than the suite: {proc.stdout!r}"
        )
        assert f"SYS_PREFIX {sys.prefix}" in proc.stdout, proc.stdout
    else:
        assert proc.returncode != 0, f"expected failure, got stdout={proc.stdout}"
    return proc


def test_f64_with_x64_runs_without_warnings():
    proc = run(
        """
        (out, fp), warns = capture(
            lambda: reprojax.reproject_interp(
                (DATA, WCS(H_IN)), H_OUT, shape_out=SHAPE
            )
        )
        assert warns == [], warns
        assert np.asarray(out).shape == SHAPE
        assert np.asarray(out).dtype == np.float32  # matches the input data
        assert np.isfinite(np.asarray(out)).any()

        (a_out, a_fp), a_warns = capture(
            lambda: reprojax.reproject_adaptive(
                (DATA, WCS(H_IN)), H_OUT, shape_out=SHAPE, max_window_radius=4
            )
        )
        assert a_warns == [], a_warns
        assert np.asarray(a_out).shape == SHAPE
        assert np.isfinite(np.asarray(a_out)).any()
        print("ok")
        """,
        x64=True,
    )
    assert "ok" in proc.stdout


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
        x64=False,
    )
    assert "interp raised" in proc.stdout
    assert "adaptive raised" in proc.stdout


def test_f32_without_x64_is_warning_free_and_correct():
    proc = run(
        """
        (out, fp), warns = capture(
            lambda: reprojax.reproject_interp(
                (DATA, WCS(H_IN)), H_OUT, shape_out=SHAPE, matrix_precision="f32"
            )
        )
        assert warns == [], ("interp", warns)
        out, fp = np.asarray(out), np.asarray(fp)
        assert out.shape == SHAPE, out.shape
        assert fp.shape == SHAPE, fp.shape
        assert out.dtype == np.float32, out.dtype
        assert fp.dtype == np.float32, fp.dtype
        assert np.isfinite(out).any(), "no valid output pixels"
        # This 40x40 output sits entirely inside the 64x64 input, so full
        # coverage is the expected result here; the partial-coverage case
        # is checked below with a 60x60 output that straddles the edge.
        assert np.isfinite(out).all(), "expected full coverage for the inner grid"
        # Footprint is exactly the finite mask, by contract.
        assert np.array_equal(fp == 1.0, np.isfinite(out)), "footprint != finite mask"

        # The 40x40 output lies entirely inside the 64x64 input, so it is
        # fully covered. A 60x60 output straddles the edge, which is what
        # exercises partial coverage and the invalid-region path.
        (p_out, p_fp), p_warns = capture(
            lambda: reprojax.reproject_interp(
                (DATA, WCS(H_IN)), header(210.02, n=60), shape_out=(60, 60),
                matrix_precision="f32",
            )
        )
        assert p_warns == [], ("partial", p_warns)
        p_out, p_fp = np.asarray(p_out), np.asarray(p_fp)
        assert p_out.shape == (60, 60), p_out.shape
        assert np.isfinite(p_out).any() and not np.isfinite(p_out).all()
        assert np.array_equal(p_fp == 1.0, np.isfinite(p_out))

        (a_out, a_fp), a_warns = capture(
            lambda: reprojax.reproject_adaptive(
                (DATA, WCS(H_IN)), H_OUT, shape_out=SHAPE,
                max_window_radius=4, matrix_precision="f32",
            )
        )
        assert a_warns == [], ("adaptive", a_warns)
        a_out, a_fp = np.asarray(a_out), np.asarray(a_fp)
        assert a_out.shape == SHAPE and a_out.dtype == np.float32
        assert np.isfinite(a_out).any()
        assert np.array_equal(a_fp == 1.0, np.isfinite(a_out))

        # Generic provider: a plain callable with no pixel_map, which takes
        # the coverage-interpolation branch the pixel-map path skips.
        import jax.numpy as jnp
        ramp = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)

        def identity(x, y):
            return x, y

        (g_out, g_fp), g_warns = capture(
            lambda: reprojax.reproject_interp_core(
                ramp,
                jnp.array([[0.5, 2.5, -1.0]]),
                jnp.array([[0.5, 0.5, 0.5]]),
                identity,
            )
        )
        assert g_warns == [], ("generic", g_warns)
        g_out, g_fp = np.asarray(g_out), np.asarray(g_fp)
        # Bilinear on the ramp value 4*row + col is exactly 4*y + x at
        # half-integer positions: 4*0.5+0.5 = 2.5, 4*0.5+2.5 = 4.5.
        assert g_out.shape == (1, 3), g_out.shape
        assert g_out.dtype == np.float32, g_out.dtype
        assert g_out[0, 0] == np.float32(2.5), g_out
        assert g_out[0, 1] == np.float32(4.5), g_out
        assert np.isnan(g_out[0, 2]), g_out
        assert np.array_equal(g_fp, np.array([[1.0, 1.0, 0.0]], dtype=np.float32)), g_fp
        print("ok")
        """,
        x64=False,
    )
    assert "ok" in proc.stdout


def test_f32_coordinates_stay_inside_the_published_cap():
    """f32 is a coordinate-precision choice; bound the coordinate error.

    Only the coordinate map is compared. Image agreement and footprint
    equality are covered by the real-pair tests, because a cap in pixels
    says nothing about resampled values.
    """
    proc = run(
        """
        from reprojax import build_transform
        import jax.numpy as jnp

        ny, nx = SHAPE
        y, x = jnp.meshgrid(jnp.arange(ny, dtype=jnp.float64),
                            jnp.arange(nx, dtype=jnp.float64), indexing="ij")

        t32 = build_transform(WCS(H_IN), WCS(H_OUT), matrix_precision="f32")
        t64 = build_transform(WCS(H_IN), WCS(H_OUT), matrix_precision="f64")
        x32, y32 = t32(x, y)
        x64, y64 = t64(x, y)
        x32, y32, x64, y64 = map(np.asarray, (x32, y32, x64, y64))

        both = np.isfinite(x32) & np.isfinite(x64) & np.isfinite(y32) & np.isfinite(y64)
        assert both.sum() > 0.5 * both.size, f"only {both.sum()} comparable pixels"
        # The published caps bound each coordinate separately, so compare
        # the worst single-axis error. The Euclidean distance is reported
        # too (it can exceed either axis by up to sqrt(2)) but is not what
        # the cap is stated against.
        dx = np.abs(x32 - x64)[both]
        dy = np.abs(y32 - y64)[both]
        per_coord = np.maximum(dx, dy)
        euclid = np.hypot(dx, dy)
        print("MEASURED_PER_COORD_MAX_PX", float(per_coord.max()))
        print("MEASURED_PER_COORD_RMS_PX", float(np.sqrt(np.mean(per_coord ** 2))))
        print("MEASURED_EUCLID_MAX_PX", float(euclid.max()))
        print("COMPARED_PX", int(both.sum()))
        """,
        x64=True,
    )
    measured = float(proc.stdout.split("MEASURED_PER_COORD_MAX_PX")[1].split()[0])
    compared = int(proc.stdout.split("COMPARED_PX")[1].split()[0])
    assert compared > 0
    assert measured <= F32_PER_COORD_CAP_PX, (
        f"f32 per-coordinate error {measured:.2e} px exceeds the published "
        f"cap {F32_PER_COORD_CAP_PX:.0e} px over {compared} pixels"
    )
