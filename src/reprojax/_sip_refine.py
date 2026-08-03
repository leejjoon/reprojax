"""Newton refinement of the SIP inverse, for WCSes whose AP/BP reverse
polynomial is not an accurate enough approximation of the true inverse.

Real-world SIP files ship AP/BP as an independently-fitted approximation of
the inverse of the forward A/B polynomial (see ``reprojax._astropy_wcs``).
For some files that approximation is excellent; for others (observed on a
real SPHEREx L2b header, where the direct AP/BP evaluation disagreed with
astropy's iterative ``all_world2pix`` by up to ~0.03 px) it is not accurate
enough for the milestone's < 1e-4 px bar.

This refines the AP/BP-based estimate by Newton's method against the
*forward* polynomial, using only mapprojax's public ``pix_to_native`` API
and JAX autodiff for the Jacobian — never mapprojax's protected internals.
Because both this refinement and astropy's own iterative solver converge to
the same fixed point (the pixel whose forward projection reproduces the
target sky direction), a single step already lands within ~1e-7 px of
astropy's answer on the real header this was diagnosed against; a second
step is kept as a safety margin at negligible extra cost.
"""

import jax
import jax.numpy as jnp

DEFAULT_REFINE_STEPS = 2


def refine_pixel_estimate(mwcs, x0, y0, xn_target, yn_target, zn_target, n_steps=DEFAULT_REFINE_STEPS):
    """Newton-refine a pixel estimate so ``mwcs.pix_to_native`` reproduces
    the target native unit vector.

    Parameters
    ----------
    mwcs : a mapprojax WCS (e.g. TanJax) with SIP.
    x0, y0 : arrays, any common shape
        Initial pixel estimate (typically from the direct AP/BP evaluation).
    xn_target, yn_target, zn_target : arrays, same shape as x0
        Target native-frame unit vector.
    n_steps : int
        Number of Newton steps (static; unrolled).

    Returns
    -------
    x, y : arrays, same shape as x0
    """
    shape = jnp.shape(x0)

    def residual(xy, target):
        x, y = xy[0], xy[1]
        xn, yn, zn = mwcs.pix_to_native(x, y)
        return jnp.array([xn, yn, zn]) - target

    def step(xy, target):
        r = residual(xy, target)
        jac = jax.jacfwd(residual)(xy, target)  # (3, 2)
        # Gauss-Newton normal equations: the sphere constraint makes this a
        # rank-deficient-by-one 3x2 system, so we solve the 2x2 normal
        # equations rather than a general least squares.
        jtj = jac.T @ jac
        jtr = jac.T @ r
        # Small Levenberg-Marquardt damping, scaled to jtj's own magnitude
        # (so it's meaningful at any pixel scale). Standard practice for a
        # Gauss-Newton solve: negligible effect on the converged answer in
        # ordinary use, but guards against jtj itself being poorly
        # conditioned (e.g. very high SIP order with widely-scaled
        # coefficients). It does *not* address the narrow, real-data-
        # irrelevant NaN-second-derivative case documented on
        # compose_transform / in tests/test_differentiability.py — that one
        # isn't a jtj-conditioning issue (empirically unaffected by this
        # damping across many orders of magnitude) but something specific
        # to differentiating through jacfwd at an exact geometric symmetry.
        damping = 1e-8 * jnp.trace(jtj)
        jtj = jtj + damping * jnp.eye(2, dtype=jtj.dtype)
        return xy - jnp.linalg.solve(jtj, jtr)

    def single_point(x0_i, y0_i, xn_i, yn_i, zn_i):
        target = jnp.array([xn_i, yn_i, zn_i])
        xy = jnp.array([x0_i, y0_i])
        for _ in range(n_steps):
            xy = step(xy, target)
        return xy[0], xy[1]

    x, y = jax.vmap(single_point)(
        jnp.ravel(x0), jnp.ravel(y0), jnp.ravel(xn_target), jnp.ravel(yn_target), jnp.ravel(zn_target)
    )
    return x.reshape(shape), y.reshape(shape)
