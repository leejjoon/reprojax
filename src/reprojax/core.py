"""The ``PixelTransform`` contract.

This is reprojax's central seam. Reprojection algorithms are written against
this contract, not against any particular WCS library — mapprojax is the
default provider (see :mod:`reprojax.transform`), and other providers (e.g.
an astropy-WCS ``pure_callback`` fallback for projections mapprojax doesn't
cover) can be added later behind the same interface.

Contract
--------
A ``PixelTransform`` is a callable::

    x_in, y_in = transform(x_out, y_out)

mapping output-image pixel coordinates to input-image pixel coordinates,
with the following conventions:

- **0-based, pixel-center convention**: pixel ``0`` is the center of the
  first pixel (the ``scipy.ndimage.map_coordinates`` / FITS-``origin=0``
  convention), not FITS's 1-based convention.
- **Shape**: ``x_out``, ``y_out`` may be arrays of any broadcastable shape;
  ``x_in``, ``y_in`` come back the same shape. Batching over many images is
  done by ``vmap``-ing the transform (and whatever consumes it), not by the
  transform itself handling a batch dimension.
- **Invalid points are NaN**: pixels with no valid corresponding input
  position (off the projection's visible hemisphere, off the celestial
  sphere) come back as NaN in both ``x_in`` and ``y_in``.
- **Units**: everything at this boundary is in pixels. Providers built on
  spherical projections will generally work in radians internally, but that
  is an implementation detail, not part of the contract.
- **Jit/grad-safe**: a conforming transform is built from pure array
  operations (no host callbacks, no data-dependent Python control flow) so
  that it can be used under ``jax.jit`` and differentiated with
  ``jax.grad``.
"""

from typing import Any, Protocol, Tuple


class PixelTransform(Protocol):
    """Callable mapping output pixel coordinates to input pixel coordinates.

    See the module docstring for the full contract. This is a
    :class:`typing.Protocol`: any callable with this signature satisfies it
    structurally, with no need to import or subclass anything from
    reprojax.
    """

    def __call__(self, x_out: Any, y_out: Any) -> Tuple[Any, Any]:
        ...
