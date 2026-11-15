"""Tier-1 validation: the refusal set and the cache-key invariants (Stage 5).

Two things are checked throughout:
  - an unsupported input is refused on **every** call, including when a warm
    cache entry exists for a WCS it would have collided with before this
    change (TPV normalises to plain TAN, so it shares every raw attribute);
  - the transform cache key is built from validated output, so equivalent
    encodings share one entry and genuinely different WCSes never do.

Module-level imports are restricted to symbols that exist before the Stage 5
migration, so the pre-fix falsification runs fail on assertions.
"""

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from reprojax import reproject_adaptive, reproject_interp
from reprojax.transform import _TRANSFORM_CACHE, build_transform

from _fixtures import make_header

PIX_TOL = 1e-4


def _tan(**extra):
    h = make_header((256.0, 256.0), (180.0, 30.0), 1.0, naxis=(64, 64))
    for key, value in extra.items():
        h[key] = value
    return h


def _random_data(header, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(header["NAXIS2"], header["NAXIS1"]))


# ---------------------------------------------------------------------------
# TPV: the hidden-distortion case.  Astropy reports ctype == RA---TAN,
# has_distortion False and get_pv() == [], so every raw attribute matches a
# plain TAN header -- and the old fingerprint therefore collided.
# ---------------------------------------------------------------------------


def test_tpv_header_is_refused():
    h = _tan(CTYPE1="RA---TPV", CTYPE2="DEC--TPV", PV1_1=3e-4, PV1_2=1e-4)
    with pytest.raises(ValueError, match="TPV"):
        build_transform(WCS(h), WCS(h))


def test_tpv_refusal_is_reachable_with_a_warm_tan_entry():
    """Validation runs before the cache lookup, so a warm TAN entry cannot
    shadow the TPV refusal.  Pre-fix this returned the TAN transform.
    """
    build_transform.cache_clear()
    tan = WCS(_tan())
    build_transform(tan, tan)
    assert len(_TRANSFORM_CACHE) == 1

    tpv = WCS(_tan(CTYPE1="RA---TPV", CTYPE2="DEC--TPV", PV1_1=3e-4, PV1_2=1e-4))
    with pytest.raises(ValueError, match="TPV"):
        build_transform(tpv, tpv)


def test_a_refusal_does_not_populate_the_cache():
    build_transform.cache_clear()
    tan = WCS(_tan())
    build_transform(tan, tan)
    before = len(_TRANSFORM_CACHE)

    tpv = WCS(_tan(CTYPE1="RA---TPV", CTYPE2="DEC--TPV", PV1_1=3e-4))
    with pytest.raises(ValueError):
        build_transform(tpv, tan)
    assert len(_TRANSFORM_CACHE) == before


def test_cubes_are_refused_with_the_celestial_message():
    """All three positions, for a Header *and* a WCS, through both wrappers.

    The Header row is why the guard lives in ``_as_wcs``: a 3-D header still
    yields a valid 2-D shape, so shape inference cannot see the third axis.
    """
    h_cube = fits.Header()
    h_cube["NAXIS"] = 3
    h_cube["NAXIS1"], h_cube["NAXIS2"], h_cube["NAXIS3"] = 64, 64, 2
    h_cube["CTYPE1"], h_cube["CTYPE2"], h_cube["CTYPE3"] = "RA---TAN", "DEC--TAN", "FREQ"
    h_cube["CRPIX1"], h_cube["CRPIX2"], h_cube["CRPIX3"] = 32.0, 32.0, 1.0
    h_cube["CRVAL1"], h_cube["CRVAL2"], h_cube["CRVAL3"] = 180.0, 30.0, 1e9
    h_cube["CD1_1"], h_cube["CD2_2"], h_cube["CD3_3"] = -1.0 / 3600, 1.0 / 3600, 1e6

    h_2d = make_header((32.0, 32.0), (180.0, 30.0), 1.0, naxis=(64, 64))
    data = _random_data(h_2d)

    for wrapper in (reproject_interp, reproject_adaptive):
        # input as a 3-D WCS and as a 3-D Header
        with pytest.raises(ValueError, match="celestial"):
            wrapper((np.zeros((2, 64, 64), dtype=np.float32), WCS(h_cube)), h_2d, shape_out=(64, 64))
        with pytest.raises(ValueError, match="celestial"):
            wrapper((np.zeros((2, 64, 64), dtype=np.float32), h_cube), h_2d, shape_out=(64, 64))
        # output_projection as a 3-D WCS and as a 3-D Header, with shape_out
        # given AND inferred -- the guard must not depend on shape inference.
        for shape_out in ((64, 64), None):
            with pytest.raises(ValueError, match="celestial"):
                wrapper((data, WCS(h_2d)), WCS(h_cube), shape_out=shape_out)
            with pytest.raises(ValueError, match="celestial"):
                wrapper((data, WCS(h_2d)), h_cube, shape_out=shape_out)

    # A 3-D WCS whose array_shape was never populated is still a cube, so the
    # axis guard catches it before shape inference can complain.
    no_shape = WCS(h_cube)
    no_shape.array_shape = None
    with pytest.raises(ValueError, match="celestial"):
        reproject_interp((data, WCS(h_2d)), no_shape)


def test_a_two_axis_wcs_without_shape_information_keeps_the_generic_message():
    """A 2-D WCS with no ``array_shape`` gets the generic message; the cube
    refusal is the separate axis guard in ``_as_wcs``.
    """
    h = make_header((256.0, 256.0), (180.0, 30.0), 1.0, naxis=(64, 64))
    for key in ("NAXIS1", "NAXIS2"):
        del h[key]
    with pytest.raises(ValueError, match="shape_out could not be inferred"):
        reproject_interp((np.zeros((64, 64), dtype=np.float32), WCS(h)), WCS(h))


def test_every_refusal_named_in_the_milestone_is_reachable():
    """cpdis1, LONPOLE != 180 and a PV card on TAN, each through
    ``build_transform`` -- these are the refusals 0.2.0 did not have.
    """
    build_transform.cache_clear()
    from astropy.wcs import DistortionLookupTable

    # A lookup table attached to the WCS itself (the programmatic path -- a
    # header card alone cannot be assembled without the distortion HDU).
    with_lut = WCS(_tan())
    with_lut.cpdis1 = DistortionLookupTable(
        np.zeros((4, 4), dtype=np.float32), [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]
    )
    assert with_lut.cpdis1 is not None
    with pytest.raises(ValueError, match="cpdis1"):
        build_transform(with_lut, with_lut)

    cases = [
        (WCS(_tan(CRVAL2=90.0)), "LONPOLE"),
        (WCS(_tan(PV1_1=0.0)), "PV"),
    ]
    for wcs, pattern in cases:
        with pytest.raises(ValueError, match=pattern):
            build_transform(wcs, wcs)

    # CRVAL2 = -90 is still accepted: its wcslib default LONPOLE is 180.
    accepted = WCS(_tan(CRVAL2=-90.0))
    build_transform(accepted, accepted)


def test_validation_runs_exactly_twice_per_call(monkeypatch):
    """Validation is unconditional and not duplicated: exactly one
    ``from_astropy`` and one ``frame_descriptor`` per WCS, twice per
    ``build_transform``, on both a miss and a warm hit.
    """
    from reprojax import _astropy_wcs

    build_transform.cache_clear()
    calls = {"from_astropy": 0, "frame_descriptor": 0}
    real_from_astropy = _astropy_wcs.from_astropy
    real_descriptor = _astropy_wcs.frame_descriptor

    def counting_from_astropy(*args, **kwargs):
        calls["from_astropy"] += 1
        return real_from_astropy(*args, **kwargs)

    def counting_descriptor(*args, **kwargs):
        calls["frame_descriptor"] += 1
        return real_descriptor(*args, **kwargs)

    monkeypatch.setattr(_astropy_wcs, "from_astropy", counting_from_astropy)
    monkeypatch.setattr(_astropy_wcs, "frame_descriptor", counting_descriptor)

    h = make_header((32.0, 32.0), (180.0, 30.0), 1.0, naxis=(64, 64))
    build_transform(WCS(h), WCS(h))  # miss
    assert calls == {"from_astropy": 2, "frame_descriptor": 2}

    build_transform(WCS(h), WCS(h))  # warm hit: still validates both
    assert calls == {"from_astropy": 4, "frame_descriptor": 4}


def test_preserve_units_is_honoured():
    """``preserve_units=True`` used to be read as if arcsec were degrees.

    This is a **cross-encoding** check, not a self round trip: two encodings
    of the same sky (``CUNIT='arcsec'`` and ``CUNIT='deg'``) must map pixel
    to pixel through ``build_transform`` with no error. A self round trip
    cannot see the bug -- both sides use the same kernel, so a wrong sky
    coordinate cancels out. On this fixture the pre-fix kernel left the
    header's raw numbers in place, putting the reference pixel 21.85 deg from
    the origin against astropy's 0.0061 deg: a 22.26 deg / 80,145 px
    disagreement between the two encodings at 1"/px. The leaves are converted
    per axis, so the two now agree to 4e-10 px, and to the astropy oracle.
    """
    base = dict(
        NAXIS=2,
        NAXIS1=64,
        NAXIS2=64,
        CRPIX1=32.0,
        CRPIX2=32.0,
        CRVAL1=10.0,
        CRVAL2=20.0,
        CDELT1=-1.0,
        CDELT2=1.0,
        CUNIT1="arcsec",
        CUNIT2="arcsec",
        CTYPE1="RA---TAN",
        CTYPE2="DEC--TAN",
    )
    preserved = WCS(base, preserve_units=True)
    degrees = WCS(
        {
            **base,
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CRVAL1": 10.0 / 3600.0,
            "CRVAL2": 20.0 / 3600.0,
            "CDELT1": -1.0 / 3600.0,
            "CDELT2": 1.0 / 3600.0,
        }
    )

    # Output pixels are read in ``degrees``; the same sky must land on the
    # same pixel when the input WCS is the arcsec encoding, so a correct
    # transform is the identity on the overlapping grid.
    build_transform.cache_clear()
    transform = build_transform(degrees, preserved)
    x = np.array([0.0, 15.75, 31.5, 47.25, 63.0])
    y = np.array([63.0, 47.25, 31.5, 15.75, 0.0])
    x_in, y_in = (np.asarray(part) for part in transform(x, y))
    assert np.isfinite(x_in).all() and np.isfinite(y_in).all()
    np.testing.assert_allclose(x_in, x, rtol=0, atol=PIX_TOL)
    np.testing.assert_allclose(y_in, y, rtol=0, atol=PIX_TOL)

    # And against astropy, which is independent of both kernels. ``preserved``
    # reports its world coordinates in arcsec, so scale before the inverse.
    world = degrees.all_pix2world(np.column_stack([x, y]), 0)
    x_ref, y_ref = preserved.all_world2pix(world * 3600.0, 0).T
    np.testing.assert_allclose(x_in, x_ref, rtol=0, atol=PIX_TOL)
    np.testing.assert_allclose(y_in, y_ref, rtol=0, atol=PIX_TOL)

    # The leaves carry the conversion too -- crval as well as cd, so a mutant
    # that scales the matrix but not the reference point is caught here.
    # ``_mapprojax_wcs`` already returns both leaves in degrees.
    from reprojax.transform import _mapprojax_wcs

    a = _mapprojax_wcs(preserved)
    b = _mapprojax_wcs(degrees)
    np.testing.assert_allclose(np.asarray(a.cd), np.asarray(b.cd), rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.asarray(a.crval), np.asarray(b.crval), rtol=1e-12, atol=0)


def test_pc_cdelt_and_cd_headers_share_one_transform():
    build_transform.cache_clear()
    cd_form = make_header((32.0, 32.0), (180.0, 30.0), 1.0, naxis=(64, 64))
    pc_form = cd_form.copy()
    for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
        del pc_form[key]
    pc_form["CDELT1"], pc_form["CDELT2"] = -1.0 / 3600.0, 1.0 / 3600.0
    pc_form["PC1_1"], pc_form["PC2_2"] = 1.0, 1.0

    first = build_transform(WCS(cd_form), WCS(cd_form))
    second = build_transform(WCS(pc_form), WCS(pc_form))
    assert first is second
    assert len(_TRANSFORM_CACHE) == 1


def test_reparsed_header_hits_the_cache():
    build_transform.cache_clear()
    h = make_header((32.0, 32.0), (180.0, 30.0), 1.0, naxis=(64, 64))
    first = build_transform(WCS(h), WCS(h))
    second = build_transform(WCS(fits.Header(h)), WCS(fits.Header(h)))
    assert first is second


def test_leaf_key_handles_contiguity_and_signed_zero():
    from reprojax.transform import _leaf_key

    base = np.array([[-1.0 / 3600, 0.0], [0.0, 1.0 / 3600]])
    neg_zero = base.copy()
    neg_zero[1, 0] = -0.0
    non_contiguous = np.asfortranarray(base)

    assert _leaf_key(base) == _leaf_key(neg_zero)
    assert _leaf_key(base) == _leaf_key(non_contiguous)
    # ... and the normalisation never mutated the caller's array.
    assert np.signbit(neg_zero[1, 0])


def test_validated_key_distinguishes_frame_descriptors():
    from reprojax._astropy_wcs import FrameDescriptor, ValidatedWcs
    from reprojax.transform import _mapprojax_wcs, _validated_key

    mwcs = _mapprojax_wcs(WCS(make_header((32.0, 32.0), (180.0, 30.0), 1.0, naxis=(64, 64))))
    ra_dec = ("RA--", "DEC-")

    def key(kind, equinox, axis_pair):
        descriptor = FrameDescriptor(kind, equinox, axis_pair, None)
        return _validated_key(ValidatedWcs(mwcs, descriptor), "f64")

    assert key("icrs", None, ra_dec) != key("fk5", 2000.0, ra_dec)
    assert key("fk5", 2000.0, ra_dec) != key("fk5", 1950.0, ra_dec)
    assert key("galactic", None, ra_dec) != key("galactic", None, ("GLON", "GLAT"))
    # matrix_precision is part of the key
    icrs = FrameDescriptor("icrs", None, ra_dec, None)
    assert _validated_key(ValidatedWcs(mwcs, icrs), "f64") != _validated_key(ValidatedWcs(mwcs, icrs), "f32")

    # SIP inverse_steps is part of the key: a different Newton budget is a
    # different kernel, so the entries must not collide.
    from mapprojax.sip import Sip
    from mapprojax.wcs import Tan

    leaves = mwcs.to_dict()

    def sip_key(steps):
        sip = Sip(a=np.full((3, 3), 1e-6), b=np.full((3, 3), 1e-6), inverse_steps=steps)
        kernel = Tan(
            leaves["crpix"], leaves["cd"], leaves["crval"], sip=sip, precision=leaves["precision"]
        )
        return _validated_key(ValidatedWcs(kernel, icrs), "f64")

    assert sip_key(3) != sip_key(5)
