"""Tier-1 validation: the celestial frame layer (Stage 5).

Two things are checked throughout:
  - the rotation between two frames is a proper rigid rotation, and it agrees
    with an independent SkyCoord round-trip at probe directions the basis
    construction never used;
  - the frame/axis classification accepts exactly the pairs reprojax can
    rotate and refuses the rest, with the CTYPE pair checked alongside the
    resolved frame (astropy resolves ELON/ELAT to ICRS, and resolves by
    ``wcs.wcs.lng``, which accepts swapped axes).

Module-level imports are restricted to symbols that exist before the Stage 5
migration, so the pre-fix falsification runs fail on assertions.
"""

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from reprojax._astropy_wcs import frame_rotation_matrix
from reprojax.transform import build_transform

from _fixtures import make_header

PIX_TOL = 1e-4
_ATOL = 1e-12


def _wcs(ctype=("RA---TAN", "DEC--TAN"), crpix=(256.0, 256.0), crval=(180.0, 30.0), **extra):
    h = make_header(crpix, crval, 1.0, ctype=ctype, naxis=(512, 512))
    for key, value in extra.items():
        h[key] = value
    return WCS(h)


def _composed_error(wcs_in, wcs_out, seed=7, n=200):
    """Composed pixel error against an astropy round-trip, in pixels."""
    transform = build_transform(wcs_in, wcs_out)
    rng = np.random.default_rng(seed)
    x_out = rng.uniform(0.0, 511.0, size=n)
    y_out = rng.uniform(0.0, 511.0, size=n)
    x_in, y_in = (np.asarray(part) for part in transform(x_out, y_out))

    from astropy.wcs.utils import wcs_to_celestial_frame

    sky = SkyCoord(
        *wcs_out.all_pix2world(x_out, y_out, 0), unit="deg", frame=wcs_to_celestial_frame(wcs_out)
    ).transform_to(wcs_to_celestial_frame(wcs_in))
    x_ref, y_ref = wcs_in.all_world2pix(sky.spherical.lon.deg, sky.spherical.lat.deg, 0)
    return np.hypot(x_in - x_ref, y_in - y_ref)


# ---------------------------------------------------------------------------
# The rotation itself.
# ---------------------------------------------------------------------------


def test_icrs_pair_is_exact_identity():
    wcs_a = _wcs()
    wcs_b = _wcs(crpix=(300.0, 240.0), crval=(180.1, 30.2))
    rotation = frame_rotation_matrix(wcs_a, wcs_b)
    np.testing.assert_allclose(rotation, np.eye(3), rtol=0, atol=0)


def test_icrs_to_fk5_j2000_already_rotates():
    """Retained 0.2.0 behaviour, now pinned: 'icrs' != 'fk5' made the old
    ``name`` shortcut take the basis path, so this pair always rotated.
    """
    rotation = frame_rotation_matrix(_wcs(), _wcs(RADESYS="FK5", EQUINOX=2000))
    # USNO Circular 179 frame bias: off-diagonal entries are ~1.1e-7.
    assert np.abs(rotation - np.eye(3)).max() == pytest.approx(1.11e-7, rel=0.2)


def test_fk5_j1950_vs_j2000_rotation_is_not_identity():
    """The actual defect: both resolve to name 'fk5', so the old shortcut
    returned the identity and dropped a 50-year precession.
    """
    rotation = frame_rotation_matrix(
        _wcs(RADESYS="FK5", EQUINOX=2000), _wcs(RADESYS="FK5", EQUINOX=1950)
    )
    assert not np.allclose(rotation, np.eye(3), atol=_ATOL)
    # A 50-year precession: the diagonals stay near 1, the off-diagonals are ~1e-2.
    assert np.abs(rotation - np.eye(3)).max() == pytest.approx(1.1e-2, rel=0.2)


def test_fk5_pair_rotation_is_orthonormal_and_composed_error_matches_skycoord():
    wcs_in = _wcs(RADESYS="FK5", EQUINOX=2000)
    wcs_out = _wcs(RADESYS="FK5", EQUINOX=1950)
    rotation = frame_rotation_matrix(wcs_out, wcs_in)

    assert np.isfinite(rotation).all()
    assert np.abs(rotation @ rotation.T - np.eye(3)).max() < _ATOL
    assert abs(np.linalg.det(rotation) - 1.0) < _ATOL

    error = _composed_error(wcs_in, wcs_out)
    assert np.isfinite(error).all()
    assert np.nanmax(error) < PIX_TOL


def test_rotation_probe_directions_agree_with_skycoord():
    """The three basis directions are the construction; these are not."""
    from astropy.coordinates import FK5, Galactic

    rotation = frame_rotation_matrix(_wcs(RADESYS="FK5", EQUINOX=2000), _wcs(RADESYS="FK5", EQUINOX=1950))
    from astropy.wcs.utils import wcs_to_celestial_frame

    frame_from = wcs_to_celestial_frame(_wcs(RADESYS="FK5", EQUINOX=2000))
    frame_to = wcs_to_celestial_frame(_wcs(RADESYS="FK5", EQUINOX=1950))
    assert isinstance(frame_from, FK5) and isinstance(frame_to, FK5)

    probe = SkyCoord([37.0, 10.0, 250.0, 0.0, 359.0], [-12.0, 89.0, -89.0, 0.0, 45.0], unit="deg", frame=frame_from)
    expected = probe.transform_to(frame_to).cartesian
    got = rotation @ np.stack([probe.cartesian.x.value, probe.cartesian.y.value, probe.cartesian.z.value])

    np.testing.assert_allclose(got, [expected.x.value, expected.y.value, expected.z.value], rtol=0, atol=_ATOL)


# ---------------------------------------------------------------------------
# The allow-list.
# ---------------------------------------------------------------------------


def test_fk5_without_equinox_card_is_accepted_at_j2000():
    """WCSLIB defaults RADESYS='FK5' with no EQUINOX to J2000, so the
    descriptor must read the resolved frame, not a header card.
    """
    from reprojax._astropy_wcs import frame_descriptor

    descriptor = frame_descriptor(_wcs(RADESYS="FK5"))
    assert (descriptor.kind, descriptor.equinox) == ("fk5", 2000.0)


def test_fk4_and_fk4_no_e_pairs_are_refused():
    from reprojax._astropy_wcs import frame_descriptor

    for radesys in ("FK4", "FK4-NO-E"):
        with pytest.raises(ValueError, match="FK4"):
            frame_descriptor(_wcs(RADESYS=radesys, EQUINOX=1950))


def test_ecliptic_ctypes_are_refused_despite_icrs_frame():
    from reprojax._astropy_wcs import frame_descriptor

    with pytest.raises(ValueError, match="ELON"):
        frame_descriptor(_wcs(ctype=("ELON-TAN", "ELAT-TAN")))


def test_axis_pair_must_match_the_resolved_frame():
    from reprojax._astropy_wcs import frame_descriptor

    # Programmatic: a RADESYS card on a GLON/GLAT header is dropped at parse
    # time, so the mismatch has to be built after parsing.
    mismatched = _wcs(ctype=("GLON-TAN", "GLAT-TAN"))
    mismatched.wcs.radesys = "ICRS"
    with pytest.raises(ValueError, match="axis pair"):
        frame_descriptor(mismatched)

    # FK5 polar axes: resolves FK5, so the FK5 branch must sanity-check too.
    with pytest.raises(ValueError, match="ELON"):
        frame_descriptor(_wcs(ctype=("ELON-TAN", "ELAT-TAN"), RADESYS="FK5", EQUINOX=2000))

    # An ordinary Galactic header is accepted.
    assert frame_descriptor(_wcs(ctype=("GLON-TAN", "GLAT-TAN"))).kind == "galactic"


def test_descriptor_audit_covers_every_suite_fixture():
    """Every header form the suite builds, plus both real fixtures, must have
    an allow-listed descriptor — a future convention fails loudly here.
    """
    from pathlib import Path

    from reprojax._astropy_wcs import frame_descriptor

    forms = [
        make_header((256.0, 256.0), (180.0, 30.0), 1.0, naxis=(512, 512)),
        make_header((256.0, 256.0), (180.0, 30.0), 1.0, rot_deg=15.0, naxis=(512, 512)),
    ]
    pc_form = make_header((256.0, 256.0), (180.0, 30.0), 1.0, naxis=(512, 512))
    for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
        del pc_form[key]
    pc_form["CDELT1"], pc_form["CDELT2"] = -1.0 / 3600.0, 1.0 / 3600.0
    pc_form["PC1_1"], pc_form["PC2_2"] = -1.0, 1.0
    forms.append(pc_form)

    crota_form = make_header((256.0, 256.0), (180.0, 30.0), 1.0, naxis=(512, 512))
    for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
        del crota_form[key]
    crota_form["CDELT1"], crota_form["CDELT2"] = 1.0 / 3600.0, 1.0 / 3600.0
    crota_form["CROTA2"] = 30.0
    forms.append(crota_form)

    data_dir = Path(__file__).parent / "data"
    forms.append(fits.getheader(data_dir / "spherex_l2b_v20.fits"))
    forms.append(fits.Header.fromtextfile(data_dir / "galactic_template_header.fits"))

    for header in forms:
        kind = frame_descriptor(WCS(header)).kind
        assert kind in {"icrs", "fk5", "galactic"}


def test_alternate_wcs_version_is_accepted():
    from reprojax._astropy_wcs import frame_descriptor

    header = make_header((256.0, 256.0), (180.0, 30.0), 1.0, naxis=(512, 512))
    header["CTYPE1A"], header["CTYPE2A"] = "RA---SIN", "DEC--SIN"
    assert frame_descriptor(WCS(header, key="A")).kind == "icrs"
