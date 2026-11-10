# Test data provenance

Two real-data fixtures back the integration tests in
`tests/test_real_spherex.py`. Neither is part of the installed package, and
neither is shipped in the sdist — the sdist carries the package only.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `spherex_l2b_v20.fits` | 16,666,560 | `36c9d328e3181745fcb95a26cbf26cccf4f99efd3d316cdcc2941adb856d6e12` |
| `galactic_template_header.fits` | 1,700 | `d061bb00190f131c1dad8e0518623d32d2933e124884e4cf9fc0d4a75c47f3c7` |

Verify them with:

```bash
cd tests/data && sha256sum -c <<'EOF'
36c9d328e3181745fcb95a26cbf26cccf4f99efd3d316cdcc2941adb856d6e12  spherex_l2b_v20.fits
d061bb00190f131c1dad8e0518623d32d2933e124884e4cf9fc0d4a75c47f3c7  galactic_template_header.fits
EOF
```

## `spherex_l2b_v20.fits`

The `IMAGE` extension of a real SPHEREx Level 2b spectrum-image product:

| Keyword | Value |
| --- | --- |
| `EXTNAME` | `IMAGE` |
| `NAXIS1`, `NAXIS2` | 2040, 2040 |
| `BITPIX` | -32 (float32) |
| `BUNIT` | `MJy / sr` |
| `CTYPE1`, `CTYPE2` | `RA---TAN-SIP`, `DEC--TAN-SIP` |
| `OBSID` | `2025W23_1C_0165_1` |
| `DATE-OBS` | `2025-06-02T22:24:40.376` |
| `DATE-END` | `2025-06-02T22:26:35.494` |
| `DETECTOR` | 4 |
| `SPS_TARGET` | 27386 |
| `CRVAL1`, `CRVAL2` | 306.774220556, 39.377324342 |
| SIP | order-3 `A`/`B` with fitted `AP`/`BP` inverse |

**Modification**: the original multi-extension product was trimmed to this
one extension. The `FLAGS`, `VARIANCE`, `ZODI`, `PSF` and `WCS-WAVE`
extensions were dropped; the `IMAGE` data and header here are otherwise
unchanged. The resulting file is bit-identical to the original `IMAGE`
extension's data array, which is why its checksum above is reproducible.

**Why the dropped `FLAGS` matters.** The accuracy tests report a handful of
pixels — 16 for interpolated, 7 for adaptive — that disagree with the
reference by up to ~0.01 and ~0.003 respectively. They all cluster on one
real data spike (around row 807–809, column 609–611) that `FLAGS` would
normally mark as bad. Without the mask, a sub-micropixel position difference
next to a local gradient of ~10⁴ per pixel turns into a large value
difference. This is a property of the trimmed fixture, not a reprojection
defect; the tests state the tolerance that covers it.

## `galactic_template_header.fits`

A header-only FITS text file (no pixel data — an output grid needs only the
header):

| Keyword | Value |
| --- | --- |
| `NAXIS1`, `NAXIS2` | 1201, 1201 |
| `CTYPE1`, `CTYPE2` | `GLON-TAN`, `GLAT-TAN` |
| `CRVAL1`, `CRVAL2` | 78.9293, 0.48367 |
| `LONPOLE` | 180.0 |
| `LATPOLE` | 0.48367 |
| `CRPIX1`, `CRPIX2` | 601.0, 601.0 |
| `CDELT1`, `CDELT2` | -0.0016666666666667, 0.0016666666666667 |
| `BITPIX`, `NAXIS` | -64, 2 (no pixel data; `EXTEND` = T) |
| `WCSAXES`, `MJDREF` | 2, 0.0 |

It is the Galactic-frame mosaicking template used for this exposure in the
original pre-reprojax workflow, so the tests run the same
ICRS TAN+SIP → Galactic TAN operation that workflow performed by hand.

**Provenance, separately from the exposure.** This file's origin is *not*
the same as `spherex_l2b_v20.fits`'s, and should not be described as one.

It is a **header-only extraction**: an IPAC-table-style text header
(`CDELT` + `CRVAL`, no `CD` matrix) taken from the 1201x1201 Galactic TAN
mosaic template used as the output grid in the pre-reprojax workflow. That
source template is a real mosaic with pixel data; none of it is here. The
20 cards above are exactly the source template's 20 cards, in order —
verified by card-for-card comparison.

What the header does **not** tell you: it carries no `OBSID`, no `DATE-OBS`,
no `ORIGIN`/`AUTHOR`, and no pipeline `HISTORY` trail, so the file itself
names neither its producer nor its licence.

**Terms.** It is project metadata from the author's own workflow, so it is
**not** SPHEREx mission data and SPD-41 does not apply to it. It is
distributed under reprojax's own BSD-3-Clause licence, confirmed by the
repository owner.

Because it is only WCS parameters — `CTYPE`, `CRVAL`, `CRPIX`, `CDELT` for
a Galactic TAN grid — it is trivially reproducible if the file is ever
replaced or its terms become a problem.

## Terms

SPHEREx is a NASA Astrophysics Medium Explorer mission, a joint project of
the Jet Propulsion Laboratory and the California Institute of Technology,
funded by NASA.

**Scope.** Only `spherex_l2b_v20.fits` is a SPHEREx mission product.
`galactic_template_header.fits` is project metadata from the author's own
workflow, not mission data; its provenance and terms are stated separately
above.

**Data rights (the exposure only).** NASA's Science Mission Directorate
policy, SPD-41, states that "Mission data are released into the public domain
and non-SMD-funded activities have no requirements with respect to the usage
of that data", and that "there shall be no period of exclusive access to
Mission data". `spherex_l2b_v20.fits` is therefore redistributed here as
public-domain mission data.

**Required acknowledgement** (from the IRSA SPHEREx Quick Release page), to be
included in published material that uses the exposure:

> This publication makes use of data products from the Spectro-Photometer for
the History of the Universe, Epoch of Reionization and Ices Explorer
> (SPHEREx), which is a joint project of the Jet Propulsion Laboratory and the
> California Institute of Technology, and is funded by the National
> Aeronautics and Space Administration.

**Citation.** IRSA asks users of SPHEREx Quick Release data to cite the
matching DOI. The exposure here was acquired 2025-06-02 (before QR2 superseded
QR1 in October 2025) and the file name records L2b processing version 2.0, so
it is either an original QR1 product or a QR2 reprocessing of the same
exposure. Confirm from your source before citing one; the candidates are
`10.26131/IRSA629` (QR1) and `10.26131/IRSA652` (QR2).

**Header provenance (the exposure only).** `spherex_l2b_v20.fits` carries
its original FITS header unchanged, including the pipeline `HISTORY` trail of
`SSDC-MS-*` calibration steps (`SSDC-MS-012` through `SSDC-MS-025`). The
template header has no such trail — see its provenance section above.

If you redistribute reprojax, keep these files' provenance with them. If the
upstream terms change, this file is the place to record the new ones.

## Updating a fixture

Replacing either file invalidates every tolerance quoted in
`tests/test_real_spherex.py` and the README's accuracy tables. Re-derive the
caps against the pinned reference (`reproject` 0.21.0) and update the
checksums here in the same commit.
