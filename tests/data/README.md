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

It is the Galactic-frame mosaicking template used for this exposure in the
original pre-reprojax workflow, so the tests run the same
ICRS TAN+SIP → Galactic TAN operation that workflow performed by hand.

## Terms

Both files are redistributed here for testing only, as part of the SPHEREx
public data release. SPHEREx is a NASA Medium Explorer mission; its data are
public and not subject to copyright in the United States. The files carry
their original FITS headers, including the pipeline `HISTORY` trail
(`SSDC-MS-*` calibration steps).

If you redistribute reprojax, keep these files' provenance with them. If the
upstream terms change, this file is the place to record the new ones.

## Updating a fixture

Replacing either file invalidates every tolerance quoted in
`tests/test_real_spherex.py` and the README's accuracy tables. Re-derive the
caps against the pinned reference (`reproject` 0.21.0) and update the
checksums here in the same commit.
