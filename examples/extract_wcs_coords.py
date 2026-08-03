# Provenance script, kept for documentation rather than to be re-run here:
# this is how wcs_test_data.json (the real SPHEREx WCS/SIP header + a few
# reference pixel<->sky points) was extracted from the original, full
# multi-extension SPHEREx L2b product. It expects that full original file
# (IMAGE/FLAGS/VARIANCE/ZODI/PSF/WCS-WAVE extensions), not the trimmed
# single-HDU fixture at tests/data/spherex_l2b_v20.fits used by the test
# suite (see tests/test_real_spherex.py).

import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def serialize_header_value(val):
    if isinstance(val, (int, float, str, bool)):
        return val
    # Handle astropy Header objects or undefined cards if necessary, though simpler is better
    return str(val)

def main():
    filename = "reproj_test/level2_2025W23_1C_0165_1D4_spx_l2b-v20-2025-251.fits"
    output_filename = "reproj_test/wcs_test_data.json"

    try:
        # Open the FITS file
        with fits.open(filename) as hdul:
            # Assume WCS is in the primary header or the first extension with data
            # Usually primary or first extension. Let's try header of extension 1 if primary is empty, or just primary.
            # We'll check the first extension that has a valid WCS.
            
            header = None
            wcs = None
            
            # Try finding a valid WCS
            for hdu in hdul:
                try:
                    w = WCS(hdu.header)
                    if w.wcs.ctype[0]: # Check if valid CTYPE exists
                        header = hdu.header
                        wcs = w
                        break
                except Exception:
                    continue
            
            if wcs is None:
                # Fallback to primary
                header = hdul[0].header
                wcs = WCS(header)

            print(f"Using header from extension with CTYPE: {wcs.wcs.ctype}")

            naxis1 = header.get('NAXIS1')
            naxis2 = header.get('NAXIS2')
            
            if naxis1 is None or naxis2 is None:
                # Try to infer from shape if available
                if hdul[0].data is not None:
                     naxis2, naxis1 = hdul[0].data.shape
                elif len(hdul) > 1 and hdul[1].data is not None:
                     naxis2, naxis1 = hdul[1].data.shape
                else:
                    # Fallback defaults if NAXIS not present (unlikely for valid FITS)
                    naxis1 = 100
                    naxis2 = 100
                    print("Warning: NAXIS not found, using default 100x100")

            # Define test points (0-based indices)
            # Center, BL, BR, TL, TR
            points_pix = np.array([
                [(naxis1 - 1) / 2.0, (naxis2 - 1) / 2.0], # Center
                [0, 0],                                   # Bottom-Left
                [naxis1 - 1, 0],                          # Bottom-Right
                [0, naxis2 - 1],                          # Top-Left
                [naxis1 - 1, naxis2 - 1]                  # Top-Right
            ])
            
            # Pixel to World (all_pix2world handles SIP and distortions if present)
            world_coords = wcs.all_pix2world(points_pix, 0)
            
            # Calculate intermediate focal plane coordinates (SIP corrected pixels)
            if wcs.sip is not None:
                focal_coords = wcs.sip_pix2foc(points_pix, 0)
            else:
                focal_coords = points_pix

            # Collect relevant header keywords
            # We filter for standard WCS keywords and SIP coefficients
            wcs_keywords = {}
            # Standard prefixes for celestial WCS and SIP distortion
            wcs_prefixes = ['CTYPE', 'CRVAL', 'CRPIX', 'CD', 'PC', 'PV', 'LONPOLE', 'LATPOLE', 'RADESYS', 'EQUINOX', 'MJD-OBS', 'DATE-OBS']
            sip_prefixes = ['A_', 'B_', 'AP_', 'BP_']
            
            for card in header.cards:
                key = card.keyword
                # Filter for core keywords and exclude auxiliary/tabular systems (often ending in letters like W or A)
                is_wcs = any(key.startswith(pre) for pre in wcs_prefixes)
                # For SIP, we need both coefficients (A_1_1) and orders (A_ORDER)
                is_sip = any(key.startswith(pre) for pre in sip_prefixes)
                is_naxis = key in ['NAXIS1', 'NAXIS2']
                
                # Avoid keywords like CTYPE1W or CRVAL1A which can cause reconstruction errors
                # But keep PC1_1, PV1_1, etc.
                is_auxiliary = len(key) > 5 and key[-1].isalpha() and not (
                    key.startswith(('PC', 'PV', 'AP_', 'BP_')) or key.endswith('ORDER')
                )
                
                if (is_wcs or is_sip or is_naxis) and not is_auxiliary:
                    wcs_keywords[key] = serialize_header_value(card.value)

            # Structure the output
            output_data = {
                "metadata": {
                    "pixel_convention": "0-based (Python/C convention)",
                    "header_convention": "FITS (1-based for CRPIX)",
                    "focal_plane_convention": "Relative to FITS CRPIX (u = pix_0 - CRPIX_fits)",
                    "notes": (
                        "input_pixel_coordinates are 0-based. "
                        "header_keywords follow FITS standards (CRPIX is 1-based). "
                        "focal_plane_coordinates are intermediate coordinates (U, V) where U = u + f(u,v) "
                        "and u = x_0 - CRPIX_fits. This follows the Astropy/Shupe convention."
                    )
                },
                "key_descriptions": {
                    "header_keywords": "Relevant FITS WCS and SIP keywords from the source image.",
                    "input_pixel_coordinates": "List of [x, y] coordinates (0-based) used for testing.",
                    "focal_plane_coordinates": "Intermediate [U, V] coordinates after applying forward SIP distortion.",
                    "output_world_coordinates": "Final [RA, Dec] in degrees (ICRS) from the projection.",
                    "coordinate_system": "The celestial coordinate system (e.g., ICRS).",
                    "projection": "The FITS projection code (e.g., TAN, SIN)."
                },
                "header_keywords": wcs_keywords,
                "input_pixel_coordinates": points_pix.tolist(),
                "focal_plane_coordinates": focal_coords.tolist(),
                "output_world_coordinates": world_coords.tolist(),
                "coordinate_system": wcs.wcs.radesys,
                "projection": wcs.wcs.ctype[0][-3:] if len(wcs.wcs.ctype[0]) > 3 else "Unknown"
            }

            # Write to JSON
            with open(output_filename, 'w') as f:
                json.dump(output_data, f, indent=4)
                
            print(f"Successfully wrote {output_filename}")
            print("Sample World Coords (Center):", world_coords[0])

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
