import requests, xml.etree.ElementTree as ET
from astropy.io import fits
from io import BytesIO

s = requests.Session()
s.headers.update({"User-Agent": "python-requests/cosmos-cutout"})

def get_hdul(params):
    # For Cutouts Program, Check: https://irsa.ipac.caltech.edu/applications/Cutouts/docs/CutoutsProgramInterface.html
    r = s.get("https://irsa.ipac.caltech.edu/cgi-bin/Cutouts/nph-cutouts", params=params, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    status = root.attrib.get("status", "ok")
    if status != "ok":
        raise RuntimeError(root.findtext("message", default="IRSA Cutouts error."))

    # Prefer explicit <fits> tags, but strip whitespace. Fallback: any <url> ending in .fits/.fits.gz
    fits_urls = [el.text.strip() for el in root.findall(".//images/cutouts/fits") if el.text and el.text.strip()]
    if not fits_urls:
        urls = [u.text.strip() for u in root.findall(".//url") if u.text and u.text.strip()]
        fits_urls = [u for u in urls if u.lower().endswith(".fits") or u.lower().endswith(".fits.gz")]
    if not fits_urls:
        html_url = root.findtext(".//summary/resultHtml")
        raise RuntimeError(f"No FITS URLs found. Open this to debug:\n{html_url}")

    # Fetch first FITS
    resp = s.get(fits_urls[0], timeout=120, allow_redirects=True)
    resp.raise_for_status()
    hdul = fits.open(BytesIO(resp.content))
    return hdul

def get_image(params):
    hdul = get_hdul(params)
    if len(hdul) == 1:
        return hdul[0].data
    else:
        raise ValueError(f"len(hdul)={len(hdul)}! Check what's going on.")

def param_generator(tab_row, cutout_factor_r50=3., pixel_width=0.03, verbose=True):
    # pixel_width: 0.03 arcsec/px for ACS, according to: https://irsa.ipac.caltech.edu/applications/Cutouts/docs/CutoutsProgramInterface.html#example
    assert type(tab_row) == astropy.table.row.Row # double check that this is one table item (not the entire table of
    # many targets)
    ra  = float(tab_row['ra'])
    dec = float(tab_row['dec'])
    r50_px = float(tab_row['r50']) # Pixels ZEST semi-major axis length of ellipse encompassing 50% of total light
    r50_arcsec = r50_px * pixel_width
    cutout_size = cutout_factor_r50 * r50_arcsec
    if verbose:
        print("cutout size:", cutout_size)
    #
    params = {
        "mode": "PI", # Program Interface (PI) mode
        "mission": "COSMOS", #
        "locstr": f"{ra} {dec} eq", # This is the search location parameter, required for all searches. The input can be a coordinate or astronomical object name; if it is an object name, it is resolved into coordinates using NED and, if that fails, SIMBAD.
        "sizeX": f"{cutout_size}",
        # The image cutouts box size on the sky (units of this size parameter are specified by the next parameter, called "units", which can be deg, arcmin or arcsec). The size can be any number larger than zero (interpreted as a double) and smaller than the size specified by the "max_size" parameter (note units may be different). Note, in most cases, the maximum allowed sizeX value is 2.0 degrees, but it varies for the different data collections. COSMOS program interface allows 1–180 arcsec
        "units": "arcsec",
        "min_size": "1", # The minimum allowed cutout size for COSMOS data in arcseconds.
        "max_size": "180", # The maximum allowed cutout size for COSMOS data in arcseconds.
        "ntable_cutouts": "1", # The number of metadata tables to search, for cutouts of COSMOS. Names of all N tables are listed below.
        "cutouttbl1": "acs_mosaic_2.0",
    }
    return params

#%%
# General reference for COSMOS:
# https://cosmos.astro.caltech.edu/page/astronomers
import pyvo
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import astropy

svc = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")

adql = """SELECT 
TOP 1 * 
FROM cosmos_morph_zurich_1"""

tab = svc.run_sync(adql).to_table()  # Astropy Table
colnames = tab.colnames
print("Column names:")
print(colnames)
# Check column information here:
# https://irsa.ipac.caltech.edu/data/COSMOS/gator_docs/cosmos_morph_zurich_colDescriptions.html

for i in range(len(tab)):
    param = param_generator(tab[i])
    im = get_image(param)
    plt.figure()
    plt.imshow(im, origin="lower")
    plt.show()
#    im_list.append(im)
# For Cutouts Program and information, Check:
# https://irsa.ipac.caltech.edu/applications/Cutouts/docs/CutoutsProgramInterface.html

print("DONE!")



#%%
# TYPE=1 are early-type (E/S0) in ZEST; STELLARITY=0 excludes stars
# R50 is in pixels (ACS scale = 0.03 arcsec/pix per catalog docs)
# ELL_GIM2D = 1 - (b/a); we also return b/a explicitly.
adql = """SELECT """ + \
""" TOP 10""" +\
"""
sequentialid, CAPAK_ID, ra, dec, type, ell_gim2d, (1.0 - ELL_GIM2D) AS q, ACS_MU_CLASS, R50, ACS_A_IMAGE as a, 
ACS_B_IMAGE as b
FROM cosmos_morph_zurich_1
WHERE stellarity=0 AND type=1 AND ACS_MU_CLASS=1
"""
## type: ZEST Type CLASS, 1 = Early type, 2 = Disk, 3 = Irregular Galaxy, 9 = no classification
# ACS_MU_CLASS: Type of object. 1 = galaxy, 2 = star, 3 = spurious
# STELLARITY: 0 if ACS_CLASS_STAR<0.6 (object is ASSUMED to be a galaxy; no visual inspection); 0 if ACS_CLASS_STAR>=0.6 AND object visually identified as a galaxy.
# ELL_GIM2D: GIM2D ellipticity = 1-b/a of object
# ACS_MU_CLASS: Type of object. 1 = galaxy, 2 = star, 3 = spurious
# ACS_A_IMAGE	float	 	SExtractor semi-major axis
# ACS_B_IMAGE	float	 	SExtractor semi-minor axis
# ACS_X_IMAGE	float	pixel	X-pixel position on ACS-tile
# ACS_Y_IMAGE	float	pixel	Y-pixel position on ACS-tile
tab = svc.run_sync(adql).to_table()  # Astropy Table
colnames = tab.colnames

for i in range(len(tab)):
    param = param_generator(tab[i])
    im = get_image(param)
    plt.figure()
    plt.imshow(im, origin="lower")
    plt.show()

# SELECT
#   ra, dec,
#   TYPE,                             -- ZEST class (1=E/S0)
#   R50,                              -- half-light semi-major axis (pixels)
#   R50 * 0.03 AS R50_arcsec,         -- convert to arcsec (0.03"/pix)
#   R_GIM2D,                          -- PSF-convolved half-light radius (arcsec)
#   ELL_GIM2D,                        -- ellipticity = 1 - b/a
#   (1.0 - ELL_GIM2D) AS b_over_a,
#   ACS_MAG_AUTO                      -- for optional quality cuts
# FROM cosmos_morph_zurich
# WHERE TYPE = 1        -- E/S0
#   AND

print("Done!")