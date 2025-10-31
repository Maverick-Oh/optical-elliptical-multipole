from datetime import datetime
import requests, xml.etree.ElementTree as ET
from astropy.io import fits
from io import BytesIO
import os

datetime_string = str(datetime.now()).replace(' ', '_').replace(':', '')
datetime_string = datetime_string[:datetime_string.find('.')]

def new_session():
    s = requests.Session()
    s.trust_env = False                 # ignore HTTP(S)_PROXY, etc.
    s.headers.update({
        "User-Agent": "python-requests/cosmos-cutout",
        "Connection": "close",          # avoid keep-alive reuse issues
        "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
    })
    return s

def fetch_cutout(params):
    s = new_session()
    url = "https://irsa.ipac.caltech.edu/cgi-bin/Cutouts/nph-cutouts"
    r = s.get(url, params=params, timeout=60, allow_redirects=True, verify=True)
    r.raise_for_status()

    # if a proxy or gateway returned a non-http body, r.raise_for_status() may not run;
    # add a sanity check on the first chars
    txt = r.text.lstrip()
    if not (txt.startswith("<?xml") or txt.startswith("<result")):
        # Print a snippet to debug what we actually got
        raise RuntimeError(f"Unexpected response (not XML): {txt[:200]}")

    root = ET.fromstring(txt)
    status = root.attrib.get("status", "ok")
    if status != "ok":
        raise RuntimeError(root.findtext("message", default="IRSA Cutouts error"))

    # collect FITS URLs, stripping whitespace
    fits_urls = [el.text.strip() for el in root.findall(".//images/cutouts/fits") if el.text and el.text.strip()]
    if not fits_urls:
        urls = [u.text.strip() for u in root.findall(".//url") if u.text and u.text.strip()]
        fits_urls = [u for u in urls if u.lower().endswith((".fits", ".fits.gz"))]
    if not fits_urls:
        html_url = root.findtext(".//summary/resultHtml")
        raise RuntimeError(f"No FITS URLs found. Open to inspect:\n{html_url}")

    # fetch FITS (fresh session to avoid reusing a flaky TCP connection)
    s2 = new_session()
    fr = s2.get(fits_urls[0], timeout=120, allow_redirects=True, verify=True)
    fr.raise_for_status()
    return fits.open(BytesIO(fr.content))

COSMOS_MIN = 1 # arcsec
COSMOS_MAX = 180 # arcsec

def param_generator(tab_row, cutout_factor_r50=3., pixel_width=0.03, verbose=False):
    # pixel_width: 0.03 arcsec/px for ACS, according to: https://irsa.ipac.caltech.edu/applications/Cutouts/docs/CutoutsProgramInterface.html#example
    assert type(tab_row) == astropy.table.row.Row # double check that this is one table item (not the entire table of
    # many targets)
    # print(tab_row)
    ra  = float(tab_row['ra'])
    dec = float(tab_row['dec'])
    # r50_px = float(tab_row['r50']) # Pixels ZEST semi-major axis length of ellipse encompassing 50% of total light
    # if type(tab_row['r_gim2d']) == np.ma.core.MaskedConstant:
    r50_px = float(tab_row['r50'])
    r50_arcsec = r50_px*0.03
    # else:
    #     r50_arcsec = float(tab_row['r_gim2d'])
    cutout_size = cutout_factor_r50 * r50_arcsec
    cutout_size = max(COSMOS_MIN, min(cutout_size, COSMOS_MAX))
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
        "min_size": str(int(COSMOS_MIN)), # The minimum allowed cutout size for COSMOS data in arcseconds.
        "max_size": str(int(COSMOS_MAX)), # The maximum allowed cutout size for COSMOS data in arcseconds.
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

adql = """SELECT TOP 1 * FROM cosmos_morph_zurich_1"""

tab = svc.run_sync(adql).to_table()  # Astropy Table
colnames = tab.colnames
print("Column names:")
print(colnames)
# Check column information here:
# https://irsa.ipac.caltech.edu/data/COSMOS/gator_docs/cosmos_morph_zurich_colDescriptions.html

# For Cutouts Program and information, Check:
# https://irsa.ipac.caltech.edu/applications/Cutouts/docs/CutoutsProgramInterface.html

print("DONE!")

#%%
# TYPE=1 are early-type (E/S0) in ZEST; STELLARITY=0 excludes stars
# R50 is in pixels (ACS scale = 0.03 arcsec/pix per catalog docs)
# ELL_GIM2D = 1 - (b/a); we also return b/a explicitly.
adql = """SELECT """ + \
""" TOP 10 """ +\
""" sequentialid, CAPAK_ID, ra, dec, type, 
ACS_MU_CLASS, R50, ACS_X_IMAGE, ACS_Y_IMAGE,
ACS_A_IMAGE, ACS_B_IMAGE, ACS_THETA_IMAGE, 
R_GIM2D, ell_gim2d, PA_GIM2D, SERSIC_N_GIM2D
FROM cosmos_morph_zurich_1
WHERE stellarity=0 AND type=1 AND ACS_MU_CLASS=1 ORDER BY R50 DESC
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

data_dir = '../data'
hdul_dir = os.path.join(data_dir, f'HDUL_{datetime_string}')
os.makedirs(hdul_dir, exist_ok=True)

tab = svc.run_sync(adql).to_table()  # Astropy Table
tab.write(os.path.join(hdul_dir, f"cosmos_sample_N={len(tab)}_{datetime_string}.csv"), format="csv", overwrite=True)
# SAVE ADQL (archiving purpose)
with open(os.path.join(hdul_dir, f"ADQL_Query_{datetime_string}.sql"), "w") as file:
    file.write(adql)

import time
t0 = time.perf_counter()

plot = False
for i in range(len(tab)):
    if i==1 or (i>1 and i%5==0):
        done = i + 1
        elapsed = time.perf_counter() - t0
        # items/sec (avoid div by zero)
        rate = done / elapsed if elapsed > 0 else float('inf')
        rem = len(tab) - done
        eta_sec = rem / rate if np.isfinite(rate) and rate > 0 else float('nan')
        if np.isfinite(eta_sec):
            m, s = divmod(int(round(eta_sec)), 60)
            h, m = divmod(m, 60)
            eta_str = f"{h:02d}:{m:02d}:{s:02d}"
        else:
            eta_str = "--:--:--"
        msg = f"\rProcessing: [{done:>5}/{len(tab):<5}]  ETA: {eta_str}"
        print(msg, end='', flush=True)
    param = param_generator(tab[i])
    hdul = fetch_cutout(param)
    if len(hdul) > 1:
        raise ValueError("len(hdul)>1")
    seq_id = int(tab[i]['sequentialid'])
    # Saving
    if plot:
        im = hdul[0].data
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(im, origin="lower")
        ax.set_aspect("equal")
        fig.savefig(os.path.join(hdul_dir, f"{seq_id}.pdf"))
    plt.show()
    hdul.writeto(os.path.join(hdul_dir, f"{seq_id}.fits"), overwrite=True)
print("Done!")