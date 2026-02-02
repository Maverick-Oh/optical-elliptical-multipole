import requests
import xml.etree.ElementTree as ET
from astropy.io import fits
from io import BytesIO
import astropy
import numpy as np

def new_session():
    s = requests.Session()
    s.trust_env = False                 # ignore HTTP(S)_PROXY, etc.
    s.headers.update({
        "User-Agent": "python-requests/cosmos-cutout",
        "Connection": "close",          # avoid keep-alive reuse issues
        "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
    })
    return s

def fetch_cutout(params, debug=False):
    s = new_session()
    url = "https://irsa.ipac.caltech.edu/cgi-bin/Cutouts/nph-cutouts"
    r = s.get(url, params=params, timeout=60, allow_redirects=True, verify=True)
    r.raise_for_status()

    txt = r.text.lstrip()
    if not (txt.startswith("<?xml") or txt.startswith("<result")):
        raise RuntimeError(f"Unexpected response (not XML): {txt[:200]}")

    root = ET.fromstring(txt)
    status = root.attrib.get("status", "ok")
    if status != "ok":
        raise RuntimeError(root.findtext("message", default="IRSA Cutouts error"))

    # collect *all* FITS URLs
    fits_urls = [el.text.strip()
                 for el in root.findall(".//images/cutouts/fits")
                 if el.text and el.text.strip()]
    fits_urls.sort()
    # sort in case multiple _sci and _wht files exist; to list them in order

    # if not fits_urls:
    #     urls = [u.text.strip() for u in root.findall(".//url")
    #             if u.text and u.text.strip()]
    #     fits_urls = [u for u in urls if u.lower().endswith((".fits", ".fits.gz"))]
    # if not fits_urls:
    #     html_url = root.findtext(".//summary/resultHtml")
    #     raise RuntimeError(f"No FITS URLs found. Open to inspect:\n{html_url}")

    # Prefer matched SCI/WHT pair when present
    sci_url_list = []
    wht_url_list = []
    for url in fits_urls:
        if '_sci' in url:
            sci_url_list.append(url)
        elif '_wht' in url:
            wht_url_list.append(url)
        else:
            raise RuntimeError(f"Unexpected url: {url} (without '_sci' or '_wht')")

    s2 = new_session()
    out = {"SCI": [], "WHT": []}
    for sci_url in sci_url_list:
        fr = s2.get(sci_url, timeout=120, allow_redirects=True, verify=True)
        fr.raise_for_status()
        out["SCI"].append(fits.open(BytesIO(fr.content)))
    for wht_url in wht_url_list:
        fr = s2.get(wht_url, timeout=120, allow_redirects=True, verify=True)
        fr.raise_for_status()
        out["WHT"].append(fits.open(BytesIO(fr.content)))

    # Fallback: if no sci/wht markers, return the first file under "SCI"
    if not out and fits_urls:
        # fr = s2.get(fits_urls[0], timeout=120, allow_redirects=True, verify=True)
        # fr.raise_for_status()
        # out["SCI"] = fits.open(BytesIO(fr.content))
        raise ValueError()

    return out, sci_url_list, wht_url_list

# def fetch_cutout(params):
#     s = new_session()
#     url = "https://irsa.ipac.caltech.edu/cgi-bin/Cutouts/nph-cutouts"
#     r = s.get(url, params=params, timeout=60, allow_redirects=True, verify=True)
#     r.raise_for_status()
#
#     # if a proxy or gateway returned a non-http body, r.raise_for_status() may not run;
#     # add a sanity check on the first chars
#     txt = r.text.lstrip()
#     if not (txt.startswith("<?xml") or txt.startswith("<result")):
#         # Print a snippet to debug what we actually got
#         raise RuntimeError(f"Unexpected response (not XML): {txt[:200]}")
#
#     root = ET.fromstring(txt)
#     status = root.attrib.get("status", "ok")
#     if status != "ok":
#         raise RuntimeError(root.findtext("message", default="IRSA Cutouts error"))
#
#     # collect FITS URLs, stripping whitespace
#     fits_urls = [el.text.strip() for el in root.findall(".//images/cutouts/fits") if el.text and el.text.strip()]
#     if not fits_urls:
#         urls = [u.text.strip() for u in root.findall(".//url") if u.text and u.text.strip()]
#         fits_urls = [u for u in urls if u.lower().endswith((".fits", ".fits.gz"))]
#     if not fits_urls:
#         html_url = root.findtext(".//summary/resultHtml")
#         raise RuntimeError(f"No FITS URLs found. Open to inspect:\n{html_url}")
#
#     # fetch FITS (fresh session to avoid reusing a flaky TCP connection)
#     s2 = new_session()
#     fr = s2.get(fits_urls[0], timeout=120, allow_redirects=True, verify=True)
#     fr.raise_for_status()
#     return fits.open(BytesIO(fr.content))

def param_generator(tab_row, cutouttbl1="acs_mosaic_2.0", cutout_arcsec=None, verbose=True, **kwargs):
    COSMOS_MIN = 1  # arcsec
    COSMOS_MAX = 180  # arcsec
    # pixel_width: 0.03 arcsec/px for ACS, according to: https://irsa.ipac.caltech.edu/applications/Cutouts/docs/CutoutsProgramInterface.html#example
    assert type(tab_row) == astropy.table.row.Row # double check that this is one table item (not the entire table of
    # many targets)
    # print(tab_row)
    ra  = float(tab_row['ra'])
    dec = float(tab_row['dec'])
    # r50_px = float(tab_row['r50']) # Pixels ZEST semi-major axis length of ellipse encompassing 50% of total light
    # if type(tab_row['r_gim2d']) == np.ma.core.MaskedConstant:
    # r50_px = float(tab_row['r50'])
    # r50_arcsec = r50_px * pixel_width
    # else:
    #     r50_arcsec = float(tab_row['r_gim2d'])
    if verbose:
        print("cutout_arcsec:", cutout_arcsec)
    #
    params = {
        "mode": "PI", # Program Interface (PI) mode
        "mission": kwargs['mission'] if 'mission' in kwargs.keys() else "COSMOS",
        "locstr": f"{ra} {dec} eq", # This is the search location parameter, required for all searches. The input can be a coordinate or astronomical object name; if it is an object name, it is resolved into coordinates using NED and, if that fails, SIMBAD.
        "sizeX": f"{cutout_arcsec}",
        # The image cutouts box size on the sky (units of this size parameter are specified by the next parameter, called "units", which can be deg, arcmin or arcsec). The size can be any number larger than zero (interpreted as a double) and smaller than the size specified by the "max_size" parameter (note units may be different). Note, in most cases, the maximum allowed sizeX value is 2.0 degrees, but it varies for the different data collections. COSMOS program interface allows 1–180 arcsec
        "units": "arcsec",
        "min_size": str(int(COSMOS_MIN)), # The minimum allowed cutout size for COSMOS data in arcseconds.
        "max_size": str(int(COSMOS_MAX)), # The maximum allowed cutout size for COSMOS data in arcseconds.
        "ntable_cutouts": "1", # The number of metadata tables to search, for cutouts of COSMOS. Names of all N tables are listed below.
        "cutouttbl1": cutouttbl1,
    }
    return params

def cutout_selection_metric(target_xy_coords, sci_list, pixel_width, r50_arcsec,
                            include_inside_metric=True,
                            include_distance_metric=True,
                            include_ratio_metric=True,
                            include_zero_ratio_metric=True,):
    metric_dict = {}
    # criterion 0: the target xy coords must be within the image pixels
    inside = np.array([(t_x > 0) and (t_y > 0) and (t_y < sci_.data.shape[0]) and (t_x < sci_.data.shape[1])
                       for sci_, (t_x, t_y) in zip(sci_list, target_xy_coords)], dtype=bool)
    inside_metric = ~inside * 100.  # 0 if the target x and y pixel coords are inside, 100 if outside
    metric_dict['inside'] = inside; metric_dict['inside_metric'] = inside_metric
    # criterion 0.1: distance metric for cases where the target is at the edge for two or more cases (e.g. sid=102969)
    dist_to_bdry_px = np.array([min(t_x, t_y, sci_.data.shape[1] - t_x, sci_.data.shape[0] - t_y)
                                for sci_, (t_x, t_y) in zip(sci_list, target_xy_coords)])
    distance_normalized_w_r50 = (dist_to_bdry_px * pixel_width / (20 * r50_arcsec))  # bad if this is < 1
    distance_metric = np.exp(- distance_normalized_w_r50)  # ~0 if the boundary is far away
    metric_dict['dist_to_bdry_px'] = dist_to_bdry_px; metric_dict['distance_metric'] = distance_metric
    # criterion 1: ratio between the numbers of rows vs. columns (better if square)
    rowcol_array = np.array([sci_.data.shape for sci_ in sci_list])
    ratio_array = rowcol_array[:, 0] / rowcol_array[:, 1]  # the ratio of the cut's row and colums
    ratio_log_abs = np.abs(np.log(ratio_array))  # better if close to 0; it means the ratio is close to 1.
    ratio_metric = ratio_log_abs / np.log(1.2)  # divide by np.log(1.2)~0.18; if the ratio was 1., it will get 0; if the ratio was 1.2 (or 1/1.2), it will get 1. if the ratio was 1.44 (or 1/1.44), it will get 2 and so on; better if points is lower.
    metric_dict['ratio_array'] = ratio_array; metric_dict['ratio_metric'] = ratio_metric
    # criteria 2: ratio of pixels with value 0. vs. all pixels; better if it is 0 or close
    zero_ratio = np.array([np.sum(sci_.data == 0.) / np.prod(sci_.data.shape) for sci_ in sci_list])
    zero_ratio_metric = zero_ratio / 0.1  # if zero_ratio was 0., it will get 0; if zero_ratio was 10%, it will get 1, etc.
    metric_dict['zero_ratio'] = zero_ratio; metric_dict['zero_ratio_metric'] = zero_ratio_metric
    # combine all criteria
    full_metric = 0.
    if include_inside_metric:
        full_metric += inside_metric
    if include_distance_metric:
        full_metric += distance_metric
    if include_ratio_metric:
        full_metric += ratio_metric
    if include_zero_ratio_metric:
        full_metric += zero_ratio_metric
    metric_dict['full_metric'] = full_metric
    return full_metric, metric_dict