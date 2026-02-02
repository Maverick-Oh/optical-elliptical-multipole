from pathlib import Path
import os

HERE = Path(__file__).resolve().parent # This file's location

from astroquery.vizier import Vizier
from astropy.table import join

# Always inspect column names first (VizieR column naming can differ from the paper)
Vizier.ROW_LIMIT = -1  # 5 for small preview, -1 for full

# --- hCOSMOS (Damjanov+ 2018), Table 2 ---
hc = Vizier.get_catalogs("J/ApJS/234/21/table2")[0]
print("hCOSMOS columns:", hc.colnames)
print(hc[:3])
filename = os.path.join(HERE.parent,'data/hCOSMOS.csv')
hc.write(filename, format='ascii.csv', overwrite=True)

# --- LEGA-C DR3 ---
legac = Vizier.get_catalogs("J/ApJS/256/44/legacdr3")[0]
print("LEGA-C DR3 columns:", legac.colnames)
print(legac[:3])
filename = os.path.join(HERE.parent, 'data/LEGA-C.csv')
legac.write(filename, format='ascii.csv', overwrite=True)

# ------------------------------------------------------------
# Find overlap between hCOSMOS and LEGA-C using [MMS2013] IDs
# (append this after your current code)
# ------------------------------------------------------------

import numpy as np
from astropy.table import Table

def _normalize_id(x):
    """Normalize IDs for robust matching (strip, drop leading zeros, unify case)."""
    if x is None:
        return None
    s = str(x).strip()
    if s in ("", "nan", "None", "--", "null"):
        return None
    s = s.upper()
    # If the ID is purely numeric, strip leading zeros
    if s.isdigit():
        s = str(int(s))
    return s

# 1) Identify the [MMS2013] column name in each table
# VizieR sometimes uses "MMS2013" or "[MMS2013]" or similar.
hc_id_col = '[MMS2013]'
lg_id_col = '[MMS2013]'

# 2) Build normalized ID arrays
hc_ids = np.array([_normalize_id(v) for v in hc[hc_id_col]], dtype=object)
lg_ids = np.array([_normalize_id(v) for v in legac[lg_id_col]], dtype=object)

# 3) Drop nulls
hc_mask = np.array([v is not None for v in hc_ids])
lg_mask = np.array([v is not None for v in lg_ids])

hc_ids_n = hc_ids[hc_mask]
lg_ids_n = lg_ids[lg_mask]

# 4) Compute overlap
hc_set = set(hc_ids_n.tolist())
lg_set = set(lg_ids_n.tolist())
overlap_ids = sorted(hc_set.intersection(lg_set))

print(f"Valid hCOSMOS [MMS2013] IDs: {len(hc_set)}")
print(f"Valid LEGA-C  [MMS2013] IDs: {len(lg_set)}")
print(f"Overlapping IDs: {len(overlap_ids)}")

# 5) Extract the overlapping rows from both catalogs
# Map ID -> row index (first occurrence). If you expect duplicates, we can expand this.
hc_index = {}
for i, v in zip(np.where(hc_mask)[0], hc_ids_n):
    hc_index.setdefault(v, i)

lg_index = {}
for i, v in zip(np.where(lg_mask)[0], lg_ids_n):
    lg_index.setdefault(v, i)

hc_rows = hc[[hc_index[i] for i in overlap_ids]]
lg_rows = legac[[lg_index[i] for i in overlap_ids]]

# 6) Create a merged overlap table with key physical columns (edit as desired)
# Pick some common/likely useful fields, but keep robust if a column is missing.
def _safe_cols(tab, wanted):
    return [c for c in wanted if c in tab.colnames]

hc_keep = _safe_cols(hc_rows, [hc_id_col, "RAJ2000", "DEJ2000", "z", "logM", "LogM", "sig", "sigma"])
lg_keep = _safe_cols(lg_rows, [lg_id_col, "RAJ2000", "DEJ2000", "z", "zspec", "logMvir", "Mvir", "sigma", "sig"])

hc_small = hc_rows[hc_keep]
lg_small = lg_rows[lg_keep]

# Rename ID columns to avoid collision, then join on the normalized overlap_ids list
hc_small = hc_small.copy()
lg_small = lg_small.copy()
hc_small.rename_column(hc_id_col, "MMS2013")
lg_small.rename_column(lg_id_col, "MMS2013")

overlap_merged = join(
    hc_small,
    lg_small,
    keys="MMS2013",
    join_type="inner",
    table_names=["hc", "legac"]
)
#Table.join(hc_small, lg_small, keys="MMS2013", join_type="inner", table_names=["hc", "legac"])

# 7) Save results
out_dir = os.path.join(HERE.parent, "data")
os.makedirs(out_dir, exist_ok=True)

overlap_ids_path = os.path.join(out_dir, "hCOSMOS_LEGAC_overlap_MMS2013_ids.txt")
with open(overlap_ids_path, "w") as f:
    f.write("\n".join(overlap_ids))

overlap_hc_path = os.path.join(out_dir, "hCOSMOS_overlap_MMS2013.csv")
overlap_lg_path = os.path.join(out_dir, "LEGAC_overlap_MMS2013.csv")
overlap_merged_path = os.path.join(out_dir, "hCOSMOS_LEGAC_overlap_merged.csv")

hc_rows.write(overlap_hc_path, format="ascii.csv", overwrite=True)
lg_rows.write(overlap_lg_path, format="ascii.csv", overwrite=True)
overlap_merged.write(overlap_merged_path, format="ascii.csv", overwrite=True)

print("Wrote:")
print(" -", overlap_ids_path)
print(" -", overlap_hc_path)
print(" -", overlap_lg_path)
print(" -", overlap_merged_path)

# 8) Quick peek
print(overlap_merged[:5])

# Plot
import matplotlib.pyplot as plt
stellar_mass = overlap_merged['LogM']
virial_mass = overlap_merged['logMvir']
plt.plot(stellar_mass, virial_mass, '.')
plt.xlabel('stellar mass')
plt.ylabel('virial mass')
plt.show()