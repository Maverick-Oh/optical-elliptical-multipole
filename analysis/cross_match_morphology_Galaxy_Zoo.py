import pyvo, os

adql = """SELECT 
sequentialid, CAPAK_ID, acs_ident, ra, dec, type, 
ACS_MU_CLASS, R50, ACS_X_IMAGE, ACS_Y_IMAGE,
ACS_A_IMAGE, ACS_B_IMAGE, ACS_THETA_IMAGE, 
R_GIM2D, ell_gim2d, PA_GIM2D, SERSIC_N_GIM2D
FROM cosmos_morph_zurich_1
WHERE stellarity=0 AND type=1 AND ACS_MU_CLASS=1
"""

# datetime_string = str(datetime.now()).replace(' ', '_').replace(':', '')
# datetime_string = datetime_string[:datetime_string.find('.')]
datetime_string = 'crossmatching_Galaxy_Zoo'

data_dir = '../data'
hdul_dir = os.path.join(data_dir, f'HDUL_{datetime_string}')
os.makedirs(hdul_dir, exist_ok=True)

svc = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
tab = svc.run_sync(adql).to_table()  # Astropy Table
tab.write(os.path.join(hdul_dir, f"cosmos_sample_N={len(tab)}_{datetime_string}.csv"), format="csv", overwrite=True)
# SAVE ADQL (archiving purpose)
with open(os.path.join(hdul_dir, f"ADQL_Query_{datetime_string}.sql"), "w") as file:
    file.write(adql)