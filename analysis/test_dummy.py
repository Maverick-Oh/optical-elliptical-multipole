import pandas as pd
import numpy as np
row_query = pd.Series({'sequentialid': 5, 'EXPTIME_SCI': 4056.0})
if row_query is not None and 'EXPTIME_SCI' in row_query:
    try:
        val = float(row_query['EXPTIME_SCI'])
        print(f"val: {val}")
        if np.isfinite(val):
            exptime = val
            print(f"exptime: {exptime}")
    except Exception as e:
        print(f"error: {e}")
