import pandas as pd
import glob
import os
import argparse
import numpy as np

def compare_runs(run_dirs):
    summaries = {}
    
    for d in run_dirs:
        name = os.path.basename(d)
        sum_file = os.path.join(d, "validation_summary.csv")
        
        if os.path.exists(sum_file):
            df = pd.read_csv(sum_file)
            summaries[name] = df.set_index('param')
        else:
            print(f"Warning: No validation summary found for {name}")

    if not summaries:
        print("No data to compare.")
        return

    # Assuming all have same params
    base_run = list(summaries.keys())[0] # First one as baseline? Or sort?
    # Sort: 0201-1 (1x), 0201-2 (4x), 0203-1 (9x)
    run_names = sorted(summaries.keys())
    
    params = summaries[run_names[0]].index.unique()
    
    print("\n--- Mean Chi2 Comparison ---")
    header = f"{'Param':<15} | " + " | ".join([f"{r:<20}" for r in run_names])
    print(header)
    print("-" * len(header))
    
    for p in params:
        row = f"{p:<15} | "
        for r in run_names:
            if p in summaries[r].index:
                val = summaries[r].loc[p, 'mean_chi2']
                row += f"{val:.4f}".center(20) + " | "
            else:
                row += "N/A".center(20) + " | "
        print(row)

    print("\n\n--- Mean Residual Comparison (Recovered - True) ---")
    print(header)
    print("-" * len(header))

    for p in params:
        row = f"{p:<15} | "
        for r in run_names:
            if p in summaries[r].index:
                val = summaries[r].loc[p, 'mean_residual']
                row += f"{val:.4g}".center(20) + " | "
            else:
                row += "N/A".center(20) + " | "
        print(row)
        
    # Analyze Amplitude/X0/Y0 specifically
    print("\n\n--- Key Problematic Parameters Improvement ---")
    for p in ['amplitude', 'x0', 'y0']:
        if p not in params: continue
        print(f"\nParameter: {p}")
        base_chi2 = summaries[run_names[0]].loc[p, 'mean_chi2'] if p in summaries[run_names[0]].index else None
        
        for r in run_names:
            if p in summaries[r].index:
                curr_chi2 = summaries[r].loc[p, 'mean_chi2']
                imp = 0.0
                if base_chi2 and base_chi2 > 0:
                    imp = (base_chi2 - curr_chi2) / base_chi2 * 100
                print(f"  {r}: Chi2 = {curr_chi2:.4f} (Improvement: {imp:+.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", help="Directories to compare")
    args = parser.parse_args()
    
    compare_runs(args.dirs)
