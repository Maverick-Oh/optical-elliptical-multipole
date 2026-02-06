import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def compare_runs(dirs_dict, parameter='n_sersic'):
    """
    Compare parameter recovery across multiple runs.
    dirs_dict: {label: path_to_csv}
    """
    dfs = {}
    truth = None
    
    # Load Data
    for label, csv_path in dirs_dict.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            dfs[label] = df
            print(f"Loaded {label}: {len(df)} rows")
        else:
            print(f"Warning: {csv_path} not found.")
            
    if not dfs:
        print("No data loaded.")
        return

    # Load truth from the first valid directory
    first_valid_csv = list(dirs_dict.values())[0] # Try first
    for csv in dirs_dict.values():
         if os.path.exists(csv):
             first_valid_csv = csv
             break
             
    first_dir = os.path.dirname(first_valid_csv)
    truth_csv = os.path.join(first_dir, "simulation_truth.csv")
    
    if os.path.exists(truth_csv):
        truth_df = pd.read_csv(truth_csv)
        print("Loaded truth data.")
    else:
        print(f"Truth data not found at {truth_csv}!")
        return

    # Merge
    truth_df['sequentialid'] = truth_df['seqid'].astype(str)
    
    plt.figure(figsize=(10, 6))
    
    summary_stats = []

    for label, df in dfs.items():
        # Merge with truth
        df['sequentialid'] = df['sequentialid'].astype(str)
        merged = pd.merge(df, truth_df, on='sequentialid', suffixes=('', '_true'), how='inner')
        
        param_fit = f"{parameter}_best"
        param_true = parameter
        
        if param_fit not in merged.columns:
            print(f"Column {param_fit} not found in {label}")
            continue
            
        diff = merged[param_fit] - merged[param_true]
        
        mean_bias = np.mean(diff)
        std_bias = np.std(diff)
        rmse = np.sqrt(np.mean(diff**2))
        
        print(f"{label}: Bias={mean_bias:.4f}, Scatter={std_bias:.4f}, RMSE={rmse:.4f}")
        summary_stats.append({'label': label, 'bias': mean_bias, 'scatter': std_bias, 'rmse': rmse})

        # Plot
        x = merged[param_true]
        plt.scatter(x, diff, label=f"{label} (RMSE={rmse:.3f})", alpha=0.6, s=30)
        
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel(f"True {parameter}")
    plt.ylabel(f"Error ({parameter}_best - True)")
    plt.title(f"Recovery of {parameter} by Supersampling Factor")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_fn = f"comparison_bias_{parameter}.png"
    plt.savefig(out_fn)
    print(f"Saved {out_fn}")
    plt.close()

if __name__ == "__main__":
    # Define paths
    base_dir = "/Volumes/MavSSD_T5/Dropbox/Google_Drive/code/research_Anna/optical-elliptical-multipole/data"
    
    # Paths to CSVs (mock_varying_a_m3)
    files = {
        '1x': f"{base_dir}/comparison_1x/mock_varying_a_m3/fitting_results.csv",
        '4x': f"{base_dir}/comparison_4x/mock_varying_a_m3/fitting_results.csv",
        '9x': f"{base_dir}/mock_fitting-0203-1/mock_varying_a_m3/fitting_results.csv"
    }
    
    compare_runs(files, parameter='n_sersic')
    compare_runs(files, parameter='a_m3')
    compare_runs(files, parameter='q')
