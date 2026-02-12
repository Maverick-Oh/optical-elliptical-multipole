
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys

def plot_recovery(results_path, out_dir):
    """
    Plot theta_ell recovery.
    """
    if not os.path.exists(results_path):
        print(f"File not found: {results_path}")
        return

    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} rows from {results_path}")
    
    # Filter valid fits if needed (though we want to see all)
    # df = df[df['chi2_reduced'] < 5] 
    
    plt.figure(figsize=(6, 6))
    
    # Check column names
    if 'theta_ell_best' not in df.columns or 'theta_ell_true' not in df.columns:
        print("Columns theta_ell_best or theta_ell_true not found.")
        print("Columns:", df.columns)
        return

    # Plot Best vs True
    plt.errorbar(df['theta_ell_true'], df['theta_ell_best'], 
                 yerr=df.get('theta_ell_err', None), 
                 fmt='o', color='blue', alpha=0.7, label='Recovered')
    
    # Identity line
    min_val = min(df['theta_ell_true'].min(), df['theta_ell_best'].min())
    max_val = max(df['theta_ell_true'].max(), df['theta_ell_best'].max())
    range_val = max_val - min_val
    plot_min = min_val - 0.1 * range_val
    plot_max = max_val + 0.1 * range_val
    
    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', alpha=0.5, label='1:1')
    
    plt.xlabel(r'$\theta_{\rm ell, True}$ (rad)')
    plt.ylabel(r'$\theta_{\rm ell, Best}$ (rad)')
    plt.title('Parameter Recovery: theta_ell (9x SS)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    out_name = os.path.join(out_dir, "recovery_theta_ell.pdf")
    plt.savefig(out_name)
    print(f"Saved {out_name}")
    
    # Also plot residuals
    plt.figure(figsize=(8, 4))
    resid = df['theta_ell_best'] - df['theta_ell_true']
    plt.scatter(df['theta_ell_true'], resid, color='red', alpha=0.7)
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel(r'$\theta_{\rm ell, True}$ (rad)')
    plt.ylabel(r'Residual ($\theta_{\rm rec} - \theta_{\rm true}$)')
    plt.title('Residuals: theta_ell')
    plt.grid(True, alpha=0.3)
    
    out_resid = os.path.join(out_dir, "residual_theta_ell.pdf")
    plt.savefig(out_resid)
    print(f"Saved {out_resid}")

if __name__ == "__main__":
    results_csv = "data/mock_fitting-theta_ell/mock_varying_theta_ell/fitting_results.csv"
    plot_recovery(results_csv, ".")
