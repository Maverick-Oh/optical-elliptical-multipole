import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Configuration
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

def validate_results(data_dir):
    print(f"Validating results in: {data_dir}")
    
    mock_dirs = glob.glob(os.path.join(data_dir, "mock_varying_*"))
    mock_dirs.sort()
    
    if not mock_dirs:
        print(f"No 'mock_varying_*' directories found in {data_dir}")
        return

    all_summary = []
    
    # Setup plots output directory
    PLOT_DIR = os.path.join(data_dir, "validation_figures")
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"Saving figures to: {PLOT_DIR}")
    
    for d in mock_dirs:
        param_name = os.path.basename(d).replace("mock_varying_", "")
        print(f"  Processing {param_name}...")
        
        truth_file = os.path.join(d, "simulation_truth.csv")
        fit_file = os.path.join(d, "fitting_results.csv")
        
        if not os.path.exists(truth_file) or not os.path.exists(fit_file):
            print(f"    Missing truth or fit files for {param_name}, skipping.")
            continue
            
        try:
            df_truth = pd.read_csv(truth_file)
            df_fit = pd.read_csv(fit_file)
        except Exception as e:
            print(f"    Error reading CSVs: {e}")
            continue
        
        # Merge on seqid
        df_truth['seqid'] = df_truth['seqid'].astype(str)
        df_fit['seqid'] = df_fit['sequentialid'].astype(str)
        for key in list(df_fit.keys()):
            if '_true' in key:
                print("Dropping the following column from fitting result for a safe merge: ", key)
                df_fit.drop(columns=[key], inplace=True)
        # Rename columns in df_fit: remove '_best' suffix to align with Truth for easy comparison
        rename_map = {}
        for c in df_fit.columns:
            if c.endswith('_best'):
                rename_map[c] = c.replace('_best', '')
        df_fit = df_fit.rename(columns=rename_map)
        
        # Suffixes: _true, _rec
        merged = pd.merge(df_truth, df_fit, on='seqid', suffixes=('_true', '_rec'))
        
        if len(merged) == 0:
            print("    No matching records found (check seqid matching).")
            continue
            
        # Determine which column is being varied (primary)
        col_true = f"{param_name}_true"
        col_rec = f"{param_name}_rec"
        
        # Special alias handling
        if param_name == 'amplitude':
            col_true = 'amplitude_true'
            col_rec = 'amplitude_rec'
        
        if col_true not in merged.columns or col_rec not in merged.columns:
            # Fallback: maybe we are not checking the varied param, but just general stats?
            # Or the param name doesn't match column name exactly.
            pass

        # --- General Plots ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Recovery of Varied Param
        ax = axes[0]
        if col_true in merged.columns and col_rec in merged.columns:
            x_vals = merged[col_true]
            y_vals = merged[col_rec]
            
            # Error bars?
            y_err = None
            err_col = None
            # Search for error column
            candidates = [f'{param_name}_err', f'err_{param_name}', f'{param_name}_error']
            for c in candidates:
                if c in merged.columns: err_col=c; break
            
            # If not found in merged (because it was in fit only and didn't collide), maybe it's without suffix?
            if not err_col:
                # Check df_fit columns in merged (they might have _rec suffix if collision, or no suffix if no collision?)
                # Wait, merge adds suffix to OVERLAPPING columns.
                # error columns are usually NOT in truth. So they won't have _rec suffix.
                for c in merged.columns:
                    if c.endswith('_err') and param_name in c:
                        err_col = c; break
            
            if err_col:
                y_err = merged[err_col]
                ax.errorbar(x_vals, y_vals, yerr=y_err, fmt='o', alpha=0.6, label='Rec')
                # Pull
                pull = (y_vals - x_vals) / y_err
                merged['pull'] = pull
            else:
                ax.plot(x_vals, y_vals, 'o', alpha=0.6, label='Rec')
                
            min_v = min(x_vals.min(), y_vals.min())
            max_v = max(x_vals.max(), y_vals.max())
            ax.plot([min_v, max_v], [min_v, max_v], 'k--', label='1:1')
            ax.set_xlabel(f"True {param_name}")
            ax.set_ylabel(f"Rec {param_name}")
            ax.set_title(f"Recovery: {param_name}")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Param cols not found", ha='center')

        # 2. Residuals
        ax = axes[1]
        if col_true in merged.columns and col_rec in merged.columns:
            res = merged[col_rec] - merged[col_true]
            ax.plot(merged[col_true], res, 'o', alpha=0.6)
            ax.axhline(0, color='k', linestyle='--')
            ax.set_xlabel(f"True {param_name}")
            ax.set_ylabel("Residual")
            ax.set_title("Residuals")

        # 3. Chi2 / Loss
        ax = axes[2]
        chi2_col = None
        if 'loss_final' in merged.columns: chi2_col = 'loss_final'
        elif 'chi2_reduced_final' in merged.columns: chi2_col = 'chi2_reduced_final'
        
        if chi2_col:
            ax.plot(merged.get(col_true, np.arange(len(merged))), merged[chi2_col], 's', color='r', alpha=0.6)
            ax.axhline(1.0, color='k', linestyle=':')
            ax.set_xlabel(f"True {param_name}")
            ax.set_ylabel(chi2_col)
            ax.set_title("Goodness of Fit (final reduced chi^2)")
            ax.set_yscale('log')
            
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"validate_{param_name}.pdf"))
        plt.close()
        
        # --- Multipole Reliability ---
        # Plot multipole uncertainties vs the varied parameter
        
        # Identify Multipole Params available
        mps = ['a_m3', 'phi_m3', 'a_m4', 'phi_m4']
        available_mps = [m for m in mps if f"{m}_rec" in merged.columns or f"{m}" in merged.columns] 
        
        if available_mps and col_true in merged.columns:
            # Use the VARIED parameter as x-axis (not always R_sersic!)
            fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
            
            x_varied = merged[col_true]
            xlab = f'True {param_name}'
                
            # Plot 1: Sigma(multipoles) vs Varied Parameter
            ax = axes2[0]
            for mp in available_mps:
                err_c = f"{mp}_err"
                if err_c in merged.columns:
                    ax.loglog(x_varied, merged[err_c], 'o', label=mp, alpha=0.5)
            
            ax.set_xlabel(xlab)
            ax.set_ylabel("Uncertainty (1 sigma)")
            ax.set_title(f"Multipole Uncertainty vs {param_name}")
            ax.legend()
            
            # Plot 2: Chi2 vs Varied Parameter
            ax = axes2[1]
            if chi2_col:
                ax.axhline(1.0, color='k', linestyle=':', label='reduced chi^2 =1')
                ax.axhline(2.0, color='k', linestyle='--', label='reduced chi^2 =2')
                ax.plot(x_varied, merged[chi2_col], 'o', color='r', alpha=0.5)
                ax.set_xlabel(xlab)
                ax.set_ylabel("Reduced Chi^2")
                ax.set_title(f"Fit Quality vs {param_name}")
                ax.set_yscale('log')
                ax.legend()
                    
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, f"multipoles_reliability_{param_name}.pdf"))
            plt.close()

        # Summary Metrics
        summary = {
            'param': param_name,
            'N': len(merged),
            'mean_residual': (merged[col_rec] - merged[col_true]).mean() if (col_rec in merged.columns and col_true in merged.columns) else np.nan,
            'mean_chi2': merged[chi2_col].mean() if chi2_col else np.nan
        }
        all_summary.append(summary)

    # Save Report
    if all_summary:
        df_sum = pd.DataFrame(all_summary)
        out_csv = os.path.join(data_dir, "validation_summary.csv")
        df_sum.to_csv(out_csv, index=False)
        print(f"\nValidation Summary saved to {out_csv}")
        print(df_sum)

if __name__ == "__main__":
    # usage example:
    # python mock_validate.py --data-dir ../data/mock_test
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=False, default="../data/mock_test", help="Directory containing mock_varying_* folders")
    args = parser.parse_args()
    
    if os.path.exists(args.data_dir):
        validate_results(args.data_dir)
    else:
        print(f"Directory not found: {args.data_dir}")