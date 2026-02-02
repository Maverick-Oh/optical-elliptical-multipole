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
        df_fit['seqid'] = df_fit['seqid'].astype(str)
        
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
            ax.set_title("Goodness of Fit")
            
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"validate_{param_name}.pdf"))
        plt.close()
        
        # --- Multipole Reliability ---
        # Check multipole errors vs Signal (Amplitude) and Size (R_sersic)
        # We assume R_sersic and amplitude are available (either as varied or fixed truth)
        
        # Identify Multipole Params available
        mps = ['a_m3', 'phi_m3', 'a_m4', 'phi_m4']
        available_mps = [m for m in mps if f"{m}_rec" in merged.columns or f"{m}" in merged.columns] 
        # Note: if fit results have a_m3 (renamed from a_m3_best), and truth has a_m3, we get a_m3_rec and a_m3_true.
        
        if available_mps:
            # We want to plot Uncertainty of these MPs vs R_sersic_true (or amplitude_true)
            fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
            
            # X-axis: R_sersic_true
            if 'R_sersic_true' in merged.columns:
                x_sz = merged['R_sersic_true']
                xlab = 'True R_sersic'
            elif 'R_sersic' in merged.columns: # Fixed value?
                x_sz = merged['R_sersic']
                xlab = 'R_sersic'
            else:
                x_sz = None
                
            if x_sz is not None:
                # Plot 1: Sigma(a_m) vs Size
                ax = axes2[0]
                for mp in available_mps:
                    # Find error col
                    # mp is like 'a_m3'. Error col: 'a_m3_err'
                    err_c = f"{mp}_err" # Assuming standard naming
                    # Check in columns
                     # Error columns won't have suffixes usually
                    if err_c in merged.columns:
                        ax.loglog(x_sz, merged[err_c], 'o', label=mp, alpha=0.5)
                
                ax.set_xlabel(xlab)
                ax.set_ylabel("Uncertainty (1 sigma)")
                ax.set_title("Multipole Uncertainty vs Size")
                ax.legend()
                
                # Plot 2: Chi2 vs Size
                ax = axes2[1]
                if chi2_col:
                    ax.semilogx(x_sz, merged[chi2_col], 'o', color='r', alpha=0.5)
                    ax.set_xlabel(xlab)
                    ax.set_ylabel("Chi2 / Loss")
                    ax.set_title("Fit Quality vs Size")
                    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing mock_varying_* folders")
    args = parser.parse_args()
    
    if os.path.exists(args.data_dir):
        validate_results(args.data_dir)
    else:
        print(f"Directory not found: {args.data_dir}")
