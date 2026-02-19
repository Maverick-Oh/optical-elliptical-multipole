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
            if 'theta_ell' in d:
                print('debug')
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
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        
        # 1. Recovery of Varied Param
        ax = axes[0]
        outliers = pd.DataFrame()
        n_out = 0
        
        if col_true in merged.columns and col_rec in merged.columns:
            x_vals = merged[col_true]
            y_vals = merged[col_rec]

            print('col_true:', col_true)

            if col_true == 'theta_ell_true': # Should this be checking specific name? Or was this legacy? Leaving as matches original logic intent
                # wrap y_vals within 0-pi!
                y_vals = y_vals % np.pi
            
            # Define limits based on Truth range with some padding
            # This allows us to catch "fly-away" Rec values
            x_min, x_max = x_vals.min(), x_vals.max()
            if x_max != x_min:
                span = x_max - x_min
            else:
                raise ValueError("x_max and x_min are equal, cannot compute span")
            
            # Use a generous buffer to include "good" recovery but exclude catastrophic failures
            # Or use percentiles of y? 
            # Let's use the union of Truth range and (inner 98% of Rec) to define the specific view?
            # User phrase "what points are missing" implies we slice.
            # Let's stick to Truth range + 50% buffer to be safe but catch gross outliers.
            # Actually, sometimes range is small. Let's try percentile based.
            
            # Actually, standard practice: 
            # But let's respect the user's "missing points" request by ENFORCING a view and showing who is out.
            
            view_min = x_min - 0.2 * span
            view_max = x_max + 0.2 * span
            
            # Let's expand view to include most y data if it's reasonable, 
            # but cap it if it helps invalidating.
            # Let's use the code's own robust limit logic:
            # Let's use 1st and 99th percentiles of Y as boundaries if they are not too wild.
            # y_p01 = np.percentile(y_vals, 1) if len(y_vals) > 0 else view_min
            # y_p99 = np.percentile(y_vals, 99) if len(y_vals) > 0 else view_max
            
            # final_min = min(view_min, y_p01)
            # final_max = max(view_max, y_p99)
            final_min = x_min if param_name in ['amplitude', 'background'] else view_min
            final_max = x_max if param_name in ['amplitude', 'background'] else view_max
            
            # Check for outliers outside THIS range
            mask_out = (y_vals < final_min) | (y_vals > final_max)
            n_out = mask_out.sum()
            outliers = merged[mask_out]
            if len(outliers) > 0:
                print(f"Outliers for {param_name}:\n{outliers}")
            
            # Error bars?
            y_err = None
            err_col = None
            candidates = [f'{param_name}_err', f'err_{param_name}', f'{param_name}_error']
            for c in candidates:
                if c in merged.columns: err_col=c; break
            
            if not err_col:
                for c in merged.columns:
                    if c.endswith('_err') and param_name in c:
                        err_col = c; break
            
            if err_col:
                y_err = merged[err_col]
                # Only plot error bars if not too cluttered?
                ax.errorbar(x_vals, y_vals, yerr=y_err, fmt='o', alpha=0.6, label='Rec')
                # Pull
                pull = (y_vals - x_vals) / y_err
                merged['pull'] = pull
            else:
                ax.plot(x_vals, y_vals, 'o', alpha=0.6, label='Rec')
            
            ax.plot([final_min, final_max], [final_min, final_max], 'k--', label='1:1')
            ax.set_xlabel(f"True {param_name}")
            ax.set_ylabel(f"Rec {param_name}")
            ax.set_title(f"Rec: {param_name}")
            
            # Suptitle for subplot or Figure? User said "have suptitle that says...". 
            # Usually suptitle is for Figure, but user said "In this code section... they have suptitle".
            # Maybe they meant title (for the subplot) or suptitle for the whole figure.
            # Given context "make the figure with 4 column subplots", "suptitle" usually means global title.
            # But specific to "points out of xlim", this is per-parameter.
            # I will add it to the Title of this subplot OR the Figure suptitle.
            # Let's put it in the subplot title for clarity since we loop over parameters.
            # "Recovery: {param} (N_out={n_out})"
            
            ax.set_title(f"Rec: {param_name}\n(Out of Range: {n_out})")
            ax.legend()
            # ax.set_xlim(final_min, final_max)
            ax.set_ylim(final_min, final_max)
            if param_name in ['amplitude', 'background']:
                print('debug')
        else:
            ax.text(0.5, 0.5, "Param cols not found", ha='center')

        # 2. Residuals
        ax = axes[1]
        if col_true in merged.columns and col_rec in merged.columns:
            res = merged[col_rec] - merged[col_true]
            # Use same mask if meaningful? 
            # Or just plot all residuals? 
            # Usually residuals for outliers are huge.
            # Let's just plot all.
            ax.plot(merged[col_true], res, 'o', alpha=0.6)
            ax.axhline(0, color='k', linestyle='--')
            ax.set_xlabel(f"True {param_name}")
            ax.set_ylabel("Residual")
            ax.set_title("Residuals")
            # ax.set_xlim(final_min, final_max)
        
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
            ax.set_ylim(0.5, 2)

        # Helper function to add vertical lines and seqid labels
        def add_vlines_and_seqid(ax, x_data, y_data, seqids):
            # We need to know the current y-limits to draw lines to the bottom
            # But limits might change. Let's use a transform that mixes data (x) and axes (y) coordinates?
            # Actually, easiest is to use axvline with ymax.
            # However, axvline goes from 0 to 1 (axes fraction).
            # We want it to go from 0 (bottom) to the data point y (in axes fraction).
            # The data point y might be outside [0,1].
            
            # Let's use plot with a transform for the x-axis and data for y-axis? No, that's complex.
            # Let's just use data coordinates for both, but use the current bottom limit.
            # BETTER: transform=ax.get_xaxis_transform() makes y=0 the bottom of the axis?
            # No, xaxis_transform makes y in axes coords (0=bottom, 1=top). X is data coords.
            # PERFECT!
            # So we plot from (x, 0) to (x, y_data_transformed_to_axes_fraction).
            # But y_data is in data coords.
            # So we can't easily use mixed transform for the line *height* based on data value.
            
            # Alternative: iterate and draw line from (x, y_data) to (x, y_bottom_limit).
            # We can get y_bottom from ax.get_ylim()[0].
            # But we must ensure limits are finalized? Or just use a very small number if log/linear?
            # Let's use the transform trick for the TEXT (labels) because they stay at the bottom.
            # For the lines, let's use vlines and rely on data coordinates.
            
            # Wait, if we use vlines, we need a 'ymin'.
            # If we assume the plot limits are set or will be set to include (most of) the data,
            # we can use the data min or current limit.
            # Let's wait until limits are set to draw lines? 
            # Or just draw them with a very low ymin (e.g. -1e99 for linear, 1e-99 for log) and clip_on=True?
            # User said "come down to the x-axis".
            # If we use transformed y (0), we get the axis line.
            # So we want a line from (x, y_data) to (x, y_axis_bottom).
            
            # Simple approach:
            # For each point:
            #   ax.plot([x, x], [y, y_axis_btm], ...)
            # We need y_axis_btm properly.
            
            ymin, ymax = ax.get_ylim()
            
            # For text:
            trans = ax.get_xaxis_transform() # x in data, y in axes fraction
            
            for x, y, sid in zip(x_data, y_data, seqids):
                # Draw vertical line
                # We can't use xaxis_transform for the whole line because the top point (y) is data-dependent.
                # So we use data coords for the line.
                # Use current ymin as the floor.
                ax.vlines(x, ymin, y, colors='gray', linestyles=':', alpha=0.5, linewidth=0.8, clip_on=False)
                
                # Add seqid text
                # y=-0.02 in axes coords puts it just below x-axis
                ax.text(x, -0.02, f"({sid})", transform=trans, 
                        ha='center', va='top', fontsize=6, color='gray', rotation=90, clip_on=False)

        # Apply to General Plots
        # axes[0]: Rec
        # limits are set (final_min, final_max)
        add_vlines_and_seqid(axes[0], merged[col_true], merged[col_rec], merged['seqid'])
        
        # axes[1]: Residuals
        # limits are auto (except xlim which is set to final_min, max)
        add_vlines_and_seqid(axes[1], merged[col_true], merged[col_rec]-merged[col_true], merged['seqid'])
        
        # axes[2]: Chi2
        # limits set (0.5, 2)
        if chi2_col:
            add_vlines_and_seqid(axes[2], merged.get(col_true, np.arange(len(merged))), merged[chi2_col], merged['seqid'])

        # 4. Outlier Details
        ax = axes[3]
        ax.set_axis_off()
        ax.set_title(f"Outliers (> {n_out})")
        if n_out > 0:
            # Find filename column
            fname_col = None
            for c in merged.columns:
                if 'filename' in c and 'sci' in c: fname_col = c; break
            if not fname_col:
                 for c in merged.columns:
                    if 'filename' in c: fname_col = c; break
            
            cols_to_show = ['seqid', col_true, col_rec]
            if chi2_col: cols_to_show.append(chi2_col)
            if fname_col: cols_to_show.append(fname_col)
            
            # Create a textual representation of outliers
            # Limit to top 20 to avoid overcrowding
            out_info = outliers[cols_to_show].copy()
            
            # Format nicely
            out_info[col_true] = out_info[col_true].map(lambda x: f"{x:.4g}")
            out_info[col_rec] = out_info[col_rec].map(lambda x: f"{x:.4g}")
            if chi2_col:
                out_info[chi2_col] = out_info[chi2_col].map(lambda x: f"{x:.4g}")
            
            # Header
            header = "seqid | True | Rec"
            if chi2_col: header += " | Chi2"
            if fname_col: header += " | File"
            
            text_str = header + "\n" + "-"*len(header) + "\n"
            
            rows = []
            for idx, row in out_info.head(30).iterrows():
                r_str = f"{row['seqid']} | {row[col_true]} | {row[col_rec]}"
                if chi2_col: r_str += f" | {row[chi2_col]}"
                if fname_col: 
                    # Shorten filename?
                    f_short = os.path.basename(str(row[fname_col]))
                    r_str += f" | {f_short}"
                rows.append(r_str)
            
            text_str += "\n".join(rows)
            if n_out > 30:
                text_str += f"\n... (+{n_out-30} more)"
            
            ax.text(0.0, 1.0, text_str, va='top', ha='left', family='monospace', fontsize=8, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No points out of range.", ha='center', va='center')

        # 5. Set log scale for amplitude and background
        if col_true in ['amplitude_true', 'background_true']:
            axes[0].set_xscale('log')
            axes[0].set_yscale('log')
            axes[1].set_xscale('log')    
            axes[2].set_xscale('log')
            print('debug')
            
        plt.suptitle(f"Validation: {param_name} (Total: {len(merged)}, Out: {n_out})", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
        plt.savefig(os.path.join(PLOT_DIR, f"validate_{param_name}.pdf"))
        plt.close()
        
        # --- Multipole Reliability ---
        # Plot multipole uncertainties vs the varied parameter
        
        # Identify Multipole Params available
        mps = ['a_m3', 'a_m4', 'phi_m3', 'phi_m4']
        available_mps = [m for m in mps if f"{m}_rec" in merged.columns or f"{m}" in merged.columns] 
        # available_mps.sort() # such that it is in the order of a_m3, a_m4, phi_m3, and phi_m4
        
        if available_mps and col_true in merged.columns:
            # Use the VARIED parameter as x-axis (not always R_sersic!)
            fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
            
            axes2[0].axhline(np.pi/6, color='lightgreen', linestyle='--', alpha=0.5, label='$\pi/6$')            
            axes2[0].axhline(np.pi/8, color='darkgreen', linestyle='--', alpha=0.5, label='$\pi/8$')
            axes2[0].axhline(0.005, color='blue', linestyle='-', alpha=0.5, label='0.005')

            x_varied = merged[col_true]
            xlab = f'True {param_name}'
                
            # Plot 1: Sigma(multipoles) vs Varied Parameter
            ax = axes2[0]
            
            # Custom markers
            # a_m3: light blue triangle_up
            # a_m4: dark blue square
            # phi_m3: light green triangle_up
            # phi_m4: dark green square
            
            styles = {
                'a_m3': {'fmt': '^', 'color': 'lightblue', 'size': 10},
                'a_m4': {'fmt': 's', 'color': 'darkblue', 'size': 5},
                'phi_m3': {'fmt': '^', 'color': 'lightgreen', 'size': 10},
                'phi_m4': {'fmt': 's', 'color': 'darkgreen', 'size': 5}
            }
            
            for mp in available_mps:
                err_c = f"{mp}_err"
                if err_c in merged.columns:
                    s = styles.get(mp, {'fmt': 'o', 'color': 'k'}) # fallback
                    ax.plot(x_varied, merged[err_c], marker=s['fmt'], color=s['color'], linestyle='', label=mp, alpha=0.8, markersize=s['size'])
                    
                    # Add vlines/seqid for this series? 
                    # If we have multiple series on one plot, it might get cluttered if we draw line for EACH one.
                    # But x is same for all (x_varied). y differs.
                    # Let's draw lines for one of them or better yet, iterate unique x and draw common lines?
                    # The request says "For each point that got plotted... add a vertical gray line".
                    # Since they share x, one vertical line per x is enough visually.
                    # But seqid labeling is per object.
                    # Let's just do it once per object using the x-value.
                    
            # Add vlines/seqid (once per object)
            # Use 'a_m3' or first available as "y" reference? 
            # Actually, line should go from data point DOWN.
            # If we have multiple points at same X (different params), we have multiple lines from different Ys?
            # Or just one line from the Lowest/Highest Y?
            # User expectation: "start from the data point".
            # If we have 4 data points per X (4 params), we get 4 lines overlapping. That's fine.
            # Visually it looks like lines from each marker down.

            min_multipole_uncertainty_lim = 1e-4
            max_multipole_uncertainty_lim = np.pi
            ax.set_xlabel(xlab)
            ax.set_ylabel("Uncertainty (1 sigma)")
            ax.set_title(f"Multipole Uncertainty vs {param_name}")
            ax.legend()
            ax.set_ylim(min_multipole_uncertainty_lim, max_multipole_uncertainty_lim) # if phi_m3 or phi_m4 is over np.pi, it's too big of uncertainty. a_m3 a_m4 must be <<1 too, so np.pi is a safe upper bound.
            ax.set_yscale('log')
            for mp in available_mps:
                err_c = f"{mp}_err"
                if err_c in merged.columns:
                    add_vlines_and_seqid(ax, x_varied, merged[err_c], merged['seqid'])

            # Plot 2: Chi2 vs Varied Parameter
            ax = axes2[1]
            min_chi2_lim = 0.5
            max_chi2_lim = 3.0
            if chi2_col:
                ax.axhline(1.0, color='k', linestyle=':', label='reduced chi^2 =1')
                ax.axhline(2.0, color='k', linestyle='--', label='reduced chi^2 =2')
                ax.plot(x_varied, merged[chi2_col], 'o', color='r', alpha=0.5)
                ax.set_xlabel(xlab)
                ax.set_ylabel("Reduced Chi^2")
                ax.set_title(f"Fit Quality vs {param_name}")
                ax.set_yscale('log')
                ax.set_ylim(min_chi2_lim, max_chi2_lim) # if reduced chi^2 is over 3, it's too big of a reduced chi^2.
                ax.legend()
                
                # Add vlines/seqid
                add_vlines_and_seqid(ax, x_varied, merged[chi2_col], merged['seqid'])
            
            if 'amplitude' in col_true:
                print(col_true)
                print('debug')

            if col_true in ['amplitude_true', 'background_true']:
                axes2[0].set_xscale('log')
                axes2[1].set_xscale('log')

            # Plot 3: Outliers details
            ax = axes2[2]
            ax.set_axis_off()
            
            # Identify outliers for Multipoles
            # Criteria: Unc > max_multipole_uncertainty_lim (pi) OR Chi2 > max_chi2_lim (3) OR Chi2 < min_chi2_lim (0.5)
            bad_chi2_upper = merged[chi2_col] > max_chi2_lim if chi2_col else pd.Series([False]*len(merged))
            bad_chi2_lower = merged[chi2_col] < min_chi2_lim if chi2_col else pd.Series([False]*len(merged))
            bad_unc = pd.Series([False]*len(merged))
            
            # Also want to know WHICH param failed for the report
            # Also want to know WHICH param failed for the report
            # Ensure bad_params is object type (string)
            merged['bad_params'] = ''
            merged['bad_params'] = merged['bad_params'].astype(object)
            
            for mp in available_mps:
                err_c = f"{mp}_err"
                if err_c in merged.columns:
                    is_bad = (merged[err_c] > max_multipole_uncertainty_lim) | (merged[err_c] < min_multipole_uncertainty_lim)
                    bad_unc = bad_unc | is_bad
                    if is_bad.any():
                        # Append bad param name safely
                        to_append = f"{mp}_err=" + merged.loc[is_bad, err_c].apply(lambda x: f"{x:.2g}") + "; "
                        merged.loc[is_bad, 'bad_params'] = merged.loc[is_bad, 'bad_params'] + to_append
            
            if chi2_col:
                is_bad_chi2_upper = merged[chi2_col] > max_chi2_lim
                is_bad_chi2_lower = merged[chi2_col] < min_chi2_lim
                if is_bad_chi2_upper.any():
                    to_append = f"Chi2=" + merged.loc[is_bad_chi2_upper, chi2_col].apply(lambda x: f"{x:.2g}") + "; "
                    merged.loc[is_bad_chi2_upper, 'bad_params'] = merged.loc[is_bad_chi2_upper, 'bad_params'] + to_append
                if is_bad_chi2_lower.any():
                    to_append = f"Chi2=" + merged.loc[is_bad_chi2_lower, chi2_col].apply(lambda x: f"{x:.2g}") + "; "
                    merged.loc[is_bad_chi2_lower, 'bad_params'] = merged.loc[is_bad_chi2_lower, 'bad_params'] + to_append
            
            mask_mp_out = bad_chi2_upper | bad_chi2_lower | bad_unc
            n_mp_out = mask_mp_out.sum()
            
            ax.set_title(f"Outliers (out of plotting range): {n_mp_out}")
            
            if n_mp_out > 0:
                mp_outliers = merged[mask_mp_out]
                
                # Report columns: seqid, bad_params
                text_str = "seqid | Issues\n" + "-"*30 + "\n"
                rows = []
                for idx, row in mp_outliers.head(30).iterrows():
                    rows.append(f"{row['seqid']} | {row['bad_params']}")
                
                text_str += "\n".join(rows)
                if n_mp_out > 30:
                    text_str += f"\n... (+{n_mp_out-30} more)"
                
                ax.text(0.0, 1.0, text_str, va='top', ha='left', family='monospace', fontsize=8, transform=ax.transAxes)
            else:
                 ax.text(0.5, 0.5, "No outliers found.", ha='center', va='center')
                    
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
    parser.add_argument("--data-dir", type=str, required=False, default="../data/mock_test_weak_a_m", help="Directory containing mock_varying_* folders")
    args = parser.parse_args()
    
    if os.path.exists(args.data_dir):
        validate_results(args.data_dir)
    else:
        print(f"Directory not found: {args.data_dir}")