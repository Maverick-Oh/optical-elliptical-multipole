import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = "../data/mock_test_stronger_a_m-test_PSOMCMC/mock_varying_R_sersic"

truth_file = os.path.join(data_dir, "simulation_truth.csv")
opt_file = os.path.join(data_dir, "fitting_results.csv")
mcmc_file = os.path.join(data_dir, "fitting_results_mcmc.csv")

df_truth = pd.read_csv(truth_file)
df_opt = pd.read_csv(opt_file)
df_mcmc = pd.read_csv(mcmc_file)

# We want target sid=1
sid = 1
row_truth = df_truth[df_truth['seqid'] == sid].iloc[0]
row_opt = df_opt[df_opt['sequentialid'] == sid].iloc[0]

# MCMC might use 'id' or 'sequentialid'
if 'id' in df_mcmc.columns:
    row_mcmc = df_mcmc[df_mcmc['id'] == sid].iloc[0]
else:
    row_mcmc = df_mcmc[df_mcmc['sequentialid'] == sid].iloc[0]

params = [
    'n_sersic', 'R_sersic', 'amplitude', 'q', 'theta_ell', 
    'background', 'x0', 'y0', 
    'a_m3', 'phi_m3', 'a_m4', 'phi_m4'
]

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i, p in enumerate(params):
    ax = axes[i]
    
    val_true = row_truth[f"{p}_true"]
    
    # Non-MCMC (Optimization)
    if f"{p}_pso_best" in row_opt and not pd.isna(row_opt[f"{p}_pso_best"]):
        # If PSO was used and is valid, user might want PSO or standard. 
        # But generally, we use the one with the lowest loss or just fallback if it's there.
        # Let's see which one has the best loss and use it, or just use _best if it exists.
        # Actually user said "optimized parameter values without MCMC". 
        # I'll default to the one that is best or _best. Let me just use _best for simplicity,
        # or check if _pso_best is better.
        pass
    
    # Wait, let's just use `_best` and `_err` assuming it's the standard opt. If they are NaN, use pso.
    val_opt = row_opt.get(f"{p}_best", np.nan)
    err_opt = row_opt.get(f"{p}_err", np.nan)
    
    if pd.isna(val_opt):
        val_opt = row_opt.get(f"{p}_pso_best", np.nan)
        err_opt = row_opt.get(f"{p}_pso_err", np.nan)
        
    val_mcmc = row_mcmc[f"{p}_mcmc_best"]
    err_mcmc = row_mcmc[f"{p}_mcmc_err"]
    
    # Plotting
    ax.axhline(val_true, color='k', linestyle='--', label='True Value')
    
    # Optimization 
    ax.errorbar([0], [val_opt], yerr=[err_opt], fmt='o', color='C0', capsize=5, label='Opt (Jacobian)')
    
    # MCMC
    ax.errorbar([1], [val_mcmc], yerr=[err_mcmc], fmt='s', color='C1', capsize=5, label='MCMC')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Optimization', 'MCMC'])
    ax.set_xlim(-0.5, 1.5)
    
    ax.set_title(p)
    if i == 0:
        ax.legend()
        
plt.suptitle(f"Parameter Recovery Comparison for Target SID={sid}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

out_pdf = os.path.join(data_dir, f"comparison_opt_mcmc_sid{sid}.pdf")
plt.savefig(out_pdf)
print(f"Saved plot to {out_pdf}")
