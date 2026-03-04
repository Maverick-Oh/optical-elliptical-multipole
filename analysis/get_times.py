import pandas as pd
import os

out_path = '../data/mock_test_stronger_a_m-test_PSO/mock_varying_a_m3/fitting_results.csv'
out_path_mcmc = '../data/mock_test_stronger_a_m-test_PSO/mock_varying_a_m3/fitting_results_mcmc.csv'

print(f"\n--- Execution Timing Report ---")
if os.path.exists(out_path):
    df_opt = pd.read_csv(out_path)
    if 'opt_best_strategy' in df_opt.columns and 'fit_time' in df_opt.columns:
        mask_pso = df_opt['opt_best_strategy'] == 'PSO'
        non_pso_times = df_opt.loc[~mask_pso, 'fit_time']
        pso_times = df_opt.loc[mask_pso, 'fit_time']
        
        if len(non_pso_times) > 0:
            print(f"Non-PSO Optimization: {non_pso_times.mean():.2f}s avg (N={len(non_pso_times)})")
        else:
            print("Non-PSO Optimization: None observed")
            
        if len(pso_times) > 0:
            print(f"PSO Optimization: {pso_times.mean():.2f}s avg (N={len(pso_times)})")
        else:
            print("PSO Optimization: None observed")
if os.path.exists(out_path_mcmc):
    df_mcmc = pd.read_csv(out_path_mcmc)
    if 'mcmc_time' in df_mcmc.columns:
        print(f"MCMC Inference: {df_mcmc['mcmc_time'].mean():.2f}s avg (N={len(df_mcmc['mcmc_time'])})")
print("-------------------------------------------\n")
