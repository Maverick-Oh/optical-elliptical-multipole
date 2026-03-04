import re

with open('mock_validate.py', 'r') as f:
    lines = f.readlines()

new_lines = []
in_target_block = False

for i, line in enumerate(lines):
    if i == 33: # line 34: fit_file = os.path.join(d, "fitting_results.csv")
        new_lines.append('        for suffix, fn in [("", "fitting_results.csv"), ("_mcmc", "fitting_results_mcmc.csv")]:\n')
        new_lines.append('            fit_file = os.path.join(d, fn)\n')
        continue
    if i == 34:
        continue
    
    if i >= 35 and i <= 518:
        # indent by 4 spaces
        new_lines.append('    ' + line if line.strip() else line)
    else:
        new_lines.append(line)

with open('mock_validate.py', 'w') as f:
    f.writelines(new_lines)
