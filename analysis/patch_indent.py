with open("mock_run_fitting-nonjax.py", "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    # lines 395 to 570 are indices 394 to 569
    if 394 <= i <= 569:
        if line.strip(): # Don't indent purely empty lines
            new_lines.append("    " + line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open("mock_run_fitting-nonjax.py", "w") as f:
    f.writelines(new_lines)
