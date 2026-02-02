import pandas as pd
import numpy as np

# file_path = '../data/HDUL_2025-10-22_134731_all/cosmos_analysis_result.csv' # Replace with your CSV file's path
file_path = '../data/HDUL_test6-1000/cosmos_optimization_result-at-2026-01-03_114917.csv'

def remove_nan(df, keys):
    # this drops the data frame rows with blank fields with specified keys; it is here to filter out the samples that
    # didn't go through optimization well.
    for key in keys:
        ind_to_drop = np.where(np.isnan(df[key]))[0]
        df = df.drop(ind_to_drop)
        print("Dropped: ", len(ind_to_drop), " samples due to nan fields")
    return df

def remove_outof_range(df, lower_limit_dict=None, upper_limit_dict=None):
    if lower_limit_dict is not None:
        for key in lower_limit_dict.keys():
            ind_to_drop = np.where(df[key] >= lower_limit_dict[key])[0]
            df = df.drop(ind_to_drop)
            print("Dropped: ", len(ind_to_drop), f" samples due to lower limit on {key} field")
    if upper_limit_dict is not None:
        for key in upper_limit_dict.keys():
            ind_to_drop = np.where(df[key] <= upper_limit_dict[key])[0]
            df = df.drop(ind_to_drop)
            print("Dropped: ", len(ind_to_drop), f" samples due to upper limit on {key} field")
    return df

df = pd.read_csv(file_path)
print(df.head()) # Prints the first 5 rows of the DataFrame

num_before_nan_removal = len(df)
print("number before nan removal: ", num_before_nan_removal)
df = remove_nan(df, keys=['loss_final'])
num_after_nan_removal = len(df)
print("number before nan removal: ", num_after_nan_removal)
df = remove_outof_range(df, lower_limit_dict={'loss_final': 1.0})

z = np.array(df['q_best'])
dz = np.array(df['q_err'])
x3 = np.array(df['a_m3_best'])
dx3 = np.array(df['a_m3_err'])
y3 = np.array(df['phi_m3_best'])
dy3 = np.array(df['phi_m3_err'])
x4 = np.array(df['a_m4_best'])
dx4 = np.array(df['a_m4_err'])
y4 = np.array(df['phi_m4_best'])
dy4 = np.array(df['phi_m4_err'])

import matplotlib.pyplot as plt

selection = z<=1.
num_after_q_1_removal = np.sum(selection)
print(f"Dropped: {np.sum(~selection)} samples due to q>1.0")
print("number after q==1.0 removal: ", num_after_q_1_removal)

# plt.plot(x4[selection], y4[selection], '.')
# plt.ylim([-np.pi/2/m, np.pi/2/m])
# plt.show()

from optical_elliptical_multipole.nonjax.tools import amplitude_angle_wrapper

# m=3 plotting with error bars
m=3
x3_, y3_ = amplitude_angle_wrapper(x3, y3, m)
plt.plot(x3_[selection], y3_[selection], '.')
plt.ylim([-np.pi/2/m, np.pi/2/m])
plt.show()

fig, ax = plt.subplots()
hist = ax.hist2d(x3_[selection], y3_[selection], bins=30, cmap='viridis') # 'viridis' is a common colormap
# Add labels and a title
ax.set_xlabel('a3'); ax.set_ylabel('phi3')
ax.set_title(f"2D Histogram of a3 and phi3 ({num_after_q_1_removal}/{num_before_nan_removal}\n"
          f"{num_before_nan_removal-num_after_nan_removal} removed for nan ("
          f"optimization didn't do well), and\n{num_after_nan_removal-num_after_q_1_removal} removed for q==1.0 (now "
          f"sure why; will be "
          f"tested)")
# Add a color bar to indicate density
cbar = plt.colorbar(mappable=hist[-1], label='Density', ax=ax)
ax.set_position([.1, .1, .6, 0.6])
cbar.ax.set_position([0.7837500000000002, 0.10999999999999999, 0.05, 0.6])
fig.savefig('fit_result_m=3_2D_hist.pdf')
plt.show()

alpha = 0.1
plt.errorbar(x3_[selection], y3_[selection],
             yerr=dy4[selection], xerr=dx4[selection], fmt='',
             alpha=alpha, linestyle='None', marker='.', ms=None)
plt.xlim([-0.1, 0.1])
plt.ylim([-np.pi/2/m, np.pi/2/m])
plt.xlabel('a3')
plt.ylabel('phi3')
plt.title('a3 vs. phi3')
plt.savefig("fit_result_m=3_error_bar(phi).pdf")
plt.show()

# m=4 plotting with error bars
m=4
x4_, y4_ = amplitude_angle_wrapper(x4, y4, 4)
plt.plot(x4_[selection], y4_[selection], '.')
plt.ylim([-np.pi/2/m, np.pi/2/m])
plt.show()

fig, ax = plt.subplots()
hist = ax.hist2d(x4_[selection], y4_[selection], bins=30, cmap='viridis') # 'viridis' is a common colormap
# Add labels and a title
ax.set_xlabel('a4'); ax.set_ylabel('phi4')
ax.set_title(f"2D Histogram of a4 and phi4 ({num_after_q_1_removal}/{num_before_nan_removal}\n"
          f"{num_before_nan_removal-num_after_nan_removal} removed for nan ("
          f"optimization didn't do well), and\n{num_after_nan_removal-num_after_q_1_removal} removed for q==1.0 (now "
          f"sure why; will be "
          f"tested)")
# Add a color bar to indicate density
cbar = plt.colorbar(mappable=hist[-1], label='Density', ax=ax)
ax.set_position([.1, .1, .6, 0.6])
cbar.ax.set_position([0.7837500000000002, 0.10999999999999999, 0.05, 0.6])
fig.savefig('fit_result_m=4_2D_hist.pdf')
plt.show()

plt.errorbar(x4_[selection], y4_[selection],
             yerr=dy4[selection], xerr=dx4[selection], fmt='',
             alpha=alpha, linestyle='None', marker='.', ms=None)
plt.xlim([-0.1, 0.1])
plt.ylim([-np.pi/2/m, np.pi/2/m])
plt.xlabel('a4')
plt.ylabel('phi4')
plt.title('a4 vs. phi4')
plt.savefig("fit_result_m=4_error_bar(phi).pdf")
plt.show()


print("done")