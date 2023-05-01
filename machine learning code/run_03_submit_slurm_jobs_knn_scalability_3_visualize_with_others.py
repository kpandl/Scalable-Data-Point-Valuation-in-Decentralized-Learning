import string
import os
import time
import pandas as pd

result_path = os.path.join(os.getcwd(), "results", "default0_reduced_maxtranslation_0.1_labelflip_40", "mimic_0_mimic_1_mimic_2_mimic_3_mimic_4_mimic_5", "time_knn_shapley_values_federated.csv")
result_path_or = os.path.join(os.getcwd(), "results", "scalability_results", "time_or_shapley_values.csv")
result_path_lr = os.path.join(os.getcwd(), "results", "scalability_results", "time_lr_shapley_values.csv")

df_results = pd.read_csv(result_path)
df_results_or = pd.read_csv(result_path_or)
df_results_lr = pd.read_csv(result_path_lr)

# plot df_results["compute_time"] over df_results["limit"]

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
plt.plot(df_results["limit"], df_results["compute_time"]/60, "o-", label="DDVAL, data point level")
plt.plot(df_results_or["limit"], df_results_or["compute_time"]/60, "X-", label="OR valuation, client level")
plt.plot(df_results_lr["limit"], df_results_lr["compute_time"]/60, "x-", label="SaFE, client level")
#plt.xscale("log")
#plt.yscale("log")
plt.xlabel("Number of data points [1000s]")
plt.ylabel("Compute time [min]")
#plt.title("Compute time for Shapley values")

# make y axis logarithmic
ax = plt.gca()
ax.set_yscale("log")


# divide x-axis labels by 1000
ax = plt.gca()
ax.set_xticklabels([int(x/1000) for x in ax.get_xticks()])

# show legend
plt.legend()

# set size of plot
width_inch_from_cm = 13 / 2.54
height_inch_from_cm = 13 / 2.54
fig.set_size_inches(width_inch_from_cm, height_inch_from_cm)

ax.spines[['right', 'top']].set_visible(False)

plt.tight_layout()

plt.savefig("time_knn_shapley_values_federated.png")
plt.show()