import string
import os
import time
import pandas as pd

result_path = os.path.join(os.getcwd(), "results", "default0_reduced_maxtranslation_0.1_labelflip_40", "mimic_0_mimic_1_mimic_2_mimic_3_mimic_4_mimic_5", "time_knn_shapley_values_federated.csv")

df_results = pd.read_csv(result_path)

# plot df_results["compute_time"] over df_results["limit"]

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
plt.plot(df_results["limit"], df_results["compute_time"]/60, "o-")
#plt.xscale("log")
#plt.yscale("log")
plt.xlabel("Number of data points")
plt.ylabel("Compute time [min]")
plt.title("Compute time for KNN Shapley values")

# set size of plot
width_inch_from_cm = 13 / 2.54
height_inch_from_cm = 13 / 2.54
fig.set_size_inches(width_inch_from_cm, height_inch_from_cm)

plt.tight_layout()

plt.savefig("time_knn_shapley_values_federated.png")
plt.show()