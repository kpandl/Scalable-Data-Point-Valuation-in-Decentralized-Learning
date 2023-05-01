import pandas as pd
import os
import numpy as np
import scipy.stats
import csv
from pathlib import Path

filenames = ["Shapley_original_method_predictive_performance", "Shapley_federated_performance", "Shapley_LR_method_predictive_performance", "Shapley_performance"]

path = os.path.join(os.getcwd(), "results", "plot_documents_reduced_maxtranslation_0.1_3c_iid")

df_means = []
df_hs = []
for filename in filenames:
    df = pd.read_csv(os.path.join(path, filename+"_means.csv"))
    df_means.append(df)
    df = pd.read_csv(os.path.join(path, filename+"_hs.csv"))
    df_hs.append(df)

# bar plot of dfs

import matplotlib.pyplot as plt
import numpy as np

# data to plot
n_groups = 4
means_nih = []
means_chexpert = []
means_mimic = []

hs_nih = []
hs_chexpert = []
hs_mimic = []
for i in range(n_groups):
    means_nih.append(df_means[i].iloc[0]["Shapley value"])
    hs_nih.append(df_hs[i].iloc[0]["Shapley value"])
    means_chexpert.append(df_means[i].iloc[1]["Shapley value"])
    hs_chexpert.append(df_hs[i].iloc[1]["Shapley value"])
    means_mimic.append(df_means[i].iloc[2]["Shapley value"])
    hs_mimic.append(df_hs[i].iloc[2]["Shapley value"])


# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.3
opacity = 0.8

rects1 = plt.bar(index, means_nih, bar_width,
alpha=opacity,
color='b',
label='Client 1', yerr=hs_nih, hatch="/")

rects2 = plt.bar(index + bar_width, means_chexpert, bar_width,
alpha=opacity,
color='g',
label='Client 2', yerr=hs_chexpert, hatch="\\")

rects3 = plt.bar(index + 2*bar_width, means_mimic, bar_width,
alpha=opacity,
color='r',
label='Client 3', yerr=hs_mimic, hatch="x")



plt.xlabel('Shapley value computation methods and clients')
plt.ylabel('Shapley value [%]')
#plt.title('Mean shapley value and 95% confidence intervals\nby client - iid')
plt.xticks(index + bar_width, ('Canonical SV', 'DDVal', 'SaFE', 'OR'))
plt.legend(loc='upper left')
plt.ylim(8, 13.5)

# set size of plot
width_inch_from_cm = 13 / 2.54
height_inch_from_cm = 9 / 2.54
fig.set_size_inches(width_inch_from_cm, height_inch_from_cm)

ax.spines[['right', 'top']].set_visible(False)

plt.tight_layout()
plt.show()